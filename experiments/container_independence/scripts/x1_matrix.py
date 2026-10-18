"""x1: empirical test of claim C1 on narwhals ``collect()`` semantics.

C1 (dev claim): "Polars lazy collects to Polars by default, but Narwhals' collect
also sends Dask to pandas and DuckDB to Arrow."

This script records, per native input container:

A. ``nw.from_native(x)`` -> narwhals class (DataFrame / LazyFrame), implementation
B. ``.collect()`` default and ``.collect(backend=...)`` for every candidate value
   (strings, Implementation enums, modules) -> native result type or error;
   for eager inputs ``.lazy()`` / ``.lazy(backend=...)`` -> native type
C. backend x operation matrix for the ops a narwhals VectorizedDF would need,
   executed on a narwhals LazyFrame. For each op: does it build without error,
   is ``.to_native()`` still the ORIGINAL native type (format respected), did it
   trigger hidden computation (dask: compute-callback count), and is the
   collected result equal to the polars-eager reference.
D. eager-only ops (iterate groups, to_numpy, pivot, ...) on eager frames.

Run (from worktree root)::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x1_matrix.py [--tag pd2]

Writes results/x1_matrix_<tag>.json and prints a markdown matrix.
"""

import argparse
import json
import os
import sys
import tempfile
import traceback

os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("MODIN_ENGINE", "python")  # serial engine, no ray/dask workers

import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402

import narwhals as nw  # noqa: E402
from narwhals import Implementation as Impl  # noqa: E402

from common import env_info, make_panel_fast  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

try:
    import dask
    import dask.dataframe as dd
    from dask.callbacks import Callback

    dask.config.set(scheduler="synchronous")
except ImportError:  # pragma: no cover
    dd = None
try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None
try:
    import ibis
except ImportError:
    ibis = None
try:
    import modin.pandas as mpd
except ImportError:
    mpd = None


def short_type(obj):
    t = type(obj)
    return f"{t.__module__.split('.')[0]}.{t.__name__}"


def err_str(e):
    msg = str(e).strip().splitlines()
    first = msg[0] if msg else ""
    return f"{type(e).__name__}: {first[:160]}"


# --------------------------------------------------------------------------- data
def make_long(n_series=20, n_time=30):
    pdf = make_panel_fast(n_series, n_time, n_cols=2).reset_index()
    # h0: str, time: datetime64[ns], c0/c1: float64
    return pdf


class Natives:
    """Builds each native container from the same long pandas frame."""

    def __init__(self, pdf):
        self.pdf = pdf
        self.arrow = pa.Table.from_pandas(pdf, preserve_index=False)
        self.duck_con = duckdb.connect() if duckdb else None
        if self.duck_con is not None:
            self.duck_con.execute("SET threads=4")
        self.ibis_con = ibis.duckdb.connect(threads=4) if ibis else None
        self._ibis_n = 0

    def builders(self):
        b = {
            "pandas": lambda: self.pdf.copy(),
            "polars_eager": lambda: pl.from_arrow(self.arrow),
            "polars_lazy": lambda: pl.from_arrow(self.arrow).lazy(),
            "pyarrow": lambda: self.arrow,
        }
        if dd is not None:
            b["dask"] = lambda: dd.from_pandas(self.pdf, npartitions=4)
        if duckdb is not None:
            # default (module-level) connection, as in duckdb.sql(...) / duckdb.from_arrow
            b["duckdb"] = lambda: duckdb.from_arrow(self.arrow)
            # user-owned connection, as in production code: con = duckdb.connect(...)
            b["duckdb_owncon"] = lambda: self.duck_con.from_arrow(self.arrow)
        if ibis is not None:
            b["ibis_duckdb"] = self._ibis
        if mpd is not None:
            b["modin"] = lambda: mpd.DataFrame(self.pdf)
        return b

    def _ibis(self):
        # NB: create_table(..., overwrite=True) raises a duckdb ParserException with
        # ibis 12 + duckdb 1.5.6 (seen in probing); use fresh names instead.
        self._ibis_n += 1
        return self.ibis_con.create_table(
            f"panel_{self._ibis_n}", ibis.memtable(self.arrow)
        )


# ------------------------------------------------------------- dask compute counter
class ComputeCounter:
    def __init__(self):
        self.n = 0

    def __enter__(self):
        if dd is None:
            return self
        counter = self

        class _CB(Callback):
            def _start(self, dsk):
                counter.n += 1

        self._cb = _CB()
        self._cb.__enter__()
        return self

    def __exit__(self, *a):
        if dd is not None:
            self._cb.__exit__(*a)


# ------------------------------------------------------------------- part A + B
COLLECT_BACKENDS = [
    ("None", None),
    ("'pandas'", "pandas"),
    ("'polars'", "polars"),
    ("'pyarrow'", "pyarrow"),
    ("Impl.PANDAS", Impl.PANDAS),
    ("Impl.POLARS", Impl.POLARS),
    ("Impl.PYARROW", Impl.PYARROW),
    ("module pandas", pd),
    ("module polars", pl),
    ("'dask'", "dask"),
    ("'duckdb'", "duckdb"),
    ("'ibis'", "ibis"),
    ("'modin'", "modin"),
    ("Impl.DASK", Impl.DASK),
    ("Impl.DUCKDB", Impl.DUCKDB),
    ("Impl.MODIN", Impl.MODIN),
]

LAZY_BACKENDS = [
    ("None", None),
    ("'polars'", "polars"),
    ("'duckdb'", "duckdb"),
    ("'dask'", "dask"),
    ("'ibis'", "ibis"),
    ("'pandas'", "pandas"),
    ("'pyarrow'", "pyarrow"),
]


def part_ab(nat):
    out = {}
    for name, build in nat.builders().items():
        native = build()
        rec = {"native_type": short_type(native)}
        obj = nw.from_native(native)
        rec["nw_class"] = type(obj).__name__
        rec["implementation"] = str(obj.implementation)
        # also: can it be forced to be eager?
        try:
            nw.from_native(native, eager_only=True)
            rec["eager_only=True"] = "ok"
        except Exception as e:  # noqa: BLE001
            rec["eager_only=True"] = err_str(e)

        if isinstance(obj, nw.DataFrame):
            lazies = {}
            for label, be in LAZY_BACKENDS:
                try:
                    lf = obj.lazy(backend=be) if be is not None else obj.lazy()
                    lazies[label] = (
                        f"{type(lf).__name__}[{lf.implementation}] -> "
                        f"{short_type(lf.to_native())}"
                    )
                except Exception as e:  # noqa: BLE001
                    lazies[label] = err_str(e)
            rec["DataFrame.lazy(backend)"] = lazies
            lf = obj.lazy()
        else:
            lf = obj
        collects = {}
        for label, be in COLLECT_BACKENDS:
            try:
                with ComputeCounter() as cc:
                    res = lf.collect(backend=be) if be is not None else lf.collect()
                collects[label] = short_type(res.to_native()) + (
                    f" (dask computes={cc.n})" if name == "dask" else ""
                )
            except Exception as e:  # noqa: BLE001
                collects[label] = err_str(e)
        rec["collect(backend)"] = collects
        kw = {"polars_lazy": {"engine": "streaming"}, "dask": {"scheduler": "threads"}}
        if name in kw:
            try:
                res = lf.collect(**kw[name])
                rec["collect(**kwargs)"] = f"{kw[name]} -> {short_type(res.to_native())}"
            except Exception as e:  # noqa: BLE001
                rec["collect(**kwargs)"] = err_str(e)
        out[name] = rec
    return out


# ----------------------------------------------------------------------- part C
K = 3  # last-k rows per series


def same_backend_small(lf, data):
    """Lift a small eager dict into the same backend as ``lf`` (lazy)."""
    impl = lf.implementation
    if impl in (Impl.POLARS, Impl.DUCKDB, Impl.DASK, Impl.IBIS):
        return nw.from_dict(data, backend="pyarrow").lazy(backend=impl)
    return nw.from_dict(data, backend=impl).lazy()


def op_specs(first_key):
    c = nw.col
    t0 = pd.Timestamp("2000-01-31")
    return {
        # --- elementwise / relational
        "with_columns": lambda lf: lf.with_columns(c2=c("c0") * 2),
        "filter": lambda lf: lf.filter(c("c0") > 0),
        "select_cols": lambda lf: lf.select("h0", "time", "c0"),
        "sort": lambda lf: lf.sort("h0", "time"),
        "collect_schema": lambda lf: lf.collect_schema(),
        # --- instance keys
        "unique_keys": lambda lf: lf.select("h0").unique(),
        "unique_keys_sorted": lambda lf: lf.select("h0").unique().sort("h0"),
        "n_rows_per_key": lambda lf: lf.group_by("h0").agg(nw.len()),
        "filter_one_key": lambda lf: lf.filter(c("h0") == first_key),
        # --- group_by aggregations
        "group_by_agg_mean_max": lambda lf: lf.group_by("h0").agg(
            c("c0").mean(), c("time").max().alias("cutoff")
        ),
        "group_by_agg_last(order-dep)": lambda lf: lf.group_by("h0").agg(
            c("c0").last()
        ),
        "group_by_agg_last_after_sort": lambda lf: lf.sort("time")
        .group_by("h0")
        .agg(c("c0").last()),
        "group_by_agg_last(order_by=time)": lambda lf: lf.group_by("h0").agg(
            c("c0").last(order_by="time"), c("time").max()
        ),
        # --- windows (over)
        "over_mean(no order)": lambda lf: lf.with_columns(m=c("c0").mean().over("h0")),
        "over_cum_sum(order_by)": lambda lf: lf.with_columns(
            cs=c("c0").cum_sum().over("h0", order_by="time")
        ),
        "over_shift(order_by)": lambda lf: lf.with_columns(
            lag1=c("c0").shift(1).over("h0", order_by="time")
        ),
        "over_shift(no order_by)": lambda lf: lf.with_columns(
            lag1=c("c0").shift(1).over("h0")
        ),
        "over_diff(order_by)": lambda lf: lf.with_columns(
            d=c("c0").diff().over("h0", order_by="time")
        ),
        "over_rolling_mean(order_by)": lambda lf: lf.with_columns(
            rm=c("c0").rolling_mean(3).over("h0", order_by="time")
        ),
        "over_ffill(order_by)": lambda lf: lf.with_columns(
            ff=c("c0").fill_null(strategy="forward").over("h0", order_by="time")
        ),
        # --- last-k rows per group (cutoff / window of training data)
        "last_row_per_key(max over)": lambda lf: lf.filter(
            c("time") == c("time").max().over("h0")
        ),
        "last_row_per_key(semi-join on max)": lambda lf: lf.join(
            lf.group_by("h0").agg(c("time").max()), on=["h0", "time"], how="semi"
        ),
        "last_k_per_key(rank over)": lambda lf: lf.filter(
            c("time").rank(method="ordinal", descending=True).over("h0") <= K
        ),
        "last_k_per_key(cum_count rev)": lambda lf: lf.filter(
            c("c0").cum_count(reverse=True).over("h0", order_by="time") <= K
        ),
        "with_row_index(order_by)": lambda lf: lf.with_row_index(
            "idx", order_by=["h0", "time"]
        ),
        "head": lambda lf: lf.head(5),
        "tail": lambda lf: lf.tail(5),
        "top_k": lambda lf: lf.top_k(3, by="c0"),
        "gather_every": lambda lf: lf.gather_every(2),
        # --- combining
        "join_left_on_key": lambda lf: lf.join(
            lf.group_by("h0").agg(c("c0").mean().alias("mu")), on="h0", how="left"
        ),
        "join_cross_horizon(lift small frame)": lambda lf: lf.select("h0")
        .unique()
        .join(
            same_backend_small(
                lf, {"time": [t0 + pd.Timedelta(days=i) for i in range(1, 4)]}
            ),
            how="cross",
        ),
        "concat_vertical": lambda lf: nw.concat([lf, lf], how="vertical"),
        "concat_diagonal": lambda lf: nw.concat(
            [lf, lf.select("h0", "time", "c0")], how="diagonal"
        ),
        "unpivot": lambda lf: lf.unpivot(on=["c0", "c1"], index=["h0", "time"]),
        "join_asof": lambda lf: lf.sort("time").join_asof(
            lf.select("h0", "time", c("c1").alias("c1b")).sort("time"),
            on="time",
            by="h0",
        ),
        # --- escape hatches
        "map_batches": lambda lf: lf.with_columns(
            mb=c("c0").map_batches(lambda s: s * 2, return_dtype=nw.Float64())
        ),
        # --- eager-only API probed on the LazyFrame
        "iter_groups": lambda lf: list(iter(lf.group_by("h0"))),
        "to_numpy": lambda lf: lf.to_numpy(),
        "pivot": lambda lf: lf.pivot(on="h0", index="time", values="c0"),
        "lazy_len(select nw.len)": lambda lf: lf.select(nw.len()),
    }


# row-count-preserving ops: check whether input row order survives collect()
ORDER_OPS = {
    "with_columns",
    "over_mean(no order)",
    "over_cum_sum(order_by)",
    "over_shift(order_by)",
    "join_left_on_key",
    "map_batches",
}


def order_preserved(res, pdf):
    got = res.collect(backend="polars").to_native().select("h0", "time")
    exp = pl.from_pandas(pdf[["h0", "time"]])
    got = got.with_columns(pl.col("time").cast(pl.Datetime("us")))
    exp = exp.with_columns(pl.col("time").cast(pl.Datetime("us")))
    return bool(got.equals(exp))


# ops whose output order/content is not deterministic -> only check row count
NONDET = {"head", "tail", "gather_every", "unique_keys", "top_k"}


def to_ref_polars(x):
    """narwhals LazyFrame/DataFrame -> canonical polars.DataFrame for comparison."""
    if isinstance(x, nw.LazyFrame):
        df = x.collect(backend="polars").to_native()
    elif isinstance(x, nw.DataFrame):
        df = x.to_polars()
    else:
        return None
    casts = {}
    for name, dt in df.schema.items():
        if dt.is_temporal():
            casts[name] = pl.Datetime("us")
        elif dt.is_integer() or dt.is_float():
            casts[name] = pl.Float64
        elif dt == pl.Categorical or isinstance(dt, pl.Enum):
            casts[name] = pl.String
    df = df.cast(casts) if casts else df
    df = df.select(sorted(df.columns))
    return df.sort(df.columns, nulls_last=True)


def compare(a, b, nondet):
    if a is None or b is None:
        return "n/a"
    if nondet:
        return "rows-match" if a.height == b.height else f"rows {a.height}!={b.height}"
    from polars.testing import assert_frame_equal

    try:
        assert_frame_equal(a, b, check_dtypes=False, check_exact=False, rel_tol=1e-9)
        return "match"
    except AssertionError as e:
        return "MISMATCH: " + str(e).splitlines()[0][:120]


def part_c(nat, sink_dir):
    builders = nat.builders()
    first_key = nat.pdf["h0"].iloc[0]
    specs = op_specs(first_key)
    specs["sink_parquet"] = None  # special-cased

    # reference on polars eager via lazy API
    ref = {}
    ref_lf = nw.from_native(builders["polars_eager"]()).lazy()
    for op, fn in specs.items():
        if fn is None:
            continue
        try:
            ref[op] = to_ref_polars(fn(ref_lf))
        except Exception:  # noqa: BLE001
            ref[op] = None

    out = {}
    for name, build in builders.items():
        native = build()
        orig_type = type(native)
        obj = nw.from_native(native)
        lf = obj.lazy() if isinstance(obj, nw.DataFrame) else obj
        row = {}
        for op, fn in specs.items():
            cell = {}
            try:
                if op == "sink_parquet":
                    path = os.path.join(sink_dir, f"{name}.parquet")
                    with ComputeCounter() as cc:
                        lf.sink_parquet(path)
                    cell["status"] = "ok (written)"
                    cell["computes"] = cc.n
                    row[op] = cell
                    continue
                with ComputeCounter() as cc:
                    res = fn(lf)
                cell["computes_while_building"] = cc.n
                if isinstance(res, nw.LazyFrame):
                    nat_res = res.to_native()
                    cell["native_out"] = short_type(nat_res)
                    # ibis returns Table subclasses (e.g. ibis.Join) -> isinstance
                    cell["same_native_type"] = isinstance(nat_res, orig_type)
                    cell["status"] = "ok-lazy" if lf is obj else "ok"
                    if obj is not lf:
                        cell["status"] = "ok (eager backend via .lazy())"
                    try:
                        cell["check"] = compare(
                            to_ref_polars(res), ref.get(op), op in NONDET
                        )
                    except Exception as e:  # noqa: BLE001
                        cell["check"] = "collect-error: " + err_str(e)
                    if op in ORDER_OPS and not cell["check"].startswith("collect"):
                        cell["order_preserved"] = order_preserved(res, nat.pdf)
                elif isinstance(res, (nw.Schema, dict)):
                    cell["status"] = "ok (metadata)"
                else:
                    cell["status"] = f"ok -> {short_type(res)}"
            except Exception as e:  # noqa: BLE001
                cell["status"] = "ERROR"
                cell["error"] = err_str(e)
            row[op] = cell
        out[name] = row
    return out


# ----------------------------------------------------------------------- part D
def part_d(nat):
    """Eager-only ops on eager narwhals DataFrames (native eager + collected lazy)."""
    builders = nat.builders()
    out = {}
    for name, build in builders.items():
        obj = nw.from_native(build())
        variants = {}
        if isinstance(obj, nw.DataFrame):
            variants[f"{name}"] = obj
        else:
            variants[f"{name}.collect()"] = obj.collect()
        for vname, df in variants.items():
            rec = {"eager_impl": str(df.implementation)}
            tests = {
                "iter_groups": lambda d: sum(1 for _ in d.group_by("h0")),
                "iter_groups_keys_type": lambda d: type(next(iter(d.group_by("h0")))[0]).__name__,
                "to_numpy": lambda d: d.select("c0", "c1").to_numpy().shape,
                "pivot": lambda d: d.pivot(on="h0", index="time", values="c0").shape,
                "partition_by(native-only?)": lambda d: len(d.partition_by("h0")),
                "group_by.agg.last": lambda d: d.group_by("h0").agg(nw.col("c0").last()).shape,
                "item/len": lambda d: d.select(nw.len()).item(),
            }
            for tname, fn in tests.items():
                try:
                    rec[tname] = repr(fn(df))
                except Exception as e:  # noqa: BLE001
                    rec[tname] = err_str(e)
            out[vname] = rec
    return out


def md_matrix(c_res):
    ops = list(next(iter(c_res.values())).keys())
    backs = list(c_res.keys())
    lines = ["| op | " + " | ".join(backs) + " |", "|---" * (len(backs) + 1) + "|"]
    for op in ops:
        cells = []
        for b in backs:
            cell = c_res[b][op]
            st = cell.get("status", "")
            if st == "ERROR":
                s = "ERR"
            elif st.startswith("ok"):
                if "native_out" in cell:
                    s = "lazy" if cell.get("same_native_type") else "TYPE-CHANGED"
                    if "(eager backend" in st:
                        s = "eager"
                    chk = cell.get("check", "")
                    if chk.startswith("MISMATCH"):
                        s += " WRONG"
                    elif chk.startswith("collect-error"):
                        s += " COLLECT-ERR"
                    if cell.get("order_preserved") is False:
                        s += " (reordered)"
                    if cell.get("computes_while_building"):
                        s += f" ({cell['computes_while_building']} compute)"
                else:
                    s = st.replace("ok ", "")
            else:
                s = st
            cells.append(s)
        lines.append(f"| {op} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=f"pd{pd.__version__.split('.')[0]}")
    ap.add_argument("--n-series", type=int, default=20)
    ap.add_argument("--n-time", type=int, default=30)
    args = ap.parse_args()

    pdf = make_long(args.n_series, args.n_time)
    nat = Natives(pdf)
    info = env_info()
    for m in ["ibis", "modin"]:
        try:
            info[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            info[m] = None
    info["python"] = sys.version.split()[0]

    res = {"env": info, "n_rows": len(pdf)}
    res["AB"] = part_ab(nat)
    with tempfile.TemporaryDirectory(dir=os.environ.get("X1_TMP")) as d:
        res["C"] = part_c(nat, d)
    res["D"] = part_d(nat)
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, f"x1_matrix_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=str)

    print("ENV", json.dumps(info))
    print("\n## A/B: from_native + collect")
    for b, rec in res["AB"].items():
        print(f"\n### {b}: native={rec['native_type']} nw={rec['nw_class']} "
              f"impl={rec['implementation']} eager_only={rec['eager_only=True']}")
        if "DataFrame.lazy(backend)" in rec:
            for k, v in rec["DataFrame.lazy(backend)"].items():
                print(f"  .lazy({k}) -> {v}")
        for k, v in rec["collect(backend)"].items():
            print(f"  .collect({k}) -> {v}")
        if "collect(**kwargs)" in rec:
            print(f"  .collect(**kwargs): {rec['collect(**kwargs)']}")
    print("\n## C: op matrix (LazyFrame API)")
    print(md_matrix(res["C"]))
    print("\n## C errors")
    for b, row in res["C"].items():
        for op, cell in row.items():
            if cell.get("status") == "ERROR":
                print(f"  {b} / {op}: {cell['error']}")
            elif str(cell.get("check", "")).startswith(("MISMATCH", "collect-error")):
                print(f"  {b} / {op}: {cell['check']}")
    print("\n## D: eager-only ops")
    for v, rec in res["D"].items():
        print(f"  {v}: {rec}")
    print("\nwrote", path)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
