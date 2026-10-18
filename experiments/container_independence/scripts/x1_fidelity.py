"""x1: dtype / null fidelity of narwhals collect() targets and round trips.

Input: one small table with "tricky" dtypes (tz-aware timestamps, date, int with
nulls, float with BOTH null and NaN, bool with nulls, categorical/dictionary,
uint8, duration, decimal). Built once in pyarrow, then lifted into each lazy
backend. For each backend we record per column (dtype, #null, #NaN) for:

  input                 : the native lazy input, materialised natively
  nw_collect_<be>       : nw.from_native(L).collect(backend=be)  (default/pandas/polars/pyarrow)
  rt_nw                 : nw.from_native(L).collect().lazy(backend=impl) -> native collect
  rt_manual             : backend's natural eager form -> rebuilt lazy -> native collect

and print only the columns that changed (dtype or null/NaN counts) vs input.

Run::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x1_fidelity.py
"""

import argparse
import datetime as dt
import decimal
import json
import os
import sys

import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402

import narwhals as nw  # noqa: E402

from common import env_info  # noqa: E402

import dask  # noqa: E402
import dask.dataframe as dd  # noqa: E402
import duckdb  # noqa: E402

dask.config.set(scheduler="synchronous")
try:
    import ibis
except ImportError:
    ibis = None

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def make_arrow(n=12):
    rng = np.random.default_rng(0)
    ts = [dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=i) for i in range(n)]
    i_vals = [int(x) if k % 4 else None for k, x in enumerate(rng.integers(0, 100, n))]
    f_vals = [float(x) for x in rng.normal(size=n)]
    f_vals[1] = None
    f_vals[2] = float("nan")
    b_vals = [bool(k % 2) if k % 5 else None for k in range(n)]
    cats = ["a", "b", "c"]
    return pa.table(
        {
            "key": pa.array([f"s{k % 3}" for k in range(n)], pa.string()),
            "time_tz": pa.array(ts, pa.timestamp("us", tz="UTC")),
            "time_ns": pa.array([t.replace(tzinfo=None) for t in ts], pa.timestamp("ns")),
            "date": pa.array([t.date() for t in ts], pa.date32()),
            "i_null": pa.array(i_vals, pa.int64()),
            "f_null_nan": pa.array(f_vals, pa.float64()),
            "b_null": pa.array(b_vals, pa.bool_()),
            "cat": pa.array([cats[k % 3] for k in range(n)]).dictionary_encode(),
            "u8": pa.array(list(range(n)), pa.uint8()),
            "dur": pa.array([dt.timedelta(hours=k) for k in range(n)], pa.duration("us")),
            "dec": pa.array([decimal.Decimal(f"{k}.25") for k in range(n)], pa.decimal128(10, 2)),
        }
    )


# --------------------------------------------------------------- summarise eager
def summary(obj):
    """native eager frame -> {col: (dtype, n_null, n_nan)}."""
    out = {}
    if isinstance(obj, pd.DataFrame):
        for c in obj.columns:
            s = obj[c]
            n_null = int(s.isna().sum())
            n_nan = 0
            if isinstance(s.dtype, pd.Float64Dtype):
                arr = s.array
                n_nan = int(np.isnan(arr._data[~arr._mask]).sum())
            elif isinstance(s.dtype, pd.ArrowDtype) and pa.types.is_floating(
                s.dtype.pyarrow_dtype
            ):
                n_nan = int(pc.sum(pc.is_nan(s.array._pa_array)).as_py() or 0)
            elif pd.api.types.is_float_dtype(s.dtype):
                # numpy float: NaN and null are the same thing
                n_nan = "n/a(NaN==null)"
            out[c] = (str(s.dtype), n_null, n_nan)
    elif isinstance(obj, pl.DataFrame):
        for c in obj.columns:
            s = obj[c]
            n_nan = int(s.is_nan().sum()) if s.dtype.is_float() else 0
            out[c] = (str(s.dtype), int(s.null_count()), n_nan)
    elif isinstance(obj, pa.Table):
        for c in obj.column_names:
            col = obj[c]
            n_nan = (
                int(pc.sum(pc.is_nan(col)).as_py() or 0)
                if pa.types.is_floating(col.type)
                else 0
            )
            out[c] = (str(col.type), int(col.null_count), n_nan)
    else:
        raise TypeError(type(obj))
    return out


def native_materialise(x):
    if isinstance(x, pl.LazyFrame):
        return x.collect()
    if isinstance(x, dd.DataFrame):
        return x.compute()
    if isinstance(x, duckdb.DuckDBPyRelation):
        return x.to_arrow_table()
    if ibis is not None and isinstance(x, ibis.Table):
        return x.to_pyarrow()
    return x


def build_inputs(arrow, icon):
    out = {"polars_lazy": pl.from_arrow(arrow).lazy()}
    # pandas with nullable extension dtypes, so dask starts lossless-ish
    mapper = {
        pa.int64(): pd.Int64Dtype(),
        pa.bool_(): pd.BooleanDtype(),
        pa.float64(): pd.Float64Dtype(),
        pa.string(): pd.StringDtype("pyarrow"),
    }.get
    pdf = arrow.to_pandas(types_mapper=mapper)
    out["dask"] = dd.from_pandas(pdf, npartitions=2)
    out["duckdb"] = duckdb.from_arrow(arrow)
    notes = {}
    if ibis is not None:
        t = icon.create_table("x1_fid", ibis.memtable(arrow))
        try:
            t.to_pyarrow()
        except Exception as e:  # noqa: BLE001
            notes["ibis"] = (
                f"full table not materialisable ({type(e).__name__}: {str(e)[:120]}); "
                "dropped column 'dur'"
            )
            t = icon.create_table("x1_fid2", ibis.memtable(arrow.drop_columns(["dur"])))
        out["ibis"] = t
    return out, pdf, notes


def rt_manual(name, L):
    if name == "polars_lazy":
        return L.collect().lazy()
    if name == "dask":
        return dd.from_pandas(L.compute(), npartitions=L.npartitions)
    if name == "duckdb":
        return duckdb.from_arrow(L.to_arrow_table())
    if name == "ibis":
        return ibis.memtable(L.to_pyarrow())
    raise ValueError(name)


import re  # noqa: E402


def family(dtype):
    d = dtype.lower()
    if "interval" in d:
        return "interval"
    if "dictionary" in d or "categor" in d:
        return "categorical"
    if "decimal" in d:
        return "decimal"
    if "datetime" in d or "timestamp" in d:
        m = re.search(r"tz=([^\]\)]+)|time_zone='([^']+)'|, ([A-Za-z_]+/?[A-Za-z_]*)\]", dtype)
        tz = next((g for g in (m.groups() if m else ()) if g), None)
        return f"datetime[tz={tz}]"
    if d.startswith("date"):
        return "date"
    if "duration" in d or "timedelta" in d:
        return "duration"
    if "uint" in d:
        return "uint"
    if "int" in d:
        return "int"
    if "float" in d or "double" in d:
        return "float"
    if "bool" in d:
        return "bool"
    if "string" in d or d == "str":
        return "string"
    return d


def lossy(a, b):
    """semantic loss between two (dtype, n_null, n_nan) summaries."""
    if b == "MISSING":
        return "column missing"
    reasons = []
    if family(a[0]) != family(b[0]):
        reasons.append(f"{family(a[0])}->{family(b[0])}")
    if a[1] != b[1]:
        reasons.append(f"nulls {a[1]}->{b[1]}")
    if a[2] != b[2]:
        reasons.append(f"NaN {a[2]}->{b[2]}")
    return "; ".join(reasons)


def diff(base, other):
    d = {}
    for c in base:
        if c not in other:
            d[c] = (base[c], "MISSING")
        elif base[c] != other[c]:
            d[c] = (base[c], other[c])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=f"pd{pd.__version__.split('.')[0]}")
    args = ap.parse_args()
    arrow = make_arrow()
    icon = ibis.duckdb.connect(threads=2) if ibis is not None else None
    inputs, pdf, notes = build_inputs(arrow, icon)
    res = {"env": env_info(), "arrow_input": summary(arrow), "backends": {}}
    print("ENV", json.dumps(res["env"]))
    print("arrow source:", json.dumps(res["arrow_input"]))
    print("pandas fed to dask:", json.dumps(summary(pdf)))
    for name, L in inputs.items():
        rec = {}
        base = summary(native_materialise(L))
        rec["input"] = base
        rec["ingest_changes_vs_arrow_source"] = diff(res["arrow_input"], base)
        if name in notes:
            rec["note"] = notes[name]
            print(f"\nNOTE {name}: {notes[name]}")
        lf = nw.from_native(L)
        routes = {
            "nw_collect_default": lambda: lf.collect().to_native(),
            "nw_collect_pandas": lambda: lf.collect(backend="pandas").to_native(),
            "nw_collect_polars": lambda: lf.collect(backend="polars").to_native(),
            "nw_collect_pyarrow": lambda: lf.collect(backend="pyarrow").to_native(),
            "rt_nw": lambda: native_materialise(
                lf.collect().lazy(backend=lf.implementation).to_native()
            ),
            "rt_manual": lambda: native_materialise(rt_manual(name, L)),
        }
        print(f"\n=== {name}: native ingest changes vs arrow source:")
        for c, (a, b) in rec["ingest_changes_vs_arrow_source"].items():
            tag = f"   <-- LOSSY: {lossy(a, b)}" if lossy(a, b) else ""
            print(f"      {c:11s} {a} -> {b}{tag}")
        print("  -- routes, compared with the backend's own native materialisation:")
        for rname, fn in routes.items():
            try:
                out = fn()
                s = summary(out)
                dff = diff(base, s)
                loss = {c: lossy(a, b) for c, (a, b) in dff.items() if lossy(a, b)}
                rec[rname] = {"type": type(out).__module__.split(".")[0] + "." + type(out).__name__,
                              "changed": dff, "lossy": loss}
                print(f"  {rname:20s} -> {rec[rname]['type']}: "
                      f"{len(dff)} dtype-name change(s), {len(loss)} LOSSY")
                for c, (a, b) in dff.items():
                    tag = f"   <-- LOSSY: {loss[c]}" if c in loss else ""
                    print(f"      {c:11s} {a} -> {b}{tag}")
            except Exception as e:  # noqa: BLE001
                rec[rname] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                print(f"  {rname:20s} -> ERROR {rec[rname]['error']}")
        res["backends"][name] = rec
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, f"x1_fidelity_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("\nwrote", path)


if __name__ == "__main__":
    sys.exit(main())
