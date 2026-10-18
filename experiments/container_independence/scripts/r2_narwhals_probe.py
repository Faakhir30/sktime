"""r2_narwhals: empirical verification of documented narwhals semantics.

Workstream r2_narwhals (prior art / narwhals deep-dive). This script does NOT
benchmark a full design; it verifies, on the installed narwhals version, the
claims made in the narwhals docs/source and the two contested dev claims:

  C1  "Polars lazy collects to Polars by default, but Narwhals' collect also sends
       Dask to pandas and DuckDB to Arrow."
  C2  "No MultiIndex. Narwhals dataframes are columns only. The pandas path would
       have to reset_index into columns before grouping. That is a copy, and last
       time the index that came back was a RangeIndex, so level names were gone."

Sections
  A  LazyFrame.collect / to_native / DataFrame.lazy semantics per backend    (C1)
  B  eager-vs-lazy API surface, lazy-unavailable ops, order-dependence rules
  C  pandas index policy: from_native on MultiIndex, preserve/reset per op,
     maybe_* helpers, reset_index round trip, dtype mapping, copy check     (C2)
  D  window / grouped expressions per backend (over(partition_by, order_by),
     rolling_*, ewm_mean, cum_sum, shift, diff, fill_null) + correctness
  E  grouped iteration: order guarantees and per-group cost per backend
  F  per-call narwhals overhead on tiny frames (forecaster hot-loop proxy)
  G  import footprint

Run (from worktree root):
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/r2_narwhals_probe.py --grid dev
    (also with $EXP_PY3 for pandas 3)
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

from common import DEV_GRID, LARGE_GRID, bench, env_info, make_panel, rss_mb

warnings.filterwarnings("ignore")

import narwhals as nw  # noqa: E402
import narwhals.stable.v1 as nw1  # noqa: E402
import narwhals.stable.v2 as nw2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")

OUT = {}


def tname(x):
    t = type(x)
    return f"{t.__module__.split('.')[0]}.{t.__qualname__}"


def attempt(fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip().splitlines()[0][:220] if str(e).strip() else ""
        return {"ok": False, "error": f"{type(e).__name__}: {msg}"}


def log(*a):
    print(*a, flush=True)


def flat_panel(n_series=3, n_time=6, time_kind="datetime"):
    X = make_panel(n_series=n_series, n_time=n_time, n_levels=1, time_kind=time_kind)
    return X, X.reset_index()


def lazy_natives(flat):
    """Native lazy objects for each installed lazy backend."""
    out = {}
    import polars as pl

    out["polars_lazy"] = pl.from_pandas(flat).lazy()
    try:
        import dask.dataframe as dd

        out["dask"] = dd.from_pandas(flat, npartitions=2)
    except Exception as e:  # noqa: BLE001
        log("dask unavailable", e)
    try:
        import duckdb

        out["duckdb"] = duckdb.from_df(flat)
    except Exception as e:  # noqa: BLE001
        log("duckdb unavailable", e)
    try:
        import ibis

        out["ibis"] = ibis.memtable(flat)
    except Exception as e:  # noqa: BLE001
        log("ibis unavailable", e)
    return out


def eager_natives(flat):
    import polars as pl
    import pyarrow as pa

    return {
        "pandas": flat,
        "polars": pl.from_pandas(flat),
        "pyarrow": pa.Table.from_pandas(flat, preserve_index=False),
    }


# --------------------------------------------------------------------------- A
def section_a():
    log("\n=== A: collect / to_native / lazy semantics ===")
    _, flat = flat_panel()
    res = {}
    for name, native in lazy_natives(flat).items():
        r = {"native_type": tname(native)}
        lf = nw.from_native(native)
        r["from_native_type"] = type(lf).__name__
        r["to_native_type"] = tname(lf.to_native())
        r["to_native_after_with_columns_type"] = tname(
            lf.with_columns(c1=nw.col("c0") * 2).to_native()
        )
        r["to_native_is_same_backend"] = (
            tname(lf.with_columns(c1=nw.col("c0") * 2).to_native()) == tname(native)
        )
        r["collect_default"] = tname(lf.collect().to_native())
        for b in ["pandas", "polars", "pyarrow", "dask", "duckdb", "ibis"]:
            a = attempt(lambda b=b: tname(lf.collect(b).to_native()))
            r[f"collect[{b}]"] = a["value"] if a["ok"] else a["error"]
        # stable v1: duckdb/ibis are interchange-level DataFrame there
        a = attempt(lambda native=native: type(nw1.from_native(native)).__name__)
        r["stable_v1_from_native_type"] = a["value"] if a["ok"] else a["error"]
        a = attempt(lambda native=native: type(nw2.from_native(native)).__name__)
        r["stable_v2_from_native_type"] = a["value"] if a["ok"] else a["error"]
        res[name] = r
        log(name, json.dumps(r, indent=1))

    # DataFrame.lazy(backend=...) from an eager pandas frame
    df = nw.from_native(flat)
    lz = {}
    for b in [None, "polars", "dask", "duckdb", "ibis", "pandas", "pyarrow"]:
        a = attempt(lambda b=b: tname(df.lazy(b).to_native()))
        lz[str(b)] = a["value"] if a["ok"] else a["error"]
    res["pandas_eager.lazy(backend)"] = lz
    log("pandas DataFrame.lazy(backend):", json.dumps(lz, indent=1))

    # polars eager -> lazy() without backend keeps polars
    import polars as pl

    a = attempt(lambda: tname(nw.from_native(pl.from_pandas(flat)).lazy().to_native()))
    res["polars_eager.lazy(None)"] = a["value"] if a["ok"] else a["error"]
    OUT["A_collect"] = res


# --------------------------------------------------------------------------- B
def section_b():
    log("\n=== B: eager vs lazy API surface ===")
    pub = lambda cls: {n for n in dir(cls) if not n.startswith("_")}  # noqa: E731
    only_eager = sorted(pub(nw.DataFrame) - pub(nw.LazyFrame))
    only_lazy = sorted(pub(nw.LazyFrame) - pub(nw.DataFrame))
    from narwhals.group_by import GroupBy, LazyGroupBy

    gb_only_eager = sorted(
        {n for n in dir(GroupBy) if not n.startswith("__") or n == "__iter__"}
        - {n for n in dir(LazyGroupBy) if not n.startswith("__") or n == "__iter__"}
    )
    log("DataFrame-only public methods:", only_eager)
    log("LazyFrame-only public methods:", only_lazy)
    log("GroupBy-only (vs LazyGroupBy):", gb_only_eager)

    _, flat = flat_panel()
    probes = {}
    for name, native in lazy_natives(flat).items():
        lf = nw.from_native(native)
        p = {}
        p["iter(group_by)"] = attempt(lambda lf=lf: len(list(lf.group_by("h0"))))
        p["lf['c0']"] = attempt(lambda lf=lf: tname(lf["c0"]))
        p["len(lf)"] = attempt(lambda lf=lf: len(lf))
        p["lf.to_numpy()"] = attempt(lambda lf=lf: tname(lf.to_numpy()))
        p["lf.iter_rows()"] = attempt(lambda lf=lf: next(iter(lf.iter_rows())))
        p["lf.shape"] = attempt(lambda lf=lf: lf.shape)
        p["lf.partition_by"] = attempt(lambda lf=lf: len(lf.partition_by("h0")))
        p["cum_sum (no over)"] = attempt(
            lambda lf=lf: tname(
                lf.with_columns(x=nw.col("c0").cum_sum()).collect().to_native()
            )
        )
        p["cum_sum.over(order_by)"] = attempt(
            lambda lf=lf: tname(
                lf.with_columns(x=nw.col("c0").cum_sum().over(order_by="time"))
                .collect()
                .to_native()
            )
        )
        p["group_by.agg(mean)"] = attempt(
            lambda lf=lf: tname(
                lf.group_by("h0").agg(nw.col("c0").mean()).collect().to_native()
            )
        )
        p["collect_schema"] = attempt(lambda lf=lf: str(lf.collect_schema()))
        for k, v in p.items():
            p[k] = v["value"] if v["ok"] else v["error"]
            if isinstance(p[k], tuple):
                p[k] = str(p[k])
        probes[name] = p
        log(name, json.dumps(p, indent=1, default=str))
    OUT["B_api"] = {
        "dataframe_only": only_eager,
        "lazyframe_only": only_lazy,
        "groupby_only_vs_lazygroupby": gb_only_eager,
        "lazy_probes": probes,
    }


# --------------------------------------------------------------------------- C
def _idx_info(native):
    idx = native.index
    return {
        "index_type": type(idx).__name__,
        "names": [str(n) for n in idx.names],
        "nlevels": idx.nlevels,
    }


def section_c(grid):
    log("\n=== C: pandas index policy ===")
    res = {}
    for time_kind in ["datetime", "period", "int"]:
        X, _ = flat_panel(n_series=4, n_time=5, time_kind=time_kind)
        d = nw.from_native(X)
        r = {"input": _idx_info(X), "nw_columns": d.columns}
        r["nw_schema"] = {k: str(v) for k, v in d.schema.items()}
        r["maybe_get_index"] = attempt(lambda d=d: _idx_info(pd.DataFrame(index=nw.maybe_get_index(d))))
        r["group_by(index level 'h0')"] = attempt(
            lambda d=d: len(d.group_by("h0").agg(nw.col("c0").mean()))
        )
        ops = {
            "with_columns": lambda d: d.with_columns(c1=nw.col("c0") * 2),
            "select": lambda d: d.select("c0"),
            "select(expr)": lambda d: d.select(nw.col("c0") + 1),
            "filter": lambda d: d.filter(nw.col("c0") > d["c0"].min()),
            "sort": lambda d: d.sort("c0"),
            "head": lambda d: d.head(3),
            "rename": lambda d: d.rename({"c0": "z"}),
            "cast": lambda d: d.cast({"c0": nw.Float32}),
            "drop_nulls": lambda d: d.drop_nulls(),
            "with_row_index": lambda d: d.with_row_index("r"),
            "concat_vertical": lambda d: nw.concat([d, d]),
            "unique": lambda d: d.unique(),
            "group_by(col).agg": lambda d: d.with_columns(k=nw.col("c0") > 0)
            .group_by("k")
            .agg(nw.col("c0").mean()),
            "join(on col)": lambda d: d.join(
                d.with_columns(c9=nw.col("c0") * 3), on="c0", how="left"
            ),
            "Series.__array__/to_numpy": lambda d: d,  # placeholder
            "maybe_reset_index": lambda d: nw.maybe_reset_index(d),
        }
        per_op = {}
        for opname, f in ops.items():
            a = attempt(lambda f=f, d=d: _idx_info(nw.to_native(f(d))))
            per_op[opname] = a["value"] if a["ok"] else a["error"]
        per_op.pop("Series.__array__/to_numpy")
        r["ops_index_after"] = per_op

        # iterate group_by on a column: do sub-frames keep the original MultiIndex?
        Xk = X.assign(k=X.index.get_level_values("h0"))
        dk = nw.from_native(Xk)
        first = next(iter(dk.group_by("k")))
        r["iter(group_by col) subframe index"] = _idx_info(nw.to_native(first[1]))
        r["iter(group_by col) key"] = repr(first[0])

        # the C2 path: reset_index into columns, group with narwhals, restore
        flat = X.reset_index()
        r["reset_index: columns"] = list(flat.columns)
        r["reset_index: new index"] = type(flat.index).__name__
        dfl = nw.from_native(flat)
        r["reset_index: nw schema"] = {k: str(v) for k, v in dfl.schema.items()}
        agg = dfl.group_by("h0").agg(nw.col("c0").mean())
        r["reset_index: group_by(h0).agg ok, n"] = len(agg)
        back = nw.to_native(dfl.with_columns(c1=nw.col("c0") * 2)).set_index(
            list(X.index.names)
        )
        r["round_trip set_index: names restored"] = list(back.index.names) == list(
            X.index.names
        )
        r["round_trip set_index: index equal"] = bool(back.index.equals(X.index))
        r["round_trip: time level dtype"] = str(back.index.levels[-1].dtype)
        # what if the level is unnamed?
        Xu = X.copy()
        Xu.index = Xu.index.set_names([None, None])
        r["reset_index of unnamed levels -> columns"] = list(Xu.reset_index().columns)
        res[time_kind] = r
        log(time_kind, json.dumps(r, indent=1, default=str))

    # Period-typed time column (sktime's PeriodIndex) once it is a column
    X, flat = flat_panel(n_series=3, n_time=5, time_kind="period")
    dp = nw.from_native(flat)
    per = {"schema[time]": str(dp.schema["time"])}
    per["pandas: sort('time')"] = attempt(lambda: len(dp.sort("h0", "time")))
    per["pandas: shift.over(h0, order_by=time)"] = attempt(
        lambda: str(
            dp.with_columns(x=nw.col("c0").shift(1).over("h0", order_by="time"))
            .to_native()["x"]
            .isna()
            .sum()
        )
    )
    per["pandas: time.dt.year()"] = attempt(
        lambda: dp.select(nw.col("time").dt.year()).to_native().iloc[0, 0].item()
    )
    per["pandas: time max (scalar)"] = attempt(lambda: repr(dp["time"].max()))
    per["pandas: filter time > first"] = attempt(
        lambda: len(dp.filter(nw.col("time") > flat["time"].iloc[0]))
    )
    per["to_polars()"] = attempt(lambda: str(dp.to_polars().schema["time"]))
    per["to_arrow()"] = attempt(lambda: str(dp.to_arrow().schema.field("time").type))
    per["lazy('polars')"] = attempt(
        lambda: str(dp.lazy("polars").collect().schema["time"])
    )
    for k, v in list(per.items()):
        if isinstance(v, dict):
            per[k] = v["value"] if v["ok"] else v["error"]
    res["period_column_probe"] = per
    log("period column probe", json.dumps(per, indent=1, default=str))

    # copy / cost check for reset_index (pandas version dependent: CoW in pandas 3)
    cost = []
    sizes = [(n, t) for n, t in grid if n * t <= 2_000_000]
    for n_series, n_time in sizes:
        X = make_panel(n_series=n_series, n_time=n_time, n_levels=1)
        flat = X.reset_index()
        shares = bool(np.shares_memory(X["c0"].to_numpy(), flat["c0"].to_numpy()))
        t_reset = bench(lambda X=X: X.reset_index(), repeat=5)
        t_reset_set = bench(
            lambda X=X: X.reset_index().set_index(["h0", "time"]), repeat=3
        )

        def native_iter(X=X):
            for _k, _g in X.groupby(level=0, sort=False):
                pass

        t_iter = bench(native_iter, repeat=3)
        row = {
            "n_series": n_series,
            "n_time": n_time,
            "rows": n_series * n_time,
            "value_col_shares_memory_after_reset_index": shares,
            "t_reset_index_s": t_reset,
            "t_reset_then_set_index_s": t_reset_set,
            "t_native_groupby_level_iter_s": t_iter,
            "reset_index_over_iter_ratio": t_reset / t_iter,
            "reset_and_set_over_iter_ratio": t_reset_set / t_iter,
        }
        cost.append(row)
        log("reset_index cost", json.dumps(row))
    OUT["C_index"] = {"per_time_kind": res, "reset_index_cost": cost}


# --------------------------------------------------------------------------- D
def section_d():
    log("\n=== D: window expressions per backend ===")
    _, flat = flat_panel(n_series=3, n_time=8, time_kind="datetime")
    # inject a null to test fill_null
    flat.loc[3, "c0"] = np.nan
    flat = flat.sample(frac=1.0, random_state=0).reset_index(drop=True)  # shuffle rows
    ref_sorted = flat.sort_values(["h0", "time"]).reset_index(drop=True)
    g = ref_sorted.groupby("h0")["c0"]
    refs = {
        "mean.over(h0)": g.transform("mean"),
        "cum_sum.over(h0, order_by=time)": g.cumsum(),
        "shift(1).over(h0, order_by=time)": g.shift(1),
        "diff().over(h0, order_by=time)": g.diff(),
        "rolling_mean(3).over(h0, order_by=time)": g.transform(
            lambda s: s.rolling(3).mean()
        ),
        "rolling_std(3).over(h0, order_by=time)": g.transform(
            lambda s: s.rolling(3).std()
        ),
        "ewm_mean(alpha=.5).over(h0, order_by=time)": g.transform(
            lambda s: s.ewm(alpha=0.5, ignore_na=False).mean()
        ),
        "fill_null(forward).over(h0, order_by=time)": g.ffill(),
    }
    exprs = {
        "mean.over(h0)": nw.col("c0").mean().over("h0"),
        "cum_sum.over(h0, order_by=time)": nw.col("c0")
        .cum_sum()
        .over("h0", order_by="time"),
        "shift(1).over(h0, order_by=time)": nw.col("c0")
        .shift(1)
        .over("h0", order_by="time"),
        "diff().over(h0, order_by=time)": nw.col("c0").diff().over("h0", order_by="time"),
        "rolling_mean(3).over(h0, order_by=time)": nw.col("c0")
        .rolling_mean(3)
        .over("h0", order_by="time"),
        "rolling_std(3).over(h0, order_by=time)": nw.col("c0")
        .rolling_std(3)
        .over("h0", order_by="time"),
        "ewm_mean(alpha=.5).over(h0, order_by=time)": nw.col("c0")
        .ewm_mean(alpha=0.5)
        .over("h0", order_by="time"),
        "fill_null(forward).over(h0, order_by=time)": nw.col("c0")
        .fill_null(strategy="forward")
        .over("h0", order_by="time"),
    }
    natives = {**eager_natives(flat), **lazy_natives(flat)}
    table = {}
    for bname, native in natives.items():
        row = {}
        for ename, ex in exprs.items():

            def run(native=native, ex=ex, ename=ename):
                fr = nw.from_native(native)
                out = fr.with_columns(x=ex)
                if isinstance(out, nw.LazyFrame):
                    out = out.collect("pandas")
                pdf = out.to_pandas() if not isinstance(out.to_native(), pd.DataFrame) else out.to_native()
                pdf = pdf.sort_values(["h0", "time"]).reset_index(drop=True)
                got = pd.to_numeric(pdf["x"], errors="coerce").to_numpy(dtype=float)
                exp = refs[ename].to_numpy(dtype=float)
                if np.allclose(got, exp, equal_nan=True):
                    return "ok"
                nan_diff = int(np.sum(np.isnan(got) != np.isnan(exp)))
                both = ~np.isnan(got) & ~np.isnan(exp)
                maxd = float(np.max(np.abs(got[both] - exp[both]))) if both.any() else 0.0
                where = [
                    f"row{i}: got={got[i]!r} pandas_ref={exp[i]!r}"
                    for i in np.where(np.isnan(got) != np.isnan(exp))[0][:2]
                ]
                return (
                    f"MISMATCH nan_pattern_diffs={nan_diff} max|d|(finite)={maxd:.3g} "
                    + "; ".join(where)
                )

            a = attempt(run)
            row[ename] = a["value"] if a["ok"] else a["error"]
        table[bname] = row
        log(bname, json.dumps(row, indent=1))
    OUT["D_windows"] = table


# --------------------------------------------------------------------------- H
def section_h():
    """ewm_mean null semantics: plain vs .over(partition_by, order_by), pandas vs polars."""
    log("\n=== H: ewm_mean null handling, plain vs over ===")
    import polars as pl

    data = {
        "g": ["a"] * 4 + ["b"] * 4,
        "t": list(range(4)) * 2,
        "x": [1.0, None, 3.0, 4.0, 2.0, 2.0, None, 1.0],
    }

    def clean(vals):
        return [
            None if v is None or (isinstance(v, float) and np.isnan(v)) else round(v, 4)
            for v in vals
        ]

    res = {}
    for name, nat in [("pandas", pd.DataFrame(data)), ("polars", pl.DataFrame(data))]:
        d = nw.from_native(nat)
        plain = d.with_columns(e=nw.col("x").ewm_mean(alpha=0.5)).get_column("e")
        over = (
            d.with_columns(e=nw.col("x").ewm_mean(alpha=0.5).over("g", order_by="t"))
            .sort("g", "t")
            .get_column("e")
        )
        res[name] = {"plain": clean(plain.to_list()), "over": clean(over.to_list())}
    res["x"] = data["x"]
    res["pandas_over_equals_polars_over"] = res["pandas"]["over"] == res["polars"]["over"]
    res["pandas_plain_equals_polars_plain"] = (
        res["pandas"]["plain"] == res["polars"]["plain"]
    )
    log(json.dumps(res, indent=1))
    OUT["H_ewm_nulls"] = res


# --------------------------------------------------------------------------- E
def section_e(grid):
    log("\n=== E: grouped iteration order + cost ===")
    import polars as pl
    import pyarrow as pa

    # E1 order determinism: polars group_by iteration without maintain_order
    X = make_panel(n_series=200, n_time=5, n_levels=1)
    flat = X.reset_index()
    first_appearance = list(dict.fromkeys(flat["h0"]))
    pdf_nw = nw.from_native(flat)
    pl_nw = nw.from_native(pl.from_pandas(flat))
    pa_nw = nw.from_native(pa.Table.from_pandas(flat, preserve_index=False))
    orders = {"pandas": set(), "polars": set(), "pyarrow": set()}
    match_first = {"pandas": 0, "polars": 0, "pyarrow": 0}
    n_rep = 20
    for _ in range(n_rep):
        for nm, fr in [("pandas", pdf_nw), ("polars", pl_nw), ("pyarrow", pa_nw)]:
            keys = [k[0] if isinstance(k, tuple) else k for k, _ in fr.group_by("h0")]
            orders[nm].add(tuple(keys))
            match_first[nm] += keys == first_appearance
    order_res = {
        nm: {
            "distinct_orders_in_%d_runs" % n_rep: len(orders[nm]),
            "runs_matching_first_appearance": match_first[nm],
        }
        for nm in orders
    }
    pk = next(iter(pdf_nw.group_by("h0")))[0]
    order_res["key_type_pandas"] = repr(type(pk)) + " " + repr(pk)
    pk = next(iter(pl_nw.group_by("h0")))[0]
    order_res["key_type_polars"] = repr(type(pk)) + " " + repr(pk)
    log("order:", json.dumps(order_res, indent=1))

    # E2 per-group cost
    cost = []
    for n_series, n_time in grid:
        if n_series * n_time > 2_000_000:
            continue
        X = make_panel(n_series=n_series, n_time=n_time, n_levels=1)
        flat = X.reset_index()
        pl_df = pl.from_pandas(flat)
        pa_t = pa.Table.from_pandas(flat, preserve_index=False)
        f_pd_nw = nw.from_native(flat)
        f_pl_nw = nw.from_native(pl_df)
        f_pa_nw = nw.from_native(pa_t)

        def it(gen):
            for _k, _g in gen:
                pass

        cands = {
            "pandas_native_groupby_level(sort=False) [current VectorizedDF]": lambda: it(
                X.groupby(level=0, sort=False)
            ),
            "pandas_native_groupby_col(sort=False)": lambda: it(
                flat.groupby("h0", sort=False)
            ),
            "nw[pandas].group_by(h0) iter": lambda: it(f_pd_nw.group_by("h0")),
            "polars_native_group_by(maintain_order=True) iter": lambda: it(
                pl_df.group_by("h0", maintain_order=True)
            ),
            "polars_native_partition_by(as_dict)": lambda: pl_df.partition_by(
                "h0", as_dict=True, maintain_order=True
            ),
            "nw[polars].group_by(h0) iter": lambda: it(f_pl_nw.group_by("h0")),
            "nw[pyarrow].group_by(h0) iter": lambda: it(f_pa_nw.group_by("h0")),
        }
        row = {"n_series": n_series, "n_time": n_time}
        for cname, fn in cands.items():
            if "pyarrow" in cname and n_series > 1000:
                row[cname] = None  # O(groups*rows); skipped at this size
                continue
            t0 = time.perf_counter()
            fn()
            single = time.perf_counter() - t0
            rep = 3 if single < 2 else 1
            row[cname] = bench(fn, repeat=rep, warmup=0 if single > 2 else 1)
        base = row["pandas_native_groupby_level(sort=False) [current VectorizedDF]"]
        row["ratios_vs_current"] = {
            k: (v / base if v is not None else None)
            for k, v in row.items()
            if k not in ("n_series", "n_time", "ratios_vs_current")
        }
        row["rss_mb"] = rss_mb()
        cost.append(row)
        log("iter cost", json.dumps(row, indent=1))
    OUT["E_group_iter"] = {"order": order_res, "cost": cost}


# --------------------------------------------------------------------------- F
def section_f():
    log("\n=== F: per-call narwhals overhead on tiny inputs ===")
    import polars as pl

    s = pd.Series(np.random.default_rng(0).normal(size=100), name="c0")
    df = s.to_frame()
    pls = pl.Series("c0", s.to_numpy())
    pldf = pl.DataFrame({"c0": s.to_numpy()})
    N = 2000
    cases = {
        "pd: s.iloc[-1] (naive last)": lambda: s.iloc[-1],
        "nw[pd]: from_native(s).item(-1)": lambda: nw.from_native(s, series_only=True).item(-1),
        "pd: s.mean()": lambda: s.mean(),
        "nw[pd]: from_native(s).mean()": lambda: nw.from_native(s, series_only=True).mean(),
        "pd: df.shift(1)": lambda: df.shift(1),
        "nw[pd]: from_native(df).with_columns(shift).to_native()": lambda: nw.from_native(df)
        .with_columns(nw.col("c0").shift(1))
        .to_native(),
        "pl: s.mean()": lambda: pls.mean(),
        "nw[pl]: from_native(s).mean()": lambda: nw.from_native(pls, series_only=True).mean(),
        "pl: df.with_columns(shift)": lambda: pldf.with_columns(pl.col("c0").shift(1)),
        "nw[pl]: from_native(df).with_columns(shift).to_native()": lambda: nw.from_native(pldf)
        .with_columns(nw.col("c0").shift(1))
        .to_native(),
        "nw: from_native(pd df) only": lambda: nw.from_native(df),
        "nw: from_native(pl df) only": lambda: nw.from_native(pldf),
    }
    res = {k: bench(f, repeat=5, number=N) * 1e6 for k, f in cases.items()}
    pairs = [
        ("pd: s.iloc[-1] (naive last)", "nw[pd]: from_native(s).item(-1)"),
        ("pd: s.mean()", "nw[pd]: from_native(s).mean()"),
        ("pd: df.shift(1)", "nw[pd]: from_native(df).with_columns(shift).to_native()"),
        ("pl: s.mean()", "nw[pl]: from_native(s).mean()"),
        ("pl: df.with_columns(shift)", "nw[pl]: from_native(df).with_columns(shift).to_native()"),
    ]
    ratios = {f"{b} / {a}": res[b] / res[a] for a, b in pairs}
    log(json.dumps({"us_per_call": res, "ratios": ratios}, indent=1))
    OUT["F_overhead_us"] = {"us_per_call": res, "ratios": ratios}


# --------------------------------------------------------------------------- G
def section_g():
    log("\n=== G: import footprint ===")
    py = sys.executable

    def t(cmd):
        best = 1e9
        for _ in range(5):
            t0 = time.perf_counter()
            subprocess.run([py, "-c", cmd], check=True)
            best = min(best, time.perf_counter() - t0)
        return best

    base = t("pass")
    nwt = t("import narwhals")
    mods = subprocess.run(
        [
            py,
            "-c",
            "import sys, narwhals; print(','.join(m for m in "
            "['pandas','polars','pyarrow','numpy','dask','duckdb'] if m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    r = {
        "python_startup_s": base,
        "import_narwhals_s": nwt,
        "narwhals_import_overhead_s": nwt - base,
        "heavy_modules_imported_by_import_narwhals": mods or "(none)",
    }
    log(json.dumps(r, indent=1))
    OUT["G_import"] = r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--sections", default="ABCDEFGH")
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    OUT["env"] = env_info()
    OUT["env"]["python"] = sys.version.split()[0]
    OUT["grid"] = args.grid
    log("env:", OUT["env"])
    fns = {
        "A": section_a,
        "B": section_b,
        "C": lambda: section_c(grid),
        "D": section_d,
        "E": lambda: section_e(grid),
        "F": section_f,
        "G": section_g,
        "H": section_h,
    }
    for s in args.sections:
        fns[s]()
    pdv = pd.__version__.split(".")[0]
    os.makedirs(RESULTS, exist_ok=True)
    suffix = "" if args.sections == ap.get_default("sections") else f"_{args.sections}"
    path = os.path.join(RESULTS, f"r2_narwhals_probe_pd{pdv}_{args.grid}{suffix}.json")
    with open(path, "w") as f:
        json.dump(OUT, f, indent=1, default=str)
    log("\nwrote", path, "peak-ish rss_mb", rss_mb())


if __name__ == "__main__":
    main()
