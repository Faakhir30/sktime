"""r2_narwhals_verify0: adversarial re-checks of r2_narwhals claims by DIFFERENT code paths.

Each section targets one r2_narwhals claim and tries to refute it with a method that is
independent of scripts/r2_narwhals_probe.py:

  V1  (claim .1/.2/.3, C1)  collect defaults via isinstance on native objects; count calls to
      pl.from_pandas to prove dask->polars goes through pandas; kwargs passthrough.
  V2  (claim .6, C2)        paths where index level names / values ARE lost through narwhals:
      index-name/column collision in group_by iteration, to_polars/to_arrow/lazy of a
      pandas MultiIndex frame, reset_index-path sub-frames; assert_frame_equal round trip.
  V3  (claim .7)            reset_index copy: in-place mutation test (pd2), block-level
      shares_memory (pd3), tracemalloc bytes allocated vs value-column bytes.
  V4  (claim .9)            polars order nondeterminism vs number of groups, directly in
      polars (no narwhals) and through narwhals; cost of a first-appearance reorder fix.
  V5  (claim .10)           pyarrow group iteration scaling: log-log slope over groups at
      fixed rows/group, vs a linear sort+slice pyarrow alternative.
  V6  (claim .11)           iteration cost with an END-TO-END delivery accounting (each
      method must hand the inner estimator what it needs), including the REAL current
      VectorizedDF.items(); ABAB interleaved rounds, median of per-round ratios.
  V7  (claim .12)           per-call overhead with a like-for-like native baseline
      (df.assign(c0=shift) instead of df.shift), and length scaling 100 vs 10_000 rows.
  V8  (claim .13/.14)       grouped window exprs vs a pure-numpy per-group reference
      (not pandas); ewm null semantics vs a hand-written ewm.

Run (from worktree root):
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/r2_narwhals_verify0_checks.py --grid dev
"""

import argparse
import json
import os
import statistics
import sys
import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd

from common import bench, env_info, make_panel, rss_mb

warnings.filterwarnings("ignore")

import narwhals as nw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
OUT = {}


def log(*a):
    print(*a, flush=True)


def attempt(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e).strip().splitlines()[0][:160] if str(e).strip() else ''}"


# --------------------------------------------------------------------------- V1
def v1():
    log("\n=== V1: collect defaults (isinstance) / dask->polars via pandas ===")
    import polars as pl
    import pyarrow as pa

    X = make_panel(n_series=3, n_time=6, n_levels=1)
    flat = X.reset_index()
    natives = {"polars_lazy": pl.from_pandas(flat).lazy()}
    import dask.dataframe as dd

    natives["dask"] = dd.from_pandas(flat, npartitions=2)
    import duckdb

    natives["duckdb"] = duckdb.from_df(flat)
    try:
        import ibis

        natives["ibis"] = ibis.memtable(flat)
    except ImportError:
        pass

    def kind(obj):
        if isinstance(obj, pd.DataFrame):
            return "pandas.DataFrame"
        if isinstance(obj, pl.DataFrame):
            return "polars.DataFrame"
        if isinstance(obj, pa.Table):
            return "pyarrow.Table"
        return type(obj).__module__ + "." + type(obj).__name__

    res = {}
    for name, nat in natives.items():
        lf = nw.from_native(nat)
        r = {"is_LazyFrame": isinstance(lf, nw.LazyFrame)}
        r["collect()"] = kind(nw.to_native(lf.collect()))
        for b in ["pandas", "polars", "pyarrow", "dask", "duckdb", "ibis"]:
            r[f"collect({b})"] = attempt(lambda b=b, lf=lf: kind(nw.to_native(lf.collect(b))))
        r["same_lazy_type_after_filter"] = type(
            nw.to_native(lf.filter(nw.col("c0") > 0))
        ) is type(nat)
        res[name] = r

    # dask -> polars: count pl.from_pandas calls during collect('polars')
    calls = {"n": 0}
    orig = pl.from_pandas

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    pl.from_pandas = spy
    try:
        out = nw.from_native(natives["dask"]).collect("polars")
    finally:
        pl.from_pandas = orig
    res["dask.collect('polars') pl.from_pandas calls"] = calls["n"]
    res["dask.collect('polars') result"] = kind(nw.to_native(out))

    # kwargs passthrough for polars lazy (engine) -> should work
    res["polars_lazy.collect(engine='streaming')"] = attempt(
        lambda: kind(nw.to_native(nw.from_native(natives["polars_lazy"]).collect(engine="streaming")))
    )
    # re-lazify after collect: does narwhals remember duckdb?
    rt = nw.from_native(natives["duckdb"]).collect().lazy()
    res["duckdb.collect().lazy() native"] = kind(nw.to_native(rt))
    res["duckdb.collect().lazy('duckdb') native"] = kind(
        nw.to_native(nw.from_native(natives["duckdb"]).collect().lazy("duckdb"))
    )
    log(json.dumps(res, indent=1))
    OUT["V1"] = res


# --------------------------------------------------------------------------- V2
def v2():
    log("\n=== V2: where are index level names/values lost? ===")
    res = {}
    for time_kind in ["datetime", "period", "int"]:
        X = make_panel(n_series=4, n_time=5, n_levels=1, time_kind=time_kind)
        r = {}
        d = nw.from_native(X)

        # (a) collision: a COLUMN named like an index level, then group_by iteration
        Xc = X.assign(h0=X.index.get_level_values("h0"))
        k, sub = next(iter(nw.from_native(Xc).group_by("h0")))
        sn = nw.to_native(sub)
        r["collision: subframe index type"] = type(sn.index).__name__
        r["collision: subframe index names"] = [str(n) for n in sn.index.names]
        # (a') no collision, group by index level name directly (pandas passthrough)
        k2, sub2 = next(iter(d.group_by("h0")))
        r["level-name group_by iter: subframe index names"] = [
            str(n) for n in nw.to_native(sub2).index.names
        ]

        # (b) conversion of a pandas MultiIndex frame through narwhals to other backends
        def idx_report(fn):
            def g():
                o = fn()
                cols = o.column_names if hasattr(o, "column_names") else list(o.columns)
                return {"columns": [str(c) for c in cols]}

            return attempt(g)

        r["lazy('dask').collect('pandas') columns"] = idx_report(
            lambda: nw.to_native(d.lazy("dask").collect("pandas"))
        )
        r["write_csv header"] = attempt(lambda: d.write_csv().splitlines()[0])

        r["to_polars() columns"] = idx_report(lambda: d.to_polars())
        r["to_arrow() columns"] = idx_report(lambda: d.to_arrow())
        r["lazy('polars').collect() columns"] = idx_report(
            lambda: nw.to_native(d.lazy("polars").collect())
        )
        r["lazy('duckdb').collect('pandas') columns"] = idx_report(
            lambda: nw.to_native(d.lazy("duckdb").collect("pandas"))
        )
        r["to_polars().to_pandas() index"] = attempt(
            lambda: type(d.to_polars().to_pandas().index).__name__
        )

        # (c) reset_index path: what does a sub-frame look like for the inner estimator?
        flat = X.reset_index()
        k3, sub3 = next(iter(nw.from_native(flat).group_by("h0")))
        s3 = nw.to_native(sub3)
        r["reset path: subframe index type"] = type(s3.index).__name__
        r["reset path: subframe columns"] = [str(c) for c in s3.columns]

        # (d) whole-frame round trip via a DIFFERENT equality: assert_frame_equal
        back = nw.to_native(nw.from_native(flat).with_columns(nw.col("c0"))).set_index(
            ["h0", "time"]
        )
        try:
            pd.testing.assert_frame_equal(back, X, check_index_type=True, check_exact=True)
            r["round trip assert_frame_equal"] = True
        except AssertionError as e:
            r["round trip assert_frame_equal"] = "FAIL: " + str(e).splitlines()[0][:150]
        # (e) round trip via polars backend (natural non-pandas path)
        def via_polars():
            pl_back = (
                nw.from_native(flat).lazy("polars").collect("pandas").to_native()
            ).set_index(["h0", "time"])
            pd.testing.assert_frame_equal(pl_back, X, check_index_type=True)
            return True

        r["round trip via polars backend assert_frame_equal"] = attempt(via_polars)
        res[time_kind] = r
    log(json.dumps(res, indent=1))
    OUT["V2"] = res


# --------------------------------------------------------------------------- V3
def v3(grid):
    log("\n=== V3: is reset_index a copy? (mutation / blocks / tracemalloc) ===")
    res = {"pandas": pd.__version__}
    X = make_panel(n_series=50, n_time=40, n_levels=1)
    flat = X.reset_index()
    # method 1 (pd2 only meaningful): mutate the flat value column in place
    if int(pd.__version__.split(".")[0]) < 3:
        before = float(X["c0"].iloc[0])
        flat_vals = flat["c0"].to_numpy()
        flat_vals[0] = before + 12345.0  # writes into flat's block (no CoW in pd2)
        res["mutation propagated to X"] = float(X["c0"].iloc[0]) != before
    # method 2: block-level shares_memory
    xb = [b.values for b in X._mgr.blocks]
    fb = [getattr(b.values, "_ndarray", b.values) for b in flat._mgr.blocks]
    shares = any(
        np.shares_memory(a, b)
        for a in xb
        for b in fb
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
    )
    res["any block shares memory"] = bool(shares)

    # method 3: tracemalloc allocation during reset_index, relative to value bytes
    rows = []
    for n_series, n_time in [(1000, 100), (2000, 100)]:
        Xb = make_panel(n_series=n_series, n_time=n_time, n_levels=1)
        val_bytes = Xb["c0"].to_numpy().nbytes
        tracemalloc.start()
        f2 = Xb.reset_index()
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(
            {
                "rows": n_series * n_time,
                "value_col_MB": val_bytes / 2**20,
                "traced_alloc_MB_retained": cur / 2**20,
                "retained_over_value_bytes": cur / val_bytes,
                "h0_dtype_after": str(f2["h0"].dtype),
            }
        )
        del f2, Xb
    res["tracemalloc"] = rows
    log(json.dumps(res, indent=1))
    OUT["V3"] = res


# --------------------------------------------------------------------------- V4
def v4():
    log("\n=== V4: polars group order (direct polars vs narwhals), fix cost ===")
    import polars as pl

    res = {}
    for n_groups in [5, 20, 200]:
        X = make_panel(n_series=n_groups, n_time=5, n_levels=1)
        flat = X.reset_index()
        first = list(dict.fromkeys(flat["h0"]))
        pldf = pl.from_pandas(flat)
        nwdf = nw.from_native(pldf)
        d_pl, d_nw, m_pl, m_nw = set(), set(), 0, 0
        for _ in range(30):
            k1 = [k[0] for k, _ in pldf.group_by("h0")]
            k2 = [k[0] for k, _ in nwdf.group_by("h0")]
            d_pl.add(tuple(k1))
            d_nw.add(tuple(k2))
            m_pl += k1 == first
            m_nw += k2 == first
        res[f"{n_groups} groups"] = {
            "polars_direct distinct/30": len(d_pl),
            "polars_direct first-appearance/30": m_pl,
            "narwhals distinct/30": len(d_nw),
            "narwhals first-appearance/30": m_nw,
        }

    # cost of making narwhals[polars] iteration deterministic (first appearance)
    X = make_panel(n_series=1000, n_time=100, n_levels=1)
    flat = X.reset_index()
    pldf = pl.from_pandas(flat)
    nwdf = nw.from_native(pldf)

    def nw_plain():
        for _k, _g in nwdf.group_by("h0"):
            pass

    def nw_reordered():
        order = nwdf["h0"].unique(maintain_order=True).to_list()
        groups = {k[0]: g for k, g in nwdf.group_by("h0")}
        for k in order:
            _ = groups[k]

    def pl_partition():
        for _k, _g in pldf.partition_by("h0", as_dict=True, maintain_order=True).items():
            pass

    t = {}
    for _ in range(2):  # interleave
        for nm, f in [("nw_plain", nw_plain), ("nw_reordered", nw_reordered), ("pl_partition", pl_partition)]:
            t[nm] = min(t.get(nm, 1e9), bench(f, repeat=3))
    res["fix_cost_1000x100_ms"] = {k: v * 1e3 for k, v in t.items()}
    res["fix_cost_ratio reordered/plain"] = t["nw_reordered"] / t["nw_plain"]
    log(json.dumps(res, indent=1))
    OUT["V4"] = res


# --------------------------------------------------------------------------- V5
def v5():
    log("\n=== V5: pyarrow group iteration scaling ===")
    import pyarrow as pa
    import pyarrow.compute as pc

    rows = []
    for G in [50, 100, 200, 400]:
        X = make_panel(n_series=G, n_time=50, n_levels=1)
        flat = X.reset_index()
        tbl = pa.Table.from_pandas(flat, preserve_index=False)
        f = nw.from_native(tbl)

        def nw_iter(f=f):
            for _k, _g in f.group_by("h0"):
                pass

        def pa_sort_slice(tbl=tbl):
            # linear alternative: stable sort by first-appearance code, then slice
            codes = pc.dictionary_encode(tbl["h0"]).combine_chunks().indices
            order = pc.sort_indices(codes)
            st = tbl.take(order)
            counts = np.bincount(codes.to_numpy())
            off = 0
            for c in counts:
                _ = st.slice(off, c)
                off += c

        def pd_current(X=X):
            for _k, _g in X.groupby(level=0, sort=False):
                pass

        t_nw = bench(nw_iter, repeat=3)
        t_alt = bench(pa_sort_slice, repeat=3)
        t_cur = bench(pd_current, repeat=3)
        rows.append(
            {
                "groups": G,
                "rows": G * 50,
                "nw_pyarrow_ms": t_nw * 1e3,
                "pa_sort_slice_ms": t_alt * 1e3,
                "pd_current_ms": t_cur * 1e3,
                "nw_pyarrow/current": t_nw / t_cur,
                "pa_sort_slice/current": t_alt / t_cur,
            }
        )
    g = np.log([r["groups"] for r in rows])
    slope_nw = float(np.polyfit(g, np.log([r["nw_pyarrow_ms"] for r in rows]), 1)[0])
    slope_alt = float(np.polyfit(g, np.log([r["pa_sort_slice_ms"] for r in rows]), 1)[0])
    slope_cur = float(np.polyfit(g, np.log([r["pd_current_ms"] for r in rows]), 1)[0])
    res = {
        "rows": rows,
        "loglog_slope_nw_pyarrow": slope_nw,
        "loglog_slope_pa_sort_slice": slope_alt,
        "loglog_slope_pd_current": slope_cur,
    }
    log(json.dumps(res, indent=1))
    OUT["V5"] = res


# --------------------------------------------------------------------------- V6
def v6(grid):
    log("\n=== V6: end-to-end delivery iteration cost, ABAB interleaved ===")
    import polars as pl

    from sktime.datatypes._vectorize import _enforce_index_freq, prepare_VectorizedDF

    ef = _enforce_index_freq
    out = []
    for n_series, n_time in grid:
        if n_series * n_time > 200_000:
            continue
        X = make_panel(n_series=n_series, n_time=n_time, n_levels=1)
        vec, X_mi = prepare_VectorizedDF(X, iterate_as="Series", is_scitype="Panel")
        flat = X.reset_index()
        nw_pd_mi = nw.from_native(X)
        nw_pd_flat = nw.from_native(flat)
        pldf = pl.from_pandas(flat)
        nw_pl = nw.from_native(pldf)

        cands = {
            # what the framework does today, incl. droplevel + freq enforcement
            "current VectorizedDF.items()": lambda: [None for _ in vec.items(X_mi)],
            # bare iteration (the r2 baseline)
            "bare groupby(level=0,sort=False)": lambda: [
                None for _ in X.groupby(level=0, sort=False)
            ],
            "groupby + droplevel": lambda: [
                g.droplevel(0) for _, g in X.groupby(level=0, sort=False)
            ],
            "groupby + droplevel + enforce_freq": lambda: [
                ef(g.droplevel(0)) for _, g in X.groupby(level=0, sort=False)
            ],
            # narwhals pandas on MultiIndex (undocumented level passthrough) + droplevel
            "nw[pd MI] iter + droplevel": lambda: [
                nw.to_native(g).droplevel(0) for _, g in nw_pd_mi.group_by("h0")
            ],
            "nw[pd MI] iter + droplevel + enforce_freq": lambda: [
                ef(nw.to_native(g).droplevel(0)) for _, g in nw_pd_mi.group_by("h0")
            ],
            # narwhals pandas on reset frame: inner pandas estimator needs a time index
            "nw[pd flat] iter + set_index(time)": lambda: [
                nw.to_native(g).drop(columns="h0").set_index("time")
                for _, g in nw_pd_flat.group_by("h0")
            ],
            "nw[pd flat] iter + set_index(time) + enforce_freq": lambda: [
                ef(nw.to_native(g).drop(columns="h0").set_index("time"))
                for _, g in nw_pd_flat.group_by("h0")
            ],
            # narwhals polars, inner polars estimator takes slice as is (order not fixed)
            "nw[pl] iter + to_native": lambda: [
                nw.to_native(g) for _, g in nw_pl.group_by("h0")
            ],
            "pl partition_by(as_dict,maintain_order)": lambda: list(
                pldf.partition_by("h0", as_dict=True, maintain_order=True).values()
            ),
        }
        names = list(cands)
        ratios = {n: [] for n in names}
        abs_t = {n: [] for n in names}
        for f in cands.values():  # warmup
            f()
        n_rounds = 5 if n_series <= 1000 else 3
        for _ in range(n_rounds):
            ts = {}
            for n in names:
                t0 = time.perf_counter()
                cands[n]()
                ts[n] = time.perf_counter() - t0
            base = ts["current VectorizedDF.items()"]
            for n in names:
                ratios[n].append(ts[n] / base)
                abs_t[n].append(ts[n])
        row = {
            "n_series": n_series,
            "n_time": n_time,
            "median_ratio_vs_current_items": {n: statistics.median(r) for n, r in ratios.items()},
            "min_ms": {n: min(v) * 1e3 for n, v in abs_t.items()},
            "bare_over_items (r2 baseline vs real current)": statistics.median(
                ratios["bare groupby(level=0,sort=False)"]
            ),
            "rss_mb": rss_mb(),
        }
        out.append(row)
        log(json.dumps(row, indent=1))
    OUT["V6"] = out


# --------------------------------------------------------------------------- V7
def v7():
    log("\n=== V7: per-call overhead, like-for-like natives, length scaling ===")
    import polars as pl

    res = {}
    for n in [100, 10_000]:
        s = pd.Series(np.random.default_rng(0).normal(size=n), name="c0")
        df = s.to_frame()
        pldf = pl.DataFrame({"c0": s.to_numpy()})
        N = 1000 if n == 100 else 200
        cases = {
            "pd df.shift(1) [r2 baseline]": lambda: df.shift(1),
            "pd df.assign(c0=shift) [like-for-like]": lambda: df.assign(c0=df["c0"].shift(1)),
            "nw[pd] with_columns(shift).to_native": lambda: nw.from_native(df)
            .with_columns(nw.col("c0").shift(1))
            .to_native(),
            "nw[pd] from_native only": lambda: nw.from_native(df),
            "pl with_columns(shift)": lambda: pldf.with_columns(pl.col("c0").shift(1)),
            "nw[pl] with_columns(shift).to_native": lambda: nw.from_native(pldf)
            .with_columns(nw.col("c0").shift(1))
            .to_native(),
            "pd s.mean()": lambda: s.mean(),
            "nw[pd] s.mean()": lambda: nw.from_native(s, series_only=True).mean(),
        }
        us = {}
        for _ in range(2):  # interleave twice, keep min
            for k, f in cases.items():
                us[k] = min(us.get(k, 1e12), bench(f, repeat=3, number=N) * 1e6)
        res[f"n={n}"] = {
            "us_per_call": us,
            "nw/pd shift [r2 pairing]": us["nw[pd] with_columns(shift).to_native"]
            / us["pd df.shift(1) [r2 baseline]"],
            "nw/pd assign [like-for-like]": us["nw[pd] with_columns(shift).to_native"]
            / us["pd df.assign(c0=shift) [like-for-like]"],
            "nw/pl with_columns": us["nw[pl] with_columns(shift).to_native"]
            / us["pl with_columns(shift)"],
            "nw/pd mean": us["nw[pd] s.mean()"] / us["pd s.mean()"],
            "nw[pd] with_columns added us": us["nw[pd] with_columns(shift).to_native"]
            - us["pd df.assign(c0=shift) [like-for-like]"],
        }
    log(json.dumps(res, indent=1))
    OUT["V7"] = res


# --------------------------------------------------------------------------- V8
def _np_ref(t_sorted_vals, op):
    """numpy per-group reference on one group's values sorted by time."""
    v = np.asarray(t_sorted_vals, dtype=float)
    n = len(v)
    if op == "mean":
        return np.full(n, np.nanmean(v))
    if op == "cum_sum":  # polars semantics: null stays null, sum skips nulls
        out = np.nancumsum(v)
        out[np.isnan(v)] = np.nan
        return out
    if op == "shift":
        return np.concatenate([[np.nan], v[:-1]])
    if op == "diff":
        return np.concatenate([[np.nan], v[1:] - v[:-1]])
    if op in ("rolling_mean", "rolling_std"):
        out = np.full(n, np.nan)
        for i in range(2, n):
            w = v[i - 2 : i + 1]
            if not np.isnan(w).any():
                out[i] = w.mean() if op == "rolling_mean" else w.std(ddof=1)
        return out
    if op == "ffill":
        out = v.copy()
        for i in range(1, n):
            if np.isnan(out[i]):
                out[i] = out[i - 1]
        return out
    raise ValueError(op)


def v8():
    log("\n=== V8: grouped window exprs vs numpy reference; ewm by hand ===")
    import duckdb
    import polars as pl
    import pyarrow as pa

    X = make_panel(n_series=3, n_time=8, n_levels=1)
    flat = X.reset_index()
    flat.loc[3, "c0"] = np.nan
    flat.loc[13, "c0"] = np.nan
    flat = flat.sample(frac=1.0, random_state=1).reset_index(drop=True)
    srt = flat.sort_values(["h0", "time"]).reset_index(drop=True)
    ref = {}
    for op in ["mean", "cum_sum", "shift", "diff", "rolling_mean", "rolling_std", "ffill"]:
        ref[op] = np.concatenate(
            [_np_ref(g["c0"].to_numpy(), op) for _, g in srt.groupby("h0", sort=True)]
        )
    exprs = {
        "mean": nw.col("c0").mean().over("h0"),
        "cum_sum": nw.col("c0").cum_sum().over("h0", order_by="time"),
        "shift": nw.col("c0").shift(1).over("h0", order_by="time"),
        "diff": nw.col("c0").diff().over("h0", order_by="time"),
        "rolling_mean": nw.col("c0").rolling_mean(3).over("h0", order_by="time"),
        "rolling_std": nw.col("c0").rolling_std(3).over("h0", order_by="time"),
        "ffill": nw.col("c0").fill_null(strategy="forward").over("h0", order_by="time"),
    }
    natives = {
        "pandas": flat,
        "polars": pl.from_pandas(flat),
        "polars_lazy": pl.from_pandas(flat).lazy(),
        "pyarrow": pa.Table.from_pandas(flat, preserve_index=False),
        "duckdb": duckdb.from_df(flat),
    }
    import dask.dataframe as dd

    natives["dask"] = dd.from_pandas(flat, npartitions=2)
    try:
        import ibis

        natives["ibis"] = ibis.memtable(flat)
    except ImportError:
        pass
    table = {}
    for b, nat in natives.items():
        row = {}
        for op, ex in exprs.items():

            def run(nat=nat, ex=ex, op=op):
                o = nw.from_native(nat).with_columns(x=ex)
                if isinstance(o, nw.LazyFrame):
                    o = o.collect("polars")
                o = o.to_polars().sort("h0", "time")
                got = o["x"].cast(pl.Float64).fill_nan(None).to_numpy()
                got = np.where(np.isnan(got), np.nan, got)
                exp = ref[op]
                return "ok" if np.allclose(got, exp, equal_nan=True, atol=1e-9) else "MISMATCH"

            row[op] = attempt(run)
            if row[op] not in ("ok", "MISMATCH"):
                row[op] = row[op].split(":")[0]
        row["n_ok"] = sum(v == "ok" for v in row.values())
        table[b] = row
    log(json.dumps(table, indent=1))

    # ewm by hand: adjust=True, ignore_nulls=False (polars default semantics)
    def ewm_hand(v, alpha):
        out, num, den, w = [], 0.0, 0.0, 1.0
        for x in v:
            num *= 1 - alpha
            den *= 1 - alpha
            if x is None:
                out.append(None)
                continue
            num += x
            den += 1.0
            out.append(round(num / den, 4))
        return out

    data = {"g": ["a"] * 4 + ["b"] * 4, "t": list(range(4)) * 2,
            "x": [1.0, None, 3.0, 4.0, 2.0, 2.0, None, 1.0]}
    hand = ewm_hand(data["x"][:4], 0.5) + ewm_hand(data["x"][4:], 0.5)
    ew = {"hand_ignore_nulls_False_null_at_null": hand}
    for nm, nat in [("pandas", pd.DataFrame(data)), ("polars", pl.DataFrame(data))]:
        o = nw.from_native(nat).with_columns(
            e=nw.col("x").ewm_mean(alpha=0.5).over("g", order_by="t")
        ).sort("g", "t")
        vals = o["e"].to_list()
        ew[nm] = [None if (v is None or v != v) else round(v, 4) for v in vals]
    ew["pandas==hand"] = ew["pandas"] == hand
    ew["polars==hand"] = ew["polars"] == hand
    log(json.dumps(ew, indent=1))
    OUT["V8"] = {"windows": table, "ewm": ew}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--sections", default="12345678")
    a = ap.parse_args()
    grid = (
        [(10, 100), (100, 100), (1000, 100), (2000, 50)]
        if a.grid == "dev"
        else [(10, 500), (100, 500), (1000, 500), (10000, 100)]
    )
    OUT["env"] = env_info()
    OUT["grid"] = a.grid
    log("env", OUT["env"])
    fns = {"1": v1, "2": v2, "3": lambda: v3(grid), "4": v4, "5": v5,
           "6": lambda: v6(grid), "7": v7, "8": v8}
    for s in a.sections:
        fns[s]()
    pdv = pd.__version__.split(".")[0]
    suffix = "" if a.sections == "12345678" else "_" + a.sections
    path = os.path.join(RESULTS, f"r2_narwhals_verify0_pd{pdv}_{a.grid}{suffix}.json")
    with open(path, "w") as f:
        json.dump(OUT, f, indent=1, default=str)
    log("wrote", path, "rss_mb", rss_mb())


if __name__ == "__main__":
    main()
