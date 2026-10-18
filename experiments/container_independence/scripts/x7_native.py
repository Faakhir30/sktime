"""x7: value of engine-native panel forecasting vs per-series Python loops.

Five simple forecasters, each computed for ALL series of a panel at once, in:

  a   pandas groupby vectorized ops (no per-group python), pandas MultiIndex in/out
  b   polars eager expressions (group_by().agg + cross join), polars in/out
  c   polars lazy, default in-memory engine (collect())
  c_s polars lazy, streaming engine (collect(engine="streaming"))
  d   duckdb SQL (aggregates + window functions), arrow in/out
  e   numpy, equal-length series reshaped to (n_series, T); pandas in/out
  e2  numpy GroupedArray (indptr + reduceat, works for ragged panels); pandas in/out
  ek  numpy kernel only (input already (n, T) array; output (n, H) array) = floor
  f   baseline: current sktime per-series VectorizedDF loop, pandas in/out
  f_pl baseline with sktime "polars_panel" input (converted to pandas and back)

Methods (fh = 1..H, H=14, daily datetime index):

  last     NaiveForecaster(strategy="last")
  snaive7  NaiveForecaster(strategy="last", sp=7)
  mean10   NaiveForecaster(strategy="mean", window_length=10)
  drift    NaiveForecaster(strategy="drift")
  ols      PolynomialTrendForecaster(degree=1)  (closed-form OLS on days-since-epoch)

Every engine's predictions (values AND future timestamps) are checked against (f).

Usage (from the worktree root)::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x7_native.py --grid dev --threads 1
    ... --threads 4
    ... --mem            # peak / retained RSS per engine, one subprocess each
    ... --parquet        # end-to-end from a wide parquet file (projection pushdown)
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings


def _set_threads_from_argv():
    # must happen before polars / numpy import
    n = "4"
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            n = sys.argv[i + 1]
    os.environ["POLARS_MAX_THREADS"] = n
    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ.setdefault(v, "1")
    return int(n)


N_THREADS = _set_threads_from_argv()
OLS_DEFAULT = "--ols-default" in sys.argv

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from common import bench, env_info, make_panel_fast, rss_mb, to_polars_long  # noqa

warnings.simplefilter("ignore")

H = 14
SP = 7
W = 10
HS = np.arange(1, H + 1)
STEP = np.timedelta64(1, "D")
METHODS = ["last", "snaive7", "mean10", "drift", "ols"]
ENGINES = ["a", "b", "c", "c_s", "d", "e", "e2", "ek", "f", "f_pl"]
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

DEV = [(100, 100), (1_000, 100), (10_000, 100)]
LARGE = [(1_000, 100), (10_000, 100), (100_000, 100), (10_000, 1_000)]


def silence_sktime_warnings():
    """sktime's _SuppressWarningPattern.__exit__ resets FutureWarning to "default",
    so the per-clone ``remember_data`` FutureWarning gets printed thousands of
    times (2.7MB of stderr for 5k fits). We re-ignore on exit. This makes the
    baseline FASTER than out-of-the-box sktime, i.e. speedups are conservative.
    """
    import sktime.utils.warnings as sw

    def _exit(self, *a):
        warnings.simplefilter("ignore")

    sw._SuppressWarningPattern.__exit__ = _exit
    warnings.simplefilter("ignore")


# =============================================================================
# (a) pandas groupby, vectorized
# =============================================================================


def _pd_out(keys, tlast, vals, name="c0"):
    ft = (np.asarray(tlast)[:, None] + HS * STEP).ravel()
    idx = pd.MultiIndex.from_arrays(
        [np.repeat(np.asarray(keys), H), ft], names=["h0", "time"]
    )
    return pd.DataFrame({name: vals}, index=idx)


def run_pandas(y, method, x0):
    s = y["c0"]
    g = s.groupby(level=0, sort=False)
    tail1 = g.tail(1)
    keys = tail1.index.get_level_values(0)
    tlast = tail1.index.get_level_values(-1).to_numpy()
    last = tail1.to_numpy()
    n = len(last)
    if method == "last":
        vals = np.repeat(last, H)
    elif method == "snaive7":
        season = g.tail(SP).to_numpy().reshape(n, SP)
        vals = season[:, (HS - 1) % SP].ravel()
    elif method == "mean10":
        m = g.tail(W).groupby(level=0, sort=False).mean().to_numpy()
        vals = np.repeat(m, H)
    elif method == "drift":
        first = g.first().to_numpy()
        cnt = g.size().to_numpy()
        slope = (last - first) / (cnt - 1)
        vals = (last[:, None] + slope[:, None] * HS).ravel()
    elif method == "ols":
        t = y.index.get_level_values(-1)
        x = (t - pd.Timestamp(0)) / pd.Timedelta(days=1) - x0
        x = np.asarray(x, dtype="float64")
        yv = s.to_numpy()
        d = pd.DataFrame(
            {"x": x, "y": yv, "xy": x * yv, "xx": x * x, "one": 1.0}, index=y.index
        )
        S = d.groupby(level=0, sort=False).sum()
        nn, sx, sy, sxy, sxx = (S[c].to_numpy() for c in ["one", "x", "y", "xy", "xx"])
        b = (nn * sxy - sx * sy) / (nn * sxx - sx * sx)
        a = (sy - b * sx) / nn
        xl = (tlast - np.datetime64(0, "ns")) / np.timedelta64(1, "D") - x0
        vals = (a[:, None] + b[:, None] * (xl[:, None] + HS)).ravel()
    return _pd_out(keys, tlast, vals)


# =============================================================================
# (b, c) polars eager / lazy: one plan, works on DataFrame and LazyFrame
# =============================================================================


def pl_plan(frame, method, t0):
    lazy = isinstance(frame, pl.LazyFrame)
    hor = pl.DataFrame({"h": HS.astype("int64")})
    if lazy:
        hor = hor.lazy()
    y = pl.col("c0")
    aggs = [pl.col("time").last().alias("tlast")]
    h = pl.col("h")
    if method == "last":
        aggs.append(y.last().alias("a"))
        val = pl.col("a")
    elif method == "snaive7":
        aggs.append(y.tail(SP).alias("season"))
        val = pl.col("season").list.get((h - 1) % SP)
    elif method == "mean10":
        aggs.append(y.tail(W).mean().alias("a"))
        val = pl.col("a")
    elif method == "drift":
        aggs.append(y.last().alias("a"))
        aggs.append(((y.last() - y.first()) / (pl.len() - 1)).alias("b"))
        val = pl.col("a") + pl.col("b") * h
    elif method == "ols":
        x = (pl.col("time") - t0).dt.total_nanoseconds() / 86_400e9
        b = pl.cov(x, y) / x.var()
        aggs.append(b.alias("b"))
        aggs.append((y.mean() - b * x.mean()).alias("a"))
        aggs.append(x.last().alias("xl"))
        val = pl.col("a") + pl.col("b") * (pl.col("xl") + h)
    p = frame.group_by("h0", maintain_order=True).agg(aggs)
    return p.join(hor, how="cross").select(
        "h0",
        (pl.col("tlast") + pl.duration(days=h)).alias("time"),
        val.alias("c0"),
    )


def run_polars_eager(ypl, method, t0):
    return pl_plan(ypl, method, t0)


def run_polars_lazy(ypl, method, t0, streaming=False):
    plan = pl_plan(ypl.lazy(), method, t0)
    if streaming:
        return plan.collect(engine="streaming")
    return plan.collect()


# =============================================================================
# (d) duckdb SQL
# =============================================================================

_SQL_OUT = (
    "SELECT h0, tlast + h * INTERVAL 1 DAY AS time, {val} AS c0 "
    "FROM p, range(1, {H1}) r(h) ORDER BY h0, time"
)


def duck_sql(method, x0):
    H1 = H + 1
    if method == "last":
        p = "SELECT h0, max(time) AS tlast, arg_max(c0, time) AS a FROM y GROUP BY h0"
        val = "a"
    elif method == "snaive7":
        p = (
            "SELECT h0, max(time) AS tlast, list(c0 ORDER BY time) AS season FROM "
            "(SELECT h0, time, c0, row_number() OVER "
            "(PARTITION BY h0 ORDER BY time DESC) AS rn FROM y QUALIFY rn <= 7) "
            "GROUP BY h0"
        )
        val = "season[((h - 1) % 7) + 1]"
    elif method == "mean10":
        p = (
            "SELECT h0, max(time) AS tlast, avg(c0) AS a FROM "
            "(SELECT h0, time, c0, row_number() OVER "
            "(PARTITION BY h0 ORDER BY time DESC) AS rn FROM y QUALIFY rn <= 10) "
            "GROUP BY h0"
        )
        val = "a"
    elif method == "drift":
        p = (
            "SELECT h0, max(time) AS tlast, arg_max(c0, time) AS a, "
            "(arg_max(c0, time) - arg_min(c0, time)) / (count(*) - 1) AS b "
            "FROM y GROUP BY h0"
        )
        val = "a + b * h"
    elif method == "ols":
        p = (
            "SELECT h0, max(time) AS tlast, regr_slope(c0, x) AS b, "
            "regr_intercept(c0, x) AS a, max(x) AS xl FROM "
            f"(SELECT h0, time, c0, epoch_ns(time) / 86400e9 - {x0!r} AS x FROM y) "
            "GROUP BY h0"
        )
        val = "a + b * (xl + h)"
    return f"WITH p AS ({p}) " + _SQL_OUT.format(val=val, H1=H1)


def run_duckdb(con, method, x0):
    return con.execute(duck_sql(method, x0)).fetch_arrow_table()


# =============================================================================
# (e) numpy (n, T) reshape, (e2) GroupedArray indptr, (ek) kernel only
# =============================================================================


def np_kernel(Y, method, x=None, xl=None):
    """Y: (n, T) float array; x: (T,) or (n, T) shifted day numbers. -> (n, H)."""
    if method == "last":
        return np.repeat(Y[:, -1:], H, axis=1)
    if method == "snaive7":
        return Y[:, -SP:][:, (HS - 1) % SP]
    if method == "mean10":
        return np.repeat(Y[:, -W:].mean(axis=1, keepdims=True), H, axis=1)
    if method == "drift":
        T = Y.shape[1]
        slope = (Y[:, -1] - Y[:, 0]) / (T - 1)
        return Y[:, -1:] + slope[:, None] * HS
    if method == "ols":
        xm = x.mean(axis=-1, keepdims=True)
        ym = Y.mean(axis=1, keepdims=True)
        xc = x - xm
        b = ((xc * (Y - ym)).sum(axis=1, keepdims=True)) / (xc * xc).sum(
            axis=-1, keepdims=True
        )
        a = ym - b * xm
        return a + b * (xl[:, None] + HS)
    raise ValueError(method)


def _days(t, x0):
    return (t - np.datetime64(0, "ns")) / np.timedelta64(1, "D") - x0


def run_numpy_reshape(y, method, x0):
    mi = y.index
    n = len(mi.levels[0])
    T = len(y) // n
    Y = y["c0"].to_numpy().reshape(n, T)
    tcodes = mi.codes[-1]
    tvals = mi.levels[-1].to_numpy()
    tlast = tvals[tcodes[T - 1 :: T]]
    keys = mi.levels[0].to_numpy()[mi.codes[0][::T]]
    x = xl = None
    if method == "ols":
        x = _days(tvals[tcodes].reshape(n, T), x0)
        xl = x[:, -1]
    P = np_kernel(Y, method, x, xl)
    return _pd_out(keys, tlast, P.ravel())


def run_numpy_indptr(y, method, x0):
    """GroupedArray style: contiguous rows per series, ragged allowed."""
    mi = y.index
    c0 = mi.codes[0]
    br = np.flatnonzero(c0[1:] != c0[:-1]) + 1
    start = np.r_[0, br]
    end = np.r_[br, len(c0)]
    v = y["c0"].to_numpy()
    tvals = mi.levels[-1].to_numpy()
    tcodes = mi.codes[-1]
    tlast = tvals[tcodes[end - 1]]
    keys = mi.levels[0].to_numpy()[c0[start]]
    last = v[end - 1]
    cnt = end - start
    if method == "last":
        P = np.repeat(last, H)
    elif method == "snaive7":
        P = v[(end - SP)[:, None] + (HS - 1) % SP].ravel()
    elif method == "mean10":
        cs = np.r_[0.0, np.cumsum(v)]
        P = np.repeat((cs[end] - cs[end - W]) / W, H)
    elif method == "drift":
        slope = (last - v[start]) / (cnt - 1)
        P = (last[:, None] + slope[:, None] * HS).ravel()
    elif method == "ols":
        x = _days(tvals[tcodes], x0)
        sx = np.add.reduceat(x, start)
        sy = np.add.reduceat(v, start)
        sxy = np.add.reduceat(x * v, start)
        sxx = np.add.reduceat(x * x, start)
        b = (cnt * sxy - sx * sy) / (cnt * sxx - sx * sx)
        a = (sy - b * sx) / cnt
        xl = x[end - 1]
        P = (a[:, None] + b[:, None] * (xl[:, None] + HS)).ravel()
    return _pd_out(keys, tlast, P)


# =============================================================================
# (f) sktime baseline
# =============================================================================


def make_sktime(method):
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.trend import PolynomialTrendForecaster

    return {
        "last": lambda: NaiveForecaster(strategy="last"),
        "snaive7": lambda: NaiveForecaster(strategy="last", sp=SP),
        "mean10": lambda: NaiveForecaster(strategy="mean", window_length=W),
        "drift": lambda: NaiveForecaster(strategy="drift"),
        "ols": lambda: PolynomialTrendForecaster(degree=1, regressor=_ols_regressor()),
    }[method]


def _ols_regressor():
    """LinearRegression(fit_intercept=False) as in PolynomialTrendForecaster's default.

    sklearn >= 1.9 passes ``tol`` (default 1e-6) as ``cond`` to scipy lstsq on dense
    data. With x = days since epoch (~1.1e4) the design [1, x] has singular value
    ratio ~2e-7 < 1e-6, so the default fit is TRUNCATED (intercept ~ 0) and
    differs from OLS by up to ~20 in our data (see report). We pass tol=1e-12 so the
    baseline computes true OLS; this does not change its runtime materially.
    ``--ols-default`` restores the sktime default regressor.
    """
    from sklearn.linear_model import LinearRegression

    if OLS_DEFAULT:
        return None
    try:
        return LinearRegression(fit_intercept=False, tol=1e-12)
    except TypeError:
        return LinearRegression(fit_intercept=False)


def run_sktime(y, method):
    f = make_sktime(method)()
    return f.fit(y, fh=list(HS)).predict()


# =============================================================================
# checking
# =============================================================================


def to_triplet(out):
    """Native output -> (keys:str ndarray, time:int64 ns, values:float), sorted."""
    import pyarrow as pa

    if isinstance(out, pd.DataFrame):
        if isinstance(out.index, pd.MultiIndex):
            k = out.index.get_level_values(0).astype(str).to_numpy()
            t = out.index.get_level_values(-1).to_numpy().astype("datetime64[ns]")
            v = out.iloc[:, 0].to_numpy(dtype="float64")
        else:
            raise TypeError("unexpected pandas output")
    elif isinstance(out, pl.DataFrame):
        cols = out.columns
        if any(c.startswith("__index__") for c in cols):
            ix = [c for c in cols if c.startswith("__index__")]
            kc, tc = ix[0], ix[-1]
            vc = [c for c in cols if c not in ix][0]
        else:
            kc, tc, vc = "h0", "time", "c0"
        k = out[kc].cast(pl.String).to_numpy()
        t = out[tc].cast(pl.Datetime("ns")).to_numpy()
        v = out[vc].to_numpy().astype("float64")
    elif isinstance(out, pa.Table):
        return to_triplet(pl.from_arrow(out))
    else:
        raise TypeError(type(out))
    o = np.lexsort((t.astype("int64"), k))
    return k[o], t[o].astype("int64"), v[o]


def compare(ref, out):
    k0, t0, v0 = ref
    k1, t1, v1 = to_triplet(out)
    if len(v0) != len(v1):
        return {"ok": False, "why": f"len {len(v0)} vs {len(v1)}"}
    ok_k = bool((k0 == k1).all())
    ok_t = bool((t0 == t1).all())
    maxabs = float(np.max(np.abs(v0 - v1)))
    maxrel = float(np.max(np.abs(v0 - v1) / np.maximum(np.abs(v0), 1e-12)))
    ok_v = bool(np.allclose(v0, v1, rtol=1e-7, atol=1e-9))
    return {
        "ok": ok_k and ok_t and ok_v,
        "keys": ok_k,
        "time": ok_t,
        "max_abs": maxabs,
        "max_rel": maxrel,
    }


# =============================================================================
# driver
# =============================================================================


def build_inputs(n, T):
    import duckdb

    y_pd = make_panel_fast(n, T)
    y_pl = to_polars_long(y_pd)
    tmin = y_pd.index.levels[-1].min()
    x0 = float((tmin - pd.Timestamp(0)) / pd.Timedelta(days=1))
    t0 = pl.lit(np.datetime64(tmin.to_datetime64(), "ns"))
    con = duckdb.connect()
    con.execute(f"SET threads={N_THREADS}")
    con.register("y", y_pl.to_arrow())
    y_skpl = None
    return dict(y_pd=y_pd, y_pl=y_pl, x0=x0, t0=t0, con=con, y_skpl=y_skpl, n=n, T=T)


def engine_fn(eng, method, D):
    y_pd, y_pl, x0, t0, con = D["y_pd"], D["y_pl"], D["x0"], D["t0"], D["con"]
    if eng == "a":
        return lambda: run_pandas(y_pd, method, x0)
    if eng == "b":
        return lambda: run_polars_eager(y_pl, method, t0)
    if eng == "c":
        return lambda: run_polars_lazy(y_pl, method, t0)
    if eng == "c_s":
        return lambda: run_polars_lazy(y_pl, method, t0, streaming=True)
    if eng == "d":
        return lambda: run_duckdb(con, method, x0)
    if eng == "e":
        return lambda: run_numpy_reshape(y_pd, method, x0)
    if eng == "e2":
        return lambda: run_numpy_indptr(y_pd, method, x0)
    if eng == "ek":
        n, T = D["n"], D["T"]
        Y = y_pd["c0"].to_numpy().reshape(n, T).copy()
        tv = y_pd.index.get_level_values(-1).to_numpy().reshape(n, T)
        x = _days(tv[0], x0)  # equal, aligned time grid -> one shared x row
        xl = np.full(n, x[-1])
        return lambda: np_kernel(Y, method, x, xl)
    if eng == "f":
        return lambda: run_sktime(y_pd, method)
    if eng == "f_pl":
        if D["y_skpl"] is None:
            from sktime.datatypes import convert_to

            D["y_skpl"] = convert_to(y_pd, "polars_panel")
        return lambda: run_sktime(D["y_skpl"], method)
    raise ValueError(eng)


def main_bench(args):
    silence_sktime_warnings()
    grid = DEV if args.grid == "dev" else LARGE
    if args.sizes:
        grid = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    methods = args.methods.split(",") if args.methods else METHODS
    engines = args.engines.split(",") if args.engines else ENGINES
    out = {
        "env": env_info(),
        "threads": N_THREADS,
        "H": H,
        "grid": grid,
        "methods": methods,
        "engines": engines,
        "results": [],
    }
    for n, T in grid:
        D = build_inputs(n, T)
        print(f"== n_series={n} T={T} rss={rss_mb():.0f}MB", flush=True)
        for method in methods:
            # reference: sktime baseline (computed once for checking), or (a) when
            # the baseline is skipped at this size (a is validated vs f elsewhere)
            ref_name = "f" if n <= args.baseline_max else "a"
            t_start = time.perf_counter()
            if ref_name == "f":
                ref_out = run_sktime(D["y_pd"], method)
            else:
                ref_out = run_pandas(D["y_pd"], method, D["x0"])
            t_ref_first = time.perf_counter() - t_start
            ref = to_triplet(ref_out)
            fns, checks, ek_only = {}, {}, None
            for eng in engines:
                if eng == "f" and n > args.baseline_max:
                    continue
                if eng == "f_pl" and n > args.baseline_pl_max:
                    continue
                fns[eng] = engine_fn(eng, method, D)
                if eng == "f":
                    checks[eng] = {"ok": True, "ref": True}
                    continue
                try:
                    o = fns[eng]()  # warmup + check
                except Exception as e:  # noqa: BLE001
                    checks[eng] = {"ok": False, "error": repr(e)[:300]}
                    fns.pop(eng)
                    continue
                if eng == "ek":
                    ek_only = np.asarray(o).ravel()
                    checks[eng] = {
                        "ok": bool(np.allclose(ref[2], ek_only, rtol=1e-7, atol=1e-9)),
                        "values_only": True,
                        "max_abs": float(np.max(np.abs(ref[2] - ek_only))),
                    }
                else:
                    checks[eng] = compare(ref, o)
            # interleaved timing, min over repeats
            times = {e: [] for e in fns}
            if "f" in times:
                times["f"].append(t_ref_first)
            slow = {"f", "f_pl"}
            for rep in range(args.repeat):
                for eng, fn in fns.items():
                    if eng in slow and rep >= args.baseline_repeat:
                        continue
                    t1 = time.perf_counter()
                    fn()
                    times[eng].append(time.perf_counter() - t1)
            best = {e: min(v) for e, v in times.items() if v}
            row = {
                "ref": ref_name,
                "n_series": n,
                "T": T,
                "method": method,
                "best_s": best,
                "n_runs": {e: len(v) for e, v in times.items()},
                "check": checks,
            }
            if "f" in best:
                row["speedup_vs_f"] = {e: best["f"] / t for e, t in best.items()}
            if "a" in best:
                row["speedup_vs_a"] = {e: best["a"] / t for e, t in best.items()}
            out["results"].append(row)
            msg = "  ".join(
                f"{e}={best[e] * 1e3:.2f}ms"
                + ("" if checks.get(e, {}).get("ok", True) else "(!)")
                for e in best
            )
            print(f"  {method:8s} {msg}", flush=True)
            bad = {e: c for e, c in checks.items() if not c.get("ok")}
            if bad:
                print(f"    CHECK FAIL: {bad}", flush=True)
        D["con"].close()
        del D
    fn = os.path.join(RES, f"x7_native_{args.grid}_t{N_THREADS}{args.tag}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", fn)


# ----------------------------------------------------------------------------
# memory: one subprocess per (engine, method) so allocator state is clean
# ----------------------------------------------------------------------------


def mem_one(eng, method, n, T):
    from common import peak_rss_delta_mb

    silence_sktime_warnings()
    D = build_inputs(n, T)
    fn = engine_fn(eng, method, D)
    import gc

    gc.collect()
    base = rss_mb()
    res, peak = peak_rss_delta_mb(fn, interval=0.002)
    gc.collect()
    retained = rss_mb() - base  # result (incl. fitted forecaster state) kept alive
    out_mb = None
    if isinstance(res, pd.DataFrame):
        out_mb = res.memory_usage(deep=True).sum() / 2**20
    print(
        json.dumps(
            {
                "engine": eng,
                "method": method,
                "n": n,
                "T": T,
                "peak_delta_mb": peak,
                "retained_mb": retained,
                "input_pd_mb": D["y_pd"].memory_usage(deep=True).sum() / 2**20,
                "input_pl_mb": D["y_pl"].estimated_size() / 2**20,
                "out_pd_mb": out_mb,
            }
        )
    )


def fitted_state_one(method, n, T):
    """Size of fitted state: sktime forecasters_ vs a native params table."""
    import gc
    import pickle

    silence_sktime_warnings()
    D = build_inputs(n, T)
    f = make_sktime(method)()
    gc.collect()
    base = rss_mb()
    f.fit(D["y_pd"], fh=list(HS))
    gc.collect()
    retained_fit = rss_mb() - base
    pickled = len(pickle.dumps(f)) / 2**20
    # native params table, polars: one row per series
    p = (
        D["y_pl"]
        .group_by("h0", maintain_order=True)
        .agg(pl.col("time").last(), pl.col("c0").last())
    )
    print(
        json.dumps(
            {
                "method": method,
                "n": n,
                "T": T,
                "sktime_fit_retained_rss_mb": retained_fit,
                "sktime_fitted_pickle_mb": pickled,
                "native_params_table_mb": p.estimated_size() / 2**20,
                "input_pd_mb": D["y_pd"].memory_usage(deep=True).sum() / 2**20,
            }
        )
    )


def main_mem(args):
    n, T = (10_000, 100) if args.grid == "dev" else (100_000, 100)
    if args.sizes:
        n, T = (int(v) for v in args.sizes.split(",")[0].split("x"))
    methods = args.methods.split(",") if args.methods else ["last", "ols"]
    engines = args.engines.split(",") if args.engines else ENGINES
    rows = []
    for method in methods:
        for eng in engines:
            cmd = [
                sys.executable,
                "-W",
                "ignore",
                __file__,
                "--threads",
                str(N_THREADS),
                "--mem-one",
                eng,
                method,
                str(n),
                str(T),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
            if not line:
                print("FAIL", eng, method, r.stderr[-500:])
                continue
            d = json.loads(line[-1])
            rows.append(d)
            print(
                f"{method:6s} {eng:5s} peak+{d['peak_delta_mb']:.1f}MB "
                f"retained {d['retained_mb']:.1f}MB",
                flush=True,
            )
        cmd = [
            sys.executable,
            "-W",
            "ignore",
            __file__,
            "--fitted-state",
            method,
            str(n),
            str(T),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        if line:
            d = json.loads(line[-1])
            d["engine"] = "fitted_state"
            rows.append(d)
            print(d, flush=True)
    fn = os.path.join(RES, f"x7_native_mem_{args.grid}_t{N_THREADS}{args.tag}.json")
    with open(fn, "w") as f:
        json.dump({"env": env_info(), "rows": rows}, f, indent=1)
    print("wrote", fn)


# ----------------------------------------------------------------------------
# end-to-end from a wide parquet file: what laziness / pushdown buys
# ----------------------------------------------------------------------------


def parquet_prepare(path, n, T, n_extra):
    y = make_panel_fast(n, T, n_cols=1 + n_extra)
    y = y.rename(columns={f"c{j}": f"extra{j}" for j in range(1, 1 + n_extra)})
    to_polars_long(y).write_parquet(path, row_group_size=max(n * T // 16, 1024))
    return y


def parquet_one(variant, method, path, x0):
    """Run one end-to-end variant in a fresh process, report time + peak RSS."""
    from common import peak_rss_delta_mb

    silence_sktime_warnings()
    tmin = np.datetime64("2000-01-01", "ns")
    t0 = pl.lit(tmin)

    def go():
        if variant == "sktime_today_lazy_in":
            # user hands sktime a LazyFrame-derived frame: today the whole table is
            # collected, converted to pandas MultiIndex, looped per series.
            lf = pl.scan_parquet(path)
            df = lf.collect()
            ypd = df.to_pandas().set_index(["h0", "time"])[["c0"]]
            return run_sktime(ypd, method)
        if variant == "pandas_read_all":
            ypd = pd.read_parquet(path).set_index(["h0", "time"])[["c0"]]
            return run_pandas(ypd, method, x0)
        if variant == "pandas_read_cols":
            ypd = pd.read_parquet(path, columns=["h0", "time", "c0"])
            ypd = ypd.set_index(["h0", "time"])
            return run_pandas(ypd, method, x0)
        if variant == "polars_eager_read_all":
            return pl_plan(pl.read_parquet(path), method, t0)
        if variant == "polars_lazy_scan":
            return pl_plan(pl.scan_parquet(path), method, t0).collect()
        if variant == "polars_lazy_scan_streaming":
            return pl_plan(pl.scan_parquet(path), method, t0).collect(
                engine="streaming"
            )
        if variant == "duckdb_parquet":
            import duckdb

            con = duckdb.connect()
            con.execute(f"SET threads={N_THREADS}")
            con.execute(f"CREATE VIEW y AS SELECT * FROM read_parquet('{path}')")
            return con.execute(duck_sql(method, x0)).fetch_arrow_table()
        raise ValueError(variant)

    t1 = time.perf_counter()
    res, peak = peak_rss_delta_mb(go, interval=0.002)
    dt = time.perf_counter() - t1
    print(
        json.dumps(
            {
                "variant": variant,
                "method": method,
                "wall_s": dt,
                "peak_delta_mb": peak,
                "n_out": len(res),
            }
        )
    )


PARQUET_VARIANTS = [
    "sktime_today_lazy_in",
    "pandas_read_all",
    "pandas_read_cols",
    "polars_eager_read_all",
    "polars_lazy_scan",
    "polars_lazy_scan_streaming",
    "duckdb_parquet",
]


def main_parquet(args):
    n, T = (10_000, 100) if args.grid == "dev" else (100_000, 100)
    n_extra = args.n_extra
    scratch = os.environ.get("X7_SCRATCH", "/tmp")
    path = os.path.join(scratch, f"x7_panel_{n}x{T}_extra{n_extra}.parquet")
    if not os.path.exists(path):
        parquet_prepare(path, n, T, n_extra)
    x0 = float((pd.Timestamp("2000-01-01") - pd.Timestamp(0)) / pd.Timedelta(days=1))
    rows = []
    methods = args.methods.split(",") if args.methods else ["last", "ols"]
    variants = args.engines.split(",") if args.engines else PARQUET_VARIANTS
    for rep in range(args.repeat_parquet):
        for method in methods:
            for v in variants:
                cmd = [
                    sys.executable,
                    "-W",
                    "ignore",
                    __file__,
                    "--threads",
                    str(N_THREADS),
                    "--parquet-one",
                    v,
                    method,
                    path,
                    repr(x0),
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
                line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
                if not line:
                    print("FAIL", v, method, r.stderr[-800:])
                    continue
                d = json.loads(line[-1])
                d["rep"] = rep
                rows.append(d)
                print(
                    f"{method:5s} {v:28s} {d['wall_s'] * 1e3:9.1f}ms "
                    f"peak+{d['peak_delta_mb']:.1f}MB",
                    flush=True,
                )
    out = {
        "env": env_info(),
        "threads": N_THREADS,
        "n": n,
        "T": T,
        "n_extra_cols": n_extra,
        "file_mb": os.path.getsize(path) / 2**20,
        "rows": rows,
    }
    fn = os.path.join(RES, f"x7_native_parquet_{args.grid}_t{N_THREADS}{args.tag}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", fn)


if __name__ == "__main__":
    if "--mem-one" in sys.argv:
        i = sys.argv.index("--mem-one")
        e, m, n, T = sys.argv[i + 1 : i + 5]
        mem_one(e, m, int(n), int(T))
        sys.exit(0)
    if "--fitted-state" in sys.argv:
        i = sys.argv.index("--fitted-state")
        m, n, T = sys.argv[i + 1 : i + 4]
        fitted_state_one(m, int(n), int(T))
        sys.exit(0)
    if "--parquet-one" in sys.argv:
        i = sys.argv.index("--parquet-one")
        v, m, p, x0 = sys.argv[i + 1 : i + 5]
        parquet_one(v, m, p, float(x0))
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sizes", default="", help="e.g. 100x100,1000x100")
    ap.add_argument("--methods", default="")
    ap.add_argument("--engines", default="")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--baseline-repeat", type=int, default=1)
    ap.add_argument("--baseline-max", type=int, default=10_000)
    ap.add_argument("--baseline-pl-max", type=int, default=1_000)
    ap.add_argument("--ols-default", action="store_true")
    ap.add_argument("--mem", action="store_true")
    ap.add_argument("--parquet", action="store_true")
    ap.add_argument("--n-extra", type=int, default=8)
    ap.add_argument("--repeat-parquet", type=int, default=2)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.mem:
        main_mem(args)
    elif args.parquet:
        main_parquet(args)
    else:
        main_bench(args)
