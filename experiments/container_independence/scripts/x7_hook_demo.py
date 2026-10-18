"""x7: correctness + cost of the panel-native hook prototype (protos/x7_hook.py).

Equality checks of NaivePanelNative vs stock NaiveForecaster (per-instance loop) on:
pandas Panel / multivariate / Hierarchical / NaN / int time / period(M) time /
ragged ends; polars (sktime __index__ convention); polars LazyFrame from parquet;
fallbacks (strategy="mean"; predict_interval materialization; update).

Timing: fit+predict, native pandas / polars / polars-lazy vs stock loop.

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x7_hook_demo.py --grid dev --threads 1
"""

import argparse
import json
import os
import sys
import time
import warnings

_n = "1"
if "--threads" in sys.argv:
    _n = sys.argv[sys.argv.index("--threads") + 1]
os.environ["POLARS_MAX_THREADS"] = _n

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from polars.testing import assert_frame_equal as pl_assert  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import bench, env_info, make_panel, make_panel_fast  # noqa: E402
from protos.x7_hook import NaivePanelNative, detect_panel  # noqa: E402

from sktime.datatypes import check_is_scitype, convert_to  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402

warnings.simplefilter("ignore")
import sktime.utils.warnings as _sw  # noqa: E402

_sw._SuppressWarningPattern.__exit__ = lambda self, *a: warnings.simplefilter("ignore")

FH = list(range(1, 15))
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SCRATCH = os.environ.get("X7_SCRATCH", "/tmp")


def pd_equal(a, b):
    try:
        pd.testing.assert_frame_equal(a, b, check_freq=False)
        return True, ""
    except AssertionError as e:
        return False, str(e)[:400]


def pl_equal(a, b):
    try:
        pl_assert(a, b)
        return True, ""
    except AssertionError as e:
        return False, str(e)[:400]


def checks():
    out = {}

    def rec(name, ok, why="", native=None):
        out[name] = {"ok": bool(ok), "why": why, "native_path": native}
        flag = "OK " if ok else "FAIL"
        print(f"[{flag}] {name} native={native} {why[:200]}", flush=True)

    # 1 pandas panel
    y = make_panel_fast(50, 40)
    f = NaivePanelNative().fit(y, fh=FH)
    rec("pandas_panel", *pd_equal(f.predict(), NaiveForecaster().fit(y, fh=FH).predict()),
        native=f._panel_native)
    # 2 multivariate
    y2 = make_panel_fast(30, 40, n_cols=3)
    f = NaivePanelNative().fit(y2, fh=FH)
    rec("pandas_multivariate", *pd_equal(
        f.predict(), NaiveForecaster().fit(y2, fh=FH).predict()), native=f._panel_native)
    # 3 hierarchical
    yh = make_panel(40, 30, n_levels=2)
    f = NaivePanelNative().fit(yh, fh=FH)
    rec("pandas_hierarchical", *pd_equal(
        f.predict(), NaiveForecaster().fit(yh, fh=FH).predict()), native=f._panel_native)
    # 4 NaN: trailing NaN in some series, interior NaN in others
    yn = make_panel_fast(20, 30, n_cols=2).copy()
    yn.iloc[29, 0] = np.nan  # last obs of s0, col 0
    yn.iloc[28:30, 1] = np.nan
    yn.iloc[45, 0] = np.nan
    f = NaivePanelNative().fit(yn, fh=FH)
    rec("pandas_nan", *pd_equal(
        f.predict(), NaiveForecaster().fit(yn, fh=FH).predict()), native=f._panel_native)
    # 5 integer time
    yi = make_panel_fast(20, 30, time_kind="int")
    f = NaivePanelNative().fit(yi, fh=FH)
    rec("pandas_int_time", *pd_equal(
        f.predict(), NaiveForecaster().fit(yi, fh=FH).predict()), native=f._panel_native)
    # 6 period monthly (calendar offset)
    yp = make_panel_fast(20, 30, time_kind="period")
    f = NaivePanelNative().fit(yp, fh=FH)
    rec("pandas_period_M", *pd_equal(
        f.predict(), NaiveForecaster().fit(yp, fh=FH).predict()), native=f._panel_native)
    # 7 ragged ends: drop last 3 obs of every 3rd series
    yr = make_panel_fast(30, 40)
    keep = np.ones(len(yr), bool)
    codes = yr.index.codes[0]
    tpos = yr.index.codes[1]
    keep[(codes % 3 == 0) & (tpos >= 37)] = False
    yr = yr[keep]
    f = NaivePanelNative().fit(yr, fh=FH)
    rec("pandas_ragged_ends", *pd_equal(
        f.predict(), NaiveForecaster().fit(yr, fh=FH).predict()), native=f._panel_native)
    # 8 polars sktime convention
    ypl = convert_to(make_panel_fast(50, 40, n_cols=2), "polars_panel")
    f = NaivePanelNative().fit(ypl, fh=FH)
    pn = f.predict()
    rec("polars_panel", *pl_equal(pn, NaiveForecaster().fit(ypl, fh=FH).predict()),
        native=f._panel_native)
    rec("polars_panel_output_is_polars", isinstance(pn, pl.DataFrame), str(type(pn)))
    # 9 polars lazy from parquet with extra columns -> projection pushdown
    path = os.path.join(SCRATCH, "x7_hook_demo.parquet")
    wide = ypl.with_columns(
        *[pl.lit(float(j)).alias(f"unused{j}") for j in range(6)]
    )
    wide.write_parquet(path)
    lf = pl.scan_parquet(path).select(ypl.columns)  # user selects what is the panel
    f = NaivePanelNative().fit(lf, fh=FH)
    lp = f.predict()
    rec("polars_lazy_returns_lazy", isinstance(lp, pl.LazyFrame), str(type(lp)))
    rec("polars_lazy_equals_eager", *pl_equal(lp.collect(), pn), native=f._panel_native)
    plan = (
        pl.scan_parquet(path)
        .group_by(["__index__h0"], maintain_order=True)
        .agg(pl.col("__index__time").last(), pl.col("c0").last())
        .explain()
    )
    proj = [ln.strip() for ln in plan.splitlines() if "PROJECT" in ln or "project" in ln]
    rec("polars_lazy_projection_pushdown", len(proj) > 0, " | ".join(proj)[:200])
    # stock sktime on a LazyFrame panel
    try:
        NaiveForecaster().fit(lf, fh=FH).predict()
        rec("stock_sktime_accepts_lazy_panel", True, "")
    except Exception as e:  # noqa: BLE001
        rec("stock_sktime_accepts_lazy_panel", False, repr(e)[:200])
    # 10 fallback: strategy not supported natively -> loop, identical
    f = NaivePanelNative(strategy="mean", window_length=10).fit(y, fh=FH)
    rec("fallback_mean_equal", *pd_equal(
        f.predict(),
        NaiveForecaster(strategy="mean", window_length=10).fit(y, fh=FH).predict()),
        native=f._panel_native)
    f = NaivePanelNative(sp=7).fit(y, fh=FH)
    rec("fallback_sp7_equal", *pd_equal(
        f.predict(), NaiveForecaster(sp=7).fit(y, fh=FH).predict()),
        native=f._panel_native)
    # 11 predict_interval via materialization
    f = NaivePanelNative().fit(y, fh=FH)
    pi = f.predict_interval(coverage=0.9)
    rec("materialize_predict_interval", *pd_equal(
        pi, NaiveForecaster().fit(y, fh=FH).predict_interval(coverage=0.9)),
        native=f._panel_native)
    # 12 update natively (upsert params) vs stock fit+update
    ya = y[y.index.get_level_values(-1) < y.index.levels[-1][35]]
    yb = y[y.index.get_level_values(-1) >= y.index.levels[-1][35]]
    f = NaivePanelNative().fit(ya, fh=FH)
    f.update(yb)
    s = NaiveForecaster().fit(ya, fh=FH)
    s.update(yb)
    rec("native_update_pandas", *pd_equal(f.predict(), s.predict()),
        native=f._panel_native)
    ypa = convert_to(ya, "polars_panel")
    ypb = convert_to(yb, "polars_panel")
    f = NaivePanelNative().fit(ypa, fh=FH)
    f.update(ypb)
    s = NaiveForecaster().fit(ypa, fh=FH)
    s.update(ypb)
    rec("native_update_polars", *pl_equal(f.predict(), s.predict()),
        native=f._panel_native)
    # 13 fitted params
    f = NaivePanelNative().fit(ypl, fh=FH)
    fp = f.get_fitted_params()
    rec("fitted_params_is_native_table", isinstance(fp["params_table"], pl.DataFrame),
        f"{type(fp['params_table']).__name__} shape={fp['params_table'].shape}")
    return out


def timing(grid, baseline_max):
    rows = []
    for n, T in grid:
        y = make_panel_fast(n, T)
        ypl = convert_to(y, "polars_panel")
        lf = ypl.lazy()
        fns = {
            "native_pandas": lambda: NaivePanelNative().fit(y, fh=FH).predict(),
            "native_polars": lambda: NaivePanelNative().fit(ypl, fh=FH).predict(),
            "native_polars_lazy": lambda: NaivePanelNative()
            .fit(lf, fh=FH)
            .predict()
            .collect(),
            # cost if full sktime validation were kept on the native path
            "native_pandas+check_is_scitype": lambda: (
                check_is_scitype(y, ["Panel", "Hierarchical"], return_metadata=[]),
                NaivePanelNative().fit(y, fh=FH).predict(),
            ),
            "native_polars+check_is_scitype": lambda: (
                check_is_scitype(ypl, ["Panel", "Hierarchical"], return_metadata=[]),
                NaivePanelNative().fit(ypl, fh=FH).predict(),
            ),
            "detect_panel_only_pandas": lambda: detect_panel(y),
        }
        if n <= baseline_max:
            fns["stock_loop_pandas"] = lambda: NaiveForecaster().fit(y, fh=FH).predict()
            fns["stock_loop_polars"] = lambda: NaiveForecaster().fit(ypl, fh=FH).predict()
        res = {}
        for k, fn in fns.items():
            rep = 1 if k.startswith("stock") else 5
            res[k] = bench(fn, repeat=rep, warmup=0 if k.startswith("stock") else 1)
        row = {"n_series": n, "T": T, "best_s": res}
        if "stock_loop_pandas" in res:
            row["speedup_vs_stock_pandas"] = {
                k: res["stock_loop_pandas"] / v for k, v in res.items()
            }
        rows.append(row)
        print(n, T, {k: f"{v * 1e3:.2f}ms" for k, v in res.items()}, flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--baseline-max", type=int, default=1000)
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()
    grid = [(100, 100), (1000, 100), (10000, 100)]
    if args.grid == "large":
        grid = [(1000, 100), (10000, 100), (100000, 100)]
    out = {"env": env_info(), "threads": args.threads}
    t0 = time.perf_counter()
    out["checks"] = checks()
    print("checks took", time.perf_counter() - t0)
    if not args.skip_timing:
        out["timing"] = timing(grid, args.baseline_max)
    fn = os.path.join(RES, f"x7_hook_demo_{args.grid}_t{args.threads}.json")
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", fn)
