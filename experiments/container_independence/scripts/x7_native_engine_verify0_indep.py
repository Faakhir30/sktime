"""x7 verify0: independent correctness + fairness checks for x7_native.py / x7_hook.py.

1. Independent per-series reference (plain numpy loop + np.polyfit, NOT sktime, NOT
   the x7 vectorized formulas) vs every x7 engine, on a fresh seed.
2. Reproduce the PolynomialTrendForecaster default (sklearn tol) discrepancy vs polyfit.
3. Evidence lazy/duckdb paths really execute: output row counts, time scaling 1k->10k.
4. Asymmetric accounting: cost of pandas->polars / ->arrow conversion that engines b/c/d
   do not pay in x7_native, relative to (a).
5. Adversarial inputs for engines + hook: time not sorted within instance, polars rows
   interleaved across instances.

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x7_native_engine_verify0_indep.py --grid dev
"""

import argparse
import json
import os
import sys

sys.argv += [] if "--threads" in sys.argv else ["--threads", "1"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import x7_native as X  # noqa: E402  (sets POLARS_MAX_THREADS from argv)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from common import bench, make_panel_fast, to_polars_long  # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")


def indep_reference(y, method):
    """Per-series python loop on raw arrays; np.polyfit for OLS."""
    out_k, out_t, out_v = [], [], []
    for key, sub in y["c0"].groupby(level=0, sort=True):
        v = sub.to_numpy()
        t = sub.index.get_level_values(-1)
        tl = t[-1]
        fut = [tl + pd.Timedelta(days=int(h)) for h in X.HS]
        if method == "last":
            p = [v[-1]] * X.H
        elif method == "snaive7":
            season = v[-7:]
            p = [season[(h - 1) % 7] for h in X.HS]
        elif method == "mean10":
            p = [float(np.mean(v[-10:]))] * X.H
        elif method == "drift":
            s = (v[-1] - v[0]) / (len(v) - 1)
            p = [v[-1] + s * h for h in X.HS]
        elif method == "ols":
            xd = np.asarray((t - pd.Timestamp(0)) / pd.Timedelta(days=1), float)
            xf = np.asarray([(f - pd.Timestamp(0)) / pd.Timedelta(days=1) for f in fut])
            c = np.polyfit(xd - xd.mean(), v, 1)  # centred -> well conditioned
            p = list(np.polyval(c, xf - xd.mean()))
        out_k += [key] * X.H
        out_t += [np.datetime64(f, "ns").astype("int64") for f in fut]
        out_v += list(p)
    k, t, v = np.asarray(out_k), np.asarray(out_t), np.asarray(out_v, float)
    o = np.lexsort((t, k))
    return k[o], t[o], v[o]


def part1(n, T, seed):
    y = make_panel_fast(n, T, seed=seed)
    D = X.build_inputs(n, T)
    D["y_pd"] = y
    D["y_pl"] = to_polars_long(y)
    D["con"].register("y", D["y_pl"].to_arrow())
    rows = {}
    for m in X.METHODS:
        ref = indep_reference(y, m)
        r = {}
        for eng in ["a", "b", "c", "c_s", "d", "e", "e2", "f"]:
            o = X.engine_fn(eng, m, D)()
            c = X.compare(ref, o)
            r[eng] = {"ok": c["ok"], "max_abs": c.get("max_abs"), "type": type(o).__name__}
        rows[m] = r
        print(m, {e: (v["ok"], f"{v['max_abs']:.1e}") for e, v in r.items()}, flush=True)
    D["con"].close()
    return rows


def part2(n, T):
    """stock default PolynomialTrendForecaster vs polyfit."""
    from sktime.forecasting.trend import PolynomialTrendForecaster

    y = make_panel_fast(n, T, seed=1)
    ref = indep_reference(y, "ols")
    o = PolynomialTrendForecaster(degree=1).fit(y, fh=list(X.HS)).predict()
    c = X.compare(ref, o)
    from sklearn.linear_model import LinearRegression

    o2 = PolynomialTrendForecaster(
        degree=1, regressor=LinearRegression(fit_intercept=False, tol=1e-12)
    ).fit(y, fh=list(X.HS)).predict()
    c2 = X.compare(ref, o2)
    print("stock default vs polyfit:", c, "\n tol=1e-12 vs polyfit:", c2, flush=True)
    return {"default": c, "tol1e-12": c2}


def part3_4(sizes):
    out = {}
    for n in sizes:
        y = make_panel_fast(n, 100)
        D = X.build_inputs(n, 100)
        r = {}
        for eng in ["a", "b", "c", "c_s", "d", "e2"]:
            fn = X.engine_fn(eng, "ols", D)
            res = fn()
            r[eng] = {"t_ms": bench(fn, repeat=5) * 1e3, "rows_out": len(res)}
        # conversion costs not paid by b/c/d in x7_native
        r["conv_pd_to_pl_ms"] = bench(lambda: to_polars_long(y), repeat=5) * 1e3
        r["conv_pd_to_arrow_ms"] = (
            bench(lambda: to_polars_long(y).to_arrow(), repeat=5) * 1e3
        )
        r["conv_pl_back_to_pd_mi_ms"] = bench(
            lambda: X.pl_plan(D["y_pl"], "ols", D["t0"]).to_pandas().set_index(
                ["h0", "time"]
            ),
            repeat=5,
        ) * 1e3
        for e in ["b", "c", "c_s", "d"]:
            r[f"{e}_incl_conv_ratio_vs_a"] = r["a"]["t_ms"] / (
                r[e]["t_ms"] + r["conv_pd_to_pl_ms"]
            )
            r[f"{e}_excl_conv_ratio_vs_a"] = r["a"]["t_ms"] / r[e]["t_ms"]
        D["con"].close()
        out[n] = r
        print(n, json.dumps(r, default=float)[:900], flush=True)
    if len(sizes) == 2:
        a, b = sizes
        out["scaling"] = {
            e: out[b][e]["t_ms"] / out[a][e]["t_ms"]
            for e in ["a", "b", "c", "c_s", "d", "e2"]
        }
        print("scaling", out["scaling"])
    return out


def part5():
    """unsorted time within instance; interleaved polars rows."""
    from sktime.datatypes import convert_to
    from sktime.forecasting.naive import NaiveForecaster

    from protos.x7_hook import NaivePanelNative

    out = {}
    y = make_panel_fast(20, 30, seed=3)
    rng = np.random.default_rng(0)
    ys = y.iloc[rng.permutation(len(y))]  # rows shuffled (keys + time)
    ref = indep_reference(y, "last")  # truth on sorted data
    D = X.build_inputs(20, 30)
    D["y_pd"] = ys
    D["y_pl"] = to_polars_long(ys)
    D["con"].register("y", D["y_pl"].to_arrow())
    for eng in ["a", "b", "c", "d", "e2"]:
        try:
            c = X.compare(ref, X.engine_fn(eng, "last", D)())
        except Exception as e:  # noqa: BLE001
            c = {"ok": False, "err": repr(e)[:120]}
        out[f"engine_{eng}_shuffled"] = c
    try:
        so = NaiveForecaster().fit(ys, fh=list(X.HS)).predict()
        out["stock_sktime_shuffled"] = X.compare(ref, so)
    except Exception as e:  # noqa: BLE001
        out["stock_sktime_shuffled"] = {"ok": False, "err": repr(e)[:160]}
    try:
        ho = NaivePanelNative().fit(ys, fh=list(X.HS)).predict()
        out["hook_pandas_shuffled"] = X.compare(ref, ho)
    except Exception as e:  # noqa: BLE001
        out["hook_pandas_shuffled"] = {"ok": False, "err": repr(e)[:160]}
    # polars: sktime convention, rows interleaved/shuffled
    ypl = convert_to(y, "polars_panel")
    ypls = ypl.sample(fraction=1.0, shuffle=True, seed=0)
    for nm, fr in [("polars_shuffled", ypls), ("polars_lazy_shuffled", ypls.lazy())]:
        try:
            ho = NaivePanelNative().fit(fr, fh=list(X.HS)).predict()
            if isinstance(ho, pl.LazyFrame):
                ho = ho.collect()
            out[f"hook_{nm}"] = X.compare(ref, ho)
        except Exception as e:  # noqa: BLE001
            out[f"hook_{nm}"] = {"ok": False, "err": repr(e)[:160]}
    try:
        so = NaiveForecaster().fit(ypls, fh=list(X.HS)).predict()
        out["stock_sktime_polars_shuffled"] = X.compare(ref, so)
    except Exception as e:  # noqa: BLE001
        out["stock_sktime_polars_shuffled"] = {"ok": False, "err": repr(e)[:160]}
    for k, v in out.items():
        print(k, {kk: vv for kk, vv in v.items() if kk in ("ok", "max_abs", "err", "keys", "time")})
    return out


if __name__ == "__main__":
    X.silence_sktime_warnings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--threads", default="1")
    a = ap.parse_args()
    res = {}
    res["part1_indep"] = part1(300, 100, seed=42)
    res["part2_ols_default"] = part2(100, 100)
    res["part3_4"] = part3_4([1000, 10000] if a.grid == "dev" else [10000, 100000])
    res["part5_adversarial"] = part5()
    fn = os.path.join(RES, f"x7_native_engine_verify0_indep_{a.grid}.json")
    with open(fn, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", fn)
