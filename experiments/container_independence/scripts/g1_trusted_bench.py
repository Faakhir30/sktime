"""G1: trusted per-slice path vs stock, equality + interleaved timing.

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $PY -W ignore \
        experiments/container_independence/scripts/g1_trusted_bench.py --grid dev

$PY must have statsmodels for ets/theta/ttf_deseas (own venv g1_trusted_path);
those estimators are skipped otherwise.

Workload per (estimator, scitype, container): fit(y_train, fh=[1,2,3]) ->
predict() -> predict_interval(coverage=[0.8, 0.9]) -> update(y_new) -> predict().
Each phase timed separately; stock and trusted interleaved in one process,
min over reps; ratio = stock / trusted (>1 means trusted faster).
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import env_info, make_panel, make_panel_fast  # noqa: E402

import g1_trusted as g1  # noqa: E402

FH = [1, 2, 3]
COV = [0.8, 0.9]
N_UPD = 5

HAS_SM = True
try:
    import statsmodels  # noqa: F401
except ImportError:
    HAS_SM = False


def make_estimator(name):
    if name == "naive":
        from sktime.forecasting.naive import NaiveForecaster

        return NaiveForecaster(strategy="last")
    if name == "poly":
        from sktime.forecasting.trend import PolynomialTrendForecaster

        return PolynomialTrendForecaster(degree=1, prediction_intervals=True)
    if name == "ets":
        from sktime.forecasting.exp_smoothing import ExponentialSmoothing

        return ExponentialSmoothing(trend="add")
    if name == "theta":
        from sktime.forecasting.theta import ThetaForecaster

        return ThetaForecaster(sp=1, deseasonalize=False)
    if name == "reduce":
        from sklearn.linear_model import LinearRegression

        from sktime.forecasting.compose import make_reduction

        return make_reduction(LinearRegression(), window_length=5)
    if name == "ttf_detrend":
        from sktime.forecasting.compose import TransformedTargetForecaster
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.detrend import Detrender

        return TransformedTargetForecaster(
            [("detrend", Detrender()), ("f", NaiveForecaster(strategy="last"))]
        )
    if name == "ttf_deseas":
        from sktime.forecasting.compose import TransformedTargetForecaster
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.detrend import Deseasonalizer

        return TransformedTargetForecaster(
            [("deseas", Deseasonalizer(sp=7)), ("f", NaiveForecaster(strategy="last"))]
        )
    raise ValueError(name)


ESTIMATORS = ["naive", "poly", "ets", "theta", "reduce", "ttf_detrend", "ttf_deseas"]
NEEDS_SM = {"ets", "theta", "ttf_deseas"}
# per-instance inner cost is high for statsmodels -> cap n
N_CAP = {"ets": 100, "theta": 1000, "ttf_deseas": 1000}


def make_data(scitype, n, T):
    tot = T + N_UPD
    y = make_panel_fast(n, tot) if scitype == "Panel" else make_panel(n, tot, n_levels=2)
    t = y.index.get_level_values(-1)
    cut = np.sort(t.unique())[T - 1]
    return y[t <= cut], y[t > cut]


def to_container(y, scitype, cont):
    if cont == "pd":
        return y
    from sktime.datatypes import convert_to

    return convert_to(y, "polars_panel" if scitype == "Panel" else "polars_hierarchical")


def to_pd(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    from sktime.datatypes._adapter.polars import convert_polars_to_pandas

    return convert_polars_to_pandas(obj)


def run_once(proto, y_tr, y_new, mode, times, outs, do_int, do_upd):
    """One full workload; mode in {'stock','trusted'}; fills times/outs."""
    if mode == "trusted":
        g1.install("schema_items")
    elif mode == "trusted_ve":
        g1.install("vectorize_est")
    else:
        g1.uninstall()
    try:
        f = proto.clone()
        t0 = time.perf_counter()
        f.fit(y_tr, fh=FH)
        t1 = time.perf_counter()
        p = f.predict()
        t2 = time.perf_counter()
        times["fit"] = t1 - t0
        times["predict"] = t2 - t1
        outs["pred"] = to_pd(p)
        outs["cutoff"] = f.cutoff
        outs["est"] = f
        if do_int:
            t0 = time.perf_counter()
            pi = f.predict_interval(coverage=COV)
            times["predict_interval"] = time.perf_counter() - t0
            outs["int"] = to_pd(pi)
        if do_upd:
            t0 = time.perf_counter()
            f.update(y_new)
            t1 = time.perf_counter()
            p2 = f.predict()
            t2 = time.perf_counter()
            times["update"] = t1 - t0
            times["predict_after_update"] = t2 - t1
            outs["pred_upd"] = to_pd(p2)
            outs["cutoff_upd"] = f.cutoff
    finally:
        g1.uninstall()


def _eq_obj(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, (pd.DataFrame, pd.Series)):
        try:
            if isinstance(a, pd.DataFrame):
                pd.testing.assert_frame_equal(a, b)
            else:
                pd.testing.assert_series_equal(a, b)
            return True
        except AssertionError:
            return False
    if isinstance(a, pd.Index):
        return a.equals(b) and getattr(a, "freq", None) == getattr(b, "freq", None)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=True)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_eq_obj(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_eq_obj(x, y) for x, y in zip(a, b))
    try:
        r = a == b
        if isinstance(r, (bool, np.bool_)):
            return bool(r)
        return bool(np.all(r))
    except Exception:  # noqa: BLE001
        return repr(a) == repr(b)


SLICE_ATTRS = [
    "_cutoff",
    "_y",
    "_X",
    "_y_metadata",
    "_y_mtype_last_seen",
    "_converter_store_y",
    "_is_vectorized",
    "_state",
]


def compare_state(f_s, f_t):
    """Compare outer + per-slice state; returns list of mismatch strings."""
    bad = []
    if not _eq_obj(f_s.cutoff, f_t.cutoff):
        bad.append("outer cutoff")
    if hasattr(f_s, "forecasters_") != hasattr(f_t, "forecasters_"):
        bad.append("forecasters_ presence")
        return bad
    if not hasattr(f_s, "forecasters_"):
        return bad
    A, B = f_s.forecasters_, f_t.forecasters_
    if not (A.index.equals(B.index) and A.columns.equals(B.columns)):
        bad.append("forecasters_ index/columns")
        return bad
    for a, b in zip(A.values.ravel(), B.values.ravel()):
        if type(a) is not type(b):
            bad.append("slice type")
            break
        for att in SLICE_ATTRS:
            if not _eq_obj(getattr(a, att, None), getattr(b, att, None)):
                bad.append(f"slice {att}")
        fa, fb = getattr(a, "_fh", None), getattr(b, "_fh", None)
        if (fa is None) != (fb is None) or (
            fa is not None
            and not (
                np.array_equal(fa.to_numpy(), fb.to_numpy())
                and fa.freq == fb.freq
                and fa.is_relative == fb.is_relative
            )
        ):
            bad.append("slice _fh")
        if bad:
            break
    try:
        ps, pt = f_s.get_fitted_params(), f_t.get_fitted_params()
        if ps.keys() != pt.keys():
            bad.append("fitted_params keys")
        else:
            diff = [k for k in ps if not _eq_obj(ps[k], pt[k])]
            # estimator objects compare by identity -> only compare non-estimators
            diff = [
                k for k in diff if not hasattr(ps[k], "get_params")
                and not isinstance(ps[k], pd.DataFrame)
            ]
            if diff:
                bad.append(f"fitted_params values {diff[:3]}")
    except Exception as e:  # noqa: BLE001
        bad.append(f"fitted_params err {type(e).__name__}")
    return sorted(set(bad))


def bench_case(est_name, scitype, cont, n, T, reps, check_state, modes):
    proto = make_estimator(est_name)
    y_tr_pd, y_new_pd = make_data(scitype, n, T)
    y_tr = to_container(y_tr_pd, scitype, cont)
    y_new = to_container(y_new_pd, scitype, cont)
    do_int = proto.get_tag("capability:pred_int")
    do_upd = True
    best = {m: {} for m in modes}
    first = {}
    # warmup (imports, caches) on a tiny case, untimed
    w_tr, w_new = make_data(scitype, 4, min(T, 30))
    w_tr, w_new = to_container(w_tr, scitype, cont), to_container(w_new, scitype, cont)
    for m in modes:
        try:
            run_once(proto, w_tr, w_new, m, {}, {}, do_int, do_upd)
        except Exception:  # noqa: BLE001
            pass
    g1.reset_stats()
    err = None
    for r in range(reps):
        order = modes if r % 2 == 0 else modes[::-1]
        for m in order:
            times, outs = {}, {}
            try:
                run_once(proto, y_tr, y_new, m, times, outs, do_int, do_upd)
            except Exception as e:  # noqa: BLE001
                err = f"{m}: {type(e).__name__}: {str(e)[:200]}"
                return {"error": err}
            for k, v in times.items():
                best[m][k] = min(best[m].get(k, np.inf), v)
            if r == 0:
                first[m] = outs
    stats = dict(g1.STATS)
    res = {
        "est": est_name, "scitype": scitype, "cont": cont, "n": n, "T": T,
        "reps": reps, "times": best,
        "trusted_calls": stats["trusted"], "fallbacks": stats["fallback_reasons"],
    }
    s = first["stock"]
    eq = {}
    for m in modes:
        if m == "stock":
            continue
        t = first[m]
        e = {k: _eq_obj(s[k], t[k]) for k in ("pred", "int", "pred_upd", "cutoff",
                                             "cutoff_upd") if k in s}
        if check_state:
            e["state_mismatch"] = compare_state(s["est"], t["est"])
        eq[m] = e
    res["equal"] = eq
    for m in modes:
        if m == "stock":
            continue
        res[f"ratio_{m}"] = {
            k: best["stock"][k] / best[m][k] for k in best["stock"] if k in best[m]
        }
        tot_s = sum(best["stock"].values())
        tot_t = sum(best[m].values())
        res[f"ratio_{m}"]["total"] = tot_s / tot_t
    return res


GRIDS = {
    "smoke": [(20, 50)],
    "dev": [(10, 100), (100, 100), (1000, 100)],
    "large": [(100, 500), (1000, 500), (5000, 100)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=list(GRIDS))
    ap.add_argument("--est", default=",".join(ESTIMATORS))
    ap.add_argument("--sc", default="Panel,Hierarchical")
    ap.add_argument("--cont", default="pd,pl")
    ap.add_argument("--modes", default="stock,trusted,trusted_ve")
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    modes = a.modes.split(",")
    rows = []
    for n, T in GRIDS[a.grid]:
        reps = a.reps or (3 if n <= 100 else (2 if n <= 1000 else 1))
        check_state = n <= 100
        for est in a.est.split(","):
            if est in NEEDS_SM and not HAS_SM:
                print(f"skip {est}: no statsmodels", flush=True)
                continue
            if n > N_CAP.get(est, 10**9):
                continue
            for sc in a.sc.split(","):
                for cont in a.cont.split(","):
                    t0 = time.perf_counter()
                    r = bench_case(est, sc, cont, n, T, reps, check_state, modes)
                    r.update(est=est, scitype=sc, cont=cont, n=n, T=T)
                    r["wall_s"] = time.perf_counter() - t0
                    rows.append(r)
                    _dump(a, rows)
                    if "error" in r:
                        print(f"{est:12s} {sc:12s} {cont} n={n} ERROR {r['error']}",
                              flush=True)
                        continue
                    rt = r.get("ratio_trusted", {})
                    rv = r.get("ratio_trusted_ve", {})
                    print(
                        f"{est:12s} {sc:12s} {cont} n={n:5d} T={T} "
                        f"stock_tot={sum(r['times']['stock'].values()):.3f}s "
                        "ratios(stock/trusted) "
                        + " ".join(f"{k}={v:.2f}" for k, v in rt.items())
                        + f" | ve_total={rv.get('total', float('nan')):.2f}"
                        + f" | eq={r['equal']} trusted_calls={r['trusted_calls']} "
                        f"fb={r['fallbacks']}",
                        flush=True,
                    )
    print("wrote", _dump(a, rows))


def _dump(a, rows):
    out = a.out or os.path.join(
        os.path.dirname(__file__), "..", "results", f"g1_trusted_bench_{a.grid}.json"
    )
    with open(out, "w") as fh:
        json.dump({"env": env_info(), "grid": a.grid, "rows": rows}, fh, indent=1,
                  default=str)
    return out


if __name__ == "__main__":
    main()
