"""G1: invariant / edge-case probes, trusted path vs stock (same outputs or same error).

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $PY -W ignore \
        experiments/container_independence/scripts/g1_invariants.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
sys.path.insert(0, os.path.dirname(__file__))

import g1_trusted as g1  # noqa: E402
from common import make_panel, make_panel_fast  # noqa: E402
from g1_trusted_bench import _eq_obj, compare_state  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402
from sktime.forecasting.trend import PolynomialTrendForecaster  # noqa: E402


def run(fn, mode):
    if mode == "stock":
        g1.uninstall()
    else:
        g1.install(mode)
    try:
        return ("ok", fn())
    except Exception as e:  # noqa: BLE001
        return ("err", f"{type(e).__name__}: {str(e)[:120]}")
    finally:
        g1.uninstall()


def same(a, b):
    if a[0] != b[0]:
        return False
    if a[0] == "err":
        return a[1].split(":")[0] == b[1].split(":")[0]
    va, vb = a[1], b[1]
    if isinstance(va, tuple):
        return all(_eq_obj(x, y) for x, y in zip(va, vb))
    return _eq_obj(va, vb)


Y = make_panel_fast(6, 40)
YH = make_panel(8, 40, n_levels=2)
Y_TR = Y[Y.index.get_level_values(-1) < Y.index.get_level_values(-1).unique()[35]]
Y_NEW = Y[Y.index.get_level_values(-1) >= Y.index.get_level_values(-1).unique()[35]]


def _sarimax():
    from sktime.forecasting.sarimax import SARIMAX

    return SARIMAX(order=(1, 0, 0))


def _x_for(y, k=2, extra=0, seed=1):
    rng = np.random.default_rng(seed)
    idx = y.index
    if extra:
        parts = []
        for key, g in y.groupby(level=0):
            t = g.index.get_level_values(-1)
            tt = pd.date_range(t[0], periods=len(t) + extra, freq="D")
            parts.append(pd.MultiIndex.from_product([[key], tt], names=idx.names))
        idx = parts[0].append(parts[1:])
    return pd.DataFrame(rng.normal(size=(len(idx), k)), index=idx, columns=["x0", "x1"])


def case_state(make, y, fh=(1, 2, 3)):
    def fn():
        f = make()
        f.fit(y, fh=list(fh))
        return f
    return fn


CASES = {}


def case(name):
    def deco(f):
        CASES[name] = f
        return f
    return deco


@case("multivariate y, univariate est (iterate_cols)")
def _():
    y = make_panel_fast(5, 30, n_cols=3)
    f = NaiveForecaster().fit(y, fh=[1, 2])
    return f.predict(), f.predict_interval(coverage=0.9), f.cutoff


@case("fh only in predict")
def _():
    f = NaiveForecaster().fit(Y)
    return f.predict(fh=[1, 2, 5]), f.cutoff


@case("absolute fh (DatetimeIndex) in predict")
def _():
    f = NaiveForecaster().fit(Y)
    t = Y.index.get_level_values(-1).max()
    return (f.predict(fh=pd.date_range(t + pd.Timedelta("1D"), periods=3, freq="D")),)


@case("in-sample fh, insample-capable (Naive)")
def _():
    f = NaiveForecaster().fit(Y, fh=[-2, -1, 0, 1])
    return (f.predict(),)


@case("in-sample fh, insample-incapable (SARIMAX? -> error expected)")
def _():
    from sktime.forecasting.trend import TrendForecaster

    f = TrendForecaster()
    f.set_tags(**{"capability:insample": False})
    f.fit(Y, fh=[-1, 1])
    return (f.predict(),)


@case("requires-fh-in-fit, fh missing -> error")
def _():
    from sktime.forecasting.compose import make_reduction
    from sklearn.linear_model import LinearRegression

    f = make_reduction(LinearRegression(), strategy="direct", window_length=3)
    f.fit(Y)
    return (f.predict(fh=[1, 2]),)


@case("requires-fh-in-fit, different fh in predict -> error")
def _():
    from sktime.forecasting.compose import make_reduction
    from sklearn.linear_model import LinearRegression

    f = make_reduction(LinearRegression(), strategy="direct", window_length=3)
    f.fit(Y, fh=[1, 2])
    return (f.predict(fh=[1, 2, 3]),)


@case("requires-fh-in-fit direct reducer ok")
def _():
    from sktime.forecasting.compose import make_reduction
    from sklearn.linear_model import LinearRegression

    f = make_reduction(LinearRegression(), strategy="direct", window_length=3)
    f.fit(Y, fh=[1, 2])
    return f.predict(), f.cutoff


@case("NaN in y, est without missing capability -> error")
def _():
    y = Y.copy()
    y.iloc[5, 0] = np.nan
    return (PolynomialTrendForecaster().fit(y, fh=[1]).predict(),)


@case("NaN in y, est with missing capability (Naive)")
def _():
    y = Y.copy()
    y.iloc[5, 0] = np.nan
    f = NaiveForecaster(strategy="mean").fit(y, fh=[1])
    return f.predict(), f.forecasters_.iloc[0, 0]._y_metadata.get("has_nans")


@case("X Panel, exogenous est (SARIMAX) fit/predict/update")
def _():
    X = _x_for(Y, extra=3)
    Xtr = X[X.index.get_level_values(-1).isin(Y_TR.index.get_level_values(-1))]
    f = _sarimax().fit(Y_TR, X=Xtr, fh=[1, 2, 3])
    p = f.predict(X=X)
    f.update(Y_NEW, X=X[X.index.get_level_values(-1).isin(Y_NEW.index.get_level_values(-1))])
    return p, f.predict(X=X), f.cutoff


@case("X with missing dates, X-y-must-have-same-index -> error")
def _():
    X = _x_for(Y).iloc[3:]
    return (_sarimax().fit(Y, X=X, fh=[1]).predict(),)


@case("X passed to non-exogenous est (ignored)")
def _():
    X = _x_for(Y, extra=3)
    f = NaiveForecaster().fit(Y, X=X.loc[Y.index], fh=[1, 2])
    return f.predict(X=X), f.forecasters_.iloc[0, 0]._X


@case("remember_data=False")
def _():
    f = PolynomialTrendForecaster().set_config(remember_data=False)
    f.fit(Y_TR, fh=[1, 2])
    p = f.predict()
    f.update(Y_NEW, update_params=False)
    return p, f.predict(), f.cutoff, f.forecasters_.iloc[0, 0]._y


@case("update(update_params=False)")
def _():
    f = NaiveForecaster().fit(Y_TR, fh=[1, 2])
    f.update(Y_NEW, update_params=False)
    return f.predict(), f.cutoff


@case("update_predict (fallback method) after trusted fit")
def _():
    from sktime.split import ExpandingWindowSplitter

    f = NaiveForecaster().fit(Y_TR, fh=[1])
    cv = ExpandingWindowSplitter(fh=[1], initial_window=35, step_length=2)
    return (f.update_predict(Y, cv=cv),)


@case("predict_residuals")
def _():
    f = PolynomialTrendForecaster().fit(Y, fh=[1])
    return (f.predict_residuals(),)


@case("predict_quantiles / predict_var")
def _():
    f = NaiveForecaster().fit(Y, fh=[1, 2])
    return f.predict_quantiles(alpha=[0.1, 0.5]), f.predict_var()


@case("Hierarchical PeriodIndex")
def _():
    y = make_panel(8, 30, n_levels=2, time_kind="period", freq="M")
    f = PolynomialTrendForecaster(prediction_intervals=True).fit(y, fh=[1, 2])
    return f.predict(), f.predict_interval(), f.cutoff


@case("Panel RangeIndex")
def _():
    y = make_panel_fast(5, 30, time_kind="int")
    f = NaiveForecaster().fit(y, fh=[1, 2])
    return f.predict(), f.cutoff


@case("unequal length / irregular series (freq None per slice)")
def _():
    y = Y.drop(index=Y.index[[3, 10, 50]])
    f = NaiveForecaster().fit(y, fh=[1, 2])
    return f.predict(), f.cutoff


@case("backend:parallel=joblib loky n_jobs=2")
def _():
    f = NaiveForecaster().set_config(
        **{"backend:parallel": "loky", "backend:parallel:params": {"n_jobs": 2}}
    )
    f.fit(Y, fh=[1, 2])
    return f.predict(), f.predict_interval()


@case("fitted-state equality (Naive, Hier)")
def _():
    return case_state(lambda: NaiveForecaster(), YH)()


@case("get_fitted_params keys")
def _():
    f = PolynomialTrendForecaster().fit(Y, fh=[1])
    return (sorted(f.get_fitted_params().keys()),)


class _OptOutNaive(NaiveForecaster):
    _tags = {g1.OPT_OUT_TAG: False}


class _OverridePredictNaive(NaiveForecaster):
    def predict(self, fh=None, X=None):
        return super().predict(fh=fh, X=X)


FALLBACK_SEEN = {}


@case("opt-out class tag -> fallback")
def _():
    g1.reset_stats()
    f = _OptOutNaive().fit(Y, fh=[1])
    p = f.predict()
    FALLBACK_SEEN["opt-out class tag"] = dict(g1.STATS)
    return (p,)


@case("opt-out config -> fallback")
def _():
    g1.reset_stats()
    f = NaiveForecaster().set_config(**{g1.OPT_OUT_CONFIG: False}).fit(Y, fh=[1])
    p = f.predict()
    FALLBACK_SEEN["opt-out config"] = dict(g1.STATS)
    return (p,)


@case("class overriding public predict -> fallback")
def _():
    g1.reset_stats()
    f = _OverridePredictNaive().fit(Y, fh=[1])
    p = f.predict()
    FALLBACK_SEEN["override predict"] = dict(g1.STATS)
    return (p,)


@case("in-sample fh, insample-incapable (DirectTabular) -> error")
def _():
    from sktime.forecasting.compose import make_reduction
    from sklearn.linear_model import LinearRegression

    f = make_reduction(LinearRegression(), strategy="direct", window_length=3)
    f.fit(Y, fh=[-1, 1])
    return (f.predict(),)


@case("refit same instance twice (reset)")
def _():
    f = NaiveForecaster().fit(Y, fh=[1])
    f.fit(YH, fh=[2])
    return f.predict(), f.cutoff


def main():
    out = []
    for name, fn in CASES.items():
        s = run(fn, "stock")
        r = {"case": name, "stock": s[0] if s[0] == "ok" else s[1]}
        for mode in ("schema_items", "vectorize_est"):
            t = run(fn, mode)
            ok = same(s, t)
            if ok and s[0] == "ok" and isinstance(s[1], g1.BaseForecaster):
                mism = compare_state(s[1], t[1])
                ok = not mism
                r[f"{mode}_state_mismatch"] = mism
            r[mode] = "SAME" if ok else f"DIFF stock={s[1] if s[0]=='err' else 'ok'} trusted={t[1] if t[0]=='err' else 'ok'}"
        out.append(r)
        print(f"{name:60s} stock={r['stock'][:60]:60s} si={r['schema_items'][:70]} ve={r['vectorize_est'][:30]}",
              flush=True)
    print("fallback stats (last run = vectorize_est mode):", FALLBACK_SEEN)
    p = os.path.join(os.path.dirname(__file__), "..", "results", "g1_invariants.json")
    json.dump(out, open(p, "w"), indent=1, default=str)
    n_same = sum(r["schema_items"] == "SAME" for r in out)
    print(f"{n_same}/{len(out)} cases SAME (schema_items); "
          f"{sum(r['vectorize_est'] == 'SAME' for r in out)}/{len(out)} (vectorize_est)")


if __name__ == "__main__":
    main()
