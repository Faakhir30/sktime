"""a1 verify0: independent behavioural probes (claims a1.4, a1.10, a1.11, a1.12).

Written from scratch (does not import a1_semantic_probes); uses different
forecasters / data shapes where possible.

Run from WT root:
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/a1_core_coupling_verify0_probes.py
"""

import json
import traceback
import warnings

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

from sktime.datatypes import check_is_mtype, convert_to  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402
from sktime.forecasting.trend import PolynomialTrendForecaster  # noqa: E402

OUT = {}


def run(name, fn):
    try:
        OUT[name] = {"status": "ok", "result": fn()}
    except Exception as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)[-1]
        OUT[name] = {
            "status": "exc",
            "exc": f"{type(e).__name__}: {str(e)[:200]}",
            "where": tb.filename.split("sktime/")[-1] + ":" + str(tb.lineno),
        }
    print(name, OUT[name])


def tname(x):
    return type(x).__module__ + "." + type(x).__name__


# ---------------- a1.10: X/y containment check, disjoint dates -------------
idx_y = pd.date_range("2001-01-01", periods=30, freq="D")
idx_x = pd.date_range("2010-01-01", periods=30, freq="D")  # disjoint
y_pd = pd.DataFrame({"y": np.random.default_rng(0).normal(size=30)}, index=idx_y)
X_pd = pd.DataFrame({"x": np.random.default_rng(1).normal(size=30)}, index=idx_x)
y_pl = convert_to(y_pd, "pl.DataFrame")
X_pl = convert_to(X_pd, "pl.DataFrame")
print("pl mtype ok:", check_is_mtype(y_pl, "pl.DataFrame"), check_is_mtype(X_pl, "pl.DataFrame"))


def _xy(y, X):
    # a forecaster with capability:exogenous and X-y-must-have-same-index
    from sktime.forecasting.compose import make_reduction
    from sklearn.linear_model import LinearRegression

    f = make_reduction(LinearRegression(), window_length=3, strategy="recursive")
    tags = {
        "X-y-must-have-same-index": f.get_tag("X-y-must-have-same-index"),
        "capability:exogenous": f.get_tag("capability:exogenous"),
    }
    f.fit(y, X=X, fh=[1])
    return {"fit": "no error", "tags": tags}


run("a1.10_pandas_disjoint", lambda: _xy(y_pd, X_pd))
run("a1.10_polars_disjoint", lambda: _xy(y_pl, X_pl))


# ---------------- a1.11: column named "X" ----------------------------------
def _panel(colname, n=3, T=12):
    mi = pd.MultiIndex.from_product(
        [[f"s{i}" for i in range(n)], pd.date_range("2000-01-01", periods=T, freq="D")],
        names=["inst", "time"],
    )
    return pd.DataFrame({colname: np.arange(n * T, dtype=float)}, index=mi)


def _fit_panel(col, fcls):
    y = _panel(col)
    f = fcls().fit(y, fh=[1, 2])
    return tname(f.predict())


for col in ["x", "X"]:
    run(f"a1.11_panel_col_{col}_naive", lambda c=col: _fit_panel(c, NaiveForecaster))
    run(
        f"a1.11_panel_col_{col}_polytrend",
        lambda c=col: _fit_panel(c, PolynomialTrendForecaster),
    )


# Series (non-panel) with column X -> claim only says Panel/Hierarchical
def _series_X():
    y = pd.DataFrame({"X": np.arange(20.0)}, index=pd.date_range("2000", periods=20))
    return str(NaiveForecaster().fit(y, fh=[1]).cutoff)


run("a1.11_series_col_X", _series_X)


# Panel with inner mtype that is natively panel (no vectorization) - is the bug
# vectorization-only or any Panel forecaster?
def _panel_native_X():
    from sktime.forecasting.dummy import ForecastKnownValues  # noqa: F401

    from sktime.forecasting.base import BaseForecaster

    class PanelNative(BaseForecaster):
        _tags = {
            "y_inner_mtype": "pd-multiindex",
            "X_inner_mtype": "pd-multiindex",
            "requires-fh-in-fit": False,
        }

        def _fit(self, y, X=None, fh=None):
            return self

        def _predict(self, fh, X=None):
            return None

    PanelNative().fit(_panel("X"), fh=[1])
    return "fit ok"


run("a1.11_panel_col_X_native_pd_multiindex_inner", _panel_native_X)


# ---------------- a1.12: update_predict / predict_residuals polars Series ---
ys_pd = pd.DataFrame({"v": np.arange(40.0)}, index=pd.date_range("2000", periods=40, freq="D"))
ys_pl = convert_to(ys_pd, "pl.DataFrame")


def _upd_pred(y):
    from sktime.split import SlidingWindowSplitter

    f = PolynomialTrendForecaster().fit(y[:30] if isinstance(y, pd.DataFrame) else y.head(30), fh=[1, 2])
    cv = SlidingWindowSplitter(fh=[1, 2], window_length=5, step_length=2)
    r = f.update_predict(y.tail(10) if isinstance(y, pl.DataFrame) else y.iloc[30:], cv=cv)
    return tname(r)


def _resid(y):
    f = PolynomialTrendForecaster().fit(y, fh=[1])
    return tname(f.predict_residuals(y))


run("a1.12_update_predict_pandas", lambda: _upd_pred(ys_pd))
run("a1.12_update_predict_polars", lambda: _upd_pred(ys_pl))
run("a1.12_predict_residuals_pandas", lambda: _resid(ys_pd))
run("a1.12_predict_residuals_polars", lambda: _resid(ys_pl))


# ---------------- a1.4: output types for polars Panel input ------------------
def _out_types():
    y = _panel("val", n=4, T=20)
    y_pl = convert_to(y, "polars_panel")
    f = NaiveForecaster(strategy="mean").fit(y_pl, fh=[1, 2])
    res = {"input": tname(y_pl)}
    res["predict"] = tname(f.predict())
    for m in ["predict_interval", "predict_quantiles", "predict_var"]:
        try:
            res[m] = tname(getattr(f, m)())
        except Exception as e:  # noqa: BLE001
            res[m] = "EXC " + type(e).__name__
    res["cutoff"] = tname(f.cutoff)
    res["forecasters_"] = tname(f.forecasters_)
    res["fitted_keys_sample"] = [k for k in f.get_fitted_params() if "loc" in k][:2]
    return res


run("a1.4_polars_panel_output_types", _out_types)

json.dump(OUT, open("experiments/container_independence/results/a1_core_coupling_verify0_probes.json", "w"), indent=1, default=str)
