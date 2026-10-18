"""a1: behavioural probes of pandas coupling at the BaseForecaster / VectorizedDF boundary.

Each probe runs a public-API call with a pandas input and with the equivalent
polars input (sktime polars mtypes), and records what happens. The point is to
find places where the boundary layer's pandas assumptions change BEHAVIOUR (not
just speed) for non-pandas inputs, and to record which outputs are pandas-typed
regardless of input type (= pandas as a value type in the public API contract).

Run (from WT root):
    PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
        $EXP_PY -W ignore experiments/container_independence/scripts/a1_semantic_probes.py
Output: results/a1_semantic_probes.json
"""

import json
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from a1_toy_forecasters import LastValueX  # noqa: E402
from common import env_info, make_panel  # noqa: E402

from sktime.datatypes import convert_to, prepare_VectorizedDF  # noqa: E402
from sktime.datatypes._utilities import update_data  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402
from sktime.split import ExpandingWindowSplitter  # noqa: E402

RES = {}


def probe(name):
    def deco(fn):
        try:
            out = fn()
            RES[name] = {"status": "ok", **(out or {})}
        except Exception as e:  # noqa: BLE001
            RES[name] = {
                "status": "exception",
                "exc_type": type(e).__name__,
                "exc_msg": str(e)[:300],
                "where": traceback.extract_tb(e.__traceback__)[-1].filename.split(
                    "sktime/"
                )[-1]
                + ":"
                + str(traceback.extract_tb(e.__traceback__)[-1].lineno),
            }
        print(f"[{name}] {RES[name]}")
        return fn

    return deco


def _series_pd():
    return pd.DataFrame(
        {"c0": np.arange(20.0)}, index=pd.date_range("2000-01-01", periods=20, freq="D")
    )


def _tn(x):
    return f"{type(x).__module__}.{type(x).__name__}"


# 1. X/y time-index containment check (X-y-must-have-same-index) ----------------
def _xy_check(conv):
    y = _series_pd()
    X = pd.DataFrame(
        {"x0": np.arange(20.0)}, index=pd.date_range("2001-01-01", periods=20, freq="D")
    )  # disjoint dates: must be rejected
    try:
        LastValueX().fit(conv(y), X=conv(X), fh=[1])
        return {"mismatch_detected": False}
    except ValueError as e:
        return {"mismatch_detected": True, "msg": str(e)[:80]}


@probe("xy_index_check_pandas")
def _():
    return _xy_check(lambda d: d)


@probe("xy_index_check_polars")
def _():
    return _xy_check(lambda d: convert_to(d, "pl.DataFrame"))


# 2. output types for polars Panel input --------------------------------------
@probe("output_types_polars_panel_input")
def _():
    y = make_panel(5, 30, n_levels=1)
    ypl = convert_to(y, "polars_panel")
    f = NaiveForecaster().fit(ypl, fh=[1, 2])
    out = {
        "predict": _tn(f.predict()),
        "predict_interval": _tn(f.predict_interval(coverage=0.9)),
        "predict_quantiles": _tn(f.predict_quantiles(alpha=[0.1, 0.9])),
        "predict_var": _tn(f.predict_var()),
        "cutoff": _tn(f.cutoff),
        "forecasters_": _tn(f.forecasters_),
        "fh": _tn(f.fh),
        "fitted_param_keys_sample": [
            k for k in f.get_fitted_params() if k.startswith("forecasters.loc")
        ][:2],
    }
    return out


@probe("output_types_pandas_hier_input")
def _():
    y = make_panel(4, 30, n_levels=2)
    f = NaiveForecaster().fit(y, fh=[1, 2])
    pi = f.predict_interval(coverage=0.9)
    return {
        "predict_interval_columns": [str(c) for c in pi.columns],
        "pi_columns_type": _tn(pi.columns),
        "fitted_param_keys_sample": [
            k for k in f.get_fitted_params() if k.startswith("forecasters.loc")
        ][:2],
        "forecasters_index_type": _tn(f.forecasters_.index),
    }


# 3. predict_residuals with polars y -----------------------------------------
@probe("predict_residuals_pandas_series")
def _():
    y = _series_pd()
    f = NaiveForecaster().fit(y, fh=[1])
    r = f.predict_residuals(y)
    return {"type": _tn(r), "len": len(r)}


@probe("predict_residuals_polars_series")
def _():
    y = convert_to(_series_pd(), "pl.DataFrame")
    f = NaiveForecaster().fit(y, fh=[1])
    r = f.predict_residuals(y)
    return {"type": _tn(r), "len": len(r)}


# 4. update_predict (moving cutoff) with polars y ------------------------------
@probe("update_predict_pandas_series")
def _():
    y = _series_pd()
    f = NaiveForecaster().fit(y.iloc[:10], fh=[1])
    cv = ExpandingWindowSplitter(initial_window=1, fh=[1], step_length=1)
    r = f.update_predict(y.iloc[10:], cv=cv)
    return {"type": _tn(r), "shape": list(r.shape)}


@probe("update_predict_polars_series")
def _():
    y = _series_pd()
    f = NaiveForecaster().fit(convert_to(y.iloc[:10], "pl.DataFrame"), fh=[1])
    cv = ExpandingWindowSplitter(initial_window=1, fh=[1], step_length=1)
    r = f.update_predict(convert_to(y.iloc[10:], "pl.DataFrame"), cv=cv)
    return {"type": _tn(r)}


@probe("update_predict_freq_set_vs_none")
def _():
    out = {}
    for label, freq in [("freq=D", "D"), ("freq=None", None)]:
        y = _series_pd()
        if freq is None:
            y.index = pd.DatetimeIndex(list(y.index))  # same stamps, freq dropped
        assert (y.index == _series_pd().index).all()
        try:
            f = NaiveForecaster().fit(y.iloc[:10], fh=[1])
            cv = ExpandingWindowSplitter(initial_window=1, fh=[1], step_length=1)
            r = f.update_predict(y.iloc[10:], cv=cv)
            out[label] = f"ok shape={list(r.shape)}"
        except Exception as e:  # noqa: BLE001
            out[label] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


# duck-typing hazard: get_cutoff does ``if hasattr(obj, "X"): obj = obj.X``
# (legacy VectorizedDF.X); pandas resolves ``.X`` to a COLUMN named "X".
def _fit_col(name, conv=lambda d: d):
    y = make_panel(3, 20, n_levels=1).rename(columns={"c0": name})
    try:
        f = NaiveForecaster().fit(conv(y), fh=[1])
        return f"ok cutoff={list(f.cutoff)}"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:70]}"


@probe("panel_column_named_X_pandas_vs_polars")
def _():
    return {
        "pandas col 'x'": _fit_col("x"),
        "pandas col 'X'": _fit_col("X"),
        "polars col 'X'": _fit_col("X", lambda d: convert_to(d, "polars_panel")),
    }


# 5. polars LazyFrame as y ----------------------------------------------------
@probe("lazy_polars_panel_y")
def _():
    y = convert_to(make_panel(5, 30, n_levels=1), "polars_panel").lazy()
    NaiveForecaster().fit(y, fh=[1])
    return {}


# 6. stale VectorizedDF attribute access in helpers (dead after stateless) -----
@probe("update_data_with_VectorizedDF")
def _():
    y = make_panel(3, 10, n_levels=1)
    vec, _ = prepare_VectorizedDF(y, iterate_as="Series", is_scitype="Panel")
    update_data(vec, vec)
    return {}


@probe("VectorizedDF_has_X_attr")
def _():
    y = make_panel(3, 10, n_levels=1)
    vec, _ = prepare_VectorizedDF(y, iterate_as="Series", is_scitype="Panel")
    return {
        "has_X": hasattr(vec, "X"),
        "has_X_multiindex": hasattr(vec, "X_multiindex"),
        "attrs": sorted(k for k in vars(vec)),
    }


# 7. what polars input actually costs: conversions on the vectorized path -----
N_INST = 5


@probe("conversion_count_polars_vs_pandas_panel")
def _():
    import sktime.datatypes._convert as cv

    calls = []
    orig = cv.convert

    def wrapped(obj, from_type, to_type, *a, **k):
        if obj is not None:
            resolved = cv._get_first_mtype_of_same_scitype(from_type, to_type)
            calls.append((from_type, resolved))
        return orig(obj, from_type, to_type, *a, **k)

    cv.convert = wrapped
    try:
        out = {}
        for name, conv in [
            ("pandas", lambda d: d),
            ("polars", lambda d: convert_to(d, "polars_panel")),
        ]:
            y = conv(make_panel(N_INST, 30, n_levels=1))
            calls.clear()
            f = NaiveForecaster().fit(y, fh=[1, 2])
            f.predict()
            nontrivial = [c for c in calls if c[0] != c[1]]
            from collections import Counter

            out[name] = {
                "n_convert_calls": len(calls),
                "n_nontrivial_conversions": len(nontrivial),
                "nontrivial_counts": {
                    f"{a}->{b}": n for (a, b), n in Counter(nontrivial).items()
                },
            }
        return out
    finally:
        cv.convert = orig


if __name__ == "__main__":
    RES["_env"] = env_info()
    with open(
        "experiments/container_independence/results/a1_semantic_probes.json", "w"
    ) as f:
        json.dump(RES, f, indent=1, default=str)
