"""x9 (A2): does sktime accept pandas frames backed by pyarrow (ArrowDtype)?

"pandas API, Arrow memory" only pays off if the frame that
``polars.to_pandas(use_pyarrow_extension_array=True)`` produces can be passed
through sktime unchanged. This script runs a matrix of

  inputs  : the same numbers in different pandas memory layouts
            (numpy baseline, Arrow values, Arrow values + Arrow index, Arrow
            int index, nulls, and the literal output of the polars->Arrow path),
            for Series and Panel (pd-multiindex) shapes
  x tests : check_is_scitype, forecasters (Naive last/mean/drift, intervals,
            PolynomialTrend, Theta, ExponentialSmoothing, reduction),
            TransformedTargetForecaster pipelines (Differencer, Detrender),
            Differencer transform + inverse, update, and VectorizedDF
            (prepare_VectorizedDF -> items -> reconstruct) for panels

and records, per cell: OK / ERROR (exception + innermost sktime frame + innermost
frame overall), output value/index dtypes, whether the output is still Arrow, and
whether values+index match the numpy baseline (silent-wrong-result detection).

statsmodels is not installed in the shared envs; it is made importable from a
separate --target dir (no deps) by prepending it to PYTHONPATH::

    SM=$SCRATCH/venvs/x9_smtarget
    PYTHONPATH=.:experiments/container_independence:$SM $EXP_PY -W ignore \
        experiments/container_independence/scripts/x9_arrow_compat.py
    PYTHONPATH=.:experiments/container_independence:$SM $EXP_PY3 -W ignore \
        experiments/container_independence/scripts/x9_arrow_compat.py
"""

import json
import os
import sys
import traceback
import warnings

warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402

from common import env_info, make_panel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

try:
    import statsmodels  # noqa: F401

    HAS_SM = True
except ImportError:
    HAS_SM = False


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def exc_info(e):
    """Short exception description: type, message, innermost sktime + any frame."""
    tb = traceback.extract_tb(e.__traceback__)
    sk = [f for f in tb if "/sktime/" in f.filename and "experiments" not in f.filename]
    last_sk = sk[-1] if sk else None
    last = tb[-1] if tb else None

    def fmt(fr):
        if fr is None:
            return None
        fn = fr.filename
        if fn.startswith(WT):
            fn = os.path.relpath(fn, WT)
        else:
            # site-packages/<lib>/... -> <lib>/...
            if "site-packages/" in fn:
                fn = fn.split("site-packages/")[1]
        return f"{fn}:{fr.lineno} ({fr.name}) `{(fr.line or '').strip()[:120]}`"

    return {
        "error": f"{type(e).__name__}: {str(e).splitlines()[0][:220] if str(e) else ''}",
        "at_sktime": fmt(last_sk),
        "at_innermost": fmt(last),
    }


def dtype_str(obj):
    if isinstance(obj, pd.DataFrame):
        return sorted({str(d) for d in obj.dtypes})
    if isinstance(obj, pd.Series):
        return [str(obj.dtype)]
    return [type(obj).__name__]


def index_dtype_str(idx):
    if isinstance(idx, pd.MultiIndex):
        return [str(lv.dtype) for lv in idx.levels]
    return [type(idx).__name__ + ":" + str(idx.dtype)]


def is_arrow_values(obj):
    if isinstance(obj, pd.DataFrame):
        return all(isinstance(d, pd.ArrowDtype) for d in obj.dtypes)
    if isinstance(obj, pd.Series):
        return isinstance(obj.dtype, pd.ArrowDtype)
    return False


def norm_values(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return np.asarray(obj.astype("float64").to_numpy(dtype="float64", na_value=np.nan))
    return np.asarray(obj, dtype="float64")


def norm_index(idx):
    out = []
    for x in idx.to_list():
        if isinstance(x, tuple):
            out.append(tuple(str(pd.Timestamp(v)) if _is_ts(v) else str(v) for v in x))
        else:
            out.append(str(pd.Timestamp(x)) if _is_ts(x) else str(x))
    return out


def _is_ts(v):
    return isinstance(v, (pd.Timestamp, np.datetime64)) or (
        hasattr(v, "year") and hasattr(v, "month") and hasattr(v, "day")
    )


def compare(out, base):
    """Return dict with value / index match flags vs numpy baseline output."""
    if base is None or not isinstance(out, (pd.DataFrame, pd.Series)):
        return {}
    try:
        v1, v0 = norm_values(out), norm_values(base)
        vmatch = v1.shape == v0.shape and bool(
            np.allclose(v1, v0, equal_nan=True, rtol=1e-8, atol=1e-10)
        )
    except Exception as e:  # noqa: BLE001
        vmatch = f"cmp-error {type(e).__name__}: {e}"
    try:
        imatch = norm_index(out.index) == norm_index(base.index)
    except Exception as e:  # noqa: BLE001
        imatch = f"cmp-error {type(e).__name__}: {e}"
    return {"values_match_np": vmatch, "index_match_np": imatch}


def describe(out):
    if isinstance(out, (pd.DataFrame, pd.Series)):
        return {
            "out_type": type(out).__name__,
            "out_dtypes": dtype_str(out),
            "out_index": index_dtype_str(out.index),
            "out_is_arrow": is_arrow_values(out),
        }
    return {"out_type": type(out).__name__}


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------
T = 60
ARROW_F = pd.ArrowDtype(pa.float64())
ARROW_TS = pd.ArrowDtype(pa.timestamp("ns"))
ARROW_I = pd.ArrowDtype(pa.int64())
ARROW_S = pd.ArrowDtype(pa.large_string())


def series_inputs():
    rng = np.random.default_rng(0)
    v = rng.normal(size=T).cumsum() + 50.0
    t = pd.date_range("2000-01-01", periods=T, freq="D")
    base = pd.Series(v, index=t, name="y")
    out = {}
    out["np"] = base
    out["arrow_vals"] = base.astype(ARROW_F)
    s = base.astype(ARROW_F)
    s.index = pd.Index(t.astype(ARROW_TS), name=None)
    out["arrow_vals+arrow_ts_idx"] = s
    s = base.copy()
    s.index = pd.Index(t.astype(ARROW_TS))
    out["np_vals+arrow_ts_idx"] = s
    out["np_int_idx"] = pd.Series(v, index=pd.Index(np.arange(T)), name="y")
    s = pd.Series(v, index=pd.Index(np.arange(T), dtype=ARROW_I), name="y", dtype=ARROW_F)
    out["arrow_vals+arrow_int_idx"] = s
    # literal output of the polars -> pandas(Arrow) route, sktime __index__ cols
    import polars as pl

    pl_df = pl.DataFrame({"__index__time": t.values, "y": v})
    df = pl_df.to_pandas(use_pyarrow_extension_array=True).set_index("__index__time")
    df.index.name = "time"
    out["polars_to_pandas_arrow(DataFrame)"] = df
    # hybrid route: index columns -> numpy-backed pandas index, values -> Arrow
    idx = pd.DatetimeIndex(pl_df["__index__time"].to_numpy(), name="time")
    vals = pl_df.select("y").to_pandas(use_pyarrow_extension_array=True)
    vals.index = idx
    out["hybrid:arrow_vals+np_idx(DataFrame)"] = vals
    out["np(DataFrame)"] = base.to_frame()
    base_of = {
        "np": "np",
        "np_int_idx": "np_int_idx",
        "arrow_vals+arrow_int_idx": "np_int_idx",
        "np(DataFrame)": "np(DataFrame)",
        "polars_to_pandas_arrow(DataFrame)": "np(DataFrame)",
        "hybrid:arrow_vals+np_idx(DataFrame)": "np(DataFrame)",
    }
    return {k: (o, base_of.get(k, "np")) for k, o in out.items()}


def series_inputs_nulls():
    """Missing value in the middle: NaN (numpy) vs pd.NA (Arrow)."""
    rng = np.random.default_rng(1)
    v = rng.normal(size=T).cumsum() + 50.0
    v[T - 5] = np.nan
    t = pd.date_range("2000-01-01", periods=T, freq="D")
    base = pd.Series(v, index=t, name="y")
    arr = pd.Series(pd.array(v, dtype=ARROW_F), index=t, name="y")
    # make sure it is a real Arrow null, not a NaN value
    arr = arr.where(~np.isnan(v), pd.NA)
    return {"np_nan": (base, "np_nan"), "arrow_null": (arr, "np_nan")}


def panel_inputs():
    base = make_panel(n_series=4, n_time=40, n_levels=1, seed=2) + 50.0
    out = {"P_np": base}
    out["P_arrow_vals"] = base.astype(ARROW_F)
    flat = base.reset_index()
    h0 = flat["h0"].astype(ARROW_S)
    tt = flat["time"].astype(ARROW_TS)
    mi = pd.MultiIndex.from_arrays([h0, tt], names=["h0", "time"])
    out["P_arrow_vals+arrow_levels"] = pd.DataFrame(
        {"c0": flat["c0"].astype(ARROW_F).values}, index=mi
    )
    import polars as pl  # noqa: F401

    from sktime.datatypes import convert_to

    pl_df = convert_to(base, "polars_panel")
    df = pl_df.to_pandas(use_pyarrow_extension_array=True).set_index(
        ["__index__h0", "__index__time"]
    )
    df.index.names = ["h0", "time"]
    out["P_polars_to_pandas_arrow"] = df
    # hybrid route: index columns -> numpy-backed MultiIndex, values -> Arrow
    idx_cols = ["__index__h0", "__index__time"]
    mi = pd.MultiIndex.from_frame(pl_df.select(idx_cols).to_pandas(), names=["h0", "time"])
    vals = pl_df.drop(idx_cols).to_pandas(use_pyarrow_extension_array=True)
    vals.index = mi
    out["P_hybrid:arrow_vals+np_levels"] = vals
    return {k: (o, "P_np") for k, o in out.items()}


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
FH = [1, 2, 3]


def t_check(y):
    from sktime.datatypes import check_is_scitype

    ok, msg, meta = check_is_scitype(
        y, scitype=["Series", "Panel", "Hierarchical"], return_metadata=True
    )
    rec = {"valid": bool(ok)}
    if ok:
        rec["mtype"] = meta.get("mtype")
        for k in ["is_equally_spaced", "has_nans", "is_univariate", "feature_kind"]:
            if k in meta:
                rec[k] = str(meta[k])
    else:
        if isinstance(msg, dict):
            keep = ["pd.Series", "pd.DataFrame", "pd-multiindex"]
            rec["msg"] = "; ".join(f"[{k}] {msg[k]}" for k in keep if k in msg)[:600]
        else:
            rec["msg"] = str(msg)[:600]
    return rec, None


def _fc(est):
    def run(y):
        f = est.clone().fit(y, fh=FH)
        return {}, f.predict()

    return run


def t_naive_interval(y):
    from sktime.forecasting.naive import NaiveForecaster

    f = NaiveForecaster(strategy="drift").fit(y, fh=FH)
    return {}, f.predict_interval(coverage=0.9)


def t_update(y):
    from sktime.forecasting.naive import NaiveForecaster

    n = len(y)
    if isinstance(y.index, pd.MultiIndex):
        tl = y.index.get_level_values(-1)
        cut = sorted(tl.unique())[-10]
        y1, y2 = y[tl < cut], y[tl >= cut]
    else:
        y1, y2 = y.iloc[: n - 10], y.iloc[n - 10 :]
    f = NaiveForecaster(strategy="last").fit(y1, fh=FH)
    f.update(y2)
    return {}, f.predict()


def t_differencer(y):
    from sktime.transformations.difference import Differencer

    tr = Differencer(lags=1)
    yt = tr.fit_transform(y)
    yi = tr.inverse_transform(yt)
    rec = {"transform": describe(yt)}
    # inverse transform should give back y
    rec["inverse_roundtrip_match"] = compare(yi, y)
    rec["inverse_out"] = describe(yi)
    return rec, yt


def t_detrender(y):
    from sktime.transformations.detrend import Detrender

    yt = Detrender().fit_transform(y)
    return {}, yt


def t_vectorizeddf(y):
    """prepare_VectorizedDF -> items -> reconstruct, with the identity op."""
    from sktime.datatypes._vectorize import prepare_VectorizedDF

    vec, X_mi = prepare_VectorizedDF(y, iterate_as="Series", is_scitype=None)
    rec = {
        "X_mi_dtypes": dtype_str(X_mi),
        "X_mi_index": index_dtype_str(X_mi.index),
        "schema_freq": str(vec.freq),
        "n_slices": int(len(vec)),
    }
    sl = vec.as_list(X=X_mi)
    rec["slice0_dtypes"] = dtype_str(sl[0])
    rec["slice0_index"] = index_dtype_str(sl[0].index)
    rec["slice0_freq"] = str(getattr(sl[0].index, "freq", "n/a"))
    out = vec.reconstruct(sl, convert_back=True)
    return rec, out


def build_tests(scitype):
    from sktime.forecasting.compose import (
        TransformedTargetForecaster,
        make_reduction,
    )
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.trend import PolynomialTrendForecaster
    from sktime.transformations.detrend import Detrender
    from sktime.transformations.difference import Differencer

    tests = {
        "check_is_scitype": t_check,
        "Naive(last)": _fc(NaiveForecaster(strategy="last")),
        "Naive(mean,w=7)": _fc(NaiveForecaster(strategy="mean", window_length=7)),
        "Naive(drift)": _fc(NaiveForecaster(strategy="drift")),
        "Naive(drift).predict_interval": t_naive_interval,
        "PolynomialTrend(deg=2)": _fc(PolynomialTrendForecaster(degree=2)),
        "TTF[Differencer,Naive(mean)]": _fc(
            TransformedTargetForecaster(
                [("d", Differencer()), ("f", NaiveForecaster(strategy="mean"))]
            )
        ),
        "TTF[Detrender,Naive(last)]": _fc(
            TransformedTargetForecaster(
                [("d", Detrender()), ("f", NaiveForecaster(strategy="last"))]
            )
        ),
        "Differencer.fit_transform(+inverse)": t_differencer,
        "Detrender.fit_transform": t_detrender,
        "Naive(last).update": t_update,
    }
    try:
        from sklearn.linear_model import LinearRegression

        tests["make_reduction(LinReg,w=5)"] = _fc(
            make_reduction(LinearRegression(), window_length=5, strategy="recursive")
        )
    except Exception:  # noqa: BLE001
        pass
    if HAS_SM:
        from sktime.forecasting.exp_smoothing import ExponentialSmoothing
        from sktime.forecasting.theta import ThetaForecaster

        tests["Theta(sp=1)"] = _fc(ThetaForecaster(sp=1))
        tests["ExponentialSmoothing(trend=add)"] = _fc(
            ExponentialSmoothing(trend="add")
        )
    if scitype == "Panel":
        tests["VectorizedDF(prepare/items/reconstruct)"] = t_vectorizeddf
    return tests


def run_matrix(inputs, tests):
    res = {}
    base_out = {}  # (baseline input key, test key) -> output
    bases = []
    for k, (_, b) in inputs.items():
        if b not in bases:
            bases.append(b)
    order = bases + [k for k in inputs if k not in bases]
    for in_key in order:
        y, baseline_key = inputs[in_key]
        res[in_key] = {
            "input_dtypes": dtype_str(y),
            "input_index": index_dtype_str(y.index),
            "baseline": baseline_key,
        }
        for t_key, fn in tests.items():
            try:
                rec, out = fn(y.copy())
                rec = dict(rec)
                rec["status"] = "OK"
                if out is not None:
                    rec.update(describe(out))
                    if in_key == baseline_key:
                        base_out[(in_key, t_key)] = out
                    else:
                        rec.update(compare(out, base_out.get((baseline_key, t_key))))
            except Exception as e:  # noqa: BLE001
                rec = {"status": "ERROR", **exc_info(e)}
            res[in_key][t_key] = rec
    return res


def summarize(res, baseline_key):
    lines = []
    for in_key, cells in res.items():
        lines.append(f"\n### input = {in_key}   (baseline: {cells.get('baseline')})")
        lines.append(f"  input dtypes={cells['input_dtypes']} index={cells['input_index']}")
        for t_key, rec in cells.items():
            if not isinstance(rec, dict) or "status" not in rec:
                continue
            if rec["status"] == "OK":
                bits = []
                if "valid" in rec:
                    bits.append(f"valid={rec['valid']}")
                    if not rec["valid"]:
                        bits.append(f"msg={rec.get('msg', '')[:200]}")
                    else:
                        bits.append(f"mtype={rec.get('mtype')}")
                if "out_dtypes" in rec:
                    bits.append(f"out={rec['out_dtypes']} idx={rec['out_index']}")
                    bits.append(f"arrow_kept={rec['out_is_arrow']}")
                if "values_match_np" in rec:
                    bits.append(
                        f"match(v={rec['values_match_np']},i={rec['index_match_np']})"
                    )
                if "inverse_roundtrip_match" in rec:
                    bits.append(f"inv_rt={rec['inverse_roundtrip_match']}")
                if "slice0_freq" in rec:
                    bits.append(
                        f"X_mi={rec['X_mi_dtypes']}/{rec['X_mi_index']} "
                        f"slice_idx={rec['slice0_index']} slice_freq={rec['slice0_freq']}"
                        f" schema_freq={rec['schema_freq']}"
                    )
                lines.append(f"  [OK ] {t_key}: " + "; ".join(bits))
            else:
                lines.append(
                    f"  [ERR] {t_key}: {rec['error']}\n"
                    f"        sktime: {rec['at_sktime']}\n"
                    f"        inner : {rec['at_innermost']}"
                )
    return "\n".join(lines)


def main():
    out = {"env": env_info(), "python": sys.version.split()[0], "statsmodels": HAS_SM}
    if HAS_SM:
        import statsmodels

        out["env"]["statsmodels"] = statsmodels.__version__
    import sklearn

    out["env"]["sklearn"] = sklearn.__version__

    s_tests = build_tests("Series")
    p_tests = build_tests("Panel")

    out["series"] = run_matrix(series_inputs(), s_tests)
    out["series_nulls"] = run_matrix(series_inputs_nulls(), s_tests)
    out["panel"] = run_matrix(panel_inputs(), p_tests)

    tag = "pd3" if pd.__version__.startswith("3") else "pd2"
    txt = [f"env: {out['env']}"]
    for k in ["series", "series_nulls", "panel"]:
        txt.append(f"\n================ {k} ================")
        txt.append(summarize(out[k], None))
    txt = "\n".join(txt)
    print(txt)
    with open(os.path.join(RES, f"x9_arrow_compat_{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    with open(os.path.join(RES, f"x9_arrow_compat_{tag}.txt"), "w") as fh:
        fh.write(txt)

    # compact count table
    counts = {}
    for k in ["series", "series_nulls", "panel"]:
        for in_key, cells in out[k].items():
            n_ok = sum(
                1 for r in cells.values() if isinstance(r, dict) and r.get("status") == "OK"
            )
            n_err = sum(
                1
                for r in cells.values()
                if isinstance(r, dict) and r.get("status") == "ERROR"
            )
            n_mismatch = sum(
                1
                for r in cells.values()
                if isinstance(r, dict)
                and r.get("status") == "OK"
                and (r.get("values_match_np") is False or r.get("index_match_np") is False)
            )
            n_arrow = sum(
                1
                for r in cells.values()
                if isinstance(r, dict) and r.get("out_is_arrow") is True
            )
            n_out = sum(
                1 for r in cells.values() if isinstance(r, dict) and "out_is_arrow" in r
            )
            counts[f"{k}/{in_key}"] = {
                "ok": n_ok,
                "error": n_err,
                "ok_but_mismatch": n_mismatch,
                "outputs_still_arrow": f"{n_arrow}/{n_out}",
            }
    print("\n==== counts ====")
    for k, v in counts.items():
        print(f"  {k:55s} {v}")
    out["counts"] = counts
    with open(os.path.join(RES, f"x9_arrow_compat_{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
