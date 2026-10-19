"""g5_api_surface (critic G5): API surface beyond y-only fit+predict.

Matrix: {approach} x {operation, scitype} x {pandas, polars}.

approaches
  stock   : unmodified sktime (WT checkout)
  x4      : protos/x4_adapter via x4_instrument.patched_base("native", cutoff=True)
  x4nc    : same, without the cutoff bridge (isolates the reverse_order bug)
  x4fix   : x4 + a cutoff bridge that honours reverse_order (falls back to stock
            get_cutoff when reverse_order=True)
  x6b     : x6 hub wired *at the boundary only* (hub is not wired into
            BaseForecaster): every data argument goes native -> hub -> pandas,
            stock sktime runs on pandas, every frame output goes pandas -> hub ->
            native (input's backend). Pandas input = hub round trip in and out.

Reference for equality: stock + pandas. Outputs normalised to pandas via
sktime convert_to before comparing.

Run from WT:
  PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
    $G5_PY -W ignore experiments/container_independence/scripts/g5_api_surface.py
"""

import contextlib
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression

from sktime.datatypes import convert_to
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import TransformedTargetForecaster, make_reduction
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.reconcile import ReconcilerForecaster
from sktime.split import ExpandingWindowSplitter
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.series.detrend import Deseasonalizer

OUT_JSON = "experiments/container_independence/results/g5_api_surface.json"

# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
T, NTR, NNEW = 60, 45, 6
RNG = np.random.default_rng(0)
TIDX = pd.date_range("2000-01-01", periods=T + 10, freq="D", name="time")


def _series_frame(cols, n, seed, idx=TIDX):
    rng = np.random.default_rng(seed)
    vals = rng.normal(size=(len(idx), len(cols))).cumsum(axis=0) + 50
    return pd.DataFrame(vals, index=idx, columns=cols)


def _panel(cols, seed, n_inst=4, idx=TIDX):
    parts = {f"s{i}": _series_frame(cols, len(idx), seed + i, idx) for i in range(n_inst)}
    df = pd.concat(parts, names=["h0"])
    return df


def make_data():
    D = {}
    # Series
    ys = _series_frame(["y"], T + 10, 1)
    Xs = _series_frame(["x1", "x2"], T + 10, 2)
    # y depends on X so reduction w/ X is non-trivial
    ys["y"] = ys["y"] + 0.5 * Xs["x1"].values
    D["Series"] = (ys, Xs)
    yp = _panel(["y"], 10)
    Xp = _panel(["x1", "x2"], 20)
    yp["y"] = yp["y"] + 0.5 * Xp["x1"].values
    D["Panel"] = (yp, Xp)
    return D


def sl(obj, a, b):
    """Time slice [a:b) on the time (last) index level."""
    if isinstance(obj.index, pd.MultiIndex):
        inst = obj.index.droplevel(-1)
        pos = obj.groupby(level=list(range(obj.index.nlevels - 1))).cumcount().values
        return obj[(pos >= a) & (pos < b)]
    return obj.iloc[a:b]


PL_MTYPE = {"Series": "pl.DataFrame", "Panel": "polars_panel",
            "Hierarchical": "polars_hierarchical"}
PD_MTYPES = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]


def to_backend(obj, backend, scitype):
    if obj is None or backend == "pandas":
        return obj
    return convert_to(obj, PL_MTYPE[scitype])


def to_pd(obj):
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        return convert_to(obj, PD_MTYPES)
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    return obj


# ----------------------------------------------------------------------------
# approaches
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def ctx_for(approach):
    if approach in ("stock", "x6b"):
        yield
        return
    from x4_instrument import patched_base

    if approach == "x4":
        with patched_base("native", cutoff=True):
            yield
    elif approach == "x4nc":
        with patched_base("native", cutoff=False):
            yield
    elif approach == "x4fix":
        import sktime.forecasting.base._base as B
        from x4_adapter import adapter_get_cutoff

        from sktime.datatypes import get_cutoff as stock_get_cutoff

        def fixed(obj, cutoff=0, return_index=False, reverse_order=False, **kw):
            if reverse_order:
                return stock_get_cutoff(obj, cutoff, return_index=return_index,
                                        reverse_order=True, **kw)
            return adapter_get_cutoff(obj, cutoff, return_index=return_index, **kw)

        with patched_base("native", cutoff=False):
            orig = B.get_cutoff
            B.get_cutoff = fixed
            try:
                yield
            finally:
                B.get_cutoff = orig
    else:
        raise ValueError(approach)


# x6 boundary wiring ---------------------------------------------------------
def hub_in(obj):
    """native -> hub -> pandas (sktime layout)."""
    from x6_hub import from_pandas, from_polars, to_pandas

    if obj is None:
        return None
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        hub = from_polars(obj)
        pdo = to_pandas(hub)
        return pdo, hub.meta
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        hub = from_pandas(obj)
        return to_pandas(hub), hub.meta
    return obj, None


def hub_out(obj, like_meta):
    """pandas output -> hub -> native of the input backend."""
    from x6_hub import from_pandas, to_pandas, to_polars

    if not isinstance(obj, (pd.DataFrame, pd.Series)) or like_meta is None:
        return obj
    hub = from_pandas(obj)
    if like_meta.get("source") == "polars":
        m = dict(like_meta)
        # output columns/rows differ from input: rebuild schema from like_meta
        schema = dict(m["schema"])
        keep = list(m["key_cols"]) + [m["time_col"]] + list(hub.columns)
        m["schema"] = {k: schema.get(k) for k in keep if schema.get(k) is not None}
        m["columns_order"] = keep
        m["perm"] = None
        m["null_cols"] = []
        m["lazy"] = False
        hub.meta = m
        hub.key_names = list(m["key_cols"])
        return to_polars(hub)
    return to_pandas(hub)


# ----------------------------------------------------------------------------
# operations: each takes (data dict for one backend) and returns dict of outputs
# ----------------------------------------------------------------------------
FH = [1, 2, 3]


def red():
    return make_reduction(LinearRegression(), window_length=5, strategy="recursive",
                          pooling="local")


def naive(strategy="last"):
    return NaiveForecaster(strategy=strategy)


def op_fit_naive(d):
    f = naive().fit(d["y_tr"], fh=FH)
    return {"pred": f.predict(), "cutoff": f.cutoff}


def op_fit_X(d):
    f = red().fit(d["y_tr"], X=d["X_tr"], fh=FH)
    return {"pred": f.predict(X=d["X_fut"]), "cutoff": f.cutoff}


def op_fit_X_disjoint(d):
    try:
        red().fit(d["y_tr"], X=d["X_disj"], fh=FH)
    except ValueError as e:
        return {"raised": f"ValueError: {str(e)[:80]}"}
    return {"raised": "NO ERROR (silently accepted disjoint X)"}


def op_update(d):
    f = red().fit(d["y_tr"], X=d["X_tr"], fh=FH)
    f.update(d["y_new"], X=d["X_new"])
    return {"pred": f.predict(X=d["X_fut2"]), "cutoff": f.cutoff}


def op_update_predict(d):
    f = naive().fit(d["y_tr"], fh=[1, 2])
    out = f.update_predict(d["y_test"])
    return {"pred": out, "cutoff": f.cutoff}


def op_update_predict_X(d):
    f = red().fit(d["y_tr"], X=d["X_tr"], fh=[1, 2])
    out = f.update_predict(d["y_test"], X=d["X_all"])
    return {"pred": out, "cutoff": f.cutoff}


def op_predict_residuals(d):
    f = naive().fit(d["y_tr"], fh=FH)
    return {"pred": f.predict_residuals(d["y_next3"])}


def op_evaluate(d):
    cv = ExpandingWindowSplitter(initial_window=40, step_length=5, fh=FH)
    res = evaluate(naive(), cv, d["y_all"], return_data=True)
    return {"scores": res.filter(like="test_").reset_index(drop=True),
            "y_pred0": res["y_pred"].iloc[0]}


def op_evaluate_X(d):
    cv = ExpandingWindowSplitter(initial_window=40, step_length=5, fh=FH)
    res = evaluate(red(), cv, d["y_all"], X=d["X_all"], return_data=True)
    return {"scores": res.filter(like="test_").reset_index(drop=True),
            "y_pred0": res["y_pred"].iloc[0]}


def op_ttf(d):
    f = TransformedTargetForecaster([("des", Deseasonalizer(sp=7)),
                                     ("f", naive())]).fit(d["y_tr"], fh=FH)
    return {"pred": f.predict(), "cutoff": f.cutoff}


def op_gridcv(d):
    cv = ExpandingWindowSplitter(initial_window=30, step_length=5, fh=FH)
    g = ForecastingGridSearchCV(naive(), cv=cv,
                                param_grid={"strategy": ["last", "mean", "drift"]})
    g.fit(d["y_tr"], fh=FH)
    bp = g.best_params_ if hasattr(g, "best_params_") else None
    if isinstance(bp, pd.DataFrame):  # panel: per-instance tuning frame
        bp = None
    return {"pred": g.predict(), "best_params": bp}


def op_reconcile(d):
    f = ReconcilerForecaster(naive(), method="ols").fit(d["y_tr"], fh=FH)
    return {"pred": f.predict()}


def op_col_X(d):
    f = naive().fit(d["y_colX"], fh=FH)
    return {"pred": f.predict()}


OPS = [
    ("baseline fit(y)+predict [Naive]", op_fit_naive, ["Series", "Panel"]),
    ("fit(y,X)+predict(X)", op_fit_X, ["Series", "Panel"]),
    ("fit(y,X_disjoint) must raise", op_fit_X_disjoint, ["Series", "Panel"]),
    ("update(y_new,X_new)+predict", op_update, ["Series", "Panel"]),
    ("update_predict(y_test) [Naive]", op_update_predict, ["Series", "Panel"]),
    ("update_predict(y_test,X) [reduction]", op_update_predict_X, ["Series", "Panel"]),
    ("predict_residuals(y_oos)", op_predict_residuals, ["Series", "Panel"]),
    ("evaluate(Naive, ExpandingWindow)", op_evaluate, ["Series", "Panel"]),
    ("evaluate(reduction, X)", op_evaluate_X, ["Series", "Panel"]),
    ("TTF([Deseasonalizer, Naive])", op_ttf, ["Series", "Panel"]),
    ("ForecastingGridSearchCV(Naive)", op_gridcv, ["Series", "Panel"]),
    ("ReconcilerForecaster(ols)", op_reconcile, ["Hierarchical"]),
    ("Panel with column named 'X'", op_col_X, ["Panel"]),
]


def build(scitype, backend, D, Dh):
    if scitype == "Hierarchical":
        y = Dh
        return {"y_tr": to_backend(sl(y, 0, NTR), backend, scitype)}
    y, X = D[scitype]
    tb = lambda o: to_backend(o, backend, scitype)  # noqa: E731
    Xd = X.copy()
    if isinstance(Xd.index, pd.MultiIndex):
        Xd.index = Xd.index.set_levels(Xd.index.levels[-1] + pd.Timedelta(days=4000),
                                       level=-1)
    else:
        Xd.index = Xd.index + pd.Timedelta(days=4000)
    ycx = sl(y, 0, NTR).rename(columns={"y": "X"})
    return {
        "y_tr": tb(sl(y, 0, NTR)),
        "X_tr": tb(sl(X, 0, NTR)),
        "X_disj": tb(sl(Xd, 0, NTR)),
        "X_fut": tb(sl(X, NTR, NTR + 3)),
        "y_new": tb(sl(y, NTR, NTR + NNEW)),
        "X_new": tb(sl(X, NTR, NTR + NNEW)),
        "X_fut2": tb(sl(X, NTR + NNEW, NTR + NNEW + 3)),
        "y_test": tb(sl(y, NTR, T)),
        "y_next3": tb(sl(y, NTR, NTR + 3)),
        "y_all": tb(sl(y, 0, T)),
        "X_all": tb(sl(X, 0, T + 2)),
        "y_colX": tb(ycx),
    }


# ----------------------------------------------------------------------------
# comparison helpers
# ----------------------------------------------------------------------------
def tname(o):
    if isinstance(o, pd.Index):
        return f"{type(o).__name__}[{o.dtype}]"
    return type(o).__module__.split(".")[0] + "." + type(o).__name__


def compare(out, ref):
    """Return (strict_equal, loose_equal, note)."""
    if ref is None:
        return None, None, "no reference"
    res = {}
    for k, v in out.items():
        r = ref.get(k)
        if isinstance(r, (pd.DataFrame, pd.Series)) or isinstance(
            v, (pl.DataFrame, pd.DataFrame, pd.Series)
        ):
            try:
                a, b = to_pd(v), to_pd(r)
            except Exception as e:  # noqa: BLE001
                res[k] = ("conv-fail", repr(e)[:120])
                continue
            try:
                pd.testing.assert_frame_equal(a, b, check_freq=False)
                res[k] = ("strict", None)
            except AssertionError as e:
                same_shape = a.shape == b.shape
                vals_ok = same_shape and np.allclose(
                    np.asarray(a.select_dtypes("number"), dtype=float),
                    np.asarray(b.select_dtypes("number"), dtype=float),
                    equal_nan=True) if same_shape else False
                if vals_ok:
                    res[k] = ("values-only", str(e).replace("\n", " ")[:220])
                else:
                    res[k] = ("DIFF", str(e).replace("\n", " ")[:220])
        elif isinstance(r, pd.Index) or isinstance(v, pd.Index):
            try:
                ok = (type(v) is type(r)) and v.equals(r) and getattr(
                    v, "freq", None) == getattr(r, "freq", None)
                res[k] = ("strict", None) if ok else (
                    ("values-only", f"freq {getattr(v,'freq',None)} vs "
                     f"{getattr(r,'freq',None)}") if v.equals(r) else
                    ("DIFF", f"{list(v)} vs {list(r)}"))
            except Exception as e:  # noqa: BLE001
                res[k] = ("DIFF", repr(e)[:120])
        else:
            res[k] = ("strict", None) if v == r else ("DIFF", f"{v!r} vs {r!r}")
    return res


def root_frames(tb_exc):
    """Deepest frames inside sktime/ or protos/ (file:line func)."""
    frames = traceback.extract_tb(tb_exc)
    keep = [f for f in frames if "/sktime/sktime/" in f.filename or "protos/" in
            f.filename]
    out = []
    for f in keep[-3:]:
        fn = f.filename.split("/sktime/sktime/")[-1]
        fn = ("sktime/" + fn) if "protos/" not in f.filename else (
            "protos/" + f.filename.split("protos/")[-1])
        out.append(f"{fn}:{f.lineno} {f.name}")
    return out


# ----------------------------------------------------------------------------
def main():
    D = make_data()
    yh = _panel(["y"], 30, n_inst=4)
    # 2-level hierarchy: h0 -> (g0,g1) groups of instances
    yh = yh.copy()
    yh.index = pd.MultiIndex.from_arrays(
        [np.where(yh.index.get_level_values(0).isin(["s0", "s1"]), "g0", "g1"),
         yh.index.get_level_values(0), yh.index.get_level_values(1)],
        names=["h0", "h1", "time"])
    Dh = Aggregator(flatten_single_levels=False).fit_transform(yh)
    Dh = Dh.sort_index()

    approaches = sys.argv[1].split(",") if len(sys.argv) > 1 else [
        "stock", "x4", "x4nc", "x4fix", "x6b"]
    results = []
    refs = {}
    for opname, fn, scitypes in OPS:
        for sci in scitypes:
            for appr in approaches:
                for backend in ["pandas", "polars"]:
                    d = build(sci, backend, D, Dh)
                    meta = None
                    if appr == "x6b":
                        try:
                            new = {}
                            for k, v in d.items():
                                pdo, m = hub_in(v)
                                new[k] = pdo
                                meta = meta or m
                            d = new
                        except Exception as e:  # noqa: BLE001
                            results.append(dict(op=opname, scitype=sci, approach=appr,
                                                backend=backend, status="FAIL",
                                                stage="hub ingress",
                                                error=repr(e)[:200],
                                                where=root_frames(e.__traceback__)))
                            print(results[-1])
                            continue
                    t0 = time.perf_counter()
                    try:
                        with ctx_for(appr):
                            out = fn(d)
                        if appr == "x6b":
                            # "scores" is a Table (not a time series): no hub egress
                            out = {k: (v if k == "scores" else hub_out(v, meta))
                                   for k, v in out.items()}
                        status = "PASS"
                        err, where = None, None
                    except Exception as e:  # noqa: BLE001
                        out, status = None, "FAIL"
                        err = f"{type(e).__name__}: {str(e)[:200]}"
                        where = root_frames(e.__traceback__)
                    dt = time.perf_counter() - t0
                    rec = dict(op=opname, scitype=sci, approach=appr, backend=backend,
                               status=status, sec=round(dt, 4), error=err, where=where)
                    key = (opname, sci)
                    if out is not None:
                        rec["types"] = {k: tname(v) for k, v in out.items()}
                        if "raised" in out:
                            rec["raised"] = out["raised"]
                            if out["raised"].startswith("NO ERROR"):
                                rec["status"] = "FAIL(silent)"
                        if appr == "stock" and backend == "pandas":
                            refs[key] = out
                        if "raised" not in out:
                            rec["eq_vs_stock_pd"] = compare(out, refs.get(key))
                        rec["out_backend_matches_in"] = all(
                            (backend == "polars") == isinstance(v, pl.DataFrame)
                            for k, v in out.items()
                            if isinstance(v, (pd.DataFrame, pd.Series, pl.DataFrame))
                            and k not in ("scores",)
                        )
                    results.append(rec)
                    print(json.dumps(rec, default=str))
    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
