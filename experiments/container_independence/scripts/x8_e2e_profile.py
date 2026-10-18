"""x8: end-to-end profile of the CURRENT sktime vectorized forecasting path.

Question: where does time go when a non-panel forecaster (NaiveForecaster,
PolynomialTrendForecaster) is fit/predicted on Panel / Hierarchical data, and
therefore which decoupling approach can deliver real gains?

Method
------
1. Uninstrumented wall clock of ``fit(y, fh=[1,2,3]); predict()`` for each
   container (pandas MI, sktime polars_panel/polars_hierarchical, dask (tag-patched,
   see below)), interleaved within one process, min over repeats.
2. Deterministic, context-aware *exclusive-time* instrumentation: thin wrappers
   are installed (in-process monkeypatch, no sktime file is modified) on the
   framework functions; each wrapper charges its exclusive time to a bucket that
   depends on the call context (top level vs. inside a per-instance
   ``_vectorize_est_single`` vs. inside the forecaster's own ``_fit/_predict``).
   Wrapper overhead is reported (instrumented vs uninstrumented wall time).
3. cProfile cross-check (cumtime of key functions, top tottime functions).
4. "Tiers" for the same computation, as ceilings for possible designs:
   T1 lite loop   - same per-series pandas ``_fit/_predict`` of the sktime
                    forecaster, but no BaseForecaster boilerplate / no clone
   T2 numpy loop  - per-series python loop over numpy slices (no pandas per series)
   T3 vectorized  - pandas groupby / polars group_by expressions (eager + lazy)
   all validated against the sktime output.

Run (from worktree root)::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x8_e2e_profile.py --grid dev
"""

import argparse
import copy
import cProfile
import gc
import io
import json
import os
import pstats
import resource
import subprocess
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.dirname(HERE)
RESULTS = os.path.join(ED, "results")

GRIDS = {
    # (n_series, n_time)
    "dev": [(10, 100), (100, 100), (1000, 100), (5000, 100)],
    "large": [(10, 500), (100, 500), (1000, 500), (5000, 100), (20000, 100)],
    "smoke": [(10, 50), (50, 50)],
}

BUCKET_ORDER = [
    "top_checks",
    "top_convert_in",
    "vdf_construct",
    "slicing_iter",
    "pi_clone",
    "pi_checks",
    "pi_convert",
    "pi_cutoff_fh",
    "pi_other",
    "warnings",
    "inner",
    "inner_nested_core",
    "inner_nested_boiler",
    "vec_assembly",
    "reconstruct",
    "top_convert_back",
    "top_other",
]

ROLLUPS = {
    # container-related, done once per call (input checks, conversion, schema)
    "container_once": [
        "top_checks",
        "top_convert_in",
        "vdf_construct",
        "reconstruct",
        "top_convert_back",
    ],
    # only the pure conversion part (polars<->pandas etc)
    "conversion_in_out": ["top_convert_in", "top_convert_back"],
    # per-instance framework overhead (python loop machinery + BaseForecaster)
    "per_instance_framework": [
        "slicing_iter",
        "pi_clone",
        "pi_checks",
        "pi_convert",
        "pi_cutoff_fh",
        "pi_other",
        "warnings",
        "vec_assembly",
    ],
    # BaseForecaster boilerplate re-run on every slice (subset of the above)
    "pi_boilerplate": ["pi_checks", "pi_convert", "pi_cutoff_fh", "pi_other"],
    # the forecaster's own algorithm (incl. nested sktime estimators it calls)
    "inner_total": ["inner", "inner_nested_core", "inner_nested_boiler"],
    "inner_algorithm_only": ["inner", "inner_nested_core"],
    "top_misc": ["top_other"],
}


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------
class Instrumenter:
    """Context-aware exclusive-time instrumentation via reversible monkeypatching."""

    def __init__(self):
        self.acc = defaultdict(float)
        self.cnt = defaultdict(int)
        self.stack = []  # frames: [label, t0, child_time]
        self.state = {"pi": 0, "clone": 0, "inner": 0}
        self._patches = []  # (owner, attr, original)
        self.n_wrapper_calls = 0

    # -- classification ----------------------------------------------------
    def _classify(self, kind, name, args, kwargs):
        st = self.state
        parent = self.stack[-1][0] if self.stack else None
        if kind == "warn":
            # sktime.utils.warnings.warn: emitted per constructed forecaster
            return "warnings", ()
        if kind == "vec_single":
            meta = kwargs.get("meta", args[2] if len(args) > 2 else None)
            if meta is not None and meta.get("method") == "clone":
                return "pi_clone", ("clone",)
            return "pi_other", ("pi",)
        if st["clone"]:
            return "pi_clone", ()
        if st["inner"]:
            if kind == "tf_core":
                return "inner_nested_core", ()
            if kind == "est_inner":
                return "inner", ("inner",)
            return "inner_nested_boiler", ()
        if st["pi"]:
            if kind == "check":
                return "pi_checks", ()
            if kind == "convert":
                return "pi_convert", ()
            if kind == "cutoff_fh":
                return "pi_cutoff_fh", ()
            if kind == "est_inner":
                return "inner", ("inner",)
            return "pi_other", ()
        # top level
        if kind == "check":
            return "top_checks", ()
        if kind == "convert":
            if parent == "top_predict":
                return "top_convert_back", ()
            return "top_convert_in", ()
        if kind == "vdf":
            return "vdf_construct", ()
        if kind == "parallelize":
            return "slicing_iter", ()
        if kind == "vectorize_est":
            return "vec_assembly", ()
        if kind == "reconstruct":
            return "reconstruct", ()
        if kind == "bf_public":
            return ("top_predict" if name == "predict" else "top_fit"), ()
        if kind == "est_inner":
            # non-vectorized path (forecaster natively supports the scitype)
            return "inner", ("inner",)
        return "top_other", ()

    def _make_wrapper(self, orig, kind, name):
        inst = self

        def wrapper(*args, **kwargs):
            label, ctx = inst._classify(kind, name, args, kwargs)
            inst.n_wrapper_calls += 1
            frame = [label, 0.0, 0.0]
            inst.stack.append(frame)
            for c in ctx:
                inst.state[c] += 1
            frame[1] = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                t1 = time.perf_counter()
                inst.stack.pop()
                for c in ctx:
                    inst.state[c] -= 1
                el = t1 - frame[1]
                inst.acc[label] += el - frame[2]
                inst.cnt[label] += 1
                if inst.stack:
                    inst.stack[-1][2] += el

        wrapper.__wrapped_x8__ = orig
        wrapper.__name__ = getattr(orig, "__name__", "wrapped")
        wrapper.__doc__ = getattr(orig, "__doc__", None)
        return wrapper

    def _patch_attr(self, owner, attr, kind):
        orig = owner.__dict__[attr] if isinstance(owner, type) else getattr(owner, attr)
        if hasattr(orig, "__wrapped_x8__"):
            return
        w = self._make_wrapper(orig, kind, attr)
        setattr(owner, attr, w)
        self._patches.append((owner, attr, orig))

    def _patch_function_everywhere(self, func, kind):
        """Replace module-level references to ``func`` in all loaded sktime modules."""
        w = self._make_wrapper(func, kind, func.__name__)
        n = 0
        for mname, mod in list(sys.modules.items()):
            if mod is None or not mname.startswith("sktime"):
                continue
            d = getattr(mod, "__dict__", None)
            if not d:
                continue
            for k, v in list(d.items()):
                if v is func:
                    setattr(mod, k, w)
                    self._patches.append((mod, k, func))
                    n += 1
        return n

    def install(self, est_classes, tf_core_classes):
        from sktime.datatypes import _check, _convert, _vectorize
        from sktime.forecasting.base._base import BaseForecaster
        from sktime.transformations.base import BaseTransformer
        from sktime.utils import parallel

        for f in [_check.check_is_scitype, _check.check_is_mtype, _check.mtype]:
            self._patch_function_everywhere(f, "check")
        self._patch_function_everywhere(_convert.convert_to, "convert")
        self._patch_function_everywhere(_convert.convert, "convert")
        for f in [_vectorize.prepare_VectorizedDF, _vectorize.coerce_to_multiindex]:
            self._patch_function_everywhere(f, "vdf")
        self._patch_function_everywhere(parallel.parallelize, "parallelize")
        from sktime.utils import warnings as sk_warnings

        self._patch_function_everywhere(sk_warnings.warn, "warn")

        V = _vectorize.VectorizedDF
        self._patch_attr(V, "__init__", "vdf")
        self._patch_attr(V, "vectorize_est", "vectorize_est")
        self._patch_attr(V, "_vectorize_est_single", "vec_single")
        self._patch_attr(V, "reconstruct", "reconstruct")

        for a in ["fit", "predict"]:
            self._patch_attr(BaseForecaster, a, "bf_public")
        for a in ["_check_X_y", "_vectorize"]:
            self._patch_attr(BaseForecaster, a, "bf_misc")
        for a in ["_check_fh", "_update_y_X", "_set_cutoff_from_y"]:
            self._patch_attr(BaseForecaster, a, "cutoff_fh")
        for cls in est_classes:
            for a in ["_fit", "_predict"]:
                if a in cls.__dict__:
                    self._patch_attr(cls, a, "est_inner")
        for a in ["fit", "transform", "fit_transform"]:
            self._patch_attr(BaseTransformer, a, "tf_public")
        for cls in tf_core_classes:
            for a in ["_fit", "_transform"]:
                if a in cls.__dict__:
                    self._patch_attr(cls, a, "tf_core")

    def uninstall(self):
        for owner, attr, orig in reversed(self._patches):
            setattr(owner, attr, orig)
        self._patches = []

    def reset(self):
        self.acc = defaultdict(float)
        self.cnt = defaultdict(int)
        self.stack = []
        self.state = {"pi": 0, "clone": 0, "inner": 0}
        self.n_wrapper_calls = 0

    def result(self):
        acc = dict(self.acc)
        # fold top_fit / top_predict exclusive time into top_other
        acc["top_other"] = (
            acc.get("top_other", 0.0)
            + acc.pop("top_fit", 0.0)
            + acc.pop("top_predict", 0.0)
        )
        out = {b: acc.get(b, 0.0) for b in BUCKET_ORDER}
        for k, v in acc.items():
            if k not in out:
                out[k] = v
        return out


# ---------------------------------------------------------------------------
# data + estimators
# ---------------------------------------------------------------------------
def patch_dask_mtype_tags():
    """Work around broken dask mtype ``python_type`` tag (in-process only).

    The dask mtypes declare ``python_type="dask.dataframe"``; ``_precheck`` then
    requires a class named ``dataframe``, which never matches dask's ``DataFrame``,
    so every dask input is rejected by ``check_is_scitype``. We patch the class tag
    in this process only (no file is modified) to profile the dask path.
    """
    import sktime.datatypes._hierarchical._check as hc
    import sktime.datatypes._panel._check as pc
    import sktime.datatypes._series._check as sc

    patched = []
    for mod in (pc, hc, sc):
        for name in dir(mod):
            c = getattr(mod, name)
            if (
                isinstance(c, type)
                and "_tags" in c.__dict__
                and c.__dict__["_tags"].get("python_type") == "dask.dataframe"
            ):
                c._tags = {**c._tags, "python_type": "dask.DataFrame"}
                patched.append(c.__name__)
    return patched


def make_data(scitype, n_series, n_time):
    from common import make_panel, make_panel_fast

    if scitype == "Panel":
        return make_panel_fast(n_series, n_time)
    return make_panel(n_series, n_time, n_levels=2)


def make_containers(y, scitype, containers):
    from sktime.datatypes import convert_to

    out = {}
    for c in containers:
        if c == "pd":
            out[c] = y
        elif c == "pl":
            out[c] = convert_to(
                y, "polars_panel" if scitype == "Panel" else "polars_hierarchical"
            )
        elif c == "dask":
            out[c] = convert_to(
                y, "dask_panel" if scitype == "Panel" else "dask_hierarchical"
            )
    return out


def make_estimator(name):
    if name == "naive":
        from sktime.forecasting.naive import NaiveForecaster

        return NaiveForecaster(strategy="last")
    if name == "poly":
        from sktime.forecasting.trend import PolynomialTrendForecaster

        return PolynomialTrendForecaster(degree=1)
    raise ValueError(name)


def to_pandas_out(y_pred):
    """Bring any forecaster output to a pandas MI frame (for validation only)."""
    from sktime.datatypes._adapter.polars import convert_polars_to_pandas

    if isinstance(y_pred, pd.DataFrame):
        return y_pred
    if type(y_pred).__module__.startswith("polars"):
        return convert_polars_to_pandas(y_pred)
    if hasattr(y_pred, "compute"):
        from sktime.datatypes._adapter.dask_to_pd import convert_dask_to_pandas

        return convert_dask_to_pandas(y_pred)
    raise TypeError(type(y_pred))


def compare(ref, other):
    """Max abs diff of values after aligning on sorted index (ref, other pandas)."""
    a = ref.sort_index()
    b = other.sort_index()
    if len(a) != len(b):
        return {"ok": False, "why": f"len {len(a)} vs {len(b)}"}
    va = np.asarray(a.iloc[:, 0], dtype=float)
    vb = np.asarray(b.iloc[:, 0], dtype=float)
    ia = a.index.get_level_values(-1)
    ib = b.index.get_level_values(-1)
    idx_ok = bool((np.asarray(ia) == np.asarray(ib)).all())
    d = float(np.nanmax(np.abs(va - vb))) if len(va) else 0.0
    return {"ok": bool(d < 1e-6 and idx_ok), "max_abs_diff": d, "time_index_equal": idx_ok}


# ---------------------------------------------------------------------------
# tiers (ceilings)
# ---------------------------------------------------------------------------
def _inst_bounds_pd(y):
    """Start/end offsets of contiguous instances in a sorted pandas MI frame."""
    codes = y.index.codes[:-1]
    if len(codes) == 1:
        key = np.asarray(codes[0], dtype=np.int64)
    else:
        key = np.zeros(len(y), dtype=np.int64)
        for c, lev in zip(codes, y.index.levels[:-1]):
            key = key * (len(lev) + 1) + np.asarray(c, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    ends = np.r_[starts[1:], len(y)]
    return starts, ends


def _pd_out(y, starts, last_t, step, H, values):
    """Build sktime pandas MI output: instances x H future time points."""
    inst = y.index.droplevel(-1)[starts]
    rep = np.repeat(np.arange(len(starts)), H)
    hs = np.tile(np.arange(1, H + 1), len(starts))
    fut = last_t[rep] + step[rep] * hs
    if isinstance(inst, pd.MultiIndex):
        arrays = [inst.get_level_values(i)[rep] for i in range(inst.nlevels)]
    else:
        arrays = [inst[rep]]
    names = list(y.index.names)
    idx = pd.MultiIndex.from_arrays(arrays + [pd.DatetimeIndex(fut)], names=names)
    return pd.DataFrame({y.columns[0]: values}, index=idx)


def tier_numpy_loop_pd(y, est_name, H=3):
    """T2: per-series python loop over numpy slices (input pandas MI)."""
    starts, ends = _inst_bounds_pd(y)
    v = y.iloc[:, 0].to_numpy()
    t = y.index.get_level_values(-1).to_numpy()
    n = len(starts)
    last_t = t[ends - 1]
    step = t[ends - 1] - t[ends - 2]
    out = np.empty(n * H)
    if est_name == "naive":
        for i in range(n):
            seg = v[starts[i] : ends[i]]
            seg = seg[~np.isnan(seg)]
            out[i * H : (i + 1) * H] = seg[-1]
    else:
        tday = (t - np.datetime64(0, "ns")) / np.timedelta64(1, "D")
        stepd = step / np.timedelta64(1, "D")
        hs = np.arange(1, H + 1)
        for i in range(n):
            s, e = starts[i], ends[i]
            b1, b0 = np.polyfit(tday[s:e], v[s:e], 1)
            out[i * H : (i + 1) * H] = b0 + b1 * (tday[e - 1] + stepd[i] * hs)
    return _pd_out(y, starts, last_t, step, H, out)


def tier_vectorized_pd(y, est_name, H=3):
    """T3: pandas groupby-vectorized (no python loop over series)."""
    lv = list(range(y.index.nlevels - 1))
    col = y.columns[0]
    tser = pd.Series(y.index.get_level_values(-1), index=y.index)
    g_t = tser.groupby(level=lv, sort=False)
    last_t = g_t.last()
    step = last_t - g_t.nth(-2).to_numpy()
    if est_name == "naive":
        vals = y[col].groupby(level=lv, sort=False).last().to_numpy()
        values = np.repeat(vals, H)
    else:
        tday = (tser - pd.Timestamp(0)) / pd.Timedelta(days=1)
        df = pd.DataFrame({"t": tday.to_numpy(), "y": y[col].to_numpy()}, index=y.index)
        g = df.groupby(level=lv, sort=False)
        tm = g["t"].transform("mean")
        ym = g["y"].transform("mean")
        dt = df["t"] - tm
        num = (dt * (df["y"] - ym)).groupby(level=lv, sort=False).sum()
        den = (dt * dt).groupby(level=lv, sort=False).sum()
        b1 = (num / den).to_numpy()
        b0 = g["y"].mean().to_numpy() - b1 * g["t"].mean().to_numpy()
        tl = g["t"].last().to_numpy()
        sd = (step / pd.Timedelta(days=1)).to_numpy()
        hs = np.arange(1, H + 1)
        values = (b0[:, None] + b1[:, None] * (tl[:, None] + sd[:, None] * hs)).ravel()
    starts, _ = _inst_bounds_pd(y)
    return _pd_out(
        y, starts, last_t.to_numpy(), step.to_numpy(), H, values
    )


def _polars_expr_pipeline(frame, est_name, H=3):
    import polars as pl

    idx = [c for c in frame.collect_schema().names() if c.startswith("__index__")]
    inst, tcol = idx[:-1], idx[-1]
    vcol = [c for c in frame.collect_schema().names() if c not in idx][0]
    t = pl.col(tcol)
    step = t.diff().last()
    if est_name == "naive":
        aggs = [
            t.last().alias("_tl"),
            step.alias("_step"),
            pl.col(vcol).drop_nulls().drop_nans().last().alias("_v"),
        ]
        pred = pl.col("_v")
    else:
        td = t.dt.epoch("s").cast(pl.Float64) / 86400.0
        y = pl.col(vcol)
        b1 = ((td - td.mean()) * (y - y.mean())).sum() / ((td - td.mean()) ** 2).sum()
        aggs = [
            t.last().alias("_tl"),
            step.alias("_step"),
            b1.alias("_b1"),
            (y.mean() - b1 * td.mean()).alias("_b0"),
            td.last().alias("_tdl"),
        ]
        pred = pl.col("_b0") + pl.col("_b1") * (
            pl.col("_tdl")
            + pl.col("_h") * (pl.col("_step").dt.total_seconds() / 86400.0)
        )
    out = (
        frame.group_by(inst, maintain_order=True)
        .agg(aggs)
        .with_columns(pl.lit(list(range(1, H + 1))).alias("_h"))
        .explode("_h")
        .with_columns(
            (pl.col("_tl") + pl.col("_step") * pl.col("_h")).alias(tcol),
            pred.alias(vcol),
        )
        .select(inst + [tcol, vcol])
    )
    return out


def tier_vectorized_pl(df, est_name, lazy=False, H=3):
    """T3: polars group_by expressions on the sktime polars mtype frame."""
    if lazy:
        return _polars_expr_pipeline(df.lazy(), est_name, H).collect()
    return _polars_expr_pipeline(df, est_name, H)


def tier_numpy_loop_pl(df, est_name, H=3):
    """T2 from polars: numpy views of polars columns, per-series python loop."""
    import polars as pl

    idx = [c for c in df.columns if c.startswith("__index__")]
    inst, tcol = idx[:-1], idx[-1]
    vcol = [c for c in df.columns if c not in idx][0]
    ids = df.select(pl.struct(inst).rle_id()).to_series().to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
    ends = np.r_[starts[1:], len(df)]
    v = df[vcol].to_numpy()
    t = df[tcol].to_numpy()
    n = len(starts)
    out = np.empty(n * H)
    last_t = t[ends - 1]
    step = t[ends - 1] - t[ends - 2]
    if est_name == "naive":
        for i in range(n):
            seg = v[starts[i] : ends[i]]
            seg = seg[~np.isnan(seg)]
            out[i * H : (i + 1) * H] = seg[-1]
    else:
        tday = (t - np.datetime64(0, "ns")) / np.timedelta64(1, "D")
        stepd = step / np.timedelta64(1, "D")
        hs = np.arange(1, H + 1)
        for i in range(n):
            s, e = starts[i], ends[i]
            b1, b0 = np.polyfit(tday[s:e], v[s:e], 1)
            out[i * H : (i + 1) * H] = b0 + b1 * (tday[e - 1] + stepd[i] * hs)
    rep = np.repeat(np.arange(n), H)
    hs = np.tile(np.arange(1, H + 1), n)
    keys = df.select(inst)[starts[rep]]
    return keys.with_columns(
        pl.Series(tcol, last_t[rep] + step[rep] * hs),
        pl.Series(vcol, out),
    )


def tier_lite_loop(est, y, H=3):
    """T1: per-series loop calling only the forecaster's own _fit/_predict.

    Skips BaseForecaster.fit/predict boilerplate (checks, conversions, cutoff/fh
    handling, clone); per-instance state is set by hand. Same pandas inner code.
    """
    from sktime.utils.validation.forecasting import check_fh

    lv = list(range(y.index.nlevels - 1))
    inner_mtype = est.get_tag("y_inner_mtype")
    cls = type(est)
    template = dict(est.__dict__)
    feat = list(y.columns)
    ymeta = {"feature_names": feat}
    fh = None
    preds, keys = [], []
    freq = None
    for key, g in y.groupby(level=lv, sort=False):
        s = g.droplevel(lv)
        if freq is None:
            freq = pd.infer_freq(s.index)
        s.index.freq = freq
        if inner_mtype == "pd.Series":
            s = s.iloc[:, 0]
        f = cls.__new__(cls)
        f.__dict__.update(copy.copy(template))
        f._cutoff = s.index[-1:]
        if fh is None:
            fh = check_fh([1, 2, 3], freq=f._cutoff)
        f._fh = fh
        f._y = s
        f._y_metadata = ymeta
        f._is_vectorized = False
        f._fit(y=s, X=None, fh=fh)
        f._is_fitted = True
        p = f._predict(fh=fh, X=None)
        if isinstance(p, pd.Series):
            p = p.to_frame(feat[0])
        preds.append(p)
        keys.append(key)
    out = pd.concat(preds, keys=keys)
    out.index = out.index.set_names(y.index.names)
    return out


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------
def run_sktime(est, data):
    f = est.clone()
    t0 = time.perf_counter()
    f.fit(data, fh=[1, 2, 3])
    t1 = time.perf_counter()
    yp = f.predict()
    t2 = time.perf_counter()
    del f
    return t1 - t0, t2 - t1, yp


def cprofile_run(est, data, est_cls_names):
    pr = cProfile.Profile()
    f = est.clone()
    pr.enable()
    t0 = time.perf_counter()
    f.fit(data, fh=[1, 2, 3])
    f.predict()
    tot = time.perf_counter() - t0
    pr.disable()
    del f
    st = pstats.Stats(pr)
    key_funcs = {
        "_vectorize_est_single": ("_vectorize.py", "_vectorize_est_single"),
        "clone": ("skbase/base/_base.py", "clone"),
        "convert_to": ("datatypes/_convert.py", "convert_to"),
        "check_is_scitype": ("datatypes/_check.py", "check_is_scitype"),
        "mtype": ("datatypes/_check.py", "mtype"),
        "coerce_to_multiindex": ("_vectorize.py", "coerce_to_multiindex"),
        "reconstruct": ("_vectorize.py", "reconstruct"),
        "items": ("_vectorize.py", "items"),
        "get_cutoff": ("datatypes/_utilities.py", "get_cutoff"),
        "_check_fh": ("forecasting/base/_base.py", "_check_fh"),
        "BaseTransformer.fit_transform": ("transformations/base/_base.py", "fit_transform"),
        "polars_check_monotonic": ("_adapter/polars.py", "is_monotonically_increasing"),
        "convert_polars_to_pandas": ("_adapter/polars.py", "convert_polars_to_pandas"),
        "convert_pandas_to_polars": ("_adapter/polars.py", "convert_pandas_to_polars"),
    }
    cum = {}
    for label, (fsub, fname) in key_funcs.items():
        tot_c, ncalls = 0.0, 0
        for (fn, _ln, func), (cc, nc, tt, ct, _callers) in st.stats.items():
            if func == fname and fn.replace(os.sep, "/").endswith(fsub):
                tot_c += ct
                ncalls += nc
        cum[label] = {"cumtime": tot_c, "ncalls": ncalls}
    inner = 0.0
    for (fn, _ln, func), (cc, nc, tt, ct, _c) in st.stats.items():
        if func in ("_fit", "_predict") and any(
            fn.endswith(x) for x in est_cls_names
        ):
            inner += ct
    cum["est_inner(_fit+_predict)"] = {"cumtime": inner}
    rows = []
    for (fn, ln, func), (cc, nc, tt, ct, _c) in st.stats.items():
        rows.append((tt, ct, nc, f"{_short(fn)}:{ln}({func})"))
    rows.sort(reverse=True)
    top = [
        {"tottime": r[0], "cumtime": r[1], "ncalls": r[2], "func": r[3]} for r in rows[:20]
    ]
    return {"total_under_cprofile": tot, "cumtime": cum, "top_tottime": top}


def _short(fn):
    fn = fn.replace(os.sep, "/")
    for marker in ["site-packages/", "decouple-forecasters-pandas-4ef33f/"]:
        if marker in fn:
            return fn.split(marker, 1)[1]
    return fn


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def one_job(spec):
    """Run all measurements for one (scitype, estimator, n_series, n_time)."""
    warnings.simplefilter("ignore")
    from common import bench, env_info, rss_mb

    dask_patched = patch_dask_mtype_tags() if "dask" in spec["containers"] else []

    from sktime.datatypes import convert_to
    from sktime.transformations.lag import Lag

    warnings.simplefilter("ignore")

    scitype, est_name = spec["scitype"], spec["est"]
    n, T = spec["n_series"], spec["n_time"]
    reps = spec["reps"]
    rss0 = rss_mb()
    y = make_data(scitype, n, T)
    conts = make_containers(y, scitype, spec["containers"])
    est = make_estimator(est_name)
    est_cls = type(est)
    res = {
        "spec": spec,
        "env": env_info(),
        "python": sys.version.split()[0],
        "dask_tag_patched": dask_patched,
        "containers": {},
        "tiers": {},
    }

    # warmup (imports, caches)
    small = make_containers(make_data(scitype, 5, 20), scitype, spec["containers"])
    for c, d in small.items():
        run_sktime(est, d)

    # --- 1. uninstrumented wall clock, interleaved over containers -------------
    best = {c: [np.inf, np.inf] for c in conts}
    outs = {}
    for _ in range(reps):
        for c, d in conts.items():
            gc.collect()
            tf, tp, yp = run_sktime(est, d)
            if tf + tp < sum(best[c]):
                best[c] = [tf, tp]
            outs[c] = yp
    ref = outs["pd"]
    for c in conts:
        res["containers"][c] = {
            "wall_fit": best[c][0],
            "wall_predict": best[c][1],
            "wall_total": best[c][0] + best[c][1],
            "out_type": type(outs[c]).__module__ + "." + type(outs[c]).__name__,
            "valid_vs_pd": compare(ref, to_pandas_out(outs[c])),
        }
    del outs
    gc.collect()

    # reference for the hand-written tiers: exact OLS for poly. With sklearn>=1.9
    # LinearRegression(tol=1e-6) is passed as lstsq ``cond`` and truncates the
    # intercept direction of [1, days-since-epoch] -> sktime default is not OLS.
    ref_tiers = ref
    if est_name == "poly":
        from sklearn.linear_model import LinearRegression
        from sktime.forecasting.trend import PolynomialTrendForecaster

        est_ols = PolynomialTrendForecaster(
            degree=1, regressor=LinearRegression(fit_intercept=False, tol=0)
        )
        ref_tiers = est_ols.clone().fit(y, fh=[1, 2, 3]).predict()
        res["poly_sktime_default_vs_exact_ols"] = compare(ref_tiers, ref)

    # --- 2. instrumented buckets ---------------------------------------------
    ins = Instrumenter()
    ins.install([est_cls], [Lag])
    try:
        for c, d in conts.items():
            best_tot, best_b, best_calls = np.inf, None, None
            for _ in range(max(1, reps)):
                ins.reset()
                gc.collect()
                tf, tp, _yp = run_sktime(est, d)
                if tf + tp < best_tot:
                    best_tot = tf + tp
                    best_b = ins.result()
                    best_calls = ins.n_wrapper_calls
                    best_cnt = dict(ins.cnt)
                del _yp
            r = res["containers"][c]
            r["instr_total"] = best_tot
            r["instr_overhead_ratio"] = best_tot / r["wall_total"]
            r["instr_wrapper_calls"] = best_calls
            r["instr_counts"] = best_cnt
            r["buckets"] = best_b
            r["bucket_sum"] = sum(best_b.values())
            r["rollups"] = {
                k: sum(best_b.get(b, 0.0) for b in v) for k, v in ROLLUPS.items()
            }
            r["rollups_pct"] = {
                k: 100 * v / r["bucket_sum"] for k, v in r["rollups"].items()
            }
            r["buckets_pct"] = {
                k: 100 * v / r["bucket_sum"] for k, v in best_b.items()
            }
    finally:
        ins.uninstall()

    # --- 3. cProfile cross-check ---------------------------------------------
    if spec["cprofile"]:
        for c, d in conts.items():
            gc.collect()
            res["containers"][c]["cprofile"] = cprofile_run(
                est, d, [est_cls.__module__.replace(".", "/") + ".py"]
            )

    # --- 4. isolated conversion micro-bench ------------------------------------
    r_conv = {}
    if "pl" in conts:
        to_pl = "polars_panel" if scitype == "Panel" else "polars_hierarchical"
        to_pd = "pd-multiindex" if scitype == "Panel" else "pd_multiindex_hier"
        r_conv["pl_to_pd"] = bench(
            lambda: convert_to(conts["pl"], to_pd), repeat=reps + 2
        )
        r_conv["pd_to_pl"] = bench(lambda: convert_to(y, to_pl), repeat=reps + 2)
        # back-conversion size: output frame has n*3 rows
        r_conv["pd_to_pl_output_sized"] = bench(
            lambda: convert_to(ref, to_pl), repeat=reps + 2
        )
    res["conversion_microbench"] = r_conv

    # --- 5. tiers ----------------------------------------------------------------
    trep = max(3, reps + 2)
    tiers = {}
    if n <= spec["lite_max_n"]:
        tiers["T1_lite_loop_pd"] = bench(lambda: tier_lite_loop(est, y), repeat=reps)
        tiers["T1_valid"] = compare(ref, tier_lite_loop(est, y))
    tiers["T2_numpy_loop_pd"] = bench(lambda: tier_numpy_loop_pd(y, est_name), repeat=trep)
    tiers["T2_valid"] = compare(ref_tiers, tier_numpy_loop_pd(y, est_name))
    tiers["T3_vectorized_pd"] = bench(lambda: tier_vectorized_pd(y, est_name), repeat=trep)
    tiers["T3_pd_valid"] = compare(ref_tiers, tier_vectorized_pd(y, est_name))
    if "pl" in conts:
        dpl = conts["pl"]
        tiers["T2_numpy_loop_pl"] = bench(
            lambda: tier_numpy_loop_pl(dpl, est_name), repeat=trep
        )
        tiers["T2_pl_valid"] = compare(
            ref_tiers, to_pandas_out(tier_numpy_loop_pl(dpl, est_name))
        )
        tiers["T3_polars_eager"] = bench(
            lambda: tier_vectorized_pl(dpl, est_name), repeat=trep
        )
        tiers["T3_polars_lazy"] = bench(
            lambda: tier_vectorized_pl(dpl, est_name, lazy=True), repeat=trep
        )
        tiers["T3_pl_valid"] = compare(
            ref_tiers, to_pandas_out(tier_vectorized_pl(dpl, est_name))
        )
        tiers["T3_pl_lazy_valid"] = compare(
            ref_tiers, to_pandas_out(tier_vectorized_pl(dpl, est_name, lazy=True))
        )
    res["tiers"] = tiers
    res["rss_after_import_mb"] = rss0
    res["peak_rss_mb"] = peak_rss_mb()
    return res


def build_jobs(grid, scitypes, ests, containers, cprofile_max_n, dask_max_n, lite_max_n):
    jobs = []
    for n, T in GRIDS[grid]:
        for sc in scitypes:
            for e in ests:
                conts = [c for c in containers if not (c == "dask" and n > dask_max_n)]
                if n <= 100:
                    reps = 3
                elif n <= 1000:
                    reps = 2
                else:
                    reps = 1
                jobs.append(
                    {
                        "scitype": sc,
                        "est": e,
                        "n_series": n,
                        "n_time": T,
                        "containers": conts,
                        "reps": reps,
                        "cprofile": n <= cprofile_max_n,
                        "lite_max_n": lite_max_n,
                    }
                )
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="dev", choices=sorted(GRIDS))
    p.add_argument("--one", default=None, help="internal: json job spec")
    p.add_argument("--scitypes", default="Panel,Hierarchical")
    p.add_argument("--ests", default="naive,poly")
    p.add_argument("--containers", default="pd,pl,dask")
    p.add_argument("--cprofile-max-n", type=int, default=1000)
    p.add_argument("--dask-max-n", type=int, default=1000)
    p.add_argument("--lite-max-n", type=int, default=100000)
    p.add_argument("--max-n", type=int, default=None)
    p.add_argument("--jobs", type=int, default=1, help="parallel subprocesses")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    if a.one is not None:
        spec = json.loads(a.one)
        res = one_job(spec)
        sys.stdout.write("\n@@X8JSON@@" + json.dumps(res, default=str) + "\n")
        return

    jobs = build_jobs(
        a.grid,
        a.scitypes.split(","),
        a.ests.split(","),
        a.containers.split(","),
        a.cprofile_max_n,
        a.dask_max_n,
        a.lite_max_n,
    )
    if a.max_n is not None:
        jobs = [j for j in jobs if j["n_series"] <= a.max_n]
    out_path = a.out or os.path.join(RESULTS, f"x8_e2e_profile_{a.grid}.json")
    results = []
    if os.path.exists(out_path):
        try:
            with open(out_path) as fh:
                results = json.load(fh)["runs"]
        except Exception:  # noqa: BLE001
            results = []
    done = {json.dumps(r["spec"], sort_keys=True) for r in results}

    env = dict(os.environ)
    pending = [j for j in jobs if json.dumps(j, sort_keys=True) not in done]
    running = []

    import tempfile

    def launch(job):
        cmd = [sys.executable, "-W", "ignore", os.path.abspath(__file__), "--one", json.dumps(job)]
        # child output goes to temp files: PIPEs deadlock once warnings fill the buffer
        fo = tempfile.TemporaryFile(mode="w+")
        fe = tempfile.TemporaryFile(mode="w+")
        pr = subprocess.Popen(cmd, stdout=fo, stderr=fe, env=env, text=True)
        pr._x8_files = (fo, fe)
        return (job, pr, time.time())

    def save():
        with open(out_path, "w") as fh:
            json.dump({"grid": a.grid, "runs": results}, fh, indent=1, default=str)

    while pending or running:
        while pending and len(running) < a.jobs:
            running.append(launch(pending.pop(0)))
        time.sleep(0.5)
        for item in list(running):
            job, pr, t0 = item
            if pr.poll() is None:
                continue
            running.remove(item)
            fo, fe = pr._x8_files
            fo.seek(0)
            fe.seek(0)
            out, err = fo.read(), fe.read()
            fo.close()
            fe.close()
            if "@@X8JSON@@" in out:
                r = json.loads(out.split("@@X8JSON@@", 1)[1])
                r["elapsed_s"] = time.time() - t0
                results.append(r)
                save()
                pd_c = r["containers"]["pd"]
                print(
                    f"[x8] {job['scitype']:12s} {job['est']:5s} n={job['n_series']:6d} "
                    f"pd={pd_c['wall_total']:.3f}s "
                    + " ".join(
                        f"{c}={v['wall_total']:.3f}s" for c, v in r["containers"].items() if c != "pd"
                    )
                    + f" rss={r['peak_rss_mb']:.0f}MB ({r['elapsed_s']:.0f}s)",
                    flush=True,
                )
            else:
                print(f"[x8] FAILED {job}\n{err[-3000:]}", flush=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
