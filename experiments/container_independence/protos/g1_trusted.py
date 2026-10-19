"""G1 prototype: trusted per-slice "fast lane" for vectorized BaseForecaster.

When ``BaseForecaster._vectorize`` runs, the outer public method has already
checked ``y``/``X`` (scitype, categorical, nans), converted them once to the
pandas multiindex frame and built a ``VectorizedDF`` schema. Stock sktime then
calls the *public* ``fit``/``predict``/... of every per-instance clone, which
re-validates every slice (``check_is_scitype``), re-infers the mtype in every
``convert_to``, re-parses the cutoff via ``get_cutoff`` (another ``convert_to``)
and re-runs ``_reset_at`` on a fresh clone.

This module replaces ``_vectorize`` with a version that, per slice, sets the
state the public method would have set (``_y_metadata``, ``_y_mtype_last_seen``,
``_converter_store_y``, ``_cutoff``, ``_fh``, ``_y``, ``_X``, ``_is_vectorized``,
``_state``) from the schema and per-call constants, and then calls the private
``_fit/_predict/_update/_predict_interval/_predict_quantiles/_predict_var``.

Nothing in ``sktime/`` is edited. Use either

* ``install()`` / ``uninstall()`` / ``trusted_path()`` (context manager):
  monkeypatches ``BaseForecaster._vectorize`` for every forecaster, or
* ``make_trusted(cls)``: a subclass with the trusted ``_vectorize``.

Safety: a forecaster falls back to the stock ``_vectorize`` when

* its proto tag ``"capability:trusted_vectorize"`` is False (opt-out tag),
* its class overrides a public method or check/state helper the trusted path
  replaces (see ``GUARDED``), so its own logic would be skipped,
* the method is not one of ``TRUSTED_METHODS`` (e.g. update_predict_single),
* ``fit`` is called in state ``"pretrained"``,
* a slice is not a plain ``pd.DataFrame`` (defensive).

Per-call constants computed once (not per slice): inner-mtype converter
functions (``convert_dict`` lookup, no ``infer_mtype``), the per-column ``y``
metadata template (one stock ``check_is_scitype`` per column set; ``has_nans``
recomputed per slice with numpy), tag values, the ``remember_data`` config.
"""

import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

from sktime.datatypes import check_is_scitype, convert_to, update_data
from sktime.datatypes._convert import _get_first_mtype_of_same_scitype, convert_dict
from sktime.datatypes._utilities import get_cutoff
from sktime.datatypes._vectorize import VectorizedDF
from sktime.forecasting.base._base import BaseForecaster, _coerce_to_list
from sktime.utils.parallel import parallelize
from sktime.utils.validation.forecasting import check_fh
from sktime.utils.validation.series import check_equal_time_index

OPT_OUT_TAG = "capability:trusted_vectorize"
OPT_OUT_CONFIG = "trusted_vectorize"

TRUSTED_METHODS = (
    "fit",
    "update",
    "predict",
    "predict_interval",
    "predict_quantiles",
    "predict_var",
)
PREDICT_METHODS = ("predict", "predict_interval", "predict_quantiles", "predict_var")

# public methods / helpers whose override would be silently skipped by the
# trusted path; if a class overrides any of them, fall back to stock.
GUARDED = (
    "fit",
    "predict",
    "update",
    "predict_interval",
    "predict_quantiles",
    "predict_var",
    "_check_X_y",
    "_check_X",
    "_update_y_X",
    "_set_cutoff_from_y",
    "_set_cutoff",
    "_check_fh",
    "check_is_fitted",
    "_reset_at",
)

_ORIG_VECTORIZE = BaseForecaster._vectorize
_ELIGIBLE_CACHE = {}

# instrumentation: counts how often the trusted path / fallback ran
STATS = {"trusted": 0, "fallback": 0, "fallback_reasons": {}}

# "schema_items": single pass over the schema (y sliced once)
# "vectorize_est": stock VectorizedDF.vectorize_est iteration (y sliced twice,
#                  separate clone pass merged), trusted per-slice entry only
ITER_MODE = {"mode": "schema_items"}


def _fallback(reason):
    STATS["fallback"] += 1
    STATS["fallback_reasons"][reason] = STATS["fallback_reasons"].get(reason, 0) + 1


def class_eligible(cls):
    """Return (bool, reason): whether cls can use the trusted path (no override)."""
    if cls in _ELIGIBLE_CACHE:
        return _ELIGIBLE_CACHE[cls]
    res = (True, "")
    for name in GUARDED:
        if getattr(cls, name, None) is not getattr(BaseForecaster, name, None):
            # properties / methods compared by identity of the class attribute
            res = (False, f"overrides {name}")
            break
    _ELIGIBLE_CACHE[cls] = res
    return res


def estimator_eligible(est):
    """Return (bool, reason) for an estimator instance (class check + opt-out tag)."""
    ok, reason = class_eligible(type(est))
    if not ok:
        return ok, reason
    # class tag (set by estimator authors); instance set_tags is wiped by the
    # _reset_at() at the start of fit, so users opt out via set_config instead
    if not est.get_tag(OPT_OUT_TAG, tag_value_default=True, raise_error=False):
        return False, "opt-out tag"
    if not est.get_config().get(OPT_OUT_CONFIG, True):
        return False, "opt-out config"
    return True, ""


# --------------------------------------------------------------------------
# per-call context: everything that is the same for every slice
# --------------------------------------------------------------------------
class _Ctx:
    __slots__ = (
        "y_inner_mtype",
        "y_to",
        "y_conv",
        "X_inner_mtype",
        "X_to",
        "X_conv",
        "exog",
        "xy_same_index",
        "remember",
        "requires_fh",
        "insample",
        "pred_int_insample",
        "multivariate",
        "missing_ok",
        "meta_template",
        "meta_req",
        "fh_oos_cache",
    )


def _converter(from_type, to_mtypes, scitype="Series"):
    to_type = _get_first_mtype_of_same_scitype(from_type, _coerce_to_list(to_mtypes))
    key = (from_type, to_type, scitype)
    if key not in convert_dict:
        return to_type, None
    return to_type, convert_dict[key]


def _make_ctx(est):
    c = _Ctx()
    c.y_inner_mtype = _coerce_to_list(est.get_tag("y_inner_mtype"))
    c.X_inner_mtype = _coerce_to_list(est.get_tag("X_inner_mtype"))
    c.y_to, c.y_conv = _converter("pd.DataFrame", c.y_inner_mtype)
    c.X_to, c.X_conv = _converter("pd.DataFrame", c.X_inner_mtype)
    c.exog = est.get_tag("capability:exogenous")
    c.xy_same_index = est.get_tag("X-y-must-have-same-index")
    c.remember = est.get_config()["remember_data"]
    c.requires_fh = est.get_tag("requires-fh-in-fit")
    c.insample = est.get_tag("capability:insample")
    c.pred_int_insample = est.get_tag("capability:pred_int:insample")
    c.multivariate = est.get_tag("capability:multivariate")
    c.missing_ok = est.get_tag("capability:missing_values")
    req = ["n_features", "feature_names", "feature_kind"]
    if not c.multivariate:
        req += ["is_univariate"]
    if not c.missing_ok:
        req += ["has_nans"]
    c.meta_req = req
    c.meta_template = {}
    c.fh_oos_cache = {}
    return c


def _slice_metadata(ctx, y_slice):
    """Per-slice y metadata; stock check once per column set, then reused."""
    key = tuple(y_slice.columns)
    tmpl = ctx.meta_template.get(key)
    if tmpl is None:
        valid, msg, tmpl = check_is_scitype(
            y_slice,
            scitype=["Series", "Panel", "Hierarchical"],
            return_metadata=ctx.meta_req,
            var_name="y",
        )
        if not valid:  # pragma: no cover - outer check already passed
            raise TypeError(msg)
        ctx.meta_template[key] = tmpl
        return tmpl.copy()
    meta = tmpl.copy()
    if "has_nans" in meta:
        try:
            meta["has_nans"] = bool(np.isnan(y_slice.to_numpy(dtype=float)).any())
        except (TypeError, ValueError):
            meta["has_nans"] = bool(y_slice.isna().to_numpy().any())
    return meta


def _convert_y(ctx, y_slice, store):
    store.clear()  # store_behaviour="reset"
    if ctx.y_conv is None:
        return convert_to(
            y_slice, ctx.y_inner_mtype, as_scitype="Series", store=store,
            store_behaviour="reset",
        )
    return ctx.y_conv(y_slice, store=store)


def _convert_X(ctx, X_slice):
    if X_slice is None or not ctx.exog:
        return None
    if not isinstance(X_slice, pd.DataFrame) or isinstance(X_slice.index, pd.MultiIndex):
        # broadcast X (not vectorized) or unexpected: stock conversion
        return convert_to(X_slice, ctx.X_inner_mtype)
    if ctx.X_conv is None:
        return convert_to(X_slice, ctx.X_inner_mtype, as_scitype="Series")
    return ctx.X_conv(X_slice, store=None)


_DIRECT_CUTOFF_TYPES = (pd.Series, pd.DataFrame, np.ndarray)


def _cutoff_from(y_inner, prev):
    """Same result as BaseForecaster._set_cutoff_from_y, without re-inference.

    Fast case: pandas Series/DataFrame on a DatetimeIndex with freq set (the
    slices carry the freq from _enforce_index_freq): ``idx[-1:]`` keeps the freq
    without re-validating it (stock does ``idx[[-1]]`` + ``res.freq = freq``,
    which re-validates the offset; ~0.6 ms per call). Other cases: stock
    get_cutoff with ``convert_input=False`` (no infer_mtype) when the type is
    one get_cutoff handles directly, else stock get_cutoff.
    """
    if isinstance(y_inner, (pd.Series, pd.DataFrame)) and len(y_inner) > 0:
        idx = y_inner.index
        if (
            type(idx) is pd.DatetimeIndex
            and idx.freq is not None
            and (
                isinstance(y_inner, pd.Series)
                or not (y_inner.dtypes == "object").any()
            )
        ):
            return idx[-1:]
    if isinstance(y_inner, _DIRECT_CUTOFF_TYPES) and not (
        isinstance(y_inner, pd.DataFrame) and isinstance(y_inner.index, pd.MultiIndex)
    ):
        return get_cutoff(y_inner, prev, return_index=True, convert_input=False)
    return get_cutoff(y_inner, prev, return_index=True)


def _update_data(X, X_new):
    """sktime update_data for plain pandas inner types, without convert_to.

    For pd.Series/pd.DataFrame (non-MultiIndex) both convert_to calls in
    update_data are identities; the rest is copied verbatim.
    """
    plain = (pd.Series, pd.DataFrame)
    if (
        isinstance(X, plain)
        and isinstance(X_new, plain)
        and not isinstance(X.index, pd.MultiIndex)
        and not isinstance(X_new.index, pd.MultiIndex)
        and len(X_new) > 0
    ):
        X_new = X_new.combine_first(X)
        if hasattr(X.index, "freq") and hasattr(X_new.index, "freq"):
            if X.index.freq == X_new.index.freq:
                if X.index.freqstr != X_new.index.freqstr:
                    X_new.index.freq = X.index.freq
        return X_new
    return update_data(X, X_new)


def _fh_out_of_sample_ok(ctx, est, pred_int):
    """In-sample check of BaseForecaster._check_fh; cached for relative fh."""
    ok_insample = ctx.pred_int_insample if pred_int else ctx.insample
    fh = est._fh
    if ok_insample or fh is None:
        return
    if fh.is_relative:
        k = (id(fh), pred_int)
        res = ctx.fh_oos_cache.get(k)
        if res is None:
            res = fh.is_all_out_of_sample(est._cutoff)
            ctx.fh_oos_cache[k] = res
    else:
        res = fh.is_all_out_of_sample(est._cutoff)
    if not res:
        raise NotImplementedError(
            f"{est.__class__.__name__} can not perform in-sample prediction. "
            f"Found fh with in sample index: {fh}"
        )


def _trusted_check_fh(ctx, est, fh, pred_int=False):
    """BaseForecaster._check_fh for an already-validated slice estimator."""
    if fh is None:
        # rare branches (errors / nothing to do): delegate to stock
        return est._check_fh(fh, pred_int=pred_int)
    # vectorized slices are never themselves vectorized: freq from slice cutoff
    fh = check_fh(fh=fh, freq=est._cutoff)
    if not ctx.requires_fh or not est._fh or not est.is_fitted:
        est._fh = fh
    elif est._fh and not np.array_equal(fh, est._fh):
        return est._check_fh(fh, pred_int=pred_int)  # raises the stock error
    _fh_out_of_sample_ok(ctx, est, pred_int)
    return est._fh


# --------------------------------------------------------------------------
# per-slice trusted entries
# --------------------------------------------------------------------------
def trusted_fit_slice(est, y_slice, X_slice, fh, ctx):
    """Fit a fresh clone ``est`` on a validated pd.DataFrame Series slice."""
    if est._state != "new":  # fresh clones are already in reset state
        est._reset_at("pretrained")
    meta = _slice_metadata(ctx, y_slice)
    est._y_metadata = meta
    est._y_mtype_last_seen = meta["mtype"]
    if not ctx.missing_ok and meta.get("has_nans"):
        raise ValueError(
            f"{type(est).__name__} cannot handle missing data (nans), "
            "but y passed contained missing data."
        )
    y_inner = _convert_y(ctx, y_slice, est._converter_store_y)
    X_inner = _convert_X(ctx, X_slice)
    if X_inner is not None and ctx.xy_same_index and X_slice is not None:
        check_equal_time_index(X_slice, y_slice, mode="contains")
    # _update_y_X
    est._cutoff = _cutoff_from(y_inner, est.cutoff)
    if ctx.remember:
        est._y = y_inner
        if X_inner is not None:
            est._X = X_inner
    est._is_vectorized = False
    # _check_fh (state "new": fh written to self)
    if fh is None:
        fh = est._check_fh(None)
    else:
        fh = check_fh(fh=fh, freq=est._cutoff)
        est._fh = fh
        _fh_out_of_sample_ok(ctx, est, pred_int=False)
    est._fit(y=y_inner, X=X_inner, fh=fh)
    est._state = "fitted"
    return est


def trusted_update_slice(est, y_slice, X_slice, update_params, ctx):
    if y_slice is None or len(y_slice) == 0:
        return est.update(y=y_slice, X=X_slice, update_params=update_params)
    meta = _slice_metadata(ctx, y_slice)
    est._y_metadata = meta
    est._y_mtype_last_seen = meta["mtype"]
    if not ctx.missing_ok and meta.get("has_nans"):
        raise ValueError(f"{type(est).__name__} cannot handle missing data (nans)")
    y_inner = _convert_y(ctx, y_slice, est._converter_store_y)
    X_inner = _convert_X(ctx, X_slice)
    if X_inner is not None and ctx.xy_same_index and X_slice is not None:
        check_equal_time_index(X_slice, y_slice, mode="contains")
    est._cutoff = _cutoff_from(y_inner, est.cutoff)
    if ctx.remember:
        if getattr(est, "_y", None) is None or not est.is_fitted:
            est._y = y_inner
        else:
            est._y = _update_data(est._y, y_inner)
        est._cutoff = _cutoff_from(y_inner, est.cutoff)
        if X_inner is not None:
            if getattr(est, "_X", None) is None or not est.is_fitted:
                est._X = X_inner
            else:
                est._X = _update_data(est._X, X_inner)
    if (
        TRUSTED_REFIT["on"]
        and update_params
        and ctx.remember
        and type(est)._update is BaseForecaster._update
    ):
        _trusted_default_refit(est, ctx)
    else:
        est._update(y=y_inner, X=X_inner, update_params=update_params)
    return est


TRUSTED_REFIT = {"on": True}


def _trusted_default_refit(est, ctx):
    """BaseForecaster._update default (refit on remembered data), trusted.

    Stock calls the public ``self.fit(self._y, self._X, self._fh)`` per slice,
    i.e. a full re-validation of data the forecaster already holds in its inner
    mtype. Here: same warning, same reset, same state writes, then ``_fit``.
    """
    from sktime.utils.warnings import warn

    warn(
        f"NotImplementedWarning: {est.__class__.__name__} "
        f"does not have a custom `update` method implemented. "
        f"{est.__class__.__name__} will be refit each time "
        f"`update` is called with update_params=True. "
        "To pool data and control refit frequency, wrap with "
        "``UpdateRefitsEvery`` from ``sktime.forecasting.stream`` "
        "(e.g. ``refit_interval=0`` to refit every update, or a "
        "larger interval / ``UpdateEvery`` to refit less often).",
        obj=est,
    )
    mtype_last_seen = est._y_mtype_last_seen
    y_metadata = est._y_metadata
    store = est._converter_store_y
    y_all, X_all, fh = est._y, est._X, est._fh
    # --- public fit(y=_y, X=_X, fh=_fh), trusted ---
    if not est._state == "pretrained":
        est._reset_at("pretrained")
    X_all = X_all if ctx.exog else None
    if X_all is not None and ctx.xy_same_index:
        check_equal_time_index(X_all, y_all, mode="contains")
    est._cutoff = _cutoff_from(y_all, est.cutoff)
    est._y = y_all
    if X_all is not None:
        est._X = X_all
    est._is_vectorized = False
    if fh is None:
        fh = est._check_fh(None)
    else:
        fh = check_fh(fh=fh, freq=est._cutoff)
        est._fh = fh
        _fh_out_of_sample_ok(ctx, est, pred_int=False)
    est._fit(y=y_all, X=X_all, fh=fh)
    est._state = "fitted"
    # --- restore, as BaseForecaster._update does ---
    est._y_mtype_last_seen = mtype_last_seen
    est._y_metadata = y_metadata
    est._converter_store_y = store


def _to_outer_mtype(est, y_pred):
    """predict's convert_to(y_pred, _y_metadata['mtype'], store, 'freeze')."""
    target = est._y_metadata["mtype"]
    if target == "pd.DataFrame":
        if type(y_pred) is pd.DataFrame and not isinstance(y_pred.index, pd.MultiIndex):
            return y_pred
        if type(y_pred) is pd.Series:
            fn = convert_dict.get(("pd.Series", "pd.DataFrame", "Series"))
            if fn is not None:
                store = est._converter_store_y
                return fn(y_pred, store=dict(store) if store else store)
    return convert_to(
        y_pred, target, store=est._converter_store_y, store_behaviour="freeze"
    )


def trusted_predict_slice(est, method, X_slice, fh, ctx, kw):
    X_inner = _convert_X(ctx, X_slice)
    pred_int = method != "predict"
    fh = _trusted_check_fh(ctx, est, fh, pred_int=pred_int)
    if method == "predict":
        return _to_outer_mtype(est, est._predict(fh=fh, X=X_inner))
    if method == "predict_interval":
        return est._predict_interval(fh=fh, X=X_inner, coverage=kw["coverage"])
    if method == "predict_quantiles":
        return est._predict_quantiles(fh=fh, X=X_inner, alpha=kw["alpha"])
    if method == "predict_var":
        return est._predict_var(fh=fh, X=X_inner, cov=kw.get("cov", False))
    raise ValueError(method)


def _single(vec_tuple, meta):
    """One slice; module-level so parallel backends can pickle it."""
    method = meta["method"]
    ctx = meta["ctx"]
    key, col, est, y_i, X_i = vec_tuple
    if method == "fit":
        res = trusted_fit_slice(est.clone(), y_i, X_i, meta["fh"], ctx)
    elif method == "update":
        res = trusted_update_slice(est, y_i, X_i, meta["update_params"], ctx)
    else:
        res = trusted_predict_slice(est, method, X_i, meta["fh"], ctx, meta["kw"])
    return key, col, res


# skbase-style methods for the "vectorize_est" iteration mode
def _g1_fit_entry(self, y, X=None, fh=None, ctx=None):
    return trusted_fit_slice(self.clone(), y, X, fh, ctx)


def _g1_update_entry(self, y, X=None, update_params=True, ctx=None):
    return trusted_update_slice(self, y, X, update_params, ctx)


def _make_pred_entry(method):
    def entry(self, X=None, fh=None, ctx=None, **kw):
        return trusted_predict_slice(self, method, X, fh, ctx, kw)

    return entry


# --------------------------------------------------------------------------
# the trusted _vectorize
# --------------------------------------------------------------------------
def _iter_slices(schema, frame, iterate_as=None, iterate_cols=None):
    if frame is None:
        return None
    return (inst for _, _, inst in schema.items(
        X=frame, iterate_as=iterate_as, iterate_cols=iterate_cols))


def _estimators_frame(ret):
    """Same assembly as VectorizedDF.vectorize_est(return_type='pd.DataFrame')."""
    df_long = pd.DataFrame(ret)
    cols_right_order = df_long.loc[:, 1].unique()
    rows_right_order = df_long.loc[:, 0].unique()
    df = df_long.pivot(index=0, columns=1, values=2)
    df = df.reindex(cols_right_order, axis=1)
    df = df.reindex(rows_right_order, axis=0)
    df.index.names = [None] * len(df.index.names)
    df.columns.name = None
    try:
        df.index = pd.MultiIndex.from_tuples(df.index)
    except (TypeError, ValueError):
        pass
    return df


def trusted_vectorize(self, methodname, **kwargs):
    """Drop-in for BaseForecaster._vectorize with a trusted per-slice entry."""
    if methodname not in TRUSTED_METHODS:
        _fallback(f"method {methodname}")
        return _ORIG_VECTORIZE(self, methodname, **kwargs)
    ok, reason = estimator_eligible(self)
    if not ok:
        _fallback(reason)
        return _ORIG_VECTORIZE(self, methodname, **kwargs)
    if methodname == "fit" and getattr(self, "_state", "new") == "pretrained":
        _fallback("pretrained")
        return _ORIG_VECTORIZE(self, methodname, **kwargs)

    X = kwargs.get("X", None)
    X_frame, X_schema = X if isinstance(X, tuple) else (X, None)
    y = kwargs.get("y", None)
    y_frame, y_schema = y if y else (None, None)

    if methodname in ("fit", "update"):
        if not isinstance(y_frame, pd.DataFrame):
            _fallback("y not pd.DataFrame")
            return _ORIG_VECTORIZE(self, methodname, **kwargs)
        schema = y_schema
    else:
        schema = self._y_schema
    if not isinstance(schema, VectorizedDF):
        _fallback("no schema")
        return _ORIG_VECTORIZE(self, methodname, **kwargs)

    STATS["trusted"] += 1
    ctx = _make_ctx(self)
    backend = self.get_config()["backend:parallel"]
    backend_params = self.get_config()["backend:parallel:params"]
    n_cols = schema.shape[1]

    if ITER_MODE["mode"] == "vectorize_est":
        return _trusted_via_vectorize_est(
            self, methodname, kwargs, schema, ctx, backend, backend_params
        )

    # ---- single-pass iteration over the schema ----
    keys = schema.items()  # (row, col, None) keys, no data
    if methodname in ("fit", "update"):
        y_it = _iter_slices(schema, y_frame)
    else:
        y_it = None

    # X: row-vectorized, repeated over columns (as stock explode does)
    if X_schema is not None and isinstance(X_schema, VectorizedDF):
        X_it = _iter_slices(X_schema, X_frame, schema.iterate_as, False)
        if schema.iterate_cols:
            X_it = (x for x in X_it for _ in range(n_cols))
    else:
        X_it = None

    if methodname == "fit":
        ests = (self for _ in iter(int, 1))
    else:
        ests = (cell for _, row in self.forecasters_.iterrows() for cell in row.values)

    def gen():
        for (rk, ck, _), est in zip(keys, ests):
            y_i = next(y_it) if y_it is not None else None
            X_i = next(X_it) if X_it is not None else X_frame
            yield rk, ck, est, y_i, X_i

    meta = {
        "method": methodname,
        "ctx": ctx,
        "fh": kwargs.get("fh", None),
        "update_params": kwargs.get("update_params", True),
        "kw": {k: kwargs[k] for k in ("coverage", "alpha", "cov") if k in kwargs},
    }
    ret = parallelize(
        fun=_single, iter=gen(), meta=meta, backend=backend,
        backend_params=backend_params,
    )
    ret = [
        ("forecasters" if rk is None else rk, "forecasters" if ck is None else ck, r)
        for rk, ck, r in ret
    ]
    if methodname in ("fit", "update"):
        self._y_schema = y_schema
        self.forecasters_ = _estimators_frame(ret)
        return self
    return _reconstruct(self, [r for _, _, r in ret])


def _reconstruct(self, y_preds):
    m = len(self.forecasters_.columns)
    col_multiindex = "multiindex" if m > 1 else "none"
    y_pred = self._y_schema.reconstruct(
        y_preds, overwrite_index=True, col_multiindex=col_multiindex
    )
    if col_multiindex == "multiindex":
        y_pred.columns = y_pred.columns.droplevel(1)
    return y_pred


def _trusted_via_vectorize_est(self, methodname, kwargs, schema, ctx, backend, bp):
    """Ablation: stock VectorizedDF.vectorize_est iteration, trusted entry."""
    X = kwargs.get("X", None)
    common = dict(
        args_rowvec={"X": X},
        rowname_default="forecasters",
        colname_default="forecasters",
        varname_of_self="y",
        backend=backend,
        backend_params=bp,
        ctx=ctx,
    )
    if methodname == "fit":
        self._y_schema = kwargs["y"][1]
        self.forecasters_ = schema.vectorize_est(
            self, method="_g1_fit_entry", y=kwargs["y"], fh=kwargs.get("fh"), **common
        )
        return self
    if methodname == "update":
        self._y_schema = kwargs["y"][1]
        self.forecasters_ = schema.vectorize_est(
            self.forecasters_, method="_g1_update_entry", y=kwargs["y"],
            update_params=kwargs.get("update_params", True), **common,
        )
        return self
    extra = {k: kwargs[k] for k in ("fh", "coverage", "alpha", "cov") if k in kwargs}
    y_preds = schema.vectorize_est(
        self.forecasters_, method=f"_g1_{methodname}_entry", return_type="list",
        **extra, **common,
    )
    return _reconstruct(self, y_preds)


# --------------------------------------------------------------------------
# installation
# --------------------------------------------------------------------------
def _install_entries():
    BaseForecaster._g1_fit_entry = _g1_fit_entry
    BaseForecaster._g1_update_entry = _g1_update_entry
    for m in PREDICT_METHODS:
        setattr(BaseForecaster, f"_g1_{m}_entry", _make_pred_entry(m))


def install(mode="schema_items"):
    """Monkeypatch BaseForecaster._vectorize with the trusted path."""
    ITER_MODE["mode"] = mode
    _install_entries()
    BaseForecaster._vectorize = trusted_vectorize


def uninstall():
    BaseForecaster._vectorize = _ORIG_VECTORIZE


@contextmanager
def trusted_path(mode="schema_items"):
    install(mode)
    try:
        yield
    finally:
        uninstall()


def make_trusted(cls):
    """Subclass of forecaster class ``cls`` that uses the trusted path."""
    _install_entries()
    return type(f"Trusted{cls.__name__}", (cls,), {"_vectorize": trusted_vectorize})


def reset_stats():
    STATS["trusted"] = 0
    STATS["fallback"] = 0
    STATS["fallback_reasons"] = {}


warnings.filterwarnings("ignore", category=FutureWarning)
