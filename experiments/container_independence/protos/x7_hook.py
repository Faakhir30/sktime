"""x7: optional engine-native panel hook for BaseForecaster (prototype, no sktime edits).

Idea
----
A forecaster MAY implement a panel-level fast path that computes all instances of a
Panel / Hierarchical input at once, in the input's own engine (pandas, polars eager,
polars lazy, ...). It declares this via tags::

    "capability:panel_native": True
    "panel_native:backends": ["pandas", "polars", "polars_lazy"]

and implements two methods::

    _fit_panel(self, pf: PanelFrame) -> params      # one row per instance, native
    _predict_panel(self, params, steps) -> frame    # native frame, same layout as in

plus an optional param-dependent applicability predicate::

    _panel_native_supported(self) -> bool           # e.g. only strategy="last"

``BaseForecaster.fit`` (here: ``PanelNativeMixin.fit``) routes to the native path
INSTEAD of ``prepare_VectorizedDF`` + the per-instance loop when ALL of:

  * the tag is set and ``_panel_native_supported()`` is True,
  * the container is recognised as one of the declared backends (cheap structural
    check, no conversion),
  * the input is a Panel/Hierarchical (>= 1 instance level), and X is None
    (exogenous handling left out of the sketch).

Otherwise the existing per-instance loop runs unchanged, so behaviour is opt-in and
partial implementations are safe.

Fitted state is a *params table* (one row per instance, in the input's engine)
instead of ``forecasters_`` (a pandas DataFrame of fitted estimator objects).

Methods the native path does not implement (predict_interval, predict_quantiles,
predict_var, get_fitted_params) MATERIALIZE the per-instance forecasters on first
use: ``super().fit`` on the remembered input (converted to pandas as today), then
delegate. So the full sktime API keeps working; only the fast path is new.

Containers
----------
``pandas``      pd.DataFrame with row MultiIndex (instance levels..., time)
``polars``      pl.DataFrame in sktime's polars convention: ``__index__<name>``
                columns (instance levels..., time) + value columns
``polars_lazy`` pl.LazyFrame with the same convention; fit runs ONE collect of the
                (small) params table; projection pushdown applies, so only key,
                time and value columns are read from e.g. a parquet scan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sktime.forecasting.naive import NaiveForecaster

IX = "__index__"


# =============================================================================
# container recognition (structural, no conversion)
# =============================================================================


@dataclass
class PanelFrame:
    """Thin descriptor of a panel container: which parts are keys/time/values."""

    frame: object
    backend: str  # "pandas" | "polars" | "polars_lazy"
    key_names: list  # instance level names (pandas level names / polars columns)
    time_name: str
    value_names: list


def detect_panel(obj):
    """Return PanelFrame if obj is a recognised panel container, else None."""
    if isinstance(obj, pd.DataFrame) and isinstance(obj.index, pd.MultiIndex):
        if obj.index.nlevels < 2:
            return None
        names = list(obj.index.names)
        return PanelFrame(obj, "pandas", names[:-1], names[-1], list(obj.columns))
    try:
        import polars as pl
    except ImportError:  # pragma: no cover
        return None
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        cols = (
            obj.collect_schema().names()
            if isinstance(obj, pl.LazyFrame)
            else obj.columns
        )
        ix = [c for c in cols if c.startswith(IX)]
        if len(ix) < 2:
            return None  # a single Series -> not a panel
        backend = "polars_lazy" if isinstance(obj, pl.LazyFrame) else "polars"
        vals = [c for c in cols if c not in ix]
        return PanelFrame(obj, backend, ix[:-1], ix[-1], vals)
    return None


def _freq_from_pandas(y):
    lvl = y.index.levels[-1]
    t = y.index.get_level_values(-1)
    if isinstance(t, pd.PeriodIndex):
        return t.freq
    if isinstance(t, pd.DatetimeIndex):
        f = lvl.freq or pd.infer_freq(lvl[: min(len(lvl), 50)])
        return pd.tseries.frequencies.to_offset(f) if f is not None else None
    return None  # integer time: step 1


# =============================================================================
# the hook (what BaseForecaster would do)
# =============================================================================


class PanelNativeMixin:
    """Put before a BaseForecaster subclass in the MRO."""

    _tags = {
        "capability:panel_native": True,
        "panel_native:backends": ["pandas", "polars", "polars_lazy"],
    }

    # --- to be implemented by the forecaster -------------------------------
    def _panel_native_supported(self):
        return True

    def _fit_panel(self, pf):
        raise NotImplementedError

    def _predict_panel(self, params, steps):
        raise NotImplementedError

    # --- routing -----------------------------------------------------------
    def _panel_route(self, y, X):
        if not self.get_tag("capability:panel_native", False, raise_error=False):
            return None
        if X is not None or not self._panel_native_supported():
            return None
        pf = detect_panel(y)
        if pf is None:
            return None
        if pf.backend not in self.get_tag("panel_native:backends", []):
            return None
        return pf

    def fit(self, y, X=None, fh=None):
        pf = self._panel_route(y, X)
        if pf is None:
            self._panel_native = False
            return super().fit(y, X=X, fh=fh)

        if not self._state == "pretrained":
            self._reset_at("pretrained")
        self._panel_native = True
        self._panel_meta = pf
        self._panel_y = y  # reference only (no copy); for materialization fallback

        # one engine pass: params table, also yields cutoff (max of instance ends)
        params, cutoff_scalar, freq = self._fit_panel(pf)
        self._panel_params = params
        self._panel_freq = freq
        cut = pd.Index([cutoff_scalar])
        if isinstance(cut, pd.DatetimeIndex) and freq is not None:
            cut = pd.DatetimeIndex(cut, freq=freq)
        self._cutoff = cut
        # per-instance relative fh semantics, as in the vectorized loop
        self._is_vectorized = True
        self._check_fh(fh)
        self._state = "fitted"
        return self

    def predict(self, fh=None, X=None):
        if not getattr(self, "_panel_native", False):
            return super().predict(fh=fh, X=X)
        self.check_is_fitted()
        fh = self._check_fh(fh)
        if not fh.is_relative:
            # absolute fh: valid only if all instances share the cutoff; the sketch
            # converts w.r.t. the global cutoff (documented limitation)
            fh = fh.to_relative(self.cutoff)
        steps = np.asarray(fh.to_numpy(), dtype="int64")
        return self._predict_panel(self._panel_params, steps)

    # --- materialization fallback for everything not implemented natively --
    def _materialize(self):
        if getattr(self, "_panel_native", False):
            fh = self._fh
            self._panel_native = False
            super().fit(self._panel_y, fh=fh)

    def predict_interval(self, fh=None, X=None, coverage=0.90):
        self._materialize()
        return super().predict_interval(fh=fh, X=X, coverage=coverage)

    def predict_quantiles(self, fh=None, X=None, alpha=None):
        self._materialize()
        return super().predict_quantiles(fh=fh, X=X, alpha=alpha)

    def predict_var(self, fh=None, X=None, cov=False):
        self._materialize()
        return super().predict_var(fh=fh, X=X, cov=cov)

    def update(self, y, X=None, update_params=True):
        if getattr(self, "_panel_native", False) and hasattr(self, "_update_panel"):
            pf = detect_panel(y)
            if pf is not None and pf.backend == self._panel_meta.backend:
                return self._update_panel(pf)
        self._materialize()
        return super().update(y, X=X, update_params=update_params)

    def get_fitted_params(self, deep=True):
        if getattr(self, "_panel_native", False):
            return {"params_table": self._panel_params}
        return super().get_fitted_params(deep=deep)


# =============================================================================
# example: NaiveForecaster(strategy="last", sp=1) panel-native in pandas + polars
# =============================================================================


class NaivePanelNative(PanelNativeMixin, NaiveForecaster):
    """NaiveForecaster with a panel-native fast path for strategy="last", sp=1.

    Other strategies / sp fall back to the per-instance loop (identical results).
    NaN semantics as NaiveForecaster "last": last non-missing value per column;
    future time stamps = each instance's own last time stamp + step * freq.
    """

    def _panel_native_supported(self):
        return self.strategy == "last" and (self.sp or 1) == 1

    # ---- fit: one pass, params table ---------------------------------------
    def _fit_panel(self, pf):
        if pf.backend == "pandas":
            return self._fit_panel_pandas(pf)
        return self._fit_panel_polars(pf)

    def _fit_panel_pandas(self, pf):
        y = pf.frame
        lv = list(range(y.index.nlevels - 1))
        lv = lv[0] if len(lv) == 1 else lv
        g = y.groupby(level=lv, sort=False)
        last_vals = g.last()  # last non-NaN per column (cythonized, no python loop)
        tail = g.tail(1)  # last row per instance -> instance end time stamp
        tlast = tail.index.get_level_values(-1)
        params = last_vals.copy()
        params["__tlast__"] = np.asarray(tlast)
        cutoff = tlast.max()
        return params, cutoff, _freq_from_pandas(y)

    def _fit_panel_polars(self, pf):
        import polars as pl

        keys, t = pf.key_names, pf.time_name

        def last_valid(c):
            col = pl.col(c)
            ok = col.is_not_null()
            if pf.frame.collect_schema()[c].is_float():
                ok = ok & col.is_not_nan()
            return col.filter(ok).last().alias(c)

        plan = pf.frame.group_by(keys, maintain_order=True).agg(
            pl.col(t).last().alias("__tlast__"),
            # step from the last two stamps of each instance (fixed-step sketch)
            (pl.col(t).last() - pl.col(t).slice(-2, 1).first()).alias("__step__"),
            *[last_valid(c) for c in pf.value_names],
        )
        params = plan.collect() if pf.backend == "polars_lazy" else plan
        cutoff = params["__tlast__"].max()
        step = params["__step__"].min()
        freq = None
        if isinstance(step, (pd.Timedelta, np.timedelta64)) or hasattr(
            step, "total_seconds"
        ):
            freq = pd.tseries.frequencies.to_offset(pd.Timedelta(step))
        params = params.drop("__step__")
        self._panel_step = step
        return params, pd.Timestamp(cutoff) if freq is not None else cutoff, freq

    # ---- predict: params x steps, native ------------------------------------
    def _predict_panel(self, params, steps):
        if self._panel_meta.backend == "pandas":
            return self._predict_panel_pandas(params, steps)
        return self._predict_panel_polars(params, steps)

    def _predict_panel_pandas(self, params, steps):
        y = self._panel_meta.frame
        H = len(steps)
        n = len(params)
        tl = params["__tlast__"]
        vals = params.drop(columns="__tlast__")
        freq = self._panel_freq
        tl_idx = pd.Index(tl.to_numpy())
        if isinstance(tl_idx, (pd.DatetimeIndex, pd.PeriodIndex)) and freq is not None:
            if isinstance(tl_idx, pd.DatetimeIndex) and isinstance(
                freq, pd.offsets.Tick
            ):
                ft = (
                    tl_idx.to_numpy()[:, None] + steps * pd.Timedelta(freq).to_numpy()
                ).ravel()
            else:  # calendar offsets / periods: H vectorized offset additions
                ft = np.stack([(tl_idx + int(k) * freq).to_numpy() for k in steps], 1)
                ft = ft.ravel()
        else:  # integer time
            ft = (tl_idx.to_numpy()[:, None] + steps).ravel()
        key_ix = params.index
        rep = np.repeat(np.arange(n), H)
        if isinstance(key_ix, pd.MultiIndex):
            arrays = [key_ix.get_level_values(i)[rep] for i in range(key_ix.nlevels)]
        else:
            arrays = [key_ix[rep]]
        idx = pd.MultiIndex.from_arrays(arrays + [ft], names=list(y.index.names))
        out = pd.DataFrame(np.repeat(vals.to_numpy(), H, axis=0), index=idx)
        out.columns = vals.columns
        return out

    def _predict_panel_polars(self, params, steps):
        import polars as pl

        pf = self._panel_meta
        hor = pl.DataFrame({"__h__": steps})
        step = self._panel_step
        if isinstance(step, int) or np.issubdtype(type(step), np.integer):
            tnew = pl.col("__tlast__") + pl.col("__h__") * step
        else:
            ns = int(pd.Timedelta(step).value)
            tnew = pl.col("__tlast__") + pl.duration(nanoseconds=pl.col("__h__") * ns)
        out = params.join(hor, how="cross").select(
            *pf.key_names,
            tnew.cast(params.schema["__tlast__"]).alias(pf.time_name),
            *pf.value_names,
        )
        if pf.backend == "polars_lazy":
            return out.lazy()
        return out

    # ---- update: upsert params from new data ---------------------------------
    def _update_panel(self, pf):
        new, cutoff, _ = self._fit_panel(pf)
        old = self._panel_params
        if pf.backend == "pandas":
            merged = new.combine_first(old)[old.columns]
            # keep instance order: old first, then any new instances
            order = old.index.append(new.index.difference(old.index))
            self._panel_params = merged.loc[order]
        else:
            import polars as pl

            keys = pf.key_names
            self._panel_params = pl.concat([old, new]).group_by(
                keys, maintain_order=True
            ).agg(pl.all().last())
        cut = pd.Index([max(self._cutoff[0], pd.Index([cutoff])[0])])
        if isinstance(cut, pd.DatetimeIndex) and self._panel_freq is not None:
            cut = pd.DatetimeIndex(cut, freq=self._panel_freq)
        self._cutoff = cut
        return self
