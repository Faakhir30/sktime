"""g2: three forecasters ported to the proto "np_grouped" inner mtype.

* ``G2NaiveForecaster``          last / mean / drift, ``window_length``; batched over
                                 all instances (indptr reductions) for sp == 1,
                                 per-instance loop on views for sp > 1 (points only).
                                 predict_interval (sp == 1) batched, same formulas
                                 as ``NaiveForecaster._predict_var``.
* ``G2PolynomialTrendForecaster`` default regressor: batched QR least squares over
                                 equal-length instances (per-instance lstsq on views
                                 for ragged); custom regressor: per-instance sklearn
                                 clone on views. predict_interval as stock.
* ``G2RecursiveReducer``         ``make_reduction(est, strategy="recursive")`` local
                                 pooling, per-series sklearn clone on zero-copy
                                 ``sliding_window_view`` windows.

None of ``_fit`` / ``_predict`` / ``_predict_interval`` touch pandas except
``_predict_interval``'s output (proba outputs are pandas types in the public API).
The fh -> per-instance step grid uses only the H-element ``fh``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from protos.g2_np_grouped import (
    NPG_MTYPES,
    fh_to_grid,
    npg_append,
    pandas_row_index,
    register_np_grouped,
    time_as_float_days,
)
from sktime.forecasting.base import BaseForecaster

register_np_grouped()

_NPG_TAGS = {
    "y_inner_mtype": NPG_MTYPES,  # opt-in: the only change visible to the framework
    "X_inner_mtype": "pd.DataFrame",
    "capability:exogenous": False,
    "capability:multivariate": True,  # kernels are column-wise
    "requires-fh-in-fit": False,
}


class _G2Base(BaseForecaster):
    """Shared: grid of (instance x horizon), output hub, proba output frame."""

    def _grid(self, fh):
        K, T = fh_to_grid(self._npg, fh, self.cutoff)
        if (K < 1).any():
            raise NotImplementedError("g2 proto: in-sample fh not ported")
        return K, T

    def _out(self, T, Y):
        """T (U,H) times, Y (U,H,C) values -> NpGrouped of predictions."""
        y = self._npg
        U, H = T.shape
        out = y.with_rows(np.arange(U + 1, dtype=np.int64) * H,
                          np.ascontiguousarray(T).reshape(-1),
                          Y.reshape(U * H, Y.shape[2]))
        if not y.key_names:  # stock: single-series prediction index is unnamed
            out.time_name = None
        return out

    def _update(self, y, X=None, update_params=True):
        """Stock window forecasters keep fitted params and slide the data window."""
        self._npg = (self._y if self.get_config()["remember_data"]
                     else npg_append(self._npg, y))
        if update_params and self._refit_on_update:
            self._fit(self._npg, X, self._fh)
        return self

    def _interval_frame(self, T, mean, var, coverage):
        """pandas output of predict_interval (public API type), built once."""
        import pandas as pd
        from scipy.stats import norm

        idx = pandas_row_index(self._out(T, mean))
        cols, data = [], []
        names = self._get_varnames()
        for j, name in enumerate(names):
            sd = np.sqrt(var[..., j]).reshape(-1)
            m = mean[..., j].reshape(-1)
            for c in coverage:
                for side, a in (("lower", 0.5 - c / 2), ("upper", 0.5 + c / 2)):
                    cols.append((name, c, side))
                    data.append(m + sd * norm.ppf(a))
        return pd.DataFrame(np.column_stack(data), index=idx,
                            columns=pd.MultiIndex.from_tuples(cols))


# =============================================================================
# NaiveForecaster
# =============================================================================
def _seg_last_valid(v, ip, ws):
    pos = np.where(np.isnan(v), -1, np.arange(len(v)))
    last = np.maximum.reduceat(pos, ip[:-1])
    return np.where(last >= ws, v[np.clip(last, 0, None)], np.nan)


def _seg_nanmean(v, ws, ends):
    b = np.empty(2 * len(ws), dtype=np.int64)
    b[0::2], b[1::2] = ws, ends
    vz = np.append(np.nan_to_num(v), 0.0)
    cnt = np.append((~np.isnan(v)).astype(np.int64), 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.add.reduceat(vz, b)[0::2] / np.add.reduceat(cnt, b)[0::2]


class G2NaiveForecaster(_G2Base):
    _tags = {**_NPG_TAGS, "capability:pred_int": True,
             "capability:missing_values": True}
    _refit_on_update = False

    def __init__(self, strategy="last", window_length=None, sp=1):
        self.strategy = strategy
        self.window_length = window_length
        self.sp = sp
        super().__init__()

    def _fit(self, y, X, fh):
        if self.strategy not in ("last", "mean", "drift"):
            raise ValueError(self.strategy)
        lens = y.lengths
        wl = self.window_length
        if wl is not None and (lens < wl).any():
            raise ValueError("window_length larger than a training series")
        if self.strategy == "drift" and wl == 1:
            raise ValueError("drift needs window_length > 1")
        self._npg = y  # a reference, no copy (stock: _y + one clone per instance)
        # stock: window_length_ = len(y) at fit, frozen on update
        self.window_length_ = y.lengths.copy() if wl is None else wl
        return self

    def _point(self, K):
        y, sp, s = self._npg, self.sp or 1, self.strategy
        ip, ends, lens = y.indptr, y.indptr[1:], y.lengths
        wl = np.minimum(self.window_length_, lens)
        ws = ends - wl
        U, H = K.shape
        C = y.values.shape[1]
        Y = np.empty((U, H, C))
        for j in range(C):
            v = y.values[:, j]
            if sp > 1 and s in ("last", "mean"):  # per-instance loop on views
                Y[:, :, j] = [self._seasonal(v[a:b], w, K[i])
                              for i, (a, b, w) in enumerate(zip(ws, ends, wl))]
            elif s == "last":
                Y[:, :, j] = _seg_last_valid(v, ip, ws)[:, None]
            elif s == "mean":
                Y[:, :, j] = _seg_nanmean(v, ws, ends)[:, None]
            else:  # drift
                last, first = v[ends - 1], v[ws]
                slope = (last - first) / (wl - 1)
                Y[:, :, j] = last[:, None] + K * slope[:, None]
        return Y

    def _seasonal(self, lw, wl, k):
        sp = self.sp
        if np.all(np.isnan(lw)):
            return np.full(len(k), np.nan)
        pad = (sp - wl % sp if wl % sp else 0) + wl - len(lw)
        w = np.concatenate([np.full(pad, np.nan), lw]).reshape(-1, sp)
        if self.strategy == "last":
            yp = w[(~np.isnan(w)).cumsum(0).argmax(0), np.arange(sp)]
        else:
            yp = np.nanmean(w, axis=0)
        reps = int(np.ceil(k.max() / sp)) if k.max() > sp else 1
        return np.tile(yp, reps)[k - 1]

    def _predict(self, fh, X=None):
        K, T = self._grid(fh)
        return self._out(T, self._point(K))

    def _predict_interval(self, fh, X=None, coverage=None):
        if (self.sp or 1) != 1:
            raise NotImplementedError("g2 proto: intervals ported for sp == 1")
        K, T = self._grid(fh)
        mean = self._point(K)
        y = self._npg
        ip, n = y.indptr, y.n_rows
        lens = y.lengths.astype(float)
        inst = np.repeat(np.arange(len(lens)), y.lengths)
        starts = ip[:-1]
        s, wlp = self.strategy, self.window_length
        var = np.empty_like(mean)
        for j in range(mean.shape[2]):
            v = y.values[:, j]
            prev = np.empty(n)
            prev[1:] = v[:-1]
            prev[starts] = np.nan
            if s == "last":
                res = v - prev
            elif s == "mean" and wlp is None:
                res = v - np.repeat(_seg_nanmean(v, starts, ip[1:]), y.lengths)
            elif s == "mean":
                from numpy.lib.stride_tricks import sliding_window_view

                rm = np.full(n, np.nan)
                if n >= wlp:
                    rm[wlp - 1:] = sliding_window_view(v, wlp).mean(axis=1)
                rm[(np.arange(n) - starts[inst]) < wlp - 1] = np.nan
                res = v - rm
            else:  # drift: same (quirky) slope as stock: y[-wl or 0] over T-1
                first = v[ip[1:] - wlp] if wlp else v[starts]
                slope = (v[ip[1:] - 1] - first) / (lens - 1)
                res = v - (prev + slope[inst])
            nn = np.bincount(inst, weights=np.isnan(res), minlength=len(lens))
            ss = np.bincount(inst, weights=np.nan_to_num(res) ** 2,
                             minlength=len(lens))
            se = np.sqrt(ss / (lens - nn - (s == "drift")))
            h = K.astype(float)
            if s == "last":
                f = np.sqrt(np.maximum(h, 1))
            elif s == "mean":
                w = float(wlp) if wlp else lens[:, None]
                f = np.broadcast_to(np.sqrt(np.maximum(1 + 1 / w, 1)), h.shape)
            else:
                Tm1 = (lens - 1)[:, None]
                f = np.sqrt(np.maximum(h * (1 + h / Tm1), 1))
            var[:, :, j] = (se[:, None] * f) ** 2
        return self._interval_frame(T, mean, var, coverage)


# =============================================================================
# PolynomialTrendForecaster
# =============================================================================
def _lstsq_cond(A, Y, cond):
    """Batched scipy.linalg.lstsq(A, Y, cond=cond): A (U,T,p), Y (U,T,C)."""
    Uu, S, Vt = np.linalg.svd(A, full_matrices=False)
    keep = S > cond * S[:, :1]
    Sinv = np.where(keep, 1.0 / np.where(keep, S, 1.0), 0.0)
    return Vt.transpose(0, 2, 1) @ (Sinv[:, :, None] * (Uu.transpose(0, 2, 1) @ Y))


class G2PolynomialTrendForecaster(_G2Base):
    _tags = {**_NPG_TAGS, "capability:pred_int": True,
             "capability:missing_values": False}
    _refit_on_update = True

    def __init__(self, regressor=None, degree=1, with_intercept=True,
                 prediction_intervals=False):
        self.regressor = regressor
        self.degree = degree
        self.with_intercept = with_intercept
        self.prediction_intervals = prediction_intervals
        super().__init__()
        self.set_tags(**{"capability:pred_int": prediction_intervals})

    tol = LinearRegression().tol  # default regressor's lstsq cutoff

    def _design(self, x):
        V = np.polynomial.polynomial.polyvander(x, self.degree)
        return V if self.with_intercept else V[..., 1:]

    def _fit(self, y, X, fh):
        self._npg = y
        x = time_as_float_days(y, y.time)
        lens, ip = y.lengths, y.indptr
        C = y.values.shape[1]
        self._p = self.degree + int(self.with_intercept)
        if self.regressor is not None:  # generic: per-instance sklearn on views
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import PolynomialFeatures

            self.regressors_ = [[make_pipeline(
                PolynomialFeatures(self.degree, include_bias=self.with_intercept),
                clone(self.regressor)).fit(x[a:b, None], y.values[a:b, j])
                for j in range(C)] for a, b in zip(ip[:-1], ip[1:])]
            return self
        self.regressors_ = None
        # default regressor = LinearRegression(fit_intercept=False); sklearn solves
        # it with scipy lstsq(cond=tol): singular values < tol * s_max are dropped.
        # Replicated exactly (batched SVD), else results differ from stock.
        tol = self.tol
        if (lens == lens[0]).all():  # batched over (U, T, p)
            U, Tn = len(lens), lens[0]
            A = self._design(x.reshape(U, Tn))
            Yv = y.values.reshape(U, Tn, C)
            self.coef_ = _lstsq_cond(A, Yv, tol)  # (U, p, C)
            self._R = np.linalg.qr(A)[1]
            if self.prediction_intervals:
                self.s2_ = ((Yv - A @ self.coef_) ** 2).sum(1) / (Tn - self._p)
        else:  # ragged: per-instance on views
            coefs, Rs, s2 = [], [], []
            for a, b in zip(ip[:-1], ip[1:]):
                A = self._design(x[a:b])
                c = _lstsq_cond(A[None], y.values[None, a:b], tol)[0]
                coefs.append(c)
                Rs.append(np.linalg.qr(A)[1])
                s2.append(((y.values[a:b] - A @ c) ** 2).sum(0) / (b - a - self._p))
            self.coef_, self._R, self.s2_ = np.stack(coefs), np.stack(Rs), np.stack(s2)
        return self

    def _predict(self, fh, X=None):
        K, T = self._grid(fh)
        return self._out(T, self._mean(T))

    def _mean(self, T):
        xf = time_as_float_days(self._npg, T)  # (U, H)
        if self.regressors_ is not None:
            return np.stack([np.column_stack([r.predict(xf[i, :, None]) for r in rs])
                             for i, rs in enumerate(self.regressors_)])
        return self._design(xf) @ self.coef_  # (U, H, p) @ (U, p, C)

    def _predict_interval(self, fh, X=None, coverage=None):
        if not self.prediction_intervals:
            raise ValueError("Set prediction_intervals=True at initialization.")
        K, T = self._grid(fh)
        mean = self._mean(T)
        W = self._design(time_as_float_days(self._npg, T))  # (U, H, p)
        # w^T (X^T X)^-1 w = || R^-T w ||^2
        Z = np.linalg.solve(self._R.transpose(0, 2, 1), W.transpose(0, 2, 1))
        v = 1 + (Z ** 2).sum(1)  # (U, H)
        var = v[:, :, None] * self.s2_[:, None, :]
        return self._interval_frame(T, mean, var, coverage)


# =============================================================================
# RecursiveTabularRegressionForecaster (make_reduction, local pooling)
# =============================================================================
class G2RecursiveReducer(_G2Base):
    _tags = {**_NPG_TAGS, "capability:pred_int": False,
             "capability:missing_values": False}
    _refit_on_update = False

    def __init__(self, estimator=None, window_length=10):
        self.estimator = estimator
        self.window_length = window_length
        super().__init__()

    def _fit(self, y, X, fh):
        from numpy.lib.stride_tricks import sliding_window_view

        self._npg = y
        wl, ip = self.window_length, y.indptr
        est = LinearRegression() if self.estimator is None else self.estimator
        if (y.lengths <= wl + 1).any():  # stock: window_length + 1 >= n raises
            raise ValueError("The `window_length` and `fh` are incompatible with y")
        self.estimators_ = []
        for a, b in zip(ip[:-1], ip[1:]):
            row = []
            for j in range(y.values.shape[1]):
                z = sliding_window_view(y.values[a:b, j], wl + 1)  # n - wl rows
                row.append(clone(est).fit(z[:, :wl], z[:, wl]))  # same rows as stock
            self.estimators_.append(row)
        return self

    def _predict(self, fh, X=None):
        K, T = self._grid(fh)
        y, wl = self._npg, self.window_length
        hmax = int(K.max())
        U, H = K.shape
        C = y.values.shape[1]
        Y = np.full((U, H, C), np.nan)
        for i, (row, b) in enumerate(zip(self.estimators_, y.indptr[1:])):
            for j, est in enumerate(row):
                last = np.empty(wl + hmax)
                last[:wl] = y.values[b - wl:b, j]
                if not np.isfinite(last[:wl]).all():
                    continue  # stock: _predict_nan
                for s in range(hmax):
                    last[wl + s] = est.predict(last[s:wl + s].reshape(1, -1))[0]
                Y[i, :, j] = last[wl + K[i] - 1]
        return self._out(T, Y)
