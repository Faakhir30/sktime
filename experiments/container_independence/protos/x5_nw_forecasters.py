"""x5 prototype: panel-native naive forecasters written in pure narwhals.

All instances are handled at once by window / group_by expressions - there is
no Python loop over series. The only Python loop is over the forecasting
horizon steps (H, typically <= a few dozen), each step producing one
union-part of the output.

Strategies (match sktime ``NaiveForecaster`` for out-of-sample integer fh):

* ``last``  (sp=1)   : last non-missing value
* ``last``  (sp>1)   : last non-missing value of the same season
* ``mean``  (sp>=1)  : mean of the last ``window_length`` values (per season)
* ``drift``          : line through first and last value of the window
* ``ses``            : simple exponential smoothing level with fixed ``alpha``
                       (``ewm_mean(adjust=False)``; pandas / polars only)

Portability rules learnt from ``x5_probe_narwhals`` (narwhals 2.27.1):

* rank per group: ``with_row_index(order_by=keys+time)`` and
  ``max().over(keys)`` instead of ``rank().over()``  (rank: no dask, no pyarrow)
* never put a window expression inside ``filter``: materialize it with
  ``with_columns`` first (dask raises on ``filter(expr.over(...))``)
* conditional aggregations are precomputed as columns, then aggregated with a
  plain ``max()``: ``group_by.agg(when(..).then(..).max())`` silently ignores
  the condition on the pandas and pyarrow backends
* future timestamps: ``cutoff + lit(k * timedelta)`` per step k (works on all
  backends); calendar steps use ``dt.offset_by`` (not on pyarrow/dask)
* foreign helper frames (e.g. a steps table for a cross join) are avoided:
  duckdb refuses to join relations from different connections
"""

from __future__ import annotations

from datetime import timedelta

import narwhals as nw

from protos.x5_narwhals import long_to_pandas, prepare_nw_vectorized

RI, RMAX, R, CUT, S, N = "__ri", "__rmax", "__r", "__cut", "__s", "__n"
ORD = "__ord"


def _over(expr, keys):
    return expr.over(*keys) if keys else expr


def _group_broadcast(f, keys, exprs):
    """Add per-group aggregates ``exprs`` (aliased) as columns, broadcast to rows.

    polars / duckdb / dask / pandas / ibis: ``expr.over(*keys)`` (a window).
    pyarrow: narwhals 2.27.1 implements ``over`` as group_by + ``pa.Table.join``
    and returns the joined COLUMN, but the pyarrow hash join does not keep the
    left row order -> values land on the wrong rows, silently (see
    scripts/x5_pa_over_bug.py). Workaround: join the aggregate table back as
    whole ROWS (row order may change, row content stays aligned).
    """
    if not keys:
        return f.with_columns(exprs)
    if f.implementation.is_pyarrow():
        agg = f.group_by(*keys).agg(exprs)
        return f.join(agg, on=keys, how="left")
    return f.with_columns([e.over(*keys) for e in exprs])


def freq_to_step(freq):
    """Pandas-style freq -> ("int", n) | ("td", timedelta) | ("cal", unit, n)."""
    import pandas as pd

    if freq is None:
        raise ValueError("cannot generate future time points without a frequency")
    if isinstance(freq, int):
        return ("int", freq)
    off = pd.tseries.frequencies.to_offset(freq)
    if isinstance(off, pd.offsets.Tick):
        return ("td", pd.Timedelta(off).to_pytimedelta())
    n = off.n
    # pandas 3: Day is a calendar offset, no longer a Tick (DST-aware). For
    # tz-naive timestamps it is still exactly 24h.
    if isinstance(off, pd.offsets.Day):
        return ("td", timedelta(days=n))
    # month/quarter/year END anchors: shift to the 1st of next month, offset,
    # shift back (offset_by alone maps Feb-28 + 1mo -> Mar-28, not Mar-31)
    if isinstance(off, pd.offsets.MonthEnd):
        return ("cal_end", "mo", n)
    if isinstance(off, pd.offsets.QuarterEnd):
        return ("cal_end", "q", n)
    if isinstance(off, pd.offsets.YearEnd):
        return ("cal_end", "y", n)
    if isinstance(off, pd.offsets.MonthBegin):
        return ("cal", "mo", n)
    if isinstance(off, pd.offsets.QuarterBegin):
        return ("cal", "q", n)
    if isinstance(off, pd.offsets.YearBegin):
        return ("cal", "y", n)
    if isinstance(off, pd.offsets.Week) and off.weekday is not None:
        return ("td", timedelta(weeks=n))
    raise NotImplementedError(f"freq {freq!r} not expressible in narwhals")


def future_time(step, k):
    """Expression: cutoff + k steps."""
    kind = step[0]
    if kind == "int":
        return nw.col(CUT) + k * step[1]
    if kind == "td":
        return nw.col(CUT) + nw.lit(step[1] * k)
    _, unit, n = step
    if kind == "cal_end":
        one = nw.lit(timedelta(days=1))
        return (nw.col(CUT) + one).dt.offset_by(f"{n * k}{unit}") - one
    return nw.col(CUT).dt.offset_by(f"{n * k}{unit}")


class NwPanelNaive:
    """Panel-native naive forecaster on any narwhals backend.

    ``fit`` records a (possibly lazy) narwhals plan of the fitted state - one
    row per series (or per series x season); nothing is collected for lazy
    inputs unless ``materialize_state=True``. ``predict`` returns the same kind
    of object as the input: pandas MI frame for pandas MI input, polars for
    polars, ``polars.LazyFrame`` / dask / duckdb relation (NOT collected) for
    lazy inputs.
    """

    def __init__(
        self,
        strategy="last",
        sp=1,
        window_length=None,
        alpha=None,
        materialize_state=False,
    ):
        self.strategy = strategy
        self.sp = sp or 1
        self.window_length = window_length
        self.alpha = alpha
        self.materialize_state = materialize_state

    # ------------------------------------------------------------------ fit
    def fit(self, y, keys=None, time=None, freq=None):
        schema, f = prepare_nw_vectorized(
            y, keys=keys, time=time, infer_freq=freq is None
        )
        self.schema_ = schema
        self.freq_ = freq if freq is not None else schema.freq
        self.keys_ = list(schema.keys)
        self.time_ = schema.time
        self.vcols_ = list(schema.value_cols)
        # eager inputs: remember first-appearance order of instances so that the
        # output order matches sktime (pandas groupby(sort=False)); lazy inputs
        # have no row order -> output sorted by keys
        self.order_ = None
        if isinstance(f, nw.DataFrame) and self.keys_:
            self.order_ = (
                f.select(self.keys_).unique(maintain_order=True).with_row_index(ORD)
            )
        state = self._fit_state(f)
        if self.materialize_state and isinstance(state, nw.LazyFrame):
            state = state.collect(schema._collect_backend()).lazy(
                backend=schema.backend
            )
        self.state_ = state
        return self

    def _ranked(self, f):
        """Add __r (1 = last obs per series), __cut (cutoff per series)."""
        keys, t = self.keys_, self.time_
        f = f.with_row_index(RI, order_by=[*keys, t])
        f = _group_broadcast(
            f, keys, [nw.col(RI).max().alias(RMAX), nw.col(t).max().alias(CUT)]
        )
        f = f.with_columns((nw.col(RMAX) - nw.col(RI) + 1).alias(R))
        # NaN -> null so that "missing" means one thing on every backend
        flt = [c for c in self.vcols_ if f.collect_schema()[c].is_float()]
        if flt:
            f = f.with_columns([nw.col(c).fill_nan(None) for c in flt])
        return f

    def _fit_state(self, f):
        keys, vcols, sp = self.keys_, self.vcols_, self.sp
        strat = self.strategy
        f = self._ranked(f)
        # derive all columns BEFORE filtering: on dask, with_columns after a
        # filter that follows with_row_index(order_by) raises
        # "cannot reindex on an axis with duplicate labels"
        f = f.with_columns(((nw.col(R) - 1) % sp).alias(S))
        if strat in ("mean", "drift") and self.window_length is not None:
            f = f.filter(nw.col(R) <= self.window_length)
        part = [*keys, S]

        if strat == "last":
            # last non-missing value per (series, season)
            f = f.with_columns(
                [
                    nw.when(nw.col(v).is_null())
                    .then(None)
                    .otherwise(nw.col(R))
                    .alias(f"__rv_{v}")
                    for v in vcols
                ]
            )
            f = _group_broadcast(
                f, part, [nw.col(f"__rv_{v}").min().alias(f"__rm_{v}") for v in vcols]
            )
            f = f.with_columns(
                [
                    nw.when(nw.col(R) == nw.col(f"__rm_{v}"))
                    .then(nw.col(v))
                    .otherwise(None)
                    .alias(f"__val_{v}")
                    for v in vcols
                ]
            )
            return f.group_by(*part).agg(
                nw.col(CUT).max(), *[nw.col(f"__val_{v}").max().alias(v) for v in vcols]
            )

        if strat == "mean":
            return f.group_by(*part).agg(
                nw.col(CUT).max(), *[nw.col(v).mean() for v in vcols]
            )

        if strat == "drift":
            f = _group_broadcast(f, keys, [nw.col(R).max().alias(N)])
            f = f.with_columns(
                *[
                    nw.when(nw.col(R) == nw.col(N)).then(nw.col(v)).otherwise(None).alias(
                        f"__first_{v}"
                    )
                    for v in vcols
                ],
                *[
                    nw.when(nw.col(R) == 1).then(nw.col(v)).otherwise(None).alias(
                        f"__last_{v}"
                    )
                    for v in vcols
                ],
            )
            g = f.group_by(*keys).agg(
                nw.col(CUT).max(),
                nw.col(N).max(),
                *[nw.col(f"__first_{v}").max() for v in vcols],
                *[nw.col(f"__last_{v}").max().alias(v) for v in vcols],
            )
            return g.with_columns(
                [
                    ((nw.col(v) - nw.col(f"__first_{v}")) / (nw.col(N) - 1)).alias(
                        f"__slope_{v}"
                    )
                    for v in vcols
                ]
            )

        if strat == "ses":
            a = self.alpha
            f = f.with_columns(
                [
                    nw.col(v)
                    .ewm_mean(alpha=a, adjust=False)
                    .over(*keys, order_by=self.time_)
                    .alias(v)
                    for v in vcols
                ]
            )
            # __r is a plain column here, so filtering on it is portable
            return f.filter(nw.col(R) == 1).select(*keys, CUT, *vcols)

        raise ValueError(f"unknown strategy {strat}")

    # -------------------------------------------------------------- predict
    def predict_long(self, fh):
        """Narwhals long frame (keys, time, values) of forecasts; not collected."""
        fh = [int(k) for k in fh]
        step = freq_to_step(self.freq_)
        keys, t, vcols, sp = self.keys_, self.time_, self.vcols_, self.sp
        st = self.state_
        parts = []
        for k in fh:
            p = st
            if self.strategy in ("last", "mean") and sp > 1:
                p = p.filter(nw.col(S) == (-k) % sp)
            if self.strategy == "drift":
                vals = [(nw.col(v) + k * nw.col(f"__slope_{v}")).alias(v) for v in vcols]
            else:
                vals = [nw.col(v) for v in vcols]
            parts.append(p.select(*keys, future_time(step, k).alias(t), *vals))
        out = nw.concat(parts, how="vertical")
        if self.order_ is not None:
            return out.join(self.order_, on=keys, how="left").sort(ORD, t).drop(ORD)
        return out.sort(*keys, t)

    def predict(self, fh):
        out = self.predict_long(fh)
        meta = self.schema_.pandas_meta
        if meta is not None:
            native = out.to_native()
            return long_to_pandas(native.reset_index(drop=True), meta)
        return out.to_native()
