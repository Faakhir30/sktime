"""x2 prototype: minimal recipe to use narwhals on sktime MultiIndex data
without losing level names, level dtypes, level order, row order or freq.

The idea: the MultiIndex is not something narwhals has to carry. The VectorizedDF
schema already stores the instance keys, the level names and the freq, so we
store the level dtypes too and re-apply all of them at the pandas boundary.

    flat, schema = mi_to_long(df)            # pandas: index levels -> columns
    nf = nw.from_native(flat)                # any narwhals backend from here on
    ... narwhals ops ...
    out = long_to_mi(nf, schema)             # back to the sktime pandas MultiIndex
    for key, s in iter_slices(nf, schema):   # per-instance pandas slices, schema order
        ...

Rules this encodes (each one has an experiment behind it, see reports/x2_*.md):
  R1 move levels to columns with the *native* ``reset_index()`` (or the sktime
     polars adapter). ``nw.maybe_reset_index`` is ``reset_index(drop=True)`` and
     throws the levels away.
  R2 prefix level columns (``__index__<name>``, the convention sktime's polars
     mtypes already use) so they can never collide with data columns. If index
     names equal column names, narwhals' pandas group_by silently drops the index.
  R3 restore names, order and dtypes from the schema, not from what comes back.
     Aggregations, joins and unpivot always return a RangeIndex.
  R4 restore freq from the schema. pandas does not keep freq through groupby or
     set_index either. The current code re-infers it for every slice.
  R5 iterate in schema order. narwhals group_by iteration order is backend
     dependent: pandas gives first-appearance order, polars gives an arbitrary one.
  R6 Period is pandas-only. narwhals reports Unknown. On polars it is an extension
     type that cannot be filtered, diffed or used with .dt, so it is converted to
     a timestamp before the data leaves pandas and converted back at the end.
"""

from dataclasses import dataclass, field

import narwhals as nw
import numpy as np
import pandas as pd

PREFIX = "__index__"


@dataclass
class IndexSchema:
    names: list  # original level names (may contain None)
    cols: list  # column names used in the long frame
    dtypes: list  # original per-level pandas dtypes
    freq: object = None  # freq of the time level (DateOffset) or None
    period_freq: object = None  # set if the time level was a PeriodIndex
    data_cols: list = field(default_factory=list)
    row_index: object = None  # instance keys in original order (like VectorizedDF)
    time_grid: object = None  # regular DatetimeIndex (with freq) covering all times

    @property
    def inst_cols(self):
        return self.cols[:-1]

    @property
    def time_col(self):
        return self.cols[-1]


def _time_freq(df):
    ix = df.index
    if not isinstance(ix, pd.MultiIndex):
        return getattr(ix, "freq", None)
    lvl = ix.levels[-1]
    if getattr(lvl, "freq", None) is not None:
        return lvl.freq
    first = df.xs(ix.droplevel(-1)[0]).index
    if getattr(first, "freq", None) is not None:
        return first.freq
    if isinstance(first, pd.DatetimeIndex) and len(first) > 2:
        f = pd.infer_freq(first)
        return None if f is None else pd.tseries.frequencies.to_offset(f)
    return None


def mi_to_long(df, period_to_timestamp=False):
    """pandas MultiIndex frame -> (flat pandas frame, IndexSchema). Data not copied under CoW."""
    names = list(df.index.names)
    cols = [f"{PREFIX}{n if n is not None else i}" for i, n in enumerate(names)]
    dtypes = [df.index.get_level_values(i).dtype for i in range(df.index.nlevels)]
    schema = IndexSchema(
        names=names,
        cols=cols,
        dtypes=dtypes,
        freq=_time_freq(df),
        data_cols=list(df.columns),
        row_index=df.index.droplevel(-1).unique() if df.index.nlevels > 1 else None,
    )
    lvl = df.index.levels[-1] if isinstance(df.index, pd.MultiIndex) else df.index
    if isinstance(lvl, pd.DatetimeIndex) and schema.freq is not None and len(lvl):
        # one regular grid for the whole frame, built once; per-slice time indexes
        # are cut out of it, so they carry freq without per-slice validation (R4)
        grid = pd.date_range(lvl[0], lvl[-1], freq=schema.freq, unit=lvl.unit, name=names[-1])
        if len(grid) <= 4 * len(lvl) + 16:
            schema.time_grid = grid
    flat = df.rename_axis(cols).reset_index()  # R1, R2
    if isinstance(dtypes[-1], pd.PeriodDtype):
        schema.period_freq = dtypes[-1].freq
        if period_to_timestamp:  # R6, needed before handing to polars/arrow
            flat[cols[-1]] = flat[cols[-1]].dt.to_timestamp()
    return flat, schema


def _to_pandas_native(nf):
    native = nw.to_native(nf)
    if isinstance(native, pd.DataFrame):
        return native
    return nf.to_pandas()


def _restore_dtype(values, dtype, schema, is_time):
    """Cast one level back to its original dtype (R3/R6)."""
    ix = pd.Index(values)
    if is_time and schema.period_freq is not None and not isinstance(
        ix.dtype, pd.PeriodDtype
    ):
        return pd.DatetimeIndex(ix).to_period(schema.period_freq)
    if isinstance(dtype, pd.CategoricalDtype):
        # unordered CategoricalDtypes compare equal as *sets*; polars returns the
        # categories in appearance order, so re-code to the original category order
        return pd.CategoricalIndex(ix, categories=dtype.categories, ordered=dtype.ordered)
    if ix.dtype != dtype:
        try:
            return ix.astype(dtype)
        except (TypeError, ValueError):
            return ix
    return ix


def long_to_mi(nf, schema, sort_like_schema=False):
    """narwhals/native long frame -> pandas frame with the original MultiIndex (R3)."""
    if not isinstance(nf, (nw.DataFrame, nw.LazyFrame)):
        nf = nw.from_native(nf)
    if isinstance(nf, nw.LazyFrame):
        nf = nf.collect()
    pdf = _to_pandas_native(nf)
    last = len(schema.cols) - 1
    arrays = [
        _restore_dtype(pdf[c], dt, schema, i == last)
        for i, (c, dt) in enumerate(zip(schema.cols, schema.dtypes))
    ]
    mi = pd.MultiIndex.from_arrays(arrays, names=schema.names)
    data_cols = [c for c in pdf.columns if c not in schema.cols]
    out = pd.DataFrame(
        {c: pdf[c].to_numpy() for c in data_cols}, index=mi, columns=data_cols
    )
    if sort_like_schema and schema.row_index is not None:  # R5
        # instance position from the schema keys, then time within instance
        inst_pos = schema.row_index.get_indexer(mi.droplevel(-1))
        order = np.lexsort((mi.codes[-1], inst_pos))
        out = out.iloc[order]
    return out


def _time_index(values, schema):
    """Build the per-slice time index with schema freq (R4), no infer_freq."""
    tname = schema.names[-1]
    dtype = schema.dtypes[-1]
    if schema.period_freq is not None:
        ix = pd.Index(values)
        if not isinstance(ix.dtype, pd.PeriodDtype):
            ix = pd.DatetimeIndex(ix).to_period(schema.period_freq)
        return ix.rename(tname)
    ix = pd.Index(values)
    if isinstance(ix, pd.DatetimeIndex):
        ix = ix.rename(tname)
        if ix.dtype != dtype:
            try:
                ix = ix.astype(dtype)
            except (TypeError, ValueError):
                pass
        f = schema.freq
        if f is not None and len(ix) > 0:
            grid = schema.time_grid
            if grid is not None and grid.tz == ix.tz and grid.unit == ix.unit:
                a = grid.searchsorted(ix[0])
                cand = grid[a : a + len(ix)]  # slicing keeps freq, O(1)
            else:
                cand = pd.date_range(ix[0], periods=len(ix), freq=f, name=tname, unit=ix.unit)
            if len(cand) == len(ix) and np.array_equal(cand.asi8, ix.asi8):
                ix = cand
        return ix
    return pd.Index(values, name=tname, dtype=dtype)


def iter_slices(nf, schema):
    """Yield (key, pandas slice with time index) per instance, in schema order (R5)."""
    if not isinstance(nf, (nw.DataFrame,)):
        nf = nw.from_native(nf, eager_only=True)
    groups = {}
    for key, g in nf.group_by(schema.inst_cols):
        gn = _to_pandas_native(g)
        ix = _time_index(gn[schema.time_col].array, schema)
        # per-column numpy + one 2D block: ~5x cheaper than gn[cols] / set_index / drop
        vals = np.column_stack([gn[c].to_numpy() for c in schema.data_cols])
        groups[key if len(key) > 1 else key[0]] = pd.DataFrame(
            vals, index=ix, columns=schema.data_cols, copy=False
        )
    for k in schema.row_index:
        yield k, groups[k]
