"""x6: hub-and-spoke canonical internal representation for panel time series.

Prototype only, not part of sktime. Cf. Nixtla ``GroupedArray`` (utilsforecast /
mlforecast: ``data`` 2D array + ``indptr``) and Darts ``TimeSeries`` (numpy/xarray
values + time index + static covariates).

Hub
---
``PanelData`` is a numpy-only, library-neutral container:

* ``indptr``     int64 offsets, length ``n_instances + 1``; instance ``i`` is rows
                 ``indptr[i]:indptr[i+1]`` (rows of one instance are contiguous)
* ``time``       per-row time array: ``datetime64[unit]`` (UTC if tz-aware),
                 or int64 (period ordinals, or plain integer time)
* ``values``     2D float array ``(n_rows, n_cols)``; may be a read-only *view* of the
                 source buffer (zero-copy ingress)
* ``columns``    value column names
* ``key_names`` / ``key_values``  instance key table: one 1D array per instance level,
                 each of length ``n_instances`` (small: O(n_instances))
* ``aux``        non-float columns (strings, categoricals), 1D arrays, row aligned
* ``tmeta``      time semantics: kind, unit, tz, pandas freq string, numeric step
* ``meta``       everything needed to rebuild the *exact* native container
                 (source library, layout, column order, native dtypes, row permutation
                 if the source was not grouped contiguously, npartitions, ...)

Spokes
------
Style (i), hand written, one per library: ``from_pandas``/``to_pandas`` (MultiIndex,
plain index, pd.Series), ``from_pandas_long``/``to_pandas_long`` (keys as columns),
``from_polars``/``to_polars`` (eager + lazy, natural layout or sktime ``__index__``
convention), ``from_arrow``/``to_arrow``, ``from_dask``/``to_dask``.

Style (ii-a), narwhals: ``nw_from_native``/``nw_to_native`` (one generic code path).
Style (ii-b), Arrow PyCapsule ingress: ``capsule_from_native``/``capsule_to_native``
(``pa.table(obj)`` on anything exposing ``__arrow_c_stream__``, egress via
``narwhals.from_arrow`` for eager targets).

Forecaster side
---------------
``SeriesView`` (one instance; views into hub arrays), ``SeriesView.to_pandas``
(lazy pandas materialization for pandas-semantic estimators), ``future_time``
(fh arithmetic without pandas for int/period/fixed-step datetime), ``HubNaive``
(per-slice numpy kernel mirroring sktime ``NaiveForecaster`` math), and
``hub_naive_batched`` (all instances at once with ``indptr`` reductions).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

INDEX_PREFIX = "__index__"  # sktime polars / dask convention

# =============================================================================
# hub
# =============================================================================


@dataclass
class TimeMeta:
    """Time semantics of the hub's ``time`` array."""

    kind: str  # "datetime" | "period" | "int"
    unit: str | None = None  # datetime unit, e.g. "ns", "us"
    tz: str | None = None  # tz name; time stored as naive UTC datetime64
    freq: str | None = None  # pandas freqstr (datetime/period), None if unknown
    step: object = None  # np.timedelta64 (fixed dt step) / int (period, int) / None
    is_range: bool = False  # int time that came from a RangeIndex (single series)


@dataclass
class SeriesView:
    """One instance slice of a ``PanelData``. All arrays are views (no copy)."""

    key: tuple
    time: np.ndarray
    values: np.ndarray  # 2D (n_t, n_cols)
    columns: list
    tmeta: TimeMeta
    time_name: object = None

    def to_pandas(self, as_series=False, freq_mode="trusted"):
        """Lazily materialize a pandas object, for pandas-semantic estimators.

        freq_mode: "none" (no freq), "ctor" (pd.DatetimeIndex(..., freq=f), validated),
        "trusted" (freq set from hub metadata without re-validation),
        "infer" (pd.infer_freq per slice, = current ``_enforce_index_freq``).
        """
        import pandas as pd

        idx = time_to_pandas_index(self.time, self.tmeta, freq_mode=freq_mode)
        idx.name = self.time_name
        if as_series:
            return pd.Series(
                self.values[:, 0], index=idx, name=self.columns[0], copy=False
            )
        return pd.DataFrame(self.values, index=idx, columns=self.columns, copy=False)


@dataclass
class PanelData:
    """Canonical hub container, see module docstring."""

    indptr: np.ndarray
    time: np.ndarray
    values: np.ndarray
    columns: list
    key_names: list
    key_values: list
    time_name: object
    tmeta: TimeMeta
    aux: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # --- shape -------------------------------------------------------------
    @property
    def n_instances(self):
        return len(self.indptr) - 1

    @property
    def n_rows(self):
        return int(self.indptr[-1])

    @property
    def lengths(self):
        return np.diff(self.indptr)

    def key(self, i):
        return tuple(kv[i] for kv in self.key_values)

    # --- slicing (views) ---------------------------------------------------
    def slice(self, i):
        a, b = self.indptr[i], self.indptr[i + 1]
        return SeriesView(
            key=self.key(i),
            time=self.time[a:b],
            values=self.values[a:b],
            columns=self.columns,
            tmeta=self.tmeta,
            time_name=self.time_name,
        )

    def iter_slices(self):
        indptr, time, values = self.indptr, self.time, self.values
        for i in range(len(indptr) - 1):
            a, b = indptr[i], indptr[i + 1]
            yield SeriesView(
                self.key(i), time[a:b], values[a:b], self.columns, self.tmeta,
                self.time_name,
            )

    # --- reconstruct -------------------------------------------------------
    def with_data(self, indptr, time, values, columns=None):
        """New hub with the same keys + egress metadata, different rows."""
        if columns is None and values.shape[1] != len(self.columns):
            columns = self.columns[: values.shape[1]]
        meta = dict(self.meta)
        meta["perm"] = None  # rows are new, no original order to restore
        meta["aux_order"] = None
        meta["col_dtypes"] = None if columns is not None else meta.get("col_dtypes")
        return PanelData(
            indptr=np.asarray(indptr, dtype=np.int64),
            time=time,
            values=values,
            columns=list(self.columns if columns is None else columns),
            key_names=self.key_names,
            key_values=self.key_values,
            time_name=self.time_name,
            tmeta=self.tmeta,
            aux={},
            meta=meta,
        )

    def from_outputs(self, outputs, columns=None):
        """Reconstruct from per-instance ``(time, values)`` outputs (one concat)."""
        lens = np.fromiter((len(t) for t, _ in outputs), dtype=np.int64,
                           count=len(outputs))
        indptr = np.zeros(len(outputs) + 1, dtype=np.int64)
        np.cumsum(lens, out=indptr[1:])
        time = np.concatenate([t for t, _ in outputs])
        vals = np.concatenate([np.asarray(v).reshape(len(v), -1) for _, v in outputs])
        return self.with_data(indptr, time, vals, columns=columns)

    def nbytes(self):
        n = self.indptr.nbytes + self.time.nbytes + self.values.nbytes
        n += sum(np.asarray(a).nbytes for a in self.aux.values())
        return n


# =============================================================================
# shared numpy core (library neutral)
# =============================================================================


def _run_starts_from_codes(code_arrays, n):
    """Start offsets of runs of equal composite key; code_arrays: list of 1D ints."""
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if not code_arrays:
        return np.zeros(1, dtype=np.int64)
    chg = np.empty(n, dtype=bool)
    chg[0] = True
    c = code_arrays[0]
    np.not_equal(c[1:], c[:-1], out=chg[1:])
    for c in code_arrays[1:]:
        chg[1:] |= c[1:] != c[:-1]
    return np.flatnonzero(chg)


def _composite(code_arrays):
    """Combine per-level integer codes into one int64 code (mixed radix)."""
    if len(code_arrays) == 1:
        return np.asarray(code_arrays[0], dtype=np.int64)
    dims = [int(np.max(c)) + 2 for c in code_arrays]  # +2: codes may be -1 (NaN)
    return np.ravel_multi_index([np.asarray(c, np.int64) + 1 for c in code_arrays], dims)


def _first_appearance_perm(code_arrays):
    """Stable permutation grouping rows by composite key, groups in first-seen order."""
    comb = _composite(code_arrays)
    _, first_idx, inv = np.unique(comb, return_index=True, return_inverse=True)
    rank = np.empty(len(first_idx), dtype=np.int64)
    rank[np.argsort(first_idx, kind="stable")] = np.arange(len(first_idx))
    return np.argsort(rank[inv], kind="stable")


def _starts_to_indptr(starts, n):
    indptr = np.empty(len(starts) + 1, dtype=np.int64)
    indptr[:-1] = starts
    indptr[-1] = n
    return indptr


def _stack_cols(cols, n):
    """1D columns -> 2D (n, k). Zero-copy for k == 1 (reshape view), else 1 copy."""
    if len(cols) == 1:
        return np.asarray(cols[0]).reshape(n, 1)
    out = np.empty((n, len(cols)), dtype=np.float64, order="F")
    for j, c in enumerate(cols):
        out[:, j] = c
    return out


_FIXED_FREQ = {
    np.timedelta64(1, "D"): "D",
    np.timedelta64(1, "h"): "h",
    np.timedelta64(1, "m"): "min",
    np.timedelta64(1, "s"): "s",
    np.timedelta64(7, "D"): "W",
}


def infer_step_numpy(t, kind):
    """Library-neutral step/freq inference from one instance's time array."""
    if len(t) < 2:
        return None, None
    d = np.diff(t)
    if kind != "datetime":
        return (int(d[0]), None) if (d == d[0]).all() else (None, None)
    if (d == d[0]).all():
        step = d[0]
        return step, _FIXED_FREQ.get(step)
    # calendar monthly: month start / month end
    months = t.astype("datetime64[M]")
    md = np.diff(months).astype(np.int64)
    if (md == md[0]).all():
        k = "" if md[0] == 1 else str(md[0])
        if (t == months.astype(t.dtype)).all():
            return None, f"{k}MS"
        if (t == (months + 1).astype(t.dtype) - np.timedelta64(1, "D")).all():
            return None, f"{k}ME"
    return None, None


# =============================================================================
# time helpers (pandas only where pandas is the target)
# =============================================================================


_OFFSETS = {}


def _offset(freq):
    import pandas as pd

    off = _OFFSETS.get(freq)
    if off is None:
        off = _OFFSETS[freq] = pd.tseries.frequencies.to_offset(freq)
    return off


def _tm_freq(tm):
    import pandas as pd

    if tm.freq is None and tm.step is not None and tm.kind == "datetime":
        tm.freq = pd.tseries.frequencies.to_offset(pd.Timedelta(tm.step)).freqstr
    return tm.freq


def time_to_pandas_index(t, tm, freq_mode="trusted"):
    """Hub time slice -> pandas index (DatetimeIndex / PeriodIndex / Index).

    freq_mode:
      "public"  pd.DatetimeIndex(t) public constructor, no freq
      "none"    no freq, fast private constructor (no validation, zero-copy)
      "trusted" freq from hub metadata, fast private constructor, no O(n) check
      "ctor"    pd.DatetimeIndex(t, freq=f): public, validates conformity (O(n))
      "infer"   pd.infer_freq per slice (= current ``_enforce_index_freq``)
    """
    import pandas as pd

    if tm.kind == "datetime":
        freq = _tm_freq(tm)
        if tm.tz is None and freq_mode in ("none", "trusted"):
            from pandas.core.arrays import DatetimeArray

            off = _offset(freq) if freq_mode == "trusted" and freq else None
            arr = DatetimeArray._simple_new(t, freq=off, dtype=t.dtype)
            return pd.DatetimeIndex._simple_new(arr, name=None)
        if freq_mode == "ctor" and freq is not None and tm.tz is None and len(t) > 0:
            return pd.DatetimeIndex(t, freq=freq)
        idx = pd.DatetimeIndex(t)
        if tm.tz is not None:
            idx = idx.tz_localize("UTC").tz_convert(tm.tz)
            if freq_mode == "trusted" and freq is not None and len(t) > 0:
                idx = pd.DatetimeIndex(idx, freq=freq)
        elif freq_mode == "infer" and len(t) > 2:
            idx.freq = pd.infer_freq(idx)
        return idx
    if tm.kind == "period":
        return pd.PeriodIndex.from_ordinals(t, freq=tm.freq)
    if tm.is_range and len(t) > 0 and tm.step:
        return pd.RangeIndex(t[0], t[-1] + tm.step, tm.step)
    return pd.Index(t)


def future_time(cutoff, fh, tm):
    """Absolute future time points: cutoff + fh * step, numpy only when possible.

    Works without pandas for int, period (ordinals) and fixed-step datetimes.
    Calendar frequencies (MS, ME, Q, B, ...) need offset arithmetic: delegated to
    pandas here (polars ``dt.offset_by`` would be the polars-native alternative).
    """
    fh = np.asarray(fh, dtype=np.int64)
    if tm.kind in ("int", "period"):
        return cutoff + fh * (tm.step or 1)
    if tm.step is not None:
        return cutoff + fh * tm.step
    import pandas as pd  # calendar offsets

    off = pd.tseries.frequencies.to_offset(tm.freq)
    c = pd.Timestamp(cutoff)
    return np.array([(c + k * off).to_datetime64() for k in fh]).astype(cutoff.dtype)


# =============================================================================
# spoke (i): pandas  (MultiIndex / plain index / pd.Series)
# =============================================================================


def _pd_time_meta(lev_or_idx, first_slice_idx):
    """TimeMeta + converter for a pandas time index (level)."""
    import pandas as pd

    if isinstance(lev_or_idx, pd.DatetimeIndex):
        tz = None if lev_or_idx.tz is None else str(lev_or_idx.tz)
        unit = np.datetime_data(lev_or_idx.values.dtype)[0]
        freq = getattr(first_slice_idx, "freqstr", None)
        if freq is None and len(first_slice_idx) > 2:
            try:
                freq = pd.infer_freq(first_slice_idx)
            except (TypeError, ValueError):
                freq = None
        step = None
        if freq is not None:
            off = pd.tseries.frequencies.to_offset(freq)
            try:  # fixed (Tick) offsets have a timedelta step
                step = np.timedelta64(pd.Timedelta(off)).astype(f"m8[{unit}]")
            except ValueError:
                step = None
        return TimeMeta("datetime", unit=unit, tz=tz, freq=freq, step=step), (
            lambda ix: ix.values if tz is None else ix.tz_convert("UTC").tz_localize(
                None).values
        )
    if isinstance(lev_or_idx, pd.PeriodIndex):
        return TimeMeta("period", freq=lev_or_idx.freqstr, step=1), (
            lambda ix: ix.asi8
        )
    is_range = isinstance(lev_or_idx, pd.RangeIndex)
    t0 = np.asarray(first_slice_idx)
    step = int(t0[1] - t0[0]) if len(t0) > 1 else 1
    return TimeMeta("int", step=step, is_range=is_range), (lambda ix: np.asarray(ix))


def _split_value_cols(dtypes_by_col):
    """Partition columns into float-hub columns and aux columns."""
    num, aux = [], []
    for c, dt in dtypes_by_col.items():
        kind = getattr(dt, "kind", "O")
        (num if kind in "fiub" else aux).append(c)
    return num, aux


def from_pandas(obj):
    """pandas Series/DataFrame (plain index or MultiIndex, time = last level) -> hub."""
    import pandas as pd

    meta = {"source": "pandas"}
    if isinstance(obj, pd.Series):
        meta["container"] = "series"
        meta["series_name"] = obj.name
        obj = obj.to_frame(name=obj.name if obj.name is not None else 0)
    else:
        meta["container"] = "frame"
    idx = obj.index
    n = len(obj)
    meta["columns_order"] = list(obj.columns)
    meta["col_dtypes"] = dict(obj.dtypes.items())
    meta["columns_index_name"] = obj.columns.name

    if isinstance(idx, pd.MultiIndex):
        meta["layout"] = "multiindex"
        codes = [np.asarray(c) for c in idx.codes]
        inst_codes = codes[:-1]
        starts = _run_starts_from_codes(inst_codes, n)
        run_comb = _composite([c[starts] for c in inst_codes])
        perm = None
        if len(np.unique(run_comb)) != len(starts):  # not grouped contiguously
            perm = _first_appearance_perm(inst_codes)
            codes = [c[perm] for c in codes]
            inst_codes = codes[:-1]
            starts = _run_starts_from_codes(inst_codes, n)
        indptr = _starts_to_indptr(starts, n)
        key_values = [
            np.asarray(idx.levels[lv])[inst_codes[lv][starts]]
            for lv in range(len(inst_codes))
        ]
        key_names = list(idx.names[:-1])
        time_name = idx.names[-1]
        tlev = idx.levels[-1]
        first = tlev.take(codes[-1][indptr[0]:indptr[1]]) if n else tlev
        tmeta, conv = _pd_time_meta(tlev, first)
        time = conv(tlev).take(codes[-1])  # one copy, 8 B/row
        meta["key_dtypes"] = [idx.levels[lv].dtype for lv in range(len(inst_codes))]
    else:
        meta["layout"] = "index"
        perm = None
        indptr = np.array([0, n], dtype=np.int64)
        key_values, key_names = [], []
        time_name = idx.name
        tmeta, conv = _pd_time_meta(idx, idx)
        time = conv(idx)  # view of index data where possible

    num, aux_cols = _split_value_cols(meta["col_dtypes"])
    same_float = len({meta["col_dtypes"][c] for c in num}) == 1 and all(
        meta["col_dtypes"][c].kind == "f" for c in num
    )
    if not aux_cols and same_float:
        values = obj.to_numpy(copy=False)  # view of the single float block
    else:
        values = _stack_cols([obj[c].to_numpy(dtype=np.float64) for c in num], n)
    aux = {c: obj[c].to_numpy() for c in aux_cols}
    if perm is not None:
        values = values[perm]
        aux = {c: a[perm] for c, a in aux.items()}
    meta["perm"] = perm
    meta["aux_order"] = aux_cols
    return PanelData(indptr, time, values, num, key_names, key_values, time_name,
                     tmeta, aux, meta)


def to_pandas(hub, restore_order=True):
    """hub -> pandas object of the recorded source layout."""
    import pandas as pd

    m = hub.meta
    n = hub.n_rows
    tm = hub.tmeta
    lengths = hub.lengths
    layout = m.get("layout", "multiindex")
    if layout == "index" or (hub.n_instances == 1 and not hub.key_names):
        index = time_to_pandas_index(hub.time, tm, freq_mode="trusted")
        index.name = hub.time_name
    else:
        levels, codes = [], []
        kdt = m.get("key_dtypes") or [None] * len(hub.key_values)
        for kv, dt in zip(hub.key_values, kdt):
            kc, ku = pd.factorize(kv)
            ku = pd.Index(ku)
            if dt is not None and ku.dtype != dt:
                ku = ku.astype(dt)
            levels.append(ku)
            codes.append(np.repeat(kc, lengths))
        tc, tu = pd.factorize(hub.time, sort=True)
        levels.append(time_to_pandas_index(tu, tm, freq_mode="none"))
        codes.append(tc)
        index = pd.MultiIndex(levels=levels, codes=codes,
                              names=[*hub.key_names, hub.time_name],
                              verify_integrity=False)
    df = pd.DataFrame(hub.values, index=index, columns=hub.columns, copy=False)
    dtypes = m.get("col_dtypes") or {}
    for c in hub.columns:
        if c in dtypes and df[c].dtype != dtypes[c]:
            df[c] = df[c].astype(dtypes[c])
    if hub.aux:
        for c, a in hub.aux.items():
            df[c] = a
        df = df[[c for c in m["columns_order"]]]
    if m.get("columns_index_name") is not None:
        df.columns.name = m["columns_index_name"]
    perm = m.get("perm")
    if restore_order and perm is not None and len(perm) == n:
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        df = df.iloc[inv]
    if m.get("container") == "series":
        s = df.iloc[:, 0]
        s.name = m.get("series_name")
        return s
    return df


# =============================================================================
# spoke (i): pandas long (keys/time as columns) -- used by dask
# =============================================================================


def _detect_long_cols(columns, key_cols, time_col):
    if key_cols is None and time_col is None:
        idx_cols = [c for c in columns if isinstance(c, str) and c.startswith(
            INDEX_PREFIX)]
        if idx_cols:
            return idx_cols[:-1], idx_cols[-1]
        raise ValueError("pass key_cols and time_col (no __index__ columns found)")
    return list(key_cols or []), time_col


def from_pandas_long(df, key_cols=None, time_col=None):
    """pandas frame with key/time columns -> hub."""
    import pandas as pd

    key_cols, time_col = _detect_long_cols(list(df.columns), key_cols, time_col)
    n = len(df)
    kcodes, kuniq = zip(*[pd.factorize(df[k]) for k in key_cols]) if key_cols else (
        (), ())
    kcodes = [np.asarray(c) for c in kcodes]
    starts = _run_starts_from_codes(kcodes, n)
    perm = None
    if kcodes and len(np.unique(_composite([c[starts] for c in kcodes]))) != len(
            starts):
        perm = _first_appearance_perm(kcodes)
        kcodes = [c[perm] for c in kcodes]
        starts = _run_starts_from_codes(kcodes, n)
    indptr = _starts_to_indptr(starts, n)
    key_values = [np.asarray(u)[c[starts]] for c, u in zip(kcodes, kuniq)]
    tcol = df[time_col]
    tix = pd.Index(tcol)
    first = tix[indptr[0]:indptr[1]] if n else tix
    tmeta, conv = _pd_time_meta(tix, first)
    time = conv(tix)
    value_cols = [c for c in df.columns if c not in key_cols and c != time_col]
    dtypes = {c: df[c].dtype for c in value_cols}
    num, aux_cols = _split_value_cols(dtypes)
    values = _stack_cols([df[c].to_numpy(dtype=np.float64) for c in num], n)
    aux = {c: df[c].to_numpy() for c in aux_cols}
    if perm is not None:
        time, values = time[perm], values[perm]
        aux = {c: a[perm] for c, a in aux.items()}
    meta = {
        "source": "pandas_long",
        "columns_order": list(df.columns),
        "col_dtypes": dtypes,
        "key_cols": key_cols,
        "time_col": time_col,
        "key_dtypes": [df[k].dtype for k in key_cols],
        "time_dtype": tcol.dtype,
        "perm": perm,
        "aux_order": aux_cols,
    }
    return PanelData(indptr, time, values, num, list(key_cols), key_values, time_col,
                     tmeta, aux, meta)


def to_pandas_long(hub, restore_order=True):
    import pandas as pd

    m = hub.meta
    lengths = hub.lengths
    data = {}
    for k, kv, dt in zip(m["key_cols"], hub.key_values, m["key_dtypes"]):
        data[k] = pd.array(np.repeat(kv, lengths), dtype=dt)
    tix = time_to_pandas_index(hub.time, hub.tmeta, freq_mode="none")
    data[m["time_col"]] = tix.astype(m["time_dtype"]) if m.get("time_dtype") is not None \
        and tix.dtype != m["time_dtype"] else tix
    for j, c in enumerate(hub.columns):
        dt = (m.get("col_dtypes") or {}).get(c)
        col = hub.values[:, j]
        data[c] = col.astype(dt) if dt is not None and col.dtype != dt else col
    for c, a in hub.aux.items():
        data[c] = a
    df = pd.DataFrame(data, copy=False)
    df = df[[c for c in m["columns_order"] if c in df.columns]]
    perm = m.get("perm")
    if restore_order and perm is not None and len(perm) == len(df):
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        df = df.iloc[inv].reset_index(drop=True)
    return df


# =============================================================================
# spoke (i): polars (eager + lazy)
# =============================================================================


def _pl_time_meta(dtype, t_first):
    import polars as pl

    if isinstance(dtype, pl.Datetime):
        step, freq = infer_step_numpy(t_first, "datetime")
        return TimeMeta("datetime", unit=dtype.time_unit, tz=dtype.time_zone,
                        freq=freq, step=step)
    if dtype == pl.Date:
        step, freq = infer_step_numpy(t_first, "datetime")
        return TimeMeta("datetime", unit="D", freq=freq, step=step)
    step, _ = infer_step_numpy(t_first, "int")
    return TimeMeta("int", step=step)


def from_polars(df, key_cols=None, time_col=None, lazy_hint=None):
    """polars DataFrame / LazyFrame -> hub (LazyFrame is collected once).

    lazy_hint: optional dict(tail=w) - push "last w rows per instance" into the
    lazy query plan before collecting (forecasters with bounded lookback).
    """
    import polars as pl

    lazy = isinstance(df, pl.LazyFrame)
    schema = df.collect_schema() if lazy else df.schema
    key_cols, time_col = _detect_long_cols(list(schema.names()), key_cols, time_col)
    if lazy:
        if lazy_hint and lazy_hint.get("tail"):
            df = df.group_by(key_cols, maintain_order=True).tail(lazy_hint["tail"])
        df = df.collect()
    n = df.height
    if len(key_cols) == 1:
        rid = df[key_cols[0]].rle_id()
    elif key_cols:
        rid = df.select(pl.struct(key_cols).rle_id()).to_series()
    else:
        rid = pl.Series(np.zeros(n, dtype=np.uint32))
    rid_np = rid.to_numpy()
    starts = _run_starts_from_codes([rid_np], n)
    perm = None
    if key_cols and df[starts.tolist()].select(key_cols).is_duplicated().any():
        df = df.with_row_index("__x6_row").sort(
            pl.col("__x6_row").min().over(key_cols), maintain_order=True)
        perm = df["__x6_row"].to_numpy()
        df = df.drop("__x6_row")
        rid_np = (df.select(pl.struct(key_cols).rle_id()).to_series().to_numpy()
                  if len(key_cols) > 1 else df[key_cols[0]].rle_id().to_numpy())
        starts = _run_starts_from_codes([rid_np], n)
    indptr = _starts_to_indptr(starts, n)
    keys_at = df[starts.tolist()].select(key_cols) if key_cols else None
    key_values = [keys_at[k].to_numpy() for k in key_cols]
    tser = df[time_col]
    time = tser.to_numpy()  # zero-copy for Datetime/int without nulls
    tmeta = _pl_time_meta(tser.dtype, time[indptr[0]:indptr[1]] if n else time)
    value_cols = [c for c in df.columns if c not in key_cols and c != time_col]
    num = [c for c in value_cols if df.schema[c].is_numeric()]
    aux_cols = [c for c in value_cols if c not in num]
    null_cols = [c for c in num if df[c].null_count() > 0]
    if len(num) == 1:
        values = df[num[0]].to_numpy().reshape(n, 1)  # zero-copy if Float64, no nulls
        if values.dtype != np.float64:
            values = values.astype(np.float64)
    elif num:
        values = df.select(num).to_numpy(order="fortran")
        if values.dtype != np.float64:
            values = values.astype(np.float64)
    else:
        values = np.empty((n, 0))
    aux = {c: df[c].to_numpy() for c in aux_cols}
    meta = {
        "source": "polars",
        "lazy": lazy,
        "schema": dict(df.schema),
        "columns_order": list(df.columns),
        "key_cols": key_cols,
        "time_col": time_col,
        "null_cols": null_cols,
        "perm": perm,
        "aux_order": aux_cols,
    }
    return PanelData(indptr, time, values, num, list(key_cols), key_values, time_col,
                     tmeta, aux, meta)


def to_polars(hub, restore_order=True):
    import polars as pl

    m = hub.meta
    schema = m["schema"]
    lengths = hub.lengths
    inst = pl.Series(np.repeat(np.arange(hub.n_instances, dtype=np.uint32), lengths))
    series = []
    for k, kv in zip(m["key_cols"], hub.key_values):
        series.append(pl.Series(k, kv, dtype=schema.get(k)).gather(inst))
    tdt = schema.get(m["time_col"])
    series.append(pl.Series(m["time_col"], hub.time).cast(tdt) if tdt is not None
                  else pl.Series(m["time_col"], hub.time))
    for j, c in enumerate(hub.columns):
        s = pl.Series(c, hub.values[:, j])
        if c in m.get("null_cols", []):
            s = s.fill_nan(None)
        dt = schema.get(c)
        if dt is not None and s.dtype != dt:
            s = s.cast(dt)
        series.append(s)
    for c, a in hub.aux.items():
        series.append(pl.Series(c, a, dtype=schema.get(c)))
    df = pl.DataFrame(series)
    df = df.select([c for c in m["columns_order"] if c in df.columns])
    perm = m.get("perm")
    if restore_order and perm is not None and len(perm) == df.height:
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        df = df[inv]
    return df.lazy() if m.get("lazy") else df


# =============================================================================
# spoke (i): pyarrow
# =============================================================================


def _np_from_arrow(col):
    """ChunkedArray -> numpy, zero-copy when single chunk and no nulls."""
    if col.num_chunks != 1:
        col = col.combine_chunks()
    else:
        col = col.chunk(0)
    try:
        return col.to_numpy(zero_copy_only=True)
    except Exception:  # noqa: BLE001  nulls / non-primitive -> copy
        return col.to_numpy(zero_copy_only=False)


def _arrow_codes(col):
    """Codes + dictionary of a key column (string, string_view, dictionary types).

    The dictionary is cast to a plain type because pyarrow 26 has no ``take``
    kernel for ``string_view`` (what polars exports through the PyCapsule).
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if pa.types.is_dictionary(col.type):
        col = col.unify_dictionaries() if col.num_chunks > 1 else col
        arr = col.combine_chunks()
    else:
        arr = pc.dictionary_encode(col.combine_chunks())
    dic = arr.dictionary
    if pa.types.is_string_view(dic.type):
        dic = dic.cast(pa.large_string())
    return arr.indices.to_numpy(zero_copy_only=False), dic


def from_arrow(table, key_cols=None, time_col=None):
    import pyarrow as pa

    key_cols, time_col = _detect_long_cols(table.column_names, key_cols, time_col)
    n = table.num_rows
    codes_dicts = [_arrow_codes(table.column(k)) for k in key_cols]
    codes = [c for c, _ in codes_dicts]
    starts = _run_starts_from_codes(codes, n)
    perm = None
    if codes and len(np.unique(_composite([c[starts] for c in codes]))) != len(starts):
        perm = _first_appearance_perm(codes)
        # no take() kernel for string_view in pyarrow 26: widen first (restored at
        # egress by the schema cast)
        sv = [i for i, f in enumerate(table.schema) if pa.types.is_string_view(f.type)]
        for i in sv:
            table = table.set_column(i, table.field(i).name,
                                     table.column(i).cast(pa.large_string()))
        table = table.take(pa.array(perm))
        codes = [c[perm] for c in codes]
        starts = _run_starts_from_codes(codes, n)
    indptr = _starts_to_indptr(starts, n)
    key_values = [d.to_numpy(zero_copy_only=False)[c[starts]]
                  for c, (_, d) in zip(codes, codes_dicts)]
    tcol = table.column(time_col)
    time = _np_from_arrow(tcol)
    tdt = tcol.type
    if pa.types.is_timestamp(tdt):
        step, freq = infer_step_numpy(time[indptr[0]:indptr[1]], "datetime")
        tmeta = TimeMeta("datetime", unit=tdt.unit, tz=tdt.tz, freq=freq, step=step)
    elif pa.types.is_date(tdt):
        time = time.astype("datetime64[D]")
        step, freq = infer_step_numpy(time[indptr[0]:indptr[1]], "datetime")
        tmeta = TimeMeta("datetime", unit="D", freq=freq, step=step)
    else:
        step, _ = infer_step_numpy(time[indptr[0]:indptr[1]], "int")
        tmeta = TimeMeta("int", step=step)
    value_cols = [c for c in table.column_names if c not in key_cols and c != time_col]
    num = [c for c in value_cols if pa.types.is_floating(table.schema.field(c).type)
           or pa.types.is_integer(table.schema.field(c).type)]
    aux_cols = [c for c in value_cols if c not in num]
    cols = [_np_from_arrow(table.column(c)) for c in num]
    cols = [c if c.dtype == np.float64 else c.astype(np.float64) for c in cols]
    values = _stack_cols(cols, n) if cols else np.empty((n, 0))
    aux = {c: table.column(c).to_numpy() for c in aux_cols}
    meta = {
        "source": "arrow",
        "null_cols": [c for c in num if table.column(c).null_count > 0],
        "schema": table.schema,
        "columns_order": table.column_names,
        "key_cols": key_cols,
        "time_col": time_col,
        "perm": perm,
        "aux_order": aux_cols,
    }
    return PanelData(indptr, time, values, num, list(key_cols), key_values, time_col,
                     tmeta, aux, meta)


def to_arrow(hub, restore_order=True):
    import pyarrow as pa

    m = hub.meta
    schema = m["schema"]
    inst = pa.array(np.repeat(np.arange(hub.n_instances, dtype=np.int32),
                              hub.lengths))
    arrays, names = [], []
    for k, kv in zip(m["key_cols"], hub.key_values):
        ft = schema.field(k).type
        if pa.types.is_dictionary(ft):  # codes over the (small) key table
            vt = ft.value_type
            dic = pa.array(kv, type=pa.large_string() if pa.types.is_string_view(vt)
                           else vt)
            a = pa.DictionaryArray.from_arrays(inst.cast(ft.index_type), dic)
            a = a if a.type == ft else a.cast(ft)
        else:
            plain = pa.large_string() if pa.types.is_string_view(ft) else ft
            a = pa.array(kv, type=plain).take(inst)
            a = a if a.type == ft else a.cast(ft)
        arrays.append(a)
        names.append(k)
    t = hub.time
    if hub.tmeta.unit == "D":
        t = t.astype("datetime64[D]")
    arrays.append(pa.array(t).cast(schema.field(m["time_col"]).type))
    names.append(m["time_col"])
    null_cols = set(m.get("null_cols", []))
    for j, c in enumerate(hub.columns):
        col = np.ascontiguousarray(hub.values[:, j])
        # zero-copy if contiguous; NaN -> null for columns that had nulls
        a = pa.array(col, mask=np.isnan(col)) if c in null_cols else pa.array(col)
        ft = schema.field(c).type if c in schema.names else a.type
        arrays.append(a if a.type == ft else a.cast(ft))
        names.append(c)
    for c, a in hub.aux.items():
        arrays.append(pa.array(a, type=schema.field(c).type))
        names.append(c)
    tab = pa.Table.from_arrays(arrays, names=names)
    tab = tab.select([c for c in m["columns_order"] if c in names])
    if schema.metadata:
        tab = tab.replace_schema_metadata(schema.metadata)
    perm = m.get("perm")
    if restore_order and perm is not None and len(perm) == tab.num_rows:
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        tab = tab.take(pa.array(inv))
    return tab


# =============================================================================
# spoke (i): dask  (compute once -> pandas long spoke)
# =============================================================================


def from_dask(ddf, key_cols=None, time_col=None):
    pdf = ddf.compute()
    hub = from_pandas_long(pdf.reset_index(drop=True), key_cols, time_col)
    hub.meta["source"] = "dask"
    hub.meta["npartitions"] = ddf.npartitions
    return hub


def to_dask(hub, restore_order=True):
    import dask.dataframe as dd

    pdf = to_pandas_long(hub, restore_order=restore_order)
    return dd.from_pandas(pdf, npartitions=min(hub.meta.get("npartitions", 1),
                                               max(1, len(pdf))))


# =============================================================================
# dispatch for style (i)
# =============================================================================


def from_native(obj, key_cols=None, time_col=None, **kw):
    mod = type(obj).__module__.split(".")[0]
    if mod == "pandas":
        if key_cols is not None or (hasattr(obj, "columns") and any(
                isinstance(c, str) and c.startswith(INDEX_PREFIX) for c in obj.columns)):
            return from_pandas_long(obj, key_cols, time_col)
        return from_pandas(obj)
    if mod == "polars":
        return from_polars(obj, key_cols, time_col, **kw)
    if mod == "pyarrow":
        return from_arrow(obj, key_cols, time_col)
    if mod in ("dask", "dask_expr"):
        return from_dask(obj, key_cols, time_col)
    raise TypeError(type(obj))


def to_native(hub, restore_order=True):
    src = hub.meta["source"]
    return {
        "pandas": to_pandas,
        "pandas_long": to_pandas_long,
        "polars": to_polars,
        "arrow": to_arrow,
        "dask": to_dask,
    }[src](hub, restore_order=restore_order)


HAND_SPOKES = [
    _pd_time_meta, from_pandas, to_pandas, from_pandas_long, to_pandas_long,
    _pl_time_meta, from_polars, to_polars, _np_from_arrow, from_arrow, to_arrow,
    from_dask, to_dask, from_native, to_native,
]

# =============================================================================
# spoke (ii-a): narwhals (one generic code path)
# =============================================================================


def nw_from_native(native, key_cols=None, time_col=None, collect_backend=None):
    """Any narwhals-supported frame -> hub. pandas MultiIndex must be reset first."""
    import narwhals as nw

    pd_index_names = None
    pd_series_name = False
    if type(native).__module__.split(".")[0] == "pandas" and key_cols is None:
        # narwhals cannot see the pandas index (C2): move it into columns
        if not hasattr(native, "columns"):
            pd_series_name = (native.name,)
            native = native.to_frame(name=native.name if native.name is not None
                                     else "__x6_series")
        pd_index_names = list(native.index.names)
        native = native.reset_index()
        names = [c for c in native.columns[: len(pd_index_names)]]
        key_cols, time_col = names[:-1], names[-1]
    f = nw.from_native(native)
    impl = f.implementation
    lazy = isinstance(f, nw.LazyFrame)
    if lazy:
        f = f.collect(backend=collect_backend)  # default: dask->pandas, duckdb->arrow
    key_cols, time_col = _detect_long_cols(f.columns, key_cols, time_col)
    schema = dict(f.schema)
    n = len(f)
    if key_cols:
        chg = f.select(nw.any_horizontal(
            *[(nw.col(k) != nw.col(k).shift(1)).fill_null(True) for k in key_cols],
            ignore_nulls=False).alias("c"))["c"].to_numpy()
    else:
        chg = np.zeros(n, dtype=bool)
        chg[:1] = True
    starts = np.flatnonzero(chg)
    perm = None
    heads = f[starts.tolist()].select(key_cols) if key_cols else None
    if key_cols and heads.is_duplicated().any():
        f = f.with_row_index("__x6_row").with_columns(
            nw.col("__x6_row").min().over(*key_cols).alias("__x6_g")
        ).sort("__x6_g", "__x6_row")
        perm = f["__x6_row"].to_numpy()
        f = f.drop("__x6_row", "__x6_g")
        chg = f.select(nw.any_horizontal(
            *[(nw.col(k) != nw.col(k).shift(1)).fill_null(True) for k in key_cols],
            ignore_nulls=False).alias("c"))["c"].to_numpy()
        starts = np.flatnonzero(chg)
        heads = f[starts.tolist()].select(key_cols)
    indptr = _starts_to_indptr(starts, n)
    key_values = [heads[k].to_numpy() for k in key_cols]
    time = f[time_col].to_numpy()
    tdt = schema[time_col]
    if isinstance(tdt, nw.Datetime):
        kind, unit, tz = "datetime", tdt.time_unit, tdt.time_zone
    elif tdt == nw.Date:
        kind, unit, tz = "datetime", "D", None
        time = time.astype("datetime64[D]")
    else:
        kind, unit, tz = "int", None, None
    if tz is not None and time.dtype == object:  # pandas tz-aware -> object array
        import pandas as pd

        time = pd.DatetimeIndex(time).tz_convert("UTC").tz_localize(None).values
    elif kind == "datetime" and time.dtype.kind == "M" and tz is None:
        pass
    step, freq = infer_step_numpy(time[indptr[0]:indptr[1]], kind)
    tmeta = TimeMeta(kind, unit=unit, tz=tz, freq=freq, step=step)
    value_cols = [c for c in f.columns if c not in key_cols and c != time_col]
    num = [c for c in value_cols if schema[c].is_numeric()]
    aux_cols = [c for c in value_cols if c not in num]
    null_cols = [c for c in num if f[c].null_count() > 0]
    values = _stack_cols([f[c].to_numpy().astype(np.float64, copy=False)
                          for c in num], n)
    aux = {c: f[c].to_numpy() for c in aux_cols}
    meta = {
        "source": "narwhals",
        "null_cols": null_cols,
        "pd_series_name": pd_series_name,
        "impl": impl,
        "eager_impl": f.implementation,
        "lazy": lazy,
        "schema": schema,
        "columns_order": list(f.columns),
        "key_cols": key_cols,
        "time_col": time_col,
        "pd_index_names": pd_index_names,
        "perm": perm,
        "aux_order": aux_cols,
    }
    return PanelData(indptr, time, values, num, list(key_cols), key_values, time_col,
                     tmeta, aux, meta)


def nw_to_native(hub, restore_order=True):
    import narwhals as nw

    m = hub.meta
    lengths = hub.lengths
    data = {k: np.repeat(kv, lengths) for k, kv in zip(m["key_cols"], hub.key_values)}
    data[m["time_col"]] = hub.time
    for j, c in enumerate(hub.columns):
        data[c] = np.ascontiguousarray(hub.values[:, j])
    data.update(hub.aux)
    f = nw.from_dict(data, backend=m["eager_impl"])
    schema = m["schema"]
    tz = hub.tmeta.tz
    if tz is not None:  # hub keeps naive UTC
        tc = m["time_col"]
        f = f.with_columns(nw.col(tc).dt.replace_time_zone("UTC")
                           .dt.convert_time_zone(tz))
    if m.get("null_cols"):
        f = f.with_columns(*[nw.col(c).fill_nan(None) for c in m["null_cols"]])
    f = f.with_columns(*[nw.col(c).cast(schema[c]) for c in f.columns if c in schema])
    f = f.select([c for c in m["columns_order"] if c in f.columns])
    perm = m.get("perm")
    if restore_order and perm is not None and len(perm) == len(f):
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        f = f[inv.tolist()]
    if m["lazy"]:
        f = f.lazy(backend=m["impl"])
    out = f.to_native()
    if m["pd_index_names"] is not None:
        idx_cols = list(out.columns[: len(m["pd_index_names"])])
        out = out.set_index(idx_cols if len(idx_cols) > 1 else idx_cols[0])
        out.index.names = m["pd_index_names"]
        if m["pd_series_name"]:
            out = out.iloc[:, 0]
            out.name = m["pd_series_name"][0]
    return out


NW_SPOKES = [nw_from_native, nw_to_native]

# =============================================================================
# spoke (ii-b): Arrow PyCapsule ingress (+ pyarrow spoke + narwhals.from_arrow)
# =============================================================================


def capsule_from_native(native, key_cols=None, time_col=None):
    """Anything with ``__arrow_c_stream__`` -> pa.Table -> hub (via pyarrow spoke)."""
    import pyarrow as pa

    lib = type(native).__module__.split(".")[0].lstrip("_")  # "_duckdb" -> "duckdb"
    lazy = False
    if hasattr(native, "to_frame") and not hasattr(native, "columns"):
        native = native.to_frame()  # pandas/polars Series
    if not hasattr(native, "__arrow_c_stream__"):
        lazy = True  # polars LazyFrame / dask: no capsule, materialize first
        native = native.collect() if hasattr(native, "collect") else native.compute()
    table = pa.table(native)  # PyCapsule stream (pandas: index -> columns + metadata)
    pd_meta = None
    has_index_cols = any(c.startswith(INDEX_PREFIX) for c in table.column_names)
    if table.schema.metadata and b"pandas" in table.schema.metadata:
        import json

        pd_meta = json.loads(table.schema.metadata[b"pandas"])
        idx_cols = [c for c in pd_meta["index_columns"] if isinstance(c, str)]
        if key_cols is not None or has_index_cols:  # index is not the time axis
            used = set(key_cols or []) | {time_col}
            table = table.drop_columns([c for c in idx_cols if c not in used])
        elif len(idx_cols) > 1:
            key_cols, time_col = idx_cols[:-1], idx_cols[-1]
        elif len(idx_cols) == 1:
            key_cols, time_col = [], idx_cols[0]
    hub = from_arrow(table, key_cols, time_col)
    hub.meta.update(capsule_lib=lib, capsule_lazy=lazy, source="capsule")
    return hub


def capsule_to_native(hub, restore_order=True):
    import narwhals as nw

    table = to_arrow(hub, restore_order=restore_order)
    lib = hub.meta["capsule_lib"]
    if lib == "duckdb":
        f = nw.from_arrow(table, backend="pyarrow").lazy(backend="duckdb")
        return f.to_native()
    if lib in ("dask", "dask_expr"):
        return nw.from_arrow(table, backend="pandas").lazy(backend="dask").to_native()
    out = nw.from_arrow(table, backend=lib).to_native()  # pandas: metadata -> MI
    return out.lazy() if hub.meta["capsule_lazy"] else out


CAPSULE_SPOKES = [capsule_from_native, capsule_to_native]

# =============================================================================
# forecaster side
# =============================================================================


class HubNaive:
    """NaiveForecaster-like kernel on numpy views (no pandas, no estimator clone).

    Mirrors sktime ``NaiveForecaster`` for strategies last/mean/drift with
    ``window_length`` and ``sp`` (``sp`` only for last/mean, as in sktime).
    """

    def __init__(self, strategy="last", window_length=None, sp=1):
        self.strategy = strategy
        self.window_length = window_length
        self.sp = sp or 1

    def fit(self, sv):
        y = sv.values[:, 0]
        n = len(y)
        wl = n if self.window_length is None else self.window_length
        self.window_length_ = wl
        self.last_window_ = y[max(0, n - wl):]  # view
        self.cutoff_ = sv.time[-1]
        self.tmeta_ = sv.tmeta
        return self

    def predict(self, fh):
        fh = np.asarray(fh, dtype=np.int64)
        lw = self.last_window_
        sp, s = self.sp, self.strategy
        wl = self.window_length_
        if len(lw) == 0 or np.all(np.isnan(lw)):
            y = np.full(len(fh), np.nan)
        elif s == "last" and sp == 1:
            valid = ~np.isnan(lw)
            y = np.repeat(lw[np.flatnonzero(valid)[-1]], len(fh))
        elif s in ("last", "mean") and sp > 1:
            rem = wl % sp
            pad = (sp - rem if rem else 0) + wl - len(lw)
            w = np.concatenate([np.full(pad, np.nan), lw]).reshape(-1, sp)
            if s == "last":
                valid = ~np.isnan(w)
                rows = valid.cumsum(0).argmax(0)
                yp = w[rows, np.arange(sp)]
            else:
                yp = np.nanmean(w, axis=0)
            reps = int(np.ceil(fh[-1] / sp)) if fh[-1] > sp else 1
            y = np.tile(yp, reps)[fh - 1]
        elif s == "mean":
            y = np.repeat(np.nanmean(lw), len(fh))
        elif s == "drift":
            slope = (lw[-1] - lw[0]) / (wl - 1)
            y = lw[-1] + fh * slope
        else:
            raise ValueError(s)
        return future_time(self.cutoff_, fh, self.tmeta_), y.reshape(-1, 1)


def hub_vectorize(hub, make_kernel, fh):
    """Per-instance loop on views: fit + predict, reconstruct to a hub."""
    outs = []
    for sv in hub.iter_slices():
        outs.append(make_kernel().fit(sv).predict(fh))
    return hub.from_outputs(outs)


def hub_naive_batched(hub, fh, strategy="last", window_length=None):
    """All instances at once via indptr reductions (sp == 1)."""
    fh = np.asarray(fh, dtype=np.int64)
    v = hub.values[:, 0]
    ip = hub.indptr
    ends = ip[1:]
    lens = np.diff(ip)
    wl = lens if window_length is None else np.minimum(window_length, lens)
    ws = ends - wl
    h = len(fh)
    if strategy == "last":
        isn = np.isnan(v)
        pos = np.where(isn, -1, np.arange(len(v)))
        lastpos = np.maximum.reduceat(pos, ip[:-1]) if len(v) else pos
        ok = lastpos >= ws
        yl = np.where(ok, v[np.clip(lastpos, 0, None)], np.nan)
        y = np.repeat(yl[:, None], h, axis=1)
    elif strategy == "mean":
        # per-window sums with reduceat over [ws_i, end_i) pairs (numerically
        # stable, unlike differences of a global cumsum)
        bounds = np.empty(2 * len(ws), dtype=np.int64)
        bounds[0::2], bounds[1::2] = ws, ends
        vz = np.append(np.nan_to_num(v), 0.0)  # sentinel so end == len(v) is valid
        cnt = np.append((~np.isnan(v)).astype(np.int64), 0)
        sums = np.add.reduceat(vz, bounds)[0::2]
        cnts = np.add.reduceat(cnt, bounds)[0::2]
        with np.errstate(invalid="ignore", divide="ignore"):
            ym = sums / cnts
        y = np.repeat(ym[:, None], h, axis=1)
    elif strategy == "drift":
        last, first = v[ends - 1], v[ws]
        slope = (last - first) / (wl - 1)
        y = last[:, None] + fh[None, :] * slope[:, None]
    else:
        raise ValueError(strategy)
    cut = hub.time[ends - 1]
    tm = hub.tmeta
    if tm.kind in ("int", "period") or tm.step is not None:
        step = tm.step if tm.step is not None else 1
        tf = cut[:, None] + fh[None, :] * step
    else:  # calendar freq: compute per unique cutoff
        uc, inv = np.unique(cut, return_inverse=True)
        grid = np.stack([future_time(c, fh, tm) for c in uc])
        tf = grid[inv]
    indptr = np.arange(hub.n_instances + 1, dtype=np.int64) * h
    return hub.with_data(indptr, tf.reshape(-1), y.reshape(-1, 1))
