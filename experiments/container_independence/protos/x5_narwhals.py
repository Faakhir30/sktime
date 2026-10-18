"""x5 prototype: a narwhals-based, stateless VectorizedDF.

Same functional surface as ``sktime.datatypes._vectorize.VectorizedDF``:

* build a schema from the input (instance keys, columns, key/time names, freq)
* iterate slices of the data per instance (``items``)
* reconstruct a frame from a list of per-instance results (``reconstruct``)

but written against narwhals so that the frame is never converted to pandas.

Data model
----------
Narwhals frames have no index, so every input is brought to LONG format:
instance key columns + one time column + value columns.

* pandas (Multi)Index input: ``reset_index`` once (a copy), the index
  metadata (level names incl. ``None``, column Index object, freq) is kept in
  ``PandasIndexMeta`` and used to restore the exact MultiIndex on the way out.
* polars / pyarrow / duckdb / dask / ibis input: either the sktime
  ``__index__<name>`` column convention, or explicit ``keys=`` and ``time=``.

Lazy handling (decision, see report)
------------------------------------
* The schema NEVER collects the data. Instance keys of a lazy input are only
  computed on demand (``schema.instance_keys(frame)``), as a small
  keys-only query (projection pushdown: only key columns are read).
* ``items`` on a lazy frame has to materialize: narwhals ``LazyGroupBy`` is not
  iterable, and filtering per key would issue one query per instance.
  We collect ONCE, explicitly, with ``collect_backend`` (default: the eager
  counterpart that keeps the data in its own memory format:
  polars.LazyFrame -> polars, duckdb -> pyarrow, dask -> pandas, ibis -> pyarrow).
* ``reconstruct`` of lazy inputs returns an eager frame of the collect backend,
  then (optionally, ``relazy=True``) re-wraps it with ``.lazy(backend=input)``,
  so the *type* matches the input. The data has been materialized regardless.
* Panel-native estimators (``x5_nw_forecasters``) never call ``items`` and keep
  lazy inputs lazy end to end - that is the only path where laziness pays off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import narwhals as nw
import numpy as np

INDEX_PREFIX = "__index__"
SERIES_SCITYPES = ["Series", "Panel", "Hierarchical"]

# eager backend used when a lazy frame must be materialized (items/reconstruct)
DEFAULT_COLLECT_BACKEND = {
    "polars": "polars",
    "dask": "pandas",
    "duckdb": "pyarrow",
    "ibis": "pyarrow",
    "pyspark": "pyarrow",
    "sqlframe": "pyarrow",
}


# --------------------------------------------------------------------------- #
# pandas index metadata (only used for pandas (Multi)Index inputs)
# --------------------------------------------------------------------------- #
@dataclass
class PandasIndexMeta:
    """What is needed to rebuild the exact pandas index after reset_index."""

    index_names: list  # original level names, may contain None
    flat_names: list  # names used as columns in the long frame
    columns: object  # original pd.Index of columns (labels may be non-str)
    flat_columns: list  # str column names used in the long frame
    time_kind: str  # "datetime", "period", "int", "other"
    period_freq: object = None  # pandas freq of a PeriodIndex time level


def _unique_name(base, taken):
    name = base
    i = 0
    while name in taken:
        i += 1
        name = f"{base}_{i}"
    return name


def pandas_to_long(X):
    """Pandas frame with (Multi)Index -> (long pandas frame, PandasIndexMeta).

    This is the ONE pandas-specific step; it costs a ``reset_index`` copy of the
    index levels (value columns are not copied under pandas Copy-on-Write).
    """
    import pandas as pd

    if isinstance(X, pd.Series):
        X = X.to_frame()
    idx = X.index
    names = list(idx.names)
    cols = X.columns
    taken = set()
    flat_cols = []
    for j, c in enumerate(cols):
        # narwhals requires str column names; sktime frequently has int names
        s = c if isinstance(c, str) else f"__col{j}__"
        s = _unique_name(s, taken)
        taken.add(s)
        flat_cols.append(s)
    flat_names = []
    for i, n in enumerate(names):
        s = n if isinstance(n, str) else f"__level{i}__"
        s = _unique_name(s, taken)
        taken.add(s)
        flat_names.append(s)

    time_level = idx.get_level_values(-1) if isinstance(idx, pd.MultiIndex) else idx
    period_freq = None
    if isinstance(time_level, pd.PeriodIndex):
        time_kind = "period"
        period_freq = time_level.freq
    elif isinstance(time_level, pd.DatetimeIndex):
        time_kind = "datetime"
    elif pd.api.types.is_integer_dtype(time_level.dtype):
        time_kind = "int"
    else:
        time_kind = "other"

    Xr = X.copy(deep=False)
    Xr.columns = flat_cols
    Xr.index = idx.set_names(flat_names)
    flat = Xr.reset_index()
    if time_kind == "period":
        # narwhals has no Period dtype; polars has none either -> timestamps
        flat[flat_names[-1]] = flat[flat_names[-1]].dt.to_timestamp()
    meta = PandasIndexMeta(
        index_names=names,
        flat_names=flat_names,
        columns=cols,
        flat_columns=flat_cols,
        time_kind=time_kind,
        period_freq=period_freq,
    )
    return flat, meta


def long_to_pandas(flat, meta: PandasIndexMeta, value_cols=None, freq=None):
    """Long pandas frame -> pandas frame with the original (Multi)Index restored."""
    import pandas as pd

    flat = flat.copy(deep=False)
    tcol = meta.flat_names[-1]
    if meta.time_kind == "period":
        flat[tcol] = flat[tcol].dt.to_period(meta.period_freq)
    out = flat.set_index(meta.flat_names)
    out.index = out.index.set_names(meta.index_names)
    if value_cols is None:
        value_cols = meta.flat_columns
    out = out[value_cols]
    col_map = dict(zip(meta.flat_columns, meta.columns))
    out.columns = pd.Index([col_map.get(c, c) for c in out.columns])
    if freq is not None and not isinstance(out.index, pd.MultiIndex):
        try:
            out.index.freq = freq
        except ValueError:
            pass
    return out


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
@dataclass
class NwVectorizedDF:
    """Stateless iteration schema for long-format panel data, backend agnostic.

    Holds only metadata; the frame is passed separately, as in the current
    stateless sktime ``VectorizedDF``.
    """

    keys: list  # instance key columns (all levels except time)
    time: str
    value_cols: list
    is_scitype: str
    iterate_as: str = "Series"
    iterate_cols: bool = False
    backend: str = "pandas"  # narwhals Implementation name of the input
    is_lazy: bool = False
    native_type: type = None
    pandas_meta: PandasIndexMeta | None = None
    freq: object = None
    collect_backend: str | None = None
    _keys_cache: object = field(default=None, repr=False)

    # ----- schema properties ------------------------------------------------
    @property
    def iter_keys(self):
        """Key columns that define one iteration slice."""
        n_drop = {"Series": 0, "Panel": 1, "Hierarchical": 2}[self.iterate_as]
        n_drop_self = {"Series": 0, "Panel": 1, "Hierarchical": 2}[self.is_scitype]
        # iterate_as == is_scitype -> whole frame is one slice
        if n_drop == n_drop_self:
            return []
        return self.keys[: len(self.keys) - n_drop]

    def instance_keys(self, frame):
        """Eager narwhals DataFrame of iteration keys, in schema order.

        Order: first appearance for eager inputs (matches pandas
        ``groupby(sort=False)``), sorted for lazy inputs (no row order exists).
        Lazy inputs run a keys-only query (cached).
        """
        if self._keys_cache is not None:
            return self._keys_cache
        k = self.iter_keys
        if not k:
            return None
        if isinstance(frame, nw.LazyFrame):
            res = frame.select(k).unique().sort(k).collect(self._collect_backend())
        else:
            res = frame.select(k).unique(maintain_order=True)
        self._keys_cache = res
        return res

    def _collect_backend(self):
        return self.collect_backend or DEFAULT_COLLECT_BACKEND.get(
            self.backend, "pyarrow"
        )

    def shape(self, frame):
        ik = self.instance_keys(frame)
        n_rows = 1 if ik is None else len(ik)
        return n_rows, (len(self.value_cols) if self.iterate_cols else 1)

    def __len__(self):
        raise TypeError("use schema.shape(frame): key count needs the frame")

    # ----- iteration ------------------------------------------------------
    def materialize(self, frame):
        """Collect a lazy frame ONCE (explicit, documented backend)."""
        if isinstance(frame, nw.LazyFrame):
            return frame.collect(self._collect_backend())
        return frame

    def _offsets(self, frame):
        """(key rows, start, length) per slice, order = schema order.

        Requires rows of one instance to be contiguous; sorts if not.
        """
        k = self.iter_keys
        ri = "__x5_ri__"
        g = (
            frame.with_row_index(ri)
            .group_by(k)
            .agg(
                nw.col(ri).min().alias("__start"),
                nw.col(ri).max().alias("__stop"),
                nw.len().alias("__len"),
            )
        )
        ik = self.instance_keys(frame)
        g = ik.join(g, on=k, how="left")
        start = g["__start"].to_numpy()
        stop = g["__stop"].to_numpy()
        length = g["__len"].to_numpy()
        contiguous = bool(np.all(stop - start + 1 == length))
        return g.select(k), start, length, contiguous

    def items(self, frame, fast_pandas=False):
        """Yield (group_name, col_name, slice) like ``VectorizedDF.items``.

        Slices are NATIVE frames of the (collected) input backend, without the
        iteration key columns. For pandas (Multi)Index inputs the slice gets the
        time index (freq inferred per slice, as ``_enforce_index_freq`` does),
        i.e. exactly what current sktime passes to inner estimators.

        ``fast_pandas=True``: pandas slices are built directly from numpy
        blocks + a pre-built time Index (no per-slice set_index).
        """
        frame = self.materialize(frame)
        k = self.iter_keys
        if not k:
            slices = [(None, frame)]
        else:
            keys_df, start, length, contiguous = self._offsets(frame)
            if not contiguous:
                # stable sort into schema order, then recompute offsets
                order = keys_df.with_row_index("__x5_ord__")
                frame = (
                    frame.join(order, on=k, how="left")
                    .sort("__x5_ord__", self.time)
                    .drop("__x5_ord__")
                )
                keys_df, start, length, _ = self._offsets(frame)
            key_rows = keys_df.rows()
            sub = frame.drop(k)
            if (
                fast_pandas
                and self.pandas_meta is not None
                and not self.iterate_cols
                and self._rest() == [self.time]
            ):
                yield from self._fast_pandas_items(sub, key_rows, start, length)
                return
            slices = []
            for key, s, n in zip(key_rows, start, length):
                name = key[0] if len(key) == 1 else key
                slices.append((name, sub[int(s) : int(s + n)]))

        for name, sl in slices:
            if self.iterate_cols:
                labels = (
                    self.pandas_meta.columns
                    if self.pandas_meta is not None
                    else self.value_cols
                )
                for c, lab in zip(self.value_cols, labels):
                    yield name, lab, self._finalize_slice(sl.select(*self._rest(), c))
            else:
                yield name, None, self._finalize_slice(sl)

    def _fast_pandas_items(self, sub, key_rows, start, length):
        import pandas as pd

        meta = self.pandas_meta
        native = sub.to_native()
        tcol = native[self.time]
        if meta.time_kind == "period":
            tcol = tcol.dt.to_period(meta.period_freq)
        tidx = pd.Index(tcol, name=meta.index_names[-1])
        block = native[meta.flat_columns]
        homog = block.dtypes.nunique() == 1
        vals = block.to_numpy() if homog else None
        cols = meta.columns
        for key, s, n in zip(key_rows, start, length):
            s, e = int(s), int(s + n)
            name = key[0] if len(key) == 1 else key
            idx = tidx[s:e]
            if homog:
                df = pd.DataFrame(vals[s:e], index=idx, columns=cols)
            else:
                df = pd.DataFrame(
                    {c: block[fc].to_numpy()[s:e] for c, fc in zip(cols, block)},
                    index=idx,
                )
            if hasattr(df.index, "freq") and df.index.freq is None and n > 2:
                df.index.freq = pd.infer_freq(df.index)
            yield name, None, df

    def _rest(self):
        """Non-value columns that remain in a slice (non-iterated keys + time)."""
        k = self.iter_keys
        return [c for c in [*self.keys, self.time] if c not in k]

    def _finalize_slice(self, sl):
        if self.pandas_meta is None:
            return sl.to_native()
        # pandas parity: restore time (and remaining key levels) as index
        import pandas as pd

        native = sl.to_native()
        meta = self.pandas_meta
        rest = self._rest()
        n_rest = len(rest)
        sub_meta = PandasIndexMeta(
            index_names=meta.index_names[-n_rest:],
            flat_names=meta.flat_names[-n_rest:],
            columns=meta.columns,
            flat_columns=meta.flat_columns,
            time_kind=meta.time_kind,
            period_freq=meta.period_freq,
        )
        vc = [c for c in native.columns if c not in rest]
        out = long_to_pandas(native, sub_meta, value_cols=vc)
        if not isinstance(out.index, pd.MultiIndex) and hasattr(out.index, "freq"):
            if out.index.freq is None and len(out.index) > 2:
                out.index.freq = pd.infer_freq(out.index)
        return out

    def as_list(self, frame, **kw):
        return [s for _, _, s in self.items(frame, **kw)]

    # ----- reconstruction ----------------------------------------------------
    def reconstruct(self, frame, results, relazy=False):
        """Row-concatenate per-instance native results and add the key columns.

        ``results`` are native frames in the same order as ``items``; each must
        contain the time column (for pandas: the time index) and value columns.
        Key columns are added by ONE horizontal concat of the repeated keys
        (no per-slice ``with_columns``).
        """
        k = self.iter_keys
        if self.pandas_meta is not None:
            return self._reconstruct_pandas(frame, results)
        parts = [nw.from_native(r) for r in results]
        body = nw.concat(parts, how="vertical")
        if k:
            ik = self.instance_keys(frame)
            lengths = np.array([len(p) for p in parts])
            pos = np.repeat(np.arange(len(lengths)), lengths)
            keys_rep = ik[pos.tolist()] if len(pos) else ik.head(0)
            body = nw.concat([keys_rep, body], how="horizontal")
        if relazy and self.is_lazy:
            body = body.lazy(backend=self.backend)
            return body.to_native()
        return body.to_native()

    def _reconstruct_pandas(self, frame, results):
        import pandas as pd

        k = self.iter_keys
        if not k:
            return results[0]
        ik = self.instance_keys(frame)
        key_rows = ik.rows()
        names = [r[0] if len(r) == 1 else r for r in key_rows]
        out = pd.concat(results, keys=names, axis=0)
        out.index = out.index.set_names(self.pandas_meta.index_names)
        return out


def _split_index_cols(columns):
    idx_cols = [c for c in columns if isinstance(c, str) and c.startswith(INDEX_PREFIX)]
    return idx_cols


def prepare_nw_vectorized(
    X,
    iterate_as="Series",
    is_scitype=None,
    iterate_cols=False,
    keys=None,
    time=None,
    collect_backend=None,
    infer_freq=True,
):
    """Wrap ``X`` (any backend) as narwhals long frame and build the schema.

    Returns
    -------
    schema : NwVectorizedDF
    frame : narwhals DataFrame or LazyFrame (long format)
    """
    import pandas as pd

    pandas_meta = None
    is_pandas_indexed = isinstance(X, (pd.DataFrame, pd.Series)) and (
        keys is None and time is None
    )
    if is_pandas_indexed:
        flat, pandas_meta = pandas_to_long(X)
        keys = pandas_meta.flat_names[:-1]
        time = pandas_meta.flat_names[-1]
        frame = nw.from_native(flat)
        value_cols = pandas_meta.flat_columns
        native_type = type(X)
    else:
        native_type = type(X)
        frame = nw.from_native(X)
        cols = frame.columns
        if keys is None and time is None:
            idx_cols = _split_index_cols(cols)
            if not idx_cols:
                raise ValueError("pass keys= and time= or use __index__ columns")
            keys, time = idx_cols[:-1], idx_cols[-1]
        keys = list(keys or [])
        value_cols = [c for c in cols if c not in keys and c != time]

    if is_scitype is None:
        is_scitype = {0: "Series", 1: "Panel"}.get(len(keys), "Hierarchical")

    is_lazy = isinstance(frame, nw.LazyFrame)
    schema = NwVectorizedDF(
        keys=keys,
        time=time,
        value_cols=value_cols,
        is_scitype=is_scitype,
        iterate_as=iterate_as,
        iterate_cols=iterate_cols,
        backend=frame.implementation.name.lower(),
        is_lazy=is_lazy,
        native_type=native_type,
        pandas_meta=pandas_meta,
        collect_backend=collect_backend,
    )
    if infer_freq:
        schema.freq = infer_freq_nw(
            frame, keys, time, pandas_index=X.index if pandas_meta else None
        )
    return schema, frame


def infer_freq_nw(frame, keys, time, pandas_index=None, n=1000):
    """Infer the time frequency from the FIRST instance.

    pandas input: exactly as current sktime (index freqstr / pd.infer_freq).
    other input : pull the time column of one instance (lazy: 2 tiny queries:
    one key value, then its time values) and run pd.infer_freq on that small
    array; integer time -> step size.
    """
    import pandas as pd

    if pandas_index is not None:
        idx = pandas_index
        if isinstance(idx, pd.MultiIndex):
            first = idx.droplevel(-1)[0]
            tix = idx[idx.droplevel(-1) == first].get_level_values(-1)
        else:
            tix = idx
        if getattr(tix, "freqstr", None):
            return tix.freqstr
        try:
            return pd.infer_freq(tix)
        except (TypeError, ValueError):
            return None

    if keys:
        head = frame.select(keys).head(1)
        if isinstance(head, nw.LazyFrame):
            head = head.collect()
        row = head.rows()[0]
        cond = nw.all_horizontal(
            *[nw.col(k) == v for k, v in zip(keys, row)], ignore_nulls=False
        )
        tcol = frame.filter(cond).select(time).sort(time).head(n)
    else:
        tcol = frame.select(time).sort(time).head(n)
    if isinstance(tcol, nw.LazyFrame):
        tcol = tcol.collect()
    vals = tcol[time].to_numpy()
    if len(vals) < 3:
        return None
    if np.issubdtype(vals.dtype, np.integer):
        d = np.unique(np.diff(vals))
        return int(d[0]) if len(d) == 1 else None
    try:
        return pd.infer_freq(pd.DatetimeIndex(vals))
    except (TypeError, ValueError):
        return None
