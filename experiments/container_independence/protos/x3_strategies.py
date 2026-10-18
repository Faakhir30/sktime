"""x3: prototype implementations of VectorizedDF core ops for several containers.

Used by ``scripts/x3_iter_bench.py``. Not part of sktime.

Three core operations of a stateless VectorizedDF-like schema are prototyped:

(A) list instance keys (plus, for positional strategies, group offsets)
(B) iterate instance slices, materialized as what an inner estimator receives
(C) reconstruct: concatenate per-instance outputs with the instance keys

Positional strategies all rely on one idea: if rows of each instance are
contiguous (true for every sktime pandas multiindex frame built by sktime's
generators/converters, checked at schema construction), a slice is
``start:stop`` in row space, so iteration needs no hashing and, for columnar
containers (numpy, polars, arrow), no copy.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# generic helpers on integer group codes
# ---------------------------------------------------------------------------


def runs_from_gid(gid):
    """Return run start positions of an integer group id array."""
    n = len(gid)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    chg = np.flatnonzero(gid[1:] != gid[:-1]) + 1
    starts = np.empty(len(chg) + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = chg
    return starts


def offsets_from_gid(gid):
    """Group ``gid`` (int array) into first-appearance order.

    Returns
    -------
    first : int64 array, row position of first row of each group
    indptr : int64 array of length n_groups + 1, offsets in (possibly reordered) rows
    order : None if groups are contiguous (rows can be sliced in place),
        else int64 permutation (stable) that makes them contiguous
    """
    n = len(gid)
    starts = runs_from_gid(gid)
    run_gid = gid[starts]
    if len(pd.unique(run_gid)) == len(run_gid):
        indptr = np.empty(len(starts) + 1, dtype=np.int64)
        indptr[:-1] = starts
        indptr[-1] = n
        return starts, indptr, None
    codes, _ = pd.factorize(gid, sort=False)  # first-appearance numbering
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes)
    indptr = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    first = order[indptr[:-1]]
    return first, indptr, order


# ---------------------------------------------------------------------------
# S2: pandas, precomputed positions from MultiIndex codes
# ---------------------------------------------------------------------------


def pd_group_positions(X, n_inst_levels):
    """Instance keys + offsets of a pandas MultiIndex frame, from its codes.

    No hashing of key values: MultiIndex already stores integer codes per level.
    """
    idx = X.index
    k = n_inst_levels
    codes = [np.asarray(idx.codes[i]) for i in range(k)]
    if k == 1:
        gid = codes[0]
    else:
        gid = np.ravel_multi_index(codes, tuple(len(idx.levels[i]) for i in range(k)))
    first, indptr, order = offsets_from_gid(gid)
    if k == 1:
        keys = idx.levels[0].take(codes[0][first]).rename(idx.names[0])
    else:
        keys = pd.MultiIndex(
            levels=[idx.levels[i] for i in range(k)],
            codes=[c[first] for c in codes],
            names=idx.names[:k],
            verify_integrity=False,
        )
    return keys, indptr, order


class PandasPositional:
    """S2: iterate a pandas MI frame by precomputed row offsets."""

    def __init__(self, X, n_inst_levels):
        self.X = X
        self.k = n_inst_levels
        self.keys, self.indptr, order = pd_group_positions(X, n_inst_levels)
        if order is not None:
            self.X = X.take(order)
        self.levels = list(range(n_inst_levels))
        # cached for the "build" variant
        self.vals = self.X.to_numpy()
        self.time = self.X.index.get_level_values(-1)
        self.columns = self.X.columns

    def iter_iloc(self):
        """Slice with iloc, then drop instance levels (pandas index machinery)."""
        X, ip, lv = self.X, self.indptr, self.levels
        for i in range(len(ip) - 1):
            yield X.iloc[ip[i] : ip[i + 1]].droplevel(lv)

    def iter_build(self, freq=None):
        """Build each slice from cached numpy values + time index slice."""
        vals, t, ip, cols = self.vals, self.time, self.indptr, self.columns
        for i in range(len(ip) - 1):
            s, e = ip[i], ip[i + 1]
            idx = t[s:e]
            if freq is not None:
                idx = pd.DatetimeIndex(idx, freq=freq)
            yield pd.DataFrame(vals[s:e], index=idx, columns=cols, copy=False)


def pd_reconstruct_fast(outs, keys, index_names):
    """S2 reconstruct: build the MultiIndex from codes instead of pd.concat(keys=).

    ``outs`` is a list of pandas frames (one per instance, time index).
    ``keys`` is the instance key Index/MultiIndex in the same order.
    """
    lens = np.fromiter((len(o) for o in outs), dtype=np.int64, count=len(outs))
    rep = np.repeat(np.arange(len(outs)), lens)
    vals = np.concatenate([o.to_numpy() for o in outs])
    tidx = outs[0].index.append([o.index for o in outs[1:]])
    tcodes, tuniq = pd.factorize(tidx, sort=False)
    if isinstance(keys, pd.MultiIndex):
        levels = list(keys.levels) + [tuniq]
        codes = [np.asarray(c)[rep] for c in keys.codes] + [tcodes]
    else:
        levels = [keys, tuniq]
        codes = [rep, tcodes]
    mi = pd.MultiIndex(
        levels=levels, codes=codes, names=index_names, verify_integrity=False
    )
    return pd.DataFrame(vals, index=mi, columns=outs[0].columns, copy=False)


# ---------------------------------------------------------------------------
# S5: polars sorted + offsets -> zero-copy slice
# ---------------------------------------------------------------------------


def pl_offsets(df, keys):
    """Instance keys + offsets of a polars frame via run-length encoding.

    Returns (keys_df, offsets, lens, contiguous).
    """
    import polars as pl

    if len(keys) == 1:
        expr = pl.col(keys[0]).rle().alias("__r")
    else:
        expr = pl.struct(keys).rle().alias("__r")
    r = df.select(expr).unnest("__r")
    lens = r["len"].to_numpy().astype(np.int64)
    if len(keys) == 1:
        keys_df = r.select(pl.col("value").alias(keys[0]))
    else:
        keys_df = r.select("value").unnest("value")
    contiguous = keys_df.is_unique().all()
    offsets = np.zeros(len(lens), dtype=np.int64)
    np.cumsum(lens[:-1], out=offsets[1:])
    return keys_df, offsets, lens, contiguous


def pl_reconstruct_vec(outs, keys_df):
    """Polars reconstruct: one concat + one gather of the key frame."""
    import polars as pl

    lens = np.fromiter((o.height for o in outs), dtype=np.int64, count=len(outs))
    rep = np.repeat(np.arange(len(outs)), lens)
    body = pl.concat(outs, how="vertical", rechunk=True)
    kcols = keys_df[rep]
    return pl.concat([kcols, body], how="horizontal")


def pl_reconstruct_lit(outs, key_rows, key_names):
    """Polars reconstruct, naive: add literal key columns to each output."""
    import polars as pl

    return pl.concat(
        [
            o.with_columns([pl.lit(v).alias(n) for n, v in zip(key_names, k)])
            for k, o in zip(key_rows, outs)
        ],
        how="vertical",
        rechunk=True,
    )


# ---------------------------------------------------------------------------
# S6: numpy GroupedArray (values + indptr), as in statsforecast/utilsforecast
# ---------------------------------------------------------------------------


class GroupedArray:
    """2D float values + time array + indptr + instance keys.

    Slicing instance i is ``values[indptr[i]:indptr[i+1]]``, a numpy view.
    """

    __slots__ = ("values", "time", "indptr", "keys", "columns", "index_names")

    def __init__(self, values, time, indptr, keys, columns, index_names):
        self.values = values
        self.time = time
        self.indptr = indptr
        self.keys = keys
        self.columns = columns
        self.index_names = index_names

    def __len__(self):
        return len(self.indptr) - 1

    def __iter__(self):
        v, t, ip = self.values, self.time, self.indptr
        for i in range(len(ip) - 1):
            s, e = ip[i], ip[i + 1]
            yield t[s:e], v[s:e]

    def nbytes(self):
        return self.values.nbytes + self.time.nbytes + self.indptr.nbytes

    @classmethod
    def from_pandas_mi(cls, X, n_inst_levels):
        keys, indptr, order = pd_group_positions(X, n_inst_levels)
        if order is not None:
            X = X.take(order)
        values = X.to_numpy()
        time = X.index.get_level_values(-1).to_numpy()
        return cls(values, time, indptr, keys, list(X.columns), list(X.index.names))

    @classmethod
    def from_polars(cls, df, keys, time_col, value_cols):
        keys_df, offsets, lens, contiguous = pl_offsets(df, keys)
        if not contiguous:
            df = df.sort(keys, maintain_order=True)  # pragma: no cover
            keys_df, offsets, lens, contiguous = pl_offsets(df, keys)
        indptr = np.empty(len(lens) + 1, dtype=np.int64)
        indptr[:-1] = offsets
        indptr[-1] = df.height
        values = df.select(value_cols).to_numpy()
        time = df[time_col].to_numpy()
        return cls(values, time, indptr, keys_df, value_cols, keys + [time_col])

    @classmethod
    def from_outputs(cls, outs, keys, columns, index_names):
        """Reconstruct from per-instance (time, values) outputs."""
        lens = np.fromiter((len(t) for t, _ in outs), dtype=np.int64, count=len(outs))
        indptr = np.zeros(len(outs) + 1, dtype=np.int64)
        np.cumsum(lens, out=indptr[1:])
        values = np.concatenate([v for _, v in outs])
        time = np.concatenate([t for t, _ in outs])
        return cls(values, time, indptr, keys, columns, index_names)

    def to_pandas_mi(self):
        """Materialize as sktime pandas multiindex frame (codes-based, no hashing)."""
        lens = np.diff(self.indptr)
        rep = np.repeat(np.arange(len(lens)), lens)
        keys = self.keys
        if not isinstance(keys, (pd.Index, pd.MultiIndex)):
            keys = pd.MultiIndex.from_frame(keys.to_pandas()) if keys.width > 1 else (
                pd.Index(keys.to_series().to_numpy(), name=keys.columns[0])
            )
        tcodes, tuniq = pd.factorize(self.time, sort=False)
        if isinstance(keys, pd.MultiIndex):
            levels = list(keys.levels) + [pd.Index(tuniq)]
            codes = [np.asarray(c)[rep] for c in keys.codes] + [tcodes]
        else:
            levels = [keys, pd.Index(tuniq)]
            codes = [rep, tcodes]
        mi = pd.MultiIndex(
            levels=levels, codes=codes, names=self.index_names, verify_integrity=False
        )
        return pd.DataFrame(self.values, index=mi, columns=self.columns, copy=False)


# ---------------------------------------------------------------------------
# S8: pyarrow Table offsets + zero-copy slice
# ---------------------------------------------------------------------------


def pa_offsets(tab, keys):
    """Instance keys + offsets of an arrow Table (contiguous groups assumed+checked).

    Uses an order-preserving hash group_by (use_threads=False) for keys and counts,
    and run-end encoding of each key column to verify contiguity.
    """
    import pyarrow.compute as pc

    g = tab.group_by(keys, use_threads=False).aggregate([([], "count_all")])
    lens = g["count_all"].to_numpy().astype(np.int64)
    run_ends = None
    for k in keys:
        col = tab[k]
        if str(col.type) == "string_view":
            col = col.cast("large_string")
        ree = pc.run_end_encode(col.combine_chunks())
        re = ree.run_ends.to_numpy()
        run_ends = re if run_ends is None else np.union1d(run_ends, re)
    contiguous = len(run_ends) == g.num_rows
    offsets = np.zeros(len(lens), dtype=np.int64)
    np.cumsum(lens[:-1], out=offsets[1:])
    return g.select(keys), offsets, lens, contiguous


def pa_reconstruct(outs, keys_tab):
    import pyarrow as pa

    lens = np.fromiter((o.num_rows for o in outs), dtype=np.int64, count=len(outs))
    rep = pa.array(np.repeat(np.arange(len(outs)), lens))
    tab = pa.concat_tables(outs).combine_chunks()
    for i, name in enumerate(keys_tab.column_names):
        tab = tab.add_column(i, name, keys_tab[name].take(rep))
    return tab
