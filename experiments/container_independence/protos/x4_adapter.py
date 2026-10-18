"""x4: adapter/dispatcher-based ``VectorizedDF`` prototype.

Workstream x4_proto_adapter. NOT part of sktime; experiment code only.

Idea (1) from the design discussion: ``VectorizedDF`` does not talk to pandas
directly but to a ``ContainerAdapter`` dispatched on the native type of the
frame it is handed (sklearn ``ADAPTERS_MANAGER`` style). The pandas adapter is
behaviour-identical to the current ``sktime.datatypes._vectorize`` code
(MultiIndex + ``groupby(level=..., sort=False)`` + ``pd.concat(keys=...)``).

The protocol ops are derived from what ``VectorizedDF`` (items / as_list /
reconstruct / schema init / freq inference) and ``BaseForecaster._vectorize`` +
``_set_cutoff_from_y`` actually do with the frame:

=========================  ===============================================
protocol op                current pandas code it replaces
=========================  ===============================================
is_native                  isinstance(x, pd.DataFrame) in reconstruct
key_names                  X.index.names
value_columns              X.columns
n_rows                     len(X)
instance_keys(k)           X.index.droplevel(-1 / [-1,-2]).unique()
iter_groups(k)             X.groupby(level=..., sort=False) + droplevel
select_cols                inst[[col]]
finalize_slice             _enforce_index_freq(inst)
concat_rows_with_keys      pd.concat(parts, keys=row_ix, axis=0)
concat_cols                pd.concat(parts, axis=1, keys=col_keys)
flatten_columns            flatten_multiindex(X.columns)
set_key_names              X.index.set_names(...)
time_max                   get_cutoff(...)  (BaseForecaster._set_cutoff_from_y)
infer_freq                 coerce_to_multiindex.infer_freq
to_numpy                   X.values / to_numpy in numpy-based _fit code
=========================  ===============================================

Design decisions
----------------
* Dispatch: O(1) dict lookup on ``(type(obj).__module__ root, type(obj).__name__)``
  with an ``is_native`` scan fallback. sklearn dispatches on the module root
  only, which cannot tell ``pl.DataFrame`` from ``pl.LazyFrame``.
* Keys are opaque per adapter: ``instance_keys`` returns whatever the adapter
  likes (pandas: a ``pd.Index`` -> bit-identical to today; polars/pyarrow: a list
  of python scalars/tuples). ``VectorizedDF2`` only needs ``len`` + iteration.
* Level names are stored *logically* (``"h0"``, ``"time"``, ``None``); each adapter
  maps them to physical names. This lets reconstruct run in a different backend
  than the input (e.g. polars input, pandas-inner forecaster outputs).
* Polars key columns use the sktime ``__index__<name>`` convention, NOT plain key
  columns: (i) frames are self-describing, which a *stateless* schema needs, and
  (ii) a slice ``[__index__time, c0]`` is a valid sktime ``"pl.DataFrame"`` Series
  mtype that ``check_is_scitype`` inside a per-slice forecaster recognises. With
  plain key columns ``[time, c0]`` the time column would be read as a 2nd
  variable. Plain-key user frames (Nixtla-style ``unique_id, ds, y``) are handled
  by a zero-copy rename at the boundary (``from_key_columns``).
"""

from typing import Iterator, Protocol, Sequence, runtime_checkable

import numpy as np

from sktime.datatypes._vectorize import SERIES_SCITYPES, VectorizedDF

# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------


@runtime_checkable
class ContainerAdapter(Protocol):
    """Operations VectorizedDF / BaseForecaster vectorization need from a frame."""

    container_lib: str
    # mtype family of this backend, by scitype (for the inner-mtype bridge)
    mtypes: dict

    def is_native(self, obj) -> bool: ...  # noqa: D102

    def key_names(self, X) -> list: ...  # noqa: D102

    def value_columns(self, X) -> list: ...  # noqa: D102

    def n_rows(self, X) -> int: ...  # noqa: D102

    def instance_keys(self, X, k) -> Sequence: ...  # noqa: D102

    def iter_groups(self, X, k) -> Iterator: ...  # noqa: D102

    def select_cols(self, X, cols): ...  # noqa: D102

    def finalize_slice(self, X): ...  # noqa: D102

    def concat_rows_with_keys(self, parts, keys, names): ...  # noqa: D102

    def concat_cols(self, parts, keys=None): ...  # noqa: D102

    def flatten_columns(self, X): ...  # noqa: D102

    def set_key_names(self, X, names): ...  # noqa: D102

    def time_max(self, X): ...  # noqa: D102

    def infer_freq(self, X): ...  # noqa: D102

    def to_numpy(self, X): ...  # noqa: D102


PROTOCOL_METHODS = [
    "is_native",
    "key_names",
    "value_columns",
    "n_rows",
    "instance_keys",
    "iter_groups",
    "select_cols",
    "finalize_slice",
    "concat_rows_with_keys",
    "concat_cols",
    "flatten_columns",
    "set_key_names",
    "time_max",
    "infer_freq",
    "to_numpy",
]


# --------------------------------------------------------------------------
# registry / dispatcher
# --------------------------------------------------------------------------


class AdapterRegistry:
    """Dispatch a native frame to its adapter (sklearn ADAPTERS_MANAGER style)."""

    def __init__(self):
        self.by_lib = {}
        self._by_type = {}
        self._by_cls = {}
        self._order = []

    def register(self, adapter, type_keys=()):
        self.by_lib[adapter.container_lib] = adapter
        self._order.append(adapter)
        for tk in type_keys:
            self._by_type[tk] = adapter

    def get(self, obj):
        t = type(obj)
        ad = self._by_cls.get(t)  # fast path: exact class already seen
        if ad is not None:
            return ad
        key = (t.__module__.split(".")[0], t.__name__)
        ad = self._by_type.get(key)
        if ad is None:
            for cand in self._order:
                if cand.is_native(obj):
                    ad = cand
                    break
        if ad is None:
            raise TypeError(f"no ContainerAdapter registered for {t!r}")
        self._by_cls[t] = ad
        return ad

    def lib_of_mtype(self, mtype):
        for lib, ad in self.by_lib.items():
            if mtype in getattr(ad, "family", ()):
                return lib
        return None


# --------------------------------------------------------------------------
# pandas adapter (behaviour-identical to current VectorizedDF)
# --------------------------------------------------------------------------


class PandasAdapter:
    """MultiIndex-based adapter; mirrors sktime.datatypes._vectorize exactly."""

    container_lib = "pandas"
    mtypes = {
        "Series": "pd.DataFrame",
        "Panel": "pd-multiindex",
        "Hierarchical": "pd_multiindex_hier",
    }
    family = ("pd.Series", "pd.DataFrame", "pd-multiindex", "pd_multiindex_hier")

    def is_native(self, obj):
        import pandas as pd

        return isinstance(obj, (pd.DataFrame, pd.Series))

    def key_names(self, X):
        return list(X.index.names)

    def value_columns(self, X):
        return X.columns

    def n_rows(self, X):
        return len(X)

    def instance_keys(self, X, k):
        ix = X.index
        return ix.droplevel(list(range(k, ix.nlevels))).unique()

    def iter_groups(self, X, k):
        by = 0 if k == 1 else list(range(k))
        for name, group in X.groupby(level=by, sort=False):
            yield name, group.droplevel(by)

    def select_cols(self, X, cols):
        return X[list(cols)]

    def finalize_slice(self, X):
        from sktime.datatypes._vectorize import _enforce_index_freq

        return _enforce_index_freq(X)

    def concat_rows_with_keys(self, parts, keys, names):
        import pandas as pd

        if not isinstance(keys, pd.Index):
            keys = _keys_to_pd_index(keys, names)
        return pd.concat(parts, keys=keys, axis=0)

    def concat_cols(self, parts, keys=None):
        import pandas as pd

        return pd.concat(parts, axis=1, keys=keys)

    def flatten_columns(self, X):
        import pandas as pd

        from sktime.utils.multiindex import flatten_multiindex

        if isinstance(X.columns, pd.MultiIndex):
            X.columns = flatten_multiindex(X.columns)
        return X

    def set_key_names(self, X, names):
        if len(X.index.names) == len(names):
            X.index = X.index.set_names(names)
        return X

    def time_max(self, X):
        from sktime.datatypes import get_cutoff

        return get_cutoff(X, return_index=True)

    def infer_freq(self, X):
        import pandas as pd

        if len(X) == 0:
            return None
        idx = X.index
        if isinstance(idx, pd.MultiIndex):
            time_idx = X.xs(idx.droplevel(-1)[0]).index
        else:
            time_idx = idx
        if getattr(time_idx, "freqstr", None):
            return time_idx.freqstr
        try:
            return pd.infer_freq(time_idx)
        except (TypeError, ValueError):
            return None

    def to_numpy(self, X):
        return X.to_numpy()


def _keys_to_pd_index(keys, names):
    import pandas as pd

    keys = list(keys)
    if keys and isinstance(keys[0], tuple):
        return pd.MultiIndex.from_tuples(keys, names=names)
    return pd.Index(keys, name=names[0] if names else None)


# --------------------------------------------------------------------------
# polars eager adapter
# --------------------------------------------------------------------------

_IX = "__index__"


def _to_logical(phys_names):
    out = []
    for i, n in enumerate(phys_names):
        nm = n[len(_IX) :]
        out.append(None if nm == str(i) else nm)
    return out


def _to_physical(names, offset=0):
    return [
        f"{_IX}{i + offset}" if n is None else f"{_IX}{n}" for i, n in enumerate(names)
    ]


def _unwrap(key):
    # pandas groupby(level=<scalar>) yields scalar names; keep that convention
    return key[0] if isinstance(key, tuple) and len(key) == 1 else key


class PolarsAdapter:
    """polars.DataFrame with sktime ``__index__<name>`` key columns."""

    container_lib = "polars"
    mtypes = {
        "Series": "pl.DataFrame",
        "Panel": "polars_panel",
        "Hierarchical": "polars_hierarchical",
    }
    family = ("pl.DataFrame", "polars_panel", "polars_hierarchical")

    def is_native(self, obj):
        import polars as pl

        return isinstance(obj, pl.DataFrame)

    @staticmethod
    def _phys_keys(X):
        return [c for c in X.columns if isinstance(c, str) and c.startswith(_IX)]

    def key_names(self, X):
        return _to_logical(self._phys_keys(X))

    def value_columns(self, X):
        return [c for c in X.columns if not (isinstance(c, str) and c.startswith(_IX))]

    def n_rows(self, X):
        return X.height

    def instance_keys(self, X, k):
        keys = self._phys_keys(X)[:k]
        rows = X.select(keys).unique(maintain_order=True).rows()
        return [_unwrap(r) for r in rows]

    def iter_groups(self, X, k):
        keys = self._phys_keys(X)[:k]
        parts = X.partition_by(keys, maintain_order=True, include_key=False, as_dict=True)
        for key, part in parts.items():
            yield _unwrap(key), part

    def select_cols(self, X, cols):
        return X.select(self._phys_keys(X) + list(cols))

    def finalize_slice(self, X):
        return X

    def concat_rows_with_keys(self, parts, keys, names):
        import polars as pl

        phys = _to_physical(names)
        keys = list(keys)
        rows = [k if isinstance(k, tuple) else (k,) for k in keys]
        key_df = pl.DataFrame(rows, schema=phys, orient="row")
        heights = np.fromiter((p.height for p in parts), dtype=np.int64, count=len(parts))
        body = pl.concat(parts, how="vertical_relaxed")
        rep = key_df[np.repeat(np.arange(len(parts)), heights)]
        return pl.concat([rep, body], how="horizontal")

    def concat_cols(self, parts, keys=None):
        """Column-concat; polars has no column MultiIndex, keys give flat names.

        ``keys`` present -> columns become ``f"{key}__{col}"`` (same as pandas
        ``multiindex`` followed by ``flatten_multiindex``).
        """
        import polars as pl

        key_cols = self._phys_keys(parts[0])
        if keys is not None:
            parts = [
                p.rename({c: f"{k}__{c}" for c in self.value_columns(p)})
                for p, k in zip(parts, keys)
            ]
        same_keys = all(
            self._phys_keys(p) == key_cols
            and p.height == parts[0].height
            and p.select(key_cols).equals(parts[0].select(key_cols))
            for p in parts[1:]
        )
        if same_keys:
            vals = [p.select(self.value_columns(p)) for p in parts]
            return pl.concat([parts[0].select(key_cols), *vals], how="horizontal")
        # misaligned row keys: outer-join on keys (pandas concat(axis=1) aligns)
        return pl.concat(parts, how="align_full")

    def flatten_columns(self, X):
        return X

    def set_key_names(self, X, names):
        old = self._phys_keys(X)
        if len(old) != len(names):
            return X
        new = _to_physical(names)
        return X.rename(dict(zip(old, new)))

    def time_max(self, X):
        return X.get_column(self._phys_keys(X)[-1]).max()

    def infer_freq(self, X):
        """Freq from the first instance's time column (O(T) metadata, not O(N*T))."""
        import pandas as pd
        import polars as pl

        if X.height == 0:
            return None
        keys = self._phys_keys(X)
        if not keys:
            return None
        t = keys[-1]
        if len(keys) > 1:
            first = X.select(keys[:-1]).row(0)
            pred = pl.all_horizontal([pl.col(c) == v for c, v in zip(keys[:-1], first)])
            tv = X.lazy().filter(pred).select(t).collect().get_column(t)
        else:
            tv = X.get_column(t)
        if tv.dtype.is_temporal():
            try:
                return pd.infer_freq(pd.DatetimeIndex(tv.to_numpy()))
            except (TypeError, ValueError):
                return None
        return None

    def to_numpy(self, X):
        vc = self.value_columns(X)
        if len(vc) == 1:  # zero-copy for null-free numeric columns
            return X.get_column(vc[0]).to_numpy()[:, None]
        return X.select(vc).to_numpy()

    # boundary helper: user frames with plain key columns ------------------
    @staticmethod
    def from_key_columns(df, keys):
        """Rename plain key columns (e.g. Nixtla ``unique_id, ds``) to sktime keys."""
        return df.rename({k: f"{_IX}{k}" for k in keys})


# --------------------------------------------------------------------------
# polars lazy adapter
# --------------------------------------------------------------------------


class PolarsLazyAdapter:
    """polars.LazyFrame. Schema ops never collect data; key discovery collects
    only the key columns; slices are either lazy filters or eager partitions.

    ``slice_mode``:

    * ``"lazy"`` -- each slice is ``X.filter(key == k).drop(keys)``, still lazy.
      Consumers that need data collect each slice => N query executions.
    * ``"collect_once"`` -- collect the full frame once, then eager partition_by
      (materialization boundary == iteration boundary).
    """

    container_lib = "polars_lazy"
    mtypes = {"Series": None, "Panel": None, "Hierarchical": None}  # no sktime mtype
    family = ()

    def __init__(self, slice_mode="collect_once"):
        self.slice_mode = slice_mode
        self._eager = PolarsAdapter()

    def is_native(self, obj):
        import polars as pl

        return isinstance(obj, pl.LazyFrame)

    @staticmethod
    def _names(X):
        return X.collect_schema().names()

    def _phys_keys(self, X):
        return [c for c in self._names(X) if isinstance(c, str) and c.startswith(_IX)]

    def key_names(self, X):
        return _to_logical(self._phys_keys(X))

    def value_columns(self, X):
        return [c for c in self._names(X) if not (isinstance(c, str) and c.startswith(_IX))]

    def n_rows(self, X):
        import polars as pl

        return X.select(pl.len()).collect().item()

    def instance_keys(self, X, k):
        keys = self._phys_keys(X)[:k]
        rows = X.select(keys).unique(maintain_order=True).collect().rows()
        return [_unwrap(r) for r in rows]

    def iter_groups(self, X, k):
        import polars as pl

        keys = self._phys_keys(X)[:k]
        if self.slice_mode == "collect_once":
            yield from self._eager.iter_groups(X.collect(), k)
            return
        for key in self.instance_keys(X, k):
            kt = key if isinstance(key, tuple) else (key,)
            pred = pl.all_horizontal([pl.col(c) == v for c, v in zip(keys, kt)])
            yield key, X.filter(pred).drop(keys)

    def select_cols(self, X, cols):
        return X.select(self._phys_keys(X) + list(cols))

    def finalize_slice(self, X):
        return X

    def concat_rows_with_keys(self, parts, keys, names):
        import polars as pl

        phys = _to_physical(names)
        out = []
        for p, k in zip(parts, keys):
            kt = k if isinstance(k, tuple) else (k,)
            out.append(p.lazy().with_columns([pl.lit(v).alias(n) for n, v in zip(phys, kt)]))
        body = pl.concat(out, how="vertical_relaxed")
        rest = [c for c in body.collect_schema().names() if c not in phys]
        return body.select(phys + rest)

    def concat_cols(self, parts, keys=None):
        import polars as pl

        return self._eager.concat_cols(pl.collect_all(parts), keys).lazy()

    def flatten_columns(self, X):
        return X

    def set_key_names(self, X, names):
        old = self._phys_keys(X)
        if len(old) != len(names):
            return X
        return X.rename(dict(zip(old, _to_physical(names))))

    def time_max(self, X):
        import polars as pl

        return X.select(pl.col(self._phys_keys(X)[-1]).max()).collect().item()

    def infer_freq(self, X):
        keys = self._phys_keys(X)
        import polars as pl

        if len(keys) > 1:
            first = X.select(keys[:-1]).head(1).collect().row(0)
            pred = pl.all_horizontal([pl.col(c) == v for c, v in zip(keys[:-1], first)])
            sub = X.filter(pred).select(keys).collect()
        else:
            sub = X.select(keys).collect()
        return self._eager.infer_freq(sub)

    def to_numpy(self, X):
        return X.select(self.value_columns(X)).collect().to_numpy()


# --------------------------------------------------------------------------
# pyarrow adapter (optional; shows marginal cost of a backend w/o groupby-iter)
# --------------------------------------------------------------------------


class PyArrowAdapter:
    """pyarrow.Table with ``__index__`` key columns.

    pyarrow has no "iterate groups" API: groups are found as runs of equal keys
    (sktime frames are instance-contiguous), slices are zero-copy ``Table.slice``.
    Non-contiguous input falls back to a stable sort on first-appearance codes.
    """

    container_lib = "pyarrow"
    mtypes = {"Series": None, "Panel": None, "Hierarchical": None}  # no sktime mtype
    family = ()

    def is_native(self, obj):
        import pyarrow as pa

        return isinstance(obj, pa.Table)

    @staticmethod
    def _phys_keys(X):
        return [c for c in X.column_names if c.startswith(_IX)]

    def key_names(self, X):
        return _to_logical(self._phys_keys(X))

    def value_columns(self, X):
        return [c for c in X.column_names if not c.startswith(_IX)]

    def n_rows(self, X):
        return X.num_rows

    def _runs(self, X, k):
        """Return (starts, lengths, keys) of contiguous key runs; sort if needed."""
        import pyarrow as pa
        import pyarrow.compute as pc

        keys = self._phys_keys(X)[:k]
        n = X.num_rows
        if n == 0:
            return X, np.array([], dtype=np.int64), np.array([], dtype=np.int64), []
        change = None
        for c in keys:
            col = X.column(c)
            neq = pc.not_equal(col.slice(1), col.slice(0, n - 1))
            change = neq if change is None else pc.or_(change, neq)
        starts = np.concatenate(
            [[0], pc.indices_nonzero(change).to_numpy().astype(np.int64) + 1]
        )
        n_unique = X.select(keys).group_by(keys).aggregate([]).num_rows
        if len(starts) != n_unique:  # not contiguous: stable sort by first appearance
            codes = pc.dictionary_encode(
                pc.binary_join_element_wise(
                    *[pc.cast(X.column(c), pa.string()) for c in keys], "\x1f"
                )
            ).combine_chunks().indices
            X = X.take(pc.sort_indices(codes, sort_keys=[("", "ascending")]))
            return self._runs(X, k)
        lengths = np.diff(np.concatenate([starts, [n]]))
        key_tbl = X.select(keys).take(pa.array(starts))
        cols = [key_tbl.column(c).to_pylist() for c in keys]
        key_list = [_unwrap(t) for t in zip(*cols)]
        return X, starts, lengths, key_list

    def instance_keys(self, X, k):
        return self._runs(X, k)[3]

    def iter_groups(self, X, k):
        keys = self._phys_keys(X)[:k]
        X, starts, lengths, key_list = self._runs(X, k)
        body = X.drop_columns(keys)
        for s, ln, key in zip(starts, lengths, key_list):
            yield key, body.slice(int(s), int(ln))

    def select_cols(self, X, cols):
        return X.select(self._phys_keys(X) + list(cols))

    def finalize_slice(self, X):
        return X

    def concat_rows_with_keys(self, parts, keys, names):
        import pyarrow as pa

        phys = _to_physical(names)
        rows = [k if isinstance(k, tuple) else (k,) for k in keys]
        heights = np.fromiter((p.num_rows for p in parts), dtype=np.int64, count=len(parts))
        take = pa.array(np.repeat(np.arange(len(parts)), heights))
        body = pa.concat_tables(parts, promote_options="permissive")
        key_arrays = [pa.array([r[i] for r in rows]).take(take) for i in range(len(phys))]
        for i, (n, a) in enumerate(zip(phys, key_arrays)):
            body = body.add_column(i, n, a)
        return body

    def concat_cols(self, parts, keys=None):
        key_cols = self._phys_keys(parts[0])
        out = parts[0].select(key_cols)
        for j, p in enumerate(parts):
            for c in self.value_columns(p):
                name = f"{keys[j]}__{c}" if keys is not None else c
                out = out.append_column(name, p.column(c))
        return out

    def flatten_columns(self, X):
        return X

    def set_key_names(self, X, names):
        old = self._phys_keys(X)
        if len(old) != len(names):
            return X
        new = dict(zip(old, _to_physical(names)))
        return X.rename_columns([new.get(c, c) for c in X.column_names])

    def time_max(self, X):
        import pyarrow.compute as pc

        return pc.max(X.column(self._phys_keys(X)[-1])).as_py()

    def infer_freq(self, X):
        return None  # not implemented in the prototype

    def to_numpy(self, X):
        return np.column_stack(
            [X.column(c).to_numpy() for c in self.value_columns(X)]
        )


REGISTRY = AdapterRegistry()
REGISTRY.register(PandasAdapter(), [("pandas", "DataFrame"), ("pandas", "Series")])
REGISTRY.register(PolarsAdapter(), [("polars", "DataFrame")])
REGISTRY.register(PolarsLazyAdapter(), [("polars", "LazyFrame")])
REGISTRY.register(PyArrowAdapter(), [("pyarrow", "Table")])


def get_adapter(obj):
    """Return the ContainerAdapter for native frame ``obj``."""
    return REGISTRY.get(obj)


# --------------------------------------------------------------------------
# VectorizedDF2: schema on top of the adapter
# --------------------------------------------------------------------------


class VectorizedDF2(VectorizedDF):
    """Backend-agnostic stateless vectorization schema.

    Subclasses ``VectorizedDF`` only to inherit ``vectorize_est`` *unchanged*
    (and to pass its ``isinstance(e, VectorizedDF)`` check). All frame access is
    routed through ``get_adapter(frame)``; the schema keeps no pandas objects
    unless the input was pandas.
    """

    def __init__(
        self,
        X,
        iterate_as="Series",
        is_scitype="Panel",
        iterate_cols=False,
        X_orig_mtype=None,
        converter_store=None,
        freq=None,
    ):
        if is_scitype is None or is_scitype not in SERIES_SCITYPES:
            raise ValueError(f"is_scitype must be resolved, found: {is_scitype}")
        self.is_scitype = is_scitype
        self._check_iterate_as(iterate_as)
        self.iterate_as = iterate_as
        self._check_iterate_cols(iterate_cols)
        self.iterate_cols = iterate_cols
        self.X_orig_mtype = X_orig_mtype
        self.converter_store = converter_store if converter_store is not None else {}

        ad = get_adapter(X)
        self.container_lib = ad.container_lib
        self.key_names = ad.key_names(X)  # logical names, all levels
        self.X_mi_index_names = self.key_names
        self.X_mi_nlevels = len(self.key_names)
        self.X_mi_columns = ad.value_columns(X)
        k = len(self._iter_levels(iterate_as))
        row_keys = None if k == self.X_mi_nlevels else ad.instance_keys(X, k)
        col_keys = self.X_mi_columns if iterate_cols else None
        self.iter_indices = (row_keys, col_keys)
        self.shape = self._iter_shape()
        self.freq = freq

    def get_iter_indices(self):
        """Pandas view of the keys (compat with _base_proba / users)."""
        import pandas as pd

        row, col = self.iter_indices
        if row is not None and not isinstance(row, pd.Index):
            k = len(self._iter_levels(self.iterate_as))
            row = _keys_to_pd_index(row, self.key_names[:k])
        if col is not None and not isinstance(col, pd.Index):
            col = pd.Index(col)
        return row, col

    def items(self, X=None, iterate_as=None, iterate_cols=None):
        """Yield (row key, col key, slice), same order/semantics as VectorizedDF."""
        if iterate_as is None:
            iterate_as = self.iterate_as
        self._check_iterate_as(iterate_as)
        if iterate_cols is None:
            iterate_cols = self.iterate_cols
        self._check_iterate_cols(iterate_cols)

        if X is None:
            yield from super().items(X=None, iterate_as=iterate_as, iterate_cols=iterate_cols)
            return

        ad = get_adapter(X)

        def _iter_cols(inst, group_name=None):
            if iterate_cols:
                for col in ad.value_columns(inst):
                    yield group_name, col, ad.finalize_slice(ad.select_cols(inst, [col]))
            else:
                yield group_name, None, ad.finalize_slice(inst)

        k = len(self._iter_levels(iterate_as))
        if k == self.X_mi_nlevels:
            yield from _iter_cols(X)
        else:
            for name, group in ad.iter_groups(X, k):
                yield from _iter_cols(group, group_name=name)

    def reconstruct(
        self, df_list, convert_back=False, overwrite_index=True, col_multiindex="none"
    ):
        """Assemble per-slice outputs; backend is dispatched on the OUTPUTS."""
        from sktime.datatypes import convert_to

        df_list = list(df_list)
        ad = None
        for x in df_list:
            try:
                ad = get_adapter(x)
                break
            except TypeError:
                continue
        if ad is None:
            raise TypeError("reconstruct: no adapter for outputs")
        if ad.container_lib == "pandas":
            import pandas as pd

            df_list = [
                x
                if isinstance(x, pd.DataFrame)
                else self._coerce_to_df(x)  # current behaviour for pd.Series etc
                for x in df_list
            ]

        def _force_flat(blk):
            cols = [list(ad.value_columns(x)) for x in blk]
            ff = len(blk) > 1 and any(len(c) > 1 for c in cols)
            allc = [c for cs in cols for c in cs]
            return ff or len(set(allc)) != len(allc)

        row_ix, col_ix = self.iter_indices
        k = len(self._iter_levels(self.iterate_as))
        names = self.key_names[:k]
        force_flat = False
        if row_ix is None and col_ix is None:
            out = df_list[0]
            if ad.container_lib == "pandas":
                import pandas as pd

                out = pd.DataFrame(out)
        elif col_ix is None:
            out = ad.concat_rows_with_keys(df_list, row_ix, names)
        elif row_ix is None:
            force_flat = _force_flat(df_list)
            use_keys = col_multiindex in ["flat", "multiindex"] or force_flat
            out = ad.concat_cols(df_list, keys=col_ix if use_keys else None)
        else:
            col_n = len(col_ix)
            blocks = []
            for i in range(len(row_ix)):
                blk = df_list[i * col_n : (i + 1) * col_n]
                force_flat = force_flat or _force_flat(blk)
                use_keys = col_multiindex in ["flat", "multiindex"] or force_flat
                blocks.append(ad.concat_cols(blk, keys=col_ix if use_keys else None))
            out = ad.concat_rows_with_keys(blocks, row_ix, names)

        flatten = col_multiindex == "flat" or (col_multiindex == "none" and force_flat)
        if flatten:
            out = ad.flatten_columns(out)
        if overwrite_index:
            out = ad.set_key_names(out, self.key_names)
        if not convert_back:
            return out
        return convert_to(
            out,
            to_type=self.X_orig_mtype,
            as_scitype=self.is_scitype,
            store=self.converter_store,
        )


# --------------------------------------------------------------------------
# prepare: native dispatch, or convert once into a target backend family
# --------------------------------------------------------------------------


def prepare_VectorizedDF2(
    X,
    iterate_as="Series",
    is_scitype="Panel",
    iterate_cols=False,
    store=None,
    store_behaviour=None,
    target="native",
    X_orig_mtype=None,
):
    """Build (schema, frame) with frame in the adapter backend chosen by ``target``.

    target : "native" -> iterate in the input's own backend if an adapter + mtype
                family exists for it, else fall back to pandas (current behaviour)
             "pandas" -> always convert once to pandas multiindex (== current)
             "<lib>"  -> convert once to that backend's mtype family (the
                         "inner-family bridge": lib = family of y_inner_mtype)
    """
    from sktime.datatypes import check_is_scitype, convert_to, mtype

    if store is None:
        store = {}
    if is_scitype is None:
        _, _, md = check_is_scitype(X, scitype=SERIES_SCITYPES, return_metadata=[])
        is_scitype, X_orig_mtype = md["scitype"], md["mtype"]
    elif X_orig_mtype is None:
        X_orig_mtype = mtype(X, as_scitype=is_scitype)

    if target == "native":
        lib = REGISTRY.lib_of_mtype(X_orig_mtype) or "pandas"
    else:
        lib = target
    ad = REGISTRY.by_lib[lib]
    to_mtype = ad.mtypes[is_scitype]
    if X_orig_mtype == to_mtype:
        X_native = X
    else:
        X_native = convert_to(
            X,
            to_type=to_mtype,
            as_scitype=is_scitype,
            store=store,
            store_behaviour=store_behaviour,
        )
    freq = ad.infer_freq(X_native)
    vec = VectorizedDF2(
        X=X_native,
        iterate_as=iterate_as,
        is_scitype=is_scitype,
        iterate_cols=iterate_cols,
        X_orig_mtype=X_orig_mtype,
        converter_store=store,
        freq=freq,
    )
    return vec, X_native


def family_of_inner_mtype(inner_mtypes):
    """Inner-mtype bridge: which backend family does a forecaster compute in?"""
    if isinstance(inner_mtypes, str):
        inner_mtypes = [inner_mtypes]
    for m in inner_mtypes:
        lib = REGISTRY.lib_of_mtype(m)
        if lib is not None:
            return lib
    return "pandas"


# --------------------------------------------------------------------------
# backend-aware cutoff (bridge for BaseForecaster._set_cutoff_from_y)
# --------------------------------------------------------------------------


def adapter_get_cutoff(obj, cutoff=0, return_index=False, **kwargs):
    """``get_cutoff`` that does not convert non-pandas frames to pandas.

    For polars, computes ``max(time)`` in polars and only wraps the single
    resulting scalar in a ``pd.Index`` (sktime's cutoff type). Freq is taken
    from the time column of the (first) instance -- O(T) metadata.
    """
    import pandas as pd

    from sktime.datatypes import get_cutoff

    try:
        ad = get_adapter(obj)
    except TypeError:
        ad = None
    if ad is None or ad.container_lib == "pandas":
        return get_cutoff(obj, cutoff, return_index=return_index, **kwargs)
    val = ad.time_max(obj)
    if not return_index:
        return val
    idx = pd.Index([val])
    freq = ad.infer_freq(obj)
    if freq is not None and isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx, freq=freq)
    return idx


# convenient introspection for the report ---------------------------------
ADAPTER_CLASSES = [PandasAdapter, PolarsAdapter, PolarsLazyAdapter, PyArrowAdapter]

__all__ = [
    "ContainerAdapter",
    "AdapterRegistry",
    "PandasAdapter",
    "PolarsAdapter",
    "PolarsLazyAdapter",
    "PyArrowAdapter",
    "REGISTRY",
    "get_adapter",
    "VectorizedDF2",
    "prepare_VectorizedDF2",
    "family_of_inner_mtype",
    "adapter_get_cutoff",
    "PROTOCOL_METHODS",
    "ADAPTER_CLASSES",
]
