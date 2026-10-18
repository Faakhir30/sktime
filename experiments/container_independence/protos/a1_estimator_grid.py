"""a1 prototype: container-free estimator grid for ``forecasters_`` / ``transformers_``.

``VectorizedDF.vectorize_est`` currently returns the fitted estimators as a
``pd.DataFrame`` built via ``pd.DataFrame(long).pivot(...).reindex(...)`` and
``pd.MultiIndex.from_tuples``. That DataFrame is only ever used as a 2D keyed
container of Python objects. This module provides:

* ``EstimatorGrid``: row-major list of cells + ordered row/col keys, O(1) key
  lookup, no pandas. ``to_pandas()`` rebuilds the current DataFrame lazily.
* ``PandasCompatGrid``: ``EstimatorGrid`` + read-only pandas-like shims
  (``shape``, ``index``, ``columns``, ``loc[r, c]``, ``iloc[i, j]``, ``iterrows``,
  ``values``) so existing call sites / user code keep working.
* ``vectorize_est_grid``: copy of ``VectorizedDF.vectorize_est`` (sktime/datatypes/
  _vectorize.py) whose only changes are (a) accepting an ``EstimatorGrid`` as
  input, (b) building an ``EstimatorGrid`` instead of the pivot DataFrame.
* ``vectorize_forecaster_grid`` / ``get_fitted_params_grid``: copies of the
  ``BaseForecaster._vectorize`` / ``get_fitted_params`` vectorized branches using
  only the grid API (no pandas).
* ``install(mode)``: monkeypatches these into sktime (mode "strict" or "compat").

Nothing under sktime/ is modified on disk.
"""

import itertools

_GRID_CLS = None  # set by install()


class EstimatorGrid:
    """2D grid of estimators keyed by (row_key, col_key); container-free."""

    __slots__ = ("_cells", "row_keys", "col_keys", "_rpos", "_cpos")

    def __init__(self, cells, row_keys, col_keys):
        row_keys = list(row_keys)
        col_keys = list(col_keys)
        if len(cells) != len(row_keys) * len(col_keys):
            raise ValueError("cells must be row-major with len(rows)*len(cols) items")
        self._cells = list(cells)
        self.row_keys = row_keys
        self.col_keys = col_keys
        self._rpos = {k: i for i, k in enumerate(row_keys)}
        self._cpos = {k: j for j, k in enumerate(col_keys)}

    @classmethod
    def from_long(cls, triples):
        """Build from row-major (row_key, col_key, value) triples."""
        triples = list(triples)
        rows = list(dict.fromkeys(r for r, _, _ in triples))
        cols = list(dict.fromkeys(c for _, c, _ in triples))
        return cls([v for _, _, v in triples], rows, cols)

    # --- container-free API -------------------------------------------------
    @property
    def shape(self):
        return len(self.row_keys), len(self.col_keys)

    def __len__(self):
        return len(self._cells)

    def cells(self):
        """Cells in row-major order (the order vectorize_est consumes them)."""
        return list(self._cells)

    def items(self):
        """Iterate (row_key, col_key, estimator) in row-major order."""
        it = itertools.product(self.row_keys, self.col_keys)
        for (r, c), v in zip(it, self._cells):
            yield r, c, v

    def get(self, row, col):
        return self._cells[self._rpos[row] * len(self.col_keys) + self._cpos[col]]

    def to_pandas(self):
        """Rebuild the pd.DataFrame that vectorize_est returns today."""
        import numpy as np
        import pandas as pd

        vals = np.empty(self.shape, dtype=object)
        for i, j in itertools.product(range(self.shape[0]), range(self.shape[1])):
            vals[i, j] = self._cells[i * self.shape[1] + j]
        idx = pd.Index(self.row_keys, tupleize_cols=False)
        try:
            idx = pd.MultiIndex.from_tuples(self.row_keys)
        except (TypeError, ValueError):
            pass
        return pd.DataFrame(vals, index=idx, columns=pd.Index(self.col_keys))

    def __repr__(self):
        return f"EstimatorGrid(shape={self.shape})"


class _Loc:
    def __init__(self, g, positional):
        self.g = g
        self.positional = positional

    def __getitem__(self, key):
        r, c = key
        if self.positional:
            if isinstance(c, slice):
                return [self.g._cells[r * self.g.shape[1] + j]
                        for j in range(self.g.shape[1])[c]]
            return self.g._cells[r * self.g.shape[1] + c]
        return self.g.get(r, c)


class PandasCompatGrid(EstimatorGrid):
    """EstimatorGrid with read-only pandas-like shims for backwards compatibility."""

    __slots__ = ()

    @property
    def index(self):
        return self.to_pandas().index

    @property
    def columns(self):
        return self.to_pandas().columns

    @property
    def loc(self):
        return _Loc(self, positional=False)

    @property
    def iloc(self):
        return _Loc(self, positional=True)

    @property
    def values(self):
        return self.to_pandas().values

    def iterrows(self):
        return self.to_pandas().iterrows()


# ---------------------------------------------------------------------------
# patched copies of sktime methods
# ---------------------------------------------------------------------------
def vectorize_est_grid(
    self,
    estimator,
    method="clone",
    args=None,
    args_rowvec=None,
    return_type="pd.DataFrame",
    rowname_default="estimators",
    colname_default="estimators",
    varname_of_self=None,
    pass_self_slice=False,
    backend=None,
    backend_params=None,
    **kwargs,
):
    """Copy of VectorizedDF.vectorize_est returning an EstimatorGrid (see module)."""
    import pandas as pd

    from sktime.datatypes._vectorize import VectorizedDF
    from sktime.utils.parallel import parallelize

    iterate_as = self.iterate_as
    iterate_cols = self.iterate_cols

    if pass_self_slice and varname_of_self is None:
        raise ValueError(
            "Error at vectorize_est: pass_self_slice requires varname_of_self"
        )
    if args is None:
        args = kwargs
    else:
        args = args.copy()
        args.update(kwargs)
    if args_rowvec is None:
        args_rowvec = {}
    if return_type not in {"pd.DataFrame", "list"}:
        raise ValueError('return_type must be one of "pd.DataFrame" or "list"')
    if varname_of_self and not isinstance(varname_of_self, str):
        raise TypeError("varname_of_self must be a string")

    def _as_data_schema(value):
        if isinstance(value, tuple) and len(value) == 2:
            return value
        return value, None

    def explode(d, iterate_as, iterate_cols):
        if not d:
            yield from itertools.cycle([{}])

        def _to_iter(value):
            X, e = _as_data_schema(value)
            if isinstance(e, VectorizedDF):
                it = (
                    inst
                    for _, _, inst in e.items(
                        X=X, iterate_as=iterate_as, iterate_cols=iterate_cols
                    )
                )
                if self.iterate_cols and not iterate_cols:
                    it = itertools.chain.from_iterable(
                        itertools.repeat(el, self.shape[1]) for el in it
                    )
                return it
            else:
                return itertools.cycle([X])

        keys, values = zip(*d.items())
        for values_inst in zip(*map(_to_iter, values)):
            yield dict(zip(keys, values_inst))

    # --- change (a): accept grid (and, for compat, DataFrame) as input -------
    if isinstance(estimator, EstimatorGrid):
        if estimator.shape != self.shape:
            raise ValueError(
                f"The estimator grid must have the same shape as self. "
                f"Expected {self.shape}, found {estimator.shape}"
            )
        estimators = iter(estimator.cells())
    elif isinstance(estimator, pd.DataFrame):
        if estimator.shape != self.shape:
            raise ValueError(
                f"The estimator data frame must have the same shape as self. "
                f"Expected {self.shape}, found {estimator.shape}"
            )
        estimators = (cell for _, row in estimator.iterrows() for cell in row.values)
    else:
        estimators = itertools.cycle([estimator])

    vec_zip = zip(
        self.items(X=_as_data_schema(kwargs.get(varname_of_self))[0]),
        explode(args, iterate_as=iterate_as, iterate_cols=iterate_cols),
        explode(args_rowvec, iterate_as=iterate_as, iterate_cols=False),
        estimators,
    )
    meta = {
        "method": method,
        "varname_of_self": varname_of_self,
        "pass_self_slice": pass_self_slice,
        "rowname_default": rowname_default,
        "colname_default": colname_default,
    }
    ret = parallelize(
        fun=self._vectorize_est_single,
        iter=vec_zip,
        meta=meta,
        backend=backend,
        backend_params=backend_params,
    )

    # --- change (b): build grid instead of pivot DataFrame -------------------
    if return_type == "pd.DataFrame":
        return _GRID_CLS.from_long(ret)
    return [result for _, _, result in ret]


def vectorize_forecaster_grid(self, methodname, **kwargs):
    """Copy of BaseForecaster._vectorize using only the grid API."""
    FIT_METHODS = ["fit", "update"]
    PREDICT_METHODS = [
        "predict",
        "update_predict_single",
        "predict_quantiles",
        "predict_interval",
        "predict_var",
    ]
    X = kwargs.pop("X", None)
    y = kwargs.get("y", None)
    y_schema = y[1] if y else None
    kwargs["args_rowvec"] = {"X": X}
    kwargs["rowname_default"] = "forecasters"
    kwargs["colname_default"] = "forecasters"
    kwargs["varname_of_self"] = "y"
    backend = self.get_config()["backend:parallel"]
    backend_params = self.get_config()["backend:parallel:params"]

    if methodname in FIT_METHODS:
        self._y_schema = y_schema
        if methodname == "fit":
            forecasters_ = self._y_schema.vectorize_est(
                self,
                method="clone",
                rowname_default="forecasters",
                colname_default="forecasters",
                backend=backend,
                backend_params=backend_params,
            )
        else:
            forecasters_ = self.forecasters_
        self.forecasters_ = self._y_schema.vectorize_est(
            forecasters_,
            method=methodname,
            backend=backend,
            backend_params=backend_params,
            **kwargs,
        )
        return self
    elif methodname in PREDICT_METHODS:
        if methodname == "update_predict_single":
            self._y_schema = y_schema
        y_preds = self._y_schema.vectorize_est(
            self.forecasters_,
            method=methodname,
            return_type="list",
            backend=backend,
            backend_params=backend_params,
            **kwargs,
        )
        m = self.forecasters_.shape[1]  # was: len(self.forecasters_.columns)
        col_multiindex = "multiindex" if m > 1 else "none"
        y_pred = self._y_schema.reconstruct(
            y_preds, overwrite_index=True, col_multiindex=col_multiindex
        )
        if col_multiindex == "multiindex":
            y_pred.columns = y_pred.columns.droplevel(1)
        return y_pred


def _grid_fitted_params(self, grid, attr_name, deep):
    def _to_str(x):
        if isinstance(x, str):
            x = f"'{x}'"
        return str(x)

    fitted_params = {attr_name: grid}
    for ix, col, est in grid.items():
        key = f"{attr_name}.loc[{_to_str(ix)},{_to_str(col)}]"  # naming kept
        fitted_params[key] = est
        for k, v in est.get_fitted_params(deep=deep).items():
            fitted_params[f"{key}__{k}"] = v
    return fitted_params


def install(mode="strict"):
    """Monkeypatch sktime to use EstimatorGrid. mode: "strict" or "compat"."""
    global _GRID_CLS
    import skbase.base  # noqa: F401

    from sktime.datatypes._vectorize import VectorizedDF
    from sktime.forecasting.base._base import BaseForecaster

    _GRID_CLS = EstimatorGrid if mode == "strict" else PandasCompatGrid
    VectorizedDF.vectorize_est = vectorize_est_grid
    BaseForecaster._vectorize = vectorize_forecaster_grid

    orig_gfp = BaseForecaster.get_fitted_params

    def get_fitted_params(self, deep=True):
        if not getattr(self, "_is_vectorized", False):
            return orig_gfp(self, deep=deep)
        fp = _grid_fitted_params(self, self.forecasters_, "forecasters", deep)
        if hasattr(self, "_pretrained_attrs"):
            for attr in self._pretrained_attrs:
                fp.pop(attr, None)
        return fp

    BaseForecaster.get_fitted_params = get_fitted_params
    return _GRID_CLS
