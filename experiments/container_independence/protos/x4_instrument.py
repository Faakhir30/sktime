"""x4: instrumentation + monkeypatch helpers for the forecaster experiments.

* ``ConvCounter``: counts every pandas<->polars data conversion by wrapping the two
  functions every sktime polars converter funnels through
  (``pl.DataFrame.to_pandas`` and ``polars.from_pandas``), plus rows moved.
* ``patched_base(policy, cutoff)``: swaps ``prepare_VectorizedDF`` / ``get_cutoff``
  *as seen by* ``sktime.forecasting.base._base`` (no file under sktime/ changes).
"""

import contextlib

import polars as pl


class ConvCounter:
    """Count pandas<->polars conversions (calls and rows)."""

    def __init__(self):
        self.to_pandas = 0
        self.from_pandas = 0
        self.rows_to_pandas = 0
        self.rows_from_pandas = 0
        self._orig_tp = pl.DataFrame.to_pandas
        self._orig_fp = pl.from_pandas

    def __enter__(self):
        cnt = self
        orig_tp, orig_fp = self._orig_tp, self._orig_fp

        def tp(df, *a, **k):
            cnt.to_pandas += 1
            cnt.rows_to_pandas += df.height
            return orig_tp(df, *a, **k)

        def fp(obj, *a, **k):
            cnt.from_pandas += 1
            cnt.rows_from_pandas += len(obj)
            return orig_fp(obj, *a, **k)

        pl.DataFrame.to_pandas = tp
        pl.from_pandas = fp
        return self

    def __exit__(self, *exc):
        pl.DataFrame.to_pandas = self._orig_tp
        pl.from_pandas = self._orig_fp

    def as_dict(self):
        return {
            "to_pandas_calls": self.to_pandas,
            "from_pandas_calls": self.from_pandas,
            "rows_to_pandas": self.rows_to_pandas,
            "rows_from_pandas": self.rows_from_pandas,
        }


@contextlib.contextmanager
def patched_base(policy="current", cutoff=False):
    """Route BaseForecaster vectorization through VectorizedDF2.

    policy : "current" -> unpatched sktime (pandas convert-once)
             "native"  -> VectorizedDF2 iterates in the input's own backend
             "pandas"/"polars" -> VectorizedDF2, convert once to that family
             "inner"   -> family of the *estimator's* y_inner_mtype (bridge);
                          needs the estimator, so it is resolved per call via
                          the ``_x4_target`` attribute set by ``set_inner_target``
    cutoff : bool, also replace get_cutoff by the backend-aware adapter version
    """
    import sktime.forecasting.base._base as B
    from x4_adapter import adapter_get_cutoff, prepare_VectorizedDF2

    orig_prepare = B.prepare_VectorizedDF
    orig_cutoff = B.get_cutoff

    if policy != "current":

        def new_prepare(
            X,
            iterate_as="Series",
            is_scitype="Panel",
            iterate_cols=False,
            store=None,
            store_behaviour=None,
        ):
            target = _STATE["target"] if policy == "inner" else policy
            return prepare_VectorizedDF2(
                X,
                iterate_as=iterate_as,
                is_scitype=is_scitype,
                iterate_cols=iterate_cols,
                store=store,
                store_behaviour=store_behaviour,
                target=target,
            )

        B.prepare_VectorizedDF = new_prepare
    if cutoff:
        B.get_cutoff = adapter_get_cutoff
    try:
        yield
    finally:
        B.prepare_VectorizedDF = orig_prepare
        B.get_cutoff = orig_cutoff


_STATE = {"target": "pandas"}


def set_inner_target(estimator):
    """Inner-mtype bridge: target backend = family of estimator's y_inner_mtype."""
    from x4_adapter import family_of_inner_mtype

    _STATE["target"] = family_of_inner_mtype(estimator.get_tag("y_inner_mtype"))
    return _STATE["target"]
