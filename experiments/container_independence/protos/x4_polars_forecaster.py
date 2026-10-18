"""x4: toy BaseForecaster subclass that computes natively in polars.

Declares ``y_inner_mtype = "pl.DataFrame"``: sktime's boilerplate
(check_is_scitype / convert_to) hands ``_fit`` a polars frame with one
``__index__<time>`` key column. Strategy == NaiveForecaster(strategy="last").
"""

import numpy as np

from sktime.forecasting.base import BaseForecaster

_IX = "__index__"


class PolarsNaiveLast(BaseForecaster):
    """Repeat last observed value; all data handling in polars."""

    _tags = {
        "y_inner_mtype": "pl.DataFrame",
        "X_inner_mtype": "pl.DataFrame",
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:missing_values": True,
        "capability:insample": False,
        "requires-fh-in-fit": False,
        "python_dependencies": "polars",
    }

    def _fit(self, y, X, fh):
        keys = [c for c in y.columns if c.startswith(_IX)]
        self._time_col = keys[-1] if keys else f"{_IX}0"
        self._value_cols = [c for c in y.columns if c not in keys]
        self._last = y.select(self._value_cols).tail(1)  # polars op
        return self

    def _predict(self, fh, X):
        import polars as pl

        # fh/cutoff are pandas *index metadata* (size h), not data
        idx = fh.to_absolute_index(self.cutoff)
        n = len(idx)
        cols = {self._time_col: idx.to_numpy()}
        for c in self._value_cols:
            cols[c] = np.repeat(self._last.get_column(c).to_numpy(), n)
        return pl.DataFrame(cols)
