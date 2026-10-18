"""a1: toy forecasters used by a1 scripts (cheap inner logic, so boundary cost dominates).

Not part of sktime.
"""

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class LastValueX(BaseForecaster):
    """Univariate, exogenous-capable forecaster: predicts last y + mean of last X row.

    y_inner_mtype pd.Series, X_inner_mtype pd.DataFrame -> panel input is
    row-vectorized by the BaseForecaster boundary layer.
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "capability:exogenous": True,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "capability:missing_values": False,
        "capability:update": True,
    }

    def __init__(self):
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        self.last_ = float(y.iloc[-1])
        self.n_ = len(y)
        return self

    def _update(self, y, X=None, update_params=True):
        self.last_ = float(y.iloc[-1])
        return self

    def _predict(self, fh, X=None):
        idx = fh.to_absolute_index(self.cutoff)
        val = self.last_
        if X is not None and len(X) > 0:
            val = val + float(np.asarray(X.iloc[-1]).mean()) * 0.0
        return pd.Series(np.full(len(idx), val), index=idx, name=self._y.name
                         if getattr(self, "_y", None) is not None
                         and hasattr(self._y, "name") else None)
