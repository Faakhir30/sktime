"""a2_census_verify0: does a2_trace's traced-pandas subclass survive the ops NaiveForecaster's
default path uses? If not, the enforcement mode silently skips those ops (false 'ok').
Usage: PYTHONPATH=.:experiments/container_independence/scripts $A2_PY -W ignore scripts/a2_census_verify0_taintloss.py
"""
import pandas as pd
import a2_trace as T
from sktime.transformations.lag import Lag

y = T.make_y(48)
ty = T.wrap(y)
print("input type:", type(ty).__name__)
lagged = Lag(1, keep_column_names=True).fit_transform(ty)
print("Lag(1).fit_transform output type:", type(lagged).__name__, "traced:", isinstance(lagged, T._TraceMixin))
new = pd.DataFrame(index=pd.period_range(y.index[-1] + 1, periods=3, freq=y.index.freq), columns=[0], dtype="float64")
full = pd.concat([lagged, new], keys=["a", "b"])
print("pd.concat([lagged, plain]) traced:", isinstance(full, T._TraceMixin))
full2 = pd.concat([ty.to_frame() if isinstance(ty, pd.Series) else ty, new])
print("pd.concat([traced, plain]) traced:", isinstance(full2, T._TraceMixin))
print("traced.shift(1) traced:", isinstance(ty.shift(1), T._TraceMixin))
print("traced.to_numpy() type:", type(ty.to_numpy()).__name__)
