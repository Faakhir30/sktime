"""a2_census_verify0: does replacing base/_sktime.py:151 (last-window .loc) make the 7 reducers
pass numpylike enforcement, or do they hit a *second* violation (a2 only recorded the FIRST)?

Monkeypatches _BaseWindowForecaster._get_last_window with a positional/to_numpy version
(no change to sktime source), then runs a2_trace.main in the requested mode.
Usage: PYTHONPATH=.:experiments/container_independence/scripts $A2_PY -W ignore \
   experiments/container_independence/scripts/a2_census_verify0_lastwindow.py --mode numpylike --names ... --out ...
"""
import sys
from sktime.forecasting.base._sktime import _BaseWindowForecaster


def _get_last_window_positional(self):
    wl = self.window_length_
    y = self._cur_y.to_numpy()[-wl:]
    X = self._cur_X.to_numpy()[-wl:] if self._cur_X is not None else None
    return y, X


_BaseWindowForecaster._get_last_window = _get_last_window_positional
import a2_trace  # noqa: E402

a2_trace.main()
