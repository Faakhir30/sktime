"""x7: where does the per-series VectorizedDF loop spend time? (cProfile, NaiveForecaster last)

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x7_profile_loop.py [--n 200]
"""
import argparse
import cProfile
import io
import pstats
import time
import warnings

warnings.simplefilter("ignore")
import sktime.utils.warnings as _sw  # noqa: E402

_sw._SuppressWarningPattern.__exit__ = lambda self, *a: warnings.simplefilter("ignore")

from common import make_panel_fast  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
y = make_panel_fast(a.n, 100)
fh = list(range(1, 15))
NaiveForecaster().fit(y.loc[["s0000000", "s0000001"]], fh=fh).predict()
t = time.perf_counter()
f = NaiveForecaster().fit(y, fh=fh)
tf = time.perf_counter() - t
t = time.perf_counter()
f.predict()
tp = time.perf_counter() - t
print(f"n={a.n} fit {tf*1e3:.0f}ms ({tf/a.n*1e3:.2f}ms/series) "
      f"predict {tp*1e3:.0f}ms ({tp/a.n*1e3:.2f}ms/series)")
pr = cProfile.Profile()
pr.enable()
NaiveForecaster().fit(y, fh=fh).predict()
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(40)
print("\n".join(s.getvalue().splitlines()[:70]))
