"""a3: where do ~2-3 ms per ForecastingHorizon.to_absolute_index call go?

cProfile of 500 calls fh.to_absolute_index(cutoff_i) with distinct monthly /
daily cutoffs (as in vectorised predict), top functions by cumulative time
restricted to sktime + pandas frames; plus unprofiled per-call time.

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/a3_fh_call_profile.py
"""

import cProfile
import io
import pstats
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def main():
    from sktime.forecasting.base import ForecastingHorizon

    for freq in ["MS", "D"]:
        cuts = pd.date_range("2000-01-01", periods=500, freq=freq)
        fh = ForecastingHorizon(list(range(1, 13)))
        cut_idx = [pd.DatetimeIndex([c], freq=freq) for c in cuts]
        t0 = time.perf_counter()
        for c in cut_idx:
            fh.to_absolute_index(c)
        per = (time.perf_counter() - t0) / len(cut_idx)
        # cost split: construction of the cutoff index itself
        t0 = time.perf_counter()
        for c in cuts:
            pd.DatetimeIndex([c], freq=freq)
        per_idx = (time.perf_counter() - t0) / len(cuts)
        t0 = time.perf_counter()
        for c in cut_idx:
            pd.util.hash_pandas_object(c).sum()
        per_hash = (time.perf_counter() - t0) / len(cut_idx)
        print(f"freq={freq}: unprofiled per call to_absolute_index = {1e3 * per:.3f} ms;"
              f" DatetimeIndex([c], freq) = {1e3 * per_idx:.3f} ms;"
              f" _HashIndex hash = {1e3 * per_hash:.3f} ms")
        pr = cProfile.Profile()
        pr.enable()
        for c in cut_idx:
            fh.to_absolute_index(c)
        pr.disable()
        s = io.StringIO()
        st = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        st.print_stats(r"_fh\.py|datetime\.py|frequencies|hashing|datetimes\.py", 14)
        txt = s.getvalue()
        lines = [ln for ln in txt.splitlines() if ln.strip() and (
            "ncalls" in ln or "/" in ln)]
        print("\n".join(lines[:18]))


if __name__ == "__main__":
    main()
