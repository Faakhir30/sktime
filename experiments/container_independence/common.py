"""Shared harness for the container-independence experiments.

Every experiment script imports this module so data, sizes and timing
methodology are identical across experiments.

Run scripts with (from the worktree root)::

    PYTHONPATH=.:experiments/container_independence $EXP_PY <script>

where ``$EXP_PY`` is the pandas 2.x env, or ``$EXP_PY3`` the pandas 3.x env
(see README.md for the paths).
"""

import gc
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# sizes used during development; keep peak RSS < ~600MB per process.
# the final serial benchmark phase may use LARGE_GRID.
DEV_GRID = [(10, 100), (100, 100), (1_000, 100), (10_000, 50)]
LARGE_GRID = [(10, 500), (100, 500), (1_000, 500), (10_000, 100), (50_000, 50)]


def make_panel(
    n_series=100,
    n_time=100,
    n_cols=1,
    n_levels=2,
    time_kind="datetime",
    freq="D",
    seed=0,
):
    """Make a pandas frame in sktime ``pd-multiindex`` / ``pd_multiindex_hier`` format.

    Parameters
    ----------
    n_series : int
        total number of instances (leaf series)
    n_time : int
        number of time points per series (equal length)
    n_cols : int
        number of variables (columns ``c0``, ``c1``, ...)
    n_levels : int
        number of instance levels; 1 means Panel (2-level MultiIndex),
        2+ means Hierarchical. The time level is added on top of this.
        For n_levels >= 2, instances are split evenly over level 0.
    time_kind : "datetime", "period", or "int"
    freq : pandas frequency string for datetime/period
    seed : int

    Returns
    -------
    pd.DataFrame with MultiIndex (h0, ..., time), float64 columns
    """
    rng = np.random.default_rng(seed)
    if time_kind == "datetime":
        t = pd.date_range("2000-01-01", periods=n_time, freq=freq)
    elif time_kind == "period":
        t = pd.period_range("2000-01", periods=n_time, freq=freq)
    elif time_kind == "int":
        t = pd.RangeIndex(n_time)
    else:
        raise ValueError(time_kind)

    if n_levels == 1:
        inst = pd.Index([f"s{i}" for i in range(n_series)], name="h0")
        inst_tuples = [(k,) for k in inst]
    else:
        n_top = max(1, int(round(n_series ** (1 / n_levels))))
        inst_tuples = []
        for i in range(n_series):
            top = f"h0_{i % n_top}"
            rest = [f"h{lvl}_{i}" for lvl in range(1, n_levels)]
            inst_tuples.append((top, *rest))
        inst_tuples.sort()

    names = [f"h{i}" for i in range(n_levels)] + ["time"]
    rows = [(*k, ti) for k in inst_tuples for ti in t]
    idx = pd.MultiIndex.from_tuples(rows, names=names)
    vals = rng.normal(size=(len(idx), n_cols)).cumsum(axis=0)
    cols = [f"c{j}" for j in range(n_cols)]
    return pd.DataFrame(vals, index=idx, columns=cols)


def make_panel_fast(n_series=100, n_time=100, n_cols=1, time_kind="datetime", seed=0):
    """Vectorized generator for large Panel frames (2-level MultiIndex h0, time)."""
    rng = np.random.default_rng(seed)
    if time_kind == "datetime":
        t = pd.date_range("2000-01-01", periods=n_time, freq="D")
    elif time_kind == "period":
        t = pd.period_range("2000-01", periods=n_time, freq="M")
    else:
        t = pd.RangeIndex(n_time)
    inst = pd.Index([f"s{i:07d}" for i in range(n_series)], name="h0")
    idx = pd.MultiIndex.from_product([inst, t], names=["h0", "time"])
    vals = rng.normal(size=(len(idx), n_cols)).cumsum(axis=0)
    return pd.DataFrame(vals, index=idx, columns=[f"c{j}" for j in range(n_cols)])


def to_polars_long(df):
    """Pandas multiindex frame -> polars frame with index levels as plain columns.

    This is the "natural" polars layout (keys as columns), NOT the sktime
    ``__index__`` convention. Use sktime ``convert_to`` for the sktime polars mtypes.
    """
    import polars as pl

    flat = df.reset_index()
    for c in flat.columns:
        if isinstance(flat[c].dtype, pd.PeriodDtype):
            flat[c] = flat[c].dt.to_timestamp()
    return pl.from_pandas(flat)


def bench(fn, repeat=5, number=1, warmup=1):
    """Return min wall-clock seconds per call over ``repeat`` runs (after warmup)."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        best = min(best, (time.perf_counter() - t0) / number)
    return best


def rss_mb():
    """Current process resident set size in MB."""
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def peak_rss_delta_mb(fn, interval=0.005):
    """Run ``fn`` and return (result, peak RSS increase in MB) via a sampling thread.

    Works for Rust/C allocations (polars, arrow) unlike tracemalloc.
    """
    import threading

    import psutil

    proc = psutil.Process(os.getpid())
    gc.collect()
    base = proc.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(interval)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    try:
        res = fn()
    finally:
        stop.set()
        th.join()
    peak[0] = max(peak[0], proc.memory_info().rss)
    return res, (peak[0] - base) / 2**20


def env_info():
    """Versions of the relevant libraries, for recording alongside results."""
    out = {}
    for m in ["pandas", "numpy", "polars", "narwhals", "pyarrow", "dask", "duckdb"]:
        try:
            out[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            out[m] = None
    return out
