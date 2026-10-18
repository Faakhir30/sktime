"""a1: cost of protocol op P10 ``cutoff(frame)`` as implemented today vs vectorized.

``sktime.datatypes._utilities._get_cutoff_from_index`` (called by get_cutoff, called
by BaseForecaster._set_cutoff_from_y on the FULL pd-multiindex y in vectorized fit/
update - twice per call when remember_data=True) loops in Python over
``idx.droplevel(-1)`` -- i.e. over every ROW, not every instance -- doing a
``DataFrame.loc`` lookup per row. This script times it against

* ``pd_vec``: pandas vectorized ``idx.get_level_values(-1).max()`` (+ freq of first
  instance), same result value
* ``pl``    : polars ``pl.col(time).max()`` on the long polars frame
* ``pl_lazy``: same on a LazyFrame (collect of a single scalar)

and checks results are equal. Interleaved, min-of-repeats via common.bench.

Run (from WT root):
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/a1_cutoff_bench.py --grid dev
Output: results/a1_cutoff_bench_<grid>.json
"""

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from common import (  # noqa: E402
    DEV_GRID,
    LARGE_GRID,
    bench,
    env_info,
    make_panel_fast,
    to_polars_long,
)

from sktime.datatypes._utilities import get_cutoff  # noqa: E402


def pd_vec(df):
    idx = df.index
    t = idx.get_level_values(-1)
    res = pd.Index([t.max()])
    # freq: same rule as sktime (freq of any instance slice that has one)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])
    a = ap.parse_args()
    grid = DEV_GRID if a.grid == "dev" else LARGE_GRID
    out = {"env": env_info(), "grid": a.grid, "rows": []}
    for n_series, n_time in grid:
        df = make_panel_fast(n_series, n_time)
        pldf = to_polars_long(df)
        lf = pldf.lazy()
        cur = get_cutoff(df, return_index=True)
        ref = pd_vec(df)
        plv = pldf.select(pl.col("time").max()).item()
        assert cur[0] == ref[0] == pd.Timestamp(plv), (cur, ref, plv)
        rep = 3 if n_series * n_time <= 100_000 else 1
        t_cur = bench(lambda: get_cutoff(df, return_index=True), repeat=rep)
        t_vec = bench(lambda: pd_vec(df), repeat=5)
        t_pl = bench(lambda: pldf.select(pl.col("time").max()).item(), repeat=5)
        t_lz = bench(lambda: lf.select(pl.col("time").max()).collect().item(), repeat=5)
        row = {
            "n_series": n_series,
            "n_time": n_time,
            "rows": n_series * n_time,
            "sktime_get_cutoff_s": t_cur,
            "pandas_vectorized_s": t_vec,
            "polars_eager_s": t_pl,
            "polars_lazy_s": t_lz,
            "ratio_sktime_over_pd_vec": t_cur / t_vec,
            "ratio_sktime_over_polars": t_cur / t_pl,
            "sktime_us_per_row": 1e6 * t_cur / (n_series * n_time),
            "freq_kept_by_sktime": str(getattr(cur, "freq", None)),
        }
        out["rows"].append(row)
        print(
            f"{n_series:>6}x{n_time:<4} sktime={t_cur:.4f}s ({row['sktime_us_per_row']:.1f}us/row)"
            f"  pd_vec={t_vec*1e3:.3f}ms  pl={t_pl*1e3:.3f}ms  pl_lazy={t_lz*1e3:.3f}ms"
            f"  ratio sk/pdvec={row['ratio_sktime_over_pd_vec']:.0f}x"
            f"  sk/pl={row['ratio_sktime_over_polars']:.0f}x",
            flush=True,
        )
    with open(
        f"experiments/container_independence/results/a1_cutoff_bench_{a.grid}.json", "w"
    ) as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
