"""r3: what "collect at the boundary" forfeits for lazy inputs.

Prior art (functime fit, pytimetk convert_to_engine, darts eager_only=True,
statsforecast/utilsforecast process_df) materializes a polars LazyFrame in full
before any work. This measures, on a parquet file, a naive forecast for a
SUBSET of series (selectivity 1%, 10%, 100%):

  lazy_pushdown : scan_parquet -> filter(ids) -> group_by.agg(last) -> collect
                  (projection + predicate pushdown into the scan)
  collect_first : scan_parquet -> collect() -> filter -> group_by.agg(last)
                  (what a boundary that calls .collect() first does)
  to_pandas     : collect() -> to_pandas() -> boolean mask -> groupby.last()
                  (what convert_to(pandas) at the boundary does)

Reports wall time (min of repeats) and peak RSS delta per variant.

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/r3_lazy_collect_cost.py --grid dev
"""

import argparse
import json
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from common import bench, env_info, make_panel_fast, peak_rss_delta_mb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--tmpdir", default=None)
    args = ap.parse_args()
    n_series, n_time, n_cols = (10_000, 50, 4) if args.grid == "dev" else (50_000, 200, 4)

    tmpdir = args.tmpdir or tempfile.mkdtemp(prefix="r3_lazy_")
    path = os.path.join(tmpdir, f"r3_panel_{n_series}x{n_time}.parquet")
    if not os.path.exists(path):
        flat = make_panel_fast(n_series, n_time, n_cols=n_cols).reset_index()
        pl.from_pandas(flat).write_parquet(path, row_group_size=50_000)
        del flat
    res = {"env": env_info(), "grid": args.grid, "n_series": n_series,
           "n_time": n_time, "n_cols": n_cols,
           "file_mb": os.path.getsize(path) / 2**20, "rows": []}

    all_ids = pl.scan_parquet(path).select(pl.col("h0").unique()).collect()["h0"].sort()
    for sel in [0.01, 0.1, 1.0]:
        ids = all_ids[: max(1, int(len(all_ids) * sel))].to_list()
        # NB polars 2.0 perf cliff: is_in(<python list of <=100 str>) in a scan
        # took ~110ms vs ~7ms for 101 items; passing an imploded Series avoids it
        ids_expr = pl.Series(ids).implode()

        def lazy_pushdown():
            return (pl.scan_parquet(path)
                    .filter(pl.col("h0").is_in(ids_expr))
                    .group_by("h0").agg(pl.col("time").last(), pl.col("c0").last())
                    .collect())

        def collect_first():
            df = pl.scan_parquet(path).collect()
            return (df.filter(pl.col("h0").is_in(ids_expr))
                    .group_by("h0").agg(pl.col("time").last(), pl.col("c0").last()))

        def to_pandas():
            pdf = pl.scan_parquet(path).collect().to_pandas()
            m = pdf["h0"].isin(set(ids))
            return pdf[m].groupby("h0", sort=False)[["time", "c0"]].last()

        a = lazy_pushdown().sort("h0")["c0"].to_numpy()
        b = collect_first().sort("h0")["c0"].to_numpy()
        c = to_pandas().sort_index()["c0"].to_numpy()
        assert np.allclose(a, b) and np.allclose(a, c)

        row = {"selectivity": sel, "n_ids": len(ids)}
        for name, fn in [("lazy_pushdown", lazy_pushdown),
                         ("collect_first", collect_first), ("to_pandas", to_pandas)]:
            row[name + "_s"] = bench(fn, repeat=5)
            _, row[name + "_peak_mb"] = peak_rss_delta_mb(fn)
        row["ratio_collect_first_over_lazy"] = row["collect_first_s"] / row["lazy_pushdown_s"]
        row["ratio_to_pandas_over_lazy"] = row["to_pandas_s"] / row["lazy_pushdown_s"]
        print(json.dumps(row), flush=True)
        res["rows"].append(row)

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"r3_lazy_collect_cost_{args.grid}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    sys.exit(main())
