"""r3: quantify the Nixtla "hub-and-spoke" pattern (dataframe -> numpy GroupedArray).

Measures, at each grid size (n_series, n_time):
  A. utilsforecast.processing.process_df cost on pandas vs polars long frames
     (the one-time boundary conversion into data/indptr numpy arrays)
  B. StatsForecast(Naive).forecast end-to-end on pandas vs polars input, and the
     "convert-to-pandas-first" alternative (polars -> pandas -> forecast -> polars),
     i.e. what sktime does today via convert_to.
  C. per-series iteration cost: numpy indptr slicing (GroupedArray) vs
     pandas MultiIndex ``groupby(level=0, sort=False)`` (sktime VectorizedDF style)
     vs polars ``partition_by``.
  D. output container type for each input type (pandas, polars, polars lazy, pyarrow).

Must be run with a python that has utilsforecast/statsforecast installed (r3 venv):

    PYTHONPATH=.:experiments/container_independence $R3_PY -W ignore \
        experiments/container_independence/scripts/r3_nixtla_boundary.py --grid dev
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from common import DEV_GRID, LARGE_GRID, bench, env_info, make_panel_fast  # noqa: E402


def long_frames(n_series, n_time):
    mi = make_panel_fast(n_series, n_time)  # MultiIndex (h0, time), col c0
    flat = mi.reset_index().rename(columns={"h0": "unique_id", "time": "ds", "c0": "y"})
    return mi, flat, pl.from_pandas(flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID

    import utilsforecast
    import statsforecast
    import utilsforecast.processing as ufp
    from statsforecast import StatsForecast
    from statsforecast.models import Naive

    res = {
        "env": {
            **env_info(),
            "utilsforecast": utilsforecast.__version__,
            "statsforecast": statsforecast.__version__,
        },
        "grid": args.grid,
        "rows": [],
        "types": {},
    }

    # D. output container per input type
    _, flat, plf = long_frames(20, 30)
    for name, df, freq in [
        ("pandas", flat, "D"),
        ("polars", plf, "1d"),
        ("polars_lazy", plf.lazy(), "1d"),
        ("pyarrow", plf.to_arrow(), "1d"),
    ]:
        try:
            out = StatsForecast(models=[Naive()], freq=freq).forecast(h=3, df=df)
            res["types"][name] = f"{type(out).__module__}.{type(out).__name__}"
        except Exception as e:  # noqa: BLE001
            res["types"][name] = f"ERROR {type(e).__name__}: {str(e)[:120]}"
    print("output types:", json.dumps(res["types"], indent=1))

    for n_series, n_time in grid:
        mi, flat, plf = long_frames(n_series, n_time)
        rep = 3 if n_series * n_time >= 500_000 else 5
        row = {"n_series": n_series, "n_time": n_time}

        # A. boundary: dataframe -> (uids, last_times, data, indptr)
        row["process_df_pandas_s"] = bench(
            lambda: ufp.process_df(flat, "unique_id", "ds", "y"), repeat=rep
        )
        row["process_df_polars_s"] = bench(
            lambda: ufp.process_df(plf, "unique_id", "ds", "y"), repeat=rep
        )

        # B. end-to-end forecast (n_jobs=1 to avoid process pools under load)
        sf_pd = StatsForecast(models=[Naive()], freq="D", n_jobs=1)
        sf_pl = StatsForecast(models=[Naive()], freq="1d", n_jobs=1)
        row["sf_forecast_pandas_s"] = bench(
            lambda: sf_pd.forecast(h=5, df=flat), repeat=rep
        )
        row["sf_forecast_polars_native_s"] = bench(
            lambda: sf_pl.forecast(h=5, df=plf), repeat=rep
        )

        def via_pandas():
            out = sf_pd.forecast(h=5, df=plf.to_pandas())
            return pl.from_pandas(out)

        row["sf_forecast_polars_via_pandas_s"] = bench(via_pandas, repeat=rep)

        # C. per-series iteration: indptr slicing vs pandas groupby(level) vs polars
        p = ufp.process_df(flat, "unique_id", "ds", "y")
        data, indptr = p.data, p.indptr

        def iter_indptr():
            s = 0.0
            for i in range(len(indptr) - 1):
                s += data[indptr[i] : indptr[i + 1], 0][-1]
            return s

        def iter_pd_groupby():
            s = 0.0
            for _, g in mi.groupby(level=0, sort=False):
                s += g["c0"].to_numpy()[-1]
            return s

        def iter_pd_groupby_droplevel():
            # closer to sktime VectorizedDF: each instance becomes a single-index frame
            s = 0.0
            for _, g in mi.groupby(level=0, sort=False):
                g = g.droplevel(0)
                s += g["c0"].to_numpy()[-1]
            return s

        def iter_pl_partition():
            s = 0.0
            for g in plf.partition_by("unique_id", maintain_order=True):
                s += g["y"][-1]
            return s

        row["iter_indptr_s"] = bench(iter_indptr, repeat=rep)
        row["iter_pd_groupby_s"] = bench(iter_pd_groupby, repeat=rep)
        row["iter_pd_groupby_droplevel_s"] = bench(iter_pd_groupby_droplevel, repeat=rep)
        row["iter_pl_partition_s"] = bench(iter_pl_partition, repeat=rep)

        # ratios
        row["ratio_pd_groupby_over_indptr"] = row["iter_pd_groupby_s"] / row["iter_indptr_s"]
        row["ratio_pd_groupby_droplevel_over_indptr"] = (
            row["iter_pd_groupby_droplevel_s"] / row["iter_indptr_s"]
        )
        row["ratio_pl_partition_over_indptr"] = (
            row["iter_pl_partition_s"] / row["iter_indptr_s"]
        )
        row["ratio_sf_via_pandas_over_native_polars"] = (
            row["sf_forecast_polars_via_pandas_s"] / row["sf_forecast_polars_native_s"]
        )
        row["ratio_process_df_pandas_over_polars"] = (
            row["process_df_pandas_s"] / row["process_df_polars_s"]
        )
        print(json.dumps(row))
        res["rows"].append(row)

    out = args.out or os.path.join(
        os.path.dirname(__file__), "..", "results", f"r3_nixtla_boundary_{args.grid}.json"
    )
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    sys.exit(main())
