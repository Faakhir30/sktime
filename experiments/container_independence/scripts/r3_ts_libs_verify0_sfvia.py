"""r3_ts_libs_verify0: decompose claim r3_ts_libs.4 (polars->pandas->polars around statsforecast).

The original measured  via_pandas = to_pandas + sf.forecast(pandas) + from_pandas
against sf.forecast(polars) and attributed the 1.37-1.80x to "converting".
Here we time the pieces separately, interleaved in one process, to see how
much of the gap is the conversion itself and how much is statsforecast's own
pandas code path being slower than its polars code path:

  native_pl   : sf.forecast(polars)
  native_pd   : sf.forecast(pandas)          (pandas input to begin with)
  to_pandas   : plf.to_pandas()
  from_pandas : pl.from_pandas(forecast_out)
  via_pandas  : the original composite

  conversion share = (to_pandas + from_pandas) / (via_pandas - native_pl)

Run with the r3 venv (statsforecast installed):
    PYTHONPATH=.:experiments/container_independence $R3_PY -W ignore \
        experiments/container_independence/scripts/r3_ts_libs_verify0_sfvia.py --grid dev
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from common import DEV_GRID, LARGE_GRID, bench, env_info, make_panel_fast  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID

    import statsforecast
    from statsforecast import StatsForecast
    from statsforecast.models import Naive

    res = {"env": {**env_info(), "statsforecast": statsforecast.__version__},
           "grid": args.grid, "rows": []}
    for n_series, n_time in grid:
        mi = make_panel_fast(n_series, n_time)
        flat = mi.reset_index().rename(columns={"h0": "unique_id", "time": "ds", "c0": "y"})
        plf = pl.from_pandas(flat)
        sf_pd = StatsForecast(models=[Naive()], freq="D", n_jobs=1)
        sf_pl = StatsForecast(models=[Naive()], freq="1d", n_jobs=1)
        out_pd = sf_pd.forecast(h=5, df=flat)
        rep = 3 if n_series >= 10_000 else 7

        fns = {
            "native_pl": lambda: sf_pl.forecast(h=5, df=plf),
            "native_pd": lambda: sf_pd.forecast(h=5, df=flat),
            "to_pandas": lambda: plf.to_pandas(),
            "from_pandas": lambda: pl.from_pandas(out_pd),
            "via_pandas": lambda: pl.from_pandas(sf_pd.forecast(h=5, df=plf.to_pandas())),
        }
        # correctness: via_pandas equals native polars
        a = fns["native_pl"]().sort("unique_id", "ds")["Naive"].to_numpy()
        b = fns["via_pandas"]().sort("unique_id", "ds")["Naive"].to_numpy()
        assert np.allclose(a, b)
        row = {"n_series": n_series, "n_time": n_time}
        for rnd in range(3):  # interleaved rounds, keep min
            for k, fn in fns.items():
                t = bench(fn, repeat=rep)
                row[k + "_s"] = min(row.get(k + "_s", float("inf")), t)
        gap = row["via_pandas_s"] - row["native_pl_s"]
        conv = row["to_pandas_s"] + row["from_pandas_s"]
        row["ratio_via_pandas_over_native_pl"] = row["via_pandas_s"] / row["native_pl_s"]
        row["ratio_native_pd_over_native_pl"] = row["native_pd_s"] / row["native_pl_s"]
        row["gap_s"] = gap
        row["conversion_s"] = conv
        row["conversion_share_of_gap"] = conv / gap if gap > 0 else None
        row["ratio_conversion_only"] = (row["native_pl_s"] + conv) / row["native_pl_s"]
        print(json.dumps(row), flush=True)
        res["rows"].append(row)

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"r3_ts_libs_verify0_sfvia_{args.grid}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    sys.exit(main())
