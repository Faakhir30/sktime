"""r3: polars-native core (functime style) vs numpy hub (Nixtla style) vs per-series loops.

Same polars long frame (unique_id, ds, y) at each grid size. Two workloads:

W1 "naive" (expressible as a polars aggregation):
  a. functime.forecasting.naive  (polars-native: group_by().agg(last), lazy plan)
  b. hand-written polars expression naive (group_by + agg + date arithmetic)
  c. statsforecast Naive on polars input (process_df -> numpy GroupedArray -> polars out)
  d. per-series python loop over partition_by (sktime-VectorizedDF-like, polars flavour)

W2 "local linear trend" (needs arbitrary numpy per series; NOT expressible in
polars expressions): np.polyfit(deg=1) per series, forecast 5 steps.
  e. numpy hub: process_df once, then loop over indptr slices
  f. polars group_by().agg(pl.col(y).map_batches(fn)) (UDF per group)
  g. polars group_by().map_groups(fn) (UDF per group returning a DataFrame)
  h. pandas MultiIndex groupby(level=0) loop (current sktime VectorizedDF iteration)

Run with the r3f venv (functime + statsforecast):

    PYTHONPATH=.:experiments/container_independence $R3F_PY -W ignore \
        experiments/container_independence/scripts/r3_polars_native_vs_hub.py --grid dev
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

H = 5


def trend_fc(y):
    """Local linear trend forecast of H steps for one series (numpy in, numpy out)."""
    n = y.shape[0]
    b, a = np.polyfit(np.arange(n, dtype=float), y, 1)
    return a + b * np.arange(n, n + H, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--max-series", type=int, default=10**9,
                    help="skip grid points with more series (dev grid peaks ~540MB RSS "
                         "at 10k series because of the per-series loop variants)")
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    grid = [g for g in grid if g[0] <= args.max_series]

    import functime
    import statsforecast
    import utilsforecast.processing as ufp
    from functime.forecasting import naive as ft_naive
    from statsforecast import StatsForecast
    from statsforecast.models import Naive

    res = {"env": {**env_info(), "functime": getattr(functime, "__version__", "1.0.0"),
                   "statsforecast": statsforecast.__version__}, "grid": args.grid,
           "rows": []}

    for n_series, n_time in grid:
        mi = make_panel_fast(n_series, n_time)
        flat = mi.reset_index().rename(columns={"h0": "unique_id", "time": "ds", "c0": "y"})
        plf = pl.from_pandas(flat)
        rep = 3 if n_series >= 10_000 else 5
        row = {"n_series": n_series, "n_time": n_time}

        # ---- W1 naive
        def a_functime():
            return ft_naive(freq="1d")(y=plf, fh=H)

        def b_polars_expr():
            last = plf.group_by("unique_id", maintain_order=True).agg(
                pl.col("ds").last(), pl.col("y").last())
            return (last.with_columns(h=pl.int_ranges(1, H + 1))
                .explode("h")
                .with_columns(pl.col("ds") + pl.duration(days=pl.col("h")))
                .drop("h"))

        sf = StatsForecast(models=[Naive()], freq="1d", n_jobs=1)

        def c_sf():
            return sf.forecast(h=H, df=plf)

        def d_loop():
            outs = []
            for g in plf.partition_by("unique_id", maintain_order=True):
                last_t = g["ds"][-1]
                yv = g["y"][-1]
                outs.append(pl.DataFrame({
                    "unique_id": [g["unique_id"][0]] * H,
                    "ds": pl.datetime_range(last_t, last_t + pd.Timedelta(days=H),
                                            "1d", eager=True)[1:],
                    "y": [yv] * H}))
            return pl.concat(outs)

        # correctness spot check: all naive variants agree on values
        ref = c_sf().sort("unique_id", "ds")["Naive"].to_numpy()
        for name, fn, col in [("functime", a_functime, "y"),
                              ("polars_expr", b_polars_expr, "y"),
                              ("loop", d_loop, "y")]:
            got = fn().sort("unique_id", "ds")[col].to_numpy()
            assert np.allclose(got, ref), name

        row["w1_functime_s"] = bench(a_functime, repeat=rep)
        row["w1_polars_expr_s"] = bench(b_polars_expr, repeat=rep)
        row["w1_statsforecast_hub_s"] = bench(c_sf, repeat=rep)
        row["w1_partition_loop_s"] = bench(d_loop, repeat=max(1, rep - 2),
                                           warmup=0 if n_series >= 10_000 else 1)

        # ---- W2 local linear trend
        def e_hub():
            p = ufp.process_df(plf, "unique_id", "ds", "y")
            d, ip = p.data[:, 0], p.indptr
            out = np.empty((len(ip) - 1, H))
            for i in range(len(ip) - 1):
                out[i] = trend_fc(d[ip[i]:ip[i + 1]])
            return out

        def f_map_batches():
            return plf.group_by("unique_id", maintain_order=True).agg(
                pl.col("y").map_batches(
                    lambda s: pl.Series(trend_fc(s.to_numpy())),
                    return_dtype=pl.Float64))

        def g_map_groups():
            return plf.group_by("unique_id", maintain_order=True).map_groups(
                lambda df: pl.DataFrame({"unique_id": [df["unique_id"][0]] * H,
                                         "yhat": trend_fc(df["y"].to_numpy())}))

        def h_pd_groupby():
            out = []
            for _, g in mi.groupby(level=0, sort=False):
                out.append(trend_fc(g["c0"].to_numpy()))
            return np.vstack(out)

        base = e_hub()
        assert np.allclose(np.vstack(f_map_batches()["y"].to_list()), base)
        assert np.allclose(h_pd_groupby(), base)

        row["w2_numpy_hub_s"] = bench(e_hub, repeat=rep)
        row["w2_polars_map_batches_s"] = bench(f_map_batches, repeat=rep)
        row["w2_polars_map_groups_s"] = bench(g_map_groups, repeat=max(1, rep - 2))
        row["w2_pandas_groupby_loop_s"] = bench(h_pd_groupby, repeat=max(1, rep - 2))

        for k in ["w1_functime_s", "w1_statsforecast_hub_s", "w1_partition_loop_s"]:
            row["ratio_" + k[:-2] + "_over_polars_expr"] = row[k] / row["w1_polars_expr_s"]
        for k in ["w2_polars_map_batches_s", "w2_polars_map_groups_s",
                  "w2_pandas_groupby_loop_s"]:
            row["ratio_" + k[:-2] + "_over_numpy_hub"] = row[k] / row["w2_numpy_hub_s"]
        print(json.dumps(row), flush=True)
        res["rows"].append(row)

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"r3_polars_native_vs_hub_{args.grid}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    sys.exit(main())
