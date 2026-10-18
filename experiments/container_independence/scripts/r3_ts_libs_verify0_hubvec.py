"""r3_ts_libs_verify0: independent check of claims r3_ts_libs.9 and r3_ts_libs.10.

Claim .9 compared a hand-written polars expression naive to the FULL
StatsForecast.forecast pipeline (validation, per-series Python model loop,
output assembly) and called the latter "the numpy hub". Claim .10 called the
local linear trend (np.polyfit deg=1) "a per-series model polars cannot express".

This script measures, interleaved in one process on the same polars frame:

W1 naive (h=5):
  polars_expr       : original hand-written expression (group_by.agg(last) + dates)
  sf_forecast       : StatsForecast(Naive).forecast  (original "hub")
  hub_vectorized    : numpy hub WITHOUT a per-series loop: utilsforecast.process_df
                      + data[indptr[1:]-1] + numpy datetime arithmetic + polars out
  hub_vectorized_own: same but hub built by hand from the polars frame (rle lengths)

W2 local linear trend (h=5):
  hub_loop_polyfit  : original numpy hub loop with np.polyfit per series
  polars_map_batches: original group_by.agg(map_batches(polyfit))
  polars_ols_expr   : closed-form OLS written as polars expressions
                      (b = cov(t,y)/var(t), a = mean(y) - b*mean(t)), no UDF
  numpy_ols_reduceat: closed-form OLS on the hub with np.add.reduceat (no loop)

All W2 variants are checked to agree with np.polyfit to 1e-8.

    PYTHONPATH=.:experiments/container_independence $R3F_PY -W ignore \
        experiments/container_independence/scripts/r3_ts_libs_verify0_hubvec.py --grid dev
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from common import DEV_GRID, LARGE_GRID, bench, env_info, make_panel_fast, rss_mb  # noqa: E402

H = 5


def trend_fc(y):
    n = y.shape[0]
    b, a = np.polyfit(np.arange(n, dtype=float), y, 1)
    return a + b * np.arange(n, n + H, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--max-series", type=int, default=10**9)
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    grid = [g for g in grid if g[0] <= args.max_series]

    import statsforecast
    import utilsforecast.processing as ufp
    from statsforecast import StatsForecast
    from statsforecast.models import Naive

    res = {"env": {**env_info(), "statsforecast": statsforecast.__version__},
           "grid": args.grid, "rows": []}
    for n_series, n_time in grid:
        mi = make_panel_fast(n_series, n_time)
        flat = mi.reset_index().rename(columns={"h0": "unique_id", "time": "ds", "c0": "y"})
        plf = pl.from_pandas(flat)
        del mi, flat
        rep = 3 if n_series >= 10_000 else 5
        row = {"n_series": n_series, "n_time": n_time}

        # ---------------- W1
        def polars_expr():
            last = plf.group_by("unique_id", maintain_order=True).agg(
                pl.col("ds").last(), pl.col("y").last())
            return (last.with_columns(h=pl.int_ranges(1, H + 1))
                    .explode("h")
                    .with_columns(pl.col("ds") + pl.duration(days=pl.col("h")))
                    .drop("h"))

        sf = StatsForecast(models=[Naive()], freq="1d", n_jobs=1)

        def sf_forecast():
            return sf.forecast(h=H, df=plf)

        def _hub_out(uids, last_t, last_y):
            n = len(last_y)
            steps = np.arange(1, H + 1, dtype="int64") * np.timedelta64(1, "D")
            ds = (last_t.astype("datetime64[ns]")[:, None] + steps[None, :]).ravel()
            return pl.DataFrame({"unique_id": np.repeat(uids, H),
                                 "ds": ds, "y": np.repeat(last_y, H)})

        def hub_vectorized():
            p = ufp.process_df(plf, "unique_id", "ds", "y")
            last_idx = p.indptr[1:] - 1
            return _hub_out(p.uids.to_numpy(), p.last_times.to_numpy()
                            if hasattr(p.last_times, "to_numpy") else np.asarray(p.last_times),
                            p.data[last_idx, 0])

        def hub_vectorized_own():
            # assumes rows grouped by id and sorted by time (checked once below)
            r = plf["unique_id"].rle()
            ln = r.struct.field("len").to_numpy().astype(np.int64)
            ip = np.concatenate([[0], np.cumsum(ln)])
            last_idx = ip[1:] - 1
            return _hub_out(r.struct.field("value").to_numpy(),
                            plf["ds"].to_numpy()[last_idx],
                            plf["y"].to_numpy()[last_idx])

        assert plf.select(pl.col("ds").diff().over("unique_id").dt.total_nanoseconds()
                          .drop_nulls().min()).item() > 0
        ref = sf_forecast().sort("unique_id", "ds")
        for fn in [polars_expr, hub_vectorized, hub_vectorized_own]:
            got = fn().sort("unique_id", "ds")
            assert np.allclose(got["y"].to_numpy(), ref["Naive"].to_numpy()), fn.__name__
            assert (got["ds"].cast(pl.Datetime("ns")).to_numpy()
                    == ref["ds"].cast(pl.Datetime("ns")).to_numpy()).all(), fn.__name__

        # ---------------- W2
        def hub_loop_polyfit():
            p = ufp.process_df(plf, "unique_id", "ds", "y")
            d, ip = p.data[:, 0], p.indptr
            out = np.empty((len(ip) - 1, H))
            for i in range(len(ip) - 1):
                out[i] = trend_fc(d[ip[i]:ip[i + 1]])
            return out

        def polars_map_batches():
            return plf.group_by("unique_id", maintain_order=True).agg(
                pl.col("y").map_batches(lambda s: pl.Series(trend_fc(s.to_numpy())),
                                        return_dtype=pl.Float64))

        def polars_ols_expr():
            t = pl.int_range(pl.len()).over("unique_id").cast(pl.Float64)
            g = (plf.with_columns(t=t)
                 .group_by("unique_id", maintain_order=True)
                 .agg(b=pl.cov("t", "y") / pl.col("t").var(),
                      mt=pl.col("t").mean(), my=pl.col("y").mean(), n=pl.len()))
            g = g.with_columns(a=pl.col("my") - pl.col("b") * pl.col("mt"))
            return (g.with_columns(h=pl.int_ranges(0, H))
                    .explode("h")
                    .select("unique_id",
                            yhat=pl.col("a") + pl.col("b") * (pl.col("n") + pl.col("h"))))

        def numpy_ols_reduceat():
            r = plf["unique_id"].rle()
            ln = r.struct.field("len").to_numpy().astype(np.int64)
            ip = np.concatenate([[0], np.cumsum(ln)])[:-1]
            y = plf["y"].to_numpy()
            t = np.arange(len(y), dtype=float) - np.repeat(ip, ln)
            n = ln.astype(float)
            st, sy = np.add.reduceat(t, ip), np.add.reduceat(y, ip)
            stt, sty = np.add.reduceat(t * t, ip), np.add.reduceat(t * y, ip)
            b = (n * sty - st * sy) / (n * stt - st * st)
            a = (sy - b * st) / n
            return a[:, None] + b[:, None] * (n[:, None] + np.arange(H)[None, :])

        base = hub_loop_polyfit()
        assert np.allclose(np.vstack(polars_map_batches()["y"].to_list()), base, atol=1e-8)
        assert np.allclose(polars_ols_expr()["yhat"].to_numpy().reshape(-1, H), base, atol=1e-8)
        assert np.allclose(numpy_ols_reduceat(), base, atol=1e-8)

        fns = [polars_expr, sf_forecast, hub_vectorized, hub_vectorized_own,
               hub_loop_polyfit, polars_map_batches, polars_ols_expr, numpy_ols_reduceat]
        for rnd in range(2):
            for fn in fns:
                t_ = bench(fn, repeat=rep)
                k = fn.__name__ + "_s"
                row[k] = min(row.get(k, float("inf")), t_)
        pe = row["polars_expr_s"]
        for k in ["sf_forecast_s", "hub_vectorized_s", "hub_vectorized_own_s"]:
            row["w1_ratio_" + k[:-2] + "_over_polars_expr"] = row[k] / pe
        hl = row["hub_loop_polyfit_s"]
        for k in ["polars_map_batches_s", "polars_ols_expr_s", "numpy_ols_reduceat_s"]:
            row["w2_ratio_" + k[:-2] + "_over_hub_loop_polyfit"] = row[k] / hl
        row["w2_ratio_polars_ols_expr_over_numpy_ols_reduceat"] = (
            row["polars_ols_expr_s"] / row["numpy_ols_reduceat_s"])
        row["rss_mb"] = rss_mb()
        print(json.dumps(row), flush=True)
        res["rows"].append(row)

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"r3_ts_libs_verify0_hubvec_{args.grid}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    sys.exit(main())
