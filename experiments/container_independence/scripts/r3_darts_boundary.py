"""r3: quantify Darts' narwhals-at-the-boundary design.

Darts >= 0.33 accepts any narwhals-compatible eager frame in
TimeSeries.from_dataframe / from_group_dataframe and exports via
to_dataframe(backend=...). Internally a TimeSeries is a 3D numpy array
(time, component, sample) plus a pandas Index for time.

Measures at each grid size (n_series, n_time):
  A. from_group_dataframe on pandas vs polars long frames (boundary in)
  B. to_dataframe(backend=pandas|polars) for the resulting series list,
     concatenated to one long frame (boundary out)
  C. probes (once): lazy frames / pyarrow / pandas MultiIndex input behaviour,
     whether output type follows input type.

Run with a python that has darts installed (r3d venv):

    PYTHONPATH=.:experiments/container_independence $R3D_PY -W ignore \
        experiments/container_independence/scripts/r3_darts_boundary.py --grid dev
"""

import argparse
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from common import DEV_GRID, LARGE_GRID, bench, env_info, make_panel_fast  # noqa: E402


def long_frames(n_series, n_time):
    mi = make_panel_fast(n_series, n_time)
    flat = mi.reset_index()
    return mi, flat, pl.from_pandas(flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-series", type=int, default=10_000,
                    help="skip grid points with more series (darts builds one python "
                         "TimeSeries per group: ~1.2ms and ~50KB each; 10k series "
                         "peaks at ~500MB RSS)")
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    grid = [g for g in grid if g[0] <= args.max_series]

    import darts
    from darts import TimeSeries

    logging.getLogger("darts").setLevel(logging.ERROR)
    res = {"env": {**env_info(), "darts": darts.__version__}, "grid": args.grid,
           "rows": [], "probes": {}}

    # C. probes
    mi, flat, plf = long_frames(5, 20)
    probes = res["probes"]

    def probe(name, fn):
        try:
            out = fn()
            probes[name] = out
        except Exception as e:  # noqa: BLE001
            probes[name] = f"ERROR {type(e).__name__}: {str(e)[:160]}"

    probe("from_group_dataframe(polars).n", lambda: len(TimeSeries.from_group_dataframe(
        plf, group_cols="h0", time_col="time", value_cols="c0")))
    probe("from_group_dataframe(polars_lazy)", lambda: len(TimeSeries.from_group_dataframe(
        plf.lazy(), group_cols="h0", time_col="time", value_cols="c0")))
    probe("from_group_dataframe(pyarrow).n", lambda: len(TimeSeries.from_group_dataframe(
        plf.to_arrow(), group_cols="h0", time_col="time", value_cols="c0")))
    probe("from_dataframe(pandas MultiIndex, no time_col)", lambda: str(
        TimeSeries.from_dataframe(mi).time_index[:2]))
    probe("from_dataframe(pandas MultiIndex.droplevel->DatetimeIndex)", lambda: str(
        type(TimeSeries.from_dataframe(mi.xs("s0000000", level="h0")).time_index).__name__))
    ts = TimeSeries.from_group_dataframe(plf, group_cols="h0", time_col="time",
                                         value_cols="c0")[0]
    probe("time_index type after polars input", lambda: type(ts.time_index).__name__)
    probe("to_dataframe() default type after polars input",
          lambda: f"{type(ts.to_dataframe()).__module__}.{type(ts.to_dataframe()).__name__}")
    probe("to_dataframe(backend=polars) type",
          lambda: type(ts.to_dataframe(backend="polars")).__module__)
    probe("static_covariates type after polars input",
          lambda: type(ts.static_covariates).__module__)
    # period index: pandas PeriodIndex input
    pmi = make_panel_fast(3, 10, time_kind="period")
    probe("from_dataframe(PeriodIndex)", lambda: type(
        TimeSeries.from_dataframe(pmi.xs("s0000000", level="h0")).time_index).__name__)
    print("probes:", json.dumps(probes, indent=1))

    for n_series, n_time in grid:
        mi, flat, plf = long_frames(n_series, n_time)
        rep = 3 if n_series >= 1000 else 5
        row = {"n_series": n_series, "n_time": n_time}

        def in_pd():
            return TimeSeries.from_group_dataframe(
                flat, group_cols="h0", time_col="time", value_cols="c0")

        def in_pl():
            return TimeSeries.from_group_dataframe(
                plf, group_cols="h0", time_col="time", value_cols="c0")

        row["from_group_pandas_s"] = bench(in_pd, repeat=rep, warmup=0 if n_series >= 10_000 else 1)
        row["from_group_polars_s"] = bench(in_pl, repeat=rep, warmup=0 if n_series >= 10_000 else 1)
        series = in_pl()
        row["per_series_us_polars"] = row["from_group_polars_s"] / n_series * 1e6

        def out_pd():
            return pd.concat([s.to_dataframe(time_as_index=False) for s in series])

        def out_pl():
            return pl.concat([s.to_dataframe(backend="polars") for s in series])

        row["to_dataframe_concat_pandas_s"] = bench(out_pd, repeat=rep, warmup=0)
        row["to_dataframe_concat_polars_s"] = bench(out_pl, repeat=rep, warmup=0)

        # reference: the pure array work the boundary wraps
        def numpy_ref():
            v = flat["c0"].to_numpy()
            return [v[i * n_time:(i + 1) * n_time] for i in range(n_series)]

        row["numpy_split_ref_s"] = bench(numpy_ref, repeat=rep)
        row["ratio_from_group_polars_over_pandas"] = (
            row["from_group_polars_s"] / row["from_group_pandas_s"])
        row["ratio_from_group_polars_over_numpy_ref"] = (
            row["from_group_polars_s"] / row["numpy_split_ref_s"])
        print(json.dumps(row), flush=True)
        res["rows"].append(row)

    out = args.out or os.path.join(
        os.path.dirname(__file__), "..", "results",
        f"r3_darts_boundary_{args.grid}_darts{darts.__version__}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    sys.exit(main())
