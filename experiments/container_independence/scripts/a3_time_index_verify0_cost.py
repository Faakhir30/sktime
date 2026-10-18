"""a3_time_index_verify0: independent check of a3 time-engine cost claims.

1. P1 (a3 original pure polars, per-row offset_by on N*H rows) vs
   P1d (pure polars, DEDUPLICATED: unique cutoffs -> U*H offsets -> join) vs
   P2 (a3 hybrid: unique cutoffs -> pandas offsets) - all interleaved in-process,
   outputs compared with DataFrame.equals after sort.
   Tests whether "hybrid is faster than pure polars at 10k series" is an artifact
   of P1 not deduplicating.
2. Real sktime vectorised NaiveForecaster fit+predict on a pd-multiindex panel:
   number of _to_absolute / _to_relative calls and lru_cache hits (is the
   _HashIndex cache effective?), and total time spent inside them (wrapper).

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/a3_time_index_verify0_cost.py --grid dev
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
sys.path.insert(0, os.path.dirname(__file__))
from a3_time_engine_cost import H, make_polars_panel, p1_polars, p2_hybrid_pandas_vec  # noqa
from a3_timeindex import PolarsTimeOps, Step  # noqa: E402
from common import DEV_GRID, LARGE_GRID, bench, env_info, rss_mb  # noqa: E402

warnings.filterwarnings("ignore")


def p1_dedup(df, step):
    cut = df.group_by("h0").agg(pl.col("time").max().alias("c"))
    u = cut.select(pl.col("c").unique())
    k = pl.DataFrame({"k": np.arange(1, H + 1, dtype=np.int64)})
    m = u.join(k, how="cross").with_columns(
        t=PolarsTimeOps.add_steps_expr(pl.col("c"), pl.col("k"), step))
    return cut.join(m, on="c").select("h0", "t")


def cost_table(grid, freqs, repeat):
    rows = []
    for freq in freqs:
        step = Step.from_pandas(freq)
        for n, t in grid:
            for ragged in (False, True):
                df = make_polars_panel(n, t, freq, ragged)
                r1 = p1_polars(df, step).sort("h0", "t")
                r1d = p1_dedup(df, step).sort("h0", "t")
                r2 = p2_hybrid_pandas_vec(df, step).sort("h0", "t")
                eq = r1.equals(r1d) and r1.equals(r2) and r1.height == n * H
                # interleave: alternate the three functions, min over repeats
                best = {"P1": 1e9, "P1d": 1e9, "P2": 1e9}
                fns = {"P1": lambda: p1_polars(df, step),
                       "P1d": lambda: p1_dedup(df, step),
                       "P2": lambda: p2_hybrid_pandas_vec(df, step)}
                for f in fns.values():
                    f()
                for _ in range(repeat):
                    for name, f in fns.items():
                        t0 = time.perf_counter()
                        f()
                        best[name] = min(best[name], time.perf_counter() - t0)
                rec = {"freq": freq, "ragged": ragged, "N": n, "T": t, "eq": eq,
                       **{k: 1e3 * v for k, v in best.items()}}
                rec["P2/P1"] = best["P2"] / best["P1"]
                rec["P2/P1d"] = best["P2"] / best["P1d"]
                rec["P1/P1d"] = best["P1"] / best["P1d"]
                rows.append(rec)
                print(f"{freq:3s} {'ragged' if ragged else 'aligned':7s} N={n:6d} "
                      f"P1={rec['P1']:7.2f} P1d={rec['P1d']:7.2f} P2={rec['P2']:7.2f} ms"
                      f"  P2/P1={rec['P2/P1']:.2f} P2/P1d={rec['P2/P1d']:.2f} "
                      f"P1/P1d={rec['P1/P1d']:.2f} eq={eq} rss={rss_mb():.0f}",
                      flush=True)
    return rows


def real_forecaster(n_series, n_time, freq):
    """Count/time fh conversions inside a real vectorised NaiveForecaster run."""
    import sktime.forecasting.base._fh as fhmod
    from sktime.forecasting.naive import NaiveForecaster

    df = make_polars_panel(n_series, n_time, freq, ragged=False)
    y = df.to_pandas().set_index(["h0", "time"])
    acc = {"_to_absolute": [0, 0.0], "_to_relative": [0, 0.0]}
    orig = {name: getattr(fhmod, name) for name in acc}
    for name in acc:
        orig[name].cache_clear()

    def wrap(name):
        f = orig[name]

        def w(*a, **k):
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                acc[name][0] += 1
                acc[name][1] += time.perf_counter() - t0
        w.cache_info = f.cache_info
        return w

    for name in acc:
        setattr(fhmod, name, wrap(name))
    try:
        t0 = time.perf_counter()
        fc = NaiveForecaster(strategy="last").fit(y, fh=list(range(1, H + 1)))
        t1 = time.perf_counter()
        yp = fc.predict()
        t2 = time.perf_counter()
    finally:
        for name in acc:
            setattr(fhmod, name, orig[name])
    info = {name: orig[name].cache_info()._asdict() for name in acc}
    rec = {"N": n_series, "T": n_time, "freq": freq, "fit_s": t1 - t0,
           "predict_s": t2 - t1, "calls": {k: v[0] for k, v in acc.items()},
           "inclusive_s": {k: v[1] for k, v in acc.items()}, "cache_info": info,
           "pred_rows": len(yp)}
    rec["fh_conv_share_of_fit_predict"] = sum(v[1] for v in acc.values()) / (t2 - t0)
    rec["fit_predict_ms_per_instance"] = 1e3 * (t2 - t0) / n_series
    print("real NaiveForecaster:", json.dumps(rec, default=str), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--real-n", default="100,1000")
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    out = {"env": env_info(), "cost": cost_table(grid, ["MS", "D"], args.repeat),
           "real": [real_forecaster(int(n), 100, f) for n in args.real_n.split(",")
                    for f in ("MS", "D")]}
    path = (f"experiments/container_independence/results/"
            f"a3_time_index_verify0_cost_{args.grid}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
