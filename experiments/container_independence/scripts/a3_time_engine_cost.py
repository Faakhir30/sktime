"""a3: cost of time-axis work when data lives in polars.

Task measured: given a long polars panel (keys h0, time, value), produce the
prediction-index frame  {(h0, t) : t = cutoff(h0) + k steps, k = 1..H}
i.e. per-series cutoff + fh.to_absolute + building the output keys.

P1 pure polars        group_by max -> cross join k -> TimeIndex polars expr
P2 hybrid/pandas-vec  polars group_by max -> unique cutoffs -> pandas offsets
                      (vectorised, U x H) -> back to polars -> join
P3 hybrid/sktime-fh   as P2 but each unique cutoff via sktime
                      ForecastingHorizon.to_absolute (exact sktime semantics)
P4 current sktime     N x fh.to_absolute_index(cutoff_i) on pandas cutoffs
                      (what vectorised predict does once per instance), plus
                      get_cutoff on the pandas MultiIndex frame
P5 (profile)          cProfile of NaiveForecaster fit+predict on pd-multiindex:
                      share of wall time spent in fh / cutoff code

Two end-alignment regimes: aligned (U=1 unique cutoff) and ragged
(U = min(N, T//2) unique cutoffs).

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/a3_time_engine_cost.py --grid dev
"""

import argparse
import cProfile
import json
import os
import pstats
import sys
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
from a3_timeindex import PandasTimeOps, PolarsTimeOps, Step  # noqa: E402
from common import DEV_GRID, LARGE_GRID, bench, env_info, rss_mb  # noqa: E402

warnings.filterwarnings("ignore")
PD3 = int(pd.__version__.split(".")[0]) >= 3
H = 12


def make_polars_panel(n_series, n_time, freq, ragged, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2000-01-01", periods=n_time, freq=freq)
    keys = np.repeat(np.array([f"s{i:07d}" for i in range(n_series)]), n_time)
    times = np.tile(np.asarray(t, dtype="M8[us]"), n_series)
    vals = rng.normal(size=n_series * n_time)
    df = pl.DataFrame({"h0": keys, "time": times, "c0": vals})
    if ragged:
        r = max(1, min(n_series, n_time // 2))
        lens = n_time - (np.arange(n_series) % r)
        mask = np.tile(np.arange(n_time), n_series) < np.repeat(lens, n_time)
        df = df.filter(pl.Series(mask))
    return df


def p1_polars(df, step):
    cut = df.group_by("h0").agg(pl.col("time").max().alias("c"))
    k = pl.DataFrame({"k": np.arange(1, H + 1, dtype=np.int64)})
    return cut.join(k, how="cross").select(
        "h0", PolarsTimeOps.add_steps_expr(pl.col("c"), pl.col("k"), step).alias("t")
    )


def _back_to_polars(cut, u_pl, abs_np):
    m = pl.DataFrame({
        "c": np.repeat(u_pl.to_numpy(), H),
        "k": np.tile(np.arange(1, H + 1, dtype=np.int64), len(u_pl)),
        "t": abs_np.ravel().astype("M8[us]"),
    }).with_columns(pl.col("c").cast(u_pl.dtype))
    return cut.join(m, on="c").select("h0", "t")


def p2_hybrid_pandas_vec(df, step):
    cut = df.group_by("h0").agg(pl.col("time").max().alias("c"))
    u = cut["c"].unique()
    u_pd = pd.DatetimeIndex(u.to_numpy())
    abs_np = PandasTimeOps.add_steps(u_pd, np.arange(1, H + 1), step)
    return _back_to_polars(cut, u, abs_np)


def p3_hybrid_sktime_fh(df, freq):
    from sktime.forecasting.base import ForecastingHorizon

    cut = df.group_by("h0").agg(pl.col("time").max().alias("c"))
    u = cut["c"].unique()
    fh = ForecastingHorizon(list(range(1, H + 1)))
    rows = [
        np.asarray(fh.to_absolute_index(pd.DatetimeIndex([c], freq=freq)),
                   dtype="M8[us]")
        for c in u.to_numpy()
    ]
    return _back_to_polars(cut, u, np.stack(rows))


def p4_current_sktime(cutoffs_pd, freq):
    """N x fh.to_absolute_index(cutoff_i): per-instance work in vectorised predict."""
    from sktime.forecasting.base import ForecastingHorizon

    fh = ForecastingHorizon(list(range(1, H + 1)))
    return [fh.to_absolute_index(pd.DatetimeIndex([c], freq=freq)) for c in cutoffs_pd]


def p4_get_cutoff(y_pd):
    from sktime.datatypes import get_cutoff

    return get_cutoff(y_pd, return_index=True)


TIME_FUNCS = ("_fh.py", "get_cutoff", "_get_cutoff_from_index", "infer_freq",
              "check_fh", "_check_fh", "_set_cutoff")


def p5_profile(y_pd):
    from sktime.forecasting.naive import NaiveForecaster

    f = NaiveForecaster(strategy="last")
    f.set_config(remember_data=False) if "remember_data" in f.get_config() else None
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    f.fit(y_pd, fh=list(range(1, H + 1)))
    f.predict()
    pr.disable()
    wall = time.perf_counter() - t0
    st = pstats.Stats(pr)
    tot = st.total_tt
    # exclusive (tottime) time of functions defined in time-axis code
    excl = 0.0
    by_file = {}
    for (fname, _, func), (_, _, tt, ct, _) in st.stats.items():
        hit = fname.endswith("forecasting/base/_fh.py") or func in (
            "get_cutoff", "_get_cutoff_from_index", "infer_freq", "_infer_freq_from_index",
        ) or fname.endswith("pandas/tseries/frequencies.py") or "offsets" in fname
        if hit:
            excl += tt
            key = os.path.basename(fname) + ":" + func
            by_file[key] = by_file.get(key, 0) + tt
    # inclusive time of top-level time-axis entry points (may overlap)
    incl = {}
    for (fname, _, func), (_, _, tt, ct, _) in st.stats.items():
        if (fname.endswith("forecasting/base/_fh.py") and func in (
                "to_absolute", "to_absolute_index", "to_relative",
                "is_all_out_of_sample", "_check_values", "__init__")) or func in (
                "get_cutoff", "_set_cutoff_from_y", "check_fh"):
            incl[os.path.basename(fname) + ":" + func] = incl.get(
                os.path.basename(fname) + ":" + func, 0) + ct
    top = sorted(by_file.items(), key=lambda x: -x[1])[:8]
    return {"wall_s": wall, "profiled_total_s": tot, "time_axis_exclusive_s": excl,
            "time_axis_exclusive_share": excl / tot if tot else None,
            "inclusive_entry_points_s": incl, "top_exclusive": top}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])
    ap.add_argument("--freqs", default="MS,D")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--profile-max-series", type=int, default=1000)
    ap.add_argument("--p4-max-series", type=int, default=10_000)
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    out = {"env": env_info(), "H": H, "grid": args.grid, "rows": []}
    print(f"{'freq':4s} {'regime':7s} {'N':>6s} {'T':>4s} {'U':>6s} "
          f"{'P1 pl ms':>9s} {'P2 hyb ms':>9s} {'P3 skfh ms':>10s} {'P4 cur ms':>10s} "
          f"{'P4 getcut':>9s} {'P2/P1':>6s} {'P3/P1':>6s} {'P4/P1':>7s} rss")
    for freq in args.freqs.split(","):
        step = Step.from_pandas(freq)
        for n_series, n_time in grid:
            for ragged in (False, True):
                df = make_polars_panel(n_series, n_time, freq, ragged)
                cut = df.group_by("h0").agg(pl.col("time").max().alias("c"))
                U = cut["c"].n_unique()
                # correctness: P1 == P2 == P3
                r1 = p1_polars(df, step).sort("h0", "t")
                r2 = p2_hybrid_pandas_vec(df, step).sort("h0", "t")
                r3 = p3_hybrid_sktime_fh(df, freq).sort("h0", "t")
                eq12 = r1.equals(r2)
                eq13 = r1.equals(r3)
                rec = {"freq": freq, "ragged": ragged, "n_series": n_series,
                       "n_time": n_time, "rows": df.height, "U": U,
                       "P1==P2": eq12, "P1==P3": eq13}
                rec["P1_s"] = bench(lambda: p1_polars(df, step), repeat=args.repeat)
                rec["P2_s"] = bench(lambda: p2_hybrid_pandas_vec(df, step),
                                    repeat=args.repeat)
                rep3 = args.repeat if U <= 1000 else 2
                rec["P3_s"] = bench(lambda: p3_hybrid_sktime_fh(df, freq), repeat=rep3)
                if n_series <= args.p4_max_series:
                    cut_pd = pd.DatetimeIndex(cut["c"].to_numpy())
                    rep4 = args.repeat if n_series <= 1000 else 2
                    rec["P4_s"] = bench(lambda: p4_current_sktime(cut_pd, freq),
                                        repeat=rep4)
                    y_pd = df.to_pandas().set_index(["h0", "time"])
                    rec["P4_get_cutoff_s"] = bench(lambda: p4_get_cutoff(y_pd),
                                                   repeat=rep4)
                else:
                    rec["P4_s"] = rec["P4_get_cutoff_s"] = None
                rec["rss_mb"] = rss_mb()
                out["rows"].append(rec)

                def ms(x):
                    return f"{1e3 * x:9.2f}" if x is not None else "      n/a"

                def ratio(a, b):
                    return f"{a / b:6.1f}" if a is not None else "   n/a"

                print(f"{freq:4s} {'ragged' if ragged else 'aligned':7s} "
                      f"{n_series:6d} {n_time:4d} {U:6d} {ms(rec['P1_s'])} "
                      f"{ms(rec['P2_s'])} {ms(rec['P3_s']):>10s} {ms(rec['P4_s']):>10s} "
                      f"{ms(rec['P4_get_cutoff_s'])} {ratio(rec['P2_s'], rec['P1_s'])} "
                      f"{ratio(rec['P3_s'], rec['P1_s'])} "
                      f"{ratio(rec['P4_s'], rec['P1_s']):>7s} {rec['rss_mb']:.0f} "
                      f"eq={eq12},{eq13}", flush=True)
    # P5: profile NaiveForecaster on pandas multiindex
    out["profile"] = []
    for n_series, n_time in grid:
        if n_series > args.profile_max_series:
            continue
        for freq in args.freqs.split(","):
            df = make_polars_panel(n_series, n_time, freq, ragged=False)
            y_pd = df.to_pandas().set_index(["h0", "time"])
            prof = p5_profile(y_pd)
            prof.update({"n_series": n_series, "n_time": n_time, "freq": freq})
            out["profile"].append(prof)
            print(f"P5 NaiveForecaster pd-multiindex N={n_series} T={n_time} {freq}: "
                  f"wall={prof['wall_s']:.2f}s time-axis exclusive share="
                  f"{100 * prof['time_axis_exclusive_share']:.1f}% "
                  f"top={[(k, round(v, 3)) for k, v in prof['top_exclusive'][:4]]}",
                  flush=True)
            print("    inclusive entry points (s, overlapping):",
                  {k: round(v, 3) for k, v in prof["inclusive_entry_points_s"].items()})
    tag = "pd3" if PD3 else "pd2"
    path = (f"experiments/container_independence/results/"
            f"a3_time_engine_cost_{args.grid}_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
