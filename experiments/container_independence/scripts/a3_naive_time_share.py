"""a3: share of NaiveForecaster wall time spent in time-axis (fh / cutoff) code.

No profiler: wraps the outermost time-axis entry points in-process (no file is
modified) with a re-entrancy guard, so nested calls are not double counted:
ForecastingHorizon.{__init__, to_absolute, to_relative, to_absolute_index,
to_absolute_int, to_indexer, is_all_in_sample, is_all_out_of_sample,
get_expected_pred_idx}, and get_cutoff / check_fh as referenced from
sktime.forecasting.base._base.

Runs fit(fh=1..12)+predict on a pd-multiindex panel (vectorised over N
instances) and on the same data as polars_panel.

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/a3_naive_time_share.py --grid dev
"""

import argparse
import functools
import json
import os
import sys
import time
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from a3_time_engine_cost import make_polars_panel  # noqa: E402
from common import env_info  # noqa: E402

warnings.filterwarnings("ignore")
PD3 = int(pd.__version__.split(".")[0]) >= 3
H = 12
STATE = {"depth": 0, "time": 0.0, "calls": {}}


def wrap(owner, name, label):
    orig = getattr(owner, name)

    @functools.wraps(orig)
    def w(*a, **k):
        STATE["calls"][label] = STATE["calls"].get(label, 0) + 1
        if STATE["depth"]:
            return orig(*a, **k)
        STATE["depth"] += 1
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            STATE["time"] += time.perf_counter() - t0
            STATE["depth"] -= 1

    setattr(owner, name, w)
    return orig


def install():
    import sktime.forecasting.base._base as b
    from sktime.forecasting.base._fh import ForecastingHorizon as FH

    for m in ["__init__", "to_absolute", "to_relative", "to_absolute_index",
              "to_absolute_int", "to_indexer", "is_all_in_sample",
              "is_all_out_of_sample", "get_expected_pred_idx"]:
        wrap(FH, m, f"FH.{m}")
    wrap(b, "get_cutoff", "get_cutoff")
    wrap(b, "check_fh", "check_fh")


def run(y):
    from sktime.forecasting.naive import NaiveForecaster

    f = NaiveForecaster(strategy="last")
    t0 = time.perf_counter()
    f.fit(y, fh=list(range(1, H + 1)))
    f.predict()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])
    args = ap.parse_args()
    sizes = [(10, 100), (100, 100), (1000, 100)]
    if args.grid == "large":
        sizes += [(10_000, 50)]
    from sktime.datatypes import convert_to

    out = {"env": env_info(), "rows": []}
    # warm-up imports
    run(make_polars_panel(3, 30, "MS", False).to_pandas().set_index(["h0", "time"]))
    # 1) unwrapped wall times
    data = {}
    for n, t in sizes:
        for freq in ("MS", "D"):
            df = make_polars_panel(n, t, freq, False)
            y_pd = df.to_pandas().set_index(["h0", "time"])
            y_pl = convert_to(y_pd, "polars_panel")
            data[(n, t, freq)] = (y_pd, y_pl)
            w_pd = run(y_pd)
            w_pl = run(y_pl)
            out["rows"].append({"n_series": n, "n_time": t, "freq": freq,
                                "wall_pd_unwrapped_s": w_pd,
                                "wall_pl_unwrapped_s": w_pl})
    # 2) wrapped: time inside time-axis entry points
    install()
    for row in out["rows"]:
        y_pd, y_pl = data[(row["n_series"], row["n_time"], row["freq"])]
        for kind, y in (("pd", y_pd), ("pl", y_pl)):
            STATE.update(time=0.0, calls={})
            w = run(y)
            row[f"wall_{kind}_wrapped_s"] = w
            row[f"time_axis_{kind}_s"] = STATE["time"]
            row[f"time_axis_share_{kind}"] = STATE["time"] / w
            row[f"calls_{kind}"] = dict(STATE["calls"])
        print(f"N={row['n_series']:5d} T={row['n_time']} {row['freq']:2s} "
              f"pd: wall={row['wall_pd_unwrapped_s']:.2f}s (wrapped "
              f"{row['wall_pd_wrapped_s']:.2f}s) time-axis={row['time_axis_pd_s']:.2f}s "
              f"share={100 * row['time_axis_share_pd']:.1f}% | "
              f"pl: wall={row['wall_pl_unwrapped_s']:.2f}s time-axis="
              f"{row['time_axis_pl_s']:.2f}s share={100 * row['time_axis_share_pl']:.1f}%"
              f" | per-instance time-axis "
              f"{1e3 * row['time_axis_pd_s'] / row['n_series']:.2f} ms", flush=True)
        print("    calls(pd):", row["calls_pd"], flush=True)
    tag = "pd3" if PD3 else "pd2"
    path = (f"experiments/container_independence/results/"
            f"a3_naive_time_share_{args.grid}_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
