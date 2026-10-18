"""x8 verify0: is the polars-vs-pandas T3 gap (claim x8.9) an artefact of the pandas tier code?

Compares x8's ``tier_vectorized_pd`` for naive-last and poly with a leaner pandas
groupby implementation (same output), and with polars eager/lazy, interleaved,
min-of-repeats. Also re-counts warnings (claim x8.11).

Run (from WT root)::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x8_e2e_profile_verify0_tiers.py --grid dev
"""

import argparse
import contextlib
import io
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
GRIDS = {"dev": [1000, 5000, 20000], "large": [5000, 20000, 100000]}


def lean_pd_naive(y, H=3):
    """naive-last via one groupby on a flat frame (level 0 only: Panel)."""
    col = y.columns[0]
    t = y.index.get_level_values(-1)
    flat = pd.DataFrame({"k": y.index.get_level_values(0), "t": t, "v": y[col].to_numpy()})
    g = flat.groupby("k", sort=False)
    agg = g.agg(tl=("t", "last"), v=("v", "last"))
    step = agg["tl"] - g["t"].nth(-2).to_numpy()
    rep = np.repeat(np.arange(len(agg)), H)
    hs = np.tile(np.arange(1, H + 1), len(agg))
    fut = agg["tl"].to_numpy()[rep] + step.to_numpy()[rep] * hs
    idx = pd.MultiIndex.from_arrays([agg.index[rep], pd.DatetimeIndex(fut)], names=y.index.names)
    return pd.DataFrame({col: agg["v"].to_numpy()[rep]}, index=idx)


def lean_pd_poly(y, H=3):
    col = y.columns[0]
    t = y.index.get_level_values(-1)
    td = (t - pd.Timestamp(0)) / pd.Timedelta(days=1)
    k = y.index.get_level_values(0)
    v = y[col].to_numpy()
    flat = pd.DataFrame({"k": k, "t": np.asarray(td), "v": v, "tv": np.asarray(td) * v,
                         "tt": np.asarray(td) ** 2})
    g = flat.groupby("k", sort=False)
    s = g.agg(n=("t", "size"), st=("t", "sum"), sv=("v", "sum"), stv=("tv", "sum"),
              stt=("tt", "sum"), tl=("t", "last"))
    b1 = (s.n * s.stv - s.st * s.sv) / (s.n * s.stt - s.st ** 2)
    b0 = (s.sv - b1 * s.st) / s.n
    tl_ts = pd.Series(t).groupby(np.asarray(k), sort=False).last()
    step = tl_ts - pd.Series(t).groupby(np.asarray(k), sort=False).nth(-2).to_numpy()
    rep = np.repeat(np.arange(len(s)), H)
    hs = np.tile(np.arange(1, H + 1), len(s))
    sd = (step / pd.Timedelta(days=1)).to_numpy()
    vals = (b0.to_numpy()[rep] + b1.to_numpy()[rep] * (s.tl.to_numpy()[rep] + sd[rep] * hs))
    fut = tl_ts.to_numpy()[rep] + step.to_numpy()[rep] * hs
    idx = pd.MultiIndex.from_arrays([s.index[rep], pd.DatetimeIndex(fut)], names=y.index.names)
    return pd.DataFrame({col: vals}, index=idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="dev", choices=sorted(GRIDS))
    a = p.parse_args()
    warnings.simplefilter("ignore")
    from common import bench, env_info, make_panel_fast

    from sktime.datatypes import convert_to

    import x8_e2e_profile as x8
    from x8_e2e_profile_verify0_sampling import strict_compare

    res = {"env": env_info(), "rows": []}
    for n in GRIDS[a.grid]:
        y = make_panel_fast(n, 100)
        ypl = convert_to(y, "polars_panel")
        for est in ["naive", "poly"]:
            lean = lean_pd_naive if est == "naive" else lean_pd_poly
            fns = {
                "x8_T3_pd": lambda: x8.tier_vectorized_pd(y, est),
                "lean_pd": lambda: lean(y),
                "pl_eager": lambda: x8.tier_vectorized_pl(ypl, est),
                "pl_lazy": lambda: x8.tier_vectorized_pl(ypl, est, lazy=True),
            }
            t = {k: np.inf for k in fns}
            for _ in range(5):  # interleave
                for k, f in fns.items():
                    t[k] = min(t[k], bench(f, repeat=1, warmup=0))
            row = {"n": n, "est": est, "t": t,
                   "x8pd/pl_eager": t["x8_T3_pd"] / t["pl_eager"],
                   "lean_pd/pl_eager": t["lean_pd"] / t["pl_eager"],
                   "lean_valid_vs_x8": strict_compare(x8.tier_vectorized_pd(y, est), lean(y))}
            res["rows"].append(row)
            print(json.dumps(row, default=float), flush=True)

    # warnings re-count (claim 11)
    from sktime.forecasting.naive import NaiveForecaster

    for n in [50, 200]:
        y = make_panel_fast(n, 50)
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            NaiveForecaster().fit(y, fh=[1, 2, 3]).predict()
        txt = buf.getvalue()
        res[f"warn_{n}"] = {"remember_data": txt.count("remember_data"),
                            "FutureWarning": txt.count("FutureWarning"), "bytes": len(txt),
                            "filters0": str(warnings.filters[0][:3])}
        print("warnings", n, res[f"warn_{n}"], flush=True)
    out = os.path.join(os.path.dirname(HERE), "results", f"x8_e2e_profile_verify0_tiers_{a.grid}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
