"""x2 verify0: re-check of the iteration claims (x2 claims 8, 9, 10).

(a) symmetric accounting: groupby(level) iteration WITH and WITHOUT droplevel vs
    narwhals group_by on the flat frame, with and without the reset_index charged.
    Paired alternating rounds, median of per-round ratios.
(b) correctness of the "nw_offsets_iloc" method (min/max row position per group,
    then iloc on the original frame) when instance rows are NOT contiguous
    (time-major row order, which is a valid pd-multiindex frame).
(c) narwhals-on-polars group order: first appearance? stable over runs?

Usage::
    PYTHONPATH=.:experiments/container_independence $EXP_PY{,3} -W ignore \
        experiments/container_independence/scripts/x2_nw_index_fidelity_verify0_iter.py --grid dev
"""

import argparse
import gc
import json
import os
import time

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl

from common import make_panel_fast

HERE = os.path.dirname(os.path.abspath(__file__))


def paired(fns, rounds):
    for f in fns.values():
        f()
    ts = {k: [] for k in fns}
    for _ in range(rounds):
        for k, f in fns.items():
            gc.collect()
            t0 = time.perf_counter_ns()
            f()
            ts[k].append((time.perf_counter_ns() - t0) / 1e6)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    args = ap.parse_args()
    grid = [(100, 100), (1000, 100), (10000, 50)]
    if args.grid == "large":
        grid = [(1000, 500), (10000, 100), (50000, 50)]
    out = {"pandas": pd.__version__, "timing": [], }
    for ns, nt in grid:
        df = make_panel_fast(ns, nt)
        flat = df.reset_index()
        nf = nw.from_native(flat, eager_only=True)
        fns = {
            "gb_level_droplevel": lambda: [(k, g.droplevel(0)) for k, g in df.groupby(level=0, sort=False)],
            "gb_level_plain": lambda: list(df.groupby(level=0, sort=False)),
            "nw_pd_flat_prebuilt": lambda: list(nf.group_by("h0")),
            "nw_pd_incl_reset": lambda: list(nw.from_native(df.reset_index(), eager_only=True).group_by("h0")),
        }
        ts = paired(fns, 5 if ns <= 1000 else 3)
        med = {k: float(np.median(v)) for k, v in ts.items()}
        r = {"n_series": ns, "n_time": nt, **{k + "_ms": round(v, 2) for k, v in med.items()}}
        for k in ("nw_pd_flat_prebuilt", "nw_pd_incl_reset"):
            for base in ("gb_level_droplevel", "gb_level_plain"):
                r[f"{k}/{base}"] = round(float(np.median([a / b for a, b in zip(ts[k], ts[base])])), 3)
        out["timing"].append(r)
        print(json.dumps(r), flush=True)

    # (b) non-contiguous instance rows
    df = make_panel_fast(5, 6)
    dft = df.swaplevel().sort_index().swaplevel()  # rows time-major, MI names unchanged
    nfl = nw.from_native(dft.reset_index(), eager_only=True).with_row_index("__r", order_by=None)
    b = nw.to_native(nfl.group_by("h0").agg(nw.col("__r").min().alias("a"), nw.col("__r").max().alias("b")))
    ok = []
    for k, a, e in zip(b["h0"].to_numpy(), b["a"].to_numpy(), b["b"].to_numpy()):
        sl = dft.iloc[a:e + 1]
        ok.append(bool((sl.index.get_level_values(0) == k).all()) and len(sl) == 6)
    out["offsets_iloc_correct_on_time_major_rows"] = ok
    print("offsets_iloc correct per group on time-major rows:", ok)

    # (c) narwhals polars order
    df = make_panel_fast(200, 5)
    perm = np.random.default_rng(0).permutation(200)
    keys = df.index.get_level_values(0).unique()[perm]
    df = df.loc[keys]
    pdf = pl.from_pandas(df.reset_index())
    first = list(dict.fromkeys(pdf["h0"].to_list()))
    orders = [[k[0] for k, _ in nw.from_native(pdf, eager_only=True).group_by("h0")] for _ in range(5)]
    out["nw_pl_first_appearance"] = [o == first for o in orders]
    out["nw_pl_stable"] = all(o == orders[0] for o in orders)
    nwpd = [k[0] for k, _ in nw.from_native(df.reset_index(), eager_only=True).group_by("h0")]
    out["nw_pd_first_appearance"] = nwpd == [str(x) for x in first] or nwpd == first
    print({k: v for k, v in out.items() if k.startswith("nw_")})

    tag = "pd" + pd.__version__[0]
    with open(os.path.join(HERE, "..", "results", f"x2_nw_index_fidelity_verify0_iter_{tag}_{args.grid}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
