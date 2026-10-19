"""design_narwhals: engine routing for narwhals panel kernels on pandas-MI input.

Question: a narwhals expression kernel run on the pandas backend is 4-16x slower than on polars
(x5) and ~10x slower than a plain pandas groupby (x5 verifier). Should the boundary route
pandas input to the polars engine (convert once in, convert once out) when polars is installed?

Routes for NwPanelNaive (x5 proto), pandas MI in -> pandas MI out, fh=1..12:
  A  nw kernel on the pandas backend (x5 'nw_pdmi')
  B  pandas MI -> reset_index -> pl.from_pandas -> nw kernel on polars -> to_pandas -> set_index MI
  C  plain pandas groupby(level=0).last()-style lower bound (strategy=last only, sorted data)
Equality: A vs B max abs diff and index equality.

Usage: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
   experiments/container_independence/scripts/design_narwhals_routing.py --grid dev
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import polars as pl

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from common import bench, make_panel_fast  # noqa: E402
from protos.x5_nw_forecasters import NwPanelNaive  # noqa: E402

GRIDS = {"smoke": [(50, 60)], "dev": [(1000, 100), (10000, 100)]}
FH = list(range(1, 13))
STRATS = {"last": dict(strategy="last"), "last_sp7": dict(strategy="last", sp=7), "mean": dict(strategy="mean")}


def route_A(y, kw):
    return NwPanelNaive(**kw).fit(y).predict(FH)


def route_B(y, kw):
    names = list(y.index.names)
    flat = y.reset_index()
    p = pl.from_pandas(flat)
    out = NwPanelNaive(**kw).fit(p, keys=names[:-1], time=names[-1]).predict(FH)
    o = out.to_pandas()
    # egress recipe step: restore the schema's time dtype (polars kernel drifts ns -> us)
    o[names[-1]] = o[names[-1]].astype(flat[names[-1]].dtype)
    return o.set_index(names)


def route_C(y):
    last = y.groupby(level=0, sort=False).last()
    cut = y.index.get_level_values(-1).max()
    t = pd.date_range(cut, periods=len(FH) + 1, freq="D")[1:]
    idx = pd.MultiIndex.from_product([last.index, t], names=y.index.names)
    return pd.DataFrame(np.repeat(last.to_numpy(), len(FH), axis=0), index=idx, columns=y.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=list(GRIDS))
    a = ap.parse_args()
    rows = []
    for n, T in GRIDS[a.grid]:
        y = make_panel_fast(n, T)
        for sname, kw in STRATS.items():
            ra, rb = route_A(y, kw), route_B(y, kw)
            ra2 = ra.sort_index()
            rb2 = rb.sort_index()
            eq_idx = ra2.index.equals(rb2.index)
            diff = float(np.nanmax(np.abs(ra2.to_numpy() - rb2.to_numpy()))) if eq_idx else None
            r = {"n": n, "T": T, "strategy": sname, "index_equal": eq_idx, "max_abs_diff": diff,
                 "A_nw_pandas_s": bench(lambda: route_A(y, kw), repeat=3),
                 "B_route_polars_s": bench(lambda: route_B(y, kw), repeat=3)}
            r["A_over_B"] = r["A_nw_pandas_s"] / r["B_route_polars_s"]
            if sname == "last":
                rc = route_C(y)
                r["C_plain_pandas_s"] = bench(lambda: route_C(y), repeat=3)
                r["C_equal_A"] = bool(np.allclose(rc.sort_index().to_numpy(), ra2.to_numpy()))
                r["B_over_C"] = r["B_route_polars_s"] / r["C_plain_pandas_s"]
            rows.append(r)
            print(json.dumps(r), flush=True)
    import importlib.metadata as md
    out = {"versions": {p: md.version(p) for p in ["narwhals", "polars", "pandas"]}, "rows": rows}
    p = os.path.join(HERE, "..", "results", f"design_narwhals_routing_{a.grid}.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
