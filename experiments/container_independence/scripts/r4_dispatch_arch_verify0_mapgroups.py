"""Verifier: polars map_groups 'slowest' claim with a FAIR baseline (both sides build a 1-row
frame per group), and expression speedup attribution (pandas vectorized agg vs polars expr)."""
import argparse, json, os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, polars as pl
sys.path.insert(0, os.path.dirname(__file__))
from common import DEV_GRID, LARGE_GRID, bench, make_panel_fast
def f(a): return float(a.mean() + a[-1])
ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="dev", choices=["dev","large"])
grid = DEV_GRID[1:] if ap.parse_args().grid == "dev" else LARGE_GRID
rows = []
for ns, nt in grid:
    pdf = make_panel_fast(ns, nt); flat = pdf.reset_index()[["h0","time","c0"]]; pldf = pl.from_pandas(flat)
    C = {
     "pd_loop_numpy": lambda: {k: f(g["c0"].to_numpy()) for k, g in pdf.groupby(level=0, sort=False)},
     "pd_loop_build_pdframe": lambda: pd.concat([pd.DataFrame({"h0": [k], "r": [f(g["c0"].to_numpy())]}) for k, g in pdf.groupby(level=0, sort=False)]),
     "pl_loop_build_plframe(partition_by)": lambda: pl.concat([pl.DataFrame({"h0": [k[0]], "r": [f(g["c0"].to_numpy())]}) for k, g in pldf.partition_by("h0", maintain_order=True, as_dict=True).items()]),
     "pl_map_groups_build_plframe": lambda: pldf.group_by("h0", maintain_order=True).map_groups(lambda g: pl.DataFrame({"h0": [g["h0"][0]], "r": [f(g["c0"].to_numpy())]})),
     "pd_vectorized_agg(mean+last)": lambda: (lambda a: a["mean"] + a["last"])(pdf.groupby(level=0, sort=False)["c0"].agg(["mean", "last"])),
     "pl_expr(mean+last)": lambda: pldf.group_by("h0", maintain_order=True).agg((pl.col("c0").mean() + pl.col("c0").last()).alias("r")),
    }
    t = {k: bench(v, repeat=3) for k, v in C.items()}
    b = t["pd_loop_numpy"]; bf = t["pd_loop_build_pdframe"]
    row = {"n_series": ns, "n_time": nt, **{k: {"ms": v*1e3, "x_vs_pd_loop_numpy": v/b, "x_vs_pd_loop_build_frame": v/bf} for k, v in t.items()}}
    print(json.dumps(row)); rows.append(row)
json.dump(rows, open(os.path.join(os.path.dirname(__file__), "..", "results", "r4_dispatch_arch_verify0_mapgroups_dev.json"), "w"), indent=1)
