"""Verifier: Fugue fidelity (output type for polars-annotated fn) and a FAIR baseline
(pandas loop that also builds the same 1-row pd.DataFrame per group as fn_pandas)."""
import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from common import DEV_GRID, LARGE_GRID, bench, make_panel_fast

def f(a): return float(a.mean() + a[-1])
def fn_pandas(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"h0": [df["h0"].iloc[0]], "r": [f(df["c0"].to_numpy())]})

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="dev", choices=["dev","large"])
    a = ap.parse_args(); grid = DEV_GRID[:3] if a.grid == "dev" else LARGE_GRID
    import fugue.api as fa, polars as pl, pyarrow as pa
    out = {"fidelity": {}, "rows": []}
    flat = make_panel_fast(3, 4).reset_index()
    pldf = pl.from_pandas(flat)
    def fn_pl(df: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame({"h0": [df["h0"][0]], "r": [f(df["c0"].to_numpy())]})
    seen = {}
    def fn_pl_probe(df: pl.DataFrame) -> pl.DataFrame:
        seen["type_in_fn"] = type(df).__module__; return fn_pl(df)
    part = {"by": "h0", "presort": "time"}
    for nm, d, fn in [("pl in, fn(pl)", pldf, fn_pl_probe), ("pl in, fn(pd)", pldf, fn_pandas),
                      ("pa in, fn(pd)", pa.table(flat), fn_pandas), ("pd in, fn(pl)", flat, fn_pl)]:
        r = fa.transform(d, fn, schema="h0:str,r:double", partition=part)
        out["fidelity"][nm] = type(r).__module__ + "." + type(r).__name__
    out["fidelity"]["polars fn sees"] = seen
    # Does no-partition (whole frame) path keep polars? (coarse use)
    r = fa.transform(pldf, fn_pl, schema="h0:str,r:double")
    out["fidelity"]["pl in, fn(pl), no partition"] = type(r).__module__ + "." + type(r).__name__
    print(json.dumps(out["fidelity"], indent=1))
    for ns, nt in grid:
        pdf = make_panel_fast(ns, nt); flat = pdf.reset_index()[["h0","time","c0"]]
        def base_np(): return {k: f(g["c0"].to_numpy()) for k, g in pdf.groupby(level=0, sort=False)}
        def base_fair():  # same per-group output construction as fugue's fn_pandas
            return pd.concat([fn_pandas(g.reset_index()) for _, g in pdf.groupby(level=0, sort=False)], ignore_index=True)
        def fug(): return fa.transform(flat, fn_pandas, schema="h0:str,r:double", partition=part)
        t = {k: bench(v, repeat=3) for k, v in [("base_numpy_loop", base_np), ("base_fair_frame_loop", base_fair), ("fugue_pd", fug)]}
        row = {"n_series": ns, "n_time": nt, **{k: v*1e3 for k, v in t.items()},
               "fugue/base_numpy": t["fugue_pd"]/t["base_numpy_loop"], "fugue/base_fair": t["fugue_pd"]/t["base_fair_frame_loop"],
               "fugue_extra_us_per_group_vs_fair": (t["fugue_pd"]-t["base_fair_frame_loop"])/ns*1e6}
        print(json.dumps(row)); out["rows"].append(row)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"r4_dispatch_arch_verify0_fugue_{a.grid}.json"), "w"), indent=1)
main()
