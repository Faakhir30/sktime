"""x4 verify0: is the 683 ms (polars) vs 0.6 ms (pandas) mtype-check gap a caching artifact?

pandas Index caches is_monotonic/is_unique etc.; x4_bench_vectorize benches
mtype(y_pd) repeatedly on the SAME object (warm cache). Here: warm (same obj)
vs cold (fresh deep copy of index per call, copy time excluded).
Run: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore <this> --grid dev
"""
import argparse, json, os, time, warnings, gc
warnings.filterwarnings("ignore")
import pandas as pd
from common import DEV_GRID, LARGE_GRID, bench, env_info, make_panel_fast
from sktime.datatypes import convert_to, mtype

ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="dev"); ap.add_argument("--max-series", type=int, default=10**9)
a = ap.parse_args()
grid = [g for g in (DEV_GRID if a.grid == "dev" else LARGE_GRID) if g[0] <= a.max_series]
rows = []
for n, t in grid:
    y_pd = make_panel_fast(n, t, 1); y_pl = convert_to(y_pd, "polars_panel")
    def cold_pd(reps=3):
        best = 1e9
        for _ in range(reps):
            y = y_pd.copy(); y.index = pd.MultiIndex.from_arrays([y_pd.index.get_level_values(i).copy() for i in range(2)], names=y_pd.index.names)
            gc.collect(); t0 = time.perf_counter(); mtype(y, as_scitype="Panel"); best = min(best, time.perf_counter() - t0)
        return best
    r = {"n_series": n, "n_time": t,
         "pd_warm": bench(lambda: mtype(y_pd, as_scitype="Panel"), repeat=3),
         "pd_cold": cold_pd(),
         "pl": bench(lambda: mtype(y_pl, as_scitype="Panel"), repeat=3)}
    r["pl/pd_warm"] = r["pl"] / r["pd_warm"]; r["pl/pd_cold"] = r["pl"] / r["pd_cold"]
    rows.append(r); print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
json.dump({"env": env_info(), "rows": rows}, open(os.path.join("experiments/container_independence/results", f"x4_proto_adapter_verify0_mtype_check_{a.grid}.json"), "w"), indent=1, default=str)
