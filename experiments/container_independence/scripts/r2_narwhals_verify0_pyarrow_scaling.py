"""r2_narwhals_verify0: does nw[pyarrow] group_by iteration scale O(G*N)? Larger G than V5.

Fixed rows/group (n_time); if per-key full-table filter dominates, time ~ G^2 (slope 2).
Also reproduces r2 claim .8's '20-28x at 1000x100' vs bare pandas groupby(level=0).
Usage: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore <this> --grid dev
"""
import argparse, json, os
import numpy as np
import narwhals as nw
import pyarrow as pa
from common import bench, make_panel, env_info, rss_mb

ap = argparse.ArgumentParser(); ap.add_argument("--grid", choices=["dev", "large"], default="dev")
a = ap.parse_args()
Gs = [250, 500, 1000, 2000] if a.grid == "dev" else [1000, 2000, 4000, 8000]
rows = []
for n_time in [10, 100]:
    for G in Gs:
        if G * n_time > 200_000:
            continue
        X = make_panel(n_series=G, n_time=n_time, n_levels=1)
        tbl = pa.Table.from_pandas(X.reset_index(), preserve_index=False)
        f = nw.from_native(tbl)
        def nwi(): 
            for _ in f.group_by("h0"): pass
        def cur():
            for _ in X.groupby(level=0, sort=False): pass
        t = {"nw": 1e9, "cur": 1e9}
        for _ in range(2):
            t["nw"] = min(t["nw"], bench(nwi, repeat=2)); t["cur"] = min(t["cur"], bench(cur, repeat=2))
        rows.append(dict(G=G, n_time=n_time, nw_ms=t["nw"]*1e3, cur_ms=t["cur"]*1e3, ratio=t["nw"]/t["cur"]))
        print(rows[-1], flush=True)
out = {"env": env_info(), "rows": rows}
for n_time in [10, 100]:
    r = [x for x in rows if x["n_time"] == n_time]
    if len(r) > 1:
        out[f"slope_nw_n_time{n_time}"] = float(np.polyfit(np.log([x["G"] for x in r]), np.log([x["nw_ms"] for x in r]), 1)[0])
        out[f"slope_cur_n_time{n_time}"] = float(np.polyfit(np.log([x["G"] for x in r]), np.log([x["cur_ms"] for x in r]), 1)[0])
print(json.dumps({k: v for k, v in out.items() if k.startswith("slope")}), "rss", rss_mb())
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"r2_narwhals_verify0_pyarrow_scaling_{a.grid}.json"), "w"), indent=1)
