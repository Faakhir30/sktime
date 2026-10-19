"""Summarize design_engine_lazy pushdown sweep: min wall / min peak over reps, ratios vs pl_eager_all."""
import json
import sys
from collections import defaultdict

f = sys.argv[1] if len(sys.argv) > 1 else "experiments/container_independence/results/design_engine_lazy_pushdown_20000x400.json"
rows = json.load(open(f))
agg = defaultdict(lambda: {"wall": 1e9, "rss": 1e9, "chk": set()})
for r in rows:
    k = (r["layout"], r["strat"], r["variant"])
    agg[k]["wall"] = min(agg[k]["wall"], r["wall_s"])
    agg[k]["rss"] = min(agg[k]["rss"], r["rss_peak_delta_mb"])
    agg[k]["chk"].add(round(r["checksum"], 4))
for layout in ["series_major", "time_major"]:
    for strat in ["last", "mean"]:
        base = agg[(layout, strat, "pl_eager_all")]
        refchk = base["chk"]
        print(f"\n{layout} {strat}: variant | wall s (x vs pl_eager_all) | peak dRSS MB (x) | checksum==ref")
        for (l, s, v), d in agg.items():
            if l == layout and s == strat:
                print(f"| {v} | {d['wall']:.3f} ({base['wall']/d['wall']:.2f}x) | {d['rss']:.0f} ({base['rss']/max(d['rss'],1e-9):.2f}x) | {d['chk']==refchk} |")
