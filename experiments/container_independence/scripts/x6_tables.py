"""x6: render markdown tables from results/x6_*.json (numbers in the report come
from here).

Run:  python experiments/container_independence/scripts/x6_tables.py
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if x != x:
            return "-"
        return f"{x:.{nd}f}"
    return str(x)


def bench_tables(path):
    d = json.load(open(path))
    print(f"\n### {os.path.basename(path)}  env={d['env']}\n")
    rows = d["rows"]
    ing = [r for r in rows if r["part"] == "ingress"]
    if ing:
        print("| source | n_series x n_time | hub ingress ms | current ingress ms | "
              "speedup | values zero-copy | time zero-copy | hub copied MB | "
              "current copied MB |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in ing:
            print(f"| {r['source']} | {r['n_series']}x{r['n_time']} | "
                  f"{f(r['t_hub_s']*1e3)} | {f(r['t_current_s']*1e3)} | "
                  f"{f(r['speedup'], 1)} | {f(r['values_zero_copy'])} | "
                  f"{f(r['time_zero_copy'])} | {f(r['hub_copied_MB'])} | "
                  f"{f(r['current_copied_MB'])} |")
    eg = [r for r in rows if r["part"] == "egress"]
    if eg:
        print("\n| target | n_series x n_time | hub egress ms | of which concat ms | "
              "current reconstruct(+convert) ms | speedup | native type |")
        print("|---|---|---|---|---|---|---|")
        for r in eg:
            cur = r.get("t_current_s")
            print(f"| {r['source']} | {r['n_series']}x{r['n_time']} | "
                  f"{f(r['t_hub_s']*1e3)} | {f(r['t_hub_reconstruct_only_s']*1e3)} | "
                  f"{f(cur*1e3) if cur else '-'} | {f(r.get('speedup'), 1)} | "
                  f"{r['native_type']} |")
    sl = [r for r in rows if r["part"] == "slicing"]
    if sl:
        print("\n| n_series x n_time | current items() us/slice | hub view us/slice | "
              "pandas public | pandas none | pandas trusted-freq | pandas ctor-freq | "
              "pandas infer-freq | pandas view shares memory |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in sl:
            print(f"| {r['n_series']}x{r['n_time']} | "
                  f"{f(r['t_current_items_us_per_slice'], 1)} | "
                  f"{f(r['t_hub_views_us_per_slice'], 2)} | "
                  f"{f(r.get('t_hub_pandas_public_us_per_slice'), 1)} | "
                  f"{f(r['t_hub_pandas_none_us_per_slice'], 1)} | "
                  f"{f(r['t_hub_pandas_trusted_us_per_slice'], 1)} | "
                  f"{f(r['t_hub_pandas_ctor_us_per_slice'], 1)} | "
                  f"{f(r['t_hub_pandas_infer_us_per_slice'], 1)} | "
                  f"{f(r['pandas_view_shares_memory'])} |")
    fc = [r for r in rows if r["part"] == "forecaster"]
    if fc:
        print("\n| strategy | source | n_series x n_time | current s | hub + sktime "
              "per slice s | hub + pandas inner s | hub numpy loop s | hub batched s | "
              "x loop | x batched | equal to current |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in fc:
            print(f"| {r['strategy']} | {r['source']} | {r['n_series']}x{r['n_time']} | "
                  f"{f(r.get('t_current_s'), 3)} | "
                  f"{f(r.get('t_hub_pandas_semantic_s'), 3)} | "
                  f"{f(r.get('t_hub_pandas_inner_s'), 3)} | "
                  f"{f(r['t_hub_loop_s'], 4)} | {f(r['t_hub_batched_s'], 4)} | "
                  f"{f(r.get('speedup_loop'), 0)} | {f(r.get('speedup_batched'), 0)} | "
                  f"{f(r.get('equal_to_current'))} |")


def fidelity_tables():
    for tag in ("pd2", "pd3"):
        d = load(f"x6_fidelity_{tag}.json")
        if not d:
            continue
        print(f"\n### fidelity {tag} env={d['env']}\n")
        fixtures = []
        for r in d["rows"]:
            if r["fixture"] not in fixtures:
                fixtures.append(r["fixture"])
        print("| fixture | hand | narwhals | capsule |")
        print("|---|---|---|---|")
        for fx in fixtures:
            cells = []
            for st in ("hand", "nw", "cap"):
                r = [x for x in d["rows"] if x["fixture"] == fx and x["style"] == st][0]
                if r["roundtrip_ok"] is None:
                    cells.append("n/a")
                elif r["roundtrip_ok"]:
                    cells.append("ok" + (" (zc)" if r.get("values_zero_copy") else ""))
                else:
                    cells.append("FAIL: " + r["reason"].replace("\n", " ")
                                 .replace("|", "/")[:70])
            print(f"| {fx} | " + " | ".join(cells) + " |")
        print("\nsummary:", d["summary"])
        print("LOC:", d["loc"])


def mem_tables():
    for p in sorted(glob.glob(os.path.join(RES, "x6_mem_*.json"))):
        d = json.load(open(p))
        print(f"\n### {os.path.basename(p)}\n")
        print("| case | n_series x n_time x cols | peak delta MB (median of reps) | "
              "reps | retained MB | op s |")
        print("|---|---|---|---|---|---|")
        for r in d["rows"]:
            print(f"| {r['case']} | {r['n_series']}x{r['n_time']}x{r['n_cols']} | "
                  f"{f(r['peak_delta_mb'], 1)} | "
                  f"{[round(x, 1) for x in r['reps_peak_delta_mb']]} | "
                  f"{f(r['retained_delta_mb'], 1)} | {f(r['op_s'], 3)} |")


def pushdown_tables():
    for p in sorted(glob.glob(os.path.join(RES, "x6_pushdown_*.json"))):
        d = json.load(open(p))
        print(f"\n### {os.path.basename(p)} (W={d['W']}, predictions allclose: "
              f"{d.get('predictions_allclose_rtol1e-12')})\n")
        print("| case | n_series x n_time | file MB | rows ingested | t min s | "
              "peak delta MB (median) |")
        print("|---|---|---|---|---|---|")
        for r in d["rows"]:
            print(f"| {r['case']} | {r['n_series']}x{r['n_time']} | {f(r['file_mb'], 1)} | "
                  f"{r.get('n_rows_ingested', '-')} | {f(r['t_min_s'], 3)} | "
                  f"{f(r['peak_delta_mb_median'], 1)} |")


def main():
    for p in sorted(glob.glob(os.path.join(RES, "x6_bench_*.json"))):
        bench_tables(p)
    fidelity_tables()
    mem_tables()
    pushdown_tables()
    for tag in ("pd2", "pd3"):
        d = load(f"x6_zerocopy_{tag}.json")
        if d:
            print(f"\n### zerocopy {tag}\n")
            for r in d["contract"] + d["unsorted"]:
                print("-", json.dumps(r))
    d = load("x6_profile.json")
    if d:
        print("\n### profile (cProfile cumulative, current path)\n")
        for run in d["runs"]:
            print(f"\n{run['label']}: total {run['total_s']:.3f}s")
            for k, v in sorted(run["items"].items(), key=lambda kv: -kv[1]["cum_s"]):
                print(f"  - {k}: {v['cum_s']:.3f}s ({100*v['share']:.1f}%), "
                      f"calls {v['calls']}")


if __name__ == "__main__":
    main()
