"""g3: merge bench + memory results into results/g3_stack.json and print markdown tables.

    python3 experiments/container_independence/scripts/g3_tables.py
"""

import json
import os

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LV = ["C0", "C1", "C2", "C3", "C4", "C5"]


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    rows, env = [], None
    for e in ("naive", "poly"):
        d = load(f"g3_stack_bench_{e}.json")
        if d:
            rows += d["rows"]
            env = d["env"]
    mem = None
    for nm in ("g3_stack_mem_a.json", "g3_stack_mem_b.json"):
        d = load(nm)
        if d:
            mem = mem or {"mem": []}
            mem["mem"] += d["mem"]
    pd3 = load("g3_stack_eq_pd3.json")

    out = {"env": env, "workload": "fit(y, fh=[1,2,3]) + predict(), Panel, daily, 1 col",
           "levels": {
               "C0": "stock sktime",
               "C1": "+ vectorized polars monotonic check + known-mtype pass-through",
               "C2": "+ g1 trusted per-slice path",
               "C3": "+ vectorized outer cutoff, panel step once (trusted slice freq), "
                     "fh.to_absolute once per unique cutoff",
               "C4": "+ hub ingress/positional iteration, trusted pandas views to "
                     "unchanged forecasters, native egress",
               "C5": "+ panel-native g2 kernel (G2Naive / G2PolynomialTrend)"},
           "rows": [], "memory_fresh_process": mem["mem"] if mem else None,
           "equality_pd3": pd3["rows"] if pd3 else None}
    print("| est | cont | size | " + " | ".join(f"{lv} s (×C0 / ×prev)" for lv in LV)
          + " | all eq |")
    print("|---|---|---|" + "---|" * (len(LV) + 1))
    for r in rows:
        L = r["levels"]
        cells = []
        alleq = True
        slim = {}
        for lv in LV:
            x = L[lv]
            eq = x["equality_vs_C0"]
            if eq is not None:
                ok = (eq["pred_close_1e9"] and eq["cutoff_same"] and eq["type_same"]
                      and (eq.get("forecasters_index_same") in (True, None))
                      and not eq.get("slice_state_mismatch"))
                alleq &= bool(ok)
            prev = "" if x["x_vs_prev"] is None else f" / {x['x_vs_prev']:.2f}"
            cells.append(f"{x['time_s']:.3g} ({x['x_vs_C0']:.2f}{prev})")
            slim[lv] = {k: x[k] for k in ("time_s", "x_vs_C0", "x_vs_prev",
                                          "peak_rss_delta_mb_inproc", "equality_vs_C0",
                                          "stats")}
        print(f"| {r['est']} | {r['cont']} | {r['n_series']}x{r['n_time']} | "
              + " | ".join(cells) + f" | {'yes' if alleq else 'NO'} |")
        out["rows"].append({"est": r["est"], "cont": r["cont"], "n_series": r["n_series"],
                            "n_time": r["n_time"], "reps": r["reps"],
                            "all_equal_vs_C0": alleq, "levels": slim})
    print()
    print("in-process peak RSS delta (MB)")
    print("| est | cont | size | " + " | ".join(LV) + " |")
    print("|---|---|---|" + "---|" * len(LV))
    for r in rows:
        L = r["levels"]
        print(f"| {r['est']} | {r['cont']} | {r['n_series']}x{r['n_time']} | "
              + " | ".join(f"{L[lv]['peak_rss_delta_mb_inproc']:.0f}"
                           if L[lv]['peak_rss_delta_mb_inproc'] is not None else "-"
                           for lv in LV) + " |")
    if mem:
        print()
        print("fresh-process peak RSS delta (MB) / time (s)")
        by = {}
        for m in mem["mem"]:
            k = (m["size"], m["est"], m["cont"])
            key = f"C{m['level']}" + ("_nog2" if m.get("no_g2") else "")
            by.setdefault(k, {})[key] = m
        cols = LV + ["C0_nog2"]
        print("| size | est | cont | " + " | ".join(cols) + " |")
        print("|---|---|---|" + "---|" * len(cols))
        for k, v in by.items():
            print(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(
                f"{v[c]['peak_rss_delta_mb']:.0f} / {v[c]['time_s']:.2f}" if c in v
                else "-" for c in cols) + " |")
    json.dump(out, open(os.path.join(R, "g3_stack.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
