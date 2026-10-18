"""a2_validate: accuracy of the static census vs manual labels and dynamic trace.

Inputs (all produced earlier in this workstream):
  results/a2_census.csv          static AST census (a2_census.py)
  results/a2_manual_labels.csv   manual spot-check labels (read code by hand)
  results/a2_trace_part*.json    dynamic pandas-API trace (a2_trace.py)

Outputs results/a2_validation.json and prints a summary.

Run::

    PYTHONPATH=.:experiments/container_independence $EXP_PY \
        experiments/container_independence/scripts/a2_validate.py
"""

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

ED = Path(__file__).resolve().parents[1]
RES = ED / "results"
HARD = ("label_select", "time_ops", "frame_ops")


def load_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def confusion(pairs):
    c = Counter(pairs)
    keys = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    return {f"{a}->{b}": c[(a, b)] for a in keys for b in keys if c[(a, b)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])  # unused
    ap.parse_args()

    census = {r["name"]: r for r in load_csv(RES / "a2_census.csv")}
    manual = load_csv(RES / "a2_manual_labels.csv")
    out = {}

    # ---------------- static vs manual ----------------
    n = len(manual)
    pb, ps, pv = [], [], []
    mism = []
    for m in manual:
        s = census[m["name"]]
        pb.append((s["bucket"], m["manual_bucket"]))
        ps.append((s["core_style"], m["manual_core_style"]))
        pv.append((s["static_verdict"], m["manual_verdict"]))
        if (
            s["bucket"] != m["manual_bucket"]
            or s["core_style"] != m["manual_core_style"]
            or s["static_verdict"] != m["manual_verdict"]
        ):
            mism.append(
                {
                    "name": m["name"],
                    "static": (s["bucket"], s["core_style"], s["static_verdict"]),
                    "manual": (m["manual_bucket"], m["manual_core_style"], m["manual_verdict"]),
                }
            )
    acc = {
        "n_spotchecked": n,
        "bucket_acc": sum(a == b for a, b in pb) / n,
        "core_style_acc": sum(a == b for a, b in ps) / n,
        "verdict_acc": sum(a == b for a, b in pv) / n,
        "verdict_acc_V1V2_vs_V3_vs_V4": sum(
            (a in ("V1", "V2")) == (b in ("V1", "V2")) and (a == "V4") == (b == "V4")
            for a, b in pv
        )
        / n,
        "bucket_confusion(static->manual)": confusion(pb),
        "core_style_confusion": confusion(ps),
        "verdict_confusion": confusion(pv),
        "mismatches": mism,
    }
    out["static_vs_manual"] = acc
    print(f"spot-checked n={n}")
    print(f"  bucket accuracy      {acc['bucket_acc']:.3f}")
    print(f"  core_style accuracy  {acc['core_style_acc']:.3f}")
    print(f"  verdict accuracy     {acc['verdict_acc']:.3f}")
    print(f"  verdict acc (coarse: V1|V2 / V3 / V4)  {acc['verdict_acc_V1V2_vs_V3_vs_V4']:.3f}")
    print("  verdict confusion", acc["verdict_confusion"])
    print("  core_style confusion", acc["core_style_confusion"])
    for x in mism:
        print("   mismatch", x)

    # ---------------- calibrated population estimate ----------------
    pop = Counter(r["static_verdict"] for r in census.values())
    cond = defaultdict(Counter)
    for a, b in pv:
        cond[a][b] += 1
    est = Counter()
    for sv, cnt in pop.items():
        tot = sum(cond[sv].values())
        if tot == 0:
            est[sv] += cnt
            continue
        for mv, k in cond[sv].items():
            est[mv] += cnt * k / tot
    N = len(census)
    out["population_static_verdict"] = dict(pop)
    out["population_calibrated_verdict"] = {k: round(v, 1) for k, v in est.items()}
    out["calibration_cells"] = {k: dict(v) for k, v in cond.items()}
    print("population static verdict:", {k: (v, f"{100 * v / N:.1f}%") for k, v in sorted(pop.items())})
    print("population calibrated   :", {k: (round(v, 1), f"{100 * v / N:.1f}%") for k, v in sorted(est.items())})
    print("calibration cells (static->manual counts):", {k: dict(v) for k, v in cond.items()})

    # distinct implementations per verdict
    impl = defaultdict(set)
    for r in census.values():
        impl[r["static_verdict"]].add((r["fit_owner"], r["predict_owner"]))
    out["distinct_impls_per_static_verdict"] = {k: len(v) for k, v in impl.items()}
    print("distinct (_fit,_predict) impls per static verdict:", out["distinct_impls_per_static_verdict"])

    # fh/cutoff reliance per verdict
    fhrel = defaultdict(Counter)
    for r in census.values():
        v = r["static_verdict"]
        fhrel[v]["n"] += 1
        fhrel[v]["fh_to_absolute"] += int(r["n_fh_to_absolute"]) > 0
        fhrel[v]["cutoff"] += int(r["n_cutoff"]) > 0
        fhrel[v]["fh_or_cutoff"] += (int(r["n_fh_ops"]) + int(r["n_cutoff"])) > 0
        fhrel[v]["stored_history"] += int(r["n_hist"]) > 0
    out["fh_reliance_by_verdict"] = {k: dict(v) for k, v in fhrel.items()}
    print("fh/cutoff reliance by verdict:", out["fh_reliance_by_verdict"])

    # ---------------- dynamic trace vs static ----------------
    trace = []
    for p in sorted(glob.glob(str(RES / "a2_trace_part*.json"))):
        trace += json.load(open(p))["results"]
    meta = json.load(open(sorted(glob.glob(str(RES / "a2_trace_part*.json")))[0]))["meta"]
    out["trace_meta"] = meta
    st = Counter(t["status"] for t in trace)
    print("trace status:", dict(st))
    rows = []
    for t in trace:
        if t["status"] not in ("ok", "error"):
            continue
        cats = t.get("cats", {})
        own = Counter(cats.get("class", {}))
        child = Counter(cats.get("sktime", {}))
        fw = Counter(cats.get("framework", {}))
        tp = Counter(cats.get("thirdparty", {}))
        own_hard = sum(own.get(c, 0) for c in HARD)
        own_np = own.get("to_numpy", 0)
        dyn = (
            "pandas-semantic-input"
            if own_hard
            else "numpy" if own_np else "no-hard-ops" if sum(own.values()) else "untouched"
        )
        s = census[t["name"]]
        hard_attrs = sorted(
            a
            for a, c in t.get("attrs", {}).get("class", {}).items()
            if a in __import__("a2_census").ATTR2CAT
            and __import__("a2_census").ATTR2CAT[a] in HARD
        )
        tp_attrs = sorted(t.get("attrs", {}).get("thirdparty", {}))
        rows.append(
            {
                "name": t["name"],
                "status": t["status"],
                "steps": "/".join(t.get("steps", [])),
                "bucket": s["bucket"],
                "static_core_style": s["core_style"],
                "static_input_ops": s["input_hard_ops"],
                "dyn_class_style": dyn,
                "dyn_class_hard_attrs": "|".join(hard_attrs),
                "dyn_class_attrs": "|".join(sorted(t.get("attrs", {}).get("class", {}))),
                "dyn_children_hard": sum(child.get(c, 0) for c in HARD),
                "dyn_framework_hard": sum(fw.get(c, 0) for c in HARD),
                "dyn_thirdparty_attrs": "|".join(tp_attrs),
                "dyn_thirdparty_hard": sum(tp.get(c, 0) for c in HARD),
            }
        )
    # agreement on "does own code apply hard pandas ops to input-derived data"
    agree = Counter()
    for r in rows:
        s_hard = r["static_core_style"] == "pandas-semantic-input"
        d_hard = r["dyn_class_style"] == "pandas-semantic-input"
        agree[(s_hard, d_hard)] += 1
    n_tr = len(rows)
    out["trace_rows"] = rows
    out["trace_agreement"] = {
        "n": n_tr,
        "both_hard": agree[(True, True)],
        "both_not_hard": agree[(False, False)],
        "static_only_hard": agree[(True, False)],
        "dynamic_only_hard": agree[(False, True)],
    }
    # coverage: were the statically flagged input-side sites executed in the run?
    cj = json.load(open(RES / "a2_census.json"))["details"]
    by_name = {t["name"]: t for t in trace}
    cov_rows = []
    for r in rows:
        sites = [
            loc for c, a, loc in cj[r["name"]]["hard_origin"] if c == "hard_input"
        ]
        ex = by_name[r["name"]].get("executed", {})
        hit = []
        for loc in sites:
            f, ln = loc.rsplit(":", 1)
            f = f[len("sktime/"):] if f.startswith("sktime/") else f
            if int(ln) in set(ex.get(f, [])):
                hit.append(loc)
        r["static_sites"] = len(sites)
        r["static_sites_executed"] = len(set(hit))
        r["static_sites_executed_list"] = "|".join(sorted(set(hit)))
        cov_rows.append(r)
    so = [r for r in rows if r["static_core_style"] == "pandas-semantic-input" and r["dyn_class_style"] != "pandas-semantic-input"]
    so_notexec = [r["name"] for r in so if r["static_sites_executed"] == 0]
    so_exec = [r["name"] for r in so if r["static_sites_executed"] > 0]
    bh = [r for r in rows if r["static_core_style"] == "pandas-semantic-input" and r["dyn_class_style"] == "pandas-semantic-input"]
    tot_sites = sum(r["static_sites"] for r in rows)
    tot_exec = sum(r["static_sites_executed"] for r in rows)
    out["trace_agreement"]["static_only_hard_sites_not_executed"] = len(so_notexec)
    out["trace_agreement"]["static_only_hard_sites_executed_untraced"] = len(so_exec)
    out["trace_agreement"]["static_input_sites_total"] = tot_sites
    out["trace_agreement"]["static_input_sites_executed"] = tot_exec
    print("trace vs static (own-code hard pandas ops on inputs):", out["trace_agreement"])
    print("  static-only-hard, flagged sites NOT executed in run:", so_notexec)
    print("  static-only-hard, sites executed but no traced hard access:", [(r["name"], r["static_sites_executed_list"]) for r in so if r["static_sites_executed"] > 0])
    print("  dynamic-only-hard (static missed):", [r["name"] for r in rows if r["static_core_style"] != "pandas-semantic-input" and r["dyn_class_style"] == "pandas-semantic-input"])
    print("  static-only-hard (path not exercised or receiver not input):", [r["name"] for r in rows if r["static_core_style"] == "pandas-semantic-input" and r["dyn_class_style"] != "pandas-semantic-input"])
    dc = Counter((r["bucket"], r["dyn_class_style"]) for r in rows)
    print("dyn class style by bucket:", sorted(dc.items()))
    out["dyn_style_by_bucket"] = {f"{a}|{b}": c for (a, b), c in dc.items()}
    # third-party libs touching the pandas API of inputs
    tpu = {r["name"]: r["dyn_thirdparty_attrs"] for r in rows if r["dyn_thirdparty_attrs"]}
    out["thirdparty_pandas_api_touch"] = tpu
    print("third-party code touching input pandas API:")
    for k, v in tpu.items():
        print(f"   {k:40s} {census[k]['lib_family']:22s} {v}")
    attr_freq = Counter()
    for r in rows:
        for a in r["dyn_class_hard_attrs"].split("|"):
            if a:
                attr_freq[a] += 1
    out["dyn_hard_attr_class_counts"] = dict(attr_freq)
    print("hard pandas attrs called by own code (n classes):", attr_freq.most_common())

    with open(RES / "a2_validation.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    with open(RES / "a2_trace_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
