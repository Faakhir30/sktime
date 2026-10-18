"""x8 (aux): turn results/x8_e2e_profile_<grid>.json into markdown tables.

Run (from worktree root)::

    PYTHONPATH=.:experiments/container_independence $EXP_PY \
        experiments/container_independence/scripts/x8_summarize.py --grid dev
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

ROLL = [
    ("container_once", "container-once"),
    ("conversion_in_out", "  of which convert in/out"),
    ("per_instance_framework", "per-instance framework"),
    ("pi_boilerplate", "  of which BaseForecaster boilerplate"),
    ("inner_algorithm_only", "inner algorithm"),
    ("inner_total", "inner incl. nested sktime boilerplate"),
]


def f(x, nd=3):
    return f"{x:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="dev")
    a = p.parse_args()
    path = os.path.join(RES, f"x8_e2e_profile_{a.grid}.json")
    runs = json.load(open(path))["runs"]
    runs.sort(key=lambda r: (r["spec"]["est"], r["spec"]["scitype"], r["spec"]["n_series"]))
    out = []
    out.append(f"## Wall clock (uninstrumented, min of reps) - grid {a.grid}\n")
    out.append(
        "| est | scitype | n | pd s | pl s | dask s | pl/pd | ms/inst (pd) | valid pl | valid dask |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        s, c = r["spec"], r["containers"]
        pdw = c["pd"]["wall_total"]
        plw = c.get("pl", {}).get("wall_total")
        dw = c.get("dask", {}).get("wall_total")
        out.append(
            f"| {s['est']} | {s['scitype']} | {s['n_series']} | {f(pdw)} | "
            f"{f(plw) if plw else '-'} | {f(dw) if dw else '-'} | "
            f"{f(plw / pdw, 2) if plw else '-'} | {f(1000 * pdw / s['n_series'], 2)} | "
            f"{c.get('pl', {}).get('valid_vs_pd', {}).get('ok', '-')} | "
            f"{c.get('dask', {}).get('valid_vs_pd', {}).get('ok', '-')} |"
        )
    out.append("\n## Bucket rollups (% of instrumented exclusive time; abs s in brackets)\n")
    out.append("| est | scitype | n | cont | instr/wall | " + " | ".join(l for _, l in ROLL) + " |")
    out.append("|---|---|---|---|---|" + "---|" * len(ROLL))
    for r in runs:
        s = r["spec"]
        for cn, c in r["containers"].items():
            cells = [
                f"{c['rollups_pct'][k]:.1f}% ({c['rollups'][k]:.3f})" for k, _ in ROLL
            ]
            out.append(
                f"| {s['est']} | {s['scitype']} | {s['n_series']} | {cn} | "
                f"{c['instr_overhead_ratio']:.2f} | " + " | ".join(cells) + " |"
            )
    out.append("\n## Fine buckets, polars container (seconds)\n")
    keys = list(runs[0]["containers"]["pd"]["buckets"].keys())
    out.append("| est | scitype | n | " + " | ".join(keys) + " |")
    out.append("|---|---|---|" + "---|" * len(keys))
    for r in runs:
        s = r["spec"]
        for cn in ["pd", "pl"]:
            b = r["containers"][cn]["buckets"]
            out.append(
                f"| {s['est']} | {s['scitype']} | {s['n_series']} {cn} | "
                + " | ".join(f"{b.get(k, 0):.3f}" for k in keys)
                + " |"
            )
    out.append("\n## Tiers vs sktime (pandas input); speedup = sktime_pd / tier\n")
    out.append(
        "| est | scitype | n | sktime pd | T1 lite | T2 np loop pd | T3 pd groupby | "
        "T2 np loop pl | T3 pl eager | T3 pl lazy | x T1 | x T3pd | x T3pl | all valid |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        s, t = r["spec"], r["tiers"]
        sk = r["containers"]["pd"]["wall_total"]
        skpl = r["containers"]["pl"]["wall_total"]
        valid = all(v["ok"] for k, v in t.items() if k.endswith("valid"))
        g = lambda k: t.get(k)  # noqa: E731
        out.append(
            f"| {s['est']} | {s['scitype']} | {s['n_series']} | {f(sk)} | "
            f"{f(g('T1_lite_loop_pd')) if g('T1_lite_loop_pd') else '-'} | "
            f"{f(g('T2_numpy_loop_pd'), 4)} | {f(g('T3_vectorized_pd'), 4)} | "
            f"{f(g('T2_numpy_loop_pl'), 4)} | {f(g('T3_polars_eager'), 4)} | "
            f"{f(g('T3_polars_lazy'), 4)} | "
            f"{f(sk / g('T1_lite_loop_pd'), 2) if g('T1_lite_loop_pd') else '-'} | "
            f"{sk / g('T3_vectorized_pd'):.0f} | {skpl / g('T3_polars_eager'):.0f} | {valid} |"
        )
    out.append("\n## Conversion micro-bench (s) vs sktime pl wall\n")
    out.append("| est | scitype | n | pl->pd | pd->pl (input) | pd->pl (output-sized) | conv in+out / pl wall |")
    out.append("|---|---|---|---|---|---|---|")
    for r in runs:
        s, cm = r["spec"], r["conversion_microbench"]
        w = r["containers"]["pl"]["wall_total"]
        out.append(
            f"| {s['est']} | {s['scitype']} | {s['n_series']} | {f(cm['pl_to_pd'], 4)} | "
            f"{f(cm['pd_to_pl'], 4)} | {f(cm['pd_to_pl_output_sized'], 4)} | "
            f"{100 * (cm['pl_to_pd'] + cm['pd_to_pl_output_sized']) / w:.2f}% |"
        )
    out.append("\n## cProfile cross-check (cumtime s, n<=cprofile_max_n)\n")
    for r in runs:
        s = r["spec"]
        for cn, c in r["containers"].items():
            if "cprofile" not in c:
                continue
            cp = c["cprofile"]
            cum = cp["cumtime"]
            tot = cp["total_under_cprofile"]
            parts = ", ".join(
                f"{k}={v['cumtime']:.3f}({100 * v['cumtime'] / tot:.0f}%, n={v.get('ncalls', '')})"
                for k, v in cum.items()
                if v["cumtime"] > 0
            )
            out.append(f"- {s['est']}/{s['scitype']}/n={s['n_series']}/{cn}: total={tot:.3f}s; {parts}")
    # headline json: per (est, scitype, n) key ratios
    head = []
    for r in runs:
        s, c, t = r["spec"], r["containers"], r["tiers"]
        row = {k: s[k] for k in ["est", "scitype", "n_series", "n_time"]}
        for cn, cc in c.items():
            row[f"wall_{cn}_s"] = cc["wall_total"]
            row[f"{cn}_rollups_pct"] = {k: round(v, 2) for k, v in cc["rollups_pct"].items()}
            row[f"{cn}_buckets_s"] = {k: round(v, 5) for k, v in cc["buckets"].items()}
            row[f"{cn}_valid"] = cc["valid_vs_pd"]["ok"]
        row["pl_over_pd"] = c["pl"]["wall_total"] / c["pd"]["wall_total"]
        row["ms_per_instance_pd"] = 1000 * c["pd"]["wall_total"] / s["n_series"]
        row["tiers_s"] = {k: v for k, v in t.items() if not k.endswith("valid")}
        row["tiers_valid"] = all(v["ok"] for k, v in t.items() if k.endswith("valid"))
        row["speedup_vs_sktime_pd"] = {
            k: c["pd"]["wall_total"] / v for k, v in row["tiers_s"].items()
        }
        row["speedup_T3pl_vs_sktime_pl"] = c["pl"]["wall_total"] / t["T3_polars_eager"]
        row["conversion_microbench_s"] = r["conversion_microbench"]
        row["peak_rss_mb"] = r["peak_rss_mb"]
        head.append(row)
    with open(os.path.join(RES, "x8_e2e_profile.json"), "w") as fh:
        json.dump(
            {"grid": a.grid, "source": os.path.basename(path), "env": runs[0]["env"],
             "rows": head},
            fh, indent=1,
        )
    txt = "\n".join(out) + "\n"
    dst = os.path.join(RES, f"x8_e2e_profile_{a.grid}_tables.md")
    open(dst, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
