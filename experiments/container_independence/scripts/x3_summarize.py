"""x3: print markdown tables from results/x3_iter_bench__<tag>.json.

Usage (from worktree root)::

    python experiments/container_independence/scripts/x3_summarize.py dev_pd2 [...]
"""

import json
import os
import sys

ED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(tag):
    with open(os.path.join(ED, "results", f"x3_iter_bench__{tag}.json")) as fh:
        return json.load(fh)


def table(points, sect, scale, fmt, ratio_sect=None, title=""):
    keys = list(points[0][sect].keys())
    hdr = ["strategy"] + [f"{p['n_series']}x{p['n_time']}" for p in points]
    print(f"\n**{title}**\n")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for k in keys:
        row = [k]
        for p in points:
            v = p[sect].get(k)
            cell = "-" if v is None else fmt.format(v * scale)
            if ratio_sect and k in p.get(ratio_sect, {}):
                cell += f" ({p[ratio_sect][k]:.2f}x)"
            row.append(cell)
        print("| " + " | ".join(row) + " |")


def main(tags):
    for tag in tags:
        r = load(tag)
        pts = r["points"]
        print(f"\n### {tag}  env={r['env']}  cmd=`{r['cmd']}`")
        table(pts, "A_s", 1e3, "{:.2f}", "A_ratio_vs_S1", "(A) list keys, ms (ratio vs S1)")
        table(
            pts, "B_s", 1e3, "{:.2f}", "B_ratio_vs_S1_sktime",
            "(B) iterate all slices, total ms (ratio vs S1_sktime)",
        )
        table(pts, "B_per_slice_us", 1, "{:.1f}", None, "(B) per-slice us")
        table(
            pts, "C_s", 1e3, "{:.2f}", "C_ratio_vs_S1",
            "(C) reconstruct last-2-rows outputs, total ms (ratio vs S1_sktime)",
        )
        table(pts, "conv_s", 1e3, "{:.2f}", None, "conversion time, ms")
        if "conv_mem" in pts[0]:
            keys = list(pts[0]["conv_mem"].keys())
            print("\n**conversion peak / retained RSS delta, MB (fresh subprocess)**\n")
            hdr = ["conversion"] + [f"{p['n_series']}x{p['n_time']}" for p in pts]
            print("| " + " | ".join(hdr) + " |")
            print("|" + "---|" * len(hdr))
            for k in keys:
                row = [k]
                for p in pts:
                    m = p["conv_mem"].get(k, {})
                    if "peak_mb" in m:
                        row.append(f"{m['peak_mb']:.1f} / {m['retained_mb']:.1f}")
                    else:
                        row.append("err")
                print("| " + " | ".join(row) + " |")
        table(pts, "sizes_mb", 1, "{:.2f}", None, "in-memory size, MB")
        table(pts, "noncontig_s", 1e3, "{:.2f}", None, "contiguous vs time-major input, ms")
        table(pts, "micro_us", 1, "{:.1f}", None, "micro-ops on ONE slice, us")
        cal = [p for p in pts if "calib" in p]
        if cal:
            print("\n**calibration**\n")
            for p in cal:
                print(f"- {p['n_series']}x{p['n_time']}: {p['calib']}")
        print("\n**checks (largest point)**\n")
        print("```")
        print(json.dumps(pts[-1]["checks"], indent=1))
        print("```")


if __name__ == "__main__":
    main(sys.argv[1:])
