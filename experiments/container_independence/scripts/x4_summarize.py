"""x4: render result JSONs as markdown tables (ratios computed here, not by hand).

Usage: python experiments/container_independence/scripts/x4_summarize.py [--grid dev]
"""

import argparse
import json
import os

ED = os.path.join(os.path.dirname(__file__), "..")


def load(name):
    p = os.path.join(ED, "results", name)
    return json.load(open(p)) if os.path.exists(p) else None


def ms(x):
    return f"{x * 1e3:.1f}" if x < 10 else f"{x * 1e3:.0f}"


def vec_tables(b):
    out = []
    out.append("#### P: pandas input, current VectorizedDF vs VectorizedDF2(PandasAdapter) [ms]\n")
    out.append("| n_series x T | prepare cur | prepare new | iterate cur | iterate new | reconstruct cur | reconstruct new | total cur | total new | new/cur total |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in b["rows"]:
        out.append(
            f"| {r['n_series']}x{r['n_time']} | {ms(r['P_prepare_cur'])} | {ms(r['P_prepare_new'])} | {ms(r['P_iter_cur'])} | {ms(r['P_iter_new'])} | "
            f"{ms(r['P_recon_cur'])} | {ms(r['P_recon_new'])} | {ms(r['P_total_cur'])} | {ms(r['P_total_new'])} | {r['P_total_new'] / r['P_total_cur']:.2f} |"
        )
    out.append("\n#### L: polars input, current (convert to pandas, iterate, reconstruct, convert back) vs VectorizedDF2(PolarsAdapter) native [ms]\n")
    out.append("| n_series x T | mtype check pl | mtype check pd | prepare cur | prepare new | prepare new (mtype known) | iterate cur | iterate new | iter speedup | reconstruct cur (+convert back) | reconstruct new | iter+recon cur (incl. convert) | iter+recon new | speedup | total cur | total new | speedup |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in b["rows"]:
        out.append(
            f"| {r['n_series']}x{r['n_time']} | {ms(r['L_mtype_check_pl'])} | {ms(r['L_mtype_check_pd'])} | {ms(r['L_prepare_cur'])} | {ms(r['L_prepare_new'])} | {ms(r['L_prepare_new_known_mtype'])} | "
            f"{ms(r['L_iter_cur'])} | {ms(r['L_iter_new'])} | {r['L_iter_cur'] / r['L_iter_new']:.1f}x | {ms(r['L_recon_cur_convert_back'])} | {ms(r['L_recon_new'])} | "
            f"{ms(r['L_iter_recon_cur_incl_convert'])} | {ms(r['L_iter_recon_new'])} | {r['L_iter_recon_cur_incl_convert'] / r['L_iter_recon_new']:.1f}x | "
            f"{ms(r['L_total_cur'])} | {ms(r['L_total_new'])} | {r['L_total_cur'] / r['L_total_new']:.1f}x |"
        )
    out.append("\n#### C: where to convert for a pd.Series-inner forecaster (polars input) [ms]\n")
    out.append("| n_series x T | convert once + pandas slices + slice->pd.Series | polars slices + convert_to(slice, pd.Series) | per-slice / once | raw whole.to_pandas() | raw sum(slice.to_pandas()) | raw ratio |")
    out.append("|---|---|---|---|---|---|---|")
    for r in b["rows"]:
        out.append(
            f"| {r['n_series']}x{r['n_time']} | {ms(r['C_once'])} | {ms(r['C_per_slice'])} | {r['C_per_slice'] / r['C_once']:.2f}x | "
            f"{ms(r['C_raw_whole_to_pandas'])} | {ms(r['C_raw_slices_to_pandas'])} | {r['C_raw_slices_to_pandas'] / r['C_raw_whole_to_pandas']:.0f}x |"
        )
    out.append("\n#### Z: materialising all slices of a polars LazyFrame [ms]\n")
    out.append("| n_series x T | source | schema build (keys only) | collect once + partition_by | per-slice filter+collect | collect_all(filters) | per-slice / once | collect_all / once |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in b["rows"]:
        for src, z in r.get("Z_lazy", {}).items():
            out.append(
                f"| {r['n_series']}x{r['n_time']} | {src} | {ms(z['schema_build'])} | {ms(z['collect_once_partition'])} | {ms(z['per_slice_filter_collect'])} | "
                f"{ms(z['collect_all_filters'])} | {z['per_slice_filter_collect'] / z['collect_once_partition']:.0f}x | {z['collect_all_filters'] / z['collect_once_partition']:.0f}x |"
            )
    d = b.get("D_dispatch_ns")
    if d:
        out.append("\n#### D: dispatch overhead (ns per get_adapter call)\n")
        out.append("| " + " | ".join(d) + " |")
        out.append("|" + "---|" * len(d))
        out.append("| " + " | ".join(f"{v:.0f}" for v in d.values()) + " |")
    m = b.get("M_peak_rss_delta_mb")
    if m:
        out.append(
            f"\n#### M: peak RSS delta, full polars pipeline at {m['n_series']}x{m['n_time']} (polars frame {m['frame_mb_polars']:.1f} MB)\n"
        )
        out.append(f"current (2 runs): {m['current'][0]:.1f} / {m['current'][1]:.1f} MB; native: {m['native'][0]:.1f} / {m['native'][1]:.1f} MB")
    return "\n".join(out)


def fc_tables(f):
    out = []
    runs = f["runs"]
    sizes = sorted({(r["n_series"], r["n_time"]) for r in runs})
    keys = list(f["scenarios"])
    out.append("| scenario | " + " | ".join(f"conv fit / pred (n={n})" for n, _ in sizes[-1:]) + " | out type | preds == A0 |")
    out.append("|---|---|---|---|")
    big = sizes[-1]
    for k in keys:
        r = next((x for x in runs if x["key"] == k and (x["n_series"], x["n_time"]) == big), None)
        if r is None or "conv_fit" not in r:
            continue
        cf, cp = r["conv_fit"], r["conv_predict"]
        out.append(
            f"| {k}: {f['scenarios'][k]} | {cf['to_pandas_calls']}+{cf['from_pandas_calls']} calls ({cf['rows_to_pandas'] + cf['rows_from_pandas']} rows) / "
            f"{cp['to_pandas_calls']}+{cp['from_pandas_calls']} calls ({cp['rows_to_pandas'] + cp['rows_from_pandas']} rows) | {r.get('out_type')} | {r.get('pred_equal_A0')} |"
        )
    out.append("\nTimings (s, min over interleaved rounds); ratios vs A1 (polars input, current) for A*, vs B1 for B2*, vs B3 for B4:\n")
    hdr = "| scenario | " + " | ".join(f"fit n={n} | pred n={n}" for n, _ in sizes) + " |"
    out.append(hdr)
    out.append("|---|" + "---|---|" * len(sizes))
    base_of = {"A0": "A1", "A1": "A1", "A2": "A1", "A2u": "A1", "A3": "A1", "B1": "B1", "B2u": "B1", "B2": "B1", "B3": "B3", "B4": "B3"}
    for k in keys:
        cells = []
        for n, t in sizes:
            r = next((x for x in runs if x["key"] == k and x["n_series"] == n), None)
            b = next((x for x in runs if x["key"] == base_of[k] and x["n_series"] == n), None)
            if r is None or "t_fit_s" not in r:
                cells += ["-", "-"]
                continue
            cells.append(f"{r['t_fit_s']:.3f} ({r['t_fit_s'] / b['t_fit_s']:.2f})")
            cells.append(f"{r['t_predict_s']:.3f} ({r['t_predict_s'] / b['t_predict_s']:.2f})")
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    if any("cpu_fit_s" in x for x in runs):
        out.append("\nCPU time (s, process_time, min over rounds) -- less sensitive to machine contention; same ratio bases:\n")
        out.append(hdr)
        out.append("|---|" + "---|---|" * len(sizes))
        for k in keys:
            cells = []
            for n, t in sizes:
                r = next((x for x in runs if x["key"] == k and x["n_series"] == n), None)
                b = next((x for x in runs if x["key"] == base_of[k] and x["n_series"] == n), None)
                if r is None or "cpu_fit_s" not in r:
                    cells += ["-", "-"]
                    continue
                cells.append(f"{r['cpu_fit_s']:.3f} ({r['cpu_fit_s'] / b['cpu_fit_s']:.2f})")
                cells.append(f"{r['cpu_predict_s']:.3f} ({r['cpu_predict_s'] / b['cpu_predict_s']:.2f})")
            out.append(f"| {k} | " + " | ".join(cells) + " |")
        out.append("\nA3 vs A1 is an A/A control (identical conversions, both iterate in pandas): its ratio is the noise floor.")
    mv = f.get("multivariate")
    if mv:
        out.append("\nMultivariate polars y (2 columns, univariate forecaster => column vectorization):\n")
        for k, v in mv.items():
            out.append(f"- `{k}`: " + (f"ok, cols={v['cols']}" if v["ok"] else f"FAILS `{v['error']}` at `{v['where']}`"))
    prof = f.get("profile")
    if prof:
        out.append(f"\ncProfile cumulative seconds, one fit+predict, n={prof['n_series']}x{prof['n_time']}:\n")
        cols = ["total_s", "convert_to", "check_is_scitype", "get_cutoff", "adapter_get_cutoff", "inner _fit", "inner _predict", "pl.to_pandas", "pl.from_pandas", "items(VectorizedDF)", "items(VectorizedDF2)", "reconstruct(VectorizedDF)", "reconstruct(VectorizedDF2)"]
        out.append("| scenario | " + " | ".join(cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for k, v in prof["by_scenario"].items():
            if "error" in v:
                continue
            out.append(f"| {k} | " + " | ".join(f"{v.get(c, 0):.3f}" for c in cols) + " |")
    return "\n".join(out)


def slice_tables(s):
    out = ["| n_series x T | pd_once | pd_per_slice | np_once | np_per_slice | pl_per_slice | np_reshape | pd_per_slice/pd_once | np_once/np_per_slice |", "|---|---|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        out.append(
            f"| {r['n_series']}x{r['n_time']} | {ms(r['pd_once'])} | {ms(r['pd_per_slice'])} | {ms(r['np_once'])} | {ms(r['np_per_slice'])} | {ms(r['pl_per_slice'])} | "
            f"{r['np_reshape'] * 1e6:.1f} us | {r['pd_per_slice'] / r['pd_once']:.2f}x | {r['np_once'] / r['np_per_slice']:.1f}x |"
        )
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev")
    a = ap.parse_args()
    b = load(f"x4_bench_vectorize_{a.grid}.json")
    if b:
        print("### VectorizedDF level\n")
        print(vec_tables(b))
    f = load(f"x4_forecaster_ext_{a.grid}.json")
    if f:
        print("\n### Forecaster level\n")
        print(fc_tables(f))
    s = load(f"x4_slice_extract_{a.grid}.json")
    if s:
        print("\n### Slice extraction\n")
        print(slice_tables(s))
