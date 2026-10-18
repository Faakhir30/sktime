"""final_consolidate: merge the final serial-benchmark outputs into one comparison.

Reads the result files written by the final runs (see reports/final_bench.md) and
writes results/final_bench.json + prints markdown tables (copied into the report).

Run (from WT root)::

    $EXP_PY experiments/container_independence/scripts/final_consolidate.py
"""

import json
import os

ED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ED, "results")

SOURCES = {
    "x3": "x3_iter_bench__final_large_pd2.json",
    "x8": "final_x8_large.json",
    "x4": "x4_bench_vectorize_large_final.json",
    "x4mem": "x4_memory_large.json",
    "x5v": "x5_bench_vectorize_large.json",
    "x5f": "x5_bench_forecasters_large.json",
    "x6": "x6_bench_pd2_large_c1_ingress-slicing-forecaster.json",
    "x6mem": "x6_mem_pd2_large.json",
    "x7": "x7_native_dev_t4_final.json",
    "x7mem": "x7_native_mem_dev_t4_final.json",
    "x9": "x9_arrow_convert_large_pd2.json",
    "x9l": "final_x9_lazy_pushdown_final_pd2.json",
}


def load(k):
    p = os.path.join(RES, SOURCES[k])
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def sz(n, t):
    return f"{n}x{t}"


def r3(x):
    import math

    return None if x is None or not math.isfinite(x) else float(f"{x:.4g}")


def main():
    D = {k: load(k) for k in SOURCES}
    if D["x4"] is None:  # x4 run hit the 600 s timeout at 50000x50: rows are in the log
        lg = os.path.join(RES, "final_x4_bench_vectorize.log")
        if os.path.exists(lg):
            rows = [json.loads(x) for x in open(lg) if x.startswith('{"n_series"')]
            D["x4"] = {"rows": rows, "note": "parsed from final_x4_bench_vectorize.log (timeout at 50000x50)"}
            SOURCES["x4"] = "final_x4_bench_vectorize.log"
    if D["x5v"] is None:  # same for x5_bench_vectorize: parse the printed text tables
        import ast
        import re

        lg = os.path.join(RES, "final_x5_bench_vectorize.log")
        if os.path.exists(lg):
            rows, cur = [], None
            for ln in open(lg):
                m = re.match(r"== n_series=(\d+) n_time=(\d+)", ln)
                if m:
                    cur = {"n_series": int(m[1]), "n_time": int(m[2]), "time_s": {}}
                    continue
                if cur is None:
                    continue
                m = re.match(r"\s+(prepare|items|reconstruct|roundtrip)\s+(.*)", ln)
                if m:
                    for k, v in re.findall(r"(\w+)=([\d.]+)ms", m[2]):
                        cur["time_s"][f"{m[1]}/{k}"] = float(v) / 1e3
                elif ln.strip().startswith("peak_rss_delta_mb"):
                    cur["peak_rss_delta_mb"] = ast.literal_eval(ln.split("peak_rss_delta_mb", 1)[1].strip())
                    rows.append(cur)
                    cur = None
            D["x5v"] = {"rows": rows}
            SOURCES["x5v"] = "final_x5_bench_vectorize.log"
    if D["x7mem"] is None:  # x7 --mem timed out in engine f (sktime at 50k): parse printed rows
        import re

        lg = os.path.join(RES, "final_x7_native_mem.log")
        if os.path.exists(lg):
            rows = []
            for ln in open(lg):
                m = re.match(r"(\w+)\s+(\w+)\s+peak\+([\d.]+)MB retained ([\d.]+)MB", ln)
                if m:
                    rows.append({"method": m[1], "engine": m[2], "n": 50000, "T": 50,
                                 "peak_delta_mb": float(m[3]), "retained_mb": float(m[4])})
            D["x7mem"] = {"rows": rows}
            SOURCES["x7mem"] = "final_x7_native_mem.log"
    out = {"sources": {k: (SOURCES[k] if D[k] is not None else "MISSING") for k in SOURCES}}
    runs_path = os.path.join(RES, "final_runs.jsonl")
    out["runs"] = [json.loads(x) for x in open(runs_path)] if os.path.exists(runs_path) else []

    # ---------------------------------------------------------------- iterate
    it = {}  # size -> approach -> per-slice us
    rec = {}  # size -> approach -> seconds
    conv = {}  # size -> name -> seconds
    if D["x3"]:
        for p in D["x3"]["points"]:
            s = sz(p["n_series"], p["n_time"])
            b, c = p["B_per_slice_us"], p["C_s"]
            it.setdefault(s, {}).update({
                "current_pandas (S1 VectorizedDF.items)": b["S1_sktime"],
                "current_pandas, no stored freq (S1)": b["S1_sktime_nofreq"],
                "pandas positional iloc (S2_iloc)": b["S2_iloc"],
                "pandas built from cached arrays (S2_build, loses freq)": b["S2_build"],
                "polars partition_by (S3)": b["S3"],
                "polars group_by iter (S4)": b["S4"],
                "polars offsets slice (S5, adapter-native)": b["S5"],
                "polars slice -> pandas per slice (S5p, adapter w/ pd-inner)": b["S5p"],
                "polars slice -> pandas fast (S5p_fast)": b["S5p_fast"],
                "numpy GroupedArray view (S6, hub)": b["S6"],
                "narwhals[pandas] group_by (S7a)": b["S7a"],
                "narwhals[pandas] -> time-indexed pandas (S7a_pd)": b["S7a_pd"],
                "narwhals[polars] group_by (S7b)": b["S7b"],
                "pyarrow Table.slice (S8)": b["S8"],
            })
            rec.setdefault(s, {}).update({
                "current_pandas pd.concat(keys) (S1)": c["S1_sktime"],
                "pandas codes-based (S2)": c["S2_codes"],
                "polars vectorized concat (S3/S4/S5_vec)": c["S3/S4/S5_vec"],
                "polars literal-key concat (S3/S4/S5_lit)": c["S3/S4/S5_lit"],
                "numpy hub (S6)": c["S6"],
                "numpy hub -> pandas MI (S6_to_pd)": c["S6_to_pd"],
                "narwhals[pandas] literal concat (S7a_lit)": c["S7a_lit"],
                "pyarrow (S8)": c["S8"],
            })
            conv.setdefault(s, {}).update({f"x3: {k}": v for k, v in p["conv_s"].items()})
    if D["x4"]:
        for r in D["x4"]["rows"]:
            s = sz(r["n_series"], r["n_time"])
            n = r["n_series"]
            it.setdefault(s, {}).update({
                "x4 current pandas as_list": r["P_iter_cur"] / n * 1e6,
                "x4 adapter[pandas] as_list": r["P_iter_new"] / n * 1e6,
                "x4 current polars-input as_list (after convert)": r["L_iter_cur"] / n * 1e6,
                "x4 adapter[polars] as_list (native slices)": r["L_iter_new"] / n * 1e6,
            })
            rec.setdefault(s, {}).update({
                "x4 current pandas reconstruct": r["P_recon_cur"],
                "x4 adapter[pandas] reconstruct": r["P_recon_new"],
                "x4 current polars-input reconstruct+convert_back": r["L_recon_cur_convert_back"],
                "x4 adapter[polars] reconstruct": r["L_recon_new"],
            })
            conv.setdefault(s, {}).update({
                "x4 prepare current pandas": r["P_prepare_cur"],
                "x4 prepare adapter[pandas]": r["P_prepare_new"],
                "x4 prepare current polars-input (check+convert_to)": r["L_prepare_cur"],
                "x4 prepare adapter[polars]": r["L_prepare_new"],
                "x4 prepare adapter[polars], known mtype": r["L_prepare_new_known_mtype"],
                "x4 mtype check polars_panel": r["L_mtype_check_pl"],
                "x4 mtype check pd-multiindex": r["L_mtype_check_pd"],
                "x4 round trip current pandas": r["P_total_cur"],
                "x4 round trip adapter[pandas]": r["P_total_new"],
                "x4 round trip current polars-input": r["L_total_cur"],
                "x4 round trip adapter[polars]": r["L_total_new"],
                "x4 whole-frame polars->pandas once": r["C_once"],
                "x4 per-slice polars->pandas (adapter w/ pd-inner)": r["C_per_slice"],
            })
    if D["x5v"]:
        for r in D["x5v"]["rows"]:
            s = sz(r["n_series"], r["n_time"])
            n = r["n_series"]
            t = r["time_s"]
            it.setdefault(s, {}).update({
                f"x5 {k}": v / n * 1e6 for k, v in t.items() if k.startswith("items/")
            })
            rec.setdefault(s, {}).update({f"x5 {k}": v for k, v in t.items() if k.startswith("reconstruct/")})
            conv.setdefault(s, {}).update(
                {f"x5 {k}": v for k, v in t.items() if k.startswith(("prepare/", "roundtrip/"))}
            )
    if D["x6"]:
        for r in D["x6"]["rows"]:
            s = sz(r["n_series"], r["n_time"])
            if r["part"] == "slicing":
                it.setdefault(s, {}).update({
                    "x6 current VectorizedDF.items": r["t_current_items_us_per_slice"],
                    "x6 hub views": r["t_hub_views_us_per_slice"],
                    "x6 hub -> pandas (trusted freq)": r["t_hub_pandas_trusted_us_per_slice"],
                    "x6 hub -> pandas (public ctor, validated)": r["t_hub_pandas_ctor_us_per_slice"],
                })
            elif r["part"] in ("ingress", "egress"):
                conv.setdefault(s, {}).update({
                    f"x6 {r['part']} {r['source']} current": r.get("t_current_s"),
                    f"x6 {r['part']} {r['source']} hub": r["t_hub_s"],
                })
    if D["x9"]:
        for r in D["x9"]["rows"]:
            s = sz(r["n_series"], r["n_time"]) + ("" if r["n_cols"] == 1 else f"c{r['n_cols']}")
            conv.setdefault(s, {}).update({f"x9 polars->{k}": v for k, v in r["time_s"].items()})

    # ------------------------------------------------------------ end to end
    e2e = {}  # size -> approach -> seconds (NaiveForecaster last, fit+predict)
    if D["x8"]:
        for r in D["x8"]["runs"]:
            s = sz(r["spec"]["n_series"], r["spec"]["n_time"])
            for c, v in r["containers"].items():
                e2e.setdefault(s, {})[f"x8 current sktime, {c} input"] = v["wall_total"]
            for k, v in r["tiers"].items():
                if not isinstance(v, dict):
                    e2e[s][f"x8 {k}"] = v
    if D["x5f"]:
        for r in D["x5f"]["rows"]:
            if r["strategy"] != "last":
                continue
            s = sz(r["n_series"], r["n_time"])
            e2e.setdefault(s, {}).update({f"x5 {k}": v for k, v in r["time_s"].items()})
    if D["x6"]:
        for r in D["x6"]["rows"]:
            if r["part"] != "forecaster" or r["strategy"] != "last":
                continue
            s = sz(r["n_series"], r["n_time"])
            for k in ["t_current_s", "t_hub_batched_s", "t_hub_loop_s", "t_hub_pandas_inner_s",
                      "t_hub_pandas_semantic_s"]:
                if r.get(k) is not None:
                    e2e.setdefault(s, {})[f"x6 {r['source']} {k[2:-2]}"] = r[k]
    if D["x7"]:
        names = {"a": "pandas groupby", "b": "polars eager", "c": "polars lazy",
                 "c_s": "polars streaming", "d": "duckdb", "e": "numpy (n,T)",
                 "e2": "numpy GroupedArray", "ek": "numpy kernel only",
                 "f": "current sktime pd", "f_pl": "current sktime polars input"}
        for r in D["x7"]["results"]:
            if r["method"] != "last":
                continue
            s = sz(r["n_series"], r["T"])
            for k, v in r["best_s"].items():
                e2e.setdefault(s, {})[f"x7 {names.get(k, k)}"] = v

    # ---------------------------------------------------------------- memory
    mem = {}
    if D["x3"]:
        for p in D["x3"]["points"]:
            s = sz(p["n_series"], p["n_time"])
            for k, v in p.get("conv_mem", {}).items():
                mem.setdefault(s, {})[f"x3 conv peak: {k}"] = v.get("peak_mb")
    if D["x8"]:
        for r in D["x8"]["runs"]:
            s = sz(r["spec"]["n_series"], r["spec"]["n_time"])
            mem.setdefault(s, {})["x8 process peak RSS (all containers, MB)"] = r["peak_rss_mb"]
    if D["x4mem"]:
        for r in D["x4mem"]["rows"]:
            s = sz(r["n_series"], r["n_time"])
            mem.setdefault(s, {})[f"x4 {r['variant']} peak"] = r["peak_mb"]
            mem[s][f"x4 {r['variant']} held slices"] = r["held_mb"]
    if D["x5v"]:
        for r in D["x5v"]["rows"]:
            s = sz(r["n_series"], r["n_time"])
            mem.setdefault(s, {}).update({f"x5 {k} peak": v for k, v in r["peak_rss_delta_mb"].items()})
    if D["x6mem"]:
        for r in D["x6mem"]["rows"]:
            s = sz(r["n_series"], r["n_time"]) + ("" if r["n_cols"] == 1 else f"c{r['n_cols']}")
            mem.setdefault(s, {})[f"x6 {r['case']} peak"] = r["peak_delta_mb"]
    if D["x7mem"]:
        for r in D["x7mem"]["rows"]:
            s = sz(r["n"], r["T"])
            mem.setdefault(s, {})[f"x7 {r['engine']} {r['method']} peak"] = r["peak_delta_mb"]
    if D["x9"]:
        for r in D["x9"]["rows"]:
            s = sz(r["n_series"], r["n_time"]) + ("" if r["n_cols"] == 1 else f"c{r['n_cols']}")
            for k, v in r.get("mem", {}).items():
                mem.setdefault(s, {})[f"x9 polars->{k} peak"] = v["peak_delta_mb"]
    lazy = None
    if D["x9l"]:
        lazy = {k: {"time_s": D["x9l"][k]["time_s"],
                    "peak_mb": {v: m["peak_delta_mb"] for v, m in D["x9l"][k]["mem"].items()},
                    "all_agree": D["x9l"][k]["all_agree"]}
                for k in ("contiguous", "random") if k in D["x9l"]}
        lazy["n_series"], lazy["n_time"] = D["x9l"]["n_series"], D["x9l"]["n_time"]

    def rnd(d):
        return {s: {k: r3(v) for k, v in m.items()} for s, m in d.items()}

    out.update({"iterate_per_slice_us": rnd(it), "reconstruct_s": rnd(rec), "conversion_s": rnd(conv),
                "e2e_naive_last_s": rnd(e2e), "peak_mem_mb": rnd(mem), "lazy_pushdown": lazy})
    with open(os.path.join(RES, "final_bench.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    for sect in ["iterate_per_slice_us", "reconstruct_s", "conversion_s", "e2e_naive_last_s", "peak_mem_mb"]:
        print(f"\n### {sect}")
        for s, m in out[sect].items():
            print(f"-- {s}")
            for k, v in m.items():
                print(f"   {k}: {v}")
    print("\nlazy:", json.dumps(lazy, indent=0)[:3000])
    print("missing:", [k for k, v in out["sources"].items() if v == "MISSING"])


if __name__ == "__main__":
    main()
