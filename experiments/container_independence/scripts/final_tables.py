"""final_tables: render the headline comparison tables from results/final_bench.json (markdown)."""

import json
import os

ED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = json.load(open(os.path.join(ED, "results", "final_bench.json")))
SIZES = ["10x500", "100x500", "1000x500", "5000x100", "10000x100", "50000x50"]


def get(sect, size, key):
    return B[sect].get(size, {}).get(key)


def fmt(v, ref=None, unit=""):
    if v is None:
        return "—"
    s = f"{v:.3g}{unit}"
    if ref:
        s += f" ({v / ref:.2g}×)"
    return s


def table(title, sect, rows, ref_key, unit="", sizes=SIZES):
    sizes = [s for s in sizes if any(get(sect, s, k) is not None for _, k in rows)]
    print(f"\n**{title}** (ratio vs `{ref_key}` in parentheses)\n")
    print("| approach | " + " | ".join(sizes) + " |")
    print("|---|" + "---|" * len(sizes))
    for label, k in rows:
        cells = []
        for s in sizes:
            ref = get(sect, s, ref_key)
            cells.append(fmt(get(sect, s, k), ref, unit))
        print(f"| {label} | " + " | ".join(cells) + " |")


table("T1. Iterate: per-slice cost, µs (x3 interleaved in one process)", "iterate_per_slice_us", [
    ("current sktime pandas (S1 VectorizedDF.items)", "current_pandas (S1 VectorizedDF.items)"),
    ("current sktime, slice without stored freq", "current_pandas, no stored freq (S1)"),
    ("current sktime polars-input (iterates converted pandas; x4)", "x4 current polars-input as_list (after convert)"),
    ("x4 adapter[pandas]", "x4 adapter[pandas] as_list"),
    ("x4 adapter[polars] native slices", "x4 adapter[polars] as_list (native slices)"),
    ("x4-style polars offsets slice (S5)", "polars offsets slice (S5, adapter-native)"),
    ("x4-style polars slice -> pandas per slice (S5p)", "polars slice -> pandas per slice (S5p, adapter w/ pd-inner)"),
    ("x5 narwhals[pandas] items (generic)", "x5 items/nw_pd"),
    ("x5 narwhals[pandas] items (numpy fast path)", "x5 items/nw_pd_fast"),
    ("x5 narwhals[polars] items", "x5 items/nw_pl"),
    ("x5 narwhals[polars-lazy] items", "x5 items/nw_pll"),
    ("x6 hub numpy view (S6)", "numpy GroupedArray view (S6, hub)"),
    ("x6 hub -> pandas slice, trusted freq", "x6 hub -> pandas (trusted freq)"),
    ("x6 hub -> pandas slice, validated ctor", "x6 hub -> pandas (public ctor, validated)"),
    ("pyarrow Table.slice (S8)", "pyarrow Table.slice (S8)"),
], "current_pandas (S1 VectorizedDF.items)", sizes=["10x500", "100x500", "1000x500", "10000x100"])

table("T1b. Iterate per-slice at 50000x50 (x6 only; x3/x4/x5 50k points not run)", "iterate_per_slice_us", [
    ("current sktime VectorizedDF.items (x6)", "x6 current VectorizedDF.items"),
    ("x6 hub view", "x6 hub views"),
    ("x6 hub -> pandas, trusted freq", "x6 hub -> pandas (trusted freq)"),
    ("x6 hub -> pandas, validated ctor", "x6 hub -> pandas (public ctor, validated)"),
], "x6 current VectorizedDF.items", sizes=["50000x50"])

table("T2. Reconstruct (fh=2 outputs -> one frame), seconds", "reconstruct_s", [
    ("current sktime pandas pd.concat(keys) (x3 S1)", "current_pandas pd.concat(keys) (S1)"),
    ("current sktime polars-input reconstruct+convert_back (x4)", "x4 current polars-input reconstruct+convert_back"),
    ("x4 adapter[pandas]", "x4 adapter[pandas] reconstruct"),
    ("x4 adapter[polars]", "x4 adapter[polars] reconstruct"),
    ("x5 narwhals[pandas]", "x5 reconstruct/nw_pd"),
    ("x5 narwhals[polars]", "x5 reconstruct/nw_pl"),
    ("x5 narwhals[polars-lazy]", "x5 reconstruct/nw_pll"),
    ("narwhals[pandas] literal-key concat (x3 S7a_lit)", "narwhals[pandas] literal concat (S7a_lit)"),
    ("x6 hub numpy (S6)", "numpy hub (S6)"),
    ("x6 hub -> pandas MI (S6_to_pd)", "numpy hub -> pandas MI (S6_to_pd)"),
    ("polars vectorized concat (S3/S4/S5_vec)", "polars vectorized concat (S3/S4/S5_vec)"),
    ("pyarrow (S8)", "pyarrow (S8)"),
], "current_pandas pd.concat(keys) (S1)", sizes=["10x500", "100x500", "1000x500", "10000x100"])

table("T3. Conversion / prepare (ingest), seconds", "conversion_s", [
    ("current sktime pandas prepare_VectorizedDF (x4, from_product frame)", "x4 prepare current pandas"),
    ("current sktime polars-input prepare (check+convert_to) (x4)", "x4 prepare current polars-input (check+convert_to)"),
    ("  of which mtype check polars_panel", "x4 mtype check polars_panel"),
    ("x4 adapter[polars] prepare", "x4 prepare adapter[polars]"),
    ("x4 adapter[polars] prepare, known mtype", "x4 prepare adapter[polars], known mtype"),
    ("x5 narwhals prepare, pandas input", "x5 prepare/nw_pd"),
    ("x5 narwhals prepare, polars input", "x5 prepare/nw_pl"),
    ("x5 narwhals prepare, polars-lazy input", "x5 prepare/nw_pll"),
    ("x6 hub ingress pandas MI", "x6 ingress pandas_mi hub"),
    ("x6 hub ingress sktime polars_panel", "x6 ingress polars_sktime hub"),
    ("x6 current ingress sktime polars_panel", "x6 ingress polars_sktime current"),
    ("x6 current ingress dask", "x6 ingress dask_sktime current"),
    ("x6 hub ingress dask", "x6 ingress dask_sktime hub"),
    ("x9 polars->pandas via sktime convert_to", "x9 polars->sktime_convert_to"),
    ("x9 polars->pandas via sktime adapter (no check)", "x9 polars->sktime_adapter"),
    ("x9 polars->arrow-backed pandas, flat", "x9 polars->arrow_flat"),
    ("x9 polars->hybrid MI (numpy index, arrow values)", "x9 polars->hybrid_mi"),
], "x4 prepare current pandas")

table("T3b. Round trip prepare+iterate+reconstruct, seconds", "conversion_s", [
    ("current sktime pandas (x4)", "x4 round trip current pandas"),
    ("current sktime polars-input (x4)", "x4 round trip current polars-input"),
    ("x4 adapter[pandas]", "x4 round trip adapter[pandas]"),
    ("x4 adapter[polars]", "x4 round trip adapter[polars]"),
    ("x5 narwhals[pandas] generic", "x5 roundtrip/nw_pd"),
    ("x5 narwhals[pandas] fast path", "x5 roundtrip/nw_pd_fast"),
    ("x5 narwhals[polars]", "x5 roundtrip/nw_pl"),
    ("x5 narwhals[polars-lazy]", "x5 roundtrip/nw_pll"),
    ("x4: polars->pandas once (whole frame)", "x4 whole-frame polars->pandas once"),
    ("x4: polars->pandas per slice (adapter + pd-inner forecaster)", "x4 per-slice polars->pandas (adapter w/ pd-inner)"),
], "x4 round trip current pandas", sizes=["10x500", "100x500", "1000x500", "10000x100"])

table("T4. End-to-end NaiveForecaster(strategy='last') fit+predict, seconds", "e2e_naive_last_s", [
    ("current sktime, pandas input (x8, fh=1..3)", "x8 current sktime, pd input"),
    ("current sktime, polars input (x8)", "x8 current sktime, pl input"),
    ("current sktime, dask input (x8, tag-patched)", "x8 current sktime, dask input"),
    ("current sktime, pandas (x7, fh=1..14)", "x7 current sktime pd"),
    ("x8 T1: same pandas _fit/_predict, no BaseForecaster boilerplate", "x8 T1_lite_loop_pd"),
    ("x6 hub -> pandas slices + stock inner pandas forecaster", "x6 pandas_mi hub_pandas_inner"),
    ("x6 hub numpy kernel, per-series loop", "x6 pandas_mi hub_loop"),
    ("x6 hub numpy kernel, batched", "x6 pandas_mi hub_batched"),
    ("x5 narwhals panel-native, pandas MI input", "x5 nw_pdmi"),
    ("x5 narwhals panel-native, polars", "x5 nw_pl"),
    ("x5 narwhals panel-native, polars lazy", "x5 nw_pll"),
    ("x5 narwhals panel-native, pyarrow", "x5 nw_pa"),
    ("x5 narwhals panel-native, duckdb", "x5 nw_duckdb"),
    ("x5 narwhals panel-native, dask", "x5 nw_dask"),
    ("x7 pandas groupby (engine-native)", "x7 pandas groupby"),
    ("x7 polars eager", "x7 polars eager"),
    ("x7 polars lazy", "x7 polars lazy"),
    ("x7 duckdb", "x7 duckdb"),
    ("x7 numpy GroupedArray", "x7 numpy GroupedArray"),
], "x8 current sktime, pd input")

table("T5. Peak RSS increase, MB", "peak_mem_mb", [
    ("x4 current pandas pipeline (prepare+iter+recon)", "x4 pd_current peak"),
    ("x4 current polars-input pipeline", "x4 pl_current peak"),
    ("x4 adapter[polars] native pipeline", "x4 pl_native peak"),
    ("x4 slices held: current polars-input", "x4 pl_current held slices"),
    ("x4 slices held: adapter[polars]", "x4 pl_native held slices"),
    ("x3 prepare_VectorizedDF from pandas", "x3 conv peak: pd->prepare_VectorizedDF (check+convert+freq+schema)"),
    ("x3 prepare_VectorizedDF from sktime polars", "x3 conv peak: polars_sktime->prepare_VectorizedDF (current polars path)"),
    ("x3 pandas -> GroupedArray (hub)", "x3 conv peak: pd->GroupedArray"),
    ("x3 polars -> narwhals[polars]", "x3 conv peak: polars->narwhals[polars]"),
    ("x5 prepare current polars", "x5 prepare/cur_pl peak"),
    ("x5 prepare narwhals[polars]", "x5 prepare/nw_pl peak"),
    ("x9 polars -> pandas, sktime convert_to", "x9 polars->sktime_convert_to peak"),
    ("x9 polars -> arrow-backed pandas flat", "x9 polars->arrow_flat peak"),
    ("x9 polars -> hybrid MI", "x9 polars->hybrid_mi peak"),
    ("x7 e2e naive: pandas groupby", "x7 a last peak"),
    ("x7 e2e naive: polars eager", "x7 b last peak"),
    ("x7 e2e naive: polars lazy", "x7 c last peak"),
    ("x7 e2e naive: duckdb", "x7 d last peak"),
    ("x7 e2e naive: numpy GroupedArray", "x7 e2 last peak"),
], "x4 pd_current peak", sizes=["1000x500", "10000x100", "50000x50"])

table("T5b. Peak RSS increase, MB (x6_mem, one process per case)", "peak_mem_mb", [
    ("ingress current pandas MI", "x6 ing_pd_current_ingress peak"),
    ("ingress hub pandas MI", "x6 ing_pd_hub_ingress peak"),
    ("ingress current sktime polars", "x6 ing_pl_current_ingress peak"),
    ("ingress hub polars", "x6 ing_pl_hub_ingress peak"),
    ("ingress current pyarrow", "x6 ing_pa_current_ingress peak"),
    ("ingress hub pyarrow", "x6 ing_pa_hub_ingress peak"),
    ("e2e naive current sktime pandas", "x6 e2e_pd_current_e2e peak"),
    ("e2e naive current sktime polars", "x6 e2e_pl_current_e2e peak"),
    ("e2e naive hub batched (pandas in/out)", "x6 e2e_pd_hub_batched_e2e peak"),
    ("e2e naive hub batched (polars in/out)", "x6 e2e_pl_hub_batched_e2e peak"),
], "x6 ing_pd_current_ingress peak", sizes=["50000x100", "50000x100c4", "5000x100"])

lz = B["lazy_pushdown"]
if lz:
    print(f"\n**T6. Lazy pushdown: select 1000 series × last 50 points from a "
          f"{lz['n_series']}x{lz['n_time']} parquet (x9, contiguous / random ids)**\n")
    print("| variant | time contiguous (s) | × vs sktime | peak MB | time random (s) | peak MB |")
    print("|---|---|---|---|---|---|")
    c, r = lz["contiguous"], lz["random"]
    ref = c["time_s"]["B1_sktime_eager_all"]
    for k in c["time_s"]:
        print(f"| {k} | {c['time_s'][k]:.4g} | {ref / c['time_s'][k]:.3g} | {c['peak_mb'][k]:.1f} | "
              f"{r['time_s'][k]:.4g} | {r['peak_mb'][k]:.1f} |")
