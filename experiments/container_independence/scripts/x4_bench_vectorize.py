"""x4: VectorizedDF-level performance of the adapter prototype vs current sktime.

Measured (min-of-repeats, same process, interleaved per size):
  P  pandas input : current VectorizedDF vs VectorizedDF2(PandasAdapter)
                    -> prepare / iterate (items) / reconstruct; overhead ratio
  L  polars input : current (convert to pandas once, iterate pandas, reconstruct,
                    convert back) vs VectorizedDF2(PolarsAdapter) native
  C  conversion placement for a pd.Series-inner forecaster (what the per-slice
     clone has to do), isolated from forecaster overhead:
       once      = convert_to(whole, pd-multiindex) + pandas slices + slice->pd.Series
       per-slice = polars partition + convert_to(slice, pd.Series) per slice
     plus raw library cost: whole.to_pandas() vs sum(slice.to_pandas())
  D  dispatch overhead of get_adapter (ns/call)
  Z  lazy: materialise all slices of a LazyFrame (in-memory and parquet scan):
       collect_once+partition_by vs per-slice filter+collect vs collect_all(filters)
  M  peak RSS delta of full polars pipeline: current vs native (largest size)

Usage:
  PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
    $EXP_PY -W ignore experiments/container_independence/scripts/x4_bench_vectorize.py [--grid dev|large]
"""

import argparse
import gc
import json
import os
import sys
import tempfile
import timeit
import warnings

warnings.filterwarnings("ignore")

import polars as pl  # noqa: E402
from common import (  # noqa: E402
    DEV_GRID,
    LARGE_GRID,
    bench,
    env_info,
    make_panel_fast,
    peak_rss_delta_mb,
    rss_mb,
)

from sktime.datatypes import convert_to, mtype  # noqa: E402
from sktime.datatypes._vectorize import prepare_VectorizedDF  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
from x4_adapter import (  # noqa: E402
    REGISTRY,
    VectorizedDF2,
    get_adapter,
    prepare_VectorizedDF2,
)

ED = os.path.join(os.path.dirname(__file__), "..")


def pipeline_pd_current(y):
    v, X = prepare_VectorizedDF(y, iterate_as="Series", is_scitype="Panel")
    sl = v.as_list(X)
    return v.reconstruct(sl, convert_back=True)


def pipeline_pd_adapter(y):
    v, X = prepare_VectorizedDF2(y, iterate_as="Series", is_scitype="Panel", target="pandas")
    sl = v.as_list(X)
    return v.reconstruct(sl, convert_back=True)


def pipeline_pl_current(y):
    v, X = prepare_VectorizedDF(y, iterate_as="Series", is_scitype="Panel")
    sl = v.as_list(X)
    return v.reconstruct(sl, convert_back=True)  # back to polars_panel


def pipeline_pl_native(y):
    v, X = prepare_VectorizedDF2(y, iterate_as="Series", is_scitype="Panel", target="native")
    sl = v.as_list(X)
    return v.reconstruct(sl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--lazy-max-series", type=int, default=1000)
    ap.add_argument("--max-series", type=int, default=10**9)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    grid = [g for g in grid if g[0] <= args.max_series]
    res = {"env": env_info(), "grid": grid, "rows": []}

    for n_series, n_time in grid:
        rep = 5 if n_series <= 100 else 3
        y_pd = make_panel_fast(n_series, n_time, 1)
        y_pl = convert_to(y_pd, "polars_panel")
        row = {"n_series": n_series, "n_time": n_time, "n_rows": len(y_pd)}

        # ---------------- P: pandas input, stage breakdown -----------------
        v0, X0 = prepare_VectorizedDF(y_pd, iterate_as="Series", is_scitype="Panel")
        v1, X1 = prepare_VectorizedDF2(y_pd, iterate_as="Series", is_scitype="Panel", target="pandas")
        s0, s1 = v0.as_list(X0), v1.as_list(X1)
        row["P_prepare_cur"] = bench(lambda: prepare_VectorizedDF(y_pd, iterate_as="Series", is_scitype="Panel"), repeat=rep)
        row["P_prepare_new"] = bench(lambda: prepare_VectorizedDF2(y_pd, iterate_as="Series", is_scitype="Panel", target="pandas"), repeat=rep)
        row["P_iter_cur"] = bench(lambda: v0.as_list(X0), repeat=rep)
        row["P_iter_new"] = bench(lambda: v1.as_list(X1), repeat=rep)
        row["P_recon_cur"] = bench(lambda: v0.reconstruct(s0), repeat=rep)
        row["P_recon_new"] = bench(lambda: v1.reconstruct(s1), repeat=rep)
        row["P_total_cur"] = bench(lambda: pipeline_pd_current(y_pd), repeat=rep)
        row["P_total_new"] = bench(lambda: pipeline_pd_adapter(y_pd), repeat=rep)
        del s0, s1, X0, X1  # keep peak RSS low for --grid large
        gc.collect()

        # ---------------- L: polars input -------------------------------------
        row["L_mtype_check_pl"] = bench(lambda: mtype(y_pl, as_scitype="Panel"), repeat=rep)
        row["L_mtype_check_pd"] = bench(lambda: mtype(y_pd, as_scitype="Panel"), repeat=rep)
        vc, Xc = prepare_VectorizedDF(y_pl, iterate_as="Series", is_scitype="Panel")
        vn, Xn = prepare_VectorizedDF2(y_pl, iterate_as="Series", is_scitype="Panel", target="native")
        sc, sn = vc.as_list(Xc), vn.as_list(Xn)
        row["L_prepare_cur"] = bench(lambda: prepare_VectorizedDF(y_pl, iterate_as="Series", is_scitype="Panel"), repeat=rep)
        row["L_prepare_new"] = bench(lambda: prepare_VectorizedDF2(y_pl, iterate_as="Series", is_scitype="Panel", target="native"), repeat=rep)
        row["L_prepare_new_known_mtype"] = bench(lambda: prepare_VectorizedDF2(y_pl, iterate_as="Series", is_scitype="Panel", target="native", X_orig_mtype="polars_panel"), repeat=rep)
        row["L_iter_cur"] = bench(lambda: vc.as_list(Xc), repeat=rep)
        row["L_iter_new"] = bench(lambda: vn.as_list(Xn), repeat=rep)
        row["L_recon_cur_convert_back"] = bench(lambda: vc.reconstruct(sc, convert_back=True), repeat=rep)
        row["L_recon_new"] = bench(lambda: vn.reconstruct(sn), repeat=rep)
        row["L_total_cur"] = bench(lambda: pipeline_pl_current(y_pl), repeat=rep)
        row["L_total_new"] = bench(lambda: pipeline_pl_native(y_pl), repeat=rep)
        # schema-free view: iterate+reconstruct only (excluding the shared mtype check)
        row["L_iter_recon_cur_incl_convert"] = bench(
            lambda: vc.reconstruct(
                vc.as_list(convert_to(y_pl, "pd-multiindex")), convert_back=True
            ),
            repeat=rep,
        )
        row["L_iter_recon_new"] = bench(lambda: vn.reconstruct(vn.as_list(y_pl)), repeat=rep)

        # ---------------- C: conversion placement (pd.Series inner) ------------
        def once():
            Xp = convert_to(y_pl, "pd-multiindex")
            return [convert_to(s, "pd.Series") for s in vc.as_list(Xp)]

        def per_slice():
            return [convert_to(s, "pd.Series") for s in vn.as_list(y_pl)]

        row["C_once"] = bench(once, repeat=rep)
        row["C_per_slice"] = bench(per_slice, repeat=rep)
        row["C_raw_whole_to_pandas"] = bench(lambda: y_pl.to_pandas(), repeat=rep)
        row["C_raw_slices_to_pandas"] = bench(lambda: [s.to_pandas() for s in sn], repeat=rep)
        del sc, Xc
        gc.collect()

        # ---------------- Z: lazy strategies ----------------------------------
        if n_series <= args.lazy_max_series:
            lz = y_pl.lazy()
            lad = REGISTRY.by_lib["polars_lazy"]
            res_lazy = {}
            with tempfile.TemporaryDirectory(dir=os.environ.get("X4_TMP")) as td:
                pq = os.path.join(td, "y.parquet")
                y_pl.write_parquet(pq, row_group_size=max(1, n_time * 10))
                for src_name, src in [("mem", lz), ("parquet", pl.scan_parquet(pq))]:
                    vlz = VectorizedDF2(src, iterate_as="Series", is_scitype="Panel")

                    def collect_once(src=src, vlz=vlz):
                        lad.slice_mode = "collect_once"
                        return vlz.as_list(src)

                    def per_slice_collect(src=src, vlz=vlz):
                        lad.slice_mode = "lazy"
                        return [s.collect() for s in vlz.as_list(src)]

                    def collect_all(src=src, vlz=vlz):
                        lad.slice_mode = "lazy"
                        return pl.collect_all(vlz.as_list(src))

                    r = 3 if n_series <= 100 else 1
                    res_lazy[src_name] = {
                        "schema_build": bench(lambda src=src: VectorizedDF2(src, iterate_as="Series", is_scitype="Panel"), repeat=r),
                        "collect_once_partition": bench(collect_once, repeat=r),
                        "per_slice_filter_collect": bench(per_slice_collect, repeat=r),
                        "collect_all_filters": bench(collect_all, repeat=r),
                    }
                lad.slice_mode = "collect_once"
            row["Z_lazy"] = res_lazy

        row["rss_mb_end_of_size"] = rss_mb()
        del sn, vn, vc, y_pd, y_pl
        gc.collect()
        res["rows"].append(row)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    # ---------------- D: dispatch overhead -------------------------------------
    y_pd = make_panel_fast(10, 10, 1)
    y_pl = convert_to(y_pd, "polars_panel")
    d = {}
    for name, obj in [("pandas", y_pd), ("polars", y_pl), ("polars_lazy", y_pl.lazy()), ("pyarrow", y_pl.to_arrow())]:
        n = 200_000
        d[name] = timeit.timeit(lambda obj=obj: get_adapter(obj), number=n) / n * 1e9
    d["isinstance_pd_DataFrame"] = timeit.timeit(lambda: isinstance(y_pd, type(y_pd)), number=200_000) / 200_000 * 1e9
    res["D_dispatch_ns"] = d
    print("dispatch ns/call", d)

    # ---------------- M: memory (largest size) ---------------------------------
    n_series, n_time = grid[-1]
    y_pl = convert_to(make_panel_fast(n_series, n_time, 1), "polars_panel")
    _, m_cur = peak_rss_delta_mb(lambda: pipeline_pl_current(y_pl))
    _, m_new = peak_rss_delta_mb(lambda: pipeline_pl_native(y_pl))
    _, m_cur2 = peak_rss_delta_mb(lambda: pipeline_pl_current(y_pl))
    _, m_new2 = peak_rss_delta_mb(lambda: pipeline_pl_native(y_pl))
    res["M_peak_rss_delta_mb"] = {
        "n_series": n_series,
        "n_time": n_time,
        "frame_mb_polars": y_pl.estimated_size() / 2**20,
        "current": [m_cur, m_cur2],
        "native": [m_new, m_new2],
    }
    print("memory", res["M_peak_rss_delta_mb"])

    out = os.path.join(ED, "results", f"x4_bench_vectorize_{args.grid}{args.tag}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    main()
