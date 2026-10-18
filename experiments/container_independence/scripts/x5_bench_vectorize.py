"""x5: benchmark narwhals VectorizedDF prototype vs current sktime VectorizedDF.

Phases: schema build (prepare), iterate all slices (items), reconstruct,
full round trip; plus peak RSS of prepare.

Paths
  cur_pd      current sktime, pandas pd-multiindex input
  cur_pl      current sktime, polars "polars_panel" input (convert_to pandas inside,
              reconstruct(convert_back=True) -> polars)
  nw_pd       narwhals proto, pandas MI input (reset_index once, pandas slices
              with time index - what inner sktime estimators expect)
  nw_pl       narwhals proto, polars eager long input, native polars slices
  nw_pll      narwhals proto, polars LazyFrame input (keys query + one collect)

Run:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x5_bench_vectorize.py --grid dev
"""

import argparse
import json
import warnings
from pathlib import Path

import polars as pl

from common import (
    DEV_GRID,
    LARGE_GRID,
    bench,
    env_info,
    make_panel_fast,
    peak_rss_delta_mb,
    rss_mb,
)
from protos.x5_backends import long_from_mi
from protos.x5_narwhals import prepare_nw_vectorized

warnings.filterwarnings("ignore")
ED = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID

    from sktime.datatypes import convert_to
    from sktime.datatypes._vectorize import prepare_VectorizedDF

    out = {"env": env_info(), "grid": args.grid, "rows": []}
    for n_series, n_time in grid:
        X = make_panel_fast(n_series, n_time)
        lng = long_from_mi(X)
        pdf = pl.from_pandas(lng)
        plz = pdf.lazy()
        X_plsk = convert_to(X, to_type="polars_panel")
        rep = 5 if n_series <= 1000 else 2

        def cur_prep(x):
            return prepare_VectorizedDF(x, iterate_as="Series", is_scitype="Panel")

        def nw_prep(x, **kw):
            s, f = prepare_nw_vectorized(x, iterate_as="Series", **kw)
            s.instance_keys(f)
            return s, f

        kwl = dict(keys=["h0"], time="time")
        # ---------------- prepare
        t = {}
        t["prepare/cur_pd"] = bench(lambda: cur_prep(X), repeat=rep)
        t["prepare/cur_pl"] = bench(lambda: cur_prep(X_plsk), repeat=rep)
        t["prepare/nw_pd"] = bench(lambda: nw_prep(X), repeat=rep)
        t["prepare/nw_pl"] = bench(lambda: nw_prep(pdf, **kwl), repeat=rep)
        t["prepare/nw_pll"] = bench(lambda: nw_prep(plz, **kwl), repeat=rep)

        vec_pd, Xmi = cur_prep(X)
        vec_pl, Xmi_pl = cur_prep(X_plsk)
        s_pd, f_pd = nw_prep(X)
        s_pl, f_pl = nw_prep(pdf, **kwl)
        s_pll, f_pll = nw_prep(plz, **kwl)

        # ---------------- items
        t["items/cur_pd"] = bench(lambda: list(vec_pd.items(Xmi)), repeat=rep)
        t["items/nw_pd"] = bench(lambda: list(s_pd.items(f_pd)), repeat=rep)
        t["items/nw_pd_fast"] = bench(
            lambda: list(s_pd.items(f_pd, fast_pandas=True)), repeat=rep
        )
        t["items/nw_pl"] = bench(lambda: list(s_pl.items(f_pl)), repeat=rep)
        t["items/nw_pll"] = bench(lambda: list(s_pll.items(f_pll)), repeat=rep)

        sl_pd = vec_pd.as_list(Xmi)
        sl_nwpd = s_pd.as_list(f_pd)
        sl_pl = s_pl.as_list(f_pl)
        sl_pll = s_pll.as_list(f_pll)

        # ---------------- reconstruct
        t["reconstruct/cur_pd"] = bench(lambda: vec_pd.reconstruct(sl_pd), repeat=rep)
        t["reconstruct/cur_pl"] = bench(
            lambda: vec_pl.reconstruct(sl_pd, convert_back=True), repeat=rep
        )
        t["reconstruct/nw_pd"] = bench(
            lambda: s_pd.reconstruct(f_pd, sl_nwpd), repeat=rep
        )
        t["reconstruct/nw_pl"] = bench(lambda: s_pl.reconstruct(f_pl, sl_pl), repeat=rep)
        t["reconstruct/nw_pll"] = bench(
            lambda: s_pll.reconstruct(f_pll, sl_pll, relazy=True), repeat=rep
        )

        # ---------------- full round trip
        def rt_cur(x, back):
            v, xm = cur_prep(x)
            return v.reconstruct(v.as_list(xm), convert_back=back)

        def rt_nw(x, relazy=False, fast=False, **kw):
            s, f = prepare_nw_vectorized(x, **kw)
            sl = s.as_list(f, fast_pandas=True) if fast else s.as_list(f)
            return s.reconstruct(f, sl, relazy=relazy)

        t["roundtrip/cur_pd"] = bench(lambda: rt_cur(X, False), repeat=rep)
        t["roundtrip/cur_pl"] = bench(lambda: rt_cur(X_plsk, True), repeat=rep)
        t["roundtrip/nw_pd"] = bench(lambda: rt_nw(X), repeat=rep)
        t["roundtrip/nw_pd_fast"] = bench(lambda: rt_nw(X, fast=True), repeat=rep)
        t["roundtrip/nw_pl"] = bench(lambda: rt_nw(pdf, **kwl), repeat=rep)
        t["roundtrip/nw_pll"] = bench(lambda: rt_nw(plz, relazy=True, **kwl), repeat=rep)

        # ---------------- memory (prepare only)
        mem = {}
        _, mem["prepare/cur_pd"] = peak_rss_delta_mb(lambda: cur_prep(X))
        _, mem["prepare/nw_pd"] = peak_rss_delta_mb(lambda: nw_prep(X))
        _, mem["prepare/cur_pl"] = peak_rss_delta_mb(lambda: cur_prep(X_plsk))
        _, mem["prepare/nw_pl"] = peak_rss_delta_mb(lambda: nw_prep(pdf, **kwl))
        _, mem["prepare/nw_pll"] = peak_rss_delta_mb(lambda: nw_prep(plz, **kwl))

        row = {
            "n_series": n_series,
            "n_time": n_time,
            "frame_mb": float(X.memory_usage(deep=True).sum() / 2**20),
            "time_s": t,
            "peak_rss_delta_mb": mem,
            "rss_mb_after": rss_mb(),
        }
        out["rows"].append(row)
        print(f"\n== n_series={n_series} n_time={n_time} frame={row['frame_mb']:.1f}MB")
        for phase in ["prepare", "items", "reconstruct", "roundtrip"]:
            base = t[f"{phase}/cur_pd"]
            cells = [
                f"{k.split('/')[1]}={v*1e3:.2f}ms (x{v/base:.2f})"
                for k, v in t.items()
                if k.startswith(phase + "/")
            ]
            print(f"  {phase:12s} " + "  ".join(cells))
        print("  peak_rss_delta_mb", {k: round(v, 1) for k, v in mem.items()})
        del sl_pd, sl_nwpd, sl_pl, sl_pll

    p = ED / "results" / f"x5_bench_vectorize_{args.grid}.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
