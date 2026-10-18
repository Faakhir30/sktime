"""x6 verifier: re-measure ingress and slicing with fairer baselines.

1. ingress decomposition: how much of current prepare_VectorizedDF is the mtype
   VALIDATION (which the hub skips entirely) vs conversion; plus an "unchecked pandas"
   baseline (prepare_VectorizedDF with a precomputed scitype is not possible, so we
   time mtype() separately and the plain conversion polars -> pandas MI by hand).
2. time-major (interleaved) input, as produced by SQL ORDER BY time / resampling:
   hub ingress (needs permutation) vs current on the SAME input.
3. slicing: current items() vs hub view vs hub->pandas, and what the current slice
   actually carries (freq), so like-for-like is visible.

Run: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
       experiments/container_independence/scripts/x6_proto_hub_verify0_bench.py --grid dev
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ED, "protos"))
warnings.simplefilter("ignore")
warnings.showwarning = lambda *a, **k: None

import x6_hub as H  # noqa: E402
from common import bench, env_info, make_panel_fast, to_polars_long  # noqa: E402


def main():
    import polars as pl

    from sktime.datatypes import convert_to, mtype
    from sktime.datatypes._vectorize import prepare_VectorizedDF

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    a = ap.parse_args()
    sizes = [(2_000, 50), (10_000, 50)] if a.grid == "dev" else [(50_000, 50)]
    out = {"env": env_info(), "ingress": [], "time_major": [], "slicing": []}
    kw = {"key_cols": ["h0"], "time_col": "time"}
    for n_s, n_t in sizes:
        X = make_panel_fast(n_s, n_t)
        pl_sk = convert_to(X, "polars_panel")
        pl_long = to_polars_long(X)
        r = {"n_series": n_s, "n_time": n_t}
        rep = 3
        # interleave: hub, current, components
        r["pd_hub"] = bench(lambda: H.from_native(X), repeat=rep)
        r["pd_current"] = bench(lambda: prepare_VectorizedDF(
            X, iterate_as="Series", is_scitype="Panel"), repeat=rep)
        r["pd_mtype_check_only"] = bench(lambda: mtype(X, as_scitype="Panel"),
                                         repeat=rep)
        r["plsk_hub"] = bench(lambda: H.from_native(pl_sk), repeat=rep)
        r["plsk_current"] = bench(lambda: prepare_VectorizedDF(
            pl_sk, iterate_as="Series", is_scitype="Panel"), repeat=1)
        r["plsk_mtype_check_only"] = bench(lambda: mtype(pl_sk, as_scitype="Panel"),
                                           repeat=1)
        # unchecked hand conversion polars(__index__) -> pandas MI (what a fast
        # pandas-first design without validation would do)

        def plsk_to_pd_unchecked():
            p = pl_sk.to_pandas()
            ic = [c for c in p.columns if c.startswith("__index__")]
            p = p.set_index(ic)
            p.index.names = [c[len("__index__"):] for c in ic]
            return p
        r["plsk_unchecked_to_pandas"] = bench(plsk_to_pd_unchecked, repeat=rep)
        r["pllong_hub"] = bench(lambda: H.from_native(pl_long, **kw), repeat=rep)
        r["pllong_unchecked_to_pandas"] = bench(
            lambda: pl_long.to_pandas().set_index(["h0", "time"]), repeat=rep)
        r["pllong_current"] = bench(lambda: prepare_VectorizedDF(
            pl_long.to_pandas().set_index(["h0", "time"]), iterate_as="Series",
            is_scitype="Panel"), repeat=rep)
        r["ratio_pd_current_over_hub"] = r["pd_current"] / r["pd_hub"]
        r["share_pd_current_in_mtype_check"] = r["pd_mtype_check_only"] / r[
            "pd_current"]
        r["ratio_pd_current_minus_check_over_hub"] = (
            r["pd_current"] - r["pd_mtype_check_only"]) / r["pd_hub"]
        r["ratio_plsk_current_over_hub"] = r["plsk_current"] / r["plsk_hub"]
        r["share_plsk_current_in_mtype_check"] = r["plsk_mtype_check_only"] / r[
            "plsk_current"]
        r["ratio_plsk_unchecked_pandas_over_hub"] = r["plsk_unchecked_to_pandas"] / r[
            "plsk_hub"]
        r["ratio_pllong_current_over_hub"] = r["pllong_current"] / r["pllong_hub"]
        r["ratio_pllong_unchecked_pandas_over_hub"] = r[
            "pllong_unchecked_to_pandas"] / r["pllong_hub"]
        out["ingress"].append(r)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in r.items()}), flush=True)

        # 2. time-major input
        Xt = X.sort_index(level="time")
        plt_ = pl_long.sort("time", maintain_order=True)
        t = {"n_series": n_s, "n_time": n_t}
        t["pd_hub_sorted"] = bench(lambda: H.from_native(X), repeat=rep)
        t["pd_hub_time_major"] = bench(lambda: H.from_native(Xt), repeat=rep)
        t["pd_current_sorted"] = bench(lambda: prepare_VectorizedDF(
            X, iterate_as="Series", is_scitype="Panel"), repeat=rep)
        t["pd_current_time_major"] = bench(lambda: prepare_VectorizedDF(
            Xt, iterate_as="Series", is_scitype="Panel"), repeat=rep)
        t["pl_hub_sorted"] = bench(lambda: H.from_native(pl_long, **kw), repeat=rep)
        t["pl_hub_time_major"] = bench(lambda: H.from_native(plt_, **kw), repeat=rep)
        t["pl_current_time_major"] = bench(lambda: prepare_VectorizedDF(
            plt_.to_pandas().set_index(["h0", "time"]), iterate_as="Series",
            is_scitype="Panel"), repeat=rep)
        hzc = H.from_native(plt_, **kw)
        t["pl_time_major_values_zero_copy"] = bool(np.shares_memory(
            hzc.values, plt_["c0"].to_numpy()))
        rt = H.to_native(hzc)
        t["pl_time_major_roundtrip_equal"] = bool(rt.equals(plt_))
        t["ratio_pd_time_major_over_sorted_hub"] = t["pd_hub_time_major"] / t[
            "pd_hub_sorted"]
        t["ratio_pd_current_over_hub_time_major"] = t["pd_current_time_major"] / t[
            "pd_hub_time_major"]
        t["ratio_pl_current_over_hub_time_major"] = t["pl_current_time_major"] / t[
            "pl_hub_time_major"]
        out["time_major"].append(t)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in t.items()}), flush=True)

        # 3. slicing
        vec, X_mi = prepare_VectorizedDF(X, iterate_as="Series", is_scitype="Panel")
        hub = H.from_pandas(X)
        first = next(iter(vec.items(X_mi)))
        cur_slice = first[-1]
        s = {"n_series": n_s, "n_time": n_t,
             "current_slice_type": type(cur_slice).__name__,
             "current_slice_freq": str(getattr(cur_slice.index, "freq", None))}

        def cur():
            for _ in vec.items(X_mi):
                pass

        def views():
            for _ in hub.iter_slices():
                pass

        def mode(m, as_series):
            def f():
                for v in hub.iter_slices():
                    v.to_pandas(as_series=as_series, freq_mode=m)
            return f

        s["current_items"] = bench(cur, repeat=rep)
        s["hub_views"] = bench(views, repeat=rep)
        s["hub_df_trusted"] = bench(mode("trusted", False), repeat=rep)
        s["hub_df_ctor"] = bench(mode("ctor", False), repeat=rep)
        s["hub_series_trusted"] = bench(mode("trusted", True), repeat=rep)
        for k in list(s):
            if k.startswith(("current_items", "hub_")):
                s[k + "_us"] = s[k] / n_s * 1e6
        s["ratio_items_over_views"] = s["current_items"] / s["hub_views"]
        s["ratio_items_over_df_trusted"] = s["current_items"] / s["hub_df_trusted"]
        s["ratio_items_over_df_ctor"] = s["current_items"] / s["hub_df_ctor"]
        out["slicing"].append(s)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in s.items()}), flush=True)
    tag = "pd" + pd.__version__.split(".")[0]
    with open(os.path.join(ED, "results",
                           f"x6_proto_hub_verify0_bench_{tag}_{a.grid}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
