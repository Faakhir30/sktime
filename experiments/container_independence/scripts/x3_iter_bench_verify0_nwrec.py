"""x3_iter_bench_verify0: is the S7a narwhals reconstruct (literal key per part) a strawman?

Compares, on the SAME narwhals[pandas] per-instance outputs (last 2 rows):
  nw_lit : per-part with_columns(nw.lit(key)) + nw.concat + set_index  (x3's S7a_lit)
  nw_vec : nw.concat(parts) once + one repeated key Series (np.repeat) + set_index
  S1     : VectorizedDF.reconstruct on pandas outputs (current sktime)
Also: narwhals[polars] group_by order vs first appearance and a sort-based fix.

Usage: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
  experiments/container_independence/scripts/x3_iter_bench_verify0_nwrec.py --grid dev
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from common import DEV_GRID, LARGE_GRID, bench, make_panel_fast, to_polars_long  # noqa


def main():
    import narwhals as nw

    from sktime.datatypes._vectorize import VectorizedDF

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    a = ap.parse_args()
    grid = DEV_GRID if a.grid == "dev" else LARGE_GRID
    for ns, nt in grid[:3]:
        X = make_panel_fast(ns, nt)
        v = VectorizedDF(
            X, iterate_as="Series", is_scitype="Panel",
            X_orig_mtype="pd-multiindex", freq="D",
        )
        pd_outs = [i.iloc[-2:] for _, _, i in v.items(X)]
        nwpd = nw.from_native(X.reset_index(), eager_only=True)
        pairs = [(k, g.drop(["h0"]).tail(2)) for k, g in nwpd.group_by(["h0"])]
        keys = [k for k, _ in pairs]
        parts = [g for _, g in pairs]

        def s1():
            return v.reconstruct(pd_outs, overwrite_index=True)

        def nw_lit():
            ps = [o.with_columns(nw.lit(k[0]).alias("h0")) for k, o in zip(keys, parts)]
            return nw.concat(ps, how="vertical").to_native().set_index(["h0", "time"])

        def nw_vec():
            body = nw.concat(parts, how="vertical")
            lens = np.fromiter((p.shape[0] for p in parts), dtype=np.int64)
            kv = np.repeat(np.array([k[0] for k in keys], dtype=object), lens)
            ks = nw.new_series("h0", kv, nw.String, backend=nw.get_native_namespace(body))
            return body.with_columns(ks).to_native().set_index(["h0", "time"])[["c0"]]

        ops = {"S1": s1, "nw_lit": nw_lit, "nw_vec": nw_vec}
        best = {k: float("inf") for k in ops}
        for _ in range(3):
            for k, f in ops.items():
                best[k] = min(best[k], bench(f, repeat=1))
        ref = s1()
        eq = {k: bool(np.array_equal(ref.to_numpy(), ops[k]().to_numpy())
                      and list(ops[k]().index.names) == list(ref.index.names))
              for k in ("nw_lit", "nw_vec")}
        # narwhals[polars] group_by order and sort-based fix
        nwpl = nw.from_native(to_polars_long(X), eager_only=True)
        first = list(dict.fromkeys(X.index.get_level_values(0)))
        order_raw = [k[0] for k, _ in nwpl.group_by(["h0"])] == first
        fixed = sorted(nwpl.group_by(["h0"]), key=lambda kg: first.index(kg[0][0])) \
            if ns <= 1000 else None
        order_fixed = fixed is not None and [k[0] for k, _ in fixed] == first
        print(json.dumps({
            "point": f"{ns}x{nt}", "pandas": pd.__version__, "s": best,
            "ratio_vs_S1": {k: b / best["S1"] for k, b in best.items()},
            "values_and_names_equal_S1": eq,
            "nw_polars_group_by_order_equals_first_appearance": order_raw,
            "after_python_sort_by_first_appearance": order_fixed,
        }), flush=True)


if __name__ == "__main__":
    main()
