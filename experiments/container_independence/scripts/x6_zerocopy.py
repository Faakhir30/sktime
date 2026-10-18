"""x6: zero-copy / writability contract of the hub + cost of non-contiguous input.

1. ingress: does hub.values / hub.time share memory with the source? writeable?
2. egress: does the native output share memory with hub.values?
3. per-slice pandas view: shares memory with the hub? writeable?
4. ingress time, contiguous vs shuffled rows (permutation copy), pandas + polars.

Run:
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/x6_zerocopy.py [--grid dev]
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

import x6_hub as H  # noqa: E402
from common import bench, env_info, make_panel_fast, to_polars_long  # noqa: E402


def buffers_of(native):
    import polars as pl
    import pyarrow as pa

    if isinstance(native, pd.DataFrame):
        return [native[c].to_numpy() for c in native.columns] + [
            native.to_numpy(copy=False)]
    if isinstance(native, pl.DataFrame):
        out = []
        for c in native.columns:
            try:
                out.append(native[c].to_numpy())
            except Exception:  # noqa: BLE001
                pass
        return out
    if isinstance(native, pa.Table):
        out = []
        for c in native.column_names:
            for ch in native.column(c).chunks:
                try:
                    out.append(ch.to_numpy(zero_copy_only=True))
                except Exception:  # noqa: BLE001
                    pass
        return out
    return []


def main():
    import polars as pl
    import pyarrow as pa

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    a = ap.parse_args()
    n_s, n_t = (10_000, 50) if a.grid == "dev" else (50_000, 50)
    res = {"env": env_info(), "contract": [], "unsorted": []}
    for n_cols in (1, 3):
        X = make_panel_fast(200, 20, n_cols=n_cols)
        pl_long = to_polars_long(X)
        pa_long = pa.Table.from_pandas(X.reset_index(), preserve_index=False)
        kw = {"key_cols": ["h0"], "time_col": "time"}
        for name, src, k in [("pandas_mi", X, {}), ("polars_long", pl_long, kw),
                             ("arrow_long", pa_long, kw)]:
            hub = H.from_native(src, **k)
            bufs = buffers_of(src)
            out = H.to_native(hub.with_data(hub.indptr, hub.time, hub.values))
            obufs = buffers_of(out)
            sv = hub.slice(3)
            s_pd = sv.to_pandas(as_series=n_cols == 1)
            rec = {
                "source": name, "n_cols": n_cols,
                "ingress_values_zero_copy": any(np.shares_memory(hub.values, b)
                                                for b in bufs),
                "ingress_time_zero_copy": any(np.shares_memory(hub.time, b)
                                              for b in bufs),
                "hub_values_writeable": bool(hub.values.flags.writeable),
                "hub_values_layout": "F" if hub.values.flags.f_contiguous else (
                    "C" if hub.values.flags.c_contiguous else "strided"),
                "egress_values_zero_copy": any(np.shares_memory(hub.values, b)
                                               for b in obufs),
                "pandas_slice_view_zero_copy": bool(np.shares_memory(
                    s_pd.to_numpy(), hub.values)),
                "pandas_slice_freq": str(getattr(s_pd.index, "freq", None)),
            }
            try:
                sv.values[0, 0] = sv.values[0, 0]
                rec["slice_inplace_write"] = "ok"
            except ValueError as e:
                rec["slice_inplace_write"] = f"ValueError: {e}"
            res["contract"].append(rec)
            print(json.dumps(rec))

    # non-contiguous (shuffled) input: one permutation gather at ingress
    X = make_panel_fast(n_s, n_t)
    rng = np.random.default_rng(0)
    Xs = X.iloc[rng.permutation(len(X))]
    pl_long = to_polars_long(X)
    pls = pl_long.sample(fraction=1.0, shuffle=True, seed=0)
    kw = {"key_cols": ["h0"], "time_col": "time"}
    for name, a_, b_, k in [("pandas_mi", X, Xs, {}), ("polars_long", pl_long, pls, kw)]:
        t_sorted = bench(lambda: H.from_native(a_, **k), repeat=3)
        t_shuf = bench(lambda: H.from_native(b_, **k), repeat=3)
        h_shuf = H.from_native(b_, **k)
        rt = H.to_native(h_shuf)
        if isinstance(rt, pd.DataFrame):
            pd.testing.assert_frame_equal(rt, b_)
        else:
            assert rt.equals(b_)
        rec = {"source": name, "n_series": n_s, "n_time": n_t,
               "t_sorted_s": t_sorted, "t_shuffled_s": t_shuf,
               "ratio": t_shuf / t_sorted,
               "values_zero_copy_shuffled": False,
               "roundtrip_restores_original_order": True}
        res["unsorted"].append(rec)
        print(json.dumps(rec))
    tag = "pd" + pd.__version__.split(".")[0]
    out = os.path.join(ED, "results", f"x6_zerocopy_{tag}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
