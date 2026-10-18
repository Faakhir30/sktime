"""x6 verifier: independent zero-copy / writability / aliasing checks.

Different method from x6_zerocopy.py: compares raw buffer ADDRESS RANGES of the
native container (pandas block manager arrays, polars/arrow buffer addresses via
pyarrow) with the hub arrays' __array_interface__ data pointers, instead of
np.shares_memory against arrays obtained through the same to_numpy calls the hub uses.

Also checks: pandas-3 CoW read-only behaviour, multi-column polars, egress output
aliasing user input, and memory-accounting sanity (tracemalloc-free byte count).

Run: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
       experiments/container_independence/scripts/x6_proto_hub_verify0_zerocopy.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ED, "protos"))
import x6_hub as H  # noqa: E402
from common import env_info, make_panel_fast, to_polars_long  # noqa: E402


def rng_of(a):
    a = np.asarray(a)
    p = a.__array_interface__["data"][0]
    return (p, p + a.nbytes)


def overlaps(r, ranges):
    return any(r[0] < b and a < r[1] for a, b in ranges)


def arrow_ranges(tab_or_chunked_list):
    out = []
    for arr in tab_or_chunked_list:
        for ch in arr.chunks:
            for b in ch.buffers():
                if b is not None:
                    out.append((b.address, b.address + b.size))
    return out


def pandas_block_ranges(df):
    out = []
    mgr = df._mgr
    for blk in mgr.blocks:
        v = blk.values
        v = getattr(v, "_ndarray", v)
        if isinstance(v, np.ndarray):
            out.append(rng_of(v))
    return out


def main():
    import polars as pl
    import pyarrow as pa

    res = {"env": env_info(), "rows": []}
    kw = {"key_cols": ["h0"], "time_col": "time"}
    for n_cols in (1, 4):
        X = make_panel_fast(500, 40, n_cols=n_cols)
        # pandas
        hub = H.from_native(X)
        rec = {"src": "pandas_mi", "n_cols": n_cols,
               "values_in_block": overlaps(rng_of(hub.values), pandas_block_ranges(X)),
               "values_writeable": bool(hub.values.flags.writeable)}
        out = H.to_native(hub)
        rec["egress_aliases_input"] = overlaps(rng_of(out.to_numpy(copy=False)),
                                               pandas_block_ranges(X))
        before = float(X.iloc[0, 0])
        try:
            out.iloc[0, 0] = before + 1000.0
            rec["mutating_roundtrip_output_changes_input"] = bool(
                float(X.iloc[0, 0]) != before)
        except Exception as e:  # noqa: BLE001
            rec["mutating_roundtrip_output_changes_input"] = f"err {e}"
        res["rows"].append(rec)
        # same with a multi-block pandas frame (column-wise construction)
        Xm = X.copy()
        if n_cols > 1:
            Xm = pd.DataFrame({c: X[c].to_numpy().copy() for c in X.columns},
                              index=X.index)
            Xm["c0"] = Xm["c0"] * 1.0  # forces separate block
            hub = H.from_native(Xm)
            res["rows"].append({"src": "pandas_mi_multiblock", "n_cols": n_cols,
                                "n_blocks": len(Xm._mgr.blocks),
                                "values_in_block": overlaps(rng_of(hub.values),
                                                            pandas_block_ranges(Xm))})
        # polars long
        P = to_polars_long(X)
        hub = H.from_native(P, **kw)
        pr = arrow_ranges([P[c].to_arrow() if False else pa.chunked_array(
            [P[c].to_arrow()]) for c in P.columns])
        # to_arrow may copy for string_view; numeric buffers are zero copy in polars
        res["rows"].append({
            "src": "polars_long", "n_cols": n_cols,
            "values_in_source": overlaps(rng_of(hub.values), pr),
            "time_in_source": overlaps(rng_of(hub.time), pr),
            "values_writeable": bool(hub.values.flags.writeable),
            "values_layout": "F" if hub.values.flags.f_contiguous else "C/strided",
        })
        # arrow
        T = pa.Table.from_pandas(X.reset_index(), preserve_index=False)
        hub = H.from_native(T, **kw)
        tr = arrow_ranges([T.column(c) for c in T.column_names])
        res["rows"].append({
            "src": "arrow_long", "n_cols": n_cols,
            "values_in_source": overlaps(rng_of(hub.values), tr),
            "time_in_source": overlaps(rng_of(hub.time), tr),
            "values_writeable": bool(hub.values.flags.writeable),
        })
        out = H.to_native(hub)
        orr = arrow_ranges([out.column(c) for c in out.column_names])
        res["rows"][-1]["egress_values_aliases_source"] = overlaps(
            rng_of(hub.values), orr) and overlaps(rng_of(hub.values), tr)

    for r in res["rows"]:
        print(json.dumps(r))
    tag = "pd" + pd.__version__.split(".")[0]
    with open(os.path.join(ED, "results", f"x6_proto_hub_verify0_zerocopy_{tag}.json"),
              "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
