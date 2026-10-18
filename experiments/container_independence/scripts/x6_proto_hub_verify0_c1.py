"""x6 verifier: C1 - what narwhals LazyFrame.collect() returns per backend, whether
``backend=`` can force the format, and whether lazy->lazy round trip via the hub keeps
a lazy, connection-bound object.

Run: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
       experiments/container_independence/scripts/x6_proto_hub_verify0_c1.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ED, "protos"))


def main():
    import dask.dataframe as dd
    import duckdb
    import narwhals as nw
    import pandas as pd
    import polars as pl

    import x6_hub as H
    from common import make_panel

    base = make_panel(4, 10, n_cols=1, n_levels=1).reset_index()
    con = duckdb.connect()
    srcs = {
        "polars_lazy": pl.from_pandas(base).lazy(),
        "dask": dd.from_pandas(base, npartitions=2),
        "duckdb": con.from_df(base),
    }
    out = {"narwhals": nw.__version__, "collect": {}}
    for name, s in srcs.items():
        lf = nw.from_native(s)
        r = {"default": type(lf.collect().to_native()).__module__}
        for be in ["polars", "pandas", "pyarrow", "dask", "duckdb"]:
            try:
                r[be] = type(lf.collect(backend=be).to_native()).__module__
            except Exception as e:  # noqa: BLE001
                r[be] = f"{type(e).__name__}: {str(e)[:80]}"
        out["collect"][name] = r
    # hub round trip (narwhals spoke): duckdb output bound to which connection?
    kw = {"key_cols": ["h0"], "time_col": "time"}
    hub = H.nw_from_native(srcs["duckdb"], **kw)
    o = H.nw_to_native(hub)
    out["duckdb_rt_type"] = type(o).__name__
    try:
        con.sql("select count(*) from o").fetchall()
        out["duckdb_rt_queryable_on_original_con"] = True
    except Exception as e:  # noqa: BLE001
        out["duckdb_rt_queryable_on_original_con"] = f"{type(e).__name__}: {str(e)[:100]}"
    out["duckdb_rt_explain_head"] = o.explain()[:200] if hasattr(o, "explain") else None
    hubd = H.nw_from_native(srcs["dask"], **kw)
    od = H.nw_to_native(hubd)
    out["dask_rt_npartitions"] = [srcs["dask"].npartitions, od.npartitions]
    print(json.dumps(out, indent=1))
    with open(os.path.join(ED, "results", "x6_proto_hub_verify0_c1.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
