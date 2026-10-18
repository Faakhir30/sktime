"""r4: Arrow PyCapsule Interface adoption + zero-copy checks across libraries.

Questions:
  1. Which objects expose __arrow_c_stream__ / __arrow_c_array__ / __arrow_c_schema__
     (and the older __dataframe__ interchange protocol, and the abandoned
     __dataframe_consortium_standard__) in the installed versions?
  2. Is the hand-off zero-copy? We check (a) buffer addresses where observable and
     (b) time + peak RSS scaling with data size (copy => grows linearly; zero-copy
     => ~constant).

Run (from WT root):
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/r4_arrow_capsule.py --grid dev
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

from common import bench, env_info, peak_rss_delta_mb, rss_mb

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

PROTOS = [
    "__arrow_c_stream__",
    "__arrow_c_array__",
    "__arrow_c_schema__",
    "__dataframe__",
    "__dataframe_consortium_standard__",
    "__column_consortium_standard__",
]


def objects():
    import polars as pl
    import pyarrow as pa

    out = {}
    pdf = pd.DataFrame({"a": np.arange(5.0), "b": list("abcde")})
    out["pandas.DataFrame"] = pdf
    out["pandas.Series"] = pdf["a"]
    out["pandas.DataFrame[MultiIndex]"] = pdf.set_index(["b", pdf.index])
    pldf = pl.DataFrame({"a": np.arange(5.0)})
    out["polars.DataFrame"] = pldf
    out["polars.LazyFrame"] = pldf.lazy()
    out["polars.Series"] = pldf["a"]
    tbl = pa.table({"a": np.arange(5.0)})
    out["pyarrow.Table"] = tbl
    out["pyarrow.RecordBatch"] = tbl.to_batches()[0]
    out["pyarrow.ChunkedArray"] = tbl["a"]
    out["pyarrow.Array"] = tbl["a"].chunk(0)
    out["pyarrow.RecordBatchReader"] = tbl.to_reader()
    try:
        import duckdb

        con = duckdb.connect()
        out["duckdb.DuckDBPyRelation"] = con.sql("select 1.0 as a")
    except Exception as e:  # noqa: BLE001
        print("duckdb unavailable", e)
    try:
        import narwhals as nw

        out["narwhals.DataFrame(polars)"] = nw.from_native(pldf)
        out["narwhals.DataFrame(pandas)"] = nw.from_native(pdf)
        out["narwhals.LazyFrame(polars)"] = nw.from_native(pldf.lazy())
        out["narwhals.Series(polars)"] = nw.from_native(pldf["a"], series_only=True)
    except Exception as e:  # noqa: BLE001
        print("narwhals unavailable", e)
    try:
        import dask.dataframe as dd

        out["dask.DataFrame"] = dd.from_pandas(pdf, npartitions=1)
    except Exception as e:  # noqa: BLE001
        print("dask unavailable", e)
    try:
        import ibis

        out["ibis.Table(memtable)"] = ibis.memtable(pdf)
    except Exception as e:  # noqa: BLE001
        print("ibis unavailable", type(e).__name__)
    return out


def protocol_matrix():
    rows = {}
    for name, obj in objects().items():
        rows[name] = {p: hasattr(obj, p) for p in PROTOS}
    return rows


def zero_copy_checks(n_rows_list):
    import polars as pl
    import pyarrow as pa

    res = {"address_checks": {}, "scaling": []}

    # --- address checks (n=1e6 float64) ---
    n = 1_000_000
    arr = np.arange(n, dtype="float64")
    pldf = pl.DataFrame({"a": arr})
    # polars -> pyarrow through the capsule
    tbl = pa.table(pldf)  # uses __arrow_c_stream__
    pl_addr = pldf["a"].to_arrow().buffers()[1].address
    pa_addr = tbl["a"].chunk(0).buffers()[1].address
    res["address_checks"]["polars->pyarrow (pa.table(pl_df))"] = pl_addr == pa_addr
    # pyarrow -> polars
    pl2 = pl.DataFrame(tbl)
    res["address_checks"]["pyarrow->polars (pl.DataFrame(tbl))"] = (
        pl2["a"].to_arrow().buffers()[1].address == pa_addr
    )
    # pandas(numpy) -> pyarrow through the capsule
    pdf = pd.DataFrame({"a": arr})
    tbl2 = pa.table(pdf) if hasattr(pdf, "__arrow_c_stream__") else None
    if tbl2 is not None:
        res["address_checks"]["pandas(numpy)->pyarrow (pa.table(pd_df))"] = (
            tbl2["a"].chunk(0).buffers()[1].address
            == pdf["a"].to_numpy().__array_interface__["data"][0]
        )
    # pyarrow -> pandas (numpy backed) via to_pandas: float64 no nulls can be zero-copy
    # for a single column only with split_blocks/self_destruct; default copies.
    pdf2 = tbl.to_pandas()
    res["address_checks"]["pyarrow->pandas (to_pandas default)"] = (
        pdf2["a"].to_numpy().__array_interface__["data"][0] == pa_addr
    )
    pdf3 = tbl.to_pandas(types_mapper=pd.ArrowDtype)
    res["address_checks"]["pyarrow->pandas (ArrowDtype)"] = (
        pdf3["a"].array._pa_array.chunk(0).buffers()[1].address == pa_addr
    )
    # pandas 3: DataFrame.from_arrow
    if hasattr(pd.DataFrame, "from_arrow"):
        try:
            pdf4 = pd.DataFrame.from_arrow(pldf)
            res["address_checks"]["polars->pandas (pd.DataFrame.from_arrow)"] = (
                pdf4["a"].to_numpy().__array_interface__["data"][0] == pl_addr
            )
            res["pandas_from_arrow_dtype"] = str(pdf4["a"].dtype)
        except Exception as e:  # noqa: BLE001
            res["address_checks"]["polars->pandas (pd.DataFrame.from_arrow)"] = repr(e)
    # polars -> numpy for float no-null single column
    res["address_checks"]["polars Series -> numpy (to_numpy)"] = (
        pldf["a"].to_numpy().__array_interface__["data"][0] == pl_addr
    )

    # MultiIndex frame through the capsule: what happens to the index?
    mi = pd.DataFrame(
        {"y": np.arange(6.0)},
        index=pd.MultiIndex.from_product(
            [["s0", "s1"], pd.date_range("2000", periods=3)], names=["h0", "time"]
        ),
    )
    if hasattr(mi, "__arrow_c_stream__"):
        t_mi = pa.table(mi)
        res["multiindex_capsule_export"] = {
            "columns": t_mi.column_names,
            "has_pandas_metadata": b"pandas" in (t_mi.schema.metadata or {}),
        }
        back = t_mi.to_pandas()
        res["multiindex_capsule_roundtrip_via_pyarrow_to_pandas"] = {
            "index_type": type(back.index).__name__,
            "index_names": list(back.index.names),
        }
        pl_mi = pl.DataFrame(mi)
        res["multiindex_capsule_into_polars"] = {"columns": pl_mi.columns}
    # pandas multi-column frame built from a 2D array: is the column hand-off
    # zero-copy? (2D blocks are stored transposed -> strided columns)
    p2 = pd.DataFrame(np.ones((1000, 3)), columns=list("abc"))
    t2 = pa.table(p2)
    res["address_checks"]["pandas(2D block)->pyarrow col a"] = (
        t2["a"].chunk(0).buffers()[1].address
        == p2["a"].to_numpy().__array_interface__["data"][0]
    )
    p3 = pd.DataFrame({c: np.ones(1000) for c in "abc"})
    t3 = pa.table(p3)
    res["address_checks"]["pandas(dict of 1D)->pyarrow col a"] = (
        t3["a"].chunk(0).buffers()[1].address
        == p3["a"].to_numpy().__array_interface__["data"][0]
    )

    # --- scaling: time + peak RSS of hand-offs vs size ---
    try:
        import duckdb
    except Exception:  # noqa: BLE001
        duckdb = None
    for n in n_rows_list:
        arr = np.random.default_rng(0).normal(size=(n, 4))
        # dict-of-1D construction: each column contiguous (best case for pandas)
        pdf = pd.DataFrame({c: arr[:, i].copy() for i, c in enumerate("abcd")})
        pldf = pl.DataFrame(arr, schema=list("abcd"))
        tbl = pa.table(pldf)
        row = {"n_rows": n, "data_mb": arr.nbytes / 2**20}
        cases = {
            "polars->pyarrow capsule": lambda: pa.table(pldf),
            "pyarrow->polars capsule": lambda: pl.DataFrame(tbl),
            "pandas->pyarrow (pa.table)": lambda: pa.table(pdf),
            "pandas->polars (pl.from_pandas)": lambda: pl.from_pandas(pdf),
            "polars->pandas (to_pandas)": lambda: pldf.to_pandas(),
            "pyarrow->pandas (to_pandas)": lambda: tbl.to_pandas(),
        }
        if duckdb is not None:
            con = duckdb.connect()
            con.register("pldf_view", pldf)
            cases["polars->duckdb scan sum(a)"] = lambda: con.sql(
                "select sum(a) from pldf_view"
            ).fetchall()
            cases["polars->duckdb ->arrow (select *)"] = lambda: con.sql(
                "select * from pldf_view"
            ).arrow()
        for k, fn in cases.items():
            t = bench(fn, repeat=5)
            a0 = pa.total_allocated_bytes()
            r = fn()
            a1 = pa.total_allocated_bytes()
            row[k] = {
                "sec": t,
                "new_arrow_alloc_mb": round((a1 - a0) / 2**20, 2),
            }
            del r
        res["scaling"].append(row)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sizes = [100_000, 1_000_000, 3_000_000] if args.grid == "dev" else [
        1_000_000, 10_000_000, 40_000_000
    ]
    out = {"env": env_info(), "python": sys.version.split()[0], "rss0": rss_mb()}
    out["protocols"] = protocol_matrix()
    out["zero_copy"] = zero_copy_checks(sizes)
    out["rss_end"] = rss_mb()
    tag = args.tag or f"pd{pd.__version__.split('.')[0]}"
    path = os.path.join(RES, f"r4_arrow_capsule_{tag}_{args.grid}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    print("wrote", path)


if __name__ == "__main__":
    main()
