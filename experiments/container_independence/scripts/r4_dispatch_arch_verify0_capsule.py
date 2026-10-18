"""Verifier: PyCapsule export matrix + zero-copy (buffer address) + MI roundtrip, via a different path:
consume capsules with polars (pl.from_arrow / pl.DataFrame(capsule obj)) and duckdb, not only pa.table."""
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, polars as pl, pyarrow as pa, narwhals as nw, duckdb, dask.dataframe as dd
res = {"pandas": pd.__version__, "polars": pl.__version__, "narwhals": nw.__version__}
pdf = pd.DataFrame({"y": np.arange(6.0)}, index=pd.MultiIndex.from_product([["a","b"], pd.date_range("2020", periods=3)], names=["h0","time"]))
objs = {"pd.DataFrame": pdf, "pd.Series": pdf["y"], "pl.DataFrame": pl.DataFrame({"y": [1.0]}), "pl.LazyFrame": pl.DataFrame({"y": [1.0]}).lazy(),
        "nw.DataFrame(pl)": nw.from_native(pl.DataFrame({"y": [1.0]})), "nw.LazyFrame(pl)": nw.from_native(pl.DataFrame({"y": [1.0]}).lazy()),
        "duckdb.rel": duckdb.sql("select 1 as y"), "dask.DataFrame": dd.from_pandas(pdf.reset_index(), npartitions=1)}
res["has___arrow_c_stream__"] = {k: hasattr(v, "__arrow_c_stream__") for k, v in objs.items()}
res["has___dataframe_consortium_standard__(pd.DataFrame)"] = hasattr(pdf, "__dataframe_consortium_standard__")
# MI via duckdb consumer (different consumer than pa.table)
t = duckdb.sql("select * from pdf").arrow() if hasattr(duckdb.sql("select * from pdf"), "arrow") else None
res["duckdb_sees_columns_of_MI_df"] = duckdb.sql("select * from pdf").columns
res["pl.from_pandas(MI df) columns"] = pl.from_pandas(pdf).columns
res["pl.from_pandas(MI df, include_index=True) columns"] = pl.from_pandas(pdf, include_index=True).columns
res["pl.DataFrame(MI df) columns"] = pl.DataFrame(pdf).columns
rt = pa.table(pdf).to_pandas(); res["pa roundtrip index"] = [type(rt.index).__name__, list(rt.index.names)]
# polars->arrow roundtrip via pl.DataFrame(arrow-capsule-holder): metadata survive?
rt2 = pa.table(pl.from_arrow(pa.table(pdf))).to_pandas(); res["via polars roundtrip index"] = [type(rt2.index).__name__, list(rt2.index.names)]
# zero-copy: polars -> pyarrow buffer address
s = pl.DataFrame({"x": np.random.rand(1_000_000)})
a1 = s["x"].to_arrow().buffers()[1].address; a2 = pa.table(s)["x"].chunks[0].buffers()[1].address
res["pl->pa same buffer"] = a1 == a2
tb = pa.table({"x": np.random.rand(1_000_000)}); p2 = pl.DataFrame(tb)
res["pa->pl same buffer"] = tb["x"].chunks[0].buffers()[1].address == p2["x"].to_arrow().buffers()[1].address
print(json.dumps(res, indent=1, default=str))
