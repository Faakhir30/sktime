"""verify0: independent probe of r1 claim 9 via actually CALLING the protocols (not hasattr).
Run with EXP_PY and EXP_PY3: PYTHONPATH=.:experiments/container_independence $PY -W default <this>"""
import warnings, json
import numpy as np, pandas as pd, polars as pl, pyarrow as pa, narwhals as nw
res = {"pandas": pd.__version__, "polars": pl.__version__}
mi = pd.MultiIndex.from_product([["a", "b"], pd.date_range("2000", periods=3)], names=["h0", "time"])
pdmi = pd.DataFrame({"c0": np.arange(6.0)}, index=mi)
def call(label, f):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = f(); out = f"ok {type(r).__name__}"
        except Exception as e:
            out = f"{type(e).__name__}: {str(e)[:120]}"
    ws = sorted({f"{x.category.__name__}: {str(x.message)[:80]}" for x in w})
    res[label] = {"result": out, "warnings": ws}
call("pd.__dataframe__()", lambda: pdmi.__dataframe__())
call("pl.__dataframe__()", lambda: pl.DataFrame({"a": [1]}).__dataframe__())
call("pa.table(pd_mi) via capsule", lambda: pa.RecordBatchReader.from_stream(pdmi).read_all())
call("pa.table(pl.LazyFrame)", lambda: pa.table(pl.LazyFrame({"a": [1]})))
call("pl.DataFrame(pa capsule of pd_mi)", lambda: pl.DataFrame(pa.RecordBatchReader.from_stream(pdmi).read_all()))
t = pa.RecordBatchReader.from_stream(pdmi).read_all()
back = t.to_pandas()
res["capsule_roundtrip_index_equal"] = bool(back.index.equals(pdmi.index)) and list(back.index.names) == ["h0", "time"] and bool(back.equals(pdmi))
res["capsule_columns"] = t.column_names
res["pl.from_pandas_default_cols"] = pl.from_pandas(pdmi).columns
res["pl.from_pandas_include_index_cols"] = pl.from_pandas(pdmi, include_index=True).columns
try:
    import dask.dataframe as dd
    ddf = dd.from_pandas(pdmi.reset_index(), npartitions=2)
    res["dask_has_capsule"] = hasattr(ddf, "__arrow_c_stream__")
    call("pa.table(dask)", lambda: pa.table(ddf))
except Exception as e:
    res["dask"] = repr(e)
try:
    import duckdb
    rel = duckdb.sql("select 1 as a")
    res["duckdb_has_capsule"] = hasattr(rel, "__arrow_c_stream__")
    call("pa.table(duckdb rel)", lambda: pa.RecordBatchReader.from_stream(rel).read_all())
except Exception as e:
    res["duckdb"] = repr(e)
res["nw_lazy_has_capsule"] = hasattr(nw.from_native(pl.LazyFrame({"a": [1]})), "__arrow_c_stream__")
res["pd_has_from_arrow"] = hasattr(pd.DataFrame, "from_arrow")
print(json.dumps(res, indent=1))
