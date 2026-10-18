"""x7 verify0: narwhals duckdb total_seconds + collect() return type (C1)."""
import datetime as dt
import duckdb, narwhals as nw, polars as pl, pandas as pd, pyarrow as pa

out = {}
t0 = dt.datetime(2000, 1, 1)
t1 = dt.datetime(2000, 1, 4, 0, 0, 5)
pdf = pd.DataFrame({"a": [t0], "b": [t1]})
rel = duckdb.sql("select TIMESTAMP '2000-01-01' as a, TIMESTAMP '2000-01-04 00:00:05' as b")
for name, native in [("pandas", pdf), ("polars", pl.from_pandas(pdf)),
                     ("polars_lazy", pl.from_pandas(pdf).lazy()), ("duckdb", rel),
                     ("pyarrow", pa.Table.from_pandas(pdf))]:
    f = nw.from_native(native)
    r = f.select((nw.col("b") - nw.col("a")).dt.total_seconds().alias("s"))
    col = r.collect() if hasattr(r, "collect") else r
    out[name] = {"total_seconds": float(col["s"].to_list()[0]),
                 "collect_type": type(col.to_native()).__module__ + "." + type(col.to_native()).__name__}
# can collect be forced?
r = nw.from_native(rel).select(nw.col("a"))
for b in ["polars", "pandas", "pyarrow"]:
    out[f"duckdb_collect_backend_{b}"] = type(r.collect(backend=b).to_native()).__name__
try:
    r.collect(backend="duckdb"); out["duckdb_collect_backend_duckdb"] = "ok"
except Exception as e:
    out["duckdb_collect_backend_duckdb"] = repr(e)[:150]
out["duckdb_to_native_lazy"] = type(r.to_native()).__name__
out["polars_lazy_collect_default"] = type(nw.from_native(pl.LazyFrame({"a": [1]})).collect().to_native()).__name__
for k, v in out.items():
    print(k, v)
