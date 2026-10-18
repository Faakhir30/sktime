"""x5: build the same long-format panel in every backend narwhals supports here.

Long format = instance key columns + time column + value columns (no index).
Used by the x5 probe / test / benchmark scripts only.
"""

import pandas as pd

BACKENDS = ["pandas", "polars", "polars_lazy", "pyarrow", "dask", "duckdb", "ibis"]
LAZY_BACKENDS = {"polars_lazy", "dask", "duckdb", "ibis"}


def long_from_mi(df_mi):
    """Pandas multiindex frame -> pandas long frame (index levels as columns)."""
    flat = df_mi.reset_index()
    for c in flat.columns:
        if isinstance(flat[c].dtype, pd.PeriodDtype):
            flat[c] = flat[c].dt.to_timestamp()
    return flat


def make_native(df_long, backend, npartitions=4):
    """Pandas long frame -> native frame of ``backend``."""
    if backend == "pandas":
        return df_long
    if backend == "polars":
        import polars as pl

        return pl.from_pandas(df_long)
    if backend == "polars_lazy":
        import polars as pl

        return pl.from_pandas(df_long).lazy()
    if backend == "pyarrow":
        import pyarrow as pa

        return pa.Table.from_pandas(df_long, preserve_index=False)
    if backend == "dask":
        import dask.dataframe as dd

        return dd.from_pandas(df_long, npartitions=npartitions)
    if backend == "duckdb":
        import duckdb
        import pyarrow as pa

        tbl = pa.Table.from_pandas(df_long, preserve_index=False)
        # keep a reference so the arrow table outlives the relation
        con = duckdb.connect()
        con.register("t", tbl)
        rel = con.sql("select * from t")
        _KEEPALIVE.append((con, tbl))
        return rel
    if backend == "ibis":
        import ibis

        return ibis.memtable(df_long)
    raise ValueError(backend)


_KEEPALIVE = []


def to_pandas_any(obj):
    """Materialize any native / narwhals frame to pandas (for comparisons only)."""
    import narwhals as nw

    f = nw.from_native(obj) if not isinstance(obj, (nw.DataFrame, nw.LazyFrame)) else obj
    if isinstance(f, nw.LazyFrame):
        f = f.collect()
    return f.to_pandas()
