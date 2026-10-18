"""a3: can narwhals express the TimeIndex ops portably across backends?

For each backend (pandas, polars eager, polars lazy, pyarrow, duckdb, dask),
evaluates via narwhals: dt.offset_by("{k}mo"/"{k}d"/"{k}h"), month-end
(via offset + truncate tricks), business days (not in narwhals API), and
compares against pandas offsets ground truth.  Records support / mismatch.

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/a3_narwhals_time_probe.py
"""

import json
import warnings
from datetime import datetime

import narwhals as nw
import numpy as np
import pandas as pd

from common import env_info

warnings.filterwarnings("ignore")

H = 6
CUT_ME = pd.date_range("2000-01-31", periods=24, freq="ME")
CUT_MS = pd.date_range("2000-01-01", periods=24, freq="MS")
CUT_H = pd.date_range("2000-01-01", periods=24, freq="h")


def native(backend, cut):
    import polars as pl
    import pyarrow as pa

    data = {"c": np.asarray(cut, dtype="M8[us]"), "k": np.ones(len(cut), dtype="i8")}
    if backend == "pandas":
        return pd.DataFrame(data)
    if backend == "polars":
        return pl.DataFrame(data)
    if backend == "polars-lazy":
        return pl.DataFrame(data).lazy()
    if backend == "pyarrow":
        return pa.table(data)
    if backend == "duckdb":
        import duckdb

        tbl = pa.table(data)  # noqa: F841
        return duckdb.sql("select * from tbl")
    if backend == "dask":
        import dask.dataframe as dd

        return dd.from_pandas(pd.DataFrame(data), npartitions=2)
    raise ValueError(backend)


def to_np(df):
    nat = nw.to_native(df)
    if hasattr(nat, "collect"):
        nat = nat.collect()
    if hasattr(nat, "compute"):
        nat = nat.compute()
    if hasattr(nat, "arrow"):  # duckdb relation
        nat = nat.arrow()
        if hasattr(nat, "read_all"):
            nat = nat.read_all()
    if hasattr(nat, "to_pandas"):
        nat = nat.to_pandas()
    return np.asarray(nat["out"], dtype="M8[ns]")


RULES = {
    "MS +k mo: offset_by": (CUT_MS, pd.offsets.MonthBegin,
                            lambda k: nw.col("c").dt.offset_by(f"{k}mo")),
    "ME +k mo: offset_by only": (CUT_ME, pd.offsets.MonthEnd,
                                 lambda k: nw.col("c").dt.offset_by(f"{k}mo")),
    "ME +k mo: truncate(1mo)+(k+1)mo-1d": (
        CUT_ME, pd.offsets.MonthEnd,
        lambda k: nw.col("c").dt.truncate("1mo").dt.offset_by(f"{k + 1}mo")
        .dt.offset_by("-1d")),
    "h +k h: offset_by": (CUT_H, lambda k: pd.Timedelta(hours=k),
                          lambda k: nw.col("c").dt.offset_by(f"{k}h")),
}


def main():
    res = {"env": env_info(), "narwhals_has": {}, "rows": {}}
    ns = nw.col("c").dt
    for m in ["offset_by", "truncate", "month_end", "add_business_days",
              "is_business_day", "replace_time_zone", "convert_time_zone"]:
        res["narwhals_has"][m] = hasattr(ns, m)
    print("narwhals Expr.dt has:", res["narwhals_has"])
    for backend in ["pandas", "polars", "polars-lazy", "pyarrow", "duckdb", "dask"]:
        row = {}
        for rname, (cut, off, mk) in RULES.items():
            mism, err = 0, None
            try:
                for k in range(1, H + 1):
                    df = nw.from_native(native(backend, cut))
                    got = to_np(df.select(mk(k).alias("out")))
                    truth = np.asarray(cut + (off(k) if callable(off) else off),
                                       dtype="M8[ns]")
                    mism += int((got != truth).sum())
                row[rname] = mism
            except Exception as e:  # noqa: BLE001
                row[rname] = f"ERR {type(e).__name__}: {str(e)[:90]}"
        res["rows"][backend] = row
        print(f"{backend:12s} {row}")
    with open("experiments/container_independence/results/a3_narwhals_time_probe.json",
              "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
