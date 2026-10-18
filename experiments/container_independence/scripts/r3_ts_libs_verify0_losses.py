"""r3_ts_libs_verify0: independent check of claim r3_ts_libs.5 (narwhals losses keep engine).

The original probe used m = y + 1 (constant error, so mae == 1.0 for every
series): a value check that cannot detect group mis-alignment. This probe:

  A. mae with a RANDOM forecast column on 6 engines; per-id values compared
     against a numpy reference (not against another engine), plus result type.
  B. mase (order-dependent: shift().over(id) without order_by) on ALL lazy
     engines (polars-lazy, dask, duckdb), not only polars-lazy.
  C. the obvious fix: the same lag expression written with
     over(id, order_by=time) through narwhals on each engine -> type + values.
  D. claim C1 side-check: what nw.LazyFrame.collect() returns per lazy engine
     by default, and whether collect(backend=...) can force a target.

    PYTHONPATH=.:experiments/container_independence $R3_PY -W ignore \
        experiments/container_independence/scripts/r3_ts_libs_verify0_losses.py
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")
import dask.dataframe as dd  # noqa: E402
import duckdb  # noqa: E402
import narwhals as nw  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from utilsforecast.losses import mae, mase  # noqa: E402

from common import make_panel_fast  # noqa: E402

rng = np.random.default_rng(1)
f = (make_panel_fast(6, 15).reset_index()
     .rename(columns={"h0": "unique_id", "time": "ds", "c0": "y"}))
f["m"] = f["y"] + rng.normal(scale=np.arange(1, len(f) + 1) / 10)  # id-dependent error
# shuffle rows so engines cannot rely on physical order
f = f.sample(frac=1.0, random_state=3).reset_index(drop=True)
plf = pl.from_pandas(f)
con = duckdb.connect()


def inputs():
    return {"pandas": f, "polars": plf, "polars_lazy": plf.lazy(),
            "pyarrow": plf.to_arrow(), "dask": dd.from_pandas(f, npartitions=3),
            "duckdb": con.from_arrow(plf.to_arrow())}


def to_pd(obj):
    o = nw.from_native(obj)
    if hasattr(o, "collect"):
        o = o.collect(backend="pandas")
    return o.to_pandas() if hasattr(o, "to_pandas") else o.to_native()


ref_mae = (f.assign(e=(f["y"] - f["m"]).abs()).groupby("unique_id")["e"].mean())
fs = f.sort_values(["unique_id", "ds"])
fs = fs.assign(lag=fs.groupby("unique_id")["y"].shift(1))
ref_lag_sum = fs.groupby("unique_id")["lag"].sum()

out = {"A_mae": {}, "B_mase": {}, "C_lag_order_by": {}, "D_collect": {}}
for name, df in inputs().items():
    try:
        r = mae(df, models=["m"])
        got = to_pd(r).set_index("unique_id")["m"].sort_index()
        ok = bool(np.allclose(got.to_numpy(), ref_mae.sort_index().to_numpy()))
        out["A_mae"][name] = {"type": f"{type(r).__module__}.{type(r).__name__}",
                              "values_match_numpy_ref": ok}
    except Exception as e:  # noqa: BLE001
        out["A_mae"][name] = f"ERROR {type(e).__name__}: {str(e)[:150]}"

for name, df in inputs().items():
    try:
        r = mase(df, models=["m"], seasonality=1, train_df=df)
        o = to_pd(r)
        out["B_mase"][name] = f"{type(r).__module__}.{type(r).__name__} rows={len(o)}"
    except Exception as e:  # noqa: BLE001
        out["B_mase"][name] = f"ERROR {type(e).__name__}: {str(e)[:150]}"

for name, df in inputs().items():
    try:
        r = (nw.from_native(df)
             .with_columns(lag=nw.col("y").shift(1).over("unique_id", order_by="ds"))
             .group_by("unique_id").agg(nw.col("lag").sum())
             .to_native())
        got = to_pd(r).set_index("unique_id")["lag"].sort_index()
        ok = bool(np.allclose(got.to_numpy(), ref_lag_sum.sort_index().to_numpy()))
        out["C_lag_order_by"][name] = {"type": f"{type(r).__module__}.{type(r).__name__}",
                                       "values_match_pandas_ref": ok}
    except Exception as e:  # noqa: BLE001
        out["C_lag_order_by"][name] = f"ERROR {type(e).__name__}: {str(e)[:150]}"

for name in ["polars_lazy", "dask", "duckdb"]:
    df = inputs()[name]
    d = {}
    lf = nw.from_native(df)
    d["nw_type"] = type(lf).__name__
    for be in [None, "polars", "pandas", "pyarrow"]:
        try:
            c = lf.collect() if be is None else lf.collect(backend=be)
            d[f"collect({be})"] = f"{type(c.to_native()).__module__}.{type(c.to_native()).__name__}"
        except Exception as e:  # noqa: BLE001
            d[f"collect({be})"] = f"ERROR {type(e).__name__}: {str(e)[:120]}"
    out["D_collect"][name] = d

out["versions"] = {"narwhals": nw.__version__, "polars": pl.__version__,
                   "dask": __import__("dask").__version__, "duckdb": duckdb.__version__}
print(json.dumps(out, indent=1))
p = os.path.join(os.path.dirname(__file__), "..", "results", "r3_ts_libs_verify0_losses.json")
with open(p, "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote", os.path.abspath(p))
