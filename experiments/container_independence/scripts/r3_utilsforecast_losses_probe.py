"""r3: utilsforecast.losses (narwhals-based) -- does output stay in the input engine?

    PYTHONPATH=.:experiments/container_independence $R3_PY -W ignore \
        experiments/container_independence/scripts/r3_utilsforecast_losses_probe.py
"""
import json
import warnings

warnings.filterwarnings("ignore")
import dask.dataframe as dd  # noqa: E402
import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from utilsforecast.losses import mae, mase  # noqa: E402

from common import make_panel_fast  # noqa: E402

f = (make_panel_fast(4, 12).reset_index()
     .rename(columns={"h0": "unique_id", "time": "ds", "c0": "y"}))
f["m"] = f["y"] + 1.0
plf = pl.from_pandas(f)
con = duckdb.connect()
inputs = {"pandas": f, "polars": plf, "polars_lazy": plf.lazy(),
          "pyarrow": plf.to_arrow(), "dask": dd.from_pandas(f, npartitions=2),
          "duckdb": con.from_arrow(plf.to_arrow())}
out = {}
for name, df in inputs.items():
    try:
        r = mae(df, models=["m"])
        val = None
        try:
            import narwhals as nw
            rn = nw.from_native(r)
            if hasattr(rn, "collect"):
                rn = rn.collect()
            val = float(np.mean(rn["m"].to_numpy()))
        except Exception as e:  # noqa: BLE001
            val = f"value-read error {type(e).__name__}"
        out[f"mae[{name}]"] = f"{type(r).__module__}.{type(r).__name__} (mean={val})"
    except Exception as e:  # noqa: BLE001
        out[f"mae[{name}]"] = f"ERROR {type(e).__name__}: {str(e)[:120]}"
for name in ["pandas", "polars", "polars_lazy"]:
    df = inputs[name]
    try:
        r = mase(df, models=["m"], seasonality=1, train_df=df)
        out[f"mase[{name}]"] = f"{type(r).__module__}.{type(r).__name__}"
    except Exception as e:  # noqa: BLE001
        out[f"mase[{name}]"] = f"ERROR {type(e).__name__}: {str(e)[:120]}"
print(json.dumps(out, indent=1))
