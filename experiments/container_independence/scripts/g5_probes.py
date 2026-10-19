"""g5 probes: root-cause details for the g5_api_surface matrix.

P1 where the disjoint-X ValueError is raised for Panel (stock pandas/polars) and why
   x4 native silently accepts it.
P2 x4 cutoff bridge reverse_order (direct call, Series + Panel polars) vs stock and x4fix.
P3 x6 hub pandas round trip: what happens to time-level freq / names on Panel.
P4 Series polars y-only Naive: column-name loss (stock).
"""

import json
import traceback

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sktime.datatypes import convert_to, get_cutoff
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.naive import NaiveForecaster

R = {}


def frames(e):
    out = []
    for f in traceback.extract_tb(e.__traceback__):
        if "/sktime/sktime/" in f.filename or "protos/" in f.filename:
            fn = f.filename.split("/sktime/sktime/")[-1]
            out.append(f"{fn}:{f.lineno} {f.name}")
    return out


idx = pd.date_range("2000-01-01", periods=30, freq="D", name="time")


def panel(cols, seed, shift_days=0):
    rng = np.random.default_rng(seed)
    parts = {}
    for i in range(3):
        df = pd.DataFrame(rng.normal(size=(30, len(cols))).cumsum(0), columns=cols,
                          index=idx + pd.Timedelta(days=shift_days))
        parts[f"s{i}"] = df
    return pd.concat(parts, names=["h0"])


# P1 -------------------------------------------------------------------------
y = panel(["y"], 0)
Xd = panel(["x1"], 1, shift_days=4000)
for bk in ["pandas", "polars"]:
    yy = y if bk == "pandas" else convert_to(y, "polars_panel")
    XX = Xd if bk == "pandas" else convert_to(Xd, "polars_panel")
    for appr in ["stock", "x4"]:
        f = make_reduction(LinearRegression(), window_length=3, pooling="local")
        try:
            if appr == "x4":
                from x4_instrument import patched_base

                with patched_base("native", cutoff=True):
                    f.fit(yy, X=XX, fh=[1])
            else:
                f.fit(yy, X=XX, fh=[1])
            R[f"P1_{appr}_{bk}"] = "NO ERROR"
        except Exception as e:  # noqa: BLE001
            R[f"P1_{appr}_{bk}"] = {"err": str(e)[:100], "frames": frames(e)[-6:]}

# P2 -------------------------------------------------------------------------
from x4_adapter import adapter_get_cutoff  # noqa: E402

ys = pd.DataFrame({"y": np.arange(30.0)}, index=idx)
for name, obj, mt in [("Series", ys, "pl.DataFrame"), ("Panel", y, "polars_panel")]:
    pl_obj = convert_to(obj, mt)
    stock_pd = get_cutoff(obj, return_index=True, reverse_order=True)
    stock_pl = get_cutoff(pl_obj, return_index=True, reverse_order=True)
    x4 = adapter_get_cutoff(pl_obj, return_index=True, reverse_order=True)
    R[f"P2_{name}"] = {
        "stock_pandas": [str(stock_pd[0]), str(stock_pd.freq)],
        "stock_polars": [str(stock_pl[0]), str(stock_pl.freq)],
        "x4_bridge_polars": [str(x4[0]), str(getattr(x4, "freq", None))],
        "x4_correct": bool(x4[0] == stock_pd[0]),
    }

# P3 -------------------------------------------------------------------------
from x6_hub import from_pandas, from_polars, to_pandas  # noqa: E402

for src, obj in [("pandas", y), ("polars", convert_to(y, "polars_panel"))]:
    hub = from_pandas(obj) if src == "pandas" else from_polars(obj)
    back = to_pandas(hub)
    R[f"P3_hub_roundtrip_{src}"] = {
        "index_names": list(back.index.names),
        "orig_index_names": list(y.index.names),
        "time_level_type": type(back.index.levels[-1]).__name__,
        "time_level_freq_orig": str(getattr(y.index.levels[-1], "freq", None)),
        "time_level_freq_back": str(getattr(back.index.levels[-1], "freq", None)),
        "cutoff_stock": str(get_cutoff(y, return_index=True).freq),
        "cutoff_after_hub": str(get_cutoff(back, return_index=True).freq),
        "frame_equal": back.equals(y),
    }
    try:
        NaiveForecaster().fit(back, fh=[1]).update_predict(back)
        R[f"P3_hub_roundtrip_{src}"]["update_predict"] = "ok"
    except Exception as e:  # noqa: BLE001
        R[f"P3_hub_roundtrip_{src}"]["update_predict"] = {"err": str(e)[:80],
                                                          "frames": frames(e)[-3:]}

# P4 -------------------------------------------------------------------------
for mt in ["pd.DataFrame", "pl.DataFrame"]:
    yy = convert_to(ys, mt)
    p = NaiveForecaster().fit(yy, fh=[1, 2]).predict()
    R[f"P4_naive_series_{mt}"] = {"type": type(p).__name__, "columns": list(p.columns)}
    fr = NaiveForecaster().fit(yy, fh=[1, 2])
    R[f"P4_naive_series_{mt}"]["inner_y_name"] = str(getattr(fr._y, "name", None))
    R[f"P4_naive_series_{mt}"]["converter_store_y"] = str(fr._converter_store_y)

print(json.dumps(R, indent=1, default=str))
with open("experiments/container_independence/results/g5_probes.json", "w") as fh:
    json.dump(R, fh, indent=1, default=str)
