"""r1: backend-divergence risk of hand-written per-backend code vs a normalising layer.

Reproduces skrub issue #2335 (SessionEncoder: polars branch truncates sub-second
gaps because polars ``dt.total_seconds()`` returns whole seconds while pandas'
returns fractional seconds) and checks the same primitive through narwhals.

Also probes a few semantics that matter for sktime's forecasting code paths and
that a per-backend adapter would have to reconcile by hand:
  * Duration -> seconds (the #2335 primitive)
  * mean of an all-null group, std ddof default, rolling-window min_periods default
  * group order of group_by (sorted vs first appearance vs undefined)

Run (needs skrub -> private r1 venv):
  PYTHONPATH=.:experiments/container_independence $R1_PY -W ignore \
      experiments/container_independence/scripts/r1_backend_divergence.py
"""

import datetime
import json
import os
import warnings

warnings.filterwarnings("ignore")
import narwhals.stable.v2 as nw  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

out = {"versions": {m.__name__: m.__version__ for m in (pd, pl, nw.__class__ and __import__("narwhals"), np)}}

# ---------------------------------------------------------- skrub #2335 repro
try:
    import skrub
    from skrub import SessionEncoder

    out["versions"]["skrub"] = skrub.__version__
    t0 = datetime.datetime(2024, 1, 1, 10, 0, 0)
    res = {}
    for case, (delta, gap) in {
        "gap 1800.5s vs session_gap=1800": (
            datetime.timedelta(minutes=30, milliseconds=500),
            1800,
        ),
        "gap 0.8s vs session_gap=0.5": (datetime.timedelta(milliseconds=800), 0.5),
    }.items():
        data = {"timestamp": [t0, t0 + delta]}
        res[case] = {}
        for name, df in [("pandas", pd.DataFrame(data)), ("polars", pl.DataFrame(data))]:
            o = SessionEncoder("timestamp", session_gap=gap).fit_transform(df)
            res[case][name] = [int(v) for v in list(o["timestamp_session_id"])]
    out["skrub_2335_session_ids"] = res
except Exception as e:  # noqa: BLE001
    out["skrub_2335_error"] = repr(e)

# ---------------------------------------------------------- primitive: total_seconds
t0 = datetime.datetime(2024, 1, 1)
ts = [t0, t0 + datetime.timedelta(seconds=1800, milliseconds=500)]
pd_s = pd.Series(ts)
pl_s = pl.Series(ts)
prim = {
    "pandas native diff().dt.total_seconds()": float(pd_s.diff().dt.total_seconds().iloc[1]),
    "polars native diff().dt.total_seconds()": float(pl_s.diff().dt.total_seconds()[1]),
}
try:
    prim["polars native diff().dt.total_seconds(fractional=True)"] = float(
        pl_s.diff().dt.total_seconds(fractional=True)[1]
    )
except Exception as e:  # noqa: BLE001
    prim["polars native ... fractional=True"] = repr(e)
for name, s in [("pandas", pd_s), ("polars", pl_s)]:
    ns = nw.from_native(s, series_only=True)
    prim[f"narwhals[{name}] diff().dt.total_seconds()"] = float(ns.diff().dt.total_seconds()[1])
out["total_seconds_primitive"] = prim

# ---------------------------------------------------------- other semantics
vals = [1.0, 2.0, 4.0, None, 8.0]
pdf = pd.DataFrame({"k": ["b", "a", "b", "a", "c"], "v": vals})
pldf = pl.DataFrame({"k": ["b", "a", "b", "a", "c"], "v": vals})
sem = {}
sem["std default ddof: pandas"] = float(pdf["v"].std())
sem["std default ddof: polars"] = float(pldf["v"].std())
sem["std default ddof: numpy np.nanstd"] = float(np.nanstd(np.array(vals, dtype=float)))
for name, d in [("pandas", pdf), ("polars", pldf)]:
    sem[f"std via narwhals[{name}]"] = float(nw.from_native(d)["v"].std())
sem["rolling(3).mean() pandas"] = [
    None if np.isnan(x) else round(x, 3) for x in pdf["v"].rolling(3).mean().tolist()
]
sem["rolling_mean(3) polars"] = [
    None if x is None else round(x, 3) for x in pldf["v"].rolling_mean(3).to_list()
]
sem["groupby order pandas (sort=True default)"] = pdf.groupby("k")["v"].mean().index.tolist()
sem["groupby order pandas (sort=False)"] = pdf.groupby("k", sort=False)["v"].mean().index.tolist()
sem["group_by order polars (maintain_order=False default)"] = (
    pldf.group_by("k").agg(pl.col("v").mean())["k"].to_list()
)
sem["group_by order polars (maintain_order=True)"] = (
    pldf.group_by("k", maintain_order=True).agg(pl.col("v").mean())["k"].to_list()
)
out["semantics"] = sem

print(json.dumps(out, indent=1, default=str))
os.makedirs("experiments/container_independence/results", exist_ok=True)
with open("experiments/container_independence/results/r1_backend_divergence.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
