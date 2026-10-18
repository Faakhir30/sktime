"""a3: prior-art check - month-end handling of polars-native future-date helpers.

Runs in the a3 private venv (utilsforecast, functime installed; NOT the shared env):
    $A3_PY -I experiments/container_independence/scripts/a3_prior_art_monthend.py

Cutoffs: month ends Jan..Dec 2000 (incl. Feb 29); fh = 1..12 months.
Truth: pandas MonthEnd arithmetic. Counts mismatching (cutoff, k) pairs.
"""

import json
from datetime import datetime

import pandas as pd
import polars as pl

H = 12
cut = pd.date_range("2000-01-31", periods=12, freq="ME")
truth = [[c + pd.offsets.MonthEnd(k) for k in range(1, H + 1)] for c in cut]
res = {}

# functime.ranges.make_future_ranges
try:
    from functime.ranges import make_future_ranges

    cf = pl.DataFrame({"id": [f"s{i}" for i in range(len(cut))],
                       "low": [c.to_pydatetime() for c in cut]})
    fr = make_future_ranges("time", cf, fh=H, freq="1mo")
    if isinstance(fr, pl.LazyFrame):
        fr = fr.collect()
    got = fr["time"].to_list()
    mism = sum(pd.Timestamp(g) != t for row_g, row_t in zip(got, truth)
               for g, t in zip(row_g, row_t))
    res["functime make_future_ranges(freq='1mo')"] = mism
    res["functime example cutoff 2000-02-29 ->"] = [str(x) for x in got[1][:3]]
except Exception as e:  # noqa: BLE001
    res["functime"] = f"ERR {type(e).__name__}: {e}"

# utilsforecast.processing.offset_times / time_ranges with polars
try:
    from utilsforecast.processing import offset_times, time_ranges

    s = pl.Series("c", [c.to_pydatetime() for c in cut])
    mism = 0
    for k in range(1, H + 1):
        out = offset_times(s, "1mo", k).to_list()
        mism += sum(pd.Timestamp(o) != truth[i][k - 1] for i, o in enumerate(out))
    res["utilsforecast offset_times(pl, '1mo')"] = mism
    tr = time_ranges(s, "1mo", H + 1).to_list()
    tr = [tr[i * (H + 1) + 1: (i + 1) * (H + 1)] for i in range(len(cut))]
    res["utilsforecast time_ranges(pl, '1mo')"] = sum(
        pd.Timestamp(g) != t for rg, rt in zip(tr, truth) for g, t in zip(rg, rt))
    # mixed input: one month-end series + one mid-month series -> heuristic off
    s2 = pl.Series("c", [datetime(2000, 2, 29), datetime(2000, 2, 15)])
    res["utilsforecast mixed ends: offset_times(+1mo)"] = [
        str(x) for x in offset_times(s2, "1mo", 1).to_list()]
    s3 = pl.Series("c", [datetime(2000, 2, 29), datetime(2000, 3, 31)])
    res["utilsforecast all month ends: offset_times(+1mo)"] = [
        str(x) for x in offset_times(s3, "1mo", 1).to_list()]
except Exception as e:  # noqa: BLE001
    res["utilsforecast"] = f"ERR {type(e).__name__}: {e}"

import importlib.metadata as md  # noqa: E402

res["versions"] = {p: md.version(p) for p in ["functime", "utilsforecast", "polars",
                                              "pandas"]}
print(json.dumps(res, indent=1))
with open("experiments/container_independence/results/a3_prior_art_monthend.json",
          "w") as f:
    json.dump(res, f, indent=1)
