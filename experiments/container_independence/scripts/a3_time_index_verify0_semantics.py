"""a3_time_index_verify0: independent stress test of the a3 Step/TimeOps prototype.

Differences from the original a3_timeindex_check.py:
* cutoffs are RANDOM on-grid points sampled from a long range (1950..2100),
  not 150 consecutive cutoffs starting in 2000;
* the truth is pd.date_range(cutoff, periods=H+1, freq=offset)[1:] (the pandas
  offset object, incl. holidays), plus sktime ForecastingHorizon.to_absolute;
* extra conditions: non-midnight time-of-day, other time zones (New York,
  Sydney = southern-hemisphere DST, Kolkata = no DST), tz-aware month/business
  frequencies, multiples (2B, 3MS, 2W-SUN, 4h, 2QS), anchored quarters (QS-FEB),
  integer index;
* comparison on int64 nanoseconds of UTC instants (no tz stripping).

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/a3_time_index_verify0_semantics.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
from a3_timeindex import NumpyTimeOps, PolarsTimeOps, Step  # noqa: E402

warnings.filterwarnings("ignore")
PD3 = int(pd.__version__.split(".")[0]) >= 3
H = 24
NCUT = 200
rng = np.random.default_rng(12345)

# name -> (freq, anchor start, tz)
CASES = {
    # baseline (random cutoffs instead of consecutive)
    "D": ("D", "1950-01-01", None),
    "B": ("B", "1950-01-02", None),
    "W-SUN": ("W-SUN", "1950-01-01", None),
    "MS": ("MS", "1950-01-01", None),
    "ME": ("ME", "1950-01-31", None),
    "QE": ("QE", "1950-03-31", None),
    "YE": ("YE", "1950-12-31", None),
    "SME": ("SME", "1950-01-15", None),
    "BMS": ("BMS", "1950-01-02", None),
    "BME": ("BME", "1950-01-31", None),
    # multiples / anchors
    "2B": ("2B", "1950-01-02", None),
    "3MS": ("3MS", "1950-01-01", None),
    "2W-SUN": ("2W-SUN", "1950-01-01", None),
    "QS-FEB": ("QS-FEB", "1950-02-01", None),
    "2QS": ("2QS", "1950-01-01", None),
    # time of day
    "MS 09:30": ("MS", "1950-01-01 09:30", None),
    "ME 09:30": ("ME", "1950-01-31 09:30", None),
    "B 09:30": ("B", "1950-01-02 09:30", None),
    "BMS 09:30": ("BMS", "1950-01-02 09:30", None),
    "BME 09:30": ("BME", "1950-01-31 09:30", None),
    "SME 09:30": ("SME", "1950-01-15 09:30", None),
    # time zones
    "D NY": ("D", "1990-01-01 12:00", "America/New_York"),
    "D Sydney": ("D", "1990-01-01 12:00", "Australia/Sydney"),
    "D Kolkata": ("D", "1990-01-01 12:00", "Asia/Kolkata"),
    "4h NY": ("4h", "1990-01-01", "America/New_York"),
    "MS NY": ("MS", "1990-01-01", "America/New_York"),
    "ME Sydney": ("ME", "1990-01-31", "Australia/Sydney"),
    "B NY": ("B", "1990-01-01", "America/New_York"),
    "W-SUN NY": ("W-SUN", "1990-01-07", "America/New_York"),
    "SME NY": ("SME", "1990-01-15", "America/New_York"),
}


def utc_ns(x):
    """DatetimeIndex / array / polars Series -> int64 UTC nanoseconds."""
    if isinstance(x, pl.Series):
        if getattr(x.dtype, "time_zone", None):
            x = x.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        return x.cast(pl.Datetime("ns")).cast(pl.Int64).to_numpy()
    idx = pd.DatetimeIndex(np.asarray(x).ravel()) if not isinstance(
        x, pd.DatetimeIndex) else x
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.as_unit("ns").asi8


def to_polars(idx):
    if idx.tz is not None:
        s = pl.Series("c", idx.tz_convert("UTC").tz_localize(None).as_unit("us"))
        return s.dt.replace_time_zone("UTC").dt.convert_time_zone(str(idx.tz))
    return pl.Series("c", idx.as_unit("us"))


def run_case(name, freq, start, tz):
    from sktime.forecasting.base import ForecastingHorizon

    off = pd.tseries.frequencies.to_offset(freq)
    if "h" in freq:
        grid = pd.date_range(start, periods=20000, freq=off, tz=tz)
    else:
        grid = pd.date_range(start, end="2100-01-01", freq=off, tz=tz)
        if len(grid) > 20000:
            grid = grid[:20000]
    if freq in ("YE", "YS"):
        grid = pd.date_range(start, end="2200-01-01", freq=off, tz=tz)
    ncut = min(NCUT, len(grid) - 1)
    pos = np.sort(rng.choice(len(grid) - 1, ncut, replace=False))
    cuts = grid[pos]
    k = np.arange(1, H + 1)
    truth = np.stack([utc_ns(pd.date_range(c, periods=H + 1, freq=off)[1:])
                      for c in cuts])
    fh = ForecastingHorizon(list(k))
    sk = np.stack([utc_ns(fh.to_absolute_index(pd.DatetimeIndex([c], freq=off)))
                   for c in cuts])
    step = Step.from_pandas(off, tz=tz)
    row = {"freq": freq, "tz": tz, "step": None if step is None else step.kind,
           "sktime_vs_date_range": int((sk != truth).sum())}
    if step is None:
        return row
    try:
        df = to_polars(cuts).to_frame().join(pl.DataFrame({"k": k}), how="cross")
        res = df.select(PolarsTimeOps.add_steps_expr(pl.col("c"), pl.col("k"),
                                                     step).alias("t"))["t"]
        got = utc_ns(res).reshape(ncut, H)
        row["polars_vs_sktime"] = int((got != sk).sum())
        row["polars_vs_date_range"] = int((got != truth).sum())
        row["polars_out_dtype"] = str(res.dtype)
        bad = np.argwhere(got != sk)
        if len(bad):
            i, j = bad[0]
            row["example"] = (str(cuts[i]), int(k[j]),
                              str(pd.Timestamp(sk[i, j], tz="UTC")),
                              str(pd.Timestamp(got[i, j], tz="UTC")))
    except Exception as e:  # noqa: BLE001
        row["polars"] = f"ERR {type(e).__name__}: {str(e)[:100]}"
    try:
        got = NumpyTimeOps.add_steps(np.asarray(cuts.tz_localize(None) if tz else cuts,
                                                dtype="M8[ns]"), k, step)
        row["numpy_vs_sktime"] = int((got.astype("M8[ns]").astype(np.int64) != sk).sum())
    except NotImplementedError as e:
        row["numpy"] = f"NA {e}"
    except Exception as e:  # noqa: BLE001
        row["numpy"] = f"ERR {type(e).__name__}: {str(e)[:100]}"
    return row


def run_int():
    from sktime.forecasting.base import ForecastingHorizon

    out = {}
    for stepn in (1, 5):
        cuts = rng.integers(-1000, 10**6, NCUT) * stepn
        k = np.arange(1, H + 1)
        fh = ForecastingHorizon(list(k))
        sk = np.stack([np.asarray(fh.to_absolute_index(
            pd.RangeIndex(c, c + stepn, stepn) if stepn > 1 else pd.Index([c])))
            for c in cuts])
        pl_res = pl.DataFrame({"c": cuts}).join(pl.DataFrame({"k": k}), how="cross") \
            .select(PolarsTimeOps.add_steps_expr(pl.col("c"), pl.col("k"),
                                                 Step("int", n=stepn)))
        got = pl_res.to_series().to_numpy().reshape(NCUT, H)
        out[f"int step {stepn}"] = {"polars_vs_sktime": int((got != sk).sum()),
                                   "sktime_first_row": sk[0, :3].tolist(),
                                   "polars_first_row": got[0, :3].tolist()}
    return out


def main():
    res = {"pandas": pd.__version__, "polars": pl.__version__, "cases": {}}
    for name, (freq, start, tz) in CASES.items():
        r = run_case(name, freq, start, tz)
        res["cases"][name] = r
        print(f"{name:12s} {r}", flush=True)
    res["int"] = run_int()
    print(res["int"])
    tag = "pd3" if PD3 else "pd2"
    path = ("experiments/container_independence/results/"
            f"a3_time_index_verify0_semantics_{tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
