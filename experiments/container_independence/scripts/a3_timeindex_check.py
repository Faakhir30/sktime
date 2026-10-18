"""a3: correctness check of the TimeIndex prototype (protos/a3_timeindex.py).

1. add_steps for pandas / polars / numpy backends vs sktime
   ForecastingHorizon.to_absolute ground truth (NC cutoffs x H steps per freq).
2. steps_between(cutoff, add_steps(cutoff, k)) == k round trip (polars, numpy).
3. infer_step (polars) vs pd.infer_freq: semantic agreement (same generated
   continuation), incl. irregular / tz / int cases.

Output: results/a3_timeindex_check_<pdmajor>.json + printed table.

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/a3_timeindex_check.py
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
sys.path.insert(0, os.path.dirname(__file__))

from a3_offset_equiv import FREQS, H, NC, ground_truth  # noqa: E402
from a3_timeindex import NumpyTimeOps, PandasTimeOps, PolarsTimeOps, Step  # noqa
from common import env_info  # noqa: E402

warnings.filterwarnings("ignore")
PD3 = int(pd.__version__.split(".")[0]) >= 3


def pl_times(cut_ts, tz):
    if tz:
        return (
            pl.Series("c", cut_ts.tz_convert("UTC").tz_localize(None).values)
            .cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
            .dt.convert_time_zone(tz)
        )
    return pl.Series("c", np.asarray(cut_ts, dtype="M8[us]"))


def strip(arr_or_series, tz):
    if isinstance(arr_or_series, pl.Series):
        s = arr_or_series
        if tz:
            s = s.dt.replace_time_zone(None)
        return s.cast(pl.Datetime("ns")).to_numpy()
    a = np.asarray(arr_or_series)
    if tz and a.dtype == object:
        return np.asarray(pd.DatetimeIndex(a.ravel()).tz_localize(None),
                          dtype="M8[ns]").reshape(a.shape)
    return a


def check_add_steps():
    out = {}
    k = np.arange(1, H + 1)
    for name, (kind, freq, start, tz) in FREQS.items():
        row = {}
        pfreq = freq if kind == "dt" else pd.PeriodDtype(freq).freq
        step = Step.from_pandas(pfreq, tz=tz)
        row["step"] = repr(step)
        if step is None:
            out[name] = row
            continue
        cut_ts, sk, _ = ground_truth(kind, freq, start, tz)
        if kind == "period" and step.kind == "month" and step.anchor == "end":
            # periods are compared on START timestamps -> use start anchor
            step = Step("month", n=step.n, anchor="start")
            row["step(period->start)"] = repr(step)
        if kind == "period" and step.kind == "fixed" and step.week_anchor is not None:
            pass
        # pandas backend
        try:
            got = PandasTimeOps.add_steps(cut_ts, k, step)
            row["pandas"] = int((strip(got, tz).astype("M8[ns]") != sk).sum())
        except Exception as e:  # noqa: BLE001
            row["pandas"] = f"ERR {type(e).__name__}: {str(e)[:60]}"
        # polars backend
        try:
            df = pl_times(cut_ts, tz).to_frame().join(
                pl.DataFrame({"k": k}), how="cross"
            )
            res = df.select(
                PolarsTimeOps.add_steps_expr(pl.col("c"), pl.col("k"), step).alias("t")
            )["t"]
            row["polars"] = int((strip(res, tz).reshape(NC, H) != sk).sum())
            # steps_between round trip
            df2 = df.with_columns(t=res).with_columns(
                kb=PolarsTimeOps.steps_between_expr(pl.col("c"), pl.col("t"), step)
            )
            row["polars steps_between rt"] = int((df2["kb"] != df2["k"]).sum())
        except Exception as e:  # noqa: BLE001
            row["polars"] = f"ERR {type(e).__name__}: {str(e)[:60]}"
        # numpy backend
        try:
            got = NumpyTimeOps.add_steps(np.asarray(cut_ts, dtype="M8[ns]"), k, step)
            row["numpy"] = int((got.astype("M8[ns]") != sk).sum())
            kb = NumpyTimeOps.steps_between(
                np.repeat(np.asarray(cut_ts, dtype="M8[ns]"), H), got.ravel(), step
            )
            row["numpy steps_between rt"] = int((kb != np.tile(k, NC)).sum())
        except NotImplementedError as e:
            row["numpy"] = f"NA: {e}"
        except Exception as e:  # noqa: BLE001
            row["numpy"] = f"ERR {type(e).__name__}: {str(e)[:60]}"
        out[name] = row
        print(f"{name:20s} {row}")
    # custom business day with holidays + integer index
    hol = ["2000-01-17", "2000-02-21", "2000-05-29", "2000-07-04"]
    cbd = pd.offsets.CustomBusinessDay(holidays=hol)
    cut = pd.date_range("2000-01-03", periods=NC, freq=cbd)
    from sktime.forecasting.base import ForecastingHorizon

    sk = np.stack([
        np.asarray(ForecastingHorizon(list(k)).to_absolute(
            pd.DatetimeIndex(cut[[i]], freq=cbd)).to_pandas(), dtype="M8[ns]")
        for i in range(NC)
    ])
    dr = np.stack([
        np.asarray(pd.date_range(cut[i], periods=H + 1, freq=cbd)[1:], dtype="M8[ns]")
        for i in range(NC)
    ])
    step = Step.from_pandas(cbd)
    df = pl_times(cut, None).to_frame().join(pl.DataFrame({"k": k}), how="cross")
    res = df.select(PolarsTimeOps.add_steps_expr(pl.col("c"), pl.col("k"), step))
    pl_res = strip(res.to_series(), None).reshape(NC, H)
    np_res = NumpyTimeOps.add_steps(np.asarray(cut, dtype="M8[ns]"), k, step)
    row = {"step": repr(step),
           "sktime_vs_date_range(holiday-aware truth)": int((sk != dr).sum()),
           "polars_vs_sktime": int((pl_res != sk).sum()),
           "polars_vs_date_range": int((pl_res != dr).sum()),
           "numpy_vs_date_range": int((np_res != dr).sum()),
           "sktime fh.freq string": ForecastingHorizon(
               list(k)).to_absolute(pd.DatetimeIndex(cut[[0]], freq=cbd)).freq}
    out["C (holidays)"] = row
    print(f"{'C (holidays)':20s} {row}")
    return out


def infer_cases():
    rng = np.random.default_rng(0)
    n = 60
    c = {}
    for f in ["D", "h", "15min", "B", "W-SUN", "W-WED", "MS", "ME", "QS", "QE",
              "YS", "YE", "BMS", "BME", "SME", "2ME", "3D"]:
        start = {"W-SUN": "2000-01-02", "W-WED": "2000-01-05", "B": "2000-01-03"}.get(
            f, "2000-01-01")
        c[f] = pd.date_range(start, periods=n, freq=f)
    c["D tz=Berlin (DST)"] = pd.date_range("2000-03-01 12:00", periods=n, freq="D",
                                            tz="Europe/Berlin")
    c["h tz=Berlin (DST)"] = pd.date_range("2000-03-25", periods=n, freq="h",
                                            tz="Europe/Berlin")
    full = pd.date_range("2000-01-01", periods=2 * n, freq="D")
    c["irregular"] = full[np.sort(rng.choice(2 * n, n, replace=False))]
    c["D with one gap"] = full[:n].delete(10)
    c["MS with one gap"] = pd.date_range("2000-01-01", periods=n, freq="MS").delete(5)
    return c


def same_semantics(times_pd, step_a, step_b, tz):
    """Do two steps generate the same continuation of the series?"""
    if step_a is None or step_b is None:
        return step_a is None and step_b is None
    try:
        a = PandasTimeOps.add_steps(times_pd[[-1]], np.arange(1, 13), step_a)
        b = PandasTimeOps.add_steps(times_pd[[-1]], np.arange(1, 13), step_b)
        return bool((a == b).all())
    except Exception:  # noqa: BLE001
        return step_a == step_b


def check_infer():
    out = {}
    for name, idx in infer_cases().items():
        tz = getattr(idx, "tz", None)
        tzs = str(tz) if tz is not None else None
        try:
            pf = pd.infer_freq(idx)
        except Exception as e:  # noqa: BLE001
            pf = f"ERR {e}"
        step_pd = Step.from_pandas(pf, tz=tzs) if pf and not pf.startswith("ERR") \
            else None
        s = pl_times(idx, tzs) if tzs else pl.Series(np.asarray(idx, dtype="M8[us]"))
        step_pl = PolarsTimeOps.infer_step(s)
        agree = same_semantics(idx, step_pd, step_pl, tzs)
        out[name] = {"pd.infer_freq": pf, "Step(pd)": repr(step_pd),
                     "polars infer_step": repr(step_pl), "agree": agree}
        print(f"{name:20s} pd={pf!s:10s} polars={step_pl!r:70.70s} agree={agree}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])  # unused
    ap.parse_args()
    res = {"env": env_info()}
    print("== add_steps mismatches (out of NC*H =", NC * H, ")")
    res["add_steps"] = check_add_steps()
    print("== infer_step vs pd.infer_freq")
    res["infer"] = check_infer()
    tag = "pd3" if PD3 else "pd2"
    path = f"experiments/container_independence/results/a3_timeindex_check_{tag}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
