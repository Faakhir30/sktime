"""a3 experiment (b): NaiveForecaster on polars time series inputs.

Fits NaiveForecaster(strategy="last") on polars inputs with various time
column encodings and inspects the predict output: container type, time dtype,
predicted time stamps, and whether they match the expected next periods.

Output: results/a3_naive_polars_<pdmajor>.json + printed table.

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/a3_naive_polars.py
"""

import argparse
import json
import os
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd
import polars as pl

from common import env_info

warnings.filterwarnings("ignore")
PD3 = int(pd.__version__.split(".")[0]) >= 3
T = 24
FH = [1, 2, 3]


def _series(time_col, name="__index__time"):
    n = len(time_col)
    return pl.DataFrame({name: time_col, "y": np.arange(n, dtype="float64")})


def _panel(time_col, keys=("a", "b"), hier=False):
    n = len(time_col)
    parts = []
    for i, k in enumerate(keys):
        d = {"__index__h0": [k] * n}
        if hier:
            d = {"__index__h0": ["x"] * n, "__index__h1": [k] * n}
        d["__index__time"] = time_col
        d["y"] = np.arange(n, dtype="float64") + 100 * i
        parts.append(pl.DataFrame(d))
    return pl.concat(parts)


def ms_dt():
    return pl.datetime_range(
        datetime(2000, 1, 1), datetime(2001, 12, 1), "1mo", eager=True
    )


def cases():
    from sktime.datatypes import convert_to

    ms = ms_dt()
    me = ms.dt.month_end()
    ms_date = pl.date_range(date(2000, 1, 1), date(2001, 12, 1), "1mo", eager=True)
    pidx = pd.period_range("2000-01", periods=T, freq="M")
    from_period = convert_to(
        pd.DataFrame({"y": np.arange(T, dtype="float64")}, index=pidx), "pl.DataFrame"
    )
    irregular = ms.filter(pl.Series(np.arange(T) != 5))
    tz_h = pl.datetime_range(
        datetime(2000, 3, 25, 12), datetime(2000, 3, 26, 11), "1h",
        time_zone="Europe/Berlin", eager=True,
    )
    bdays = pl.Series(pd.bdate_range("2000-01-03", periods=T)).cast(pl.Datetime("us"))
    out = {
        "series MS Datetime(us)": (_series(ms), "MS"),
        "series ME Datetime(us)": (_series(me), "ME"),
        "series MS pl.Date": (_series(ms_date), "MS"),
        "series from PeriodIndex(M) via convert_to": (from_period, "ME"),
        "series irregular (1 month missing)": (_series(irregular), None),
        "series tz Europe/Berlin hourly over DST": (_series(tz_h), "h"),
        "series business days": (_series(bdays), "B"),
        "series Int64 time col": (_series(pl.Series(np.arange(10, 10 + T))), "int"),
        "series no __index__ col": (pl.DataFrame({"y": np.arange(T, dtype="f8")}), "int"),
        "series natural layout (time col w/o __index__)": (
            pl.DataFrame({"time": ms, "y": np.arange(T, dtype="f8")}), None),
        "panel MS Datetime(us)": (_panel(ms), "MS"),
        "panel ME Datetime(us)": (_panel(me), "ME"),
        "panel MS pl.Date": (_panel(ms_date), "MS"),
        "hier MS Datetime(us)": (_panel(ms, keys=("p", "q"), hier=True), "MS"),
        "panel irregular (same month missing)": (_panel(irregular), None),
        "lazy series MS": (_series(ms).lazy(), "MS"),
        # pandas references, to attribute behaviour to polars vs sktime core
        "PANDAS ref: series MS, index name 'time', col 'y'": (
            pd.DataFrame({"y": np.arange(T, dtype="f8")},
                         index=pd.date_range("2000-01-01", periods=T, freq="MS",
                                             name="time")), "MS"),
        "PANDAS ref: series irregular (freq=None)": (
            pd.DataFrame({"y": np.arange(T - 1, dtype="f8")},
                         index=pd.DatetimeIndex(irregular.to_pandas(), name="time")),
            None),
    }
    return out


def expected_next(last, kind, k):
    if kind == "MS":
        return [pd.Timestamp(last) + pd.offsets.MonthBegin(i) for i in k]
    if kind == "ME":
        return [pd.Timestamp(last) + pd.offsets.MonthEnd(i) for i in k]
    if kind == "h":
        return [pd.Timestamp(last) + pd.Timedelta(hours=i) for i in k]
    if kind == "B":
        return [pd.Timestamp(last) + pd.offsets.BDay(i) for i in k]
    if kind == "int":
        return [last + i for i in k]
    return None


def summarize(y, y_pred, kind):
    rec = {"pred_type": type(y_pred).__name__}
    if isinstance(y_pred, pl.DataFrame):
        rec["pred_schema"] = {k: str(v) for k, v in y_pred.schema.items()}
        tcols = [c for c in y_pred.columns if c.endswith("time") or c == "__index__0"]
        tcol = tcols[-1] if tcols else None
        times = y_pred[tcol].to_list() if tcol else list(range(len(y_pred)))
        in_tcols = [c for c in y.columns if c.startswith("__index__")]
        last = y[in_tcols[-1]].max() if in_tcols else len(y) - 1
    else:
        rec["pred_index_type"] = type(y_pred.index).__name__
        rec["pred_schema"] = {
            "index_name": str(y_pred.index.name),
            "index_dtype": str(y_pred.index.dtype),
            "cols": str(
                list(y_pred.columns)
                if isinstance(y_pred, pd.DataFrame)
                else [y_pred.name]
            ),
        }
        times = list(y_pred.index)
        last = y.index.max()
    rec["pred_times"] = [str(t) for t in times]
    if kind is not None and last is not None:
        exp = expected_next(last, kind, FH)
        got = times[: len(FH)]
        try:
            ok = [pd.Timestamp(g) == pd.Timestamp(e) for g, e in zip(got, exp)] \
                if kind != "int" else [g == e for g, e in zip(got, exp)]
        except Exception as e:  # noqa: BLE001
            ok = f"cmp error {e}"
        rec["expected_first_instance"] = [str(e) for e in exp]
        rec["matches_expected"] = ok if isinstance(ok, str) else all(ok)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])  # unused
    ap.parse_args()
    from sktime.forecasting.naive import NaiveForecaster

    out = {"env": env_info(), "records": {}}
    for name, (y, kind) in cases().items():
        if isinstance(y, pd.DataFrame):
            rec = {"input_schema": {"index": str(y.index.dtype),
                                    "index_name": y.index.name,
                                    "cols": list(y.columns)}}
        else:
            rec = {"input_schema": {k: str(v) for k, v in y.collect_schema().items()}}
        try:
            f = NaiveForecaster(strategy="last")
            f.fit(y, fh=FH)
            rec["cutoff"] = repr(f.cutoff)
            rec["inner_mtype_seen"] = f._y_metadata["mtype"]
            y_pred = f.predict()
            rec.update(summarize(y, y_pred, kind))
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["records"][name] = rec
        status = (
            "ERROR " + rec["error"][:110] if "error" in rec
            else f"out={rec['pred_type']} times={rec['pred_times'][:3]} "
            f"ok={rec.get('matches_expected')} "
            f"schema={rec.get('pred_schema')}"
        )
        print(f"{name:48s} {status}")
    tag = "pd3" if PD3 else "pd2"
    path = f"experiments/container_independence/results/a3_naive_polars_{tag}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
