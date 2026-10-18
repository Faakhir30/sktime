"""a3 experiment (a): time-index round trip through sktime's polars / dask mtypes.

For each pandas time index kind, builds Series / Panel / Hierarchical data in the
pandas mtype, converts with sktime ``convert_to`` to the polars (and dask) mtype
and back, and records what is lost: index class, dtype, freq, values, names,
cutoff (get_cutoff) type/value, and the dtype of the time column in the
intermediate container.

Output: results/a3_roundtrip_<pdmajor>.json and a printed table.

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/a3_roundtrip.py
"""

import argparse
import json
import os
import traceback
import warnings

import numpy as np
import pandas as pd

from common import env_info

warnings.filterwarnings("ignore")

PD3 = int(pd.__version__.split(".")[0]) >= 3
N = 24


def time_indexes():
    """Return dict name -> pandas time index of length N."""
    out = {
        "period_M": pd.period_range("2000-01", periods=N, freq="M"),
        "period_Q": pd.period_range("2000Q1", periods=N, freq="Q"),
        "period_D": pd.period_range("2000-01-01", periods=N, freq="D"),
        "period_W": pd.period_range("2000-01-02", periods=N, freq="W-SUN"),
        "dt_D": pd.date_range("2000-01-01", periods=N, freq="D"),
        "dt_h": pd.date_range("2000-01-01", periods=N, freq="h"),
        "dt_B": pd.date_range("2000-01-03", periods=N, freq="B"),
        "dt_W-SUN": pd.date_range("2000-01-02", periods=N, freq="W-SUN"),
        "dt_MS": pd.date_range("2000-01-01", periods=N, freq="MS"),
        "dt_ME": pd.date_range("2000-01-31", periods=N, freq="ME"),
        "dt_QE": pd.date_range("2000-03-31", periods=N, freq="QE"),
        "dt_QS": pd.date_range("2000-01-01", periods=N, freq="QS"),
        "dt_YE": pd.date_range("2000-12-31", periods=N, freq="YE"),
        "dt_tz_h": pd.date_range(
            "2000-03-26 00:00", periods=N, freq="h", tz="Europe/Berlin"
        ),
        "dt_irregular": pd.DatetimeIndex(
            pd.date_range("2000-01-01", periods=N * 2, freq="D")[
                np.sort(np.random.default_rng(0).choice(N * 2, N, replace=False))
            ]
        ),
        "range0": pd.RangeIndex(N),
        "range5": pd.RangeIndex(5, 5 + N),
        "int_step10": pd.Index(np.arange(0, 10 * N, 10), dtype="int64"),
    }
    return out


def make(scitype, tix):
    rng = np.random.default_rng(0)
    if scitype == "Series":
        return pd.DataFrame({"y": rng.normal(size=len(tix))}, index=tix)
    if scitype == "Panel":
        inst = ["a", "b", "c"]
        idx = pd.MultiIndex.from_product([inst, tix], names=["h0", "time"])
    else:
        idx = pd.MultiIndex.from_tuples(
            [(a, b, t) for a, b in [("x", "p"), ("x", "q"), ("y", "r")] for t in tix],
            names=["h0", "h1", "time"],
        )
    return pd.DataFrame({"y": rng.normal(size=len(idx))}, index=idx)


PD_MTYPE = {"Series": "pd.DataFrame", "Panel": "pd-multiindex",
            "Hierarchical": "pd_multiindex_hier"}
TARGETS = {
    "Series": ["pl.DataFrame", "dask_series"],
    "Panel": ["polars_panel", "dask_panel"],
    "Hierarchical": ["polars_hierarchical", "dask_hierarchical"],
}


def time_level(df):
    if isinstance(df.index, pd.MultiIndex):
        return df.index.levels[-1], df.index.get_level_values(-1)
    return df.index, df.index


def freq_of(ix):
    f = getattr(ix, "freq", None)
    return None if f is None else str(getattr(f, "freqstr", f))


def describe_cutoff(c):
    v = c[0] if isinstance(c, pd.Index) else c
    return {
        "type": type(v).__name__,
        "value": str(v),
        "index_freq": freq_of(c) if isinstance(c, pd.Index) else None,
    }


def inter_time_dtype(obj):
    """dtype of the time column / index in the intermediate container."""
    try:
        import polars as pl

        if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
            sch = obj.collect_schema() if isinstance(obj, pl.LazyFrame) else obj.schema
            idx_cols = [c for c in sch.names() if c.startswith("__index__")]
            if not idx_cols:
                return "no __index__ column"
            return f"{idx_cols[-1]}: {sch[idx_cols[-1]]}"
    except ImportError:
        pass
    try:
        import dask.dataframe as dd

        if isinstance(obj, dd.DataFrame):
            idx_cols = [c for c in obj.columns if str(c).startswith("__index__")]
            if idx_cols:
                return f"{idx_cols[-1]}: {obj[idx_cols[-1]].dtype}"
            return f"dask index: {obj.index.dtype}"
    except ImportError:
        pass
    return type(obj).__name__


def run_case(scitype, name, tix, target):
    from sktime.datatypes import check_is_mtype, convert_to, get_cutoff
    from sktime.datatypes._utilities import get_time_index

    df = make(scitype, tix)
    src = PD_MTYPE[scitype]
    lvl0, vals0 = time_level(df)
    rec = {
        "scitype": scitype,
        "index": name,
        "target": target,
        "before": {
            "time_index_cls": type(lvl0).__name__,
            "dtype": str(vals0.dtype),
            "levels_freq": freq_of(lvl0),
            "get_time_index_freq": freq_of(get_time_index(df)),
            "cutoff": describe_cutoff(get_cutoff(df, return_index=True)),
            "names": list(df.index.names),
        },
    }
    try:
        mid = convert_to(df, to_type=target)
        rec["intermediate_time_dtype"] = inter_time_dtype(mid)
        try:
            back = convert_to(mid, to_type=src)
        except TypeError as e:
            if "dask" not in target:
                raise
            # sktime's dask mtype check fails on dask-expr collections
            # (python_type tag "dask.dataframe" -> class name "dataframe");
            # bypass the checker and call the adapter directly.
            from sktime.datatypes._adapter.dask_to_pd import convert_dask_to_pandas

            rec["note"] = "convert_to back failed (" + str(e)[:80] + "); used adapter"
            back = convert_dask_to_pandas(mid)
        lvl1, vals1 = time_level(back)
        valid = check_is_mtype(back, src, return_metadata=False)
        # value comparison, mapping periods to period-start timestamps
        v0 = vals0
        if isinstance(vals0, pd.PeriodIndex) and not isinstance(
            vals1, pd.PeriodIndex
        ):
            v0 = vals0.to_timestamp()  # period START timestamps
        try:
            same_vals = bool(np.array_equal(np.asarray(v0), np.asarray(vals1)))
            if not same_vals and len(v0) == len(vals1):
                same_vals = bool((pd.Index(v0) == pd.Index(vals1)).all())
        except Exception as e:  # noqa: BLE001
            same_vals = f"cmp error: {e}"
        rec["after"] = {
            "time_index_cls": type(lvl1).__name__,
            "dtype": str(vals1.dtype),
            "levels_freq": freq_of(lvl1),
            "get_time_index_freq": freq_of(get_time_index(back)),
            "cutoff": describe_cutoff(get_cutoff(back, return_index=True)),
            "names": list(back.index.names),
            "valid_" + src: bool(valid),
            "values_equal_as_timestamps": same_vals,
            "same_row_order": bool(
                np.array_equal(df["y"].to_numpy(), back["y"].to_numpy())
            ),
        }
        lost = []
        b, a = rec["before"], rec["after"]
        if b["time_index_cls"] != a["time_index_cls"]:
            lost.append(f"cls {b['time_index_cls']}->{a['time_index_cls']}")
        if b["dtype"] != a["dtype"]:
            lost.append(f"dtype {b['dtype']}->{a['dtype']}")
        if b["get_time_index_freq"] != a["get_time_index_freq"]:
            lost.append(f"freq {b['get_time_index_freq']}->{a['get_time_index_freq']}")
        if b["cutoff"]["type"] != a["cutoff"]["type"]:
            lost.append(f"cutoff {b['cutoff']['type']}->{a['cutoff']['type']}")
        if b["cutoff"]["index_freq"] != a["cutoff"]["index_freq"]:
            lost.append(
                f"cutoff.freq {b['cutoff']['index_freq']}->{a['cutoff']['index_freq']}"
            )
        if b["names"] != a["names"]:
            lost.append(f"names {b['names']}->{a['names']}")
        if a["values_equal_as_timestamps"] is not True:
            lost.append("VALUES CHANGED")
        if not a["valid_" + src]:
            lost.append("INVALID MTYPE")
        if "note" in rec:
            lost.append("convert_to(dask->pd) BROKEN, adapter used")
        rec["lost"] = lost
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec["trace"] = traceback.format_exc()[-600:]
        rec["lost"] = ["ERROR"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])  # unused
    ap.parse_args()
    out = {"env": env_info(), "records": []}
    for scitype in ["Series", "Panel", "Hierarchical"]:
        for target in TARGETS[scitype]:
            for name, tix in time_indexes().items():
                r = run_case(scitype, name, tix, target)
                out["records"].append(r)
                lost = "; ".join(r["lost"]) or "nothing"
                print(
                    f"{scitype:12s} {target:20s} {name:13s} "
                    f"mid={r.get('intermediate_time_dtype', '-')!s:45.45s} "
                    f"LOST: {lost}"
                )
    tag = "pd3" if PD3 else "pd2"
    os.makedirs("experiments/container_independence/results", exist_ok=True)
    path = f"experiments/container_independence/results/a3_roundtrip_{tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
