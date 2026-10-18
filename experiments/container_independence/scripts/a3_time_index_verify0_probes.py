"""a3_time_index_verify0: small independent probes of a3 claims 2, 9, 10, 11.

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/a3_time_index_verify0_probes.py
"""

import json
import traceback
import warnings

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")
out = {"pandas": pd.__version__, "polars": pl.__version__}


def probe(name):
    def deco(f):
        try:
            out[name] = f()
        except Exception as e:  # noqa: BLE001
            out[name] = f"ERR {type(e).__name__}: {e}"[:300]
        print(f"--- {name}\n{out[name]}", flush=True)
        return f
    return deco


@probe("claim2_cbd_via_forecaster")
def _():
    # end-to-end: NaiveForecaster on a CBD-indexed series, holiday inside the horizon
    from sktime.forecasting.naive import NaiveForecaster

    hol = ["2001-01-15"]  # a Monday
    cbd = pd.offsets.CustomBusinessDay(holidays=hol)
    idx = pd.date_range("2000-11-01", "2001-01-10", freq=cbd)
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    p = NaiveForecaster().fit(y, fh=[1, 2, 3, 4, 5]).predict()
    truth = pd.date_range(idx[-1], periods=6, freq=cbd)[1:]
    return {"pred_index": [str(t.date()) for t in p.index],
            "holiday_aware_truth": [str(t.date()) for t in truth],
            "holiday_in_pred": "2001-01-15" in [str(t.date()) for t in p.index],
            "y.index.freq": str(y.index.freq)}


@probe("claim9_period_roundtrip")
def _():
    from sktime.datatypes import convert_to

    res = {}
    for f in ["M", "Q", "W-SUN", "D"]:
        idx = pd.period_range("2000-01", periods=6, freq=f)
        df = pd.DataFrame({"y": np.arange(6.0)}, index=idx)
        pl_df = convert_to(df, "pl.DataFrame")
        back = convert_to(pl_df, "pd.DataFrame")
        tcol = [c for c in pl_df.columns if c.startswith("__index__")]
        res[f] = {"polars_cols": pl_df.columns,
                  "polars_time_dtype": str(pl_df[tcol[0]].dtype) if tcol else None,
                  "first_period": str(idx[0]),
                  "first_polars_value": str(pl_df[tcol[0]][0]) if tcol else None,
                  "back_index_type": type(back.index).__name__,
                  "back_first": str(back.index[0]), "back_cols": list(back.columns)}
    # panel
    mi = pd.MultiIndex.from_product([["a", "b"], pd.period_range("2000-01", periods=3,
                                                                  freq="M")],
                                    names=["h0", "time"])
    dfp = pd.DataFrame({"y": np.arange(6.0)}, index=mi)
    plp = convert_to(dfp, "polars_panel")
    backp = convert_to(plp, "pd-multiindex")
    res["panel M"] = {"polars_cols": plp.columns, "first": str(plp[plp.columns[1]][0]),
                      "back_level_type": type(backp.index.levels[-1]).__name__}
    return res


@probe("claim9_date_and_series_naming")
def _():
    from sktime.forecasting.naive import NaiveForecaster

    t = pl.date_range(pl.date(2000, 1, 1), pl.date(2001, 12, 1), "1mo", eager=True)
    y = pl.DataFrame({"__index__time": t, "y": np.arange(len(t), dtype=float)})
    p = NaiveForecaster().fit(y, fh=[1, 2, 3]).predict()
    return {"in_schema": str(y.schema), "out_type": type(p).__name__,
            "out_schema": str(p.schema) if hasattr(p, "schema") else None,
            "out_first": str(p.row(0)) if hasattr(p, "row") else None}


@probe("claim10_dask")
def _():
    import dask.dataframe as dd

    from sktime.datatypes import check_is_mtype, convert_to
    from sktime.datatypes._adapter.dask_to_pd import (
        convert_dask_to_pandas,
        convert_pandas_to_dask,
    )

    mi = pd.MultiIndex.from_product([["a", "b"], range(12)], names=["h0", "time"])
    df = pd.DataFrame({"y": np.arange(24.0)}, index=mi)
    res = {"dask": __import__("dask").__version__}
    try:
        ddf = convert_to(df, "dask_panel")
        res["convert_to dask_panel"] = type(ddf).__module__ + "." + type(ddf).__name__
        try:
            back = convert_to(ddf, "pd-multiindex")
            res["convert_to back"] = "ok " + str(back.index.get_level_values(-1)[:5].tolist())
        except Exception as e:  # noqa: BLE001
            res["convert_to back"] = f"ERR {type(e).__name__}: {str(e)[:150]}"
    except Exception as e:  # noqa: BLE001
        res["convert_to dask_panel"] = f"ERR {type(e).__name__}: {str(e)[:150]}"
    ddf2 = convert_pandas_to_dask(df, npartitions=1)
    back2 = convert_dask_to_pandas(ddf2)
    res["adapter_back_times_a"] = back2.loc["a"].index.tolist() \
        if "a" in back2.index.get_level_values(0) else str(back2.index[:5].tolist())
    res["adapter_back_is_mtype"] = bool(check_is_mtype(back2, "pd-multiindex"))
    res["dd.DataFrame class"] = dd.DataFrame.__module__ + "." + dd.DataFrame.__name__
    return res


@probe("claim11_narwhals_mo")
def _():
    import datetime as dt

    import dask.dataframe as dd
    import duckdb
    import narwhals as nw
    import pyarrow as pa

    base = pd.DataFrame({"t": pd.to_datetime(["2000-01-31", "2000-02-29"])})
    natives = {"pandas": base, "polars": pl.from_pandas(base),
               "polars_lazy": pl.from_pandas(base).lazy(),
               "pyarrow": pa.table(base), "dask": dd.from_pandas(base, npartitions=1),
               "duckdb": duckdb.sql("select * from base")}
    res = {"narwhals": nw.__version__}
    for name, nat in natives.items():
        r = {}
        for expr_name, e in {"offset_by 1mo": nw.col("t").dt.offset_by("1mo"),
                             "offset_by 1h": nw.col("t").dt.offset_by("1h"),
                             "truncate 1mo": nw.col("t").dt.truncate("1mo")}.items():
            try:
                f = nw.from_native(nat).select(e)
                if hasattr(f, "collect"):
                    f = f.collect()
                r[expr_name] = [str(x) for x in f["t"].to_list()]
            except Exception as ex:  # noqa: BLE001
                r[expr_name] = f"ERR {type(ex).__name__}: {str(ex)[:80]}"
        r["has dt.month_end"] = hasattr(nw.col("t").dt, "month_end")
        r["has add_business_days"] = hasattr(nw.col("t").dt, "add_business_days")
        res[name] = r
    _ = dt
    return res


if __name__ == "__main__":
    tag = "pd3" if int(pd.__version__.split(".")[0]) >= 3 else "pd2"
    with open("experiments/container_independence/results/"
              f"a3_time_index_verify0_probes_{tag}.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
