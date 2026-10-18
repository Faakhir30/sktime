"""r1: status of cross-library dataframe protocols, measured on the installed libs.

Part A  protocol surface: which containers expose ``__dataframe__`` (interchange
        protocol), ``__arrow_c_stream__`` / ``__arrow_c_array__`` /
        ``__arrow_c_schema__`` (Arrow PyCapsule Interface) and
        ``__dataframe_consortium_standard__`` (DataFrame API Standard).
Part B  deprecation status: call the interchange entry points with warnings
        recorded and report the warning category + message.
Part C  index fidelity: does a pandas (multi)index survive each route?
Part D  conversion cost polars <-> pandas through each route (min-of-repeats, same
        process, interleaved), plus peak RSS delta.

Run (from the worktree root; both pandas envs):
  PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore \
      experiments/container_independence/scripts/r1_interop_protocols.py --grid dev
  PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore \
      experiments/container_independence/scripts/r1_interop_protocols.py --grid dev
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

from common import DEV_GRID, LARGE_GRID, bench, env_info, make_panel_fast, peak_rss_delta_mb

ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="dev", choices=["dev", "large"])
ap.add_argument("--repeat", type=int, default=5)
args = ap.parse_args()
GRID = DEV_GRID if args.grid == "dev" else LARGE_GRID

out = {"env": env_info(), "grid": args.grid}
PROTOS = [
    "__dataframe__",
    "__arrow_c_stream__",
    "__arrow_c_array__",
    "__arrow_c_schema__",
    "__dataframe_consortium_standard__",
]

# ------------------------------------------------------------------ Part A
small = make_panel_fast(3, 4, n_cols=2)
small_flat = small.reset_index()
pl_small = pl.from_pandas(small_flat)
objs = {
    "pd.DataFrame(MultiIndex)": small,
    "pd.DataFrame(flat)": small_flat,
    "pd.Series": small["c0"],
    "pl.DataFrame": pl_small,
    "pl.LazyFrame": pl_small.lazy(),
    "pl.Series": pl_small["c0"],
    "pa.Table": pa.table(small_flat),
}
try:
    import duckdb

    con = duckdb.connect()
    objs["duckdb.DuckDBPyRelation"] = con.sql("select * from pl_small")
except Exception as e:  # noqa: BLE001
    out["duckdb_error"] = repr(e)
try:
    import dask.dataframe as dd

    objs["dask.DataFrame"] = dd.from_pandas(small_flat, npartitions=1)
except Exception as e:  # noqa: BLE001
    out["dask_error"] = repr(e)
try:
    import narwhals.stable.v2 as nw

    objs["narwhals.DataFrame(polars)"] = nw.from_native(pl_small)
    objs["narwhals.DataFrame(pandas)"] = nw.from_native(small_flat)
except Exception as e:  # noqa: BLE001
    out["narwhals_error"] = repr(e)

out["A_protocol_surface"] = {
    k: {p: hasattr(type(o), p) or hasattr(o, p) for p in PROTOS} for k, o in objs.items()
}


# ------------------------------------------------------------------ Part B
def record(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = fn()
            err = None
        except Exception as e:  # noqa: BLE001
            res, err = None, f"{type(e).__name__}: {str(e)[:200]}"
    ws = [f"{x.category.__name__}: {str(x.message)[:220]}" for x in w]
    # dedupe while keeping order
    ws = list(dict.fromkeys(ws))
    return res, err, ws


calls = {
    "pd_df.__dataframe__()": lambda: small_flat.__dataframe__(),
    "pl_df.__dataframe__()": lambda: pl_small.__dataframe__(),
    "pd.api.interchange.from_dataframe(pl_df)": lambda: pd.api.interchange.from_dataframe(
        pl_small
    ),
    "pl.from_dataframe(pd_df)": lambda: pl.from_dataframe(small_flat),
    "pyarrow.interchange.from_dataframe(pd_df)": lambda: __import__(
        "pyarrow.interchange", fromlist=["from_dataframe"]
    ).from_dataframe(small_flat),
    "pa.table(pl_df) [capsule]": lambda: pa.table(pl_small),
    "pa.table(pd_df) [capsule or pandas path]": lambda: pa.table(small_flat),
    "pa.RecordBatchReader.from_stream(pd_df)": lambda: pa.RecordBatchReader.from_stream(
        small_flat
    ).read_all(),
    "pl.DataFrame(pa_table)": lambda: pl.DataFrame(pa.table(small_flat)),
    "pl.DataFrame(pd_df) [constructor]": lambda: pl.DataFrame(small_flat),
}
if hasattr(pd.DataFrame, "from_arrow"):
    calls["pd.DataFrame.from_arrow(pl_df) [capsule]"] = lambda: pd.DataFrame.from_arrow(
        pl_small
    )
out["pandas_has_DataFrame.from_arrow"] = hasattr(pd.DataFrame, "from_arrow")
# what pandas itself says about the interchange protocol (docstring note/warning)
_doc = pd.api.interchange.from_dataframe.__doc__ or ""
out["pandas_from_dataframe_doc_note"] = " ".join(_doc.split())[:700]
out["B_deprecation"] = {}
for k, fn in calls.items():
    res, err, ws = record(fn)
    out["B_deprecation"][k] = {
        "ok": err is None,
        "error": err,
        "warnings": ws,
        "result_type": type(res).__name__ if res is not None else None,
    }

# ------------------------------------------------------------------ Part C
mi = small  # MultiIndex (h0, time), cols c0,c1


def idx_desc(df):
    if isinstance(df, pd.DataFrame):
        return {
            "index_type": type(df.index).__name__,
            "index_names": [str(n) for n in df.index.names],
            "columns": [str(c) for c in df.columns],
        }
    if isinstance(df, pl.DataFrame):
        return {"columns": df.columns}
    if isinstance(df, pa.Table):
        return {
            "columns": df.column_names,
            "has_pandas_metadata": bool(df.schema.metadata and b"pandas" in df.schema.metadata),
        }
    return {"type": type(df).__name__}


routes = {
    "interchange: pl.from_dataframe(pd_mi)": lambda: pl.from_dataframe(mi),
    "interchange: pd.api.interchange.from_dataframe(pd_mi)": lambda: pd.api.interchange.from_dataframe(
        mi
    ),
    "capsule: pa.table(pd_mi)": lambda: pa.table(mi),
    "capsule: RecordBatchReader.from_stream(pd_mi)": lambda: pa.RecordBatchReader.from_stream(
        mi
    ).read_all(),
    "capsule roundtrip: pa.table(pd_mi).to_pandas()": lambda: pa.table(mi).to_pandas(),
    "pyarrow: pa.Table.from_pandas(pd_mi).to_pandas()": lambda: pa.Table.from_pandas(
        mi
    ).to_pandas(),
    "polars: pl.from_pandas(pd_mi) [default include_index=False]": lambda: pl.from_pandas(mi),
    "polars: pl.from_pandas(pd_mi, include_index=True)": lambda: pl.from_pandas(
        mi, include_index=True
    ),
    "polars: pl.DataFrame(pd_mi) [constructor]": lambda: pl.DataFrame(mi),
}
if hasattr(pd.DataFrame, "from_arrow"):
    routes["capsule: pd.DataFrame.from_arrow(pa.table(pd_mi))"] = lambda: pd.DataFrame.from_arrow(
        pa.table(mi)
    )
out["C_index_fidelity"] = {}
for k, fn in routes.items():
    res, err, ws = record(fn)
    out["C_index_fidelity"][k] = {
        "ok": err is None,
        "error": err,
        "warnings": ws[:2],
        "result": idx_desc(res) if res is not None else None,
    }

# ------------------------------------------------------------------ Part D
rows = []
for n_series, n_time in GRID:
    pdf = make_panel_fast(n_series, n_time, n_cols=1)
    flat = pdf.reset_index()
    pldf = pl.from_pandas(flat)
    tbl = pa.table(flat)
    n_rows = len(flat)
    pl2pd = {
        "pl.to_pandas()": lambda: pldf.to_pandas(),
        "interchange pd.api.interchange.from_dataframe": lambda: pd.api.interchange.from_dataframe(
            pldf
        ),
        "capsule pa.table(pl).to_pandas()": lambda: pa.table(pldf).to_pandas(),
    }
    if hasattr(pd.DataFrame, "from_arrow"):
        pl2pd["capsule pd.DataFrame.from_arrow(pl)"] = lambda: pd.DataFrame.from_arrow(pldf)
    pd2pl = {
        "pl.from_pandas(flat)": lambda: pl.from_pandas(flat),
        "interchange pl.from_dataframe(flat)": lambda: pl.from_dataframe(flat),
        "capsule pl.DataFrame(pa.table(flat))": lambda: pl.DataFrame(pa.table(flat)),
        "pl.from_pandas(mi, include_index=True)": lambda: pl.from_pandas(pdf, include_index=True),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for direction, d in [("pl->pd", pl2pd), ("pd->pl", pd2pl)]:
            for name, fn in d.items():
                try:
                    t = bench(fn, repeat=args.repeat)
                    _, mem = peak_rss_delta_mb(fn)
                    err = None
                except Exception as e:  # noqa: BLE001
                    t, mem, err = None, None, f"{type(e).__name__}: {str(e)[:120]}"
                rows.append(
                    dict(
                        n_series=n_series,
                        n_time=n_time,
                        n_rows=n_rows,
                        direction=direction,
                        route=name,
                        sec=t,
                        peak_rss_mb=mem,
                        error=err,
                    )
                )
    del pdf, flat, pldf, tbl
out["D_conversion_cost"] = rows

# ratios vs baseline per (size, direction)
ratios = []
for (ns, nt, dirn) in sorted({(r["n_series"], r["n_time"], r["direction"]) for r in rows}):
    rs = [r for r in rows if (r["n_series"], r["n_time"], r["direction"]) == (ns, nt, dirn)]
    base = rs[0]["sec"]
    for r in rs:
        ratios.append(
            dict(
                n_series=ns,
                n_time=nt,
                direction=dirn,
                route=r["route"],
                ms=None if r["sec"] is None else round(r["sec"] * 1e3, 3),
                ratio_vs_first=None if (r["sec"] is None or not base) else round(r["sec"] / base, 2),
                peak_rss_mb=None if r["peak_rss_mb"] is None else round(r["peak_rss_mb"], 1),
                error=r["error"],
            )
        )
out["D_ratios"] = ratios

tag = "pd" + pd.__version__.split(".")[0]
print(json.dumps({k: v for k, v in out.items() if k not in ("D_conversion_cost",)}, indent=1, default=str))
os.makedirs("experiments/container_independence/results", exist_ok=True)
fn = f"experiments/container_independence/results/r1_interop_protocols_{tag}_{args.grid}.json"
with open(fn, "w") as f:
    json.dump(out, f, indent=1, default=str)
print("wrote", fn)
