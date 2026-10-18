"""r4: Fugue transform() as a per-instance "vectorize" engine -- overhead + fidelity.

Fugue's transform(df, fn, schema=..., partition={"by": keys, "presort": t},
engine=...) is the closest prior-art analogue of sktime's VectorizedDF: run a
Python function once per instance group on any engine.  We measure:
  * overhead vs a plain pandas groupby loop and a polars partition_by loop
  * how the per-group frame type is chosen (annotation-driven) and converted
  * what comes back (output container type per engine / input type)
  * what happens to a pandas MultiIndex input (fugue frames have no index)

Needs fugue (NOT in the shared envs): run with the r4 private venv:
  PYTHONPATH=experiments/container_independence \
    $SCRATCH/venvs/r4/bin/python -W ignore \
    experiments/container_independence/scripts/r4_fugue_transform.py --grid dev
"""

import argparse
import json
import os
import sys
import warnings
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from common import DEV_GRID, LARGE_GRID, bench, make_panel_fast, rss_mb

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")


def f(arr):
    return float(arr.mean() + arr[-1])


# --- per-group functions with different annotations (fugue converts by hint) ---
def fn_pandas(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"h0": [df["h0"].iloc[0]], "r": [f(df["c0"].to_numpy())]})


def fn_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    vals = np.array([r["c0"] for r in rows])
    return [{"h0": rows[0]["h0"], "r": f(vals)}]


def make_fn_polars():
    import polars as pl

    def fn_polars(df: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame({"h0": [df["h0"][0]], "r": [f(df["c0"].to_numpy())]})

    return fn_polars


def make_fn_arrow():
    import pyarrow as pa

    def fn_arrow(tbl: pa.Table) -> pa.Table:
        return pa.table(
            {"h0": [tbl["h0"][0].as_py()], "r": [f(tbl["c0"].to_numpy())]}
        )

    return fn_arrow


def to_dict(res):
    """Normalize whatever fugue returned into {key: value}."""
    import fugue.api as fa

    pdf = fa.as_pandas(res)
    return dict(zip(pdf["h0"], pdf["r"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--budget", type=float, default=30.0, help="sec per case")
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID

    import fugue
    import fugue.api as fa
    import polars as pl
    import pyarrow as pa
    from dask.distributed import Client

    client = Client(processes=False, n_workers=1, threads_per_worker=2)
    import duckdb

    out = {
        "env": {
            "fugue": fugue.__version__,
            "pandas": pd.__version__,
            "polars": pl.__version__,
            "pyarrow": pa.__version__,
            "duckdb": duckdb.__version__,
        },
        "python": sys.version.split()[0],
        "fidelity": {},
        "rows": [],
    }

    # ---------------- fidelity probes (small data) ----------------
    pdf = make_panel_fast(n_series=3, n_time=4)
    flat = pdf.reset_index()
    pldf = pl.from_pandas(flat)
    fid = out["fidelity"]
    part = {"by": "h0", "presort": "time"}
    for in_name, data in [("pandas", flat), ("polars", pldf), ("arrow", pa.table(flat))]:
        for eng in [None, "duckdb", "dask"]:
            key = f"input={in_name} engine={eng}"
            try:
                r = fa.transform(
                    data, fn_pandas, schema="h0:str,r:double", partition=part,
                    engine=eng,
                )
                fid[key] = {"returned_type": type(r).__module__ + "." + type(r).__name__}
                if eng is not None:
                    r2 = fa.transform(
                        data, fn_pandas, schema="h0:str,r:double", partition=part,
                        engine=eng, as_local=True,
                    )
                    fid[key]["as_local_type"] = (
                        type(r2).__module__ + "." + type(r2).__name__
                    )
            except Exception as e:  # noqa: BLE001
                fid[key] = {"error": repr(e)[:300]}
    # MultiIndex pandas input (sktime pd-multiindex) directly
    try:
        r = fa.transform(pdf, fn_pandas, schema="h0:str,r:double", partition=part)
        fid["pandas MultiIndex input"] = {"returned_type": type(r).__name__}
    except Exception as e:  # noqa: BLE001
        fid["pandas MultiIndex input"] = {"error": repr(e)[:300]}
    # schema is mandatory?
    try:
        fa.transform(flat, fn_pandas, partition=part)
        fid["schema omitted"] = "ok"
    except Exception as e:  # noqa: BLE001
        fid["schema omitted"] = {"error": repr(e)[:200]}
    # what does the per-group pandas frame look like (index)?
    seen = {}

    def probe(df: pd.DataFrame) -> pd.DataFrame:
        seen["index_type"] = type(df.index).__name__
        seen["columns"] = list(df.columns)
        seen["time_dtype"] = str(df["time"].dtype)
        return df.head(0)[["h0"]].assign(r=0.0)

    fa.transform(flat, probe, schema="h0:str,r:double", partition=part)
    fid["per-group frame seen by fn (pandas engine)"] = dict(seen)
    # period-typed time column
    pflat = make_panel_fast(3, 4, time_kind="period").reset_index()
    try:
        fa.transform(pflat, fn_pandas, schema="h0:str,r:double", partition=part)
        fid["period time column"] = "ok"
    except Exception as e:  # noqa: BLE001
        fid["period time column"] = {"error": repr(e)[:200]}
    print(json.dumps(fid, indent=1), flush=True)

    # ---------------- overhead ----------------
    per_group_prev = {}
    fn_polars = make_fn_polars()
    fn_arrow = make_fn_arrow()
    for n_series, n_time in grid:
        pdf = make_panel_fast(n_series=n_series, n_time=n_time)
        flat = pdf.reset_index()[["h0", "time", "c0"]]
        pldf = pl.from_pandas(flat)
        print(f"\n== n_series={n_series} n_time={n_time}", flush=True)

        def base_pd():
            return {
                k: f(g["c0"].to_numpy()) for k, g in pdf.groupby(level=0, sort=False)
            }

        def base_pl():
            parts = pldf.partition_by("h0", maintain_order=True, as_dict=True)
            return {k[0]: f(g["c0"].to_numpy()) for k, g in parts.items()}

        sch = "h0:str,r:double"
        cases = {
            "baseline pandas groupby loop": base_pd,
            "baseline polars partition_by loop": base_pl,
            "fugue native, pd in, fn(pd.DataFrame)": lambda: to_dict(
                fa.transform(flat, fn_pandas, schema=sch, partition=part)
            ),
            "fugue native, pd in, fn(List[Dict])": lambda: to_dict(
                fa.transform(flat, fn_rows, schema=sch, partition=part)
            ),
            "fugue native, pl in, fn(pl.DataFrame)": lambda: to_dict(
                fa.transform(pldf, fn_polars, schema=sch, partition=part)
            ),
            "fugue native, pd in, fn(pa.Table)": lambda: to_dict(
                fa.transform(flat, fn_arrow, schema=sch, partition=part)
            ),
            "fugue duckdb, pl in, fn(pl.DataFrame)": lambda: to_dict(
                fa.transform(
                    pldf, fn_polars, schema=sch, partition=part, engine="duckdb",
                    as_local=True,
                )
            ),
            "fugue duckdb, pd in, fn(pd.DataFrame)": lambda: to_dict(
                fa.transform(
                    flat, fn_pandas, schema=sch, partition=part, engine="duckdb",
                    as_local=True,
                )
            ),
            "fugue dask(threads=2), pd in, fn(pd.DataFrame)": lambda: to_dict(
                fa.transform(
                    flat, fn_pandas, schema=sch, partition=part, engine="dask",
                    as_local=True,
                )
            ),
            "fugue native, pd in, coarse+presort (one call/partition)": None,
        }
        cases.pop("fugue native, pd in, coarse+presort (one call/partition)")
        ref = base_pd()
        row = {"n_series": n_series, "n_time": n_time, "cases": {}}
        for name, fn in cases.items():
            # skip cases whose projected time (from previous grid point) is > budget
            prev = per_group_prev.get(name)
            if prev is not None and prev * n_series > args.budget:
                row["cases"][name] = {
                    "skipped": f"projected {prev * n_series:.0f}s > budget"
                }
                print(f"  {name:50s} SKIPPED (projected {prev*n_series:.0f}s)", flush=True)
                continue
            try:
                import time as _time

                t0 = _time.perf_counter()
                got = fn()
                first = _time.perf_counter() - t0
                per_group_prev[name] = first / n_series
                ok = len(got) == len(ref) and all(
                    np.isclose(ref[k], got[k]) for k in ref
                )
                if first > 5:  # too slow to repeat: single (cold) measurement
                    t = first
                else:
                    t = bench(fn, repeat=args.repeat, warmup=0)
                row["cases"][name] = {
                    "sec": t, "us_per_group": t / n_series * 1e6, "correct": bool(ok)
                }
            except Exception as e:  # noqa: BLE001
                row["cases"][name] = {"error": repr(e)[:300]}
            r = row["cases"][name]
            if "sec" in r:
                print(
                    f"  {name:50s} {r['sec']*1e3:10.2f} ms {r['us_per_group']:9.1f}"
                    f" us/grp correct={r['correct']}", flush=True
                )
        row["rss_mb"] = rss_mb()
        out["rows"].append(row)
        base = row["cases"]["baseline pandas groupby loop"]["sec"]
        print(f"== summary n_series={n_series} n_time={n_time} rss={row['rss_mb']:.0f}MB")
        for name, r in row["cases"].items():
            if "sec" in r:
                print(
                    f"  {name:50s} {r['sec']*1e3:10.2f} ms {r['us_per_group']:9.1f}"
                    f" us/grp x{r['sec']/base:7.2f} correct={r['correct']}"
                )
            else:
                print(f"  {name:50s} {str(r)[:150]}")
        sys.stdout.flush()
    client.close()
    path = os.path.join(RES, f"r4_fugue_transform_{args.grid}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
