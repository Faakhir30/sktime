"""r4: "canonical interchange format" pattern -- Arrow (via PyCapsule) as the
single internal representation for per-instance iteration.

Pattern under test (prior art: DuckDB/polars/pyarrow consumers, pandera's
pyarrow backend, narwhals' interchange level): accept any input exposing
``__arrow_c_stream__`` (or collect lazy inputs once), obtain a pyarrow Table
zero-copy, compute contiguous instance runs once, and hand each instance to
the per-instance function as numpy views (no per-group frame objects).

Compared with
  (a) today's sktime path for a polars input: convert to pandas MultiIndex,
      then groupby(level=0, sort=False) loop
  (b) the pandas-native loop on a pandas input (status quo baseline)

Run (from WT root):
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/r4_canonical_arrow.py --grid dev
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

from common import DEV_GRID, LARGE_GRID, bench, env_info, make_panel_fast, rss_mb

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")


def f(arr):
    return float(arr.mean() + arr[-1])


def arrow_runs(tbl, key="h0", value="c0", time="time", assume_contiguous=True):
    """Yield (key, numpy view) per contiguous instance run of an Arrow table.

    Only kernels that support Arrow *view* types are used on the key column
    (polars 2.x exports strings as ``string_view``; pyarrow 26 has no
    ``take``/``sort_indices`` kernels for view types -- see report).
    """
    import pyarrow.compute as pc

    keys = tbl[key].combine_chunks()
    de = pc.dictionary_encode(keys)
    enc = de.indices.to_numpy(zero_copy_only=False)
    vals = tbl[value].combine_chunks()
    vals = vals.to_numpy(zero_copy_only=vals.null_count == 0)
    if not assume_contiguous:
        t = tbl[time].combine_chunks().to_numpy(zero_copy_only=False)
        order = np.lexsort((t, enc))
        enc = enc[order]
        vals = vals[order]
    bounds = np.flatnonzero(enc[1:] != enc[:-1]) + 1
    starts = np.r_[0, bounds]
    uniq = de.dictionary.to_pylist()
    k_list = [uniq[i] for i in enc[starts]]
    return zip(k_list, np.split(vals, bounds))


def to_arrow(obj):
    """Canonical entry point: anything -> pyarrow.Table."""
    import pyarrow as pa

    if hasattr(obj, "collect") and not hasattr(obj, "__arrow_c_stream__"):
        obj = obj.collect()  # polars LazyFrame: single explicit materialization
    if hasattr(obj, "compute") and not hasattr(obj, "__arrow_c_stream__"):
        obj = obj.compute()  # dask: materialize partitions (pandas) once
    if hasattr(obj, "__arrow_c_stream__"):
        t = pa.table(obj) if not hasattr(obj, "read_all") else obj.read_all()
        return t
    raise TypeError(type(obj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    import dask.dataframe as dd
    import duckdb
    import polars as pl
    import pyarrow as pa

    from sktime.datatypes import convert_to

    out = {"env": env_info(), "python": sys.version.split()[0], "rows": []}
    con = duckdb.connect()
    con.execute("SET threads TO 2")
    for n_series, n_time in grid:
        pdf = make_panel_fast(n_series=n_series, n_time=n_time)
        flat = pdf.reset_index()[["h0", "time", "c0"]]
        pldf = pl.from_pandas(flat)
        lf = pldf.lazy()
        con.register("pl_v", pldf)
        rel = lambda: con.sql("SELECT h0, time, c0 FROM pl_v ORDER BY h0, time")  # noqa
        ddf = dd.from_pandas(flat, npartitions=4)
        # sktime polars mtype (__index__ convention) for the status-quo path
        pl_sk = convert_to(pdf, to_type="polars_panel") if n_series <= 1000 else None

        ref = {k: f(g["c0"].to_numpy()) for k, g in pdf.groupby(level=0, sort=False)}

        def loop_pd():
            return {k: f(g["c0"].to_numpy()) for k, g in pdf.groupby(level=0, sort=False)}

        def statusquo_polars():
            X = convert_to(pl_sk, to_type="pd-multiindex")
            return {k: f(g["c0"].to_numpy()) for k, g in X.groupby(level=0, sort=False)}

        def canon(obj_fn, contiguous=True):
            def run():
                return {
                    k: f(a)
                    for k, a in arrow_runs(
                        to_arrow(obj_fn()), assume_contiguous=contiguous
                    )
                }

            return run

        cases = {
            "status quo: pandas MI in, groupby loop": loop_pd,
            "status quo: polars(sktime mtype) in -> convert_to pd -> groupby loop": (
                statusquo_polars if pl_sk is not None else None
            ),
            "canonical arrow: pandas flat in": canon(lambda: flat),
            "canonical arrow: polars in": canon(lambda: pldf),
            "canonical arrow: polars lazy in (collect once)": canon(lambda: lf),
            "canonical arrow: duckdb relation in (ORDER BY)": canon(rel),
            "canonical arrow: dask in (compute once)": canon(lambda: ddf),
            "canonical arrow: polars in, unsorted path (lexsort on codes)": canon(
                lambda: pldf, contiguous=False
            ),
            "conversion only: polars -> pa.Table (capsule)": lambda: to_arrow(pldf),
            "conversion only: sktime convert_to polars_panel -> pd-multiindex": (
                (lambda: convert_to(pl_sk, to_type="pd-multiindex"))
                if pl_sk is not None
                else None
            ),
        }
        row = {"n_series": n_series, "n_time": n_time, "cases": {}}
        for name, fn in cases.items():
            if fn is None:
                continue
            try:
                res = fn()
                ok = (
                    "n/a"
                    if name.startswith("conversion")
                    else bool(
                        len(res) == len(ref)
                        and all(np.isclose(ref[k], res[k]) for k in ref)
                    )
                )
                t = bench(fn, repeat=args.repeat)
                row["cases"][name] = {
                    "sec": t,
                    "us_per_group": t / n_series * 1e6,
                    "correct": ok,
                }
            except Exception as e:  # noqa: BLE001
                row["cases"][name] = {"error": repr(e)[:300]}
        row["rss_mb"] = rss_mb()
        out["rows"].append(row)
        base = row["cases"]["status quo: pandas MI in, groupby loop"]["sec"]
        print(f"\n== n_series={n_series} n_time={n_time} rss={row['rss_mb']:.0f}MB", flush=True)
        for name, r in row["cases"].items():
            if "sec" in r:
                print(
                    f"  {name:68s} {r['sec']*1e3:9.2f} ms {r['us_per_group']:8.2f} us/grp"
                    f" x{r['sec']/base:6.3f} correct={r['correct']}", flush=True
                )
            else:
                print(f"  {name:68s} ERROR {r['error'][:140]}", flush=True)
    path = os.path.join(RES, f"r4_canonical_arrow_{args.grid}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
