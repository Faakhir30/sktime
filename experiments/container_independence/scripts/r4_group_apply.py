"""r4: engine-native "apply a Python function per group" mechanisms and their overhead.

This is the core operation of sktime vectorization (VectorizedDF: one estimator
per instance).  We compare, on identical data, the cost of calling a trivial
Python function once per instance group in each engine's idiomatic way, plus the
"expression-native" (no Python per group) baseline for a forecaster that *can*
be expressed as an aggregate (last value / mean).

The per-group function is deliberately trivial (mean + last of a float array) so
the measured time is framework overhead, not user work.

Run (from WT root):
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/r4_group_apply.py --grid dev
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
    """The per-group 'user function': trivial, so we measure framework overhead."""
    return float(arr.mean() + arr[-1])


def flat_frames(pdf):
    import polars as pl

    flat = pdf.reset_index()
    pldf = pl.from_pandas(flat)
    return flat, pldf


def methods(pdf, flat, pldf, skip_slow):
    import dask
    import dask.dataframe as dd
    import duckdb
    import ibis
    import narwhals as nw
    import polars as pl
    import pyarrow as pa

    M = {}

    # ---------------- pandas ----------------
    def pd_groupby_loop():  # what VectorizedDF does today
        return {k: f(g["c0"].to_numpy()) for k, g in pdf.groupby(level=0, sort=False)}

    M["pandas: for k,g in groupby(level=0)"] = pd_groupby_loop

    def pd_groupby_apply():
        s = pdf.groupby(level=0, sort=False)["c0"].apply(lambda g: f(g.to_numpy()))
        return dict(s.items())

    M["pandas: groupby(level=0)[c].apply(f)"] = pd_groupby_apply

    def pd_numpy_split():  # grouped-array lower bound: codes + offsets, no frames
        codes = pdf.index.codes[0]
        # data is sorted by instance in sktime multiindex format; offsets via diff
        bounds = np.flatnonzero(np.diff(codes)) + 1
        vals = pdf["c0"].to_numpy()
        keys = pdf.index.levels[0][codes[np.r_[0, bounds]]]
        return {k: f(a) for k, a in zip(keys, np.split(vals, bounds))}

    M["numpy: offsets + np.split (lower bound)"] = pd_numpy_split

    # ---------------- polars ----------------
    def pl_partition_by():
        parts = pldf.partition_by("h0", maintain_order=True, as_dict=True)
        return {k[0]: f(g["c0"].to_numpy()) for k, g in parts.items()}

    M["polars: partition_by(as_dict) loop"] = pl_partition_by

    def pl_group_iter():
        return {
            k[0]: f(g["c0"].to_numpy())
            for k, g in pldf.group_by("h0", maintain_order=True)
        }

    M["polars: for k,g in group_by()"] = pl_group_iter

    def pl_map_groups():
        out = pldf.group_by("h0", maintain_order=True).map_groups(
            lambda g: pl.DataFrame(
                {"h0": [g["h0"][0]], "r": [f(g["c0"].to_numpy())]}
            )
        )
        return dict(zip(out["h0"].to_list(), out["r"].to_list()))

    M["polars: group_by().map_groups(f)"] = pl_map_groups

    def pl_lazy_map_batches():
        out = (
            pldf.lazy()
            .group_by("h0", maintain_order=True)
            .agg(
                pl.col("c0")
                .map_batches(
                    lambda s: f(s.to_numpy()),
                    returns_scalar=True,
                    return_dtype=pl.Float64,
                )
                .alias("r")
            )
            .collect()
        )
        return dict(zip(out["h0"].to_list(), out["r"].to_list()))

    M["polars lazy: agg(col.map_batches(f, returns_scalar))"] = pl_lazy_map_batches

    def pl_list_agg_numpy():  # implode per group in rust, python loop over lists
        out = pldf.group_by("h0", maintain_order=True).agg(pl.col("c0"))
        return {
            k: f(np.asarray(v))
            for k, v in zip(out["h0"].to_list(), out["c0"].to_list())
        }

    M["polars: agg(list) then python loop"] = pl_list_agg_numpy

    # ---------------- narwhals ----------------
    nw_pd = nw.from_native(flat, eager_only=True)
    nw_pl = nw.from_native(pldf, eager_only=True)

    def nw_iter_pd():
        return {
            k[0]: f(g["c0"].to_numpy()) for k, g in nw_pd.group_by("h0").__iter__()
        }

    def nw_iter_pl():
        return {
            k[0]: f(g["c0"].to_numpy()) for k, g in nw_pl.group_by("h0").__iter__()
        }

    M["narwhals(pandas): for k,g in group_by()"] = nw_iter_pd
    M["narwhals(polars): for k,g in group_by()"] = nw_iter_pl

    # ---------------- duckdb ----------------
    con = duckdb.connect()
    con.execute("SET threads TO 2")
    con.register("pldf_v", pldf)

    def py_list_udf(lists):  # arrow UDF: called once per vector (~2048 groups)
        return pa.array(
            [f(np.asarray(x, dtype="float64")) for x in lists.to_pylist()],
            type=pa.float64(),
        )

    con.create_function(
        "f_arrow",
        py_list_udf,
        [duckdb.list_type(duckdb.sqltype("DOUBLE"))],
        duckdb.sqltype("DOUBLE"),
        type="arrow",
    )

    def f_native(lst):
        return f(np.asarray(lst, dtype="float64"))

    con.create_function(
        "f_native",
        f_native,
        [duckdb.list_type(duckdb.sqltype("DOUBLE"))],
        duckdb.sqltype("DOUBLE"),
        type="native",
    )

    def ddb_arrow_udf():
        r = con.sql(
            "SELECT h0, f_arrow(list(c0 ORDER BY time)) AS r FROM pldf_v GROUP BY h0"
        ).fetchall()
        return dict(r)

    def ddb_native_udf():
        r = con.sql(
            "SELECT h0, f_native(list(c0 ORDER BY time)) AS r FROM pldf_v GROUP BY h0"
        ).fetchall()
        return dict(r)

    def ddb_fetch_then_split():
        tbl = con.sql("SELECT h0, c0 FROM pldf_v ORDER BY h0, time").arrow()
        tbl = tbl.read_all() if hasattr(tbl, "read_all") else tbl
        keys = tbl["h0"].to_numpy(zero_copy_only=False)
        vals = tbl["c0"].to_numpy()
        bounds = np.flatnonzero(keys[1:] != keys[:-1]) + 1
        ks = keys[np.r_[0, bounds]]
        return {k: f(a) for k, a in zip(ks, np.split(vals, bounds))}

    M["duckdb: list() agg + arrow UDF (vector of groups)"] = ddb_arrow_udf
    M["duckdb: list() agg + native UDF (1 call/group)"] = ddb_native_udf
    M["duckdb: sorted fetch -> arrow -> numpy split"] = ddb_fetch_then_split

    # ---------------- ibis (duckdb backend) ----------------
    icon = ibis.duckdb.connect()
    icon.raw_sql("SET threads TO 2")
    # NB: icon.create_table(...) raises ParserException with ibis 12.0.0 +
    # duckdb 1.5.6 (version skew); register through the raw connection instead.
    icon.con.register("panel_v", pa.table(flat[["h0", "time", "c0"]]))
    it = icon.table("panel_v")

    @ibis.udf.scalar.python
    def f_ibis(x: list[float]) -> float:
        return f(np.asarray(x, dtype="float64"))

    def ibis_udf():
        e = it.group_by("h0").aggregate(v=it.c0.collect(order_by=it.time))
        e = e.mutate(r=f_ibis(e.v)).select("h0", "r")
        out = e.to_pyarrow()
        return dict(zip(out["h0"].to_pylist(), out["r"].to_pylist()))

    M["ibis(duckdb): collect(order_by) + python scalar UDF"] = ibis_udf

    # ---------------- dask ----------------
    dflat = dd.from_pandas(flat[["h0", "time", "c0"]], npartitions=4)

    def dask_groupby_apply():
        with dask.config.set(scheduler="sync"):
            s = (
                dflat.groupby("h0")["c0"]
                .apply(lambda g: f(g.to_numpy()), meta=("c0", "float64"))
                .compute()
            )
        return dict(s.items())

    def dask_groupby_apply_threads():
        with dask.config.set(scheduler="threads", num_workers=2):
            s = (
                dflat.groupby("h0")["c0"]
                .apply(lambda g: f(g.to_numpy()), meta=("c0", "float64"))
                .compute()
            )
        return dict(s.items())

    if not skip_slow:
        M["dask(sync): groupby().apply(f) [shuffle]"] = dask_groupby_apply
        M["dask(threads=2): groupby().apply(f) [shuffle]"] = (
            dask_groupby_apply_threads
        )

    return M


def vectorized_baselines(pdf, flat, pldf):
    """No Python per group: the forecaster is expressible as an aggregate."""
    import duckdb
    import polars as pl

    con = duckdb.connect()
    con.execute("SET threads TO 2")
    con.register("pldf_v", pldf)
    import ibis
    import pyarrow as pa

    icon = ibis.duckdb.connect()
    icon.raw_sql("SET threads TO 2")
    icon.con.register("panel_v", pa.table(flat[["h0", "time", "c0"]]))
    it = icon.table("panel_v")
    ie = it.group_by("h0").aggregate(r=it.c0.mean() + it.c0.last(order_by=it.time))
    pcon = ibis.polars.connect()
    pt = pcon.create_table("panel", pldf.select("h0", "time", "c0"))
    pe = pt.group_by("h0").aggregate(r=pt.c0.mean() + pt.c0.last(order_by=pt.time))
    B = {
        "ibis(duckdb): mean + last(order_by) -> pyarrow": lambda: icon.to_pyarrow(ie),
        "ibis(polars): mean + last(order_by) -> pyarrow": lambda: pcon.to_pyarrow(pe),
        "pandas: groupby(level=0).agg(mean,last)": lambda: pdf.groupby(
            level=0, sort=False
        )["c0"].agg(["mean", "last"]),
        "polars: group_by.agg(mean+last)": lambda: pldf.group_by(
            "h0", maintain_order=True
        ).agg((pl.col("c0").mean() + pl.col("c0").last()).alias("r")),
        "polars lazy: group_by.agg(mean+last)": lambda: pldf.lazy()
        .group_by("h0", maintain_order=True)
        .agg((pl.col("c0").mean() + pl.col("c0").last()).alias("r"))
        .collect(),
        "duckdb: avg + last(order by)": lambda: con.sql(
            "SELECT h0, avg(c0) + last(c0 ORDER BY time) FROM pldf_v GROUP BY h0"
        ).fetchall(),
    }
    return B


def check_equal(ref, got, name):
    if len(ref) != len(got):
        return f"len mismatch {len(ref)} vs {len(got)}"
    bad = [k for k in ref if not np.isclose(ref[k], got.get(k, np.nan))]
    return "ok" if not bad else f"{len(bad)} mismatches"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    out = {"env": env_info(), "python": sys.version.split()[0], "rows": []}
    import duckdb
    import ibis

    out["env"]["ibis"] = ibis.__version__
    for n_series, n_time in grid:
        pdf = make_panel_fast(n_series=n_series, n_time=n_time)
        flat, pldf = flat_frames(pdf)
        skip_slow = n_series > 20_000
        M = methods(pdf, flat, pldf, skip_slow)
        ref = M["numpy: offsets + np.split (lower bound)"]()
        row = {"n_series": n_series, "n_time": n_time, "methods": {}}
        for name, fn in M.items():
            try:
                status = check_equal(ref, fn(), name)
                t = bench(fn, repeat=args.repeat)
                row["methods"][name] = {
                    "sec": t,
                    "us_per_group": t / n_series * 1e6,
                    "check": status,
                }
            except Exception as e:  # noqa: BLE001
                row["methods"][name] = {"error": repr(e)[:300]}
        for name, fn in vectorized_baselines(pdf, flat, pldf).items():
            t = bench(fn, repeat=args.repeat)
            row["methods"]["[expr] " + name] = {
                "sec": t,
                "us_per_group": t / n_series * 1e6,
            }
        row["rss_mb"] = rss_mb()
        out["rows"].append(row)
        print(f"\n== n_series={n_series} n_time={n_time} rss={row['rss_mb']:.0f}MB")
        base = row["methods"]["pandas: for k,g in groupby(level=0)"]["sec"]
        for name, r in row["methods"].items():
            if "sec" in r:
                print(
                    f"  {name:58s} {r['sec']*1e3:10.2f} ms "
                    f"{r['us_per_group']:9.2f} us/grp  x{r['sec']/base:7.3f} "
                    f"{r.get('check', '')}"
                )
            else:
                print(f"  {name:58s} ERROR {r['error'][:120]}")
        del pdf, flat, pldf, M
    path = os.path.join(RES, f"r4_group_apply_{args.grid}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
