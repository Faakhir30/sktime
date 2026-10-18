"""x1: value of NOT collecting - "stay lazy, collect only per-series results".

Workload (VectorizedDF-flavoured, per instance): cutoff (max time), last value
(NaiveForecaster strategy="last"), mean (strategy="mean"), n obs. Output has one
row per series; input has n_series * n_time rows, stored in a parquet file and
opened lazily by each backend (polars.scan_parquet, duckdb.read_parquet,
dask.read_parquet, ibis.read_parquet).

Routes, per backend:

  R1_collect_all  : nw.from_native(scan).collect()  (default target)  -> same
                    narwhals expression evaluated EAGERLY on the collected frame
  R2_stay_lazy    : same narwhals expression on the LazyFrame, collect() only the
                    n_series-row result
  R0_pandas       : collect(backend="pandas") + pandas groupby (today's sktime
                    shape: everything becomes pandas first)
  R_native        : hand-written native lazy query (polars sort_by().last(),
                    duckdb arg_max SQL, dask groupby) for narwhals-overhead ratio
  R3_iter_filter  : "iterate groups lazily": one filter(key==k).collect() per
                    series (only for n_series <= 1000; reports ms per group)

The narwhals expression uses only ops that the x1 matrix showed lazy-ok on ALL
of polars/dask/duckdb/ibis: semi-join on per-key max time, group_by.agg(mean, len),
inner join.

Run::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x1_stay_lazy.py --grid dev
"""

import argparse
import gc
import json
import os
import sys
import tempfile

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import narwhals as nw  # noqa: E402

from common import (  # noqa: E402
    DEV_GRID,
    LARGE_GRID,
    bench,
    env_info,
    make_panel_fast,
    peak_rss_delta_mb,
    rss_mb,
)

import dask  # noqa: E402
import dask.dataframe as dd  # noqa: E402
import duckdb  # noqa: E402

dask.config.set(scheduler="synchronous")
ibis = None  # imported lazily (heavy import) only where needed


def _import_ibis():
    global ibis
    try:
        import ibis as _ibis

        ibis = _ibis
    except ImportError:
        ibis = None

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
c = nw.col


def per_series_stats(frame):
    """narwhals frame (eager or lazy) -> one row per series."""
    cut = frame.group_by("h0").agg(c("time").max())
    last = frame.join(cut, on=["h0", "time"], how="semi").select(
        "h0", cutoff=c("time"), last=c("c0")
    )
    stats = frame.group_by("h0").agg(mean=c("c0").mean(), n=nw.len())
    return last.join(stats, on="h0", how="inner")


def per_series_stats_lastby(frame):
    """same result using an order-aware aggregation (not supported on dask)."""
    return frame.group_by("h0").agg(
        cutoff=c("time").max(),
        last=c("c0").last(order_by="time"),
        mean=c("c0").mean(),
        n=nw.len(),
    )


def pandas_route(pdf):
    g = pdf.sort_values(["h0", "time"]).groupby("h0", sort=False)
    out = pd.DataFrame(
        {
            "cutoff": g["time"].max(),
            "last": g["c0"].last(),
            "mean": g["c0"].mean(),
            "n": g.size(),
        }
    )
    return out.reset_index()


def canon(x):
    """any result -> polars frame sorted by h0, columns h0,cutoff,last,mean,n."""
    if isinstance(x, (nw.DataFrame,)):
        x = x.to_polars()
    elif isinstance(x, pd.DataFrame):
        x = pl.from_pandas(x)
    elif isinstance(x, pa.Table):
        x = pl.from_arrow(x)
    x = x.select("h0", "cutoff", "last", "mean", "n").with_columns(
        pl.col("cutoff").cast(pl.Datetime("us")),
        pl.col("n").cast(pl.Int64),
        pl.col("last").cast(pl.Float64),
        pl.col("mean").cast(pl.Float64),
    )
    return x.sort("h0")


def scanners(path, dcon, icon):
    s = {
        "polars": lambda: pl.scan_parquet(path),
        "duckdb": lambda: dcon.read_parquet(path),
        "dask": lambda: dd.read_parquet(path),
    }
    if icon is not None:
        s["ibis"] = lambda: icon.read_parquet(path)
    return s


def native_route(name, path, dcon, icon):
    if name == "polars":
        return lambda: (
            pl.scan_parquet(path)
            .group_by("h0")
            .agg(
                cutoff=pl.col("time").max(),
                last=pl.col("c0").sort_by("time").last(),
                mean=pl.col("c0").mean(),
                n=pl.len(),
            )
            .collect()
        )
    if name == "duckdb":
        return lambda: dcon.sql(
            f"SELECT h0, max(time) AS cutoff, arg_max(c0, time) AS last, "
            f"avg(c0) AS mean, count(*) AS n FROM read_parquet('{path}') GROUP BY h0"
        ).to_arrow_table()
    if name == "dask":

        def f():
            d = dd.read_parquet(path)
            g = d.groupby("h0")
            agg = g.agg({"time": "max", "c0": "mean"}).rename(
                columns={"time": "cutoff", "c0": "mean"}
            )
            n = g.size().rename("n")
            mx = d.merge(agg[["cutoff"]].reset_index(), left_on=["h0", "time"],
                         right_on=["h0", "cutoff"])[["h0", "c0"]].rename(columns={"c0": "last"})
            return mx.merge(agg.reset_index(), on="h0").merge(
                n.reset_index(), on="h0"
            ).compute()

        return f
    if name == "ibis":

        def g():
            t = icon.read_parquet(path)
            return (
                t.group_by("h0")
                .aggregate(
                    cutoff=t.time.max(),
                    last=t.c0.argmax(t.time),
                    mean=t.c0.mean(),
                    n=t.count(),
                )
                .to_pyarrow()
            )

        return g
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tag", default=f"pd{pd.__version__.split('.')[0]}")
    ap.add_argument("--iter-max-series", type=int, default=1000)
    ap.add_argument("--backend", default=None,
                    help="run only this backend (driver mode spawns one process each)")
    args = ap.parse_args()
    if args.backend in (None, "ibis"):
        _import_ibis()
    if args.backend is None:
        return driver(args)
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID

    dcon = duckdb.connect()
    dcon.execute("SET threads=4")
    icon = (
        ibis.duckdb.connect(threads=4)
        if ibis is not None and args.backend == "ibis"
        else None
    )
    info = env_info()
    info["ibis"] = getattr(ibis, "__version__", None)
    print("ENV", json.dumps(info))
    results = {"env": info, "grid": args.grid, "rows": []}
    tmpdir = tempfile.mkdtemp(dir=os.environ.get("X1_TMP"))

    for n_series, n_time in grid:
        pdf = make_panel_fast(n_series, n_time, n_cols=2).reset_index()
        path = os.path.join(tmpdir, f"panel_{n_series}_{n_time}.parquet")
        pq.write_table(pa.Table.from_pandas(pdf, preserve_index=False), path,
                       row_group_size=100_000)
        n_rows = len(pdf)
        ref = canon(pandas_route(pdf))
        del pdf
        gc.collect()
        for name, scan in scanners(path, dcon, icon).items():
            if name != args.backend:
                continue
            routes = {
                "R1_collect_all": lambda scan=scan: per_series_stats(
                    nw.from_native(scan()).collect()
                ),
                "R2_stay_lazy": lambda scan=scan: per_series_stats(
                    nw.from_native(scan())
                ).collect(),
                "R2b_stay_lazy_last_order_by": lambda scan=scan: per_series_stats_lastby(
                    nw.from_native(scan())
                ).collect(),
                "R0_pandas": lambda scan=scan: pandas_route(
                    nw.from_native(scan()).collect(backend="pandas").to_native()
                ),
                "R_native": native_route(name, path, dcon, icon),
            }
            times, mems, checks, rows_in_python = {}, {}, {}, {}
            for rname, fn in routes.items():
                try:
                    out, dmem = peak_rss_delta_mb(fn)
                    mems[rname] = round(dmem, 1)
                    got = canon(out)
                    checks[rname] = "match" if got.equals(ref) else (
                        "MISMATCH" if got.shape == ref.shape else f"shape {got.shape}"
                    )
                    if not got.equals(ref) and got.shape == ref.shape:
                        # tolerate float noise in the mean
                        from polars.testing import assert_frame_equal

                        try:
                            assert_frame_equal(got, ref, check_exact=False, rel_tol=1e-9)
                            checks[rname] = "match(approx)"
                        except AssertionError:
                            pass
                    del out, got
                except Exception as e:  # noqa: BLE001
                    mems[rname] = None
                    checks[rname] = f"ERR {type(e).__name__}: {str(e)[:150]}"
            # timing: interleave routes, 2 cycles, min
            for _cyc in range(2):
                for rname, fn in routes.items():
                    if checks[rname].startswith("ERR"):
                        continue
                    t = bench(fn, repeat=args.repeat, warmup=0 if _cyc else 1)
                    times[rname] = min(times.get(rname, float("inf")), t)
            rows_in_python = {
                "R1_collect_all": n_rows,
                "R2_stay_lazy": n_series,
                "R2b_stay_lazy_last_order_by": n_series,
                "R0_pandas": n_rows,
                "R_native": n_series,
            }
            if n_series <= args.iter_max_series:
                keys = (
                    nw.from_native(scan()).select("h0").unique().collect()
                    .get_column("h0").to_list()
                )
                sample = keys[: min(50, len(keys))]

                def iter_filter(scan=scan, sample=sample):
                    lf = nw.from_native(scan())
                    return [lf.filter(c("h0") == k).collect() for k in sample]

                t = bench(iter_filter, repeat=1, warmup=1)
                times["R3_iter_filter_per_group_ms"] = t / len(sample) * 1e3
                times["R3_iter_filter_extrapolated_s"] = t / len(sample) * n_series
            base = times.get("R1_collect_all")
            row = {
                "n_series": n_series,
                "n_time": n_time,
                "n_rows": n_rows,
                "backend": name,
                "time_s": times,
                "speedup_vs_R1": {
                    k: round(base / v, 2)
                    for k, v in times.items()
                    if k in routes and base and v
                },
                "peak_rss_delta_mb": mems,
                "rows_materialised": rows_in_python,
                "check": checks,
            }
            results["rows"].append(row)
            print(f"\n[{n_series}x{n_time}={n_rows}] {name}  (rss {rss_mb():.0f}MB)")
            for rname in routes:
                t = times.get(rname)
                ts = f"{t * 1e3:9.2f} ms" if t else "   -     "
                print(f"  {rname:16s} {ts}  speedup_vs_R1=x{row['speedup_vs_R1'].get(rname)}"
                      f"  peakΔ={mems.get(rname)}MB  rows_in_py={rows_in_python[rname]}"
                      f"  {checks[rname]}")
            if "R3_iter_filter_per_group_ms" in times:
                print(f"  R3_iter_filter   {times['R3_iter_filter_per_group_ms']:.2f} ms/group"
                      f" -> {times['R3_iter_filter_extrapolated_s']:.2f} s for all groups")
        os.remove(path)
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, f"x1_stay_lazy_{args.grid}_{args.tag}_{args.backend}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("\nwrote", out)


def driver(args):
    """one subprocess per backend (bounds peak RSS), then merge the json files."""
    import subprocess

    backends = ["polars", "duckdb", "dask"] + (["ibis"] if ibis is not None else [])
    merged = {"grid": args.grid, "rows": [], "peak_rss_process_mb": {}}
    for b in backends:
        cmd = [sys.executable, "-W", "ignore", os.path.abspath(__file__),
               "--grid", args.grid, "--repeat", str(args.repeat), "--tag", args.tag,
               "--iter-max-series", str(args.iter_max_series), "--backend", b]
        print(f"\n##### backend={b}: {' '.join(cmd[3:])}", flush=True)
        import resource

        subprocess.run(cmd, check=True)
        part = os.path.join(RES, f"x1_stay_lazy_{args.grid}_{args.tag}_{b}.json")
        with open(part) as f:
            d = json.load(f)
        merged["env"] = d["env"]
        merged["rows"].extend(d["rows"])
        os.remove(part)
        merged["peak_rss_process_mb"][b] = round(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1
        )
    out = os.path.join(RES, f"x1_stay_lazy_{args.grid}_{args.tag}.json")
    with open(out, "w") as f:
        json.dump(merged, f, indent=1, default=str)
    print("\nmerged ->", out, "max child RSS so far (MB):", merged["peak_rss_process_mb"])


if __name__ == "__main__":
    sys.exit(main())
