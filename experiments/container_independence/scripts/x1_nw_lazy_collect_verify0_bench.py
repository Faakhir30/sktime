"""x1 verify0: independent re-measurement of the x1 stay-lazy / fallback numbers.

Differences vs x1_stay_lazy.py / x1_fallback.py (methodology checks):
  * every (backend, route) runs in a FRESH subprocess -> peak memory is the
    kernel high-water mark (ru_maxrss) of that process, not a 5 ms sampling
    thread in a process whose allocators were already warmed by earlier routes;
  * extra route R1p = collect(backend="polars") + the same eager narwhals
    expression. Separates "staying lazy" from "the default eager target is
    narwhals-on-pyarrow" (x1's R1 for duckdb/ibis is pyarrow-eager);
  * fallback ratios for polars from a scan_parquet source (x1 used an
    in-memory LazyFrame whose collect() is ~free, inflating ratios);
  * results checked against a pandas reference with numeric tolerance.

Run::
  PYTHONPATH=.:experiments/container_independence:experiments/container_independence/scripts \
     $EXP_PY -W ignore experiments/container_independence/scripts/x1_nw_lazy_collect_verify0_bench.py --grid dev
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("POLARS_MAX_THREADS", "4")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

ROUTES = ["R1_collect_default", "R1p_collect_polars", "R2_stay_lazy",
          "R2b_last_order_by", "R0_pandas", "F_native_collect", "F_collect_pandas",
          "F_collect_polars"]
BACKENDS = ["polars", "duckdb", "ibis", "dask"]


def child(backend, route, path, repeat):
    import gc

    import narwhals as nw
    import pandas as pd
    import polars as pl
    import psutil

    import dask
    import dask.dataframe as dd
    import duckdb

    dask.config.set(scheduler="synchronous")
    c = nw.col
    if backend == "polars":
        scan = lambda: pl.scan_parquet(path)  # noqa: E731
        native_collect = lambda: scan().collect()  # noqa: E731
    elif backend == "duckdb":
        con = duckdb.connect()
        con.execute("SET threads=4")
        scan = lambda: con.read_parquet(path)  # noqa: E731
        native_collect = lambda: scan().to_arrow_table()  # noqa: E731
    elif backend == "ibis":
        import ibis

        icon = ibis.duckdb.connect(threads=4)
        scan = lambda: icon.read_parquet(path)  # noqa: E731
        native_collect = lambda: scan().to_pyarrow()  # noqa: E731
    else:
        scan = lambda: dd.read_parquet(path)  # noqa: E731
        native_collect = lambda: scan().compute()  # noqa: E731

    def stats(frame):
        cut = frame.group_by("h0").agg(c("time").max())
        last = frame.join(cut, on=["h0", "time"], how="semi").select(
            "h0", cutoff=c("time"), last=c("c0"))
        st = frame.group_by("h0").agg(mean=c("c0").mean(), n=nw.len())
        return last.join(st, on="h0", how="inner")

    def stats_lastby(frame):
        return frame.group_by("h0").agg(cutoff=c("time").max(),
                                        last=c("c0").last(order_by="time"),
                                        mean=c("c0").mean(), n=nw.len())

    def pandas_route(pdf):
        g = pdf.sort_values(["h0", "time"]).groupby("h0", sort=False)
        return pd.DataFrame({"cutoff": g["time"].max(), "last": g["c0"].last(),
                             "mean": g["c0"].mean(), "n": g.size()}).reset_index()

    fns = {
        "R1_collect_default": lambda: stats(nw.from_native(scan()).collect()),
        "R1p_collect_polars": lambda: stats(nw.from_native(scan()).collect(backend="polars")),
        "R2_stay_lazy": lambda: stats(nw.from_native(scan())).collect(),
        "R2b_last_order_by": lambda: stats_lastby(nw.from_native(scan())).collect(),
        "R0_pandas": lambda: pandas_route(nw.from_native(scan()).collect(backend="pandas").to_native()),
        "F_native_collect": native_collect,
        "F_collect_pandas": lambda: nw.from_native(scan()).collect(backend="pandas"),
        "F_collect_polars": lambda: nw.from_native(scan()).collect(backend="polars"),
    }
    fn = fns[route]
    # touch schema so that connection/metadata setup is not in the measurement
    nw.from_native(scan()).collect_schema()
    gc.collect()
    rss0 = psutil.Process().memory_info().rss / 2**20
    maxrss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    try:
        out = fn()
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"err": f"{type(e).__name__}: {str(e)[:120]}"}))
        return
    maxrss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    # correctness (stats routes only)
    check = None
    if route.startswith("R"):
        if isinstance(out, nw.DataFrame):
            got = out.to_polars()
        elif isinstance(out, pd.DataFrame):
            got = pl.from_pandas(out)
        else:
            got = pl.from_arrow(out) if not isinstance(out, pl.DataFrame) else out
        got = got.select("h0", "cutoff", "last", "mean", "n").with_columns(
            pl.col("cutoff").cast(pl.Datetime("us")), pl.col("n").cast(pl.Int64)).sort("h0")
        ref = pl.read_parquet(path).sort("h0", "time").group_by("h0", maintain_order=True).agg(
            cutoff=pl.col("time").max().cast(pl.Datetime("us")), last=pl.col("c0").last(),
            mean=pl.col("c0").mean(), n=pl.len().cast(pl.Int64)).sort("h0")
        from polars.testing import assert_frame_equal

        try:
            assert_frame_equal(got, ref, check_dtypes=False, rel_tol=1e-9)
            check = "match"
        except AssertionError as e:
            check = "MISMATCH " + str(e).splitlines()[0][:80]
    del out
    gc.collect()
    ts = []
    fn()  # warmup
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    print(json.dumps({"t_min": min(ts), "t_med": sorted(ts)[len(ts) // 2],
                      "peak_delta_mb_maxrss": round(maxrss1 - rss0, 1),
                      "maxrss_grew_mb": round(maxrss1 - maxrss0, 1),
                      "rss_before_mb": round(rss0, 1), "check": check}))


def driver(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from common import DEV_GRID, LARGE_GRID, env_info, make_panel_fast

    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    if args.sizes:
        grid = [g for g in grid if g[0] * g[1] in args.sizes]
    tmp = tempfile.mkdtemp(dir=os.environ.get("X1_TMP"))
    out = {"env": env_info(), "grid": args.grid, "rows": []}
    for n_series, n_time in grid:
        pdf = make_panel_fast(n_series, n_time, n_cols=2).reset_index()
        path = os.path.join(tmp, f"p_{n_series}_{n_time}.parquet")
        pq.write_table(pa.Table.from_pandas(pdf, preserve_index=False), path,
                       row_group_size=100_000)
        del pdf
        for b in args.backends:
            row = {"n_series": n_series, "n_time": n_time, "backend": b, "routes": {}}
            for r in ROUTES:
                cmd = [sys.executable, "-W", "ignore", os.path.abspath(__file__),
                       "--child", b, r, path, "--repeat", str(args.repeat)]
                p = subprocess.run(cmd, capture_output=True, text=True)
                line = [x for x in p.stdout.splitlines() if x.startswith("{")]
                row["routes"][r] = json.loads(line[-1]) if line else {"err": p.stderr[-300:]}
            r1 = row["routes"]["R1_collect_default"].get("t_min")
            fn_ = row["routes"]["F_native_collect"].get("t_min")
            row["speedup_vs_R1"] = {k: round(r1 / v["t_min"], 2) for k, v in row["routes"].items()
                                    if r1 and "t_min" in v and k.startswith("R")}
            row["ratio_to_native_collect"] = {k: round(v["t_min"] / fn_, 2) for k, v in
                                              row["routes"].items() if fn_ and "t_min" in v
                                              and k.startswith("F")}
            out["rows"].append(row)
            print(f"\n[{n_series}x{n_time}] {b}")
            for k, v in row["routes"].items():
                if "t_min" in v:
                    print(f"  {k:20s} {v['t_min']*1e3:8.2f} ms  peakΔ(maxrss)={v['peak_delta_mb_maxrss']:7.1f}MB"
                          f"  {v['check'] or ''}")
                else:
                    print(f"  {k:20s} {v}")
            print("  speedup_vs_R1", row["speedup_vs_R1"])
            print("  ratio_to_native_collect", row["ratio_to_native_collect"])
        os.remove(path)
    import pandas as pd

    tag = f"pd{pd.__version__.split('.')[0]}"
    p = os.path.join(RES, f"x1_nw_lazy_collect_verify0_bench_{args.grid}_{tag}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--sizes", type=int, nargs="*", default=[100_000, 500_000])
    ap.add_argument("--backends", nargs="*", default=BACKENDS)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--child", nargs=3, default=None)
    a = ap.parse_args()
    if a.child:
        child(*a.child, repeat=a.repeat)
    else:
        driver(a)
