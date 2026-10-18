"""x9 (B): what sktime loses by collecting / converting lazy inputs up front.

Data: a parquet file with N_SERIES x N_TIME rows (dev: 10k x 200 = 2M rows),
columns series_id (str), time (timestamp[ns]), y, x1, x2 (float64), sorted by
(series_id, time), written with explicit row groups (100k rows) and statistics.

Task: the slice a forecaster actually needs -- column ``y`` for 1000 series,
last 50 time points each (2.5% of rows, 3 of 5 columns).

Two id selections: "contiguous" (1000 adjacent ids -> parquet min/max stats can
prune row groups) and "random" (1000 random ids -> every row group touched).

Variants
  B0_pandas_read_all     : pd.read_parquet (all) -> set_index -> isin -> groupby.tail
  B1_sktime_eager_all    : pl.read_parquet (all) -> sktime convert_to(pd-multiindex)
                           -> isin -> groupby.tail       (polars user, sktime today)
  B1b_eager_then_filter  : pl.read_parquet (all) -> polars filter -> convert_to
  B2_polars_lazy         : scan_parquet.filter(id in ids & time >= cut).select().collect()
  B2w_polars_lazy_window : scan_parquet.filter(id in ids) -> per-series last 50 via
                           rank().over() (no global time cutoff; unequal lengths ok)
  B3_duckdb              : SQL WHERE id IN (...) AND time >= cut  -> arrow
  B3w_duckdb_window      : SQL WHERE id IN (...) QUALIFY row_number() <= 50 -> arrow
  B4_pyarrow_dataset     : ds.dataset().to_table(columns, filter)
  B5_lazy_then_sktime    : B2 -> sktime convert_to(pd-multiindex)   (a lazy-aware
                           boundary: push selection into the scan, convert the slice)

Time = min of repeats (interleaved rounds, same process); peak RSS delta = fresh
subprocess per variant (result kept alive). Threads pinned to 4 for polars, duckdb
and pyarrow. Page cache is warm (file was just written / read) -- cold-cache I/O
savings are NOT measured.

Run::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x9_lazy_pushdown.py --grid dev
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import warnings

os.environ.setdefault("POLARS_MAX_THREADS", "4")
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from common import bench, env_info, peak_rss_delta_mb, rss_mb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
DATA = os.path.join(RES, "x9_data.parquet")
GRIDS = {"dev": (10_000, 200), "large": (50_000, 200), "tiny": (2_000, 200)}
N_SEL, LAST = 1000, 50
RG_SIZE = 100_000


def write_data(n_series, n_time, path=DATA, seed=0):
    rng = np.random.default_rng(seed)
    n = n_series * n_time
    df = pl.select(pl.int_range(0, n, dtype=pl.Int64).alias("i")).select(
        (pl.lit("s") + (pl.col("i") // n_time).cast(pl.String).str.zfill(7)).alias(
            "series_id"
        ),
        (
            pl.lit(pd.Timestamp("2000-01-01")).cast(pl.Datetime("ns"))
            + pl.duration(days=pl.col("i") % n_time)
        )
        .cast(pl.Datetime("ns"))
        .alias("time"),
    )
    df = df.with_columns(
        y=pl.Series(rng.normal(size=n).cumsum()),
        x1=pl.Series(rng.normal(size=n)),
        x2=pl.Series(rng.normal(size=n)),
    )
    df.write_parquet(path, row_group_size=RG_SIZE, statistics=True)
    return os.path.getsize(path)


def selection(n_series, n_time, kind):
    if kind == "contiguous":
        start = n_series // 5
        ids = [f"s{i:07d}" for i in range(start, start + N_SEL)]
    else:
        rng = np.random.default_rng(42)
        ids = [f"s{i:07d}" for i in sorted(rng.choice(n_series, N_SEL, replace=False))]
    cut = pd.Timestamp("2000-01-01") + pd.Timedelta(days=n_time - LAST)
    return ids, cut


# ----------------------------------------------------------------------------
# variants: each returns the selected slice (some container)
# ----------------------------------------------------------------------------
def _pandas_select(df_mi, ids):
    sel = df_mi[df_mi.index.get_level_values(0).isin(ids)]
    return sel.groupby(level=0, sort=False).tail(LAST)[["y"]]


def B0_pandas_read_all(ids, cut):
    df = pd.read_parquet(DATA).set_index(["series_id", "time"])
    return _pandas_select(df, ids)


def _to_sktime_pl(df):
    return df.rename({"series_id": "__index__series_id", "time": "__index__time"})


def B1_sktime_eager_all(ids, cut):
    from sktime.datatypes import convert_to

    y = convert_to(_to_sktime_pl(pl.read_parquet(DATA)), "pd-multiindex")
    return _pandas_select(y, ids)


def B1b_eager_then_filter(ids, cut):
    from sktime.datatypes import convert_to

    df = pl.read_parquet(DATA)
    df = df.filter(pl.col("series_id").is_in(ids) & (pl.col("time") >= cut)).select(
        "series_id", "time", "y"
    )
    return convert_to(_to_sktime_pl(df), "pd-multiindex")


def _lazy_query(ids, cut):
    return (
        pl.scan_parquet(DATA)
        .filter(pl.col("series_id").is_in(ids) & (pl.col("time") >= cut))
        .select("series_id", "time", "y")
    )


def B2_polars_lazy(ids, cut):
    return _lazy_query(ids, cut).collect()


def _lazy_window(ids):
    return (
        pl.scan_parquet(DATA)
        .filter(pl.col("series_id").is_in(ids))
        .select("series_id", "time", "y")
        .filter(
            pl.col("time").rank("ordinal", descending=True).over("series_id") <= LAST
        )
    )


def B2w_polars_lazy_window(ids, cut):
    return _lazy_window(ids).collect()


_DUCK = {}


def _duck():
    if "con" not in _DUCK:
        import duckdb

        con = duckdb.connect()
        con.execute("SET threads TO 4")
        _DUCK["con"] = con
    return _DUCK["con"]


def _sql_ids(ids):
    return ",".join(f"'{i}'" for i in ids)


def _duck_sql(ids, cut):
    return (
        f"SELECT series_id, time, y FROM read_parquet('{DATA}') "
        f"WHERE series_id IN ({_sql_ids(ids)}) AND time >= TIMESTAMP '{cut}'"
    )


def _duck_sql_window(ids):
    return (
        f"SELECT series_id, time, y FROM read_parquet('{DATA}') "
        f"WHERE series_id IN ({_sql_ids(ids)}) "
        f"QUALIFY row_number() OVER (PARTITION BY series_id ORDER BY time DESC) <= {LAST}"
    )


def _fetch_arrow(rel):
    # duckdb >= 1.4: .arrow() returns a RecordBatchReader; .fetch_arrow_table() a Table
    if hasattr(rel, "fetch_arrow_table"):
        return rel.fetch_arrow_table()
    return rel.arrow()


def B3_duckdb(ids, cut):
    return _fetch_arrow(_duck().sql(_duck_sql(ids, cut)))


def B3w_duckdb_window(ids, cut):
    return _fetch_arrow(_duck().sql(_duck_sql_window(ids)))


def B4_pyarrow_dataset(ids, cut):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dset = ds.dataset(DATA, format="parquet")
    filt = pc.field("series_id").isin(pa.array(ids)) & (
        pc.field("time") >= pa.scalar(cut.to_datetime64(), type=pa.timestamp("ns"))
    )
    return dset.to_table(columns=["series_id", "time", "y"], filter=filt)


def B5_lazy_then_sktime(ids, cut):
    from sktime.datatypes import convert_to

    return convert_to(_to_sktime_pl(_lazy_query(ids, cut).collect()), "pd-multiindex")


VARIANTS = {
    "B0_pandas_read_all": B0_pandas_read_all,
    "B1_sktime_eager_all": B1_sktime_eager_all,
    "B1b_eager_then_filter": B1b_eager_then_filter,
    "B2_polars_lazy": B2_polars_lazy,
    "B2w_polars_lazy_window": B2w_polars_lazy_window,
    "B3_duckdb": B3_duckdb,
    "B3w_duckdb_window": B3w_duckdb_window,
    "B4_pyarrow_dataset": B4_pyarrow_dataset,
    "B5_lazy_then_sktime": B5_lazy_then_sktime,
}


def _canon(res):
    """Canonical (n_rows, sum y, sorted first/last key) for cross-variant checks."""
    if isinstance(res, pd.DataFrame):
        flat = res.reset_index()
        flat.columns = ["series_id", "time", "y"]
        p = pl.from_pandas(flat)
    elif isinstance(res, pl.DataFrame):
        p = res
    else:  # pyarrow table
        p = pl.from_arrow(res)
    p = p.sort("series_id", "time")
    return {
        "rows": p.height,
        "sum_y": round(float(p["y"].sum()), 6),
        "n_ids": p["series_id"].n_unique(),
        "first": [str(x) for x in p.row(0)],
        "last": [str(x) for x in p.row(-1)],
    }


def _setup_threads():
    try:
        import pyarrow as pa

        pa.set_cpu_count(4)
        pa.set_io_thread_count(4)
    except Exception:  # noqa: BLE001
        pass


def mem_worker(variant, n_series, n_time, kind):
    _setup_threads()
    ids, cut = selection(n_series, n_time, kind)
    fn = VARIANTS[variant]
    # import cost out of the measurement: import modules used by the variant
    import pyarrow.dataset  # noqa: F401

    from sktime.datatypes import convert_to  # noqa: F401

    _duck() if variant.startswith("B3") else None
    gc.collect()
    rss0 = rss_mb()
    res, delta = peak_rss_delta_mb(lambda: fn(ids, cut))
    print(json.dumps({"variant": variant, "peak_delta_mb": delta, "rss0_mb": rss0}))
    del res


def run_mem(variant, n_series, n_time, kind):
    cmd = [
        sys.executable, "-W", "ignore", os.path.abspath(__file__),
        "--mem-worker", variant, str(n_series), str(n_time), kind,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    for line in p.stdout.splitlines()[::-1]:
        if line.startswith("{"):
            return json.loads(line)
    return {"variant": variant, "error": p.stderr[-800:]}


def plans(ids, cut):
    out = {}
    out["polars_lazy_plan"] = _lazy_query(ids, cut).explain()
    out["polars_lazy_window_plan"] = _lazy_window(ids).explain()
    rel = _duck().sql("EXPLAIN " + _duck_sql(ids[:3], cut))
    out["duckdb_plan(3 ids shown)"] = rel.fetchall()[0][1]
    import pyarrow.parquet as pq

    md = pq.ParquetFile(DATA).metadata
    out["parquet"] = {
        "row_groups": md.num_row_groups,
        "rows": md.num_rows,
        "columns": md.num_columns,
        "file_mb": os.path.getsize(DATA) / 2**20,
    }
    # row groups that min/max stats say can contain the selected ids
    col = md.schema.names.index("series_id")
    lo, hi = min(ids), max(ids)
    keep = 0
    for i in range(md.num_row_groups):
        st = md.row_group(i).column(col).statistics
        if not (st.max < lo or st.min > hi):
            keep += 1
    out["row_groups_overlapping_id_range"] = keep
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large", "tiny"], default="dev")
    ap.add_argument("--mem-worker", nargs=4, default=None)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--keep-data", action="store_true")
    ap.add_argument("--skip", default="", help="comma-separated variants to skip")
    args = ap.parse_args()

    if args.mem_worker:
        v, a, b, kind = args.mem_worker
        mem_worker(v, int(a), int(b), kind)
        return

    _setup_threads()
    n_series, n_time = GRIDS[args.grid]
    skip = set(filter(None, args.skip.split(",")))
    out = {
        "env": env_info(), "python": sys.version.split()[0], "grid": args.grid,
        "n_series": n_series, "n_time": n_time, "n_sel": N_SEL, "last": LAST,
        "row_group_size": RG_SIZE, "threads": 4,
    }
    try:
        import duckdb

        out["env"]["duckdb"] = duckdb.__version__
    except Exception:  # noqa: BLE001
        pass
    size = write_data(n_series, n_time)
    out["file_bytes"] = size
    print(f"wrote {DATA}: {size/2**20:.1f} MB, {n_series*n_time} rows")

    try:
        for kind in ["contiguous", "random"]:
            ids, cut = selection(n_series, n_time, kind)
            rec = {"plans": plans(ids, cut)}
            print(f"\n######## selection = {kind}")
            print("row groups overlapping id range:",
                  rec["plans"]["row_groups_overlapping_id_range"], "/",
                  rec["plans"]["parquet"]["row_groups"])
            print(rec["plans"]["polars_lazy_plan"])
            # correctness: all variants must agree
            canon = {}
            for v, fn in VARIANTS.items():
                if v in skip:
                    continue
                canon[v] = _canon(fn(ids, cut))
            ref = canon["B2_polars_lazy"]
            rec["canon"] = canon
            rec["all_agree"] = {
                v: (c["rows"] == ref["rows"] and abs(c["sum_y"] - ref["sum_y"]) < 1e-6
                    and c["first"] == ref["first"] and c["last"] == ref["last"])
                for v, c in canon.items()
            }
            print("result rows:", ref["rows"], "ids:", ref["n_ids"],
                  "agree:", rec["all_agree"])
            # timing, interleaved
            best = {v: float("inf") for v in VARIANTS if v not in skip}
            for _ in range(args.rounds):
                for v in best:
                    best[v] = min(
                        best[v],
                        bench(lambda v=v: VARIANTS[v](ids, cut), repeat=3, warmup=1),
                    )
            gc.collect()
            mem = {v: run_mem(v, n_series, n_time, kind) for v in best}
            ref_t = best["B1_sktime_eager_all"]
            rec["time_s"] = best
            rec["mem"] = mem
            rec["speedup_vs_B1"] = {v: ref_t / t for v, t in best.items()}
            for v in best:
                pk = mem[v].get("peak_delta_mb")
                print(
                    f"  {v:26s} {best[v]*1e3:9.1f} ms  x{ref_t/best[v]:7.1f} vs B1"
                    f"  peakRSSdelta={'ERR ' + mem[v].get('error','')[-200:] if pk is None else f'{pk:7.1f} MB'}"
                )
            out[kind] = rec
    finally:
        if not args.keep_data and os.path.exists(DATA):
            os.remove(DATA)
            print("deleted", DATA)

    tag = "pd3" if pd.__version__.startswith("3") else "pd2"
    path = os.path.join(RES, f"x9_lazy_pushdown_{args.grid}_{tag}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
