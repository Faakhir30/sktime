"""design_engine_lazy: lookback pushdown + guarded engine-native kernel, fresh process per variant.

Workload: panel-native NaiveForecaster(strategy="mean", window_length=w) and ("last"),
fh=1..H, on a parquet panel (key, time, y, 4 unused float cols).
Every variant returns the same long result (key, time, y_pred) as a polars DataFrame
(collected), and results are checked equal to the eager reference.

Kernels are ORDER-ROBUST ("guarded"): they sort by time within key inside the plan
(sort_by inside agg / ORDER BY in window), so shuffled input gives correct results.
A separate in-process mode measures the guard cost (positional vs order-robust) and
correctness on shuffled rows.

Usage:
  python design_engine_lazy_pushdown.py --make N T            # write parquet files
  python design_engine_lazy_pushdown.py --run VARIANT STRAT W  # one fresh-process cell (prints json)
  python design_engine_lazy_pushdown.py --sweep N T            # orchestrate fresh processes
  python design_engine_lazy_pushdown.py --guard N T            # in-process guard cost + shuffled correctness
"""

import json
import os
import resource
import subprocess
import sys
import time

SCR = os.environ.get(
    "DEL_SCRATCH",
    "/tmp/claude-0/-home-user-sktime/c9329e73-89f5-514d-ae5a-9180bf0e4a71/scratchpad/design_engine_lazy",
)
H = 14


def path(N, T, layout):
    return f"{SCR}/panel_{N}x{T}_{layout}.parquet"


def maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def make(N, T):
    import numpy as np
    import polars as pl

    os.makedirs(SCR, exist_ok=True)
    rng = np.random.default_rng(0)
    keys = np.repeat(np.array([f"s{i:06d}" for i in range(N)]), T)
    t0 = np.datetime64("2000-01-01", "ns")
    times = np.tile(t0 + np.arange(T).astype("timedelta64[D]"), N)
    y = np.cumsum(rng.normal(size=(N, T)), axis=1).ravel() + 100
    df = pl.DataFrame({"key": keys, "time": times, "y": y})
    for j in range(4):
        df = df.with_columns(pl.Series(f"x{j}", rng.normal(size=N * T)))
    df.write_parquet(path(N, T, "series_major"), row_group_size=500_000)
    df.sort(["time", "key"]).write_parquet(path(N, T, "time_major"), row_group_size=500_000)
    print("written", df.shape, os.path.getsize(path(N, T, "series_major")) / 2**20, "MB")


# ---------------------------------------------------------------- kernels (polars, order-robust)
def pl_kernel(lf_or_df, strat, w, pl):
    """Guarded Naive panel kernel: params per key, then cross join with horizon."""
    if strat == "last":
        agg = pl.col("y").sort_by("time").last().alias("p")
    else:
        agg = pl.col("y").sort_by("time").tail(w).mean().alias("p")
    params = lf_or_df.group_by("key").agg(agg, pl.col("time").max().alias("cutoff"))
    hz = pl.DataFrame({"h": list(range(1, H + 1))}, schema={"h": pl.Int64})
    if isinstance(params, pl.LazyFrame):
        hz = hz.lazy()
    out = params.join(hz, how="cross").select(
        "key",
        (pl.col("cutoff") + pl.duration(days=pl.col("h"))).alias("time"),
        pl.col("p").alias("y_pred"),
    )
    return out


def pl_kernel_positional(df, strat, w, pl):
    """UNGUARDED: relies on row order (x7 style)."""
    if strat == "last":
        agg = pl.col("y").last().alias("p")
    else:
        agg = pl.col("y").tail(w).mean().alias("p")
    params = df.group_by("key", maintain_order=True).agg(agg, pl.col("time").last().alias("cutoff"))
    hz = pl.DataFrame({"h": list(range(1, H + 1))}, schema={"h": pl.Int64})
    return params.join(hz, how="cross").select(
        "key", (pl.col("cutoff") + pl.duration(days=pl.col("h"))).alias("time"), pl.col("p").alias("y_pred")
    )


def tail_pushdown(lf, w, pl):
    """lookback selection: keep last w rows per key by time (order-robust)."""
    return lf.filter(pl.col("time").rank("ordinal", descending=True).over("key") <= w)


def run(variant, strat, w, N, T, layout):
    import duckdb  # noqa: F401  pre-import everything so imports are not timed (x7 verifier R6)
    import numpy as np
    import pandas as pd  # noqa: F401
    import polars as pl
    import pyarrow.parquet  # noqa: F401

    base = maxrss_mb()
    p = path(N, T, layout)
    t0 = time.perf_counter()
    if variant == "pl_eager_all":
        df = pl.read_parquet(p)
        out = pl_kernel(df, strat, w, pl)
    elif variant == "pl_eager_proj":
        df = pl.read_parquet(p, columns=["key", "time", "y"])
        out = pl_kernel(df, strat, w, pl)
    elif variant == "pl_eager_proj_select":  # selection-aware EAGER boundary (x9 verifier's alternative)
        df = pl.read_parquet(p, columns=["key", "time", "y"])
        df = tail_pushdown(df.lazy(), w if strat == "mean" else 1, pl).collect()
        out = pl_kernel(df, strat, w, pl)
    elif variant == "pl_lazy_plan":  # whole kernel in one lazy plan, no explicit lookback
        out = pl_kernel(pl.scan_parquet(p), strat, w, pl).collect()
    elif variant == "pl_lazy_lookback":  # lookback pushed down, then kernel, one plan
        lf = tail_pushdown(pl.scan_parquet(p), w if strat == "mean" else 1, pl)
        out = pl_kernel(lf, strat, w, pl).collect()
    elif variant == "pl_lazy_streaming":
        out = pl_kernel(pl.scan_parquet(p), strat, w, pl).collect(engine="streaming")
    elif variant == "duckdb_sql":
        import duckdb

        ww = w if strat == "mean" else 1
        con = duckdb.connect()
        con.execute("SET threads=4")
        q = f"""
        WITH sel AS (
          SELECT key, time, y FROM read_parquet('{p}')
          QUALIFY row_number() OVER (PARTITION BY key ORDER BY time DESC) <= {ww}),
        par AS (SELECT key, avg(y) AS p, max(time) AS cutoff FROM sel GROUP BY key)
        SELECT key, cutoff + h * INTERVAL 1 DAY AS time, p AS y_pred
        FROM par CROSS JOIN range(1, {H + 1}) t(h)
        """
        out = pl.from_arrow(con.execute(q).arrow())
        out = out.with_columns(pl.col("time").cast(pl.Datetime("ns")))
    elif variant == "pandas_eager_groupby":
        import pandas as pd

        df = pd.read_parquet(p, columns=["key", "time", "y"])
        df = df.sort_values(["key", "time"])
        g = df.groupby("key", sort=False)
        par = g["y"].last() if strat == "last" else g.tail(w).groupby("key", sort=False)["y"].mean()
        cut = g["time"].max()
        k = np.repeat(par.index.values, H)
        hh = np.tile(np.arange(1, H + 1), len(par))
        tt = np.repeat(cut.values, H) + hh.astype("timedelta64[D]")
        out = pl.DataFrame({"key": k, "time": tt, "y_pred": np.repeat(par.values, H)})
    else:
        raise ValueError(variant)
    wall = time.perf_counter() - t0
    peak = maxrss_mb()
    out = out.sort(["key", "time"])
    chk = float(out["y_pred"].sum())
    print(json.dumps({"variant": variant, "strat": strat, "w": w, "N": N, "T": T, "layout": layout,
                      "wall_s": wall, "rss_base_mb": base, "rss_peak_delta_mb": peak - base,
                      "rows_out": out.height, "checksum": chk}))


def sweep(N, T, reps=2):
    variants = ["pl_eager_all", "pl_eager_proj", "pl_eager_proj_select", "pl_lazy_plan",
                "pl_lazy_lookback", "pl_lazy_streaming", "duckdb_sql", "pandas_eager_groupby"]
    res = []
    for layout in ["series_major", "time_major"]:
        for strat, w in [("last", 1), ("mean", 10)]:
            for r in range(reps):
                for v in variants:
                    cmd = [sys.executable, "-W", "ignore", __file__, "--run", v, strat, str(w), str(N), str(T), layout]
                    o = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    line = [l for l in o.stdout.splitlines() if l.startswith("{")]
                    if not line:
                        print("FAIL", v, o.stderr[-500:], flush=True)
                        continue
                    d = json.loads(line[-1])
                    d["rep"] = r
                    res.append(d)
                    print(json.dumps(d), flush=True)
    out = f"experiments/container_independence/results/design_engine_lazy_pushdown_{N}x{T}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


def guard(N, T):
    import numpy as np
    import polars as pl

    sys.path.insert(0, "experiments/container_independence")
    from common import bench

    df = pl.read_parquet(path(N, T, "series_major"), columns=["key", "time", "y"])
    shuf = df.sample(fraction=1.0, shuffle=True, seed=1)
    res = {}
    for strat, w in [("last", 1), ("mean", 10)]:
        ref = pl_kernel(df, strat, w, pl).sort(["key", "time"])
        t_pos = bench(lambda: pl_kernel_positional(df, strat, w, pl), repeat=3)
        t_rob = bench(lambda: pl_kernel(df, strat, w, pl), repeat=3)
        t_chk = bench(lambda: df.select(((pl.col("time").diff().over("key") > pl.duration(nanoseconds=0)) | pl.col("time").diff().over("key").is_null()).all() & pl.col("time").is_not_null().all()), repeat=3)
        t_pos2 = bench(lambda: pl_kernel_positional(df, strat, w, pl), repeat=3)
        pos_sh = pl_kernel_positional(shuf, strat, w, pl).sort(["key", "time"])
        rob_sh = pl_kernel(shuf, strat, w, pl).sort(["key", "time"])
        pos_ok = pos_sh.equals(ref)
        rob_ok = rob_sh.equals(ref)
        res[strat] = {"t_positional": min(t_pos, t_pos2), "t_order_robust": t_rob, "t_sorted_check": t_chk,
                      "robust_over_positional": t_rob / min(t_pos, t_pos2),
                      "shuffled_positional_correct": pos_ok, "shuffled_robust_correct": rob_ok}
        print(strat, json.dumps(res[strat]), flush=True)
    with open(f"experiments/container_independence/results/design_engine_lazy_guard_{N}x{T}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "--make":
        make(int(a[2]), int(a[3]))
    elif a[1] == "--run":
        run(a[2], a[3], int(a[4]), int(a[5]), int(a[6]), a[7])
    elif a[1] == "--sweep":
        sweep(int(a[2]), int(a[3]))
    elif a[1] == "--guard":
        guard(int(a[2]), int(a[3]))
