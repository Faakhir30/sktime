"""x1 verify0: per-group iteration — lazy filter loop vs collect-once + eager iteration.

x1 claim 11 compares a lazy per-key filter+collect loop against ONE grouped
aggregation query. The realistic alternative for arbitrary per-series
estimators is "collect once, iterate groups eagerly, take numpy per group".
This measures all three on the same parquet input (all 1000 keys, no
extrapolation), each with warmup and min-of-repeats.

Run::
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
     experiments/container_independence/scripts/x1_nw_lazy_collect_verify0_iter.py --grid dev
"""

import argparse
import json
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("POLARS_MAX_THREADS", "4")

import narwhals as nw  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import duckdb  # noqa: E402

from common import bench, env_info, make_panel_fast  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
c = nw.col

ap = argparse.ArgumentParser()
ap.add_argument("--grid", choices=["dev", "large"], default="dev")
args = ap.parse_args()
sizes = [(1000, 100)] if args.grid == "dev" else [(1000, 500), (10000, 100)]
tmp = tempfile.mkdtemp(dir=os.environ.get("X1_TMP"))
con = duckdb.connect()
con.execute("SET threads=4")
out = {"env": env_info(), "rows": []}
for n_series, n_time in sizes:
    pdf = make_panel_fast(n_series, n_time, n_cols=2).reset_index()
    path = os.path.join(tmp, "it.parquet")
    pq.write_table(pa.Table.from_pandas(pdf, preserve_index=False), path, row_group_size=100_000)
    keys = sorted(pdf["h0"].unique().tolist())
    del pdf
    scans = {"polars": lambda: pl.scan_parquet(path), "duckdb": lambda: con.read_parquet(path)}
    for b, scan in scans.items():
        def lazy_loop():
            lf = nw.from_native(scan())
            return sum(lf.filter(c("h0") == k).collect().select("c0").to_numpy().sum() for k in keys)

        def eager_iter(backend):
            def f():
                df = nw.from_native(scan()).collect(backend=backend)
                return sum(g.select("c0").to_numpy().sum() for _, g in df.group_by("h0"))
            return f

        def grouped_query():
            return nw.from_native(scan()).group_by("h0").agg(c("c0").sum()).collect()

        t_loop = bench(lazy_loop, repeat=1, warmup=0)
        t_eager_pl = bench(eager_iter("polars"), repeat=3)
        t_eager_def = bench(eager_iter(None), repeat=3)
        t_grp = bench(grouped_query, repeat=3)
        row = {"backend": b, "n_series": n_series, "n_time": n_time,
               "lazy_filter_loop_s": t_loop, "collect_polars_iter_s": t_eager_pl,
               "collect_default_iter_s": t_eager_def, "grouped_query_s": t_grp,
               "loop_over_eager_iter_polars": round(t_loop / t_eager_pl, 1),
               "loop_over_grouped": round(t_loop / t_grp, 1),
               "eager_iter_polars_over_grouped": round(t_eager_pl / t_grp, 1)}
        out["rows"].append(row)
        print(json.dumps(row))
with open(os.path.join(RES, f"x1_nw_lazy_collect_verify0_iter_{args.grid}_pd{pd.__version__[0]}.json"), "w") as f:
    json.dump(out, f, indent=1)
