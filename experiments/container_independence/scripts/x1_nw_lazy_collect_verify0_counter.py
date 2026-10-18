"""x1 verify0: hunt for counterexamples to x1 matrix cells via different code paths.

Checks:
  K1 collect() defaults through narwhals.stable.v1 / stable.v2 (the APIs sktime
     would actually pin to) instead of top-level narwhals
  K2 dask "ERR" cells re-tested with npartitions=1 and with read_parquet input
     (are dask errors structural or partition-count dependent?)
  K3 hidden-compute check with a DIFFERENT method: a map_partitions side-effect
     counter on the input graph (instead of dask Callback)
  K4 alternative formulations for ERR cells (tail on duckdb/ibis via
     sort desc+head; ffill on ibis; last_k on dask via semi-join on rank-free
     approach; map_batches on polars lazy via native escape)
  K5 ibis on the polars backend: collect default type
  K6 duckdb default collect when polars is the only eager lib wanted (backend=None)

Run::
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/x1_nw_lazy_collect_verify0_counter.py
"""

import json
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("POLARS_MAX_THREADS", "4")

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import narwhals as nw  # noqa: E402

from common import env_info, make_panel_fast  # noqa: E402

import dask  # noqa: E402
import dask.dataframe as dd  # noqa: E402
import duckdb  # noqa: E402

dask.config.set(scheduler="synchronous")
try:
    import ibis
except ImportError:
    ibis = None

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAG = f"pd{pd.__version__.split('.')[0]}"


def tname(o):
    t = type(o)
    return f"{t.__module__.split('.')[0]}.{t.__name__}"


def err(e):
    return f"{type(e).__name__}: {str(e).splitlines()[0][:140]}" if str(e) else type(e).__name__


pdf = make_panel_fast(20, 30, n_cols=2).reset_index()
arrow = pa.Table.from_pandas(pdf, preserve_index=False)
out = {"env": env_info()}

# ---------------------------------------------------------------- K1 stable APIs
k1 = {}
import narwhals.stable.v1 as nw1  # noqa: E402
import narwhals.stable.v2 as nw2  # noqa: E402

makers = {
    "polars_lazy": lambda: pl.from_arrow(arrow).lazy(),
    "dask": lambda: dd.from_pandas(pdf, npartitions=4),
    "duckdb": lambda: duckdb.from_arrow(arrow),
}
if ibis is not None:
    icon = ibis.duckdb.connect()
    makers["ibis"] = lambda: icon.create_table(f"k1_{len(k1)}", ibis.memtable(arrow))
for api_name, api in [("nw", nw), ("stable.v1", nw1), ("stable.v2", nw2)]:
    for name, mk in makers.items():
        try:
            lf = api.from_native(mk())
            k1[f"{api_name}/{name}"] = f"{type(lf).__name__} -> collect() {tname(lf.collect().to_native())}"
        except Exception as e:  # noqa: BLE001
            k1[f"{api_name}/{name}"] = err(e)
out["K1_stable_api_collect_default"] = k1

# ---------------------------------------------------------------- K2 dask variants
c = nw.col
dask_ops = {
    "tail": lambda lf: lf.tail(5),
    "over_cum_sum(order_by)": lambda lf: lf.with_columns(
        cs=c("c0").cum_sum().over("h0", order_by="time")),
    "group_by_last(order_by)": lambda lf: lf.group_by("h0").agg(c("c0").last(order_by="time")),
    "rank_over": lambda lf: lf.filter(c("time").rank(method="ordinal", descending=True).over("h0") <= 3),
    "filter_time==max.over": lambda lf: lf.filter(c("time") == c("time").max().over("h0")),
    "over_mean(no order)": lambda lf: lf.with_columns(m=c("c0").mean().over("h0")),
}
tmp = tempfile.mkdtemp(dir=os.environ.get("X1_TMP"))
ppath = os.path.join(tmp, "k2.parquet")
pq.write_table(arrow, ppath, row_group_size=200)
dask_inputs = {
    "from_pandas(np=4)": lambda: dd.from_pandas(pdf, npartitions=4),
    "from_pandas(np=1)": lambda: dd.from_pandas(pdf, npartitions=1),
    "read_parquet": lambda: dd.read_parquet(ppath),
    "from_pandas(np=4,sort=False)": lambda: dd.from_pandas(pdf, npartitions=4, sort=False),
}
k2 = {}
for iname, mk in dask_inputs.items():
    for op, fn in dask_ops.items():
        try:
            res = fn(nw.from_native(mk()))
            got = res.collect(backend="polars").to_native()
            k2[f"{iname} / {op}"] = f"ok rows={got.height}"
        except Exception as e:  # noqa: BLE001
            k2[f"{iname} / {op}"] = err(e)
out["K2_dask_partition_variants"] = k2

# ---------------------------------------------------------------- K3 hidden compute (side-effect counter)
from x1_matrix import op_specs  # noqa: E402  reuse op list only

calls = {"n": 0}


def _touch(part):
    calls["n"] += 1
    return part


k3 = {}
specs = op_specs(pdf["h0"].iloc[0])
for op, fn in specs.items():
    base = dd.from_pandas(pdf, npartitions=4)
    instrumented = base.map_partitions(_touch, meta=base._meta)
    calls["n"] = 0
    try:
        res = fn(nw.from_native(instrumented))
        built = calls["n"]
        if isinstance(res, nw.LazyFrame):
            _ = tname(res.to_native())
            k3[op] = {"partition_calls_while_building": built,
                      "after_to_native": calls["n"]}
        else:
            k3[op] = {"partition_calls_while_building": built, "result": tname(res)}
    except Exception as e:  # noqa: BLE001
        k3[op] = {"partition_calls_while_building": calls["n"], "error": err(e)[:80]}
out["K3_dask_side_effect_counter"] = k3

# ---------------------------------------------------------------- K4 alternative formulations
k4 = {}
alts = {
    "tail_via_sort_desc_head": lambda lf: lf.sort("h0", "time", descending=True).head(5),
    "last_k_via_join_on_cutoff_minus_k": lambda lf: lf.join(
        lf.group_by("h0").agg(c("time").max().alias("cut")), on="h0"
    ).filter(c("time") > c("cut") - pd.Timedelta(days=3)).drop("cut"),
    "ffill_via_cum_count_trick(ibis)": lambda lf: lf.with_columns(
        ff=c("c0").fill_null(strategy="forward").over("h0", order_by="time")),
}
backs = {
    "duckdb": lambda: duckdb.from_arrow(arrow),
    "dask": lambda: dd.from_pandas(pdf, npartitions=4),
    "polars_lazy": lambda: pl.from_arrow(arrow).lazy(),
}
if ibis is not None:
    backs["ibis"] = lambda: icon.create_table(f"k4_{len(k4)}", ibis.memtable(arrow))
for b, mk in backs.items():
    for a, fn in alts.items():
        try:
            nat = mk()
            res = fn(nw.from_native(nat))
            same = isinstance(res.to_native(), type(nat))
            got = res.collect(backend="polars").to_native()
            k4[f"{b} / {a}"] = f"ok same_type={same} rows={got.height}"
        except Exception as e:  # noqa: BLE001
            k4[f"{b} / {a}"] = err(e)
# map_batches on polars lazy, via the polars-native escape hatch
try:
    lf = nw.from_native(pl.from_arrow(arrow).lazy())
    nat = lf.to_native().with_columns(mb=pl.col("c0").map_batches(lambda s: s * 2, return_dtype=pl.Float64))
    k4["polars_lazy / map_batches via to_native escape"] = f"ok {tname(nw.from_native(nat).to_native())}"
except Exception as e:  # noqa: BLE001
    k4["polars_lazy / map_batches via to_native escape"] = err(e)
# map_batches through nw stable.v1 (does the older API allow it on LazyFrame?)
for api_name, api in [("stable.v1", nw1), ("stable.v2", nw2)]:
    try:
        lf = api.from_native(pl.from_arrow(arrow).lazy())
        r = lf.with_columns(mb=api.col("c0").map_batches(lambda s: s * 2, return_dtype=api.Float64()))
        k4[f"polars_lazy / map_batches {api_name}"] = f"ok {tname(r.collect().to_native())}"
    except Exception as e:  # noqa: BLE001
        k4[f"polars_lazy / map_batches {api_name}"] = err(e)
out["K4_alternative_formulations"] = k4

# ---------------------------------------------------------------- K5 ibis polars backend
k5 = {}
if ibis is not None:
    try:
        pcon = ibis.polars.connect()
        t = pcon.create_table("k5", pl.from_arrow(arrow))
        lf = nw.from_native(t)
        k5["ibis[polars] collect()"] = tname(lf.collect().to_native())
        k5["ibis[polars] native to_polars"] = tname(t.to_polars())
    except Exception as e:  # noqa: BLE001
        k5["ibis[polars]"] = err(e)
out["K5_ibis_polars_backend"] = k5

os.makedirs(RES, exist_ok=True)
path = os.path.join(RES, f"x1_nw_lazy_collect_verify0_counter_{TAG}.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1, default=str)
for k, v in out.items():
    print(f"\n## {k}")
    if isinstance(v, dict):
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")
print("\nwrote", path)
