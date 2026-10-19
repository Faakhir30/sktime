"""design_narwhals: cost/feasibility of a narwhals (stable.v2) boundary layer.

Measures, for the narwhals-first architecture (reports/design_narwhals.md):
  B1  top-level panel validation with portable narwhals expressions (null keys, contiguity,
      strictly increasing time within key, duplicate (key,time)) vs stock sktime
      check_is_mtype(polars_panel) and check_is_mtype(pd-multiindex)
  B2  per-instance cutoff (max AND min time per key) as a narwhals group_by, + unique cutoffs
  B3  top-level X/y time-containment check (per-key min/max join) -- g5 invariant 4
  B4  lookback pushdown via semi-join on (key, max(time) - w*step) on lazy backends
      (polars scan_parquet, duckdb, dask read_parquet): type preserved?, rows, time vs collect-all
  B5  explicit materialisation policy: pyarrow -> polars rerouting cost (avoid nw pyarrow backend)
  API checks under narwhals.stable.v2 for the ops the design relies on.

Usage (from WT root):
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/design_narwhals_boundary.py --grid dev
Writes results/design_narwhals_boundary_<grid>.json
"""

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import bench, make_panel_fast  # noqa: E402

import narwhals.stable.v2 as nw  # noqa: E402
import polars as pl  # noqa: E402

GRIDS = {"smoke": [(100, 50)], "dev": [(1000, 100), (10000, 100)], "large": [(10000, 100), (50000, 100)]}
KEY, TIME, VAL = "h0", "time", "c0"


# ---------------------------------------------------------------- narwhals boundary ops
def nw_validate_eager(df_nw):
    """Portable eager validation: returns dict of counts. Positional shift = row order."""
    k, t = nw.col(KEY), nw.col(TIME)
    new_run = (k != k.shift(1)).fill_null(True)
    out = df_nw.select(
        n_rows=nw.len(),
        null_keys=k.is_null().sum(),
        null_time=t.is_null().sum(),
        n_runs=new_run.sum(),
        n_keys=k.n_unique(),
        bad_order=((~new_run) & ~(t > t.shift(1)).fill_null(True)).sum(),
    )
    r = out.to_dict(as_series=False)
    r = {a: int(b[0]) for a, b in r.items()}
    r["contiguous"] = r["n_runs"] == r["n_keys"]
    r["sorted_strict"] = r["bad_order"] == 0  # also excludes duplicate (key,time) inside a run
    return r


def nw_cutoffs(frame_nw):
    """Per-key max and min time (reverse_order needs min) -- works eager & lazy."""
    return frame_nw.group_by(KEY).agg(nw.col(TIME).max().alias("cut_max"), nw.col(TIME).min().alias("cut_min"))


def nw_containment(y_nw, X_nw):
    """g5 invariant 4: per key, X time range covers y time range. Returns #violating keys."""
    ya = y_nw.group_by(KEY).agg(nw.col(TIME).min().alias("ymin"), nw.col(TIME).max().alias("ymax"))
    xa = X_nw.group_by(KEY).agg(nw.col(TIME).min().alias("xmin"), nw.col(TIME).max().alias("xmax"))
    j = ya.join(xa, on=KEY, how="left")
    bad = j.filter(nw.col("xmin").is_null() | (nw.col("xmin") > nw.col("ymin")) | (nw.col("xmax") < nw.col("ymax")))
    return bad.select(nw.len())


def nw_lookback(lf_nw, w, step):
    """Keep the last w points per key: semi-join on max(time) - (w-1)*step (no over/rank)."""
    last = lf_nw.group_by(KEY).agg(nw.col(TIME).max().alias("__tmax"))
    j = lf_nw.join(last, on=KEY, how="inner")
    return j.filter(nw.col(TIME) > nw.col("__tmax") - step * (w)).drop("__tmax")


def nw_lookback_window(lf_nw, w, step):
    """Same selection via a non-ordered window max().over(key) (no join)."""
    return lf_nw.filter(nw.col(TIME) > nw.col(TIME).max().over(KEY) - step * (w))


# ---------------------------------------------------------------- helpers
def to_native_rows(obj):
    nat = nw.to_native(obj)
    if hasattr(nat, "collect") and isinstance(nat, pl.LazyFrame):
        return nat.collect().height
    if type(nat).__module__.startswith("duckdb"):
        return nat.aggregate("count(*)").fetchone()[0]
    if type(nat).__module__.startswith("dask"):
        return len(nat.compute())
    return len(nat)


def run_size(n, T, tmpdir, res):
    row = {"n": n, "T": T}
    pdf = make_panel_fast(n, T)  # pd-multiindex h0,time
    flat = pdf.reset_index()
    pl_long = pl.from_pandas(flat)
    nwp = nw.from_native(pl_long, eager_only=True)
    nwpd = nw.from_native(flat, eager_only=True)

    # ---- B1 validation
    from sktime.datatypes import check_is_mtype, convert_to

    sk_pl = convert_to(pdf, "polars_panel")
    t_sk_pl = bench(lambda: check_is_mtype(sk_pl, "polars_panel", scitype="Panel"), repeat=2)
    t_sk_pd = bench(lambda: check_is_mtype(pdf, "pd-multiindex", scitype="Panel"), repeat=2)
    t_nw_pl = bench(lambda: nw_validate_eager(nwp), repeat=5)
    t_nw_pd = bench(lambda: nw_validate_eager(nwpd), repeat=5)
    v = nw_validate_eager(nwp)
    # adversarial: shuffled and duplicated rows must be flagged
    shuf = nw.from_native(pl_long.sample(fraction=1.0, shuffle=True, seed=1), eager_only=True)
    dup = nw.from_native(pl.concat([pl_long, pl_long.head(1)]).sort([KEY, TIME]), eager_only=True)
    nullk = nw.from_native(pl_long.with_columns(pl.when(pl.int_range(pl.len()) == 5).then(None).otherwise(pl.col(KEY)).alias(KEY)), eager_only=True)
    row["B1"] = {
        "sk_check_polars_panel_s": t_sk_pl, "sk_check_pd_multiindex_s": t_sk_pd,
        "nw_validate_polars_s": t_nw_pl, "nw_validate_pandas_long_s": t_nw_pd,
        "ratio_nwpl_vs_skpl": t_nw_pl / t_sk_pl, "ratio_nwpd_vs_skpd": t_nw_pd / t_sk_pd,
        "clean": v,
        "flags_shuffled": nw_validate_eager(shuf),
        "flags_dup": nw_validate_eager(dup),
        "flags_nullkey": nw_validate_eager(nullk),
    }

    # ---- B2 cutoffs (polars eager, pandas long) + tiny pandas Index for public API
    def cut_pl():
        c = nw_cutoffs(nwp)
        u = c.select(nw.col("cut_max").unique())
        return pd.DatetimeIndex(u.to_native().to_series().to_list())
    def cut_pd():
        c = nw_cutoffs(nwpd)
        return pd.DatetimeIndex(c.to_native()["cut_max"].unique())
    def cut_stock():  # what get_cutoff-like per-instance work would do: groupby max on pandas MI
        return pdf.index.to_frame(index=False).groupby(KEY, sort=False)[TIME].max()
    row["B2"] = {"nw_polars_s": bench(cut_pl), "nw_pandas_long_s": bench(cut_pd), "pandas_mi_groupby_max_s": bench(cut_stock)}

    # ---- B3 containment
    X_ok = pl_long.rename({VAL: "x0"})
    X_bad = X_ok.filter(pl.col(TIME) < pl.col(TIME).max())  # misses last date for every key
    nX_ok, nX_bad = nw.from_native(X_ok, eager_only=True), nw.from_native(X_bad, eager_only=True)
    row["B3"] = {
        "nw_polars_s": bench(lambda: nw_containment(nwp, nX_ok)),
        "violations_ok": int(nw_containment(nwp, nX_ok).item()),
        "violations_bad": int(nw_containment(nwp, nX_bad).item()),
        "nw_lazy_polars_s": bench(lambda: nw_containment(nwp.lazy(), nX_ok.lazy()).collect()),
    }

    # ---- B4 lookback pushdown on lazy backends
    pq = os.path.join(tmpdir, f"p_{n}_{T}.parquet")
    pl_long.write_parquet(pq)
    w, step = 10, dt.timedelta(days=1)
    expected = n * w
    b4 = {}
    srcs = {}
    srcs["polars_lazy"] = lambda: pl.scan_parquet(pq)
    try:
        import duckdb
        con = duckdb.connect()
        srcs["duckdb"] = lambda: duckdb.read_parquet(pq)
    except Exception:  # pragma: no cover
        pass
    try:
        import dask.dataframe as dd
        srcs["dask"] = lambda: dd.read_parquet(pq)
    except Exception:  # pragma: no cover
        pass
    for name, mk in srcs.items():
        rec = {}
        try:
            lf = nw.from_native(mk())
            rec["from_native_type"] = type(lf).__name__
            sel = nw_lookback(lf, w, step)
            rec["native_type_after_ops"] = type(nw.to_native(sel)).__name__
            target = "polars"
            t_push = bench(lambda: nw_lookback(nw.from_native(mk()), w, step).collect(backend=target), repeat=3)
            t_all = bench(lambda: nw.from_native(mk()).collect(backend=target), repeat=3)
            got = nw_lookback(nw.from_native(mk()), w, step).collect(backend=target)
            rec.update(pushdown_collect_s=t_push, collect_all_s=t_all, ratio_all_over_push=t_all / t_push,
                       rows=len(got), rows_expected=expected, ok=len(got) == expected)
            t_win = bench(lambda: nw_lookback_window(nw.from_native(mk()), w, step).collect(backend=target), repeat=3)
            gw = nw_lookback_window(nw.from_native(mk()), w, step).collect(backend=target)
            # downstream per-instance forecasters need pandas: charge the to_pandas of what was collected
            t_push_pd = bench(lambda: nw_lookback(nw.from_native(mk()), w, step).collect(backend="pandas"), repeat=3)
            t_all_pd = bench(lambda: nw.from_native(mk()).collect(backend="pandas"), repeat=3)
            rec.update(window_collect_s=t_win, window_rows=len(gw), push_to_pandas_s=t_push_pd,
                       all_to_pandas_s=t_all_pd, ratio_all_over_push_pandas=t_all_pd / t_push_pd)
            # cutoffs lazily (no collect of values)
            rec["lazy_cutoffs_s"] = bench(lambda: nw_cutoffs(nw.from_native(mk())).collect(backend="polars"), repeat=3)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            rec["tb"] = traceback.format_exc()[-600:]
        b4[name] = rec
    row["B4"] = b4

    # ---- B5 pyarrow -> polars routing (zero-copy-ish) vs pyarrow -> pandas
    tbl = pl_long.to_arrow()
    row["B5"] = {
        "pa_to_polars_s": bench(lambda: pl.from_arrow(tbl)),
        "pa_to_pandas_s": bench(lambda: tbl.to_pandas()),
        "nw_validate_on_rerouted_polars_s": bench(lambda: nw_validate_eager(nw.from_native(pl.from_arrow(tbl), eager_only=True))),
    }
    res["rows"].append(row)
    print(json.dumps({k: v for k, v in row.items()}, default=str)[:3000], flush=True)


def api_checks():
    """Ops the design relies on, under narwhals.stable.v2."""
    out = {}
    lf = nw.from_native(pl.LazyFrame({KEY: ["a", "a", "b"], TIME: [1, 2, 1], VAL: [1.0, 2.0, 3.0]}))
    def chk(name, f):
        try:
            r = f()
            out[name] = "ok" if r is None else f"ok: {r}"
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}: {str(e)[:120]}"
    chk("with_row_index(order_by)", lambda: lf.with_row_index("r", order_by=[KEY, TIME]).collect().shape)
    chk("last(order_by) expr", lambda: lf.group_by(KEY).agg(nw.col(VAL).last(order_by=TIME)).collect().shape)
    chk("collect(backend=polars)", lambda: type(lf.collect(backend="polars").to_native()).__name__)
    chk("collect(backend=pandas)", lambda: type(lf.collect(backend="pandas").to_native()).__name__)
    try:
        import duckdb
        rel = duckdb.sql("select 'a' as h0, 1 as time, 1.0 as c0")
        o = nw.from_native(rel)
        out["duckdb_from_native_type"] = type(o).__name__
        chk("duckdb collect default", lambda: type(o.collect().to_native()).__name__)
        chk("duckdb lazy(backend) relift", lambda: type(o.collect().lazy(backend="duckdb").to_native()).__name__)
    except Exception as e:
        out["duckdb"] = f"ERR {e}"
    try:
        import dask.dataframe as dd
        d = nw.from_native(dd.from_pandas(pd.DataFrame({KEY: ["a", "b"], TIME: [1, 2], VAL: [1.0, 2.0]}), npartitions=1))
        out["dask_from_native_type"] = type(d).__name__
        chk("dask collect default", lambda: type(d.collect().to_native()).__name__)
        chk("dask over(order_by)", lambda: d.with_columns(nw.col(VAL).cum_sum().over(KEY, order_by=TIME)).collect().shape)
    except Exception as e:
        out["dask"] = f"ERR {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=list(GRIDS))
    a = ap.parse_args()
    import importlib.metadata as md
    res = {"grid": a.grid, "versions": {p: md.version(p) for p in ["narwhals", "polars", "pandas", "duckdb", "dask", "pyarrow"]},
           "api_stable_v2": api_checks(), "rows": []}
    print(json.dumps(res["api_stable_v2"], indent=1), flush=True)
    with tempfile.TemporaryDirectory() as td:
        for n, T in GRIDS[a.grid]:
            run_size(n, T, td, res)
    outp = os.path.join(os.path.dirname(__file__), "..", "results", f"design_narwhals_boundary_{a.grid}.json")
    with open(outp, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", outp)


if __name__ == "__main__":
    main()
