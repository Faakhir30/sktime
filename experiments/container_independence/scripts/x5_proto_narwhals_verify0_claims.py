"""x5 verifier: independent checks of C1, C2, pyarrow over-bug, laziness, speed attribution.

Different code paths from the original x5 scripts:
  K1  C1 collect defaults via narwhals, for polars-lazy / dask / duckdb (/ibis)
  K2  C2: maybe_reset_index vs native reset_index; reset_index allocation measured with
      tracemalloc (numpy allocations are traced) instead of memory_usage(deep=True)
  K3  pyarrow over bug: (a) root cause - does pa.Table.join keep left row order?
      (b) narwhals mean().over() checked against np.bincount group means
  K4  laziness: polars LazyFrame with a counting map_batches UDF -> does fit/predict
      execute the source before the user collects?  duckdb: output type
  K5  speed attribution: sktime NaiveForecaster(last) vs narwhals-on-pandas vs plain
      pandas groupby (no narwhals) vs narwhals-on-polars vs plain polars (no narwhals)

Run:
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
    experiments/container_independence/scripts/x5_proto_narwhals_verify0_claims.py --grid dev
"""

import argparse
import json
import time
import tracemalloc
import warnings
from datetime import timedelta
from pathlib import Path

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

from common import bench, env_info, make_panel, make_panel_fast
from protos.x5_backends import long_from_mi, make_native
from protos.x5_nw_forecasters import NwPanelNaive

warnings.filterwarnings("ignore")
ED = Path(__file__).resolve().parents[1]
OUT = {"env": env_info()}
FH = list(range(1, 13))


def tname(x):
    return type(x).__module__.split(".")[0] + "." + type(x).__name__


def k1():
    y = make_panel_fast(50, 20)
    lng = long_from_mi(y)
    res = {}
    bs = ["polars_lazy", "dask", "duckdb"]
    try:
        import ibis  # noqa: F401

        bs.append("ibis")
    except ImportError:
        pass
    for b in bs:
        lf = nw.from_native(make_native(lng, b))
        r = {"is_LazyFrame": isinstance(lf, nw.LazyFrame), "default": tname(lf.collect().to_native())}
        for tgt in ["polars", "pandas", "pyarrow"]:
            try:
                r[tgt] = tname(lf.collect(tgt).to_native())
            except Exception as e:  # noqa: BLE001
                r[tgt] = "ERR " + type(e).__name__
        # can one 'collect' into the lazy backend itself?
        try:
            r["collect(own backend)"] = tname(lf.collect(lf.implementation).to_native())
        except Exception as e:  # noqa: BLE001
            r["collect(own backend)"] = f"ERR {type(e).__name__}: {e}"[:100]
        res[b] = r
    OUT["K1_C1"] = res
    print("K1", json.dumps(res, indent=0))


def k2(sizes):
    y = make_panel(4, 5, n_levels=2)
    res = {}
    mr = nw.maybe_reset_index(nw.from_native(y)).to_native()
    res["maybe_reset_index"] = {"index": type(mr.index).__name__, "cols": list(mr.columns)}
    res["native_reset_index_cols"] = list(y.reset_index().columns)
    rows = []
    from sktime.datatypes._vectorize import prepare_VectorizedDF

    for n, T in sizes:
        X = make_panel_fast(n, T)
        deep_in = int(X.memory_usage(deep=True, index=True).sum())
        R = X.reset_index()
        deep_out = int(R.memory_usage(deep=True, index=True).sum())
        shallow_out = int(R.memory_usage(deep=False, index=True).sum())
        del R
        tracemalloc.start()
        R = X.reset_index()
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        t_reset = bench(lambda: X.reset_index(), repeat=5)
        t_prep = bench(lambda: prepare_VectorizedDF(X, iterate_as="Series", is_scitype="Panel"), repeat=3)
        rows.append({
            "n": n, "T": T, "frame_deep_bytes": deep_in,
            "reset_deep_bytes(original method)": deep_out,
            "reset_shallow_bytes": shallow_out,
            "tracemalloc_retained_bytes": cur, "tracemalloc_peak_bytes": peak,
            "ratio_deep/frame": deep_out / deep_in,
            "ratio_traced/frame": cur / deep_in,
            "t_reset_ms": t_reset * 1e3, "t_prep_ms": t_prep * 1e3,
            "reset/prep": t_reset / t_prep,
        })
        print("K2", rows[-1], flush=True)
    res["cost"] = rows
    OUT["K2_C2"] = res


def k3(trials=6):
    res = {"join_order": [], "nw_over": []}
    for n in [100, 500, 1000, 5000]:
        T = 50
        rng = np.random.default_rng(0)
        bad_join = 0
        bad_over = 0
        frac = []
        for s in range(trials):
            codes = np.repeat(np.arange(n), T)
            g = np.array([f"k{i:06d}" for i in range(n)])[codes]
            v = np.random.default_rng(s).normal(size=n * T)
            t = pa.table({"g": g, "v": v})
            # (a) raw pyarrow join keeps left order?
            agg = t.group_by("g").aggregate([("v", "mean")])
            j = t.select(["g"]).join(agg, "g", join_type="left outer")
            bad_join += int(not np.array_equal(j.column("g").to_numpy(zero_copy_only=False), g))
            # (b) narwhals over vs bincount means
            ref = (np.bincount(codes, weights=v) / np.bincount(codes))[codes]
            got = nw.from_native(t).with_columns(m=nw.col("v").mean().over("g")).to_native()
            w = np.abs(got.column("m").to_numpy() - ref) > 1e-12
            bad_over += int(w.any())
            frac.append(float(w.mean()))
        res["join_order"].append({"n": n, "trials_join_reordered": bad_join, "trials": trials})
        res["nw_over"].append({"n": n, "trials_over_wrong": bad_over, "mean_frac_rows_wrong": float(np.mean(frac))})
        print("K3", n, bad_join, bad_over, np.mean(frac), flush=True)
    OUT["K3_pa_over"] = res


def k4():
    import tempfile

    res = {}
    y = make_panel_fast(200, 40)
    lng = long_from_mi(y)
    calls = {"n": 0}

    def counted(s):
        calls["n"] += 1
        return s

    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/y.parquet"
        pl.from_pandas(lng).write_parquet(p)
        for freq in ["D", None]:
            calls["n"] = 0
            lf = pl.scan_parquet(p).with_columns(pl.col("c0").map_batches(counted, return_dtype=pl.Float64))
            m = NwPanelNaive("last", sp=7).fit(lf, keys=["h0"], time="time", freq=freq)
            after_fit = calls["n"]
            out = m.predict(FH)
            after_pred = calls["n"]
            col = out.collect()
            res[f"polars_lazy freq={freq}"] = {
                "out_type": tname(out), "udf_calls_after_fit": after_fit,
                "udf_calls_after_predict": after_pred, "udf_calls_after_collect": calls["n"],
                "rows": col.height,
            }
    import duckdb

    con = duckdb.connect()
    rel = con.from_df(lng)
    m = NwPanelNaive("last").fit(rel, keys=["h0"], time="time", freq="D")
    out = m.predict(FH)
    res["duckdb"] = {"out_type": tname(out), "sql_contains_sort": "ORDER BY" in out.sql_query().upper()}
    OUT["K4_lazy"] = res
    print("K4", res)


def k5(sizes, run_sk_max):
    from sktime.forecasting.naive import NaiveForecaster

    rows = []
    for n, T in sizes:
        y = make_panel_fast(n, T)
        lng = long_from_mi(y)
        plf = pl.from_pandas(lng)
        step = timedelta(days=1)

        def pandas_plain():
            # vectorized pandas (no narwhals): last non-NaN per series + future index
            last = y["c0"].groupby(level=0, sort=False).last()
            cut = y.index.get_level_values(1).to_series().groupby(y.index.get_level_values(0), sort=False).max()
            H = len(FH)
            keys = np.repeat(last.index.to_numpy(), H)
            times = (cut.to_numpy()[:, None] + np.array([np.timedelta64(k, "D") for k in FH])[None, :]).ravel()
            idx = pd.MultiIndex.from_arrays([keys, times], names=y.index.names)
            return pd.DataFrame({"c0": np.repeat(last.to_numpy(), H)}, index=idx)

        def polars_plain():
            st = plf.group_by("h0", maintain_order=True).agg(
                pl.col("c0").drop_nulls().last(), pl.col("time").max().alias("cut"))
            return (st.with_columns(pl.lit(FH).alias("k")).explode("k")
                    .select("h0", (pl.col("cut") + pl.duration(days=pl.col("k"))).alias("time"), "c0"))

        def nw_pdmi():
            return NwPanelNaive("last").fit(y).predict(FH)

        def nw_pl():
            return NwPanelNaive("last").fit(plf, keys=["h0"], time="time").predict(FH)

        # correctness of the plain baselines vs proto
        a = pandas_plain()
        b = nw_pdmi()
        ok_pd = bool(np.allclose(a.sort_index()["c0"].to_numpy(), b.sort_index()["c0"].to_numpy()))
        t = {}
        for name, fn in [("pandas_plain", pandas_plain), ("nw_pdmi", nw_pdmi),
                         ("polars_plain", polars_plain), ("nw_pl", nw_pl)]:
            t[name] = bench(fn, repeat=3 if n >= 10000 else 5)
        if n <= run_sk_max:
            t0 = time.perf_counter()
            NaiveForecaster(strategy="last").fit(y).predict(FH)
            t["sktime_once"] = time.perf_counter() - t0
        row = {"n": n, "T": T, "time_ms": {k: v * 1e3 for k, v in t.items()},
               "pandas_plain_equals_proto": ok_pd,
               "nw_pdmi/pandas_plain": t["nw_pdmi"] / t["pandas_plain"],
               "nw_pl/polars_plain": t["nw_pl"] / t["polars_plain"]}
        if "sktime_once" in t:
            row["sk/nw_pl"] = t["sktime_once"] / t["nw_pl"]
            row["sk/nw_pdmi"] = t["sktime_once"] / t["nw_pdmi"]
            row["sk/pandas_plain"] = t["sktime_once"] / t["pandas_plain"]
        rows.append(row)
        print("K5", json.dumps(row), flush=True)
    OUT["K5_speed"] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    sizes = [(1000, 100), (10000, 50)] if a.grid == "dev" else [(10000, 100), (50000, 50)]
    skip = set(a.skip.split(","))
    if "k1" not in skip:
        k1()
    if "k2" not in skip:
        k2(sizes)
    if "k3" not in skip:
        k3()
    if "k4" not in skip:
        k4()
    if "k5" not in skip:
        k5(sizes, run_sk_max=1000)
    tag = "pd3" if pd.__version__.startswith("3") else "pd2"
    p = ED / "results" / f"x5_proto_narwhals_verify0_claims_{tag}_{a.grid}.json"
    p.write_text(json.dumps(OUT, indent=1, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
