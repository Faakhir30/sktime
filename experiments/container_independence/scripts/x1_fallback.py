"""x1: cost + fidelity of the "collect then convert back" fallback route.

For each lazy native backend L (polars.LazyFrame, dask, duckdb relation, ibis table):

  native_collect   : the backend's own materialisation (L.collect(), L.compute(),
                     rel.to_arrow_table(), t.to_pyarrow())
  nw_collect_<be>  : nw.from_native(L).collect(backend=<be>) for be in
                     default/pandas/polars/pyarrow
  rt_manual        : collect to the backend's natural eager form, then rebuild the
                     SAME lazy type by hand (dd.from_pandas, duckdb.from_arrow,
                     ibis.memtable, DataFrame.lazy())
  rt_nw            : nw.from_native(L).collect().lazy(backend=<original impl>)

Reports min-of-repeats wall time (same process, interleaved), the ratio to
native_collect, peak RSS delta for one call, and fidelity of the round trip
(native type, schema equality, dask npartitions / known divisions,
duckdb connection identity).

Run::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x1_fallback.py --grid dev
"""

import argparse
import json
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import warnings

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402

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
try:
    import ibis
except ImportError:
    ibis = None

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def short_type(obj):
    t = type(obj)
    return f"{t.__module__.split('.')[0]}.{t.__name__}"


def nw_schema(native):
    obj = nw.from_native(native)
    return {k: str(v) for k, v in obj.collect_schema().items()}


def make_inputs(n_series, n_time, dcon, icon, tag):
    pdf = make_panel_fast(n_series, n_time, n_cols=2).reset_index()
    arrow = pa.Table.from_pandas(pdf, preserve_index=False)
    out = {}
    out["polars_lazy"] = pl.from_arrow(arrow).lazy()
    out["dask"] = dd.from_pandas(pdf, npartitions=min(8, n_series))
    # a real duckdb table (not a replacement scan over a python object)
    dcon.register("x1_src", arrow)
    dcon.execute(f"CREATE OR REPLACE TABLE x1_t_{tag} AS SELECT * FROM x1_src")
    dcon.unregister("x1_src")
    out["duckdb"] = dcon.table(f"x1_t_{tag}")
    if ibis is not None:
        icon.raw_sql(f"DROP TABLE IF EXISTS x1_it_{tag}")
        out["ibis"] = icon.create_table(f"x1_it_{tag}", ibis.memtable(arrow))
    return pdf, out


def variants(name, L, dcon):
    """name -> zero-arg callables returning a native object."""
    v = {}
    lf = nw.from_native(L)
    if name == "polars_lazy":
        v["native_collect"] = lambda: L.collect()
        v["rt_manual"] = lambda: L.collect().lazy()
    elif name == "dask":
        v["native_collect"] = lambda: L.compute()
        v["rt_manual"] = lambda: dd.from_pandas(L.compute(), npartitions=L.npartitions)
    elif name == "duckdb":
        v["native_collect"] = lambda: L.to_arrow_table()
        # back onto the SAME connection as the input relation
        v["rt_manual"] = lambda: dcon.from_arrow(L.to_arrow_table())
    elif name == "ibis":
        v["native_collect"] = lambda: L.to_pyarrow()
        v["rt_manual"] = lambda: ibis.memtable(L.to_pyarrow())
    v["nw_collect_default"] = lambda: lf.collect().to_native()
    v["nw_collect_pandas"] = lambda: lf.collect(backend="pandas").to_native()
    v["nw_collect_polars"] = lambda: lf.collect(backend="polars").to_native()
    v["nw_collect_pyarrow"] = lambda: lf.collect(backend="pyarrow").to_native()
    v["rt_nw"] = lambda: lf.collect().lazy(backend=lf.implementation).to_native()
    return v


def fidelity(name, L, out, dcon):
    rec = {
        "type": short_type(out),
        "same_type": isinstance(out, type(L)) or short_type(out) == short_type(L),
    }
    try:
        s_in, s_out = nw_schema(L), nw_schema(out)
        rec["schema_equal"] = s_in == s_out
        if s_in != s_out:
            rec["schema_diff"] = {
                k: (s_in.get(k), s_out.get(k))
                for k in set(s_in) | set(s_out)
                if s_in.get(k) != s_out.get(k)
            }
    except Exception as e:  # noqa: BLE001
        rec["schema_equal"] = f"err {type(e).__name__}: {e}"[:200]
    if name == "dask" and hasattr(out, "npartitions"):
        rec["npartitions"] = f"{L.npartitions} -> {out.npartitions}"
        rec["known_divisions"] = f"{L.known_divisions} -> {out.known_divisions}"
        rec["dask_dtypes"] = {k: str(v) for k, v in out.dtypes.items()}
    if name == "duckdb" and short_type(out) == short_type(L):
        # can it be joined with a relation from the user's connection?
        try:
            out.join(dcon.sql("select 1 as one"), condition="1=1").limit(1).fetchall()
            rec["joinable_with_user_conn"] = True
        except Exception as e:  # noqa: BLE001
            rec["joinable_with_user_conn"] = f"{type(e).__name__}: {str(e)[:120]}"
    if name == "ibis" and ibis is not None and short_type(out) == short_type(L):
        try:
            rec["ibis_backend"] = type(out._find_backend()).__module__
        except Exception as e:  # noqa: BLE001
            rec["ibis_backend"] = f"{type(e).__name__}: {str(e)[:80]}"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--tag", default=f"pd{pd.__version__.split('.')[0]}")
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID

    dcon = duckdb.connect()
    dcon.execute("SET threads=4")
    icon = ibis.duckdb.connect(threads=4) if ibis is not None else None

    info = env_info()
    info["ibis"] = getattr(ibis, "__version__", None)
    results = {"env": info, "grid": args.grid, "rows": []}
    print("ENV", json.dumps(info))
    for i, (n_series, n_time) in enumerate(grid):
        pdf, inputs = make_inputs(n_series, n_time, dcon, icon, i)
        n_rows = len(pdf)
        for name, L in inputs.items():
            vs = variants(name, L, dcon)
            times = {}
            # interleave variants across repeats: run bench per variant but with
            # small repeat, cycling twice, keep min
            for _cycle in range(2):
                for vname, fn in vs.items():
                    try:
                        t = bench(fn, repeat=max(1, args.repeat // 2), warmup=1)
                        times[vname] = min(times.get(vname, float("inf")), t)
                    except Exception as e:  # noqa: BLE001
                        times[vname] = f"ERR {type(e).__name__}: {str(e)[:100]}"
            mems, fids = {}, {}
            for vname, fn in vs.items():
                try:
                    out, d = peak_rss_delta_mb(fn)
                    mems[vname] = round(d, 1)
                    if vname.startswith("rt_"):
                        fids[vname] = fidelity(name, L, out, dcon)
                    else:
                        fids[vname] = {"type": short_type(out)}
                    del out
                except Exception as e:  # noqa: BLE001
                    mems[vname] = f"ERR {type(e).__name__}"
            base = times.get("native_collect")
            row = {
                "n_series": n_series,
                "n_time": n_time,
                "n_rows": n_rows,
                "backend": name,
                "time_s": times,
                "ratio_to_native_collect": {
                    k: (round(v / base, 2) if isinstance(v, float) and base else None)
                    for k, v in times.items()
                },
                "peak_rss_delta_mb": mems,
                "fidelity": fids,
                "rss_now_mb": round(rss_mb(), 1),
            }
            results["rows"].append(row)
            print(f"\n[{n_series}x{n_time}={n_rows} rows] {name}")
            for k in vs:
                t = times.get(k)
                ts = f"{t * 1e3:9.2f} ms" if isinstance(t, float) else str(t)
                r = row["ratio_to_native_collect"].get(k)
                print(f"  {k:20s} {ts}  x{r}  peakΔ={mems.get(k)}MB  {fids.get(k)}")
        del inputs, pdf, vs, L
        dcon.execute(f"DROP TABLE IF EXISTS x1_t_{i}")
        if icon is not None:
            icon.raw_sql(f"DROP TABLE IF EXISTS x1_it_{i}")
        import gc

        gc.collect()
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, f"x1_fallback_{args.grid}_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("\nwrote", path, "final rss", round(rss_mb(), 1))


if __name__ == "__main__":
    sys.exit(main())
