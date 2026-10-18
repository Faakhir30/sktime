"""x2 (2): cost of reset_index (MultiIndex -> columns) and set_index (inverse).

Puts the "reset_index is a copy" claim in proportion: compares, in the same
process, the time of

  reset      df.reset_index()
  set        flat.set_index(["h0", "time"])            (inverse, rebuilds MultiIndex)
  iter_raw   [g.droplevel(0) for _, g in df.groupby(level=0, sort=False)]
  iter_sk    list(vec.items(X=X_mi))                   (current sktime VectorizedDF)

and records peak RSS delta of a single reset_index / set_index call, plus whether
the float data columns of the reset frame share memory with the original
(np.shares_memory) - i.e. whether reset_index copies the *data* or only
materialises the index levels as new columns.

Usage::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x2_reset_index_cost.py --grid dev
    # pandas 2 with copy-on-write switched on:
    ... $EXP_PY ... x2_reset_index_cost.py --grid dev --cow
    # pandas 3 (CoW always on):
    ... $EXP_PY3 ... x2_reset_index_cost.py --grid dev
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd

from common import (
    DEV_GRID,
    LARGE_GRID,
    bench,
    env_info,
    make_panel_fast,
    peak_rss_delta_mb,
    rss_mb,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def shares(a, b):
    return bool(np.shares_memory(np.asarray(a), np.asarray(b)))


def alloc_mb(fn):
    """(tracemalloc peak MB, net pyarrow-pool MB retained) for one call of fn.

    numpy/pandas buffers are reported to tracemalloc; pyarrow buffers (pandas 3
    ``str`` dtype) are not, so the arrow memory pool delta is recorded as well.
    Deterministic, unlike RSS (which misses re-use of freed arena memory).
    """
    import tracemalloc

    try:
        import pyarrow as pa

        pa0 = pa.total_allocated_bytes()
    except Exception:  # noqa: BLE001
        pa = None
    gc.collect()
    tracemalloc.start()
    res = fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    pa_delta = (pa.total_allocated_bytes() - pa0) if pa is not None else 0
    del res
    return peak / 2**20, pa_delta / 2**20


def col_values(df, c):
    # .values on a float64 column is a view into its block (no copy)
    return df[c].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--cow", action="store_true", help="pandas 2: enable CoW")
    ap.add_argument("--ncols", default="1,10")
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    if args.cow:
        pd.options.mode.copy_on_write = True
    cow = True if pd.__version__ >= "3" else bool(pd.options.mode.copy_on_write)

    from sktime.datatypes._vectorize import prepare_VectorizedDF

    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    ncols_list = [int(x) for x in args.ncols.split(",")]
    rows = []
    for n_series, n_time in grid:
        for n_cols in ncols_list:
            df = make_panel_fast(n_series, n_time, n_cols=n_cols)
            n = len(df)
            mem_df = df.memory_usage(deep=True, index=True).sum() / 2**20
            mem_data = df.memory_usage(deep=False, index=False).sum() / 2**20

            # --- sharing / what is materialised
            flat = df.reset_index()
            share_c0 = shares(col_values(df, "c0"), col_values(flat, "c0"))
            share_all = all(
                shares(col_values(df, c), col_values(flat, c)) for c in df.columns
            )
            mem_new_cols = (
                flat[["h0", "time"]].memory_usage(deep=True, index=False).sum() / 2**20
            )
            mi_back = flat.set_index(["h0", "time"])
            share_back = shares(col_values(df, "c0"), col_values(mi_back, "c0"))
            # does a later write to the reset frame leak into the original?
            flat2 = df.reset_index()
            orig0 = float(df["c0"].iloc[0])
            try:
                flat2.loc[0, "c0"] = orig0 + 1e6
            except Exception:  # noqa: BLE001
                pass
            write_leaks = float(df["c0"].iloc[0]) != orig0
            del flat2

            # --- peak RSS of a single call (result kept alive inside fn)
            _, rss_reset = peak_rss_delta_mb(lambda: df.reset_index())
            gc.collect()
            _, rss_set = peak_rss_delta_mb(lambda: flat.set_index(["h0", "time"]))
            gc.collect()

            tm_reset, pa_reset = alloc_mb(lambda: df.reset_index())
            tm_set, pa_set = alloc_mb(lambda: flat.set_index(["h0", "time"]))

            # --- timings, interleaved in the same process
            vec, X_mi = prepare_VectorizedDF(df, iterate_as="Series", is_scitype="Panel")

            def f_reset():
                return df.reset_index()

            def f_set():
                return flat.set_index(["h0", "time"])

            def f_iter_raw():
                return [g.droplevel(0) for _, g in df.groupby(level=0, sort=False)]

            def f_iter_sk():
                return list(vec.items(X=X_mi))

            rep = args.repeat if n <= 200_000 else max(2, args.repeat // 2)
            t = {}
            for name, fn in [
                ("reset", f_reset),
                ("set", f_set),
                ("iter_raw", f_iter_raw),
                ("iter_sk", f_iter_sk),
            ]:
                t[name] = bench(fn, repeat=rep)

            row = {
                "n_series": n_series,
                "n_time": n_time,
                "n_cols": n_cols,
                "n_rows": n,
                "mem_df_MB(deep,with index)": round(mem_df, 2),
                "mem_data_cols_MB": round(mem_data, 2),
                "mem_new_index_cols_MB(deep)": round(mem_new_cols, 2),
                "h0_dtype_after_reset": str(flat["h0"].dtype),
                "data_cols_shared_after_reset": share_all,
                "c0_shared_after_reset": share_c0,
                "c0_shared_after_set_index_back": share_back,
                "write_to_reset_frame_leaks_into_orig": write_leaks,
                "peak_rss_reset_MB": round(rss_reset, 2),
                "tracemalloc_peak_reset_MB": round(tm_reset, 2),
                "arrow_pool_delta_reset_MB": round(pa_reset, 2),
                "tracemalloc_peak_set_index_MB": round(tm_set, 2),
                "arrow_pool_delta_set_index_MB": round(pa_set, 2),
                "peak_rss_set_index_MB": round(rss_set, 2),
                "t_reset_ms": round(t["reset"] * 1e3, 3),
                "t_set_index_ms": round(t["set"] * 1e3, 3),
                "t_iter_raw_ms": round(t["iter_raw"] * 1e3, 3),
                "t_iter_sktime_ms": round(t["iter_sk"] * 1e3, 3),
                "ratio_reset/iter_sktime": round(t["reset"] / t["iter_sk"], 4),
                "ratio_reset+set/iter_sktime": round(
                    (t["reset"] + t["set"]) / t["iter_sk"], 4
                ),
                "ratio_reset/iter_raw": round(t["reset"] / t["iter_raw"], 4),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del df, flat, mi_back, vec, X_mi
            gc.collect()

    tag = "pd" + pd.__version__.split(".")[0] + ("cow" if args.cow else "")
    out = {
        "env": env_info(),
        "python": sys.version.split()[0],
        "cow": cow,
        "grid": args.grid,
        "rows": rows,
        "final_rss_MB": round(rss_mb(), 1),
    }
    path = os.path.join(RES, f"x2_reset_index_cost_{tag}_{args.grid}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\npandas {pd.__version__}  CoW={cow}  grid={args.grid}")
    hdr = (
        f"{'series x time':>14s} {'cols':>4s} {'rows':>8s} {'shared':>6s} {'leak':>5s} "
        f"{'dataMB':>6s} {'tmRst':>6s} {'paRst':>6s} {'tmSet':>6s} {'paSet':>6s} {'rssRst':>6s} {'rssSet':>6s} {'reset_ms':>9s} "
        f"{'set_ms':>8s} {'iterRaw':>9s} {'iterSk':>9s} {'rst/itSk':>8s} {'(r+s)/itSk':>10s}"
    )
    print(hdr)
    for r in rows:
        print(
            f"{r['n_series']:>7d}x{r['n_time']:<6d} {r['n_cols']:>4d} {r['n_rows']:>8d} "
            f"{str(r['data_cols_shared_after_reset'])[0]:>6s} "
            f"{str(r['write_to_reset_frame_leaks_into_orig'])[0]:>5s} "
            f"{r['mem_data_cols_MB']:>6.1f} {r['tracemalloc_peak_reset_MB']:>6.1f} "
            f"{r['arrow_pool_delta_reset_MB']:>6.1f} {r['tracemalloc_peak_set_index_MB']:>6.1f} "
            f"{r['arrow_pool_delta_set_index_MB']:>6.1f} {r['peak_rss_reset_MB']:>6.1f} "
            f"{r['peak_rss_set_index_MB']:>6.1f} {r['t_reset_ms']:>9.3f} "
            f"{r['t_set_index_ms']:>8.3f} {r['t_iter_raw_ms']:>9.2f} "
            f"{r['t_iter_sktime_ms']:>9.2f} {r['ratio_reset/iter_sktime']:>8.4f} "
            f"{r['ratio_reset+set/iter_sktime']:>10.4f}"
        )
    print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
