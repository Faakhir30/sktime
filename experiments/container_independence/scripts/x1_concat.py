"""x1: cost of reconstructing from many per-instance LAZY pieces with nw.concat.

VectorizedDF.reconstruct today does pd.concat(keys=...) over per-instance eager
results. If per-instance results stayed lazy (to "respect the original format"),
reconstruct becomes nw.concat([lf_1, ..., lf_N], how="vertical"). This measures,
per lazy backend and N pieces (50 rows each):

  build_s    : time to build the concatenated lazy plan
  collect_s  : time to collect it (default target)
  eager_s    : baseline - collect each piece eagerly then concat eagerly
               (nw.concat of N eager frames of the default collect target)

Run::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x1_concat.py --grid dev
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("POLARS_MAX_THREADS", "4")
import warnings

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402

import narwhals as nw  # noqa: E402

from common import env_info, make_panel_fast, rss_mb  # noqa: E402

import dask  # noqa: E402
import dask.dataframe as dd  # noqa: E402
import duckdb  # noqa: E402

dask.config.set(scheduler="synchronous")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def pieces(backend, arrow_pieces):
    if backend == "polars":
        return [pl.from_arrow(t).lazy() for t in arrow_pieces]
    if backend == "duckdb":
        return [duckdb.from_arrow(t) for t in arrow_pieces]
    if backend == "dask":
        return [dd.from_pandas(t.to_pandas(), npartitions=1) for t in arrow_pieces]
    raise ValueError(backend)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--tag", default=f"pd{pd.__version__.split('.')[0]}")
    args = ap.parse_args()
    ns = [10, 100, 1000] if args.grid == "dev" else [10, 100, 1000, 5000]
    res = {"env": env_info(), "rows": []}
    print("ENV", json.dumps(res["env"]))
    for n in ns:
        pdf = make_panel_fast(n, 50, n_cols=1).reset_index()
        arrow = pa.Table.from_pandas(pdf, preserve_index=False)
        arrow_pieces = [arrow.slice(i * 50, 50) for i in range(n)]
        for backend in ["polars", "duckdb", "dask"]:
            natives = pieces(backend, arrow_pieces)
            lfs = [nw.from_native(x) for x in natives]
            rec = {"n_pieces": n, "backend": backend}
            try:
                # best of 2 for each phase
                b, c, e = [], [], []
                for _ in range(2):
                    cat, tb = timed(lambda: nw.concat(lfs, how="vertical"))
                    out, tc = timed(lambda: cat.collect())
                    assert out.shape[0] == n * 50, out.shape
                    _, te = timed(
                        lambda: nw.concat([lf.collect() for lf in lfs], how="vertical")
                    )
                    b.append(tb)
                    c.append(tc)
                    e.append(te)
                rec.update(
                    build_s=min(b),
                    collect_s=min(c),
                    lazy_total_s=min(b) + min(c),
                    eager_s=min(e),
                    native_type_after_concat=type(cat.to_native()).__name__,
                )
                rec["lazy_over_eager"] = round(rec["lazy_total_s"] / rec["eager_s"], 2)
            except Exception as ex:  # noqa: BLE001
                rec["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
            res["rows"].append(rec)
            print(
                f"N={n:5d} {backend:7s} "
                + (
                    f"build={rec['build_s']*1e3:8.1f}ms collect={rec['collect_s']*1e3:8.1f}ms "
                    f"eager(collect each+concat)={rec['eager_s']*1e3:8.1f}ms "
                    f"lazy/eager=x{rec['lazy_over_eager']} type={rec['native_type_after_concat']}"
                    if "error" not in rec
                    else rec["error"]
                )
                + f"  rss={rss_mb():.0f}MB",
                flush=True,
            )
            del natives, lfs
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, f"x1_concat_{args.grid}_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    sys.exit(main())
