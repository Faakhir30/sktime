"""x4: memory of current vs adapter-native vectorization, one fresh process per measurement.

Same-process RSS deltas are contaminated by allocator caching (pandas/numpy via
glibc malloc, polars via its own allocator), so every (variant, size, repeat) runs in
its own subprocess. Two quantities per variant:
  peak  = peak RSS increase while running the full pipeline (prepare/iterate/reconstruct)
  held  = RSS increase while all slices are held in a list (``as_list``), i.e. the
          memory a loop over per-instance estimators keeps alive

Usage:
  PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
    $EXP_PY -W ignore experiments/container_independence/scripts/x4_memory.py [--grid dev|large] [--repeat 2]
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

ED = os.path.join(os.path.dirname(__file__), "..")
VARIANTS = ["pd_current", "pl_current", "pl_native"]


def child(variant, n_series, n_time):
    sys.path.insert(0, os.path.join(ED, "protos"))
    from common import make_panel_fast, peak_rss_delta_mb, rss_mb

    from sktime.datatypes import convert_to
    from sktime.datatypes._vectorize import prepare_VectorizedDF
    from x4_adapter import prepare_VectorizedDF2

    y_pd = make_panel_fast(n_series, n_time, 1)
    y = y_pd if variant == "pd_current" else convert_to(y_pd, "polars_panel")
    if variant != "pd_current":
        del y_pd
    # warm up code paths on a tiny frame so imports/caches are not counted
    tiny = make_panel_fast(3, 5, 1)
    tiny = tiny if variant == "pd_current" else convert_to(tiny, "polars_panel")

    def prep(z):
        if variant == "pl_native":
            return prepare_VectorizedDF2(z, iterate_as="Series", is_scitype="Panel", X_orig_mtype="polars_panel")
        return prepare_VectorizedDF(z, iterate_as="Series", is_scitype="Panel")

    def pipeline(z):
        v, X = prep(z)
        sl = v.as_list(X)
        if variant == "pl_native":
            return v.reconstruct(sl)
        return v.reconstruct(sl, convert_back=True)

    pipeline(tiny)
    gc.collect()
    _, peak = peak_rss_delta_mb(lambda: pipeline(y))
    gc.collect()
    base = rss_mb()
    v, X = prep(y)
    held_list = v.as_list(X)
    held = rss_mb() - base
    out = {
        "variant": variant,
        "n_series": n_series,
        "n_time": n_time,
        "peak_mb": peak,
        "held_mb": held,
        "n_slices": len(held_list),
        "input_mb": (y.memory_usage(deep=True).sum() if variant == "pd_current" else y.estimated_size()) / 2**20,
    }
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--child", nargs=3)
    args = ap.parse_args()
    if args.child:
        child(args.child[0], int(args.child[1]), int(args.child[2]))
        return
    from common import DEV_GRID, LARGE_GRID, env_info

    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    grid = [g for g in grid if g[0] >= 1000]
    res = {"env": env_info(), "rows": []}
    for n_series, n_time in grid:
        for r in range(args.repeat):
            for variant in VARIANTS:
                p = subprocess.run(
                    [sys.executable, "-W", "ignore", __file__, "--child", variant, str(n_series), str(n_time)],
                    capture_output=True,
                    text=True,
                    env=os.environ,
                )
                line = [x for x in p.stdout.splitlines() if x.startswith("{")]
                if not line:
                    print("child failed", variant, p.stderr[-500:])
                    continue
                row = json.loads(line[-1])
                row["repeat"] = r
                res["rows"].append(row)
                print(row, flush=True)
    with open(os.path.join(ED, "results", f"x4_memory_{args.grid}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
