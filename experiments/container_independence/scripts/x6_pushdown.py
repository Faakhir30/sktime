"""x6: does a hub ingress with a forecaster "lookback hint" realize lazy-engine value?

NaiveForecaster(strategy in {last, mean}, window_length=w) only needs the last w
rows per instance. A hub spoke can push "tail w per instance" into the lazy query
(polars LazyFrame / duckdb relation) before materializing, instead of collecting
everything (what convert_to-to-pandas does today).

Cases (each in a fresh process; time + peak RSS via VmHWM reset):
  pd_current_ingress   pd.read_parquet -> set_index -> prepare_VectorizedDF
  pl_full              scan_parquet -> collect all -> hub -> batched predict
  pl_tail              scan_parquet -> group_by.tail(w) (in plan) -> hub -> predict
  pl_tail_streaming    same, collected with engine="streaming"
  duck_full            duckdb read_parquet ORDER BY -> Arrow capsule -> hub -> predict
  duck_qualify         duckdb ... QUALIFY row_number() <= w -> capsule -> hub -> predict
Predictions of all hub cases are checked equal (strategy=mean, window_length=w).

Run:
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/x6_pushdown.py --grid dev
"""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ED, "protos"))
warnings.simplefilter("ignore")
warnings.showwarning = lambda *a, **k: None

SCRATCH = os.environ.get(
    "X6_SCRATCH",
    "/tmp/claude-1000/-home-f-w-oss-e-sktime--claude-worktrees-decouple-forecasters-"
    "pandas-4ef33f/bd26a3e4-6ca7-4c9a-9645-a550f86267e7/scratchpad/x6_data",
)
W = 10
FH = np.arange(1, 11)


def _kb(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])


def make_file(n, t):
    import polars as pl

    from common import make_panel_fast, to_polars_long

    os.makedirs(SCRATCH, exist_ok=True)
    path = os.path.join(SCRATCH, f"x6_panel_{n}x{t}.parquet")
    if not os.path.exists(path):
        df = to_polars_long(make_panel_fast(n, t))
        df.with_columns(pl.col("time").dt.cast_time_unit("us")).write_parquet(
            path, row_group_size=max(1, n * t // 16))
    return path


def run_case(case, path):
    import duckdb
    import pandas as pd
    import polars as pl

    import x6_hub as H

    kw = {"key_cols": ["h0"], "time_col": "time"}
    if case == "pd_current_ingress":
        from sktime.datatypes._vectorize import prepare_VectorizedDF

        X = pd.read_parquet(path).set_index(["h0", "time"])
        prepare_VectorizedDF(X, iterate_as="Series", is_scitype="Panel")
        return None
    if case == "pl_full":
        hub = H.from_polars(pl.scan_parquet(path), **kw)
    elif case == "pl_tail":
        hub = H.from_polars(pl.scan_parquet(path), lazy_hint={"tail": W}, **kw)
    elif case == "pl_tail_streaming":
        lf = pl.scan_parquet(path).group_by("h0", maintain_order=True).tail(W)
        hub = H.from_polars(lf.collect(engine="streaming"), **kw)
    elif case in ("duck_full", "duck_qualify"):
        con = duckdb.connect()
        q = f"SELECT * FROM read_parquet('{path}')"
        if case == "duck_qualify":
            q += (f" QUALIFY row_number() OVER (PARTITION BY h0 ORDER BY time DESC)"
                  f" <= {W}")
        rel = con.sql(q + " ORDER BY h0, time")
        hub = H.capsule_from_native(rel, **kw)
    else:
        raise KeyError(case)
    pred = H.hub_naive_batched(hub, FH, "mean", window_length=W)
    np.save(os.path.join(SCRATCH, f"x6_pred_{case}.npy"), np.asarray(pred.values))
    return {"n_rows_ingested": hub.n_rows,
            "pred_hash": hashlib.md5(np.round(pred.values, 6).tobytes()).hexdigest()}


def child(case, path):
    # warm-up on a tiny file with the same schema: first-call library initialisation
    # (lazy imports, thread pools) measured at ~3.5 s otherwise
    run_case(case, make_file(10, 20))
    gc.collect()
    rss0 = _kb("VmRSS")
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")
    t0 = time.perf_counter()
    res = run_case(case, path) or {}
    dt = time.perf_counter() - t0
    res.update(case=case, t_s=dt, peak_delta_mb=(_kb("VmHWM") - rss0) / 1024)
    print(json.dumps(res))


CASES = ["pd_current_ingress", "pl_full", "pl_tail", "pl_tail_streaming",
         "duck_full", "duck_qualify"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--child", nargs=2)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.child:
        child(*a.child)
        return
    sizes = [(5_000, 200)] if a.grid == "dev" else [(50_000, 200)]
    rows = []
    for n, t in sizes:
        path = make_file(n, t)
        fsize = os.path.getsize(path) / 2**20
        for case in CASES:
            reps = []
            for _ in range(a.reps):
                out = subprocess.run([sys.executable, "-W", "ignore", __file__,
                                      "--child", case, path], capture_output=True,
                                     text=True)
                line = [x for x in out.stdout.splitlines() if x.startswith("{")]
                if not line:
                    print("FAILED", case, out.stderr[-1500:])
                    break
                reps.append(json.loads(line[-1]))
            if not reps:
                continue
            best_t = min(r["t_s"] for r in reps)
            med_peak = sorted(r["peak_delta_mb"] for r in reps)[len(reps) // 2]
            r0 = reps[0]
            r0.update(n_series=n, n_time=t, file_mb=fsize, t_min_s=best_t,
                      peak_delta_mb_median=med_peak,
                      reps_t=[r["t_s"] for r in reps],
                      reps_peak=[r["peak_delta_mb"] for r in reps])
            rows.append(r0)
            print(f"{case:20s} {n}x{t} file {fsize:.1f}MB  t_min {best_t:7.3f}s  "
                  f"peak +{med_peak:7.1f}MB  rows {r0.get('n_rows_ingested', '-')}  "
                  f"pred {r0.get('pred_hash', '-')[:10]}", flush=True)
    preds = {r["case"]: np.load(os.path.join(SCRATCH, f"x6_pred_{r['case']}.npy"))
             for r in rows if "pred_hash" in r}
    ref = preds.get("pl_full")
    same = all(np.allclose(p, ref, rtol=1e-12, atol=0) for p in preds.values())
    hashes = {0} if same else {r["pred_hash"] for r in rows if "pred_hash" in r}
    print("all hub predictions equal (rtol 1e-12):", same)
    import pandas as pd

    from common import env_info

    out = os.path.join(ED, "results",
                       f"x6_pushdown_pd{pd.__version__.split('.')[0]}_{a.grid}.json")
    with open(out, "w") as f:
        json.dump({"env": env_info(), "W": W, "rows": rows,
                   "predictions_allclose_rtol1e-12": same}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
