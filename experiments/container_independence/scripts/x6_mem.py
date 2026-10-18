"""x6: peak-memory of ingress / end-to-end paths, hub vs current, one fresh process
per measurement.

Peak RSS method (Linux): after building the input and gc, write "5" to
/proc/self/clear_refs (resets VmHWM to the current RSS), run the op, then
peak_delta = VmHWM - VmRSS_before. This catches transient peaks of fast ops that a
sampling thread would miss and is not masked by the data-building peak. The
sampling helper common.peak_rss_delta_mb is recorded alongside as a cross-check.

Run:
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/x6_mem.py --grid dev
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ED, "protos"))

warnings.simplefilter("ignore")
warnings.showwarning = lambda *a, **k: None


def _status_kb(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    raise KeyError(field)


def build(case, n, t, cols):
    import numpy as np  # noqa: F401
    import pyarrow as pa

    from common import make_panel_fast, to_polars_long
    from sktime.datatypes import convert_to

    X = make_panel_fast(n, t, n_cols=cols)
    src = case.split("_")[1]
    if src == "pd":
        return X
    if src == "pl":
        return convert_to(X, "polars_panel")
    if src == "pllong":
        return to_polars_long(X)
    if src == "pa":
        tab = pa.Table.from_pandas(X.reset_index(), preserve_index=False)
        return tab.replace_schema_metadata(None)
    raise KeyError(case)


def op(case, data):
    import numpy as np

    import x6_hub as H
    from sktime.datatypes._vectorize import prepare_VectorizedDF
    from sktime.forecasting.naive import NaiveForecaster

    fh = np.arange(1, 11)
    kind = case.split("_", 2)[2]
    if kind == "current_ingress":
        if case.startswith("ing_pa"):
            return prepare_VectorizedDF(data.to_pandas().set_index(["h0", "time"]),
                                        iterate_as="Series", is_scitype="Panel")
        return prepare_VectorizedDF(data, iterate_as="Series", is_scitype="Panel")
    if kind == "hub_ingress":
        kw = {"key_cols": ["h0"], "time_col": "time"} if case.startswith(
            ("ing_pa", "ing_pllong")) else {}
        return H.from_native(data, **kw)
    if kind == "current_e2e":
        return NaiveForecaster(strategy="last").fit(data, fh=fh).predict()
    if kind == "hub_batched_e2e":
        return H.to_native(H.hub_naive_batched(H.from_native(data), fh, "last"))
    if kind == "hub_loop_e2e":
        hub = H.from_native(data)
        return H.to_native(H.hub_vectorize(hub, lambda: H.HubNaive("last"), fh))
    raise KeyError(case)


def child(case, n, t, cols):
    import common

    # warm-up on a tiny input of the same kind: one-time library initialisation
    # (thread pools, lazily loaded kernels, sktime registries) is not op memory
    op(case, build(case, 10, 20, cols))
    data = build(case, n, t, cols)
    gc.collect()
    rss0 = _status_kb("VmRSS")
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")
    t0 = time.perf_counter()
    res = op(case, data)
    dt = time.perf_counter() - t0
    hwm = _status_kb("VmHWM")
    rss1 = _status_kb("VmRSS")
    del res
    gc.collect()
    _, sampled = common.peak_rss_delta_mb(lambda: op(case, data))
    print(json.dumps({"case": case, "n_series": n, "n_time": t, "n_cols": cols,
                      "peak_delta_mb": (hwm - rss0) / 1024,
                      "retained_delta_mb": (rss1 - rss0) / 1024,
                      "sampled_peak_delta_mb_2nd_run": sampled,
                      "rss_before_mb": rss0 / 1024, "op_s": dt}))


INGRESS = ["ing_pd_current_ingress", "ing_pd_hub_ingress",
           "ing_pl_current_ingress", "ing_pl_hub_ingress",
           "ing_pllong_hub_ingress",
           "ing_pa_current_ingress", "ing_pa_hub_ingress"]
E2E = ["e2e_pd_current_e2e", "e2e_pd_hub_batched_e2e", "e2e_pd_hub_loop_e2e",
       "e2e_pl_current_e2e", "e2e_pl_hub_batched_e2e", "e2e_pl_hub_loop_e2e"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--child", nargs=4)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    if args.child:
        case, n, t, cols = args.child
        child(case, int(n), int(t), int(cols))
        return
    if args.grid == "dev":
        ingress_sizes = [(10_000, 100, 1), (10_000, 100, 4)]
        e2e_sizes = [(1_000, 100, 1)]
    else:
        ingress_sizes = [(50_000, 100, 1), (50_000, 100, 4)]
        e2e_sizes = [(5_000, 100, 1)]
    plan = [(c, s) for s in ingress_sizes for c in INGRESS]
    plan += [(c, s) for s in e2e_sizes for c in E2E]
    rows = []
    env = dict(os.environ)
    for case, (n, t, cols) in plan:
        reps = []
        for _ in range(args.reps if "current_e2e" not in case else 1):
            out = subprocess.run(
                [sys.executable, "-W", "ignore", __file__, "--child", case, str(n),
                 str(t), str(cols)], capture_output=True, text=True, env=env)
            line = [x for x in out.stdout.splitlines() if x.startswith("{")]
            if not line:
                print("FAILED", case, out.stderr[-2000:])
                break
            reps.append(json.loads(line[-1]))
        if not reps:
            continue
        reps.sort(key=lambda r: r["peak_delta_mb"])
        med = reps[len(reps) // 2]
        med["reps_peak_delta_mb"] = [r["peak_delta_mb"] for r in reps]
        rows.append(med)
        print(f"{case:28s} {n:6d}x{t:<4d}c{cols} peak +{med['peak_delta_mb']:8.1f}MB "
              f"retained +{med['retained_delta_mb']:8.1f}MB sampled "
              f"+{med['sampled_peak_delta_mb_2nd_run']:7.1f}MB  op {med['op_s']:.3f}s "
              f"reps {[round(x, 1) for x in med['reps_peak_delta_mb']]}", flush=True)
    import pandas as pd

    from common import env_info

    tag = f"pd{pd.__version__.split('.')[0]}_{args.grid}"
    out = os.path.join(ED, "results", f"x6_mem_{tag}.json")
    with open(out, "w") as f:
        json.dump({"env": env_info(), "rows": rows}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
