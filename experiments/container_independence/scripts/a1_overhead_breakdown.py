"""a1: exclusive wall-time breakdown of a vectorized forecaster call by boundary step.

Question: in a row-vectorized fit/predict/predict_interval of a cheap forecaster
(NaiveForecaster on a Panel, y_inner_mtype=pd.Series), where does the time go?
Which share is the container layer (group/iterate, reconstruct, pivot) that a
backend protocol would replace, and which share is per-slice re-validation /
re-conversion / fh-time work that a protocol would NOT remove by itself?

Method: wrap sktime boundary functions with an exclusive timer stack (time of a
wrapped function minus time of wrapped functions it calls), run the scenario,
report per-bucket seconds and shares. Generator ``VectorizedDF.items`` is timed
per ``next()``. Wrapper overhead is ~1 us/call and is included in the buckets.

Buckets:
  inner-estimator   NaiveForecaster _fit/_predict/_predict_quantiles/... (the "work")
  mtype-check       check_is_scitype, mtype (inference incl. all checker attempts)
  mtype-convert     convert_to / convert (minus nested mtype inference)
  fh-check          BaseForecaster._check_fh
  cutoff            get_cutoff
  update_data       update_data (remember_data=True pooling)
  check_X_y-glue    rest of BaseForecaster._check_X_y
  vec-iterate       VectorizedDF.items (groupby + droplevel), minus freq
  vec-freq-enforce  _enforce_index_freq (pd.infer_freq per slice)
  vec-est-glue      VectorizedDF.vectorize_est minus nested (incl. pivot -> DataFrame)
  vec-reconstruct   VectorizedDF.reconstruct (pd.concat with keys)
  vec-prepare       prepare_VectorizedDF minus nested (schema build)
  clone             BaseObject.clone
  public-glue       BaseForecaster public methods minus all of the above
  unattributed      total wall - sum(buckets)

Run (from WT root):
    PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
        $EXP_PY -W ignore experiments/container_independence/scripts/a1_overhead_breakdown.py \
        --grid dev
Output: results/a1_overhead_breakdown_<grid>.json
"""

import argparse
import functools
import json
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

from common import env_info, make_panel_fast, peak_rss_delta_mb  # noqa: E402

STACK = []
ACC = Counter()
NCALLS = Counter()


def wrap(fn, bucket):
    @functools.wraps(fn)
    def w(*a, **k):
        t0 = time.perf_counter()
        STACK.append([bucket, 0.0])
        try:
            return fn(*a, **k)
        finally:
            el = time.perf_counter() - t0
            b, child = STACK.pop()
            ACC[b] += el - child
            NCALLS[b] += 1
            if STACK:
                STACK[-1][1] += el

    w._a1_wrapped = True
    return w


def wrap_gen(fn, bucket):
    @functools.wraps(fn)
    def w(*a, **k):
        g = fn(*a, **k)
        while True:
            t0 = time.perf_counter()
            STACK.append([bucket, 0.0])
            done = False
            try:
                item = next(g)
            except StopIteration:
                done = True
            finally:
                el = time.perf_counter() - t0
                b, child = STACK.pop()
                ACC[b] += el - child
                NCALLS[b] += 1
                if STACK:
                    STACK[-1][1] += el
            if done:
                return
            yield item

    return w


def install():
    import skbase.base

    import sktime.datatypes as dt
    import sktime.datatypes._check as chk
    import sktime.datatypes._convert as cv
    import sktime.datatypes._utilities as ut
    import sktime.datatypes._vectorize as vz
    import sktime.forecasting.base._base as fb
    import sktime.forecasting.naive._naive as nv
    from sktime.forecasting.naive import NaiveForecaster

    w_cis = wrap(chk.check_is_scitype, "mtype-check")
    w_mt = wrap(chk.mtype, "mtype-check")
    for mod in (chk, dt, fb, vz):
        if hasattr(mod, "check_is_scitype"):
            mod.check_is_scitype = w_cis
    for mod, name in ((chk, "mtype"), (vz, "mtype"), (cv, "infer_mtype")):
        setattr(mod, name, w_mt)

    w_ct = wrap(cv.convert_to, "mtype-convert")
    w_c = wrap(cv.convert, "mtype-convert")
    for mod in (cv, dt, fb, vz, nv):
        if hasattr(mod, "convert_to"):
            mod.convert_to = w_ct
    cv.convert = w_c
    nv.convert = w_c

    w_gc = wrap(ut.get_cutoff, "cutoff")
    for mod in (ut, dt, fb):
        mod.get_cutoff = w_gc
    w_ud = wrap(ut.update_data, "update_data")
    for mod in (ut, dt, fb):
        mod.update_data = w_ud

    w_prep = wrap(vz.prepare_VectorizedDF, "vec-prepare")
    vz.prepare_VectorizedDF = w_prep
    dt.prepare_VectorizedDF = w_prep
    fb.prepare_VectorizedDF = w_prep

    vz.VectorizedDF.items = wrap_gen(vz.VectorizedDF.items, "vec-iterate")
    vz._enforce_index_freq = wrap(vz._enforce_index_freq, "vec-freq-enforce")
    vz.VectorizedDF.vectorize_est = wrap(vz.VectorizedDF.vectorize_est, "vec-est-glue")
    vz.VectorizedDF.reconstruct = wrap(vz.VectorizedDF.reconstruct, "vec-reconstruct")

    skbase.base.BaseObject.clone = wrap(skbase.base.BaseObject.clone, "clone")

    B = fb.BaseForecaster
    B._check_fh = wrap(B._check_fh, "fh-check")
    B._check_X_y = wrap(B._check_X_y, "check_X_y-glue")
    for m in ("fit", "predict", "predict_interval", "predict_quantiles", "update",
              "_vectorize", "_update_y_X", "_set_cutoff_from_y"):
        setattr(B, m, wrap(getattr(B, m), "public-glue"))

    from a1_toy_forecasters import LastValueX

    for cls in (NaiveForecaster, LastValueX):
        for m in ("_fit", "_predict", "_predict_quantiles", "_predict_interval",
                  "_predict_var", "_update", "_predict_naive"):
            if m in vars(cls) or (cls is NaiveForecaster and hasattr(cls, m)):
                setattr(cls, m, wrap(getattr(cls, m), "inner-estimator"))


def make_op(op, n_series, n_time):
    """Return a zero-arg callable running ``op``; setup happens here (untimed)."""
    from a1_toy_forecasters import LastValueX

    from sktime.forecasting.naive import NaiveForecaster

    y = make_panel_fast(n_series, n_time)
    if op == "naive.fit":
        return lambda: NaiveForecaster(strategy="last").fit(y, fh=[1, 2, 3])
    f = NaiveForecaster(strategy="last").fit(y, fh=[1, 2, 3])
    if op == "naive.predict":
        return f.predict
    if op == "naive.predict_interval":
        return lambda: f.predict_interval(coverage=0.9)
    if op == "toy.fit+predict":
        return lambda: LastValueX().fit(y, fh=[1, 2, 3]).predict()
    raise ValueError(op)


OPS_DEV = {
    "naive.fit": [(10, 100), (100, 100), (1000, 100)],
    "naive.predict": [(10, 100), (100, 100), (1000, 100)],
    "naive.predict_interval": [(10, 100), (100, 100)],
    "toy.fit+predict": [(10, 100), (100, 100), (1000, 100)],
}
OPS_LARGE = {
    "naive.fit": [(10, 500), (100, 500), (1000, 500), (10_000, 100)],
    "naive.predict": [(10, 500), (100, 500), (1000, 500), (10_000, 100)],
    "naive.predict_interval": [(10, 500), (100, 500), (1000, 100)],
    "toy.fit+predict": [(10, 500), (100, 500), (1000, 500), (10_000, 100)],
}

CONTAINER = ("vec-iterate", "vec-freq-enforce", "vec-est-glue", "vec-reconstruct",
             "vec-prepare")
REENTRY = ("mtype-check", "mtype-convert", "check_X_y-glue", "fh-check", "cutoff",
           "update_data", "public-glue", "clone")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    args = ap.parse_args()
    ops = OPS_DEV if args.grid == "dev" else OPS_LARGE

    out = {"env": env_info(), "grid": args.grid, "runs": []}

    # uninstrumented reference timing first (same process, before patching)
    ref = {}
    for op, grid in ops.items():
        for ns, nt in grid:
            run = make_op(op, ns, nt)
            run()
            best = float("inf")
            for _ in range(3 if ns <= 100 else 1):
                t0 = time.perf_counter()
                run()
                best = min(best, time.perf_counter() - t0)
            _, rss = peak_rss_delta_mb(run)
            ref[(op, ns, nt)] = (best, rss)
            print(f"uninstrumented {op:24s} n_series={ns:6d} n_time={nt}: "
                  f"{best:8.3f}s ({best / ns * 1e3:7.3f} ms/instance) "
                  f"peak dRSS={rss:.0f}MB", flush=True)

    install()
    for op, grid in ops.items():
        for ns, nt in grid:
            run = make_op(op, ns, nt)
            run()  # warm
            ACC.clear()
            NCALLS.clear()
            t0 = time.perf_counter()
            run()
            total = time.perf_counter() - t0
            acc = dict(ACC)
            acc["unattributed"] = total - sum(ACC.values())
            shares = {k: round(100 * v / total, 1) for k, v in acc.items()}
            container = sum(acc.get(k, 0) for k in CONTAINER)
            reval = sum(acc.get(k, 0) for k in REENTRY)
            inner = acc.get("inner-estimator", 0)
            rec = {
                "op": op, "n_series": ns, "n_time": nt,
                "uninstrumented_s": ref[(op, ns, nt)][0],
                "uninstrumented_ms_per_instance": 1e3 * ref[(op, ns, nt)][0] / ns,
                "uninstrumented_peak_drss_mb": ref[(op, ns, nt)][1],
                "instrumented_s": total,
                "instrumentation_overhead_ratio": total / ref[(op, ns, nt)][0],
                "bucket_s": acc, "bucket_share_pct": shares,
                "bucket_calls": dict(NCALLS),
                "per_instance_ms": {k: 1e3 * v / ns for k, v in acc.items()},
                "grouped_share_pct": {
                    "container-layer (VectorizedDF ops)":
                        round(100 * container / total, 1),
                    "per-slice re-entry (check/convert/fh/cutoff/glue/clone)":
                        round(100 * reval / total, 1),
                    "inner estimator": round(100 * inner / total, 1),
                    "unattributed": round(100 * acc["unattributed"] / total, 1),
                },
            }
            out["runs"].append(rec)
            print(f"\n== {op} n_series={ns} n_time={nt} instrumented={total:.3f}s "
                  f"(x{rec['instrumentation_overhead_ratio']:.2f})", flush=True)
            for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
                print(f"   {k:18s} {v:8.3f}s  {shares[k]:5.1f}%  "
                      f"{1e3 * v / ns:7.3f} ms/inst  calls={NCALLS.get(k, '-')}")
            print("   grouped:", rec["grouped_share_pct"])

    with open(
        f"experiments/container_independence/results/a1_overhead_breakdown_{args.grid}.json",
        "w",
    ) as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
