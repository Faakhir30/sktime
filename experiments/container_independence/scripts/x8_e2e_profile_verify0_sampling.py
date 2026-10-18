"""x8 verify0: independent re-attribution of the vectorized forecasting path.

Different method from x8_e2e_profile.py (which uses monkeypatched exclusive-time
wrappers): a *statistical sampling profiler* (a background thread snapshots the
main thread stack via ``sys._current_frames()`` every ~1 ms; no sktime function
is wrapped) and stack-based bucket rules. Plus ablation/subtraction timings:

  A. sampling buckets for fit(y, fh=[1,2,3]); predict(), pd vs polars input
  B. subtraction: wall(pl) - wall(pd), and wall(pl) with polars
     ``is_monotonically_increasing`` replaced by a vectorized version
  C. per-instance boilerplate by ablation: loop of full single-series
     ``NaiveForecaster().fit(s).predict()`` vs vectorized sktime vs bare
     ``_fit/_predict`` lite loop (x8 T1) - all interleaved, same process
  D. stricter equality of the x8 T3 tiers (instance keys included)
  E. dask rejection without the tag patch

Run (from WT root)::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x8_e2e_profile_verify0_sampling.py --grid dev
"""

import argparse
import gc
import json
import os
import sys
import threading
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(os.path.dirname(HERE), "results")

GRIDS = {"dev": [200, 1000], "large": [1000, 5000], "smoke": [50]}


# ---------------------------------------------------------------------------
# sampling profiler
# ---------------------------------------------------------------------------
class Sampler:
    def __init__(self, interval=0.001):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self.tid = threading.get_ident()

    def _run(self):
        frames = sys._current_frames
        while not self._stop.is_set():
            f = frames().get(self.tid)
            stack = []
            while f is not None:
                co = f.f_code
                stack.append((co.co_filename.replace(os.sep, "/"), co.co_name))
                f = f.f_back
            self.samples.append(tuple(reversed(stack)))  # outermost first
            time.sleep(self.interval)

    def __enter__(self):
        self._old = sys.getswitchinterval()
        sys.setswitchinterval(0.0005)
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.th.join()
        sys.setswitchinterval(self._old)


def classify(stack, est_file):
    """Bucket rule on a full stack (outermost first). Precedence documented inline."""
    in_vs = in_clone = in_inner = in_tf = in_lagcore = False
    in_par = in_monot = in_plconv = in_check = in_convert = False
    in_reconstruct = in_vdf = False
    for fn, name in stack:
        if fn.endswith("datatypes/_vectorize.py") and name == "_vectorize_est_single":
            in_vs = True
        elif in_vs and name == "clone" and ("skbase" in fn or "sktime/base" in fn):
            in_clone = True
        elif in_vs and fn.endswith(est_file) and name in ("_fit", "_predict"):
            in_inner = True
        elif in_inner and fn.endswith("transformations/base/_base.py") and name in (
            "fit_transform", "fit", "transform"
        ):
            in_tf = True
        elif in_tf and fn.endswith("transformations/lag.py") and name in ("_fit", "_transform"):
            in_lagcore = True
        if fn.endswith("utils/parallel.py") and name == "parallelize":
            in_par = True
        if name == "is_monotonically_increasing" and fn.endswith("_adapter/polars.py"):
            in_monot = True
        if fn.endswith("_adapter/polars.py") and name in (
            "convert_polars_to_pandas", "convert_pandas_to_polars"
        ):
            in_plconv = True
        if fn.endswith("datatypes/_check.py") and name in ("check_is_scitype", "check_is_mtype", "mtype"):
            in_check = True
        if fn.endswith("datatypes/_convert.py") and name in ("convert_to", "convert"):
            in_convert = True
        if fn.endswith("datatypes/_vectorize.py") and name == "reconstruct":
            in_reconstruct = True
        if fn.endswith("datatypes/_vectorize.py") and name in ("prepare_VectorizedDF", "__init__"):
            in_vdf = True
    if in_vs:
        if in_clone:
            return "pi_clone"
        if in_inner:
            if in_lagcore:
                return "inner_nested_core"
            if in_tf:
                return "inner_nested_boiler"
            return "inner"
        if in_check:
            return "pi_checks"
        if in_convert:
            return "pi_convert"
        return "pi_other"
    if in_par:
        return "slicing_iter"  # in parallelize, outside the per-item function
    # top level (once per call)
    if in_monot:
        return "top_pl_monotonic_check"
    if in_plconv:
        return "top_pl_conversion"
    if in_check:
        return "top_checks_other"
    if in_convert:
        return "top_convert_other"
    if in_reconstruct:
        return "reconstruct"
    if in_vdf:
        return "vdf_construct"
    return "top_other"


ROLL = {
    "container_once": ["top_pl_monotonic_check", "top_pl_conversion", "top_checks_other",
                       "top_convert_other", "reconstruct", "vdf_construct"],
    "per_instance_framework": ["slicing_iter", "pi_clone", "pi_checks", "pi_convert", "pi_other"],
    "inner_total": ["inner", "inner_nested_core", "inner_nested_boiler"],
}


def sampled_run(est, data, est_file):
    f = est.clone()
    gc.collect()
    with Sampler() as s:
        t0 = time.perf_counter()
        f.fit(data, fh=[1, 2, 3])
        f.predict()
        wall = time.perf_counter() - t0
    # drop samples taken outside the measured call (thread start/stop)
    keep = [st for st in s.samples if any(n == "fit" or n == "predict" for _, n in st)]
    c = Counter(classify(st, est_file) for st in keep)
    tot = sum(c.values())
    pct = {k: 100.0 * v / tot for k, v in c.items()}
    roll = {k: sum(pct.get(b, 0.0) for b in v) for k, v in ROLL.items()}
    return {"wall": wall, "n_samples": tot, "pct": pct, "rollups_pct": roll}


# ---------------------------------------------------------------------------
# B. vectorized monotonic check (to subtract its cost)
# ---------------------------------------------------------------------------
def vec_is_monotonic(obj):
    import polars as pl
    from sktime.datatypes._adapter.polars import get_mi_cols

    idx = get_mi_cols(obj)
    if len(idx) == 0:
        return True
    if len(idx) == 1:
        return bool(obj[idx[0]].is_sorted())
    t = pl.col(idx[-1])
    ok = obj.lazy().group_by(idx[:-1]).agg((t.diff().drop_nulls() >= 0).all().alias("ok"))
    return bool(ok.select(pl.col("ok").all()).collect().item())


def patch_monotonic(fn):
    """Replace module-level references in loaded sktime modules; return undo list."""
    from sktime.datatypes._adapter import polars as pa

    orig = pa.is_monotonically_increasing
    undo = []
    for mname, mod in list(sys.modules.items()):
        if mod is None or not mname.startswith("sktime"):
            continue
        d = getattr(mod, "__dict__", {})
        for k, v in list(d.items()):
            if v is orig:
                setattr(mod, k, fn)
                undo.append((mod, k, orig))
    return undo


def unpatch(undo):
    for mod, k, v in undo:
        setattr(mod, k, v)


# ---------------------------------------------------------------------------
# C. single-series loops
# ---------------------------------------------------------------------------
def loop_full_single(est, y):
    lv = list(range(y.index.nlevels - 1))
    preds, keys = [], []
    for key, g in y.groupby(level=lv, sort=False):
        s = g.droplevel(lv)
        s.index.freq = "D"
        p = est.clone().fit(s, fh=[1, 2, 3]).predict()
        preds.append(p)
        keys.append(key)
    out = pd.concat(preds, keys=keys)
    out.index = out.index.set_names(y.index.names)
    return out


def loop_slice_only(y):
    lv = list(range(y.index.nlevels - 1))
    k = 0
    for _key, g in y.groupby(level=lv, sort=False):
        s = g.droplevel(lv)
        s.index.freq = "D"
        k += 1
    return k


def strict_compare(a, b):
    a = a.sort_index()
    b = b.sort_index()
    if a.shape != b.shape:
        return {"ok": False, "why": f"shape {a.shape} vs {b.shape}"}
    ia = a.index.to_frame(index=False).astype(str)
    ib = b.index.to_frame(index=False).astype(str)
    ia.columns = range(ia.shape[1])
    ib.columns = range(ib.shape[1])
    keys_ok = bool((ia.values == ib.values).all())
    names_ok = list(a.index.names) == list(b.index.names)
    d = float(np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float))))
    return {"ok": keys_ok and d < 1e-6, "keys_equal": keys_ok, "names_equal": names_ok,
            "max_abs_diff": d}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="dev", choices=sorted(GRIDS))
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    warnings.simplefilter("ignore")
    from common import bench, env_info, make_panel_fast

    from sktime.datatypes import check_is_scitype, convert, convert_to
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.trend import PolynomialTrendForecaster

    import x8_e2e_profile as x8

    warnings.simplefilter("ignore")
    res = {"env": env_info(), "grid": a.grid, "runs": []}
    ests = {
        "naive": (NaiveForecaster(strategy="last"), "forecasting/naive/_naive.py"),
        "poly": (PolynomialTrendForecaster(degree=1), "forecasting/trend/_polynomial_trend_forecaster.py"),
    }
    # check poly file path
    import inspect
    ests["poly"] = (ests["poly"][0],
                    inspect.getsourcefile(PolynomialTrendForecaster).replace(os.sep, "/").split("sktime/", 1)[1])
    # warmup
    yw = make_panel_fast(5, 20)
    for e, _ in ests.values():
        e.clone().fit(yw, fh=[1, 2, 3]).predict()
        e.clone().fit(convert_to(yw, "polars_panel"), fh=[1, 2, 3]).predict()

    for n in GRIDS[a.grid]:
        y = make_panel_fast(n, 100)
        ypl = convert_to(y, "polars_panel")
        for ename, (est, efile) in ests.items():
            row = {"n": n, "est": ename}
            # A. sampling (interleaved, keep run with max samples-normalised -> use min wall)
            samp = {}
            for c, d in [("pd", y), ("pl", ypl)]:
                best = None
                for _ in range(2):
                    r = sampled_run(est, d, efile)
                    if best is None or r["wall"] < best["wall"]:
                        best = r
                samp[c] = best
            row["sampling"] = samp
            # B. subtraction, interleaved uninstrumented walls
            fns = {
                "pd": lambda: est.clone().fit(y, fh=[1, 2, 3]).predict(),
                "pl": lambda: est.clone().fit(ypl, fh=[1, 2, 3]).predict(),
            }
            walls = {k: np.inf for k in ["pd", "pl", "pl_vecmono"]}
            for _ in range(a.reps):
                for k in ["pd", "pl", "pl_vecmono"]:
                    gc.collect()
                    undo = patch_monotonic(vec_is_monotonic) if k == "pl_vecmono" else []
                    try:
                        t0 = time.perf_counter()
                        (fns["pl"] if k == "pl_vecmono" else fns[k])()
                        walls[k] = min(walls[k], time.perf_counter() - t0)
                    finally:
                        unpatch(undo)
            row["walls"] = walls
            row["pl_over_pd"] = walls["pl"] / walls["pd"]
            row["pl_minus_pd_pct_of_pl"] = 100 * (walls["pl"] - walls["pd"]) / walls["pl"]
            row["vecmono_saving_pct_of_pl"] = 100 * (walls["pl"] - walls["pl_vecmono"]) / walls["pl"]
            from sktime.datatypes._adapter.polars import is_monotonically_increasing
            row["mono_loop_s"] = bench(lambda: is_monotonically_increasing(ypl), repeat=3)
            row["mono_vec_s"] = bench(lambda: vec_is_monotonic(ypl), repeat=3)
            row["mono_agree"] = bool(is_monotonically_increasing(ypl)) == vec_is_monotonic(ypl)
            row["conv_pl_to_pd_s"] = bench(lambda: convert_to(ypl, "pd-multiindex"), repeat=3)
            row["conv_pl_to_pd_with_from_type_s"] = bench(
                lambda: convert(ypl, "polars_panel", "pd-multiindex"), repeat=3)
            # C. ablation of per-instance boilerplate (interleaved)
            t = {k: np.inf for k in ["vectorized_sktime", "loop_full_single", "lite_T1", "slice_only"]}
            for _ in range(a.reps):
                for k in t:
                    gc.collect()
                    t0 = time.perf_counter()
                    if k == "vectorized_sktime":
                        ref = est.clone().fit(y, fh=[1, 2, 3]).predict()
                    elif k == "loop_full_single":
                        lf = loop_full_single(est, y)
                    elif k == "lite_T1":
                        lt = x8.tier_lite_loop(est, y)
                    else:
                        loop_slice_only(y)
                    t[k] = min(t[k], time.perf_counter() - t0)
            row["ablation"] = t
            row["ablation_ratios"] = {
                "vectorized/full_single_loop": t["vectorized_sktime"] / t["loop_full_single"],
                "vectorized/lite": t["vectorized_sktime"] / t["lite_T1"],
                "full_single_loop/lite": t["loop_full_single"] / t["lite_T1"],
                "slice_only_pct_of_vectorized": 100 * t["slice_only"] / t["vectorized_sktime"],
            }
            row["ablation_valid"] = {"full_single": strict_compare(ref, lf),
                                     "lite": strict_compare(ref, lt)}
            # D. strict validation of T3 tiers (naive only exact vs sktime default)
            if ename == "naive":
                row["T3_strict"] = {
                    "pd": strict_compare(ref, x8.tier_vectorized_pd(y, ename)),
                    "pl_eager": strict_compare(ref, x8.to_pandas_out(x8.tier_vectorized_pl(ypl, ename))),
                    "pl_lazy": strict_compare(ref, x8.to_pandas_out(x8.tier_vectorized_pl(ypl, ename, lazy=True))),
                }
            res["runs"].append(row)
            print(json.dumps({k: row[k] for k in ["n", "est", "walls", "pl_over_pd",
                                                  "vecmono_saving_pct_of_pl", "ablation_ratios"]},
                             default=float), flush=True)
            print("  sampling pd", {k: round(v, 1) for k, v in samp["pd"]["pct"].items()}, flush=True)
            print("  sampling pl", {k: round(v, 1) for k, v in samp["pl"]["pct"].items()}, flush=True)
            print("  rollups pd", samp["pd"]["rollups_pct"], "pl", samp["pl"]["rollups_pct"], flush=True)
            if "T3_strict" in row:
                print("  T3 strict", row["T3_strict"], flush=True)
            print("  valid", row["ablation_valid"], flush=True)

    # E. dask without tag patch
    try:
        import dask.dataframe as dd  # noqa: F401
        ddf = convert_to(make_panel_fast(5, 20), "dask_panel")
        res["dask_type"] = str(type(ddf))
        r = check_is_scitype(ddf, scitype="Panel", return_metadata=False)
        res["dask_check_unpatched"] = bool(r)
        try:
            NaiveForecaster().fit(ddf, fh=[1])
            res["dask_fit_unpatched"] = "ok"
        except Exception as e:  # noqa: BLE001
            res["dask_fit_unpatched"] = f"{type(e).__name__}: {str(e)[:300]}"
        print("dask", res["dask_check_unpatched"], res["dask_fit_unpatched"], flush=True)
    except ImportError:
        pass

    out = a.out or os.path.join(RESULTS, f"x8_e2e_profile_verify0_sampling_{a.grid}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
