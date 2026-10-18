"""x4: how far does the adapter-based VectorizedDF extend into the forecaster layer?

Runs real sktime ``BaseForecaster`` fit/predict (vectorized over Panel instances)
with ``prepare_VectorizedDF`` / ``get_cutoff`` swapped *in the _base module
namespace only* (no sktime file edited) for the adapter prototype.

Scenarios (y univariate Panel, fh=[1,2,3]):
  A0  pd in,  NaiveForecaster (y_inner pd.Series), current sktime        [pandas baseline]
  A1  pl in,  NaiveForecaster, current (convert whole frame once)
  A2  pl in,  NaiveForecaster, adapter-native slices (per-slice convert) + cutoff bridge
  A2u pl in,  NaiveForecaster, adapter-native, NO cutoff bridge
  A3  pl in,  NaiveForecaster, inner-family bridge (-> pandas, == convert once)
  B1  pl in,  PolarsNaiveLast (y_inner pl.DataFrame), current
  B2u pl in,  PolarsNaiveLast, adapter-native, NO cutoff bridge
  B2  pl in,  PolarsNaiveLast, adapter-native + cutoff bridge          [zero conversion]
  B3  pd in,  PolarsNaiveLast, current
  B4  pd in,  PolarsNaiveLast, inner-family bridge (-> polars) + cutoff bridge
Also: multivariate polars y through adapter-native (expected to hit the pandas
``columns.droplevel`` in BaseForecaster._vectorize), and a cProfile breakdown.

Usage:
  PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
    $EXP_PY -W ignore experiments/container_independence/scripts/x4_forecaster_ext.py \
    [--grid dev|large] [--max-series 1000] [--repeat 3]
"""

import argparse
import cProfile
import gc
import json
import os
import pstats
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import DEV_GRID, LARGE_GRID, env_info, make_panel_fast, rss_mb  # noqa: E402

from sktime.datatypes import convert_to  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
from x4_instrument import ConvCounter, patched_base, set_inner_target  # noqa: E402
from x4_polars_forecaster import PolarsNaiveLast  # noqa: E402

ED = os.path.join(os.path.dirname(__file__), "..")
FH = [1, 2, 3]


def naive():
    return NaiveForecaster(strategy="last").set_config(warnings="off")


def toy():
    return PolarsNaiveLast().set_config(warnings="off")


SCENARIOS = [
    # key, label, estimator factory, input ("pd"/"pl"), policy, cutoff bridge
    ("A0", "pd in | Naive(pd.Series) | current", naive, "pd", "current", False),
    ("A1", "pl in | Naive(pd.Series) | current convert-once", naive, "pl", "current", False),
    ("A2", "pl in | Naive(pd.Series) | adapter-native per-slice convert + cutoff bridge", naive, "pl", "native", True),
    ("A2u", "pl in | Naive(pd.Series) | adapter-native, no cutoff bridge", naive, "pl", "native", False),
    ("A3", "pl in | Naive(pd.Series) | inner-family bridge (->pandas)", naive, "pl", "inner", True),
    ("B1", "pl in | PolarsNaiveLast(pl.DataFrame) | current", toy, "pl", "current", False),
    ("B2u", "pl in | PolarsNaiveLast(pl.DataFrame) | adapter-native, no cutoff bridge", toy, "pl", "native", False),
    ("B2", "pl in | PolarsNaiveLast(pl.DataFrame) | adapter-native + cutoff bridge", toy, "pl", "native", True),
    ("B3", "pd in | PolarsNaiveLast(pl.DataFrame) | current", toy, "pd", "current", False),
    ("B4", "pd in | PolarsNaiveLast(pl.DataFrame) | inner-family bridge (->polars) + cutoff bridge", toy, "pd", "inner", True),
]


def to_pd_values(y_pred):
    """Normalise a prediction to a sorted pandas frame for value comparison."""
    if not isinstance(y_pred, pd.DataFrame):
        y_pred = convert_to(y_pred, "pd-multiindex")
    return y_pred


def run_once(est_factory, y, policy, cutoff):
    with patched_base(policy, cutoff):
        est = est_factory()
        if policy == "inner":
            set_inner_target(est)
        with ConvCounter() as cf:
            f = est.fit(y, fh=FH)
        with ConvCounter() as cp:
            p = f.predict()
    return f, p, cf.as_dict(), cp.as_dict()


def profile_breakdown(est_factory, y, policy, cutoff):
    """Cumulative seconds in key framework functions for one fit+predict."""
    pr = cProfile.Profile()
    with patched_base(policy, cutoff):
        est = est_factory()
        if policy == "inner":
            set_inner_target(est)
        pr.enable()
        t0 = time.perf_counter()
        f = est.fit(y, fh=FH)
        f.predict()
        total = time.perf_counter() - t0
        pr.disable()
    st = pstats.Stats(pr)
    want = {
        "convert_to": ("_convert.py", "convert_to"),
        "check_is_scitype": ("_check.py", "check_is_scitype"),
        "get_cutoff": ("_utilities.py", "get_cutoff"),
        "adapter_get_cutoff": ("x4_adapter.py", "adapter_get_cutoff"),
        "_check_fh": ("_base.py", "_check_fh"),
        "inner _fit": (None, "_fit"),
        "inner _predict": (None, "_predict"),
        "pl.to_pandas": ("frame.py", "to_pandas"),
        "pl.from_pandas": ("convert", "from_pandas"),
        "items(VectorizedDF)": ("_vectorize.py", "items"),
        "items(VectorizedDF2)": ("x4_adapter.py", "items"),
        "reconstruct(VectorizedDF)": ("_vectorize.py", "reconstruct"),
        "reconstruct(VectorizedDF2)": ("x4_adapter.py", "reconstruct"),
        "clone": ("_base.py", "clone"),
    }
    out = {"total_s": total}
    for label, (fpart, fname) in want.items():
        best = 0.0
        for (fn, _ln, name), (_cc, _nc, _tt, ct, _callers) in st.stats.items():
            if name == fname and (fpart is None or fpart in fn):
                # top-most frame has the largest cumtime (avoids double counting recursion)
                best = max(best, ct)
        out[label] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--max-series", type=int, default=1000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--profile-n", type=int, default=100)
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    grid = [g for g in grid if g[0] <= args.max_series]

    res = {
        "env": env_info(),
        "grid": grid,
        "args": vars(args),
        "scenarios": {k: lbl for k, lbl, *_ in SCENARIOS},
        "runs": [],
    }

    for n_series, n_time in grid:
        y_pd = make_panel_fast(n_series, n_time, 1)
        y_pl = convert_to(y_pd, "polars_panel")
        rounds = args.repeat if n_series <= 100 else max(1, args.repeat - 2)
        warmup_rounds = 1 if n_series <= 100 else 0
        ref = None
        rows = {}
        # 1) correctness + conversion counts (one counted run per scenario)
        for key, label, fac, inp, policy, cutoff in SCENARIOS:
            y = y_pd if inp == "pd" else y_pl
            row = {"n_series": n_series, "n_time": n_time, "key": key, "label": label}
            try:
                _, p, cf, cp = run_once(fac, y, policy, cutoff)
                row["conv_fit"] = cf
                row["conv_predict"] = cp
                row["out_type"] = type(p).__module__.split(".")[0] + "." + type(p).__name__
                vals = to_pd_values(p)
                if ref is None:
                    ref = vals
                row["pred_equal_A0"] = bool(
                    np.allclose(vals.to_numpy(), ref.to_numpy())
                    and list(vals.index.get_level_values(0)) == list(ref.index.get_level_values(0))
                    and list(pd.DatetimeIndex(vals.index.get_level_values(-1)))
                    == list(pd.DatetimeIndex(ref.index.get_level_values(-1)))
                )
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
                row["trace"] = traceback.format_exc()[-800:]
            rows[key] = row
        # 2) interleaved timing: each round runs every scenario once (rotated
        #    order), min over rounds -> robust to load drift on a shared machine
        t_fit = {k: [] for k in rows}
        t_pred = {k: [] for k in rows}
        c_fit = {k: [] for k in rows}
        c_pred = {k: [] for k in rows}
        for r in range(warmup_rounds + rounds):
            order = SCENARIOS[r % len(SCENARIOS) :] + SCENARIOS[: r % len(SCENARIOS)]
            for key, label, fac, inp, policy, cutoff in order:
                if "error" in rows[key]:
                    continue
                y = y_pd if inp == "pd" else y_pl
                with patched_base(policy, cutoff):
                    est = fac()
                    if policy == "inner":
                        set_inner_target(est)
                    gc.collect()
                    c0, t0 = time.process_time(), time.perf_counter()
                    est.fit(y, fh=FH)
                    c1, t1 = time.process_time(), time.perf_counter()
                    est.predict()
                    c2, t2 = time.process_time(), time.perf_counter()
                if r >= warmup_rounds:
                    t_fit[key].append(t1 - t0)
                    t_pred[key].append(t2 - t1)
                    c_fit[key].append(c1 - c0)
                    c_pred[key].append(c2 - c1)
        for key, row in rows.items():
            if t_fit[key]:
                row["t_fit_s"] = min(t_fit[key])
                row["t_predict_s"] = min(t_pred[key])
                row["t_fit_all"] = t_fit[key]
                row["t_predict_all"] = t_pred[key]
                row["cpu_fit_s"] = min(c_fit[key])
                row["cpu_predict_s"] = min(c_pred[key])
                row["cpu_fit_all"] = c_fit[key]
                row["cpu_predict_all"] = c_pred[key]
            row["rss_mb"] = rss_mb()
            res["runs"].append(row)
            cf = row.get("conv_fit", {})
            cp = row.get("conv_predict", {})
            print(
                f"n={n_series:>6} T={n_time} {key:>3} fit={row.get('t_fit_s', float('nan')):8.4f}s "
                f"pred={row.get('t_predict_s', float('nan')):8.4f}s "
                f"cpu fit={row.get('cpu_fit_s', float('nan')):8.4f}s pred={row.get('cpu_predict_s', float('nan')):8.4f}s "
                f"conv fit={cf.get('to_pandas_calls', '?')}+{cf.get('from_pandas_calls', '?')} "
                f"pred={cp.get('to_pandas_calls', '?')}+{cp.get('from_pandas_calls', '?')} "
                f"eq={row.get('pred_equal_A0')} {row.get('error', '')[:120]}",
                flush=True,
            )

    # multivariate polars y through adapter-native: where does it break?
    y2 = convert_to(make_panel_fast(5, 20, 2), "polars_panel")
    mv = {}
    for key, fac, policy, cutoff in [
        ("mv_current_naive", naive, "current", False),
        ("mv_native_naive", naive, "native", True),
        ("mv_native_toy", toy, "native", True),
    ]:
        try:
            _, p, cf, cp = run_once(fac, y2, policy, cutoff)
            mv[key] = {"ok": True, "conv_fit": cf, "conv_predict": cp, "cols": list(p.columns)}
        except Exception as e:  # noqa: BLE001
            tb = traceback.extract_tb(e.__traceback__)
            last_sktime = [f for f in tb if "sktime" in f.filename and "x4_" not in f.filename]
            where = f"{last_sktime[-1].filename.split('worktrees/')[-1]}:{last_sktime[-1].lineno} `{last_sktime[-1].line}`" if last_sktime else ""
            mv[key] = {"ok": False, "error": f"{type(e).__name__}: {e}", "where": where}
        print("multivariate", key, mv[key])
    res["multivariate"] = mv

    # profile breakdown at profile-n
    n_prof = args.profile_n
    y_pd = make_panel_fast(n_prof, 100, 1)
    y_pl = convert_to(y_pd, "polars_panel")
    prof = {}
    for key, label, fac, inp, policy, cutoff in SCENARIOS:
        y = y_pd if inp == "pd" else y_pl
        try:
            prof[key] = profile_breakdown(fac, y, policy, cutoff)
        except Exception as e:  # noqa: BLE001
            prof[key] = {"error": str(e)}
        print("profile", key, {k: round(v, 4) for k, v in prof[key].items() if isinstance(v, float)})
    res["profile"] = {"n_series": n_prof, "n_time": 100, "by_scenario": prof}

    out = os.path.join(ED, "results", f"x4_forecaster_ext_{args.grid}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    main()
