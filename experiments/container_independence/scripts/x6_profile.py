"""x6: where does the CURRENT path spend time? cProfile attribution.

Profiles (a) NaiveForecaster vectorized fit+predict on pd-multiindex and polars_panel
input, and (b) prepare_VectorizedDF on polars_panel input, and reports cumulative time
of key framework / pandas functions as a share of the total.

Run:
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/x6_profile.py [--n 200 --t 50]
"""

import argparse
import cProfile
import json
import os
import pstats
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ED, "protos"))

warnings.simplefilter("ignore")
warnings.showwarning = lambda *a, **k: None

from common import env_info, make_panel_fast  # noqa: E402

KEYS = [
    # (label, filename substring, function name)
    ("BaseForecaster.fit", "forecasting/base/_base.py", "fit"),
    ("BaseForecaster.predict", "forecasting/base/_base.py", "predict"),
    ("BaseForecaster._vectorize", "forecasting/base/_base.py", "_vectorize"),
    ("_check_X_y", "forecasting/base/_base.py", "_check_X_y"),
    ("_check_fh", "forecasting/base/_base.py", "_check_fh"),
    ("check_is_scitype", "datatypes/_check.py", "check_is_scitype"),
    ("mtype()", "datatypes/_check.py", "mtype"),
    ("convert_to", "datatypes/_convert.py", "convert_to"),
    ("prepare_VectorizedDF", "datatypes/_vectorize.py", "prepare_VectorizedDF"),
    ("VectorizedDF.items", "datatypes/_vectorize.py", "items"),
    ("VectorizedDF.reconstruct", "datatypes/_vectorize.py", "reconstruct"),
    ("VectorizedDF.vectorize_est", "datatypes/_vectorize.py", "vectorize_est"),
    ("_enforce_index_freq", "datatypes/_vectorize.py", "_enforce_index_freq"),
    ("clone (skbase)", "skbase/base/_base.py", "clone"),
    ("NaiveForecaster._fit", "naive/_naive.py", "_fit"),
    ("NaiveForecaster._predict", "naive/_naive.py", "_predict"),
    ("NaiveForecaster._predict_naive (Lag+concat+ffill)", "naive/_naive.py",
     "_predict_naive"),
    ("ForecastingHorizon.to_absolute", "base/_fh.py", "to_absolute"),
    ("pandas.infer_freq", "tseries/frequencies.py", "infer_freq"),
    ("pandas.concat", "reshape/concat.py", "concat"),
    ("DataFrame.pivot", "reshape/pivot.py", "pivot"),
    ("polars is_monotonically_increasing", "_adapter/polars.py",
     "is_monotonically_increasing"),
    ("convert_polars_to_pandas", "_adapter/polars.py", "convert_polars_to_pandas"),
    ("convert_pandas_to_polars", "_adapter/polars.py", "convert_pandas_to_polars"),
]


def attribute(prof):
    st = pstats.Stats(prof)
    total = max(v[3] for v in st.stats.values())  # max cumulative = top-level
    out = {}
    for label, fsub, fname in KEYS:
        cum = 0.0
        ncalls = 0
        for (fn, _, name), (cc, nc, tt, ct, _) in st.stats.items():
            if name == fname and fsub in fn.replace("\\", "/"):
                cum = max(cum, ct)  # recursive entries: take outermost
                ncalls += nc
        if cum > 0:
            out[label] = {"cum_s": cum, "share": cum / total, "calls": ncalls}
    return total, out


def run(label, fn):
    prof = cProfile.Profile()
    prof.enable()
    fn()
    prof.disable()
    total, out = attribute(prof)
    print(f"\n== {label}: total {total:.3f}s")
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]["cum_s"]):
        print(f"  {k:52s} {v['cum_s']:8.3f}s {100*v['share']:5.1f}%  calls {v['calls']}")
    return {"label": label, "total_s": total, "items": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--t", type=int, default=50)
    args = ap.parse_args()
    from sktime.datatypes import convert_to
    from sktime.datatypes._vectorize import prepare_VectorizedDF
    from sktime.forecasting.naive import NaiveForecaster

    X = make_panel_fast(args.n, args.t)
    pl_sk = convert_to(X, "polars_panel")
    fh = np.arange(1, 11)
    NaiveForecaster().fit(X.iloc[:40], fh=fh).predict()  # warm caches/registries
    res = []
    for strat in ["last", "drift"]:
        res.append(run(f"NaiveForecaster({strat}) fit+predict pd-multiindex "
                       f"{args.n}x{args.t}",
                       lambda: NaiveForecaster(strategy=strat).fit(X, fh=fh).predict()))
    res.append(run(f"NaiveForecaster(last) fit+predict polars_panel {args.n}x{args.t}",
                   lambda: NaiveForecaster(strategy="last").fit(pl_sk, fh=fh).predict()))
    big = convert_to(make_panel_fast(10 * args.n, args.t), "polars_panel")
    res.append(run(f"prepare_VectorizedDF(polars_panel) {10*args.n}x{args.t}",
                   lambda: prepare_VectorizedDF(big, iterate_as="Series",
                                                is_scitype="Panel")))
    out = os.path.join(ED, "results", "x6_profile.json")
    with open(out, "w") as f:
        json.dump({"env": env_info(), "args": vars(args), "runs": res}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
