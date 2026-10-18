"""a1: dynamic trace of pandas (and polars) API entry calls on the forecaster boundary.

For a set of public-API scenarios (vectorized Panel/Hierarchical, with X, polars
input, non-vectorized baseline, update_predict) this script records, via
``sys.setprofile``:

* every Python-level call from an sktime frame INTO pandas / polars (entry calls;
  pandas-internal calls are not counted), grouped by the calling sktime module group
  (boundary layer, mtype checks, mtype conversion, fh/time utilities, inner estimator)
* per boundary source line: which pandas functions are entered, how often
* call counts of key sktime boundary functions (check_is_scitype, convert_to,
  get_cutoff, _check_X_y, VectorizedDF.items, ...)

and via ``sys.settrace`` (local tracing of boundary files only) which source lines
of the boundary files executed -> joined with the static touchpoint census
(results/a1_touchpoints.json) to mark touchpoints "hot" (executed) vs "cold".

Each scenario is run at two instance counts n1 < n2; the per-instance slope
(count(n2)-count(n1))/(n2-n1) separates O(1)-per-call from O(n_instances) work.

Limitations: properties implemented in Cython (e.g. ``DataFrame.index``,
``.columns``) and C-level functions do not generate Python call events, so they
are not counted; counts are therefore LOWER bounds of pandas API usage.

Run (from WT root):
    PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
        $EXP_PY -W ignore experiments/container_independence/scripts/a1_dynamic_trace.py \
        [--grid dev|large]
Output: results/a1_dynamic_trace.json
"""

import argparse
import json
import os
import sys
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from a1_toy_forecasters import LastValueX  # noqa: E402
from common import env_info, make_panel  # noqa: E402

from sktime.datatypes import convert_to  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402
from sktime.split import ExpandingWindowSplitter  # noqa: E402

WT = os.path.abspath(".")
SK = os.path.join(WT, "sktime") + os.sep
PROTOS = os.path.join(WT, "experiments", "container_independence", "protos")
PD_DIR = os.path.dirname(pd.__file__) + os.sep
PL_DIR = os.path.dirname(pl.__file__) + os.sep

BOUNDARY_FILES = {
    "datatypes/_vectorize.py",
    "forecasting/base/_base.py",
    "datatypes/_utilities.py",
    "utils/multiindex.py",
    "utils/validation/series.py",
    "utils/parallel.py",
}

KEY_FUNCS = {
    ("datatypes/_check.py", "check_is_scitype"),
    ("datatypes/_check.py", "mtype"),
    ("datatypes/_convert.py", "convert_to"),
    ("datatypes/_convert.py", "convert"),
    ("datatypes/_utilities.py", "get_cutoff"),
    ("datatypes/_utilities.py", "update_data"),
    ("forecasting/base/_base.py", "BaseForecaster._check_X_y"),
    ("forecasting/base/_base.py", "BaseForecaster._check_fh"),
    ("forecasting/base/_base.py", "BaseForecaster.fit"),
    ("forecasting/base/_base.py", "BaseForecaster.predict"),
    ("datatypes/_vectorize.py", "prepare_VectorizedDF"),
    ("datatypes/_vectorize.py", "VectorizedDF.items"),
    ("datatypes/_vectorize.py", "VectorizedDF.vectorize_est"),
    ("datatypes/_vectorize.py", "VectorizedDF.reconstruct"),
    ("datatypes/_vectorize.py", "_enforce_index_freq"),
    ("utils/validation/series.py", "check_equal_time_index"),
}


def group_of(rel):
    if rel in BOUNDARY_FILES:
        return "boundary"
    if rel.startswith("datatypes/") and (
        "_check" in rel or "/_base/" in rel or "_dtypekind" in rel
    ):
        return "mtype-check"
    if rel.startswith("datatypes/") and (
        "_convert" in rel or "/_adapter/" in rel or "_registry" in rel
    ):
        return "mtype-convert"
    if rel.startswith("datatypes/"):
        return "datatypes-other"
    if rel in (
        "forecasting/base/_fh.py",
        "utils/datetime.py",
        "utils/validation/forecasting.py",
    ) or rel.startswith(("split/", "utils/validation/")):
        return "fh-time-split"
    if rel.startswith("forecasting/naive/") or rel == "<toy>":
        return "inner-estimator"
    return "sktime-other"


class Tracer:
    def __init__(self):
        self.entry = Counter()  # (lib, group) -> n
        self.entry_callee = defaultdict(Counter)  # (lib, group) -> callee -> n
        self.line_calls = defaultdict(Counter)  # "rel:line" -> callee -> n (pd)
        self.key_calls = Counter()
        self.lines = set()  # executed (rel, line) in boundary files

    def _rel(self, fn):
        if fn.startswith(SK):
            return fn[len(SK):]
        if fn.startswith(PROTOS):
            return "<toy>"
        return None

    def profile(self, frame, event, arg):
        if event != "call":
            return
        code = frame.f_code
        fn = code.co_filename
        if fn.startswith(PD_DIR):
            lib = "pandas"
        elif fn.startswith(PL_DIR):
            lib = "polars"
        else:
            rel = self._rel(fn)
            if rel is not None:
                k = (rel, code.co_qualname)
                if k in KEY_FUNCS:
                    self.key_calls[f"{rel}:{code.co_qualname}"] += 1
            return
        caller = frame.f_back
        if caller is None:
            return
        crel = self._rel(caller.f_code.co_filename)
        if crel is None:
            return
        grp = group_of(crel)
        self.entry[(lib, grp)] += 1
        self.entry_callee[(lib, grp)][code.co_qualname] += 1
        if grp == "boundary":
            self.line_calls[f"{crel}:{caller.f_lineno}"][f"{lib}:{code.co_qualname}"] += 1

    def _local(self, frame, event, arg):
        if event == "line":
            self.lines.add((self._rel(frame.f_code.co_filename), frame.f_lineno))
        return self._local

    def trace(self, frame, event, arg):
        fn = frame.f_code.co_filename
        if fn.startswith(SK) and fn[len(SK):] in BOUNDARY_FILES:
            self.lines.add((fn[len(SK):], frame.f_lineno))
            return self._local
        return None

    def __enter__(self):
        sys.setprofile(self.profile)
        sys.settrace(self.trace)
        return self

    def __exit__(self, *a):
        sys.setprofile(None)
        sys.settrace(None)


# --------------------------------------------------------------------------
# scenarios: each returns a callable running the public API sequence
# --------------------------------------------------------------------------
def sc_panel_univ(n):
    y = make_panel(n, 40, n_levels=1)
    y_new = make_panel(n, 45, n_levels=1, seed=1)
    y_new = y_new.groupby(level=0, group_keys=False).tail(5)

    def run():
        f = NaiveForecaster(strategy="last").fit(y, fh=[1, 2, 3])
        f.predict()
        f.predict_interval(coverage=0.9)
        f.predict_quantiles(alpha=[0.1, 0.9])
        f.update(y_new, update_params=True)
        f.predict()
        f.get_fitted_params()

    return run


def sc_hier_multivar(n):
    y = make_panel(n, 40, n_cols=2, n_levels=2)

    def run():
        f = NaiveForecaster(strategy="last").fit(y, fh=[1, 2, 3])
        f.predict()
        f.predict_interval(coverage=0.9)

    return run


def sc_panel_with_X(n):
    y = make_panel(n, 40, n_levels=1)
    X = make_panel(n, 43, n_cols=2, n_levels=1, seed=2)
    X_train = X.groupby(level=0, group_keys=False).head(40)
    X_fut = X.groupby(level=0, group_keys=False).tail(3)

    def run():
        f = LastValueX().fit(y, X=X_train, fh=[1, 2, 3])
        f.predict(X=X_fut)

    return run


def sc_polars_panel(n):
    y = convert_to(make_panel(n, 40, n_levels=1), "polars_panel")

    def run():
        f = NaiveForecaster(strategy="last").fit(y, fh=[1, 2, 3])
        f.predict()

    return run


def sc_series_baseline(n):
    # n is ignored for the length of y; used to keep the same CLI
    y = make_panel(1, 40, n_levels=1).droplevel(0)

    def run():
        f = NaiveForecaster(strategy="last").fit(y, fh=[1, 2, 3])
        f.predict()
        f.predict_interval(coverage=0.9)

    return run


def sc_update_predict_series(n):
    # n = number of cv windows (moving cutoffs)
    yfull = make_panel(1, 20 + n, n_levels=1).droplevel(0)
    # droplevel leaves freq=None; set it (freq=None breaks update_predict, see
    # a1_semantic_probes "update_predict_freq_none")
    yfull.index = pd.date_range(yfull.index[0], periods=len(yfull), freq="D")
    y0, y1 = yfull.iloc[:20], yfull.iloc[20:]

    def run():
        f = NaiveForecaster(strategy="last").fit(y0, fh=[1])
        cv = ExpandingWindowSplitter(initial_window=1, fh=[1], step_length=1)
        f.update_predict(y1, cv=cv)

    return run


SCENARIOS = {
    "panel_univ[fit,predict,pred_int,pred_q,update,predict,get_fitted_params]":
        sc_panel_univ,
    "hier_multivar_2col[fit,predict,pred_int]": sc_hier_multivar,
    "panel_with_X[fit,predict]": sc_panel_with_X,
    "polars_panel[fit,predict]": sc_polars_panel,
    "series_baseline_novec[fit,predict,pred_int]": sc_series_baseline,
    "update_predict_series[n=cv windows]": sc_update_predict_series,
}


def run_scenario(build, n):
    run = build(n)
    run()  # warm imports / caches outside the trace
    with Tracer() as t:
        run()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    args = ap.parse_args()
    ns = (5, 25) if args.grid == "dev" else (20, 200)

    with open("experiments/container_independence/results/a1_touchpoints.json") as f:
        static = json.load(f)
    static_tp = static["touchpoints"]

    out = {"env": env_info(), "grid": args.grid, "n_instances": ns, "scenarios": {}}
    hot_lines = set()
    for name, build in SCENARIOS.items():
        res = {}
        traces = {}
        for n in ns:
            t = run_scenario(build, n)
            traces[n] = t
            hot_lines |= t.lines
        n1, n2 = ns
        t1, t2 = traces[n1], traces[n2]
        groups = sorted(set(t1.entry) | set(t2.entry))
        res["entry_calls"] = {
            f"{lib}|{g}": {
                f"n={n1}": t1.entry[(lib, g)],
                f"n={n2}": t2.entry[(lib, g)],
                "per_instance_slope": round(
                    (t2.entry[(lib, g)] - t1.entry[(lib, g)]) / (n2 - n1), 2
                ),
            }
            for lib, g in groups
        }
        tot1 = sum(v for (lib, _), v in t1.entry.items() if lib == "pandas")
        tot2 = sum(v for (lib, _), v in t2.entry.items() if lib == "pandas")
        res["pandas_entry_total"] = {
            f"n={n1}": tot1, f"n={n2}": tot2,
            "per_instance_slope": round((tot2 - tot1) / (n2 - n1), 2),
        }
        res["top_pandas_callees_by_group_n2"] = {
            g: t2.entry_callee[("pandas", g)].most_common(8)
            for (lib, g) in groups if lib == "pandas"
        }
        res["key_function_calls"] = {
            k: {
                f"n={n1}": t1.key_calls[k], f"n={n2}": t2.key_calls[k],
                "per_instance_slope": round(
                    (t2.key_calls[k] - t1.key_calls[k]) / (n2 - n1), 2
                ),
            }
            for k in sorted(set(t1.key_calls) | set(t2.key_calls))
        }
        # boundary lines that call into pandas, with slopes
        lines = sorted(set(t1.line_calls) | set(t2.line_calls))
        res["boundary_line_pandas_calls"] = {
            ln: {
                f"n={n1}": sum(t1.line_calls[ln].values()),
                f"n={n2}": sum(t2.line_calls[ln].values()),
                "per_instance_slope": round(
                    (sum(t2.line_calls[ln].values()) - sum(t1.line_calls[ln].values()))
                    / (n2 - n1), 2
                ),
                "callees": dict(t2.line_calls[ln].most_common(4)),
            }
            for ln in lines
        }
        out["scenarios"][name] = res
        print(f"== {name}")
        print("   pandas entry calls total:", res["pandas_entry_total"])
        for k, v in res["entry_calls"].items():
            print(f"   {k:32s} {v}")
        for k, v in res["key_function_calls"].items():
            print(f"   key {k:60s} {v}")

    # hot/cold static touchpoints
    hot, cold = [], []
    for tp in static_tp:
        rel = tp["file"].replace("sktime/", "", 1)
        (hot if (rel, tp["line"]) in hot_lines else cold).append(tp)
    by_cat_hot = Counter(tp["cat"] for tp in hot)
    by_cat_cold = Counter(tp["cat"] for tp in cold)
    by_proto_hot = Counter(tp["proto"] for tp in hot)
    by_proto_all = Counter(tp["proto"] for tp in static_tp)
    out["static_touchpoints_hot"] = len(hot)
    out["static_touchpoints_cold"] = len(cold)
    out["hot_by_category"] = dict(by_cat_hot)
    out["cold_by_category"] = dict(by_cat_cold)
    out["hot_by_protocol_op"] = {k: f"{by_proto_hot[k]}/{v}" for k, v in sorted(
        by_proto_all.items())}
    out["cold_touchpoints"] = [
        f'{tp["file"]}:{tp["line"]} {tp["api"]} [{tp["cat"]}/{tp["proto"]}] {tp["op"]}'
        for tp in cold
    ]
    print("static touchpoints hot:", len(hot), "cold:", len(cold))
    print("hot by protocol op:", out["hot_by_protocol_op"])
    with open("experiments/container_independence/results/a1_dynamic_trace.json", "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
