"""x7: which forecasters could get a panel-native (engine-pushdown) implementation?

Heuristic classification of all sktime forecasters (registry, no soft deps needed):

  closed_form   simple statistical / closed-form per-series fits that map to
                group-by aggregates, window functions or (n, T) array ops
                (naive, trend/OLS, theta-like SES, croston, mean/window, ...)
  composite     wrappers / pipelines / reducers / ensembles / tuning: native path
                only if *all* components are native (delegation, not own kernel)
  iterative     per-series iterative optimisation in a 3rd-party lib (statsmodels,
                pmdarima, prophet, tbats, statsforecast, ...): no SQL/expr kernel;
                best hope = batch numpy/numba over (n, T), not engine pushdown
  global_dl     global / deep-learning / foundation models: already "panel-native"
                in the sense of one model over all series, need tensors not frames
  other         unclassified

Prints counts + the members of each class; writes results/x7_family_count.json.

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x7_family_count.py
"""

import json
import os
import re
import warnings

warnings.simplefilter("ignore")

from sktime.registry import all_estimators  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

CLOSED = {
    "NaiveForecaster", "PolynomialTrendForecaster", "TrendForecaster",
    "ThetaForecaster", "Croston", "STLForecaster", "ExponentialSmoothing",
    "ThetaLinesForecaster", "SquaringResiduals", "ConformalIntervals",
    "NaiveVariance", "DirectReductionForecaster", "ProphetPiecewiseLinearTrendForecaster",
    "SplineTrendForecaster", "CurveFitForecaster", "PolynomialTrendForecaster",
    "ForecastKnownValues", "AutoEnsembleForecaster",
}
# names of forecasters whose fit is a closed-form or O(T) recursion that maps to
# group-by aggregates / window functions / cumulative ops
CLOSED_STRICT = {
    "NaiveForecaster", "PolynomialTrendForecaster", "TrendForecaster",
    "ThetaForecaster", "Croston", "ForecastKnownValues",
    "SplineTrendForecaster", "ProphetPiecewiseLinearTrendForecaster",
    "NaiveVariance", "ThetaLinesForecaster",
}
GLOBAL_DL_DEPS = re.compile(
    r"torch|tensorflow|keras|neuralforecast|transformers|chronos|moirai|"
    r"timesfm|uni2ts|momentfm|tinytimemixer|granite|lag-llama|pytorch|"
    r"huggingface|einops|gluonts|darts"
)
ITER_DEPS = re.compile(
    r"statsmodels|pmdarima|prophet|tbats|statsforecast|arch|pykalman|"
    r"filterpy|simdkalman|sklearn|scikit-learn|pydlm|orbit|skpro|hmmlearn|"
    r"numba|scipy|esig|stumpy|dtw|tslearn|pyts|nolds|cmdstanpy|pymc|"
    r"autots|fbprophet|bats|skforecast|catboost|lightgbm|xgboost|"
    r"pyaf|dysts|pystan"
)
COMPOSITE_PAT = re.compile(
    r"Pipeline|Reduc|Ensemble|Search|Wrapper|Transformed|Composite|Column|"
    r"Hierarchy|Reconciler|MultiplexForecaster|Stacking|Online|Bagging|"
    r"Recursive|Direct|MultioutputTimeSeries|DirRec|Yfrom|FhPlex|Permute|"
    r"Update|DontUpdate|Ignore|Skip|Conformal|SquaringResiduals|Variance|"
    r"Plugin|Composition|Multiplexer|Grouped|ForecastByLevel|Hierarchical|"
    r"BaggingForecaster|Fallback|Reconcile|TSB|Tune|Optuna|Grid|Random|"
    r"Ensembl|Combin"
)


def classify(name, cls):
    deps = cls.get_class_tag("python_dependencies", None) or []
    if isinstance(deps, str):
        deps = [deps]
    dep_s = " ".join(str(d).lower() for d in deps)
    if name in CLOSED_STRICT:
        return "closed_form"
    if GLOBAL_DL_DEPS.search(dep_s) or cls.get_class_tag(
        "capability:global_forecasting", False
    ):
        return "global_dl"
    if COMPOSITE_PAT.search(name):
        return "composite"
    if ITER_DEPS.search(dep_s):
        return "iterative"
    if not deps:
        return "other_no_deps"
    return "other"


def main():
    ests = all_estimators("forecaster", return_names=True, exclude_estimators=None)
    rows = []
    for name, cls in ests:
        deps = cls.get_class_tag("python_dependencies", None)
        rows.append(
            {
                "name": name,
                "module": cls.__module__,
                "deps": deps,
                "class": classify(name, cls),
                "y_inner_mtype": cls.get_class_tag("y_inner_mtype", None),
            }
        )
    counts = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    # inner mtype support: how many already accept Panel/Hierarchical natively
    def accepts_panel(m):
        m = m if isinstance(m, list) else [m]
        return any(
            x in ("pd-multiindex", "pd_multiindex_hier", "polars_panel",
                  "polars_hierarchical") for x in m
        )

    n_panel_inner = sum(accepts_panel(r["y_inner_mtype"]) for r in rows)
    print("total forecasters:", len(rows))
    print("counts:", json.dumps(counts, indent=1))
    print("y_inner_mtype already includes a Panel/Hierarchical mtype:", n_panel_inner)
    for c in sorted(counts):
        mem = sorted(r["name"] for r in rows if r["class"] == c)
        print(f"\n[{c}] ({len(mem)}):", ", ".join(mem))
    out = {
        "total": len(rows),
        "counts": counts,
        "n_inner_panel": n_panel_inner,
        "panel_inner": sorted(r["name"] for r in rows if accepts_panel(r["y_inner_mtype"])),
        "rows": rows,
    }
    with open(os.path.join(RES, "x7_family_count.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
