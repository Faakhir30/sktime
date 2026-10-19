"""design_tiered census: porting leverage of a per-estimator inner-mtype opt-in.

For every registered forecaster: which class in the MRO owns _fit/_predict/_update/
_predict_interval/_predict_quantiles/_predict_var, which shared sktime base classes are in
the MRO, the y_inner_mtype tag, panel-native flag, composite flag. Then: greedy coverage
curve "port K owner implementations -> N classes switch tier". Static, no benchmark.
"""
import json, sys, time, inspect, collections, warnings
warnings.filterwarnings("ignore")
t0 = time.time()
from sktime.registry import all_estimators
from sktime.forecasting.base import BaseForecaster

ests = all_estimators("forecaster", return_names=True)
METHODS = ["_fit", "_predict", "_update", "_predict_interval", "_predict_quantiles", "_predict_var"]
SHARED = ["_StatsModelsAdapter", "_StatsForecastAdapter", "_PmdArimaAdapter", "_BaseWindowForecaster",
          "_Reducer", "_DelegatedForecaster", "_HeterogenousEnsembleForecaster", "BaseGridSearch",
          "_Pipeline", "_GeneralisedStatsForecastAdapter", "_PytorchForecastingAdapter",
          "_TbatsAdapter", "_ProphetAdapter", "BaseDeepNetworkPyTorch", "_SkforecastAdapter",
          "_NeuralForecastAdapter", "_DartsAdapter", "_HeterogenousMetaEstimator"]
PANEL = {"pd-multiindex", "pd_multiindex_hier"}
rows = []
for name, cls in ests:
    mro = cls.__mro__
    owner = {}
    for m in METHODS:
        for k in mro:
            if m in k.__dict__:
                owner[m] = k.__name__ if k is not BaseForecaster else "BaseForecaster"
                break
    tags = cls.get_class_tags()
    yin = tags.get("y_inner_mtype")
    yin = [yin] if isinstance(yin, str) else list(yin or [])
    mro_names = [k.__name__ for k in mro]
    shared = [s for s in SHARED if s in mro_names]
    rows.append(dict(name=name, fit_owner=owner.get("_fit"), predict_owner=owner.get("_predict"),
                     owners=owner, y_inner=yin, panel_native=bool(PANEL & set(yin)),
                     shared=shared, pred_int=bool(tags.get("capability:pred_int")),
                     composite=any(s in shared for s in ["_DelegatedForecaster", "_HeterogenousEnsembleForecaster",
                                                        "BaseGridSearch", "_Pipeline", "_HeterogenousMetaEstimator"])))

n = len(rows)
vec = [r for r in rows if not r["panel_native"]]
impl = collections.Counter((r["fit_owner"], r["predict_owner"]) for r in rows)
impl_vec = collections.Counter((r["fit_owner"], r["predict_owner"]) for r in vec)
# greedy coverage over shared-base-class "port units": porting a shared base covers all classes having it
by_shared = collections.Counter(s for r in vec for s in r["shared"])
# greedy by owner pair (port one _fit/_predict implementation)
cov = []
acc = 0
for k, (pair, c) in enumerate(impl_vec.most_common(), 1):
    acc += c
    if k in (1, 3, 5, 10, 20, 30, 50) or k == len(impl_vec):
        cov.append((k, acc, round(acc / len(vec), 3)))
top = impl_vec.most_common(12)
out = dict(n_forecasters=n, n_vectorized=len(vec), n_panel_native=n - len(vec),
           n_distinct_impl=len(impl), n_distinct_impl_vectorized=len(impl_vec),
           coverage_curve_vectorized=cov, top_impl_pairs=[[list(p), c] for p, c in top],
           shared_base_counts_vectorized=by_shared.most_common(),
           n_pred_int=sum(r["pred_int"] for r in rows),
           n_composite_flag=sum(r["composite"] for r in rows),
           runtime_s=round(time.time() - t0, 1))
json.dump(dict(summary=out, rows=rows), open(sys.argv[1], "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
