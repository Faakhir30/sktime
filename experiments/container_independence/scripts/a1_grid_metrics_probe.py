"""a1: does a container-free estimator grid break vectorized performance metrics?

``BaseForecastingErrorMetric`` uses ``vectorize_est`` (default return_type "pd.DataFrame")
as a numeric RESULT table (``.iloc``, ``.index``, ``.mean``). This probe evaluates
MeanAbsoluteError on Hierarchical y (2 columns) for all multioutput x multilevel combos,
baseline vs EstimatorGrid installed (strict, then compat).

Run (from WT root):
    PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
        $EXP_PY -W ignore experiments/container_independence/scripts/a1_grid_metrics_probe.py
Output: results/a1_grid_metrics_probe.json
"""
import json
import warnings

warnings.filterwarnings("ignore")
import a1_estimator_grid as g  # noqa: E402
from common import make_panel  # noqa: E402

from sktime.performance_metrics.forecasting import MeanAbsoluteError  # noqa: E402

y = make_panel(4, 20, n_levels=2, n_cols=2)
yp = y * 1.1
RES = {}


def run(tag):
    for mo in ["uniform_average", "raw_values"]:
        for ml in ["raw_values", "uniform_average"]:
            k = f"{tag}|multioutput={mo}|multilevel={ml}"
            try:
                r = MeanAbsoluteError(multioutput=mo, multilevel=ml)(y, yp)
                RES[k] = f"ok {type(r).__name__} {getattr(r, 'shape', None)}"
            except Exception as e:  # noqa: BLE001
                RES[k] = f"FAIL {type(e).__name__}: {str(e)[:90]}"
            print(k, RES[k])


run("off")
for mode in ["strict", "compat"]:
    g.install(mode)
    run(mode)
with open("experiments/container_independence/results/a1_grid_metrics_probe.json", "w") as f:
    json.dump(RES, f, indent=1)
