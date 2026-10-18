#!/usr/bin/env bash
# a1: run sktime's own tests with the container-free EstimatorGrid
# (protos/a1_estimator_grid.py) monkeypatched in, modes off|strict|compat.
#
# off    = baseline (pd.DataFrame from pivot)
# strict = vectorize_est returns EstimatorGrid (no pandas API at all)
# compat = PandasCompatGrid (grid + read-only .loc/.iloc/.index/.columns/.values shims)
#
# pytest is NOT in the shared env; it is installed with
#   uv pip install --target $SP/venvs/a1_pytest/site pytest   (packaging removed)
# and put on PYTHONPATH.
#
# usage (from WT root):
#   bash experiments/container_independence/scripts/a1_grid_pytest.sh [modes...]
#   default modes: off strict compat ; modes run in parallel (one process each)
set -u
SP=/tmp/claude-0/-home-user-sktime/c9329e73-89f5-514d-ae5a-9180bf0e4a71/scratchpad
PY=${EXP_PY:-$SP/expenv/bin/python}
ED=experiments/container_independence
export PYTHONPATH=.:$ED:$ED/protos:$SP/venvs/a1_pytest/site
TESTS="sktime/datatypes/tests/test_vectorize.py sktime/forecasting/base/tests/test_base.py sktime/forecasting/base/tests/test_base_bugs.py sktime/performance_metrics/forecasting/tests/test_metrics.py sktime/transformations/tests/test_base.py"
MODES=${*:-off strict compat}
for mode in $MODES; do
  (
    out=$ED/results/a1_grid_pytest_${mode}.txt
    A1_GRID_MODE=$mode timeout 3600 $PY -W ignore -m pytest -o addopts="" -p no:cacheprovider \
      -p a1_grid_plugin -q -rf --tb=line $TESTS > $out 2>&1
    echo "== $mode: $(tail -1 $out)"
  ) &
done
wait
