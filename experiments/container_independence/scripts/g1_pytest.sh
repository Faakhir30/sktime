#!/usr/bin/env bash
# G1: sktime forecasting base tests (+ vectorize tests) with the trusted path
# monkeypatched in. usage (WT root): bash .../g1_pytest.sh off schema_items
set -u
SP=/tmp/claude-0/-home-user-sktime/c9329e73-89f5-514d-ae5a-9180bf0e4a71/scratchpad
PY=${G1PY:-$SP/venvs/g1_trusted_path/bin/python}
ED=experiments/container_independence
export PYTHONPATH=.:$ED:$ED/protos
TESTS=${G1_TESTS:-"sktime/forecasting/base/tests sktime/datatypes/tests/test_vectorize.py"}
TAG=${G1_TAG:-base}
KARGS=()
[ -n "${G1_K:-}" ] && KARGS=(-k "$G1_K")
for mode in ${*:-off schema_items}; do
  out=$ED/results/g1_pytest_${TAG}_${mode}.txt
  G1_MODE=$mode timeout 1200 $PY -W ignore -m pytest -o addopts="" -p no:cacheprovider \
    -p g1_pytest_plugin -q -rf --tb=line "${KARGS[@]}" $TESTS > $out 2>&1
  echo "== $mode: $(grep -E 'passed|failed' $out | tail -1)"
done
