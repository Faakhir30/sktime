#!/usr/bin/env bash
# Recreate the two experiment envs used by the scripts (pandas 2.x and pandas 3.x).
# Usage: bash experiments/container_independence/setup_envs.sh <env_root_dir>
# Then: EXP_PY=<env_root_dir>/expenv/bin/python, EXP_PY3=<env_root_dir>/expenv_pd3/bin/python
set -euo pipefail
ROOT="${1:?pass a directory for the venvs, e.g. a scratch dir}"
mkdir -p "$ROOT"
uv venv "$ROOT/expenv" --python 3.12 -q
uv pip install -q --python "$ROOT/expenv/bin/python" "pandas>=2.2,<3" "numpy<2.3" polars narwhals \
  "dask[dataframe]" duckdb pyarrow "ibis-framework[duckdb]" scikit-base scikit-learn scipy packaging \
  modin psutil memray
uv venv "$ROOT/expenv_pd3" --python 3.12 -q
uv pip install -q --python "$ROOT/expenv_pd3/bin/python" "pandas>=3" "numpy<2.3" polars narwhals \
  "dask[dataframe]" duckdb pyarrow scikit-base scikit-learn scipy packaging psutil
for p in expenv expenv_pd3; do
  "$ROOT/$p/bin/python" -c "import pandas, polars, narwhals; print('$p', pandas.__version__, polars.__version__, narwhals.__version__)"
done
