# Container-independence experiments (forecasters + VectorizedDF)

Scratch experiments backing the design report. Not part of the sktime package.

Python envs (created with uv, outside the repo):

- pandas 2.3: `EXP_PY=/tmp/claude-1000/-home-f-w-oss-e-sktime--claude-worktrees-decouple-forecasters-pandas-4ef33f/bd26a3e4-6ca7-4c9a-9645-a550f86267e7/scratchpad/expenv/bin/python`
- pandas 3.0: `EXP_PY3=/tmp/claude-1000/-home-f-w-oss-e-sktime--claude-worktrees-decouple-forecasters-pandas-4ef33f/bd26a3e4-6ca7-4c9a-9645-a550f86267e7/scratchpad/expenv_pd3/bin/python`

Run from the worktree root:

    PYTHONPATH=.:experiments/container_independence $EXP_PY experiments/container_independence/scripts/<script>.py

Layout:

- `common.py` shared data generators + timing/memory helpers
- `scripts/` one script per experiment (prefix = experiment key)
- `protos/` prototype implementations (adapter, narwhals, grouped-array, ...)
- `results/` raw outputs (json/txt) written by scripts
- `reports/` one markdown report per workstream

## Status (moved to cloud mid-run)

The first investigation workflow (`workflow_investigation.js`) was stopped part-way
through when the session moved to the cloud.

- Finished, with structured results in `results/_local_run_completed_results.json`:
  `r1_sklearn_skrub`, `r2_narwhals`, `r3_ts_libs`, `x1_nw_lazy_collect`
- Report written, but no structured result yet: `a3_time_index`, `x2_nw_index_fidelity`, `x4_proto_adapter`
- In progress, with partial scripts/results only: `a1_core_coupling`, `a2_census`, `r4_dispatch_arch`,
  `x3_iter_bench`, `x5_proto_narwhals`, `x6_proto_hub`, `x7_native_engine`, `x8_e2e_profile`,
  `x9_cheap_paths`
- Not started yet: adversarial verification, the completeness critic, the serial large-grid benchmark

To resume: run `setup_envs.sh <dir>`, then run the workflow script with
`args = {wt, sp: <dir>, completed: <contents of results/_local_run_completed_results.json>}`.
