#!/usr/bin/env bash
# final_queue: run the remaining final benchmarks ONE AT A TIME (serial), from WT root.
# Each run goes through final_run.py (timeout, uptime, max child RSS -> results/final_runs.jsonl).
set -u
cd /home/user/sktime
PY=/tmp/claude-0/-home-user-sktime/c9329e73-89f5-514d-ae5a-9180bf0e4a71/scratchpad/expenv/bin/python
S=experiments/container_independence/scripts
R="$PY $S/final_run.py"
$R x4_bench_vectorize 600 -- $PY -W ignore $S/x4_bench_vectorize.py --grid large --tag _final
$R x4_memory 600 -- $PY -W ignore $S/x4_memory.py --grid large --repeat 1
$R x5_bench_vectorize 600 -- $PY -W ignore $S/x5_bench_vectorize.py --grid large
$R x5_bench_forecasters 600 -- $PY -W ignore $S/x5_bench_forecasters.py --grid large --skip-sktime-above 1000
$R x6_bench 600 -- $PY -W ignore $S/x6_bench.py --grid large --current-max 1000
$R x6_mem 600 -- $PY -W ignore $S/x6_mem.py --grid large --reps 1
$R x7_native 600 -- $PY -W ignore $S/x7_native.py --sizes 1000x500,10000x100,50000x50 --baseline-max 1000 --baseline-pl-max 1000 --repeat 3 --tag _final
$R x7_native_mem 600 -- $PY -W ignore $S/x7_native.py --mem --sizes 50000x50 --methods last --tag _final
$R x9_arrow_convert 600 -- $PY -W ignore $S/x9_arrow_convert.py --grid large --rounds 3
$R x9_lazy_pushdown 600 -- $PY -W ignore $S/final_x9_lazy_pushdown.py --grid final --rounds 2
echo QUEUE_DONE
