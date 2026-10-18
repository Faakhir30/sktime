# final_bench — serial low-noise benchmark of all container-independence approaches

Runner: serial benchmark phase (all parallel agents finished). Env: `EXP_PY` (pandas 2.3.3, numpy 2.2.6,
polars 2.0.0, narwhals 2.27.1, pyarrow 26, dask 2026.8, duckdb 1.5.6), 4 cores, 15 GB RAM.
pandas 3 (`EXP_PY3`) was **not** re-run in this phase (time budget); per-workstream reports hold pd3 dev numbers.

## TL;DR

1. **The container layer is not where the time goes for per-instance forecasters.** Current sktime
   NaiveForecaster(last) fit+predict costs 11.3–13 ms per instance on pandas input and 12–15 ms on
   polars input at every size from 10x500 to 5000x100. The polars path is only 1.04–1.18× slower than
   pandas (x8).
2. **Each decoupling approach, compared with the current pandas path (same process, min-of-repeats):**
   - **Adapter (x4), pandas input:** same speed as today. Round trip 0.94–0.99×, iteration 0.95–1.2×.
   - **Adapter (x4), polars input with native slices:** iteration falls to 0.034–0.15× of today's
     per-slice cost at ≥100 series. The round trip is still 0.7–1.9× today's pandas time, because the
     polars mtype check takes 1.2 s at 10k series. With a known mtype, prepare takes only 0.24×.
   - **Adapter (x4) feeding a pandas-inner forecaster:** converting each slice from polars to pandas costs
     2.9–5.3× today's per-slice iteration. Converting per slice costs 1.1–2.7× more than converting the
     whole frame once.
   - **Narwhals VectorizedDF (x5):** on pandas input it is slower, 1.7–2.5× with the fast path and
     3.8–5.7× generic. On polars it is 0.24–0.5× of today's pandas round trip at ≥100 series.
   - **Hub / grouped array (x6):** a slice view costs 0.6–5 µs, 54–236× cheaper than
     `VectorizedDF.items` at 115–274 µs. Ingress is 0.06–0.3× of `prepare_VectorizedDF`, and for
     sktime polars input it is 2.84 s → 11 ms at 10k (258×) and 9.84 s → 28 ms at 50k (358×).
     Reconstruct is 0.05–0.15×. Rebuilding a pandas slice costs 24–27 µs if the stored freq is trusted,
     but 173–248 µs (≈ today's cost) if it is validated or inferred.
   - **Engine-native panel kernels (x7, x5 forecasters, x6 batched):** this is where the large gains
     are. At 1000x500, current sktime takes 11.3 s. The same work takes 65 ms with pandas groupby (173×),
     7.8 ms with polars eager (1,460×), 6.5 ms with polars lazy (1,740×) and 2.2 ms with a numpy grouped
     array (5,200×). Panel-native narwhals forecasters take 0.45 s on pandas (25×) and 0.077 s on polars
     (148×). These ratios are taken against x8's baseline. Against each script's own in-process
     baseline they are x7 pandas groupby 190× and x5 polars 169×. x7 uses fh=1..14 and x8 uses fh=1..3.
   - **Arrow-backed pandas (x9):** polars → arrow-backed flat pandas takes 0.05–0.07× the time of the
     sktime adapter conversion at ≥1000 series, with ≤8 MB extra RSS. sktime does not accept that
     layout. The accepted hybrid MI (numpy index, arrow values) is only 0.66–0.88× of adapter time. It
     is 0.05× of `convert_to` at 10k (1.33 s → 0.073 s), but almost all of that gain comes from skipping
     the polars mtype check, not from Arrow.
3. **Laziness pays only when the query pushes down before sktime sees the data.** The test selects
   1000 series × last 50 points from a 4M-row parquet. Eager sktime read + convert_to takes 5.15 s and
   +786 MB. A polars lazy scan takes 5.9 ms and +31 MB (875×). duckdb takes 14 ms and +11 MB. Lazy
   pushdown followed by converting only the slice takes 15–27 ms and +44–50 MB (190–330×).


## What was run (serial, one process tree at a time)

All commands run from the WT root with
`PYTHONPATH=.:experiments/container_independence[:experiments/container_independence/protos]`.
From x4 onward each run went through `scripts/final_run.py`, which applies a 600 s timeout and records
uptime and the max RSS of the largest waited-for descendant in `results/final_runs.jsonl`. Logs are in
`results/final_<name>.log`. The queue is in `scripts/final_queue.sh`.

| # | script & args | start | load avg before (1/5/15) | wall s | exit | max child RSS MB | output |
|---|---|---|---|---|---|---|---|
| 1 | `x3_iter_bench.py --grid large --rounds 3 --tag final_large_pd2` | 21:14 | 0.16, 1.01, 1.41 | ~480 (killed before 50000x50) | killed | 824 (ru_maxrss) | `results/x3_iter_bench__final_large_pd2.json`, `final_x3_large.log` |
| 2 | `x8_e2e_profile.py --grid large --scitypes Panel --ests naive --containers pd,pl,dask --max-n 5000 --cprofile-max-n 100 --out results/final_x8_large.json` | 21:22 | 1.13, 1.22, 1.34 | 665 | 0 | 658 (job peak) | `results/final_x8_large.json`, `final_x8_large.log` |
| 3 | `x4_bench_vectorize.py --grid large --tag _final` | 21:34 | 0.76, 0.97, 1.13 | 600 | timeout at 50000x50 | 1379 | rows parsed from `final_x4_bench_vectorize.log` |
| 4 | `x4_memory.py --grid large --repeat 1` | 21:44 | 1.03, 1.02, 1.07 | 103 | 0 | 778 | `results/x4_memory_large.json` |
| 5 | `x5_bench_vectorize.py --grid large` | 21:45 | 1.08, 1.04, 1.07 | 600 | timeout at 50000x50 | 1368 | parsed from `final_x5_bench_vectorize.log` |
| 6 | `x5_bench_forecasters.py --grid large --skip-sktime-above 1000` | 21:55 | 1.00, 1.00, 1.01 | 600 | timeout after 50000x50 `last` | 2259 | `results/x5_bench_forecasters_large.json` (written per row) |
| 7 | `x6_bench.py --grid large --current-max 1000` | 22:05 | 1.18, 1.19, 1.07 | 538 | 0 | 2088 | `results/x6_bench_pd2_large_c1_ingress-slicing-forecaster.json` |
| 8 | `x6_mem.py --grid large --reps 1` | 22:14 | 1.00, 1.02, 1.02 | 393 | 0 | 1216 | `results/x6_mem_pd2_large.json` |
| 9 | `x7_native.py --sizes 1000x500,10000x100,50000x50 --baseline-max 1000 --baseline-pl-max 1000 --repeat 3 --tag _final` | 22:21 | 1.04, 1.03, 1.01 | 341 | 0 | 1267 | `results/x7_native_dev_t4_final.json` ("dev" in the name is only the unused default flag; the sizes are explicit) |
| 10 | `x7_native.py --mem --sizes 50000x50 --methods last --tag _final` | 22:27 | 1.41, 1.10, 1.03 | 600 | timeout in engine `f` (sktime at 50k) | 619 (+ orphan 1380, killed) | parsed from `final_x7_native_mem.log` |
| 11 | `x9_arrow_convert.py --grid large --rounds 3` | 22:37 | 1.01, 1.05, 1.02 | 211 | 0 | 972 | `results/x9_arrow_convert_large_pd2.json` |
| 12 | `final_x9_lazy_pushdown.py --grid final --rounds 2` (a copy of x9_lazy_pushdown with a 20000x200 = 4M-row grid and final_-prefixed outputs; the original was not edited) | 22:40 | 1.15, 1.09, 1.03 | 231 | 0 | 1711 | `results/final_x9_lazy_pushdown_final_pd2.json` |

Consolidation: `python scripts/final_consolidate.py` writes `results/final_bench.json`, with
`results/final_consolidate.out` as the full dump. `python scripts/final_tables.py` writes
`results/final_tables.md`, which holds the tables below. A load average of about 1.0 throughout means
only the benchmark process itself was running on the 4-core machine.

Grid: `LARGE_GRID` = 10x500, 100x500, 1000x500, 10000x100, 50000x50. Some scripts have their own grids:
x8 uses 10x500, 100x500, 1000x500, 5000x100 (20000x100 dropped, see below); x5 forecasters use
1000x500, 10000x100, 50000x50; x6_mem uses 50000x100 (1 and 4 columns) for ingress and 5000x100 for e2e;
x9_lazy uses 20000x200.

## Comparison tables

Ratios in parentheses are against the current sktime pandas path in the same process (min of repeats).
"—" means not measured at that size (see "not run" below).

**T1. Iterate: per-slice cost, µs (x3 interleaved in one process)** (ratio vs `current_pandas (S1 VectorizedDF.items)` in parentheses)

| approach | 10x500 | 100x500 | 1000x500 | 10000x100 |
|---|---|---|---|---|
| current sktime pandas (S1 VectorizedDF.items) | 274 (1×) | 177 (1×) | 162 (1×) | 130 (1×) |
| current sktime, slice without stored freq | 709 (2.6×) | 415 (2.3×) | 416 (2.6×) | 418 (3.2×) |
| current sktime polars-input (iterates converted pandas; x4) | 256 (0.93×) | 185 (1×) | 154 (0.96×) | 136 (1×) |
| x4 adapter[pandas] | 284 (1×) | 172 (0.97×) | 154 (0.95×) | 155 (1.2×) |
| x4 adapter[polars] native slices | 149 (0.54×) | 25.8 (0.15×) | 18.6 (0.11×) | 4.43 (0.034×) |
| x4-style polars offsets slice (S5) | 10.2 (0.037×) | 2.57 (0.015×) | 1.47 (0.0091×) | 1.35 (0.01×) |
| x4-style polars slice -> pandas per slice (S5p) | 788 (2.9×) | 661 (3.7×) | 634 (3.9×) | 685 (5.3×) |
| x5 narwhals[pandas] items (generic) | 1.7e+03 (6.2×) | 1.05e+03 (5.9×) | 921 (5.7×) | 1.08e+03 (8.3×) |
| x5 narwhals[pandas] items (numpy fast path) | 1.06e+03 (3.9×) | 358 (2×) | 272 (1.7×) | 303 (2.3×) |
| x5 narwhals[polars] items | 493 (1.8×) | 63.5 (0.36×) | 29 (0.18×) | 23.7 (0.18×) |
| x5 narwhals[polars-lazy] items | 477 (1.7×) | 71 (0.4×) | 32.4 (0.2×) | 21.6 (0.17×) |
| x6 hub numpy view (S6) | 5.04 (0.018×) | 1.04 (0.0059×) | 0.683 (0.0042×) | 0.629 (0.0048×) |
| x6 hub -> pandas slice, trusted freq | 26.8 (0.098×) | 26.2 (0.15×) | 25.2 (0.16×) | 23.9 (0.18×) |
| x6 hub -> pandas slice, validated ctor | 248 (0.9×) | 183 (1×) | 173 (1.1×) | 219 (1.7×) |
| pyarrow Table.slice (S8) | 9.66 (0.035×) | 1.99 (0.011×) | 1.07 (0.0067×) | 1.07 (0.0082×) |

**T1b. Iterate per-slice at 50000x50 (x6 only; x3/x4/x5 50k points not run)** (ratio vs `x6 current VectorizedDF.items` in parentheses)

| approach | 50000x50 |
|---|---|
| current sktime VectorizedDF.items (x6) | 125 (1×) |
| x6 hub view | 1.55 (0.012×) |
| x6 hub -> pandas, trusted freq | 24.9 (0.2×) |
| x6 hub -> pandas, validated ctor | 220 (1.8×) |

**T2. Reconstruct (fh=2 outputs -> one frame), seconds** (ratio vs `current_pandas pd.concat(keys) (S1)` in parentheses)

| approach | 10x500 | 100x500 | 1000x500 | 10000x100 |
|---|---|---|---|---|
| current sktime pandas pd.concat(keys) (x3 S1) | 0.00105 (1×) | 0.00255 (1×) | 0.0166 (1×) | 0.172 (1×) |
| current sktime polars-input reconstruct+convert_back (x4) | 0.00402 (3.8×) | 0.0102 (4×) | 0.0696 (4.2×) | 0.311 (1.8×) |
| x4 adapter[pandas] | 0.0011 (1×) | 0.00284 (1.1×) | 0.0186 (1.1×) | 0.192 (1.1×) |
| x4 adapter[polars] | 0.00112 (1.1×) | 0.00192 (0.75×) | 0.0106 (0.64×) | 0.0705 (0.41×) |
| x5 narwhals[pandas] | 0.00135 (1.3×) | 0.00328 (1.3×) | 0.0251 (1.5×) | 0.289 (1.7×) |
| x5 narwhals[polars] | 0.00125 (1.2×) | 0.00367 (1.4×) | 0.0393 (2.4×) | 0.221 (1.3×) |
| x5 narwhals[polars-lazy] | 0.00135 (1.3×) | 0.00428 (1.7×) | 0.0479 (2.9×) | 0.228 (1.3×) |
| narwhals[pandas] literal-key concat (x3 S7a_lit) | 0.00807 (7.7×) | 0.0519 (20×) | 0.477 (29×) | 5.33 (31×) |
| x6 hub numpy (S6) | 0.000155 (0.15×) | 0.000287 (0.11×) | 0.00126 (0.076×) | 0.0091 (0.053×) |
| x6 hub -> pandas MI (S6_to_pd) | 0.000784 (0.74×) | 0.000831 (0.33×) | 0.00196 (0.12×) | 0.00966 (0.056×) |
| polars vectorized concat (S3/S4/S5_vec) | 0.000858 (0.82×) | 0.00135 (0.53×) | 0.00215 (0.13×) | 0.0101 (0.059×) |
| pyarrow (S8) | 0.000434 (0.41×) | 0.000615 (0.24×) | 0.00145 (0.088×) | 0.0106 (0.062×) |

**T3. Conversion / prepare (ingest), seconds** (ratio vs `x4 prepare current pandas` in parentheses)

| approach | 10x500 | 100x500 | 1000x500 | 10000x100 | 50000x50 |
|---|---|---|---|---|---|
| current sktime pandas prepare_VectorizedDF (x4, from_product frame) | 0.00205 (1×) | 0.00507 (1×) | 0.0355 (1×) | 0.078 (1×) | — |
| current sktime polars-input prepare (check+convert_to) (x4) | 0.0151 (7.4×) | 0.092 (18×) | 0.828 (23×) | 2.62 (34×) | — |
|   of which mtype check polars_panel | 0.0055 (2.7×) | 0.0394 (7.8×) | 0.392 (11×) | 1.2 (15×) | — |
| x4 adapter[polars] prepare | 0.00676 (3.3×) | 0.0446 (8.8×) | 0.366 (10×) | 1.24 (16×) | — |
| x4 adapter[polars] prepare, known mtype | 0.00222 (1.1×) | 0.0028 (0.55×) | 0.00859 (0.24×) | 0.0185 (0.24×) | — |
| x5 narwhals prepare, pandas input | 0.00355 (1.7×) | 0.0106 (2.1×) | 0.0728 (2.1×) | 0.142 (1.8×) | — |
| x5 narwhals prepare, polars input | 0.003 (1.5×) | 0.00353 (0.7×) | 0.00849 (0.24×) | 0.0145 (0.19×) | — |
| x5 narwhals prepare, polars-lazy input | 0.00619 (3×) | 0.00595 (1.2×) | 0.00786 (0.22×) | 0.0129 (0.17×) | — |
| x6 hub ingress pandas MI | 0.000574 (0.28×) | 0.000425 (0.084×) | 0.00223 (0.063×) | 0.00473 (0.061×) | 0.0138 |
| x6 hub ingress sktime polars_panel | 0.0017 (0.83×) | 0.00194 (0.38×) | 0.00661 (0.19×) | 0.011 (0.14×) | 0.0275 |
| x6 current ingress sktime polars_panel | 0.0155 (7.6×) | 0.0924 (18×) | 0.83 (23×) | 2.84 (36×) | 9.84 |
| x6 current ingress dask | 0.00883 (4.3×) | 0.0295 (5.8×) | 0.249 (7×) | 0.53 (6.8×) | 1.49 |
| x6 hub ingress dask | 0.0064 (3.1×) | 0.0219 (4.3×) | 0.169 (4.8×) | 0.322 (4.1×) | 0.909 |
| x9 polars->pandas via sktime convert_to | 0.00914 (4.5×) | 0.0452 (8.9×) | 0.408 (12×) | 1.33 (17×) | — |
| x9 polars->pandas via sktime adapter (no check) | 0.00356 (1.7×) | 0.00778 (1.5×) | 0.0543 (1.5×) | 0.11 (1.4×) | 0.285 |
| x9 polars->arrow-backed pandas, flat | 0.00102 (0.5×) | 0.00133 (0.26×) | 0.00373 (0.11×) | 0.00624 (0.08×) | 0.0146 |
| x9 polars->hybrid MI (numpy index, arrow values) | 0.00309 (1.5×) | 0.00683 (1.3×) | 0.0362 (1×) | 0.0725 (0.93×) | 0.229 |

**T3b. Round trip prepare+iterate+reconstruct, seconds** (ratio vs `x4 round trip current pandas` in parentheses)

| approach | 10x500 | 100x500 | 1000x500 | 10000x100 |
|---|---|---|---|---|
| current sktime pandas (x4) | 0.00577 (1×) | 0.026 (1×) | 0.217 (1×) | 1.95 (1×) |
| current sktime polars-input (x4) | 0.0233 (4×) | 0.117 (4.5×) | 1.06 (4.9×) | 4.81 (2.5×) |
| x4 adapter[pandas] | 0.00554 (0.96×) | 0.0251 (0.97×) | 0.216 (0.99×) | 1.84 (0.94×) |
| x4 adapter[polars] | 0.0101 (1.8×) | 0.0451 (1.7×) | 0.408 (1.9×) | 1.36 (0.7×) |
| x5 narwhals[pandas] generic | 0.0221 (3.8×) | 0.116 (4.5×) | 0.995 (4.6×) | 11.1 (5.7×) |
| x5 narwhals[pandas] fast path | 0.0143 (2.5×) | 0.0481 (1.8×) | 0.367 (1.7×) | 3.5 (1.8×) |
| x5 narwhals[polars] | 0.00785 (1.4×) | 0.0129 (0.5×) | 0.0717 (0.33×) | 0.458 (0.24×) |
| x5 narwhals[polars-lazy] | 0.0102 (1.8×) | 0.0175 (0.67×) | 0.0706 (0.32×) | 0.461 (0.24×) |
| x4: polars->pandas once (whole frame) | 0.0156 (2.7×) | 0.0905 (3.5×) | 0.847 (3.9×) | 5.65 (2.9×) |
| x4: polars->pandas per slice (adapter + pd-inner forecaster) | 0.0176 (3×) | 0.138 (5.3×) | 1.4 (6.5×) | 15.2 (7.8×) |

**T4. End-to-end NaiveForecaster(strategy='last') fit+predict, seconds** (ratio vs `x8 current sktime, pd input` in parentheses)

| approach | 10x500 | 100x500 | 1000x500 | 5000x100 | 10000x100 | 50000x50 |
|---|---|---|---|---|---|---|
| current sktime, pandas input (x8, fh=1..3) | 0.13 (1×) | 1.13 (1×) | 11.3 (1×) | 57.8 (1×) | — | — |
| current sktime, polars input (x8) | 0.154 (1.2×) | 1.29 (1.1×) | 13.3 (1.2×) | 60.1 (1×) | — | — |
| current sktime, dask input (x8, tag-patched) | 0.168 (1.3×) | 1.37 (1.2×) | 14.2 (1.3×) | — | — | — |
| current sktime, pandas (x7, fh=1..14) | — | — | 12.4 (1.1×) | — | — | — |
| x8 T1: same pandas _fit/_predict, no BaseForecaster boilerplate | 0.0918 (0.71×) | 0.833 (0.74×) | 8.93 (0.79×) | 47.5 (0.82×) | — | — |
| x6 hub -> pandas slices + stock inner pandas forecaster | 0.0887 (0.68×) | 1.03 (0.91×) | 9.67 (0.85×) | — | — | — |
| x6 hub numpy kernel, per-series loop | 0.00125 (0.0097×) | 0.00329 (0.0029×) | 0.0257 (0.0023×) | — | 0.189 | 0.896 |
| x6 hub numpy kernel, batched | 0.00095 (0.0073×) | 0.00173 (0.0015×) | 0.00454 (0.0004×) | — | 0.0118 | 0.0415 |
| x5 narwhals panel-native, pandas MI input | — | — | 0.447 (0.039×) | — | 1.32 | 4.74 |
| x5 narwhals panel-native, polars | — | — | 0.0765 (0.0068×) | — | 0.146 | 0.357 |
| x5 narwhals panel-native, polars lazy | — | — | 0.08 (0.0071×) | — | 0.134 | 0.393 |
| x5 narwhals panel-native, pyarrow | — | — | 0.128 (0.011×) | — | 0.336 | 0.716 |
| x5 narwhals panel-native, duckdb | — | — | 0.249 (0.022×) | — | 0.386 | 1.03 |
| x5 narwhals panel-native, dask | — | — | 2.4 (0.21×) | — | 3.7 | 8.06 |
| x7 pandas groupby (engine-native) | — | — | 0.0655 (0.0058×) | — | 0.133 | 0.45 |
| x7 polars eager | — | — | 0.00778 (0.00069×) | — | 0.0157 | 0.0632 |
| x7 polars lazy | — | — | 0.00653 (0.00058×) | — | 0.00937 | 0.0243 |
| x7 duckdb | — | — | 0.0519 (0.0046×) | — | 0.129 | 0.427 |
| x7 numpy GroupedArray | — | — | 0.00217 (0.00019×) | — | 0.0127 | 0.0649 |

**T5. Peak RSS increase, MB** (ratio vs `x4 pd_current peak` in parentheses)

| approach | 1000x500 | 10000x100 | 50000x50 |
|---|---|---|---|
| x4 current pandas pipeline (prepare+iter+recon) | 32.8 (1×) | 90.9 (1×) | 316 (1×) |
| x4 current polars-input pipeline | 81.5 (2.5×) | 142 (1.6×) | 461 (1.5×) |
| x4 adapter[polars] native pipeline | 35.9 (1.1×) | 108 (1.2×) | 288 (0.91×) |
| x4 slices held: current polars-input | 8.94 (0.27×) | 37.2 (0.41×) | 171 (0.54×) |
| x4 slices held: adapter[polars] | 7.97 (0.24×) | 16.1 (0.18×) | 66.2 (0.21×) |
| x3 prepare_VectorizedDF from pandas | 12.2 (0.37×) | 31.5 (0.35×) | — |
| x3 prepare_VectorizedDF from sktime polars | 69 (2.1×) | 100 (1.1×) | — |
| x3 pandas -> GroupedArray (hub) | 0.137 (0.0042×) | 0.137 (0.0015×) | — |
| x3 polars -> narwhals[polars] | 0.137 (0.0042×) | 0.137 (0.0015×) | — |
| x5 prepare current polars | 27.1 (0.83×) | 46.2 (0.51×) | — |
| x5 prepare narwhals[polars] | 6.8 (0.21×) | 28.8 (0.32×) | — |
| x9 polars -> pandas, sktime convert_to | 69.5 (2.1×) | 111 (1.2×) | 302 (0.95×) |
| x9 polars -> arrow-backed pandas flat | 3.85 (0.12×) | 7.79 (0.086×) | 0.137 (0.00043×) |
| x9 polars -> hybrid MI | 26.4 (0.81×) | 62.7 (0.69×) | 140 (0.44×) |
| x7 e2e naive: pandas groupby | — | — | 188 (0.59×) |
| x7 e2e naive: polars eager | — | — | 124 (0.39×) |
| x7 e2e naive: polars lazy | — | — | 58.2 (0.18×) |
| x7 e2e naive: duckdb | — | — | 92.4 (0.29×) |
| x7 e2e naive: numpy GroupedArray | — | — | 50.5 (0.16×) |

**T5b. Peak RSS increase, MB (x6_mem, one process per case)** (ratio vs `x6 ing_pd_current_ingress peak` in parentheses)

| approach | 50000x100 | 50000x100c4 | 5000x100 |
|---|---|---|---|
| ingress current pandas MI | 113 (1×) | 111 (1×) | — |
| ingress hub pandas MI | 80.7 (0.71×) | 80.8 (0.73×) | — |
| ingress current sktime polars | 451 (4×) | 679 (6.1×) | — |
| ingress hub polars | 29.7 (0.26×) | 29.9 (0.27×) | — |
| ingress current pyarrow | 410 (3.6×) | 656 (5.9×) | — |
| ingress hub pyarrow | 41.7 (0.37×) | 196 (1.8×) | — |
| e2e naive current sktime pandas | — | — | 122 |
| e2e naive current sktime polars | — | — | 140 |
| e2e naive hub batched (pandas in/out) | — | — | 15.1 |
| e2e naive hub batched (polars in/out) | — | — | 6.75 |

**T6. Lazy pushdown: select 1000 series × last 50 points from a 20000x200 parquet (x9, contiguous / random ids)**

| variant | time contiguous (s) | × vs sktime | peak MB | time random (s) | peak MB |
|---|---|---|---|---|---|
| B0_pandas_read_all | 0.6885 | 7.48 | 872.3 | 0.6764 | 870.3 |
| B1_sktime_eager_all | 5.146 | 1 | 786.5 | 5.151 | 784.1 |
| B1b_eager_then_filter | 0.1612 | 31.9 | 247.8 | 0.1592 | 228.4 |
| B1c_eager_all_adapter | 0.7422 | 6.93 | 659.3 | 0.7644 | 677.5 |
| B2_polars_lazy | 0.005878 | 875 | 30.9 | 0.0186 | 36.0 |
| B2w_polars_lazy_window | 0.01317 | 391 | 47.7 | 0.02445 | 53.3 |
| B3_duckdb | 0.0143 | 360 | 11.2 | 0.03545 | 20.8 |
| B3w_duckdb_window | 0.02736 | 188 | 25.4 | 0.04599 | 31.2 |
| B4_pyarrow_dataset | 0.05847 | 88 | 177.7 | 0.0581 | 183.8 |
| B5_lazy_then_sktime | 0.1053 | 48.9 | 50.9 | 0.1193 | 57.8 |
| B5b_lazy_then_adapter | 0.01549 | 332 | 43.9 | 0.02673 | 50.3 |

### Reading the tables per approach

| approach | iterate per slice | reconstruct | conversion / ingest | e2e NaiveForecaster | peak memory |
|---|---|---|---|---|---|
| current sktime, pandas | 115–274 µs (baseline). 2.3–3.2× worse when the slice has no stored freq | baseline 1 ms → 0.17 s (10x500 → 10000x100) | prepare 2 ms → 78 ms | 11.3–13 ms per instance (0.13 s → 57.8 s) | pipeline +33 / +91 / +316 MB at 1k/10k/50k |
| current sktime, polars input | ≈ pandas (0.93–1.0×): it iterates the converted pandas frame | 1.8–4.2× (convert back) | 7–36× pandas prepare. Mtype check is 1.2 s of 2.6 s at 10k; ingress 9.8 s at 50k | 1.04–1.18× pandas | pipeline 1.5–2.5× pandas; ingress +451 MB vs +113 MB at 50000x100 |
| x4 adapter | pandas 0.95–1.2×. polars native 0.034–0.15× (≥100 series) | pandas 1.0–1.1×, polars 0.41–0.75× | pandas = today. Polars prepare 0.24× only with a known mtype, else 10–16× | unchanged unless the inner forecaster consumes polars. A polars→pandas slice costs 2.9–5.3× (S5p) | polars native pipeline 0.91–1.2× pandas. Slices held 0.18–0.24× |
| x5 narwhals | pandas 1.7–2.3× (fast) or 5.7–8.3× (generic). polars 0.17–0.36× (≥100 series) | 1.3–2.9× | polars prepare 0.19–0.24× of pandas prepare (≥1000 series), pandas prepare 1.7–2.1× | panel-native forecasters: pandas 25× faster, polars 148×, pyarrow 88×, duckdb 45×, dask 4.7× (1000x500) | prepare polars +6.8–29 MB vs +27–46 MB current |
| x6 hub | 0.6–5 µs (0.004–0.018×). Pandas rebuild: 24–27 µs trusted freq, 173–248 µs validated | 0.05–0.15× (0.06–0.74× when output is pandas MI) | 0.06–0.28× for pandas MI. 0.004–0.14× for sktime polars (258× at 10k, 358× at 50k) | stock pandas forecaster on hub slices 0.68–0.91×. Numpy kernel loop 0.002–0.01×, batched 0.0004–0.007× | ingress +30 MB vs +451 MB (polars), +81 vs +113 (pandas). e2e batched +7–15 MB vs +122–140 MB |
| x7 engine-native | n/a (no per-slice loop) | n/a | n/a | at 1000x500: pandas groupby 65 ms, polars eager 7.8 ms, polars lazy 6.5 ms, duckdb 52 ms, GroupedArray 2.2 ms, vs 11.3–12.4 s. At 50000x50: 24 ms (polars lazy) to 450 ms (pandas groupby) | at 50000x50: polars lazy +58 MB, numpy +50 MB, pandas groupby +188 MB |
| x9 arrow-backed pandas | not separately benchmarked (x9 has no per-slice bench; 13/14 Series and 14/15 Panel forecaster tests pass on the hybrid layout per x9 report) | n/a | flat 0.05–0.07× adapter, hybrid MI 0.66–0.88× adapter | not run: the arrow MI is rejected by `check_is_scitype`; e2e on the hybrid is unmeasured | flat ≤8 MB vs +111–302 MB convert_to; hybrid 0.44–0.81× |

### Not run, and why

- **x3 / x4 / x5-vectorize at 50000x50.** x3 costs about 5 ms per slice summed over its 15 strategies,
  so a 50k pass would exceed 15 min. It was stopped after 10000x100. x4 and x5_bench_vectorize hit the
  600 s timeout while measuring 50000x50, and their JSON is written only at the end, so the 50k rows
  are lost. The completed rows were recovered from their printed logs. At 50k only x6 iterate and
  ingress, and x4 memory, are available.
- **Current sktime e2e at 10000x100 and 50000x50.** This was not measured, because the cost is linear
  at 11.3–13 ms per instance (x8). Extrapolated: about 115 s at 10k and about 580 s at 50k. This is
  **unmeasured/expected**. x8 20000x100 was dropped for the same reason (about 4 min per container per
  repeat).
- **x8 Hierarchical scitype and PolynomialTrendForecaster:** dropped for time. x8's dev report covers
  them. dask input in x8 is limited to n ≤ 1000 by the script default.
- **x5 forecasters at 50000x50, strategies other than `last`:** timeout. sktime baselines only at
  1000x500 (`--skip-sktime-above 1000`).
- **x7 memory for engine `f` (sktime at 50k):** timeout. The orphaned child (1.38 GB RSS after
  10 min) was killed by hand at 22:37:16, so it overlapped the first ~16 s of run 11 (the x9 10x500
  case only).
- **pandas 3:** not re-run in this phase.
- **x9 lazy pushdown at the script's "large" 50000x200 (10M rows):** not run. The eager sktime path
  already reaches +786 MB at 4M rows and would exceed the 3 GB cap at 10M. 4M rows were used instead.

### Caveats

- **Different scripts use different baselines.** x3's `pd->prepare_VectorizedDF` (1.73 s at 10k)
  is about 22× x4's (78 ms), because x3 builds its frame with `make_panel` (MultiIndex.from_tuples).
  Its per-slice freq re-inference is the effect x2 reported. x4, x5 and x6 use `make_panel_fast`
  (from_product). Ratios are therefore only taken within one script.
- **Horizons differ.** x8 uses fh=1..3, x7 fh=1..14, and x5/x6 their own fh. The current-sktime e2e
  cost barely depends on fh: x7's own baseline at 1000x500 is 12.4 s against x8's 11.3 s.
- **x8 instrumentation.** The instrumented wall is 1.03–1.08× the uninstrumented wall. The tables
  report the uninstrumented wall.
- **Speedup ratios across scripts** (for example x7 vs the x8 baseline) mix processes. The
  within-script ratios are listed next to them in the TL;DR.
- **Memory measures differ.** "peak RSS increase" is sampled every 5 ms in a fresh subprocess for
  x4_memory, x6_mem, x7 --mem and x9. It is in-process for x3 conv_mem and x5, where allocator
  caching makes the deltas lower bounds. x8's figure is the whole-process peak RSS, not a delta.
- **Repeats are small at large sizes:** 1–3 repeats, min taken. Single-process load average was about
  1.0 throughout.
