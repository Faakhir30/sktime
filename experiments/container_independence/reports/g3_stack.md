# g3_stack: combined-architecture waterfall C0 → C5 (critic gap G3)

**Question (wrap_critic G3).** Each lever was measured on its own: the monotonic-check fix (x9),
known-mtype pass-through (x4), the trusted path (g1), the vectorized cutoff (a1 P10), the time axis
once per panel (a3), hub iteration (x6) and panel-native kernels (x7/g2). This report stacks them
and measures them **end to end**, with an equality check at every step. The question is whether
the gains add up, and which steps depend on the container.

## Short answer

| step | what it adds | marginal gain (×prev) | cumulative vs C0 | depends on container? |
|---|---|---|---|---|
| C1 | vectorized polars monotonic check + known-mtype pass-through | 1.05–1.34 | 1.05–1.34 | the mechanism works for any container; the gain is somewhat larger on polars/lazy (1.10–1.34) than on pandas (1.05–1.22) |
| C2 | g1 trusted per-slice path | 1.17–1.73 | 1.33–2.19 | no |
| C3 | outer cutoff vectorized, panel step once, `fh.to_absolute` once per unique cutoff | 1.05–1.53 | 1.55–3.03 | no (works on the pandas MI after ingress) |
| C4 | hub ingress + positional iteration, with trusted pandas views passed to *unchanged* forecasters, native egress | **0.93–1.14** (±noise) | 1.59–3.31 | ingress/egress spokes: yes. Iteration: no |
| C5 | panel-native g2 kernel (`G2Naive`, `G2PolynomialTrend`) | 32–2,330 | **106–4,560** | the kernel is numpy (no); ingress spokes per container: yes |

- **Without any forecaster rewrite (C4), the full stack reaches 1.6–1.9× for NaiveForecaster and
  2.5–3.3× for PolynomialTrend** end to end (fit + predict). This holds on pandas, polars and
  polars-lazy input, from 100 to 5000 series and T = 100–500. All outputs and the sampled
  per-slice state are identical to stock.
- **Rewriting the forecaster as a panel-native kernel (C5) gives 10²–10³×**:
  - Naive: 138–4,560×
  - Poly: 106–1,125×
- **Gains multiply roughly to C3, then stop.** Once re-entry is removed (C2–C3), the remaining time
  is the forecasters' own per-instance code:
  - Naive: nested `Lag` transformer, 63% of C4 time; `_predict_naive` 92%.
  - Poly: sklearn `_fit`, 63%.
  - Changing the container or the iteration at C4 adds at most ~1.1×.
- **Containers.** Only C1(a) and the C4/C5 ingress/egress spokes depend on the container. The
  levers that matter for unchanged forecasters (C2, C3) do not. At the same level, polars and
  lazy input end within ±20% of pandas. Laziness gives nothing here, because every level must
  `collect()` at the boundary.
- **Memory** (fresh-process peak RSS, 1000×100):
  - stock: 26–28 MB (pandas), 45–47 MB (polars), 57–59 MB (lazy)
  - C4: 12/22–24/35–37 MB
  - C5: 3–7/18–23/30–37 MB

  At 5000×100 (poly), peak RSS falls from 129/161 MB (C0) to 50/54 MB (C4) and 36/65 MB (C5).

## 1. What was built

- `protos/g3_stack.py`: `configure(level)` installs layers C1..C<level> cumulatively as
  monkeypatches, and `uninstall_all()` removes them. Nothing under `sktime/` is edited.
- `scripts/g3_stack_bench.py`: the benchmark, equality checks, and the fresh-process memory mode.
- `scripts/g3_tables.py`: produces the tables and `results/g3_stack.json`.
- `scripts/g3_profile.py`: cProfile of the residual time.

| level | layer (cumulative) | implementation |
|---|---|---|
| C0 | stock | `NaiveForecaster("last")`, `PolynomialTrendForecaster(degree=1, prediction_intervals=True)`, `remember_data=True` set explicitly (the current default; it silences the FutureWarning) |
| C1 | (a) x9 vectorized `is_monotonically_increasing`: one `(t >= t.shift(1).over(keys)).all()` expression, with a null guard that falls back to stock. (b) known-mtype pass-through: the mtype found by `check_is_scitype` / `check_is_mtype` (BaseForecaster, `_vectorize`, BaseTransformer namespaces), or produced by `convert` / `VectorizedDF.reconstruct`, is memoized per object (weakref, id-checked). `infer_mtype` in `convert_to` and `_vectorize.mtype` return it instead of re-checking. | patches in `_adapter.polars`, `_convert`, `_vectorize`, `forecasting.base._base`, `transformations.base._base` |
| C2 | g1 trusted per-slice path (`protos/g1_trusted.py`, `install("schema_items")`), unchanged | — |
| C3 | (a) a1 P10: outer `get_cutoff` on a pandas MI frame becomes one max over the time-level codes. (b) a3: the panel step is verified once in numpy over all within-instance diffs, and per-slice `_enforce_index_freq` sets that freq without re-validating it; Tick/Day offsets only, otherwise stock. (c) `fh._to_absolute` / `_to_relative` are memoized per (fh values, freq, cutoff value/type/freq). The stock `lru_cache` never hits, because `_HashIndex` has no `__eq__` (critic bug list #3). | patches in `_base`, `_vectorize`, `_fh` |
| C4 | `prepare_VectorizedDF` is replaced for univariate Panel/Hierarchical pandas-MI or eager-polars input with fixed-step datetime time. Ingress is a single hub pass: x6 `from_pandas` (MI codes, no groupby) or `from_polars`, then the g2 `NpGrouped` + a3 `Step`. **No pandas MultiIndex is built for polars input.** `_vectorize` then iterates positionally: per instance, a zero-copy `pd.DataFrame` view with a trusted-freq `DatetimeIndex` goes to g1 `trusted_fit_slice` / `trusted_predict_slice` of the **unchanged** forecaster. Predictions are concatenated in numpy and egressed once to the input container (pandas MI or polars). `forecasters_`, `cutoff` and per-slice state stay as in stock. | `hub_vectorize`, `_hub_egress` |
| C5 | g2 `G2NaiveForecaster("last")` / `G2PolynomialTrendForecaster(degree=1, prediction_intervals=True)` (`np_grouped` inner mtype, batched numpy) through the public API, with all of C1–C4 installed. Only C1 has any effect, because no vectorization happens. | `protos/g2_forecasters.py` |

The `pl_lazy` input is `pl.scan_parquet(<panel>.parquet)`. sktime has no lazy Series/Panel
mtype, and the forecasters need materialized data, so **every level collects once at the
boundary**, inside the timed region. The output is eager polars.

## 2. Commands

```
cd /home/user/sktime; export PYTHONPATH=.:experiments/container_independence
# main waterfall (EXP_PY: pandas 2.3.3, polars 2.0.0, numpy 2.2.6, sklearn 1.9.1); naive and poly ran as 2 parallel processes
$EXP_PY -W ignore experiments/container_independence/scripts/g3_stack_bench.py --sizes 100x100,1000x100,5000x100,1000x500 --est naive --cont pd,pl,pl_lazy --out experiments/container_independence/results/g3_stack_bench_naive.json
$EXP_PY -W ignore .../g3_stack_bench.py  (same) --est poly --out .../results/g3_stack_bench_poly.json
# fresh-process peak RSS (one process per case; C0 also without the g2 registration as a control)
$EXP_PY -W ignore .../g3_stack_bench.py --mem-sweep --sizes 1000x100 --est naive,poly --cont pd,pl,pl_lazy --control-no-g2 --out .../results/g3_stack_mem_a.json
$EXP_PY -W ignore .../g3_stack_bench.py --mem-sweep --sizes 5000x100 --est poly --cont pd,pl --control-no-g2 --out .../results/g3_stack_mem_b.json
# equality re-check with the final code (pd2) and under pandas 3.0.6
$EXP_PY  -W ignore .../g3_stack_bench.py --sizes 100x100 --reps 1 --out .../results/g3_stack_eq_pd2_final.json
G3_SCRATCH=$SCRATCH/pd3 $EXP_PY3 -W ignore .../g3_stack_bench.py --sizes 100x100 --reps 1 --out .../results/g3_stack_eq_pd3.json
$EXP_PY -W ignore .../g3_profile.py {4 naive pd | 4 poly pd | 5 naive pl}   # -> results/g3_profile_200x100.txt
python3 experiments/container_independence/scripts/g3_tables.py > experiments/container_independence/results/g3_tables.md   # also writes results/g3_stack.json
```

**Method.**
- Workload: a fresh estimator, then `fit(y, fh=[1,2,3])` + `predict()`, timed from native input
  to native output.
- Data: `make_panel_fast` (Panel, daily, 1 column). polars input is `convert_to(y, "polars_panel")`.
- Interleaving: levels C0..C5 run round-robin within each repeat in one process, after an untimed
  warm-up of every (est, cont, level) on a 20×30 panel.
- Repeats: 3 at 100×100, 2 at 1000×100, 1 at 5000×100 and 1000×500. The minimum is reported.
- Equality, at every level, against C0 on the same container:
  - prediction values, index and columns: `assert_frame_equal(check_exact=True)`, plus `allclose`
    at rtol 1e-9;
  - output type and polars schema;
  - pandas time-level `freq`;
  - `cutoff` (values + freq);
  - `forecasters_` index and shape;
  - per-slice state for the first and last instance: `_cutoff` + freq, `_y` + freq, `_y_metadata`,
    `_state`, `_is_vectorized`, `_fh`.

## 3. Results

### 3.1 Waterfall: seconds (× vs C0 / × vs previous level)

From `results/g3_tables.md`. Each row is fit + predict.

| est | cont | size | C0 | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|---|---|
| naive | pd | 100x100 | 1.33 | 1.20 (1.11/1.11) | 0.894 (1.49/1.34) | 0.772 (1.72/1.16) | 0.703 (1.89/1.10) | 0.0044 (305/161) |
| naive | pl | 100x100 | 1.29 | 1.11 (1.17/1.17) | 0.797 (1.62/1.39) | 0.745 (1.74/1.07) | 0.688 (1.88/1.08) | 0.0087 (148/79) |
| naive | pl_lazy | 100x100 | 1.34 | 1.16 (1.15/1.15) | 0.898 (1.49/1.30) | 0.737 (1.82/1.22) | 0.736 (1.82/1.00) | 0.0097 (138/76) |
| naive | pd | 1000x100 | 13.1 | 11.7 (1.11/1.11) | 8.95 (1.46/1.31) | 8.00 (1.63/1.12) | 7.38 (1.77/1.08) | 0.0086 (1,518/858) |
| naive | pl | 1000x100 | 13.7 | 11.6 (1.17/1.17) | 8.53 (1.60/1.37) | 8.09 (1.69/1.05) | 7.18 (1.91/1.13) | 0.0207 (660/347) |
| naive | pl_lazy | 1000x100 | 13.3 | 11.6 (1.15/1.15) | 9.74 (1.37/1.19) | 8.60 (1.55/1.13) | 7.97 (1.67/1.08) | 0.0546 (244/146) |
| naive | pd | 5000x100 | 73.9 | 62.2 (1.19/1.19) | 47.0 (1.57/1.32) | 36.7 (2.02/1.28) | 37.8 (1.96/0.97) | 0.0162 (4,560/2,330) |
| naive | pl | 5000x100 | 70.7 | 60.7 (1.17/1.17) | 47.8 (1.48/1.27) | 40.1 (1.76/1.19) | 39.4 (1.80/1.02) | 0.0519 (1,363/758) |
| naive | pl_lazy | 5000x100 | 68.1 | 59.0 (1.15/1.15) | 48.3 (1.41/1.22) | 39.2 (1.74/1.23) | 39.4 (1.73/0.99) | 0.083 (820/475) |
| naive | pd | 1000x500 | 12.9 | 12.3 (1.05/1.05) | 9.73 (1.33/1.27) | 7.72 (1.67/1.26) | 8.15 (1.59/0.95) | 0.0147 (880/555) |
| naive | pl | 1000x500 | 14.4 | 13.1 (1.10/1.10) | 10.3 (1.40/1.27) | 8.39 (1.71/1.23) | 7.95 (1.81/1.06) | 0.0473 (304/168) |
| naive | pl_lazy | 1000x500 | 15.1 | 12.5 (1.20/1.20) | 10.7 (1.40/1.17) | 8.69 (1.73/1.23) | 9.07 (1.66/0.96) | 0.0659 (228/138) |
| poly | pd | 100x100 | 1.09 | 0.896 (1.22/1.22) | 0.541 (2.01/1.65) | 0.404 (2.70/1.34) | 0.434 (2.51/0.93) | 0.0056 (193/77) |
| poly | pl | 100x100 | 1.01 | 0.809 (1.24/1.24) | 0.500 (2.01/1.62) | 0.387 (2.60/1.29) | 0.343 (2.93/1.13) | 0.0087 (116/40) |
| poly | pl_lazy | 100x100 | 1.10 | 0.854 (1.28/1.28) | 0.500 (2.19/1.71) | 0.376 (2.92/1.33) | 0.331 (3.31/1.14) | 0.0104 (105/32) |
| poly | pd | 1000x100 | 10.3 | 8.46 (1.22/1.22) | 5.83 (1.77/1.45) | 3.80 (2.72/1.53) | 3.72 (2.78/1.02) | 0.013 (797/286) |
| poly | pl | 1000x100 | 10.8 | 8.78 (1.23/1.23) | 5.39 (2.01/1.63) | 3.57 (3.03/1.51) | 3.72 (2.91/0.96) | 0.0239 (453/156) |
| poly | pl_lazy | 1000x100 | 10.8 | 8.63 (1.25/1.25) | 5.37 (2.01/1.61) | 3.74 (2.88/1.43) | 3.83 (2.81/0.98) | 0.0375 (287/102) |
| poly | pd | 5000x100 | 53.2 | 48.8 (1.09/1.09) | 32.3 (1.64/1.51) | 23.9 (2.23/1.35) | 21.8 (2.44/1.10) | 0.0473 (1,125/461) |
| poly | pl | 5000x100 | 54.6 | 45.2 (1.21/1.21) | 26.1 (2.09/1.73) | 19.6 (2.79/1.33) | 17.9 (3.06/1.10) | 0.096 (570/186) |
| poly | pl_lazy | 5000x100 | 59.6 | 45.0 (1.33/1.33) | 27.5 (2.17/1.64) | 20.3 (2.94/1.35) | 18.4 (3.24/1.10) | 0.189 (315/97) |
| poly | pd | 1000x500 | 11.7 | 9.76 (1.20/1.20) | 5.88 (1.99/1.66) | 4.28 (2.73/1.37) | 4.00 (2.93/1.07) | 0.0399 (293/100) |
| poly | pl | 1000x500 | 12.8 | 9.67 (1.32/1.32) | 5.90 (2.17/1.64) | 4.69 (2.73/1.26) | 4.18 (3.06/1.12) | 0.0724 (177/58) |
| poly | pl_lazy | 1000x500 | 12.7 | 9.43 (1.34/1.34) | 6.49 (1.95/1.45) | 4.58 (2.76/1.42) | 4.45 (2.84/1.03) | 0.107 (119/42) |

**Ranges of the marginal gain (× vs the previous level) over all 24 rows:**

| | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| naive | 1.05–1.20 | 1.17–1.39 | 1.05–1.28 | 0.95–1.13 | 76–2,330 |
| poly | 1.09–1.34 | 1.45–1.73 | 1.26–1.53 | 0.93–1.14 | 32–461 |

### 3.2 Equality

- **Main runs.** In all **24/24 rows**, every level C1–C4 is output-identical to C0 with
  `check_exact=True`. That covers type, polars schema, pandas time-level freq, cutoff + freq,
  `forecasters_` index/shape and the sampled per-slice state: 0 mismatches.
- **C5** (g2 kernels):
  - predictions match to rtol 1e-9; max relative error 0 (naive) and 4.5e-16 (poly);
  - cutoff is identical;
  - two documented differences:
    1. no per-instance `forecasters_` grid (API change, as in g2);
    2. for pandas output the MI time level has `freq=None`, where stock has `<Day>`. g2's pandas
       egress drops it; `assert_frame_equal` does not check level freq.
- **Re-check with the final code at 100×100** (`results/g3_stack_eq_pd2_final.json`): identical
  under pandas 2.3.3.
- **pandas 3.0.6** (`results/g3_stack_eq_pd3.json`, 100×100, all 6 est/cont pairs): identical at
  C1–C4. C5 has the same two documented differences.
  - The first pd3 run found **two hub bugs**, both fixed in the proto before the final pd3 and pd2
    runs:
    1. `to_offset(Timedelta(1 day))` is `24h`, not `Day`, in pandas 3, so `check_fh` raised
       "Frequencies from two sources do not coincide". The hub now takes the freq string from the
       source or Step.
    2. Casting to `datetime64[ns]` broke `_y` equality under pandas 3's `[us]` default.

  This is a concrete instance of K11/a3-type pandas-version fragility in any trusted
  time-axis code.
- The main pd2 waterfall ran with the pre-fix module. Neither fix changes behaviour for
  datetime64[ns] daily data on pandas 2, and the pd2 re-check confirms this.

### 3.3 What each layer actually did

Counters are from `STATS`, per fit+predict at 1000×100 (`results/g3_stack.json`, `levels.*.stats`).

- **C1:** 4,005 (naive) / 3,005 (poly) mtype re-inferences answered from the memo; 2,000 / 1,000
  still inferred.
  - The remaining naive misses are `convert_to` on objects freshly created inside `Lag`.
  - With C2, the slice re-entries vanish, so only 1,005 / 5 hits remain.
  - The pandas C1 gain (1.05–1.22) is mostly this per-slice/nested re-inference.
  - On polars, C1 additionally removes the per-series Python loop of the monotonic check. It
    runs once per `_check_X_y` and once per `convert_to`. That is worth about 1 s at 500k rows
    (x9), but only 1–2% of a 60–70 s stock run.
- **C3:**
  - `fh.to_absolute`: 999 hits / 1 miss per predict. Every instance has the same cutoff, so U=1,
    and the absolute horizon is built once per panel instead of 1000×.
  - Outer cutoff fast path: 2 calls (fit + `_update_y_X`). a1 measured this at ~20 ms at
    1000×100, so it is a small part of C3.
  - Trusted slice freq: **0 uses**. All 1000 slices already carry `freq` from the time level
    (`freq_present`=1000), so stock `_enforce_index_freq` never calls `pd.infer_freq` on this
    data.
  - **So C3's 1.05–1.53× is essentially the per-unique-cutoff fh memo.** By the counts above, the
    vectorized cutoff contributes ≤ ~0.3% of C0, and the step-once layer is a no-op on regular
    data where the level carries freq. The per-component split was not separately timed.
- **C4:** one hub ingress per fit and one egress per predict (`hub_fit`/`hub_predict` = 1); the
  groupby and the pandas-MI conversion of polars input are gone. The marginal 0.93–1.14×
  (median ≈ 1.05) is at the noise level of single-rep rows. Iteration and container were not the
  bottleneck after C3, consistent with g1's "slicing once vs twice makes no difference".

### 3.4 Where the time is after C4 (cProfile, 200×100, `results/g3_profile_200x100.txt`)

| case | total | dominant |
|---|---|---|
| C4 naive pd | 3.03 s | `NaiveForecaster._predict_naive` 2.79 s (92%), of which nested `Lag.fit_transform` 1.90 s (63%); 4,600 `check_is_mtype` calls inside the transformer (0.62 s) |
| C4 poly pd | 1.56 s | `trusted_fit_slice` 1.11 s, of which `PolynomialTrendForecaster._fit` (sklearn pipeline) 0.98 s (63%); `_predict` 0.29 s |
| C5 naive pl | 0.016 s | polars `check_is_scitype` 7 ms (`_check_polars_panel`) ≈ 45%; the kernel is the rest |

The residual cost after C3/C4 is per-instance **inner** code: nested-transformer framework
re-entry for Naive, and sklearn per-instance fitting for Poly. No container layer reaches it.
Only a trusted BaseTransformer entry (unmeasured, g1 §3.3) or a kernel rewrite (C5) does.

### 3.5 Peak RSS (fresh process per case, VmHWM delta; MB / s)

From `results/g3_stack_mem_{a,b}.json`. Single cold run, no warm-up.

| size | est | cont | C0 | C1 | C2 | C3 | C4 | C5 | C0 without g2 registered |
|---|---|---|---|---|---|---|---|---|---|
| 1000x100 | naive | pd | 26 / 13.1 | 26 / 12.2 | 21 / 10.0 | 18 / 7.7 | 12 / 7.3 | 3 / 0.01 | 27 / 13.0 |
| 1000x100 | naive | pl | 45 / 13.9 | 46 / 12.2 | 40 / 9.8 | 37 / 7.6 | 24 / 7.9 | 18 / 0.04 | 45 / 13.3 |
| 1000x100 | naive | pl_lazy | 57 / 14.9 | 57 / 12.5 | 52 / 10.1 | 53 / 8.1 | 37 / 8.0 | 30 / 0.04 | 57 / 13.2 |
| 1000x100 | poly | pd | 28 / 10.9 | 28 / 8.9 | 22 / 6.5 | 19 / 4.7 | 12 / 4.1 | 7 / 0.03 | 29 / 10.2 |
| 1000x100 | poly | pl | 47 / 11.5 | 48 / 10.3 | 42 / 6.7 | 39 / 5.1 | 22 / 3.9 | 23 / 0.04 | 47 / 10.3 |
| 1000x100 | poly | pl_lazy | 59 / 11.0 | 59 / 9.2 | 53 / 7.1 | 50 / 4.9 | 35 / 4.5 | 37 / 0.04 | 59 / 11.1 |
| 5000x100 | poly | pd | 129 / 54.4 | 127 / 46.3 | 98 / 28.4 | 87 / 22.5 | 50 / 19.7 | 36 / 0.17 | 129 / 51.2 |
| 5000x100 | poly | pl | 161 / 57.3 | 161 / 47.7 | 132 / 32.6 | 121 / 20.6 | 54 / 20.3 | 65 / 0.13 | 161 / 56.6 |

**Reading:**
- **C4 is the memory step, not a speed step.** It roughly halves peak RSS on pandas
  (18→12 MB, 87→50 MB) and cuts it by 1.4–2.2× on polars (121→54 MB at 5000). No pandas
  MultiIndex copy of the polars frame and no groupby materialization take place; slices are views.
- Polars/lazy peaks stay above pandas at C4/C5. They include the polars frame and the parquet
  collect, which the pandas case already holds before the run.
- **C1–C3 change memory little** (≤ ~33%, e.g. 129→87 MB, via fewer temporaries).
- The in-process VmHWM deltas from the main runs (`peak_rss_delta_mb_inproc`) are mostly 0–60 MB
  and not monotone. Allocator reuse after earlier runs masks them, so they are not used for any
  claim.
- **Control:** registering the g2 mtypes (needed for C4/C5 in the same process) changes C0 by
  1.00–1.06× in time (51.2 vs 54.4 s; noise level) and 0–2 MB in memory.

### 3.6 Container-dependent vs container-independent

| step | container-dependent part | container-independent part | measured on pandas vs polars |
|---|---|---|---|
| C1(a) monotonic | polars checker only | — | polars-only effect, a few % of e2e at these sizes |
| C1(b) known mtype | the check it skips is far more expensive for polars (x4/x9) | the memo mechanism is generic and also removes nested-transformer re-inference | pd 1.05–1.22, pl/lazy 1.10–1.34 |
| C2 trusted path | — | entirely (slices are pandas either way) | pd 1.27–1.66, pl/lazy 1.17–1.73 |
| C3 time once / fh memo / cutoff | — | entirely (operates after ingress on pandas objects) | similar on all inputs |
| C4 hub | ingress/egress spokes (x6 `from_pandas`/`from_polars`, native egress) | positional iteration, trusted views | ≈1.0–1.14 speed on every container; memory gain larger on polars |
| C5 kernel | ingress spokes (polars still pays a polars mtype check: 7 ms at 200×100) | numpy kernel | pd 193–4,560×, pl 116–1,363×, lazy 105–820× (pd fastest because ingress reads MI codes zero-copy and egress builds no polars frame) |

## 4. Impact statements

1. **Combined stack, unchanged forecasters:**
   - **Naive 1.59–1.91× and PolynomialTrend 2.44–3.31×** end to end (C4 vs stock), on pandas,
     polars and polars-lazy, 100–5000 series, T 100–500, 100% output- and state-identical.
   - The levers that deliver this (C1b, C2, C3) are container-independent.
   - Changing the container or iteration (C4) adds ≈1.0–1.14× in time but **1.4–2.4× lower
     peak RSS**.
2. **The levers multiply up to C3, then saturate.** For example, at 1000×100: 1.11 × 1.31 × 1.12
   ≈ 1.63 for naive, and 1.22 × 1.45 × 1.53 ≈ 2.72 for poly. What remains is per-instance inner
   code (Naive: nested `Lag` 63%; Poly: sklearn fit 63%). No container-layer work reaches it.
3. **Panel-native kernels are the only 10²–10³× lever:** C5 runs 105–4,560× faster than stock and
   57–2,330× faster than C4. With C1 in place, g2 on polars no longer pays the stock monotonic
   loop: 0.021 s at 1000×100 vs g2's own 0.33 s (g2 report, fh=1..10; a cross-report comparison
   with a different fh, *indicative*).
4. **Laziness does not pay** for these workloads. Every level must collect, because sktime has
   no lazy Series/Panel mtype. At C0–C4, lazy input is within about ±15% of eager polars at the same
   level. At C5 it is 1.1–2.6× slower than eager, because scan+collect dominates.
5. **Cheapest high-value step:** C3(c), memoizing `fh.to_absolute` per unique cutoff. In practice
   this means giving `_HashIndex` a cheap key and `__eq__`. It is a few lines in `_fh.py` and
   gives 1.05–1.53× on top of the trusted path (container-independent). C1(b) and C2 are next.

## 5. Caveats

1. **Noise.**
   - The machine has 4 cores; naive and poly ran as 2 parallel processes, and another agent's load
     may have been present.
   - Rows at 5000×100 and 1000×500 are single-rep, so marginal ratios within ±10% of 1 (most of C4)
     are noise.
   - Ratios come from interleaved runs in one process; absolute seconds are indicative.
2. **Scope.**
   - Only fit + predict, univariate Panel, daily fixed frequency, equal-length aligned series
     (U=1 unique cutoff). That is the best case for the fh memo.
   - With ragged ends, C3(c) computes once per unique cutoff. a3 P3 measured 1.6–3.8 ms per
     unique cutoff, so the gain shrinks as U approaches n (unmeasured here).
   - The calendar / irregular / tz paths of C3(b) and C4 fall back to stock or are not handled
     (C4 raises nothing; it simply does not take the hub route).
   - Not ported: `predict_interval`, `update`, exogenous X, Hierarchical input.
   - C4 raises `NotImplementedError` for other methods on a hub-fitted forecaster. That is a proto
     limitation, not a measured behaviour.
3. **C1(b) is a memo by object identity** (weakref + id check). It assumes objects are not
   mutated between check and convert in a way that changes their mtype. A real implementation
   would pass `from_type` explicitly (x4 recommendation) rather than memoize.
4. **C3(b) relies on private pandas internals** (`DatetimeArray._freq`, `_simple_new`). It had no
   effect on this data. Its value appears only where slices lack freq (e.g. pandas-3 equal-length
   panels / the K11 cases); this was not measured.
5. **C5 is g2's port, not stock.** It has no `forecasters_` grid. The pandas output time level
   loses `freq`. Poly matches only because g2 reproduces sklearn's `cond=tol` truncation (g2
   §1.2).
6. **Fresh-process memory runs are cold** (no warm-up), so their times include first-call
   overheads. The waterfall times come from the warm, interleaved runs.
7. **g2 registration** is active in the same-process waterfall for all levels. The control in §3.5
   bounds its effect on C0 at ≤ ~6% (noise level).
