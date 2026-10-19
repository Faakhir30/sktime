# g1_trusted_path: a trusted per-slice "fast lane" for vectorized forecasters (critic gap G1)

**Question (wrap_critic G1).** The per-slice public-API re-entry was measured as the largest bucket
(a1 verifier 52–63% of vectorized `fit`; x8 33–61%). Earlier work built no prototype. If
`BaseForecaster._vectorize` hands slices that are already validated to the per-instance estimators
through a trusted internal entry, instead of their public `fit/predict/update/predict_interval`,
how much faster is it end to end? Does it stay output- and state-identical, pass sktime's tests,
and which invariants must it keep?

**Short answer.**

| outcome | result |
|---|---|
| speed-up by phase | fit 1.2–5.5x, update 1.3–3.2x, predict 0.9–1.45x, predict_interval ≈1.0x (0.87–1.16) |
| speed-up end to end | fit+predict 1.15–1.88x; whole workload (fit, predict, predict_interval, update, predict) 1.07–1.72x, at n ≥ 1000 |
| correctness | 100% output- and state-identical, on pandas and polars, Panel and Hierarchical |
| sktime tests | 7710/7710 pass in both modes, 0 pass delta |
| class coverage | 159/160 registered forecasters can use it; the remaining one falls back |

The gain is real, but it is mostly in `fit` and `update`. Re-entry is small on the predict side:
there the forecasters' own private code dominates, including the nested sktime transformers inside
it. So G1 resolves contradiction K2 as follows: *removing re-entry is worth about 1.2–1.9x end to
end, not 2–3x*.

## 1. What was built

`protos/g1_trusted.py` (no edits to `sktime/`). It works in either of two ways:

- `install()` / `trusted_path()` monkeypatches `BaseForecaster._vectorize`;
- `make_trusted(cls)` creates a subclass.

**Per call (once, not per slice),** it computes:

- converter functions looked up directly in `convert_dict` for `pd.DataFrame → y_inner_mtype` and
  `→ X_inner_mtype` (no `infer_mtype`);
- the y-metadata template, from one stock `check_is_scitype` per distinct column set;
- tag and config values: `capability:exogenous`, `X-y-must-have-same-index`, `remember_data`,
  `requires-fh-in-fit`, `capability:insample` / `pred_int:insample`, `capability:missing_values`
  and `capability:multivariate`;
- the in-sample check for a relative `fh`, cached per fh object.

**Per slice** there is a single pass over `y_schema.items(X=y_mi)`. y is sliced once; stock slices
it twice. Each slice then goes through one of these trusted entries:

| entry | what it does instead of the public method |
|---|---|
| `trusted_fit_slice(clone, y_i, X_i, fh)` | Skips `_reset_at` on a fresh clone (already in post-init state). Sets `_y_metadata` from the template; `has_nans` is recomputed per slice with numpy. Sets `_y_mtype_last_seen` and resets `_converter_store_y` before the direct converter call. Runs the per-slice X/y containment check, as stock does. Sets `_cutoff` (`idx[-1:]` keeps freq; stock `idx[[-1]]` + `freq=` re-validates it, about 0.6 ms). Writes `_y`/`_X` if `remember_data`, sets `_is_vectorized=False`, runs `check_fh(fh, freq=cutoff)` and writes `_fh`, does the in-sample check, then calls `_fit` and sets `_state="fitted"`. |
| `trusted_update_slice` | Same state writes as public `update`. `update_data` runs without its two `convert_to` calls (identity for plain pandas). For classes that inherit `BaseForecaster._update` (refit on the remembered data), it uses a trusted refit that writes the same warning, reset and state, and restores the same metadata. Stock calls public `fit` per slice here. |
| `trusted_predict_slice` | Converts the X slice, runs a trusted `_check_fh`, calls `_predict` / `_predict_interval` / `_predict_quantiles` / `_predict_var`, and converts back to `pd.DataFrame` through the direct converter with a frozen store. |

**Parallel backends.** The per-slice function is at module level, so it runs under
`backend:parallel` (verified with loky, n_jobs=2).

**Ablation mode `vectorize_est`.** This mode keeps stock `VectorizedDF.vectorize_est` iteration and
swaps only the per-slice entry. It cannot be used with process backends, because the entry methods
are monkeypatched onto the class in the parent process only.

## 2. Commands

All commands run from WT. `G1PY` is my own venv `venvs/g1_trusted_path`: the expenv freeze plus
pytest 9.1.1 and statsmodels 0.15.0. pandas 2.3.3, polars 2.0.0 and sklearn 1.9.1 are unchanged.

```
export PYTHONPATH=.:experiments/container_independence
$G1PY -W ignore experiments/container_independence/scripts/g1_trusted_bench.py --grid dev --modes stock,trusted
$G1PY -W ignore .../g1_trusted_bench.py --grid large --modes stock,trusted --est naive,poly --sc Panel,Hierarchical --cont pd --out results/g1_trusted_bench_large_pd.json
$G1PY -W ignore .../g1_trusted_bench.py --grid large --modes stock,trusted --est naive,poly --sc Panel --cont pl --out results/g1_trusted_bench_large_pl.json
$G1PY -W ignore .../g1_trusted_bench.py --grid dev --modes stock,trusted,trusted_ve --est naive,poly --sc Panel --cont pd --out results/g1_trusted_bench_ablation_ve.json
$G1PY -W ignore .../g1_invariants.py
$G1PY -W ignore .../g1_census.py
$G1PY -W ignore .../g1_profile.py --n 300 --est {naive,poly}        # -> results/g1_profile_n300_v2.txt
G1_TESTS="..." G1_TAG=unit|suite [G1_K=...] bash .../g1_pytest.sh schema_items off
python3 .../g1_tables.py > results/g1_tables.md
```

**Benchmark workload.** Each run is `fit(y_train, fh=[1,2,3])`, then `predict()`, then
`predict_interval(coverage=[.8,.9])` where the forecaster supports it, then `update(y_new)`
(5 new points per series), then `predict()`.

- Each phase is timed separately.
- Stock and trusted are interleaved in one process, after an untimed warm-up, and the minimum over
  reps is taken. That is 3, 2 and 1 reps for n ≤ 100, ≤ 1000 and 5000.
- Ratio = stock / trusted, so >1 means the trusted path is faster.

**Estimators.**

- `NaiveForecaster("last")` (naive)
- `PolynomialTrendForecaster(degree=1, prediction_intervals=True)` (poly)
- `ExponentialSmoothing(trend="add")` (ets): n ≤ 100, because statsmodels costs about 165–300 ms per series
- `ThetaForecaster(sp=1, deseasonalize=False)` (theta)
- `make_reduction(LinearRegression(), window_length=5)` (reduce, recursive)
- `TransformedTargetForecaster([Detrender, Naive])` (ttf_detrend)
- `TransformedTargetForecaster([Deseasonalizer(sp=7), Naive])` (ttf_deseas)

**Data.** Containers are pandas and sktime `polars_panel` / `polars_hierarchical`, on Panel
(`make_panel_fast`) and Hierarchical (`make_panel(n_levels=2)`) data with a daily DatetimeIndex.

**Equality checks.** For every row:

- point predictions, intervals, the predictions after update, and cutoff (incl. freq), both before
  and after update, compared with `assert_frame_equal` / `Index.equals` + freq;
- for n ≤ 100 additionally the full per-slice state: `_cutoff`, `_fh` (values, freq,
  is_relative), `_y`, `_X`, `_y_metadata`, `_y_mtype_last_seen`, `_converter_store_y`,
  `_is_vectorized`, `_state`, plus `get_fitted_params()` keys and non-estimator values.

## 3. Results

### 3.1 Equality: every check passed

- **Benchmarks.** 132 benchmark rows: dev 84, large 18, ablation 6 × 2 modes. Every output equality
  check is True. Every state comparison run at n ≤ 100 has no mismatch
  (`results/g1_trusted_bench_*.json`, `results/g1_tables.md`).
- **Invariant probes (`g1_invariants.py`).** **29/29 SAME** for `schema_items`: same output, or the
  same exception type. The `vectorize_est` ablation mode gets 28/29; it fails the loky case as
  explained in §1. The probes cover:
  - multivariate y with a univariate forecaster (`iterate_cols`), and multi-column `fh` reconstruct;
  - fh passed only in predict; absolute DatetimeIndex fh; in-sample fh;
  - `requires-fh-in-fit` with missing fh, or with a different fh in predict (same errors);
  - NaN in y, with and without `capability:missing_values`;
  - Panel X with SARIMAX through fit, predict and update;
  - X missing dates under `X-y-must-have-same-index` (same error);
  - X passed to a non-exogenous forecaster;
  - `remember_data=False`; `update(update_params=False)`;
  - `update_predict`, which falls back; `predict_residuals`, which fails identically in stock and is
    a pre-existing bug (a1.12);
  - `predict_quantiles` / `predict_var`;
  - PeriodIndex Hierarchical data; RangeIndex Panel data; irregular series (same stock error);
  - the loky backend; refitting the same instance;
  - opt-out by class tag, opt-out by config, and a class that overrides public `predict` (all three
    fall back, confirmed by `STATS`).

### 3.2 Speed (ratio stock / trusted; n ≥ 1000 rows from dev and large; full tables in `results/g1_tables.md`)

| estimator | fit | predict | fit+predict | predict_interval | update | whole workload |
|---|---|---|---|---|---|---|
| naive (pd) | 3.50–5.52 | 0.91–1.15 | 1.23–1.50 | 0.93–1.11 | 2.11–2.47 | 1.14–1.31 |
| naive (pl) | 2.47–5.05 | 1.01–1.07 | 1.28–1.46 | 0.87–1.12 | 1.94–2.26 | 1.16–1.27 |
| poly (pd) | 1.88–2.32 | 1.16–1.45 | 1.63–1.86 | 0.99–1.09 | 2.22–3.21 | 1.48–1.71 |
| poly (pl) | 1.89–2.14 | 1.05–1.34 | 1.68–1.88 | 0.94–1.06 | 2.43–2.81 | 1.46–1.56 |
| theta | 1.17–1.28 | 0.98–1.13 | 1.15–1.19 | 0.99–1.07 | 1.92–2.34 | 1.17–1.22 |
| reduce (recursive, LinearRegression) | 2.08–2.21 | 1.10–1.28 | 1.61–1.68 | n/a | 1.95–2.33 | 1.61–1.72 |
| ttf [Detrender, Naive] | 2.46–2.78 | 1.03–1.11 | 1.51–1.67 | 1.04–1.16 | 2.23–2.53 | 1.38–1.46 |
| ttf [Deseasonalizer, Naive] | 1.33–1.60 | 0.97–1.02 | 1.15–1.18 | 0.95–1.00 | 1.28–1.63 | 1.07–1.13 |
| ets (n = 100 only) | 0.83–1.20 | 0.91–1.17 | 0.84–1.20 | n/a | 0.65–1.14 | 0.75–1.11 (noise) |

**What the ratios show.**

- Ratios are flat in n from 100 to 5000 and in T from 100 to 500. They do not depend on pandas
  vs polars input: the slices are pandas either way, and polars conversion happens once per call.
- **Agreement with earlier estimates.**
  - fit+predict here: naive 1.23–1.50, poly 1.63–1.88.
  - x8's hand-written lite loop: naive 1.09–1.67, poly 1.66–2.42.
  - x8 verifier: naive 1.26–1.33, poly 2.01.
  - The integrated trusted path therefore reaches **most of the lite-loop ceiling** while keeping
    full API and state fidelity.
  - The a1 verifier's "2.1–2.7x" was for `fit` alone. Here fit alone is 1.9–2.3x (poly) and
    3.5–5.5x (naive).
- **Slicing once vs stock double slicing** (`vectorize_est` ablation, n=1000 Panel pd). Totals are
  naive 1.18 vs 1.19 and poly 1.46 vs 1.52. Single-pass iteration adds nothing measurable: slicing
  is a few % (x8: 85–96 µs per group).

### 3.3 Where the remaining time is (cProfile, n=300 Panel; shares only, `results/g1_profile_n300_v2.txt`)

| phase | stock cProfile s | trusted cProfile s | what dominates after the trusted path |
|---|---|---|---|
| naive fit | 2.30 | 0.35 (6.5x) | `items` (slicing) 32%, `clone` 26%, the rest small. No `check_is_scitype`/`convert_to` left; `get_cutoff` falls from 2.0 to ~0 calls per instance. |
| naive predict | 5.00 | 5.07 | `NaiveForecaster._predict` 97%. Inside it, `Lag.fit_transform`/`transform` run their own BaseTransformer checks: 22 `check_is_mtype` and 3 `convert_to` per instance remain *inside the inner code*. |
| naive predict_interval | 8.55 | 9.25 | `_predict_interval` 98%: `_predict` + `_predict_var` + 5 `convert_to` + 3 `infer_freq` inside the mixin/inner code. |
| naive update | 4.77 | 1.83 | `NaiveForecaster._update` → `update_data` (through `convert_to`) inside the inner `_update`. |
| poly fit | 4.40 | 2.26 | the forecaster's own `_fit` 76% (sklearn pipeline + Lag-free trend features). |
| poly predict | 1.32 | 1.07 | `_predict` 91% (incl. `infer_freq` 16%). |
| poly predict_interval | 6.50 | 7.15 | `_predict_interval` 97% (`_predict_var`, nested `convert_to`/`infer_freq`). |
| poly update | 8.33 | 3.29 | trusted refit: `_fit` 51%; stock had public refit re-entry plus 94 `check_is_mtype` per instance. |

So after the trusted path the per-slice framework work in `fit` is gone. What remains:

- In `fit`: slicing plus clone, about 0.6 ms per instance on naive.
- In predict-like methods: the inner `_predict*` implementations, which **re-enter the framework
  themselves**. That happens through nested transformers (`Lag` in Naive, Detrender/Deseasonalizer
  in TTF), through `convert_to` in the `_predict_interval` / `_predict_var` defaults of the proba
  mixin, and through `infer_freq`.

The trusted outer path cannot reach these. They need the same treatment for BaseTransformer (a
trusted `_fit/_transform` entry for nested transformers) and mtype-aware helpers inside the proba
mixin. Those are *unmeasured/expected*. x8 put the nested-transformer share at about 4.8 ms of
11.6 ms per instance for naive.

### 3.4 sktime tests with the trusted path monkeypatched in (`g1_pytest_plugin`, `-o addopts=""`)

| suite | stock | trusted (`schema_items`) | trusted-path calls / fallbacks |
|---|---|---|---|
| `sktime/forecasting/base/tests` + `forecasting/tests/{test_naive,test_theta,test_exp_smoothing,test_reconcile}.py` + `forecasting/compose/tests` | 940 passed, 151 skipped | **940 passed, 151 skipped** | 576 / 0 |
| `forecasting/tests/test_all_forecasters.py -k "Naive… or PolynomialTrend… or ExponentialSmoothing or Theta… or TransformedTarget… or RecursiveTabular… or DirectTabular… or SARIMAX or Croston or STLForecaster"` | 6770 passed | **6770 passed** | 8157 / 1444 (`update_predict_single`, by design) |
| first run, base tests only (v1 code) | 354 passed, 106 skipped | 354 passed, 106 skipped | 42 / 0 |

**Pass delta: 0** (7710/7710 pass in both modes). Run times were 13.3 min stock vs 14.2 min trusted
for the suite, and 7.3 vs 6.6 min for the unit tests. The test data are tiny, so test time is
dominated by other things.

Notes:

- `sktime/datatypes/tests/test_vectorize.py` was dropped because it fails *in stock* on the dask
  mtype (`python_type` tag bug, as already found by x8). It never calls `BaseForecaster._vectorize`
  (0 trusted calls).
- The first combined attempt hit the 20-minute timebox before finishing, and was replaced by the
  split runs above.

## 4. Invariants the trusted path must preserve, and the opt-out

Census (`g1_census.py`, 160 registered forecasters):

- 124 forecasters are not panel-native, so they vectorize via VectorizedDF; 123 of those are
  eligible.
- 1 class overrides a guarded public method: `ChronosForecaster` overrides `predict`.
- 42 have `requires-fh-in-fit`, 71 have `capability:insample=False`, and 16 have
  `capability:pretrain`.
- Series inner mtypes: 79 `pd.Series`, 67 `pd.DataFrame`, 6 both, 8 "anything" (incl. numpy and
  polars).

**Invariants, each tested in §3.1 or §3.4:**

1. **State written by the public method must be identical.** That means `_y_metadata` (per slice:
   feature names; `has_nans` per slice; mtype `pd.DataFrame`), `_y_mtype_last_seen`,
   `_converter_store_y` (reset and then filled by the y conversion; frozen copy for the back
   conversion), `_cutoff` (pandas Index, freq kept), `_fh` (the *shared* outer FH object, with
   `check_fh(fh, freq=slice_cutoff)` per slice), `_y`/`_X` per `remember_data`, `_is_vectorized=False`
   and `_state`. Note that stock `check_fh` *mutates* the shared FH's freq per slice, and the trusted
   path does the same on purpose.
2. **No skipped overrides.** If a class overrides any of `fit`, `predict`, `update`,
   `predict_interval/quantiles/var`, `_check_X_y`, `_check_X`, `_update_y_X`, `_set_cutoff_from_y`,
   `_set_cutoff`, `_check_fh`, `check_is_fitted` or `_reset_at`, it falls back to stock
   (`GUARDED`, checked per class by identity).
3. **Validation that only happens per slice in stock must stay per slice, or be done once
   equivalently.**
   - `check_equal_time_index(X, y, "contains")` under `X-y-must-have-same-index`. The outer call
     skips it, because X is Panel there.
   - The has_nans error.
   - The `requires-fh-in-fit` errors and the in-sample check. These run per slice for absolute fh
     and are cached for relative fh.
   - Validation that the outer call already did is dropped: scitype, categorical, missing values on
     the full frame.
4. **Fresh-clone semantics.** `_reset_at("pretrained")` is skipped only when the clone is in state
   `"new"` (reset of a fresh clone is a no-op). A `"pretrained"` outer estimator falls back for
   `fit`.
5. **Default `_update` (refit on `_y`).** The trusted refit must reproduce stock exactly: warn,
   reset, rewrite `_y`, cutoff and `_fh`, call `_fit`, then restore `_y_mtype_last_seen`,
   `_y_metadata` and `_converter_store_y`. It applies only when the class does not override
   `_update`.
6. **Methods not covered fall back.** `update_predict_single` (and therefore `update_predict`) and
   `predict_proba` keep the stock per-slice public path.
7. **Tags with no effect.** `fit_is_empty` is declared but not read by `BaseForecaster.fit`.
   `capability:categorical_in_X` and `capability:exogenous` are applied once at the outer level, and
   X is dropped per slice when the forecaster is not exogenous.

**Opt-out.**

- **Class tag `"capability:trusted_vectorize": False`**, for estimator authors (default True).
- **Config `trusted_vectorize=False`**, for users.
- Instance-level `set_tags` does **not** work as an opt-out. `fit` calls `_reset_at` → `reset()`,
  which wipes dynamic tags before `_vectorize` runs (observed: the tag read back False before fit,
  but the path still ran trusted).

## 5. Impact statements

1. **Trusted per-slice path.**
   - Phase speed-ups: fit 1.2–5.5x, update 1.3–3.2x, predict 0.9–1.45x, predict_interval ≈1.0x.
   - End to end: fit+predict 1.15–1.88x; the full fit/predict/interval/update/predict workload
     1.07–1.72x (n = 1000–5000, pandas and polars, Panel and Hierarchical, 7 forecasters).
   - Fidelity: 100% output- and state-identical, 7710/7710 sktime tests pass, and 159/160
     registered forecasters are eligible.
2. **It works for every container.** It is container-agnostic. The polars ratios match the pandas
   ratios because slices are pandas either way. It is therefore the framework lever that should
   come *before* any adapter-vs-narwhals decision. On top of a ≤1.05x container-only change it
   raises the realistic e2e gain to about 1.2–1.9x.
3. **Re-entry is a fit/update cost.** Predict-side gains are capped by inner code that re-enters
   the framework itself: nested BaseTransformer calls, proba-mixin `convert_to` and `infer_freq`.
   A follow-up "trusted nested transformer + mtype-aware proba mixin" is needed before predict
   gains appear (unmeasured/expected).
4. **Iteration is not the bottleneck.** Slicing once vs twice makes no measurable difference
   (ablation 1.18 vs 1.19, 1.46 vs 1.52).
5. **Engineering cost.** The prototype is about 700 LOC in a single module. Moving it into
   `_base.py` would mostly mean refactoring the public methods into "validate" plus a shared
   "trusted core" that both use (unmeasured LOC).

## 6. Caveats

1. **Noise.** The box is a shared 4-core machine; up to 3 of my processes ran concurrently with
   other agents. Ratios come from interleaved runs in one process, and phase ratios within ±10% of
   1 are noise, e.g. predict_interval 0.87–1.16. The ets rows (n ≤ 100, 165–300 ms per series inside
   statsmodels) are pure noise around 1.0. n=5000 rows are single-rep.
2. **Scope.** Only y-only workloads were timed. X paths were verified for equality (SARIMAX
   invariants) but not timed. `predict_proba`, `update_predict` and `predict_residuals` were not
   accelerated.
3. **Fit includes clone.** Stock's separate `clone` pass is merged into the trusted fit pass, so the
   fit ratio includes clone in both modes.
4. **Partial state check at large n.** Per-slice state comparison was run for n ≤ 100. For
   n ≥ 1000 only outputs and cutoffs were compared.
5. **The trusted path copies stock logic** (`update_data` fast path, `_check_fh`, default
   `_update`). A real implementation must share it with the public methods, or the two will drift.
6. **Environment.** Tests and benchmarks used my own venv (expenv freeze + pytest + statsmodels),
   not EXP_PY, because EXP_PY lacks both.
