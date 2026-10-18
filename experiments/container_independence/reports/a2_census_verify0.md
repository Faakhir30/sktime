# a2_census_verify0 — adversarial verification of the a2_census forecaster census

Lens: independent re-classification of a random sample of 25 forecasters, plus independent
re-counts and re-runs of the decision-relevant claims.

Environments: `EXP_PY` (pandas 2.3.3) for the census and recounts; the a2 venv
(`scratchpad/venvs/a2`, which has statsmodels, pmdarima and the other adapter libraries) for the
enforcement re-runs. No sktime source was modified; monkeypatches live in my own scripts.

## Scripts and commands (all run in this session)

```
cd /home/user/sktime
EXP_PY=/tmp/claude-0/-home-user-sktime/c9329e73-89f5-514d-ae5a-9180bf0e4a71/scratchpad/expenv/bin/python
A2_PY=/tmp/claude-0/-home-user-sktime/c9329e73-89f5-514d-ae5a-9180bf0e4a71/scratchpad/venvs/a2/bin/python
# 1. re-run original census (deterministic) -> identical numbers, 7.3 s
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/a2_census.py
# 2. independent recount (tags via get_class_tag, regex over inspect.getsource of class chain), random sample seed 20261018
PYTHONPATH=. $EXP_PY -W ignore experiments/container_independence/scripts/a2_census_verify0_recount.py
# 3. last-window fix experiment (monkeypatch _BaseWindowForecaster._get_last_window -> to_numpy()[-wl:])
N=DirRecTabularRegressionForecaster,DirRecTimeSeriesRegressionForecaster,DirectTabularRegressionForecaster,DirectTimeSeriesRegressionForecaster,MultioutputTabularRegressionForecaster,MultioutputTimeSeriesRegressionForecaster,RecursiveTabularRegressionForecaster,RecursiveTimeSeriesRegressionForecaster
PYTHONPATH=.:experiments/container_independence/scripts $A2_PY -W ignore experiments/container_independence/scripts/a2_trace.py --mode numpylike --names $N --out experiments/container_independence/results/a2_census_verify0_lw_baseline.json
PYTHONPATH=.:experiments/container_independence/scripts $A2_PY -W ignore experiments/container_independence/scripts/a2_census_verify0_lastwindow.py --mode numpylike --names $N --out experiments/container_independence/results/a2_census_verify0_lw_patched.json
# 4. taint-loss test of a2_trace's instrumentation
PYTHONPATH=.:experiments/container_independence/scripts $A2_PY -W ignore experiments/container_independence/scripts/a2_census_verify0_taintloss.py
# 5. third-party source reads: darts 0.48.1 and neuralforecast 3.3.0 wheels from PyPI, unzipped into
#    scratchpad/a2v0_dl and read with grep only (not imported or executed)
```

## Raw outputs

**Recount (script 2)**
```
all_estimators n = 160
non-pandas y_inner: 8 ['ConformalIntervals','DontUpdate','FallbackForecaster','ForecastByLevel','MultiplexForecaster','Permute','UpdateEvery','UpdateRefitsEvery']
panel/hier y_inner: 36          # = a2's 29 + the 7 "all-mtypes" composites that also list pd-multiindex (checked)
regex chain-source: to_absolute 122 (exact match with a2), self.cutoff|self._cutoff 149, either 149
```
The cutoff count of 149 comes from a regex over every method in the chain. a2's 127 / 128
scanned only the inner methods plus their helpers, so a2's 80% is a lower bound and the true
figure is about 93%.

**Last-window experiment (script 3)**
```
baseline numpylike: 0/8 ok   (all 8 first-violate at forecasting/base/_sktime.py:151, step predict)
patched  numpylike: 8/8 ok   (fit / predict / update+predict all complete)
```

**Taint-loss test (script 4)**
```
input type: TSeries
Lag(1).fit_transform output type: Series traced: False
pd.concat([lagged, plain]) traced: False
pd.concat([traced, plain]) traced: True
traced.shift(1) traced: True
```
a2's trace recorded these class-level accesses for `NaiveForecaster`: shape, `__len__`, index,
name, shift, arith, ufunc. Yet its default `strategy="last"` predict path runs `pd.concat`,
`sort_index`, `ffill`, `bfill` and `.loc["b"]` at `naive/_naive.py:377-381`, on the output of
the `Lag` transformer. None of these were recorded or enforced.

## Independent re-classification of 25 random forecasters (seed 20261018)

Each verdict below comes from reading the code: file:line plus the relevant ops.

| class | a2 static | my verdict | evidence |
|---|---|---|---|
| AutoResearchForecaster | B3/V1 | V1 (really a meta/AutoML composite, B4) | agentic/_autoresearch.py:733-834: LLM picks blueprint, `craft()` + child fit/predict |
| BATS | V1 | V1 | adapters/_tbats.py:95-117 passes y to tbats (check_array → numpy); output wrap only |
| BoxCoxBiasAdjustedForecaster | V2 | V2 | boxcox_biasadj.py:175-191 column-MultiIndex masks + `.loc` on the child's interval output |
| ChronosForecaster | V1 | V1 | chronos.py:597 `.values.reshape`; :632-635 output `.loc` |
| Croston | V1 | V1 | croston.py:113,172 `to_numpy().flatten()` |
| DartsXGBModel | V4 | V4 (own type, but not pandas-bound, see claim 8) | adapters/_darts.py:194 `TimeSeries.from_dataframe` |
| DirRecTabularRegressionForecaster | V2 | V2 | numpy sliding window (_reduce.py:150-191) + `_sktime.py:151` last-window `.loc` |
| DirectTabularRegressionForecaster | **V3** | **V2** | default local path is numpy (`_concat_y_X` → `to_numpy`, `Zt` array); groupby/concat only on the `pooling="global"` / transformers path (_reduce.py:194-216, 1833) |
| ForecastByLevel | V1 | V1 | `_DelegatedForecaster`; per-level split is done by framework vectorization |
| ForecastingOptCV | V1 | V1 | delegated tuner |
| ForecastingRandomizedSearchCV | **V2** | **V1** | model_selection/_base.py:233-277: `.loc`/`.iloc` on the CV results table, not on series data |
| GreykiteForecaster | V4 | V4 | greykite.py:450 builds `pd.DataFrame({"ts","y"})` for greykite |
| HierarchyEnsembleForecaster | **V2** | **V3** | compose/_hierarchy_ensemble.py:224-304, 322-340: Aggregator plus routing by MultiIndex `droplevel` / `get_level_values` / `"__total"`. Hierarchy semantics are the algorithm (same reasoning a2 applied to ReconcilerForecaster) |
| KronosForecaster | **V1** | **V4** (vendored) | kronos.py:268-300 builds a `pd.DataFrame` plus timestamp `pd.Series` for `sktime.libs.kronos.KronosPredictor.predict`, which requires `isinstance(df, pd.DataFrame)` and `.dt` time features (libs/kronos/kronos.py:593-616) |
| LTSFTransformerForecaster | V2 | V2 | adapters/_pytorch.py:29-33 per-instance `y.loc[instance_id]`; :514 `.values` |
| NeuralForecastGRU | V4 | V4 (accepts polars natively, see claim 8) | adapters/_neuralforecast.py:420 `fit(df=long df)` |
| StackingForecaster | V1 | V1 | compose/_stack.py:119-132 positional `iloc` splits, `np.column_stack` |
| SundialForecaster | V2 | V2 | groupby(level) only in the `_pretrain` instance split (sundial.py:715-723); fit/predict `to_numpy` |
| TBATS | V1 | V1 | same as BATS |
| TimeMoEForecaster | V2 | V2 | timemoe.py:471-474 `.values.reshape`; MultiIndex rebuilt for output |
| TimerForecaster | V1 | V1 | timer.py:164 `.values.astype` |
| TransformSelectForecaster | V1 | V1 | category via `iloc[0,0]` on transformer output; delegates |
| UpdateRefitsEvery | V1 | V1 | stream/_update.py:416-470: cutoff arithmetic on index metadata, `update_data` (framework) |
| VARReduce | V2 | V2 (but a2's own V3 rule should have fired) | var_reduce.py:205 `pd.concat([data.shift(i)…])` positional lag matrix. The static taint missed `shift` because the parameter is named `data`; the dynamic trace caught it |
| WindFMForecaster | **V1** | **V4** (vendored) | windfm.py:340-366, 417-423 `pd.DataFrame` + timestamps into `sktime.libs.windfm` predictor (same API pattern as Kronos) |

**Agreement.**
* Exact verdict: 20/25 = **80%**. a2 reported 84.6% on its own sample.
* Coarse (V1|V2 / V3 / V4): 21/25 = **84%**.
* Static V1 was wrong in **2 of 13** sampled cases (Kronos, WindFM). This refutes a2's
  "static V1 and V4 right in every sampled case".
* Static V3 was wrong in 1 of 1 (DirectTabular). This matches a2's finding that V3 over-counts.
* Static V2 hid one V3 (HierarchyEnsemble). That is the same failure mode a2 found for
  ReconcilerForecaster.
* Sample mix: static V1 13 / V2 8 / V3 1 / V4 3. My labels: V1 12 / V2 7 / V3 1 / V4 5.
  V1+V2 is 21/25 static and 19/25 manual (76%).

**Population follow-up for the V1→V4 error.** A grep of `sktime/libs` for pandas-typed APIs
(`isinstance(..., pd.DataFrame)`, `.dt.`, `pd.to_datetime`) hits only `kronos` and `windfm`. A
grep of `sktime/forecasting` for DataFrame-taking foundation APIs (`PandasDataset`,
`from_long_dataframe`, `forecast_on_df`, …) hits only the already-V4 gluonts wrappers. So the
population-level V1→V4 correction is +2 classes, not 15% of 63.

## Claim verdicts

* **a2_census.1 — confirmed.** Re-running gives identical output: 160 classes, 120
  implementations, B1 7 / B2 3 / B3 99 / B4 51, 61 composite, 99 third-party, 10 both. The
  registry count was re-checked independently (160).
* **a2_census.2 — partially confirmed.** Static numbers reproduce exactly (V1 63 / V2 56 /
  V3 13 / V4 28; 74.4% V1+V2).
  * My independent sample gives 76% V1+V2 (19/25), so the headline holds directionally.
  * The per-verdict split is shakier than "±15pp V2/V3" suggests:
    * V1 is not error-free (Kronos and WindFM are V4).
    * V3 mixes optional-path ops (DirectTabular) with real hierarchy logic missed as V2
      (HierarchyEnsemble).
    * The V3 rule is not applied consistently (VARReduce `shift` was missed).
  * My corrected population estimate: V1 ≈ 61, V4 ≈ 30, V1+V2 ≈ 116-117 / 160 ≈ 73%.
  * "Zero core-math change" for V1 composites is non-transitive, which a2 itself notes. Tuners
    run on `evaluate()` and the CV splitters, which are pandas-index based.
* **a2_census.3 — partially confirmed.** The reported confusion matrix is internally
  consistent, but on an independent sample:
  * verdict accuracy is 80% (not 84.6%);
  * "static V1 and V4 were right in every sampled case" is **refuted** for V1 (2/13 wrong).
* **a2_census.4 — partially confirmed, and an upper bound.** The JSON files show 57 / 68 / 71
  out of 85 as reported. But the enforcement only covers objects that still carry the traced
  pandas subclass:
  * internal sktime transformers (e.g. `Lag`) return plain pandas, and so does `pd.concat`
    with an untraced first frame (script 4);
  * so NaiveForecaster's default `strategy="last"` path (`concat` / `sort_index` / `ffill` /
    `.loc["b"]`, `_naive.py:377-381`) passed predict unenforced. It failed only later, at
    `_predict_var`'s `shift`.
  * Forecasters whose pandas ops run on derived frames (Theta's Deseasonalizer, Lag-based
    paths, Aggregator) can be false "ok"s.
  * So 83.5% / 67% are upper bounds. How many are false "ok"s is unmeasured.
* **a2_census.5 — partially confirmed (understated).** It is **8** of the 14 numpy-like
  failures, not 7: the 8 tabular / time-series reducer classes in `a2_enforce_numpylike.json`
  all first-violate at `_sktime.py:151`. The report's own list of "the other 7" names only 6
  classes.
  * Going beyond first-violation, I replaced that one method with
    `to_numpy()[-window_length:]`: all **8/8** then complete fit / predict / update+predict
    under numpylike enforcement (0/8 before).
  * So fixing one line removes 57% of the remaining failures, not 50%.
  * Caveat: only the univariate, no-exog, 48-point test case was run.
* **a2_census.6 — confirmed.** Independent tag read: exactly the same 8 non-pandas
  `y_inner_mtype` classes, all passthrough or delegating composites, and no leaf forecaster
  declares a non-pandas inner mtype.
  * Panel / hier is 36 if the 7 all-mtype composites are counted, and 29 without them. That
    matches a2.
* **a2_census.7 — confirmed (to_absolute exact; cutoff is a lower bound).**
  * A regex over the class-chain source gives exactly 122 for `to_absolute`.
  * `self.cutoff` / `self._cutoff` appears in 149 / 160 (93%) chains, against a2's 127 / 128.
    a2 scanned only the inner methods, so its 80% understates the reliance.
* **a2_census.8 — partially confirmed; the framing is refuted for 8 of the 28.** I read the
  package source from PyPI wheels:
  * **darts 0.48.1** `TimeSeries.from_dataframe` takes "anything which can be converted to a
    narwhals DataFrame (pandas, polars, …)" (`darts/timeseries.py:55,697-699,784`). It also
    has `from_times_and_values` / `from_values` (`:1210,1321`), which is exactly a "numpy
    values + time index" constructor.
  * **neuralforecast 3.3.0** `NeuralForecast.fit(df=…)` accepts "pandas, polars or spark
    DataFrame" (`neuralforecast/core.py:26,850,998`).
  * So the 3 darts and 5 neuralforecast classes are **not** stuck with pandas whatever sktime
    chooses. A polars long frame (sktime's `polars_panel` layout is close to
    `unique_id/ds/y`) could be passed natively. That is a place where the polars backend
    actually pays off.
  * In the other direction, KronosForecaster and WindFMForecaster (vendored, pandas-typed
    APIs) are missing from V4.
  * Corrected: about 30 classes are bound to a third-party input type. Of these:
    * strictly pandas-only: about 22;
    * if the 7 gluonts / pytorch-forecasting loaders are bypassed: about 15;
    * if the vendored Kronos / WindFM APIs are edited: about 13 (8%).
* **a2_census.11 — confirmed.** The line citations check out (`_naive.py:377-381`
  concat / ffill / bfill / `loc["b"]`; `_naive.py:522-571` shift / rolling / groupby;
  `mira.py:316` `to_numpy(float32)`; `mira.py:219-227` output wrap).
  * Nuance: Naive's "last" uses the `Lag` transformer plus ffill. Every op has a trivial
    positional equivalent, so it is a small rewrite.

## Things a2 missed or got wrong
1. **Arithmetic error:** the `_sktime.py:151` cluster is 8 classes, not 7. Fixing it was
   measured here (8/8 then pass), not extrapolated.
2. **Instrumentation hole:** the traced pandas subclass is lost through sktime transformers
   and `pd.concat` with an untraced first frame. Enforcement pass rates are upper bounds, and
   "0 dynamic-only flags" partly reflects this blind spot. NaiveForecaster's main path is an
   example.
3. **V1 is not error-free:** Kronos and WindFM feed pandas-typed vendored APIs. They should be
   V4, or V2 if the vendored code is edited.
4. **V4 is not uniformly pandas-bound:** darts and neuralforecast accept polars natively, and
   darts accepts numpy + times. This bears directly on the polars-native argument: 8 classes
   could get real polars value with no conversion to pandas.
5. **Inconsistent V3 rule:** VARReduce's `shift` lag matrix was missed by name-based taint,
   and HierarchyEnsembleForecaster's MultiIndex-level routing was scored V2. It has the same
   C2-relevant dependency on MultiIndex level names as ReconcilerForecaster. Any narwhals /
   columns-only design has to provide named instance-key columns for these.
6. **Third-party "index only" claim:** the trace records `.index` access but does not trace
   what statsmodels does with the index object. statsmodels uses pandas date / freq semantics
   on it. So "array plus optional index" means a *pandas* Index for those adapters. That is
   metadata-sized, but it is still pandas.
7. **AutoResearchForecaster** is bucketed B3 (LLM) but is a meta / AutoML composite. The
   verdict is unaffected.

## Caveats of this verification
* One reader, 25 classes, so verdict judgements (especially V2 vs V3) are subjective. The
  evidence lines are listed so each one can be re-checked.
* The last-window experiment ran only the 8 reducers on a univariate monthly series with no
  exog, as in a2's setup.
* darts and neuralforecast claims come from reading their source and docstrings. Passing
  polars through sktime's adapters end to end was not run (unmeasured/expected).
* No timing claims were made or checked; this workstream is a static / dynamic census.
