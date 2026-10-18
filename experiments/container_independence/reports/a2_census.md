# a2_census — how real is the "100s of forecasters" problem?

**Question.** If sktime dropped pandas as the only inner container, how many forecasters
would need their core math rewritten? How many only need changes at the boundary (data in,
predictions out)? And how many are tied to a third-party library's own input format?

**Short answer.** There are 160 forecaster classes, built from 120 distinct
`(_fit owner, _predict owner)` implementations. Across them, the static census finds 180
input-side "hard pandas" call sites.

| verdict (inner repr = numpy values + time-index metadata) | static n | % | calibrated by 26 manual spot checks |
|---|---|---|---|
| **V1 boundary-only**: core is already numpy (or passthrough); pandas is used only to wrap outputs on `fh.to_absolute(cutoff)` | 63 | 39.4% | 63.0 (39.4%) |
| **V2 glue rewrite**: label ops (`.loc`, `reindex`, `X.loc[y.index]`, `reset_index`, output `concat`/`xs`) in glue; core math unchanged; mechanical positional replacements | 56 | 35.0% | 55.5 (34.7%) |
| **V3 core rewrite**: the algorithm itself is written in pandas (Lag/rolling/shift/groupby/combine_first, hierarchy via MultiIndex levels) | 13 | 8.1% | 13.5 (8.4%) |
| **V4 third-party pandas-bound**: the wrapped library consumes pandas or its own type (prophet, darts, neuralforecast, pytorch-forecasting, gluonts, skforecast, autots, greykite, hcrystalball, prophetverse, rpy2/fable, hypertrees) | 28 | 17.5% | 28.0 (17.5%) |

* **No change to core math (V1+V2): 119/160 = 74.4%.** Strictly boundary-only (V1): 39.4%.
* **Core rewrite (V3): 13 classes (8%).** These are the reducer family
  (Direct/Recursive/MultiOutput/DirRec tabular/TS, `RecursiveReductionForecaster`,
  `MCRecursiveProbaReductionForecaster`, `YfromX`, `ForecastX`), plus `NaiveForecaster`,
  `DummyGlobalForecaster`, `ForecastKnownValues` and `SquaringResiduals`. They come from 11
  distinct implementations. The manual check moved 3 of the 6 checked V3s to V2, so the true
  V3 count is probably smaller (see the calibration note below).
* **Bound to a third-party input format (V4): 28 (17.5%).** For 7 of these (LagLlama,
  MOIRAI, Moirai2, 4× PytorchForecasting) the model is torch, which accepts numpy. Only the
  data-loading wrapper (gluonts / pytorch-forecasting `TimeSeriesDataSet`) needs pandas.
  Libraries that are pandas-bound at their core cover 21 classes (13%).
* **Dynamic check (85 runnable forecasters, own code limited to a numpy-like surface).**
  71/85 (83.5%) complete `fit` / `predict` / `predict_interval` / `update` / `predict`.
  Under the strict "values + index metadata" surface, 57/85 (67%) complete. 7 of the 14
  numpy-like failures come from **one line**:
  `sktime/forecasting/base/_sktime.py:151` (`_cur_y.loc[start:cutoff]` in
  `_BaseWindowForecaster._get_last_window`).
* **Output index construction is centralized.** 122/160 (76%) call
  `fh.to_absolute*`, 127/160 (79%) read `self.cutoff` / `self._cutoff`, and 128 do at least
  one of the two. 25 use `get_expected_pred_idx`. The 31 that use neither are almost all
  composites or tuners that pass child output through. A time-index metadata object that
  implements `ForecastingHorizon.to_absolute(cutoff)` covers how all of them build their
  output index.

So the "100s of forecasters" concern is real in number, not in kind. About three quarters
of the forecasters only touch pandas at the edges. The pandas-heavy code is concentrated in
a few shared files: the statsmodels adapter, `_BaseWindowForecaster`, `compose/_reduce.py`,
`naive/_naive.py`, the `_pytorch.py` instance splitter and the gluonts / Prophet adapters.
Third-party lock-in affects about 1 in 6 forecasters and cannot be fixed on the sktime side.

## Methods

### Static census (`scripts/a2_census.py`)
* Registry: `all_estimators("forecaster", return_names=True)` → 160 classes. It lists
  classes whose soft dependencies are missing too.
* Tags read for each class: `y_inner_mtype`, `X_inner_mtype`, `capability:multivariate`,
  `capability:exogenous`, `capability:pred_int`, `python_dependencies`.
* **Composite detection.** A class counts as composite if a meta base class is in its MRO
  (`_HeterogenousMetaEstimator`, `_DelegatedForecaster`, `BaseGridSearch`, `_Pipeline`,
  `_HeterogenousEnsembleForecaster`, `_Reducer`, `_ReducerMixin`, `_StreamDataPoolMixin`),
  or if `get_test_params` has a parameter whose value is an estimator.
* **AST scan.**
  * Methods scanned: `_fit`, `_predict`, `_update`, `_predict_interval`,
    `_predict_quantiles`, `_predict_var`, `_predict_proba`, `_pretrain`,
    `_pretrain_update`.
  * Each is resolved through the MRO up to `BaseForecaster` (not included). The scan then
    follows `self.<m>` calls, `super().<m>` calls, and bare-name calls into sktime helper
    functions in the same module or in other modules, to depth 2. It does not follow into
    datatypes, base, `_fh`, utils.validation or libs.
  * Pandas API usage is counted in 5 categories:
    * label_select: `loc`, `xs`, `droplevel`, …
    * time_ops: `shift`, `rolling`, `resample`, `asfreq`, `to_period`, …
    * frame_ops: `groupby`, `reindex`, `merge`, `combine_first`, `set_index`,
      `reset_index`, …
    * soft_ops: `iloc`, `dropna`, `diff`, …
    * meta: `.index`, `.columns`, `.freq`, …
  * Also counted: `pd.*` calls (wrap / time / ops), numpy conversion (`.values`,
    `.to_numpy()`, `np.asarray`), `fh.to_absolute*`, `get_expected_pred_idx`,
    `self.cutoff` / `self._cutoff`, and stored history (`self._y`, `_cur_y`, …).
  * A 2-pass taint analysis tags the receiver of each hard op as *input-derived* (from
    `y` / `X` / `self._y`), *output* (built by a `pd.` constructor or returned by a
    predict call), or *other*. Ops whose receiver looks like an index (`.index`,
    `cutoff`, `fh`, `idx`, …) are counted separately as index-level ops.
* **Bucket precedence.**
  1. B3: third-party library dependency, or code vendored under `sktime.libs`.
  2. B4: composite.
  3. B2: at least one hard pandas op in core code.
  4. B1: numpy conversion present.
  5. B5: everything else.
* **core_style.**
  * pandas-semantic-input: at least one hard op on input-derived data.
  * output-assembly-only: hard ops only on outputs or untraced objects.
  * numpy-or-passthrough: no hard ops.
* **Static verdict rules.**
  * B1 → V1.
  * B2 → V3 if an algorithmic op (groupby, rolling, shift, reindex, combine_first,
    merge, …) is applied to input data, or there are 3 or more hard input ops; otherwise V2.
  * B3 → V4 if any wrapped library family is pandas-bound. The `LIB_INPUT` table in the
    script records evidence for each library as adapter file:line. Otherwise V2 if the
    class is pandas-semantic-input, else V1.
  * B4 → V1, V2 or V3 by core_style; V3 only if an algorithmic op is applied to input.

### Dynamic trace (`scripts/a2_trace.py`)
* Runs in an own venv (`scratchpad/venvs/a2`: pandas 2.3.3, numpy 2.2, statsmodels,
  pmdarima, statsforecast, arch, tbats, sklearn 1.9.1) so that adapters can run. 85 of the
  160 forecasters are runnable there; torch / prophet / darts / gluonts are not installed.
* Inputs to the inner methods are replaced by instrumented pandas subclasses. Every attribute
  access is recorded and attributed to the caller: own class chain, framework, other sktime
  code, or third party. Line coverage is recorded too.
* Steps run: `fit(fh=[1,2,3])`, `predict`, `predict_interval` (if supported),
  `update` + `predict`, on a 48-point monthly PeriodIndex series. If a class needs
  multivariate data, the run uses 2 columns.
* **Enforce modes** raise on the first use of an attribute outside an allowed set, by code
  in the forecaster's own class-chain files:
  * `strict` = `to_numpy`, `values`, `__array__`, `index`, `columns`, `name`, `shape`,
    `ndim`, `dtype(s)`, `size`, `__len__`, `copy`, `empty`. This is the surface a
    "values + time-index metadata" container offers.
  * `numpylike` = strict plus positional / elementwise numpy-equivalents: `iloc`, `iat`,
    `[]`, iteration, arithmetic, comparisons, ufuncs, reductions, `isna`, `astype`,
    `squeeze`, …
  * `numpylike_pow` = numpylike plus `.pow`.

  Not enforced: child estimators' own files, the framework, and third-party code.

### Validation (`scripts/a2_validate.py`)
* 26 manual labels (`results/a2_manual_labels.csv`), each with file:line evidence from
  reading the code.
* The script computes confusion matrices against the static census, a calibrated population
  estimate (the static→manual conditional mix applied to the population), and static vs
  dynamic agreement.

## Commands (all run in this cloud session)

```
cd /home/user/sktime
EXP_PY=/tmp/claude-0/-home-user-sktime/c9329e73-89f5-514d-ae5a-9180bf0e4a71/scratchpad/expenv/bin/python
A2_PY=/tmp/claude-0/-home-user-sktime/c9329e73-89f5-514d-ae5a-9180bf0e4a71/scratchpad/venvs/a2/bin/python
# static census (13.3 s)
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/a2_census.py
# dynamic trace (record), two halves of the 85 runnable names
PYTHONPATH=. $A2_PY -W ignore experiments/container_independence/scripts/a2_trace.py --names <half1> --out experiments/container_independence/results/a2_trace_part1.json
PYTHONPATH=. $A2_PY -W ignore experiments/container_independence/scripts/a2_trace.py --names <half2> --out experiments/container_independence/results/a2_trace_part2.json
# enforcement
PYTHONPATH=. $A2_PY -W ignore experiments/container_independence/scripts/a2_trace.py --names <all 85> --mode strict    --out .../results/a2_enforce_strict.json
PYTHONPATH=. $A2_PY -W ignore experiments/container_independence/scripts/a2_trace.py --names <all 85> --mode numpylike --out .../results/a2_enforce_numpylike.json
PYTHONPATH=. $A2_PY -W ignore experiments/container_independence/scripts/a2_trace.py --names VAR,VARMAX,VECM --mode numpylike_pow --out .../results/a2_enforce_numpylike_pow_var.json
# validation
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/a2_validate.py
```

Every script accepts `--grid {dev,large}`. The flag has no effect because this is a static
census, not a benchmark.

## Raw results

### Census (stdout of a2_census.py)
```
n_forecasters=160  runtime=13.3s
bucket counts: B1 7 (4.4%)  B2 3 (1.9%)  B3 99 (61.9%)  B4 51 (31.9%)   [B5 0]
composite flag: 61   third-party flag: 99   composite & third-party: 10
y_inner_mtype: pd.Series 70, pd.DataFrame 50, pd.DataFrame|pd-multiindex|pd_multiindex_hier 17,
  <all/most mtypes> 8, pd.Series|pd-multiindex|pd_multiindex_hier 5, pd-multiindex|pd_multiindex_hier|pd.Series 4,
  pd.Series|pd.DataFrame 2, pd.Series|pd.DataFrame|pd-multiindex|pd_multiindex_hier 2, pd.DataFrame|pd.Series 1,
  pd.DataFrame|pd.Series|pd-multiindex|pd_multiindex_hier 1
non-pandas in y_inner: 8
B3 families: torch/foundation 47, statsmodels 15, statsforecast 9, neuralforecast 5, pmdarima 4, pytorch-forecasting 4,
  darts 3, gluonts 3, prophetverse 2, hypertrees 2, prophet 2, skforecast 2, arch 1, LLM (litellm) 1, autots 1, xgboost 1,
  tsbootstrap 1, remote HTTP API 1, greykite 1, hcrystalball 1, neuralprophet 1, R (rpy2/fable) 1
uses fh.to_absolute*: 122   uses get_expected_pred_idx: 25   uses cutoff: 127   any fh op or cutoff: 128
uses stored history (_y/_X/_cur_y/_cur_X): 114
distinct (_fit owner, _predict owner) implementations: 120
B1: n=7 distinct_impls=5  core_style={numpy-or-passthrough: 7}
B2: n=3 distinct_impls=3  core_style={pandas-semantic-input: 3}
B3: n=99 distinct_impls=73 core_style={pandas-semantic-input: 46, output-assembly-only: 35, numpy-or-passthrough: 18}
B4: n=51 distinct_impls=40 core_style={output-assembly-only: 9, pandas-semantic-input: 24, numpy-or-passthrough: 18}
composite param kinds: sktime 38, sklearn 14, sklearn|sktime 5, <meta-base only> 4
native panel/hier y_inner (pd-multiindex listed): 29  (B4 16, B3 13)
static verdict: V1 63 (39.4%), V2 56 (35.0%), V3 13 (8.1%), V4 28 (17.5%)
static verdict x bucket: B1/V1 7, B2/V3 3, B3/V1 38, B3/V2 33, B3/V4 28, B4/V1 18, B4/V2 23, B4/V3 10
B3 lib_input: numpy-capable 71, pandas-bound 19, numpy-capable|pandas-bound 9
```

### Inner mtypes
* `y_inner_mtype`: every one of the 160 lists a pandas mtype. 152 list **only** pandas
  mtypes.
* The 8 that list non-pandas mtypes are all passthrough or delegating composites that
  declare "all mtypes":
  * `ConformalIntervals`, `DontUpdate`, `FallbackForecaster`, `ForecastByLevel`,
    `MultiplexForecaster`, `Permute`, `UpdateEvery`, `UpdateRefitsEvery`.
  * Non-pandas mtypes they list: `pl.DataFrame` ×8, `polars_panel` / `polars_hierarchical`
    ×7, `dask_*`, `np.ndarray`, `numpy3D`, `xr.DataArray`, `gluonts_*`, `df-list`,
    `nested_univ`.
* **No leaf forecaster declares a non-pandas inner mtype.**
* `X_inner_mtype`: `pd.DataFrame` 121; `pd.DataFrame|pd-multiindex|pd_multiindex_hier` 24
  (plus 4 with the same set in a different order); <many> 8; None 2.
* Only 29 classes (18%) declare a panel or hierarchical inner mtype, i.e. are natively
  global or vectorize-free: 16 in B4, 13 in B3. All the others go through `_vectorize`.

### Composites (B4 plus the 10 composites in B3): 61
* Estimator-valued parameters only: 22.
* Reducer: 12.
* Delegate: 8.
* ensemble + hetero_meta: 6.
* delegate + tuning: 4.
* delegate + hetero_meta: 3.
* hetero_meta + pipeline: 2.
* hetero_meta: 2.
* delegate + stream: 2.

Kinds of child estimator: sktime 38, sklearn 14, both 5, meta-base only 4.

A composite's real verdict is **transitive**. Example: `STLForecaster` is numpy at its own
level (V1), but by default it delegates components to `NaiveForecaster` (V3).

### Concentration of input-side hard pandas sites
* 180 distinct file:line sites, and 275 (class, site) pairs.
* Files with the most sites:
  * `lagllama.py` 28, `compose/_reduce.py` 19, `moirai.py` 15,
    `compose/_reduce_proba.py` 14, `moirai2.py` 13, `naive/_naive.py` 9,
    `adapters/_fbprophet.py` 7.
  * 4 each: `adapters/_pmdarima.py`, `adapters/_statsmodels.py`, `compose/_pipeline.py`,
    `mapa.py`, `adapters/_pytorchforecasting.py`.
* Sites shared by the most classes:
  * `adapters/_statsmodels.py:129-131` (×11).
  * `adapters/_pytorch.py:33` (×9). This is the `y.loc[instance_id]` per-instance split,
    i.e. a re-implementation of what VectorizedDF iteration already does.
  * `base/_sktime.py:151,155` (×9). This is the last-window `.loc[start:cutoff]`.
  * `adapters/_statsmodels.py:164` (×5).
  * `compose/_reduce.py:1833,1835,493` (×4 each).

### Manual spot check (a2_validate.py; 26 classes, labels with evidence in results/a2_manual_labels.csv)
```
spot-checked n=26
  bucket accuracy      1.000
  core_style accuracy  0.962
  verdict accuracy     0.846   (coarse V1|V2 / V3 / V4: 0.846)
  verdict confusion {'V1->V1': 10, 'V2->V2': 7, 'V2->V3': 1, 'V3->V2': 3, 'V3->V3': 3, 'V4->V4': 2}
  mismatches:
   RecursiveTabularRegressionForecaster static V3 -> manual V2 (default local path is numpy sliding window; groupby only on global/transformer path)
   TransformedTargetForecaster  core_style static pandas-semantic-input -> manual output-assembly-only (xs/loc act on interval OUTPUT passed as param named y)
   ReconcilerForecaster         static V2 -> manual V3 (Aggregator groupby, unstack, MultiIndex level semantics)
   ForecastKnownValues          static V3 -> manual V2 (reindex = label lookup; get_indexer + take)
   YfromX                       static V3 -> manual V2 (combine_first + .loc[fh_idx] row lookup, sklearn on rows)
population calibrated: V1 63.0 (39.4%), V2 55.5 (34.7%), V3 13.5 (8.4%), V4 28.0 (17.5%)
distinct (_fit,_predict) impls per static verdict: V1 47, V2 45, V4 18, V3 11
fh/cutoff reliance by verdict: V1 46/63 to_absolute, 49 cutoff; V2 41/56, 41; V4 23/28, 25; V3 12/13, 12
```

**Spot-checked classes and labels (manual bucket / verdict):**
* NaiveForecaster B2/V3
* MIRAForecaster (`sktime/forecasting/mira.py`) B3/V1
* Croston B1/V1
* TSB B1/V1
* ARARForecaster B1/V1
* CurveFitForecaster B1/V1
* PolynomialTrendForecaster B4/V1
* ThetaForecaster B3/V2
* ExponentialSmoothing B3/V2
* VAR B3/V2
* AutoARIMA B3/V2
* StatsForecastAutoARIMA B3/V1
* Prophet B3/V4
* ChronosForecaster B3/V1
* TBATS B3/V1
* ARCH B3/V2
* STLForecaster B3/V1
* SkforecastAutoreg B3/V4
* DirectReductionForecaster B4/V3
* RecursiveTabularRegressionForecaster B4/V2
* TransformedTargetForecaster B4/V2
* EnsembleForecaster B4/V2
* ReconcilerForecaster B4/V3
* ForecastKnownValues B2/V2
* DummyGlobalForecaster B2/V3
* YfromX B4/V2

**Heuristic accuracy:**
* Bucket: 100%.
* core_style: 96%.
* Verdict: 85%.
* Static V1 and V4 were always right (12/12).
* Static V3 overcounts: only 3 of 6 sampled V3s were real core rewrites. Its rule
  (combine_first / groupby / reindex anywhere on an input path) also catches optional
  feature paths and label lookups.
* Static V2 missed 1 real V3 (ReconcilerForecaster).

NaiveForecaster and MIRAForecaster, in detail:
* **NaiveForecaster (V3).**
  * `strategy="last"` is implemented as `pd.concat(keys=)` + `sort_index` + `ffill` /
    `bfill` + `.loc["b"]` (`_naive.py:373-395`).
  * `_predict_var` uses `y.shift` / `rolling` / `groupby("__sp__").rolling`
    (`_naive.py:522-571`).
  * The mean and drift strategies already use a numpy last window.
  * It is a small rewrite, but it is in the core.
* **MIRAForecaster (V1).**
  * Input: `y.iloc[:,0].to_numpy(float32)` (`mira.py:316`), passed to torch.
  * Output: `pd.DataFrame(values, index=fh.to_absolute(cutoff))` + `.loc[isin(...)]`
    (`mira.py:221-227`).

### Dynamic trace, record mode (84 ok, 1 error)
* The error is `RecursiveReductionForecaster`:
  `ValueError: Invalid frequency: ME`. This is a pandas / PeriodIndex freq incompatibility,
  not instrumentation.
* Own-code style by bucket:
  * B1: numpy 7.
  * B2: numpy 1, no-hard-ops 1, pandas-semantic 1.
  * B3: numpy 12, no-hard-ops 15, pandas-semantic 1.
  * B4: pandas-semantic 12, no-hard-ops 15, untouched 18, numpy 2.
* Static vs dynamic, on "does own code apply hard pandas ops to input":
  * Both say yes: 14. Both say no: 45. Dynamic-only: **0** (no false negatives in the
    static scan).
  * Static-only: 26. For 21 of these, the flagged sites were not executed in this run
    (update / exog / in-sample paths). For 5, the sites were executed but the receiver was
    not the traced input; these are output-receiver false positives (ARCH,
    ForecastKnownValues, SquaringResiduals, TransformedTargetForecaster, VARMAX).
  * Of 154 static input sites, 26 were executed in the run.
* Hard pandas attributes called by own code on traced inputs, as number of classes:
  `loc` 11, `shift` 2, `groupby` 1.
* **Third-party code accessing the pandas API of inputs.**
  * statsmodels, pmdarima, arch and tbats only accessed `__array__`, `index`, `name`,
    `__len__`, `dtype`, `iloc`, `columns`, `shape`, `ndim`.
  * statsmodels reads `.index` for date handling. pmdarima / sklearn `check_array` uses
    `iloc` / `dtype`.
  * The sklearn regressors in reducers additionally use `apply`, `dtypes` and `sparse`.
  * None of them use label or time operations. **In the traced set, the numpy-capable
    libraries need only an array plus, optionally, an index.**

### Enforcement (own code only on the allowed surface)
| mode | ok / 85 | % | first-violation attributes |
|---|---|---|---|
| strict (values + index metadata) | 57 | 67.1% | iloc 12, loc 9, notnull 2, shift 2, `[]` 1, ffill 1, arithmetic 1 |
| numpylike (+ positional / elementwise) | 68 | 80.0% | loc 11, pow 3, shift 2, ffill 1 |
| numpylike_pow (+ `.pow`; VAR, VARMAX and VECM then pass) | 71 | 83.5% | loc 11, shift 2, ffill 1 |

The 14 remaining numpy-like failures:
* 7 reducers fail at `base/_sktime.py:151`. This is one line, the last-window label slice;
  the positional equivalent is `values[-window_length:]`.
* `DirectReductionForecaster` (`_reduce.py:2218`) and `RecursiveReductionForecaster`
  (`_reduce.py:2652`): `.loc` on lagged frames.
* `ColumnEnsembleForecaster` (`_column_ensemble.py:210`).
* `MAPAForecaster` (`ffill`).
* `NaiveForecaster` (`shift`).
* `VARReduce` (`shift`).

Passing in strict mode, by static verdict: V1 30, V2 24, V3 3. In numpylike mode: V1 37,
V2 26, V3 5.

## Conclusions

1. **Boundary-only versus rewrite.** Under a "numpy values + time-index metadata" inner
   representation:
   * About 39% need boundary changes only (V1).
   * About 35% more need only mechanical glue edits (V2), so **about 74% need zero
     changes to core math**.
   * About 8% (13 classes, 11 implementations; probably fewer per the calibration note)
     have pandas in the algorithm itself, mainly the reducers and NaiveForecaster.
   * About 17.5% are bound to a third-party pandas or own-type input format. The 85-class
     dynamic run agrees: 83.5% run with their own code restricted to a numpy-like surface.
2. **The work is concentrated.**
   * 120 implementations, not 160.
   * Three shared sites (statsmodels adapter `_update` / `_predict`, `_sktime.py:151`,
     `_pytorch.py:33`) cover 29 (class, site) pairs.
   * The torch foundation-model wrappers (47 classes) feed numpy or tensors. Their pandas
     use is per-instance splitting and output assembly, both of which the framework could
     provide.
3. **Output index construction already goes through one API.** 128/160 build output
   indexes via `fh.to_absolute*` / `cutoff`, so time-index metadata must implement
   `ForecastingHorizon` semantics (to_absolute, to_relative, to_indexer,
   get_expected_pred_idx). That is a single object to port. `.index` is also read by
   statsmodels for dates; that read is metadata, not data.
4. **Third-party lock-in (V4, 28) cannot be removed in sktime**, and those classes must
   keep converting to pandas at their boundary. 7 of the 28 (gluonts or
   pytorch-forecasting loaders around torch models) could be moved to numpy by bypassing
   the loader.
5. **Implication for options (1) and (2).**
   * Neither a per-backend adapter nor a narwhals rewrite has to touch about 74% of
     forecasters' core math, as long as the inner representation they receive offers
     `values` + `index` / `columns` metadata and a ForecastingHorizon that works on it.
   * The rewrite cost of (2) applies only to the about 13 V3 classes plus the about 56 V2
     glue sites.
   * The unavoidable pandas conversions are the V4 classes, about 17%.
   * Today no leaf forecaster declares a non-pandas inner mtype, so getting any real value
     from polars or dask requires a boundary change in all 152 pandas-only classes, or in
     the framework conversion that feeds them.

## Caveats
* The static scan is depth-limited: helpers to depth 2, and it does not follow into
  sktime transformers used internally (e.g. Theta's Deseasonalizer, the reducers' Lag,
  Aggregator). Composite verdicts are not transitive.
* The taint analysis is name-based. It over-flags ops on outputs passed under input-like
  names (TransformedTargetForecaster) and on optional feature paths (reducers with
  transformers).
* The verdict rules are judgement calls. V2 versus V3 is the soft boundary; V1 and V4 were
  100% right in the sample. 26 spot checks give a coarse calibration, about ±15pp on the
  V2/V3 split.
* The dynamic trace covers only the 85 forecasters runnable without torch, prophet, darts
  or gluonts. That means 0 of the 47 torch / foundation models. It uses one 48-point
  monthly series with no exog, so exog, panel and in-sample paths are under-exercised: 21
  of 26 static-only flags were not executed. Enforcement checks only the top class's own
  files, not children or the framework.
* "Zero changes to core math" assumes the inner container provides `values`, positional
  slicing, and an index object with ForecastingHorizon support. It does not assume pandas
  label semantics.
* `capability:global_forecasting` returned None for all classes (the tag name probably
  differs), so it was not used.
