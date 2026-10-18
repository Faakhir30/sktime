# a1_core_coupling — pandas touchpoints at the VectorizedDF + BaseForecaster boundary, and the minimal container protocol

Workstream key: `a1_core_coupling` (file prefix `a1_`). Worktree `/home/user/sktime` at the
stateless-`VectorizedDF` state (schema-only `VectorizedDF`, data passed as `(frame, schema)`).
Env: pandas 2.3.3, numpy 2.2.6, polars 2.0.0, narwhals 2.27.1, pyarrow 26.0.0
(`EXP_PY`). All numbers below were regenerated in this (cloud) run; commands and outputs are
quoted. Machine: 4 cores, ~2 concurrent agents → absolute timings are noisy; ratios are measured
within one process.

## TL;DR

1. **Census.** 229 classified pandas touchpoints on 170 distinct source lines in the boundary layer
   (6 false positives kept for transparency, 0 unclassified), using **46 distinct pandas APIs**.
   Biggest category is *time index / freq / cutoff* (78 touchpoints, 16 APIs) — not grouping. The
   pure container ops (group/iterate + slice/select + concat/reconstruct + pivot) are only
   42 touchpoints (18%). 138/229 touchpoints are executed ("hot") in 6 typical public-API scenarios.
2. **Minimal protocol.** All non-contract touchpoints collapse into **14 abstract operations**
   (P1–P14, plus the existing `convert` table P15), of which **only 4 are true container ops**
   (`iter_groups`, `select_columns`, `concat_rows`, `concat_cols`); the rest are
   metadata (2), time semantics (5: `time_freq`, `cutoff`, `time_contains`, `take/sort_rows`,
   `upsert_rows`), dispatch (1) and an aligned binary op (1). See §3.
3. **API contracts (pandas as a value type in the public API): 44 touchpoints / 4 contracts** —
   `forecasters_` is a `pd.DataFrame` (C1, 21 touchpoints), `cutoff`/fh/cv are pandas `Index`
   with `freq` (C2, 11), `predict_interval/quantiles` column MultiIndex (C3, 2),
   `get_fitted_params` key strings `"forecasters.loc[...]"` (C4, 1). Probe: with a **polars Panel
   input, `predict()` returns polars but `predict_interval`, `predict_quantiles`, `predict_var`,
   `cutoff`, and `forecasters_` are pandas** — these need a decision independent of the
   implementation.
4. **Where the time goes** (cheap forecasters, 1000 series × 100): the container layer
   (VectorizedDF iterate/reconstruct/pivot/prepare) is **13.5–15.8%** of fit wall time and **1.8%** of
   NaiveForecaster predict; **per-slice re-entry** into the public API (mtype check, convert,
   fh check, cutoff, glue, clone) is **64.7–83.7%** of fit. A backend protocol that only swaps the
   container ops therefore removes at most ~16% of the boundary cost; the dominant cost is the
   per-instance re-validation, which is container-independent design, not pandas API.
5. **forecasters_ container-free.** A pandas-free `EstimatorGrid` (protos/a1_estimator_grid.py)
   replaces the pivot (4–82× faster build, 26–314× faster consume, but only ≲1% of fit cost).
   With it monkeypatched in, forecasting-base / transformer-base / metric tests (204) show **0 new
   failures**; the 2112 new failures are all one type assertion (`isinstance(result, pd.DataFrame)`)
   in `test_vectorize_est`, and a direct probe shows **4/4 hierarchical metric configurations
   break**. Breakage is concentrated in callers that treat `vectorize_est(return_type="pd.DataFrame")`
   as a *data* table (performance metrics: `.iloc`, `.index`, `.mean`) and in user-facing
   attribute access (`forecasters_.loc/.iloc/.values`, e.g. `cinn.py:707`).
6. **Behavioural bugs found by probes** (pandas assumptions that silently change behaviour for
   non-pandas input): X/y index containment check silently **passes** for polars inputs with
   disjoint dates (`hasattr(obj,"index")` duck typing → RangeIndex); `update_predict` and
   `predict_residuals` **raise** for polars Series; `update_data`/`get_cutoff` still duck-type the
   removed `VectorizedDF.X` attribute (AttributeError if a VectorizedDF is passed).

## 1. Method

| step | script | output |
|---|---|---|
| static AST census + hand classification of every hit | `scripts/a1_static_touchpoints.py` | `results/a1_touchpoints.json`, `results/a1_touchpoints_table.md` |
| dynamic trace (`sys.setprofile`) of pandas entry calls per module group, hot/cold join with census | `scripts/a1_dynamic_trace.py --grid dev` | `results/a1_dynamic_trace.json` |
| exclusive wall-time breakdown by boundary step | `scripts/a1_overhead_breakdown.py --grid dev` | `results/a1_overhead_breakdown_dev.json` |
| behavioural probes pandas vs polars inputs | `scripts/a1_semantic_probes.py` | `results/a1_semantic_probes.json` |
| cutoff op cost: sktime vs pandas-vectorized vs polars | `scripts/a1_cutoff_bench.py --grid dev` | `results/a1_cutoff_bench_dev.json` |
| estimator grid build/consume cost: pivot vs grid | `scripts/a1_grid_bench.py --grid dev` | `results/a1_grid_bench_dev.json` |
| vectorized metrics with grid installed | `scripts/a1_grid_metrics_probe.py` | `results/a1_grid_metrics_probe.json` |
| container-free estimator grid, sktime tests off/strict/compat | `protos/a1_estimator_grid.py`, `protos/a1_grid_plugin.py`, `scripts/a1_grid_pytest.sh` | `results/a1_grid_pytest_{off,strict,compat}.txt` |

All commands are run from the WT root with
`PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos $EXP_PY -W ignore <script>`.

**Static census scope** (functions, not whole files, except `_vectorize.py` and `parallel.py`):
`_vectorize.py` (all); `_base.py`: `fit, predict, fit_predict, predict_quantiles,
predict_interval, predict_var, predict_proba, pretrain, update, update_predict,
update_predict_single, predict_residuals, get_fitted_params, _check_X_y, _check_X, _update_X,
_update_y_X, cutoff, _set_cutoff, _set_cutoff_from_y, _check_fh, _vectorize, _update,
_predict_moving_cutoff, _get_varnames, _get_columns, _format_moving_cutoff_predictions`, class
tags; `_convert.py: convert, convert_to, ...`; `_check.py: check_is_scitype, mtype, ...`;
`utils/multiindex.py: flatten_multiindex, underscore_join`; `_utilities.py: _get_index,
get_time_index, get_index_for_series, _get_cutoff_from_index, get_cutoff, update_data, module
consts`; `validation/series.py: check_equal_time_index`; `utils/parallel.py` (all).
Detection: (1) `pd.<...>` references, (2) pandas-specific attribute names on non-self objects
(`groupby, droplevel, xs, loc, iloc, index, columns, names, nlevels, levels, pivot, reindex,
unique, iterrows, set_names, freq, freqstr, combine_first, ...`), (3) `hasattr/getattr` duck
typing on pandas attribute names, (4) pandas mtype strings in code (not docstrings),
(5) 11 manual entries for implicit semantics (index-aligned `y - y_pred`, `len(frame)`,
`cv.split(y)`, ...). Every hit must be in a hand-written `CLASSIFY` table keyed by
(file, line, token) — the script exits non-zero on any unclassified hit, and reports stale keys;
this run: **0 unclassified, 0 stale keys** (so the classification matches the current source).

Caveat: `_convert.py`, `_check.py`, `parallel.py` contain **no** in-scope pandas touchpoints —
they are pure dispatchers; pandas lives in the registered checkers/converters (out of scope here;
see workstream a2 for the estimator census). Static counts are of *syntactic sites*, dynamic
counts are lower bounds (Cython properties such as `DataFrame.index` emit no profile events).

## 2. Census results

Command: `PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/a1_static_touchpoints.py`

```
distinct source lines: 170
touchpoints: 229  false positives: 6  unclassified: 0
by category:
     9  container-op: concat/reconstruct
    34  type-dispatch: isinstance/duck-typing
     6  container-op: group/iterate
    25  mtype-pinning (pandas mtype as conversion target)
    23  index-semantics: MultiIndex levels/names
    13  metadata: columns/dtypes/shape/nans
    14  output-construction
    15  container-op: pivot/reshape (estimator grid)
    12  container-op: slice/select
    78  index-semantics: time index/freq/cutoff
distinct canonical pandas API: 46
by file: {'_vectorize.py': 71, '_base.py': 43, 'multiindex.py': 1, '_utilities.py': 110, 'validation/series.py': 4}
API-contract touchpoints: 44
```

### 2.1 Touchpoints and distinct pandas APIs per category

| category | touchpoints | hot (executed) | distinct pandas APIs | APIs |
|---|---:|---:|---:|---|
| container-op: group/iterate | 6 | 6 | 5 | `DataFrame.groupby`, `GroupBy.nth`, `DataFrame/Series.iloc`, `Index.to_frame`, `Series.agg` |
| container-op: slice/select | 12 | 4 | 6 | `DataFrame.__getitem__(list)`, `DataFrame.loc`, `DataFrame.xs`, `DataFrame/Series.iloc`, `DataFrame.sort_index`, `pd.DataFrame` |
| container-op: concat/reconstruct | 9 | 5 | 4 | `pd.concat`, `pd.DataFrame`, `DataFrame.combine_first`, `DataFrame.__sub__` (index-aligned) |
| container-op: pivot/reshape (estimator grid) | 15 | 15 | 10 | `pd.DataFrame`, `DataFrame.pivot`, `DataFrame.reindex`, `DataFrame.loc`, `DataFrame.iterrows`, `DataFrame.shape`, `DataFrame.columns`, `.index`, `Index/Series.unique`, `Series.values` |
| index-semantics: MultiIndex levels/names | 23 | 19 | 6 | `.index`, `Index.droplevel`, `Index.names`, `Index.nlevels`, `Index.set_names`, `Index/Series.unique` |
| index-semantics: time index/freq/cutoff | 78 | 40 | 16 | `.index`, `Index.__getitem__(int)`, `Index.__getitem__(list)`, `Index.difference`, `Index.equals`, `Index.freq`, `Index.freqstr`, `Index.is_monotonic_increasing`, `Index.isin`, `Index.sort_values`, `Index.symmetric_difference`, `MultiIndex.get_level_values`, `MultiIndex.levels`, `pd.Index`, `pd.RangeIndex`, `pd.infer_freq` |
| metadata: columns/dtypes/shape/nans | 13 | 11 | 3 | `DataFrame.columns`, `DataFrame.dtypes`, `Index.equals` |
| output-construction | 14 | 11 | 9 | `pd.DataFrame`, `pd.Index`, `pd.MultiIndex.from_product`, `pd.MultiIndex.from_tuples`, `.columns`, `.index`, `Index.droplevel`, `Index.name`, `Index.names` |
| type-dispatch: isinstance/duck-typing | 34 | 22 | 6 | `pd.DataFrame`, `pd.Series`, `pd.Index`, `pd.MultiIndex`, `.columns`, `.index` (via hasattr) |
| mtype-pinning (pandas mtype string as target) | 25 | 5 | 0 | (mtype strings `"pd.DataFrame"`, `"pd-multiindex"`, `"pd_multiindex_hier"`, `"pd.Series"`, ...; `convert_to` calls) |
| **total** | **229** | **138** | **46** (distinct, across categories) | |

"hot" = line executed in at least one dynamic-trace scenario (§4). Per-category API sets overlap
(e.g. `.index` appears in 6 categories), hence 46 distinct overall.

### 2.2 Touchpoints by file × category

| file | G | S | C | P | L | T | M | O | D | K | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `datatypes/_vectorize.py` | 1 | 2 | 5 | 11 | 16 | 11 | 6 | 7 | 4 | 8 | 71 |
| `forecasting/base/_base.py` | 0 | 3 | 3 | 4 | 0 | 8 | 5 | 6 | 9 | 5 | 43 |
| `datatypes/_utilities.py` | 5 | 7 | 1 | 0 | 7 | 55 | 2 | 0 | 21 | 12 | 110 |
| `utils/validation/series.py` | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 4 |
| `utils/multiindex.py` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |

G group/iterate, S slice/select, C concat, P pivot (estimator grid), L MultiIndex levels/names,
T time/freq/cutoff, M metadata, O output construction, D type dispatch, K mtype pinning.

Observation: **`_utilities.py` (get_cutoff / get_time_index / update_data) holds 48% of all
touchpoints** and 55 of the 78 time-semantics touchpoints. VectorizedDF itself is mostly
MultiIndex level bookkeeping (16) + estimator grid (11 P + 7 O).

The full per-touchpoint table (229 rows: file:line, function, token, canonical pandas API,
category, abstract op, protocol op, API-contract flag) is in Appendix A.

## 3. (a) The minimal container-operation protocol

Grouping the abstract operation of every touchpoint (122 distinct free-text ops) yields the
following protocol. Counts: touchpoints mapped to the op / of which hot in the trace.

| op | signature (sketch) | replaces (examples, file:line) | n | hot |
|---|---|---|---:|---:|
| **P1** `backend_of` | `backend_of(obj) -> Backend` (single dispatch point) | `isinstance(x, pd.DataFrame/Series/Index/MultiIndex)`, `hasattr(obj,"index"/"columns"/"freq"/"X")` — `_base.py:1715, 2214, 2837..2866`, `_utilities.py:130..298`, `_vectorize.py:94, 516, 564` | 29 | 21 |
| **P2** `index_spec` | `index_spec(frame) -> (key_names: tuple[str], time_name: str, n_levels: int)` | `X.index.names`, `.nlevels`, `pd.RangeIndex` synthesis for numpy — `_vectorize.py:242-244`, `_utilities.py:13-17, 40, 55-58, 311-316` | 22 | 15 |
| **P3** `column_names` | `column_names(frame) -> tuple; feature_kinds(frame)` | `X.columns`, `dtypes` — `_vectorize.py:90, 241, 368, 527, 529, 564`; `_base.py:2842, 2853, 2866` | 12 | 11 |
| **P4** `instance_keys` | `instance_keys(frame, n_key_levels) -> list[tuple]` (first-appearance order, unique) | `X_ix.droplevel(-1).unique()` — `_vectorize.py:292, 294, 95`; `_utilities.py:143` | 6 | 3 |
| **P5** `time_freq` | `time_freq(frame_or_slice) -> freq | None` (infer, attach) | `idx.freq/freqstr`, `pd.infer_freq`, `_enforce_index_freq` — `_vectorize.py:99-102, 866-868`, `_utilities.py:148-168, 280-291` | 36 | 24 |
| **P6** `iter_groups` | `iter_groups(frame, n_key_levels) -> Iterator[(key_tuple, slice_without_key_levels)]` | `X.groupby(level=..., sort=False)` + `group.droplevel(levels)` — `_vectorize.py:396-397`; `.xs` `_vectorize.py:95` | 5 | 3 |
| **P7** `select_columns` | `select_columns(slice, cols) -> slice` | `inst[[col]]` — `_vectorize.py:369`; `y_pred.loc[:, cols]` `_base.py:2868` | 2 | 2 |
| **P8** `concat_rows` | `concat_rows(slices, keys: list[tuple], key_names) -> frame` | `pd.concat(df_list, keys=row_ix)` — `_vectorize.py:536, 538, 559`; `_base.py:2836, 2858` | 5 | 3 |
| **P9** `concat_cols` | `concat_cols(slices, col_keys | None, flatten: bool) -> frame` | `pd.concat(axis=1, keys=col_keys)`, `flatten_multiindex` — `_vectorize.py:545, 557, 565`; `_base.py:2466, 2861`; `multiindex.py:39` | 7 | 4 |
| **P10** `cutoff` | `time_extreme(frame, n_key_levels, "max"|"min") -> scalar` | `get_cutoff` groupby-nth/agg path, `_get_cutoff_from_index` — `_utilities.py:142-168, 278-324` | 15 | 8 |
| **P11** `upsert_rows` | `upsert_rows(frame, new) -> frame` (new wins on equal keys) | `X_new.combine_first(X)` + freq repair — `_utilities.py:405-409` | 5 | 4 |
| **P12** `take_rows / sort_rows` | `take_rows(frame, positions)`, `sort_by_time(frame)` | `y.iloc[new_window]` `_base.py:2726`, `sort_index` `_base.py:2863-2864` | 4 | 3 |
| **P13** `time_contains` | `time_contains(X, y) -> bool` (+ diff for the error message) | `check_equal_time_index` `series.py:268-277`, `_base.py:2073`, `get_index_for_series` `_utilities.py:88` | 6 | 4 |
| **P14** `aligned_sub` | `aligned_binary(a, b, op)` (join on keys+time, then op) | `y - y_pred` in `predict_residuals` `_base.py:1735` | 1 | 0 |
| P15 `convert` (exists) | `convert_to(obj, to_type)` — already a dispatch table; *not* a new op | `_PANDAS_MTYPE_BY_SCITYPE` targets, tag defaults `y_inner_mtype="pd.Series"` | 25 | 5 |

So the backend-specific protocol is **14 ops**, and only **P6, P7, P8, P9 (19 touchpoints)** are
the "group / slice / concat" operations that the adapter idea (1) usually has in mind. **P5 + P10 +
P11 + P12 + P13 (66 touchpoints) are time-index semantics**: freq inference/attachment and cutoff
are the hard part, because pandas carries `freq` *on the index object* and the boundary layer
reads/writes it in 36 places (P5). Polars/arrow/duckdb have no per-index freq; a protocol must
therefore either carry freq in the schema (VectorizedDF already stores `freq`) or return it from
`time_freq`. This dovetails with the a3 time-index workstream.

Proposed protocol (Python sketch, all ops are stateless functions over a native frame plus the
existing VectorizedDF schema):

```python
class FrameBackend(Protocol):
    def matches(self, obj) -> bool: ...                                  # P1
    def index_spec(self, f) -> tuple[tuple[str, ...], str, int]: ...    # P2 key names, time name, nlevels
    def column_names(self, f) -> tuple: ...                              # P3
    def instance_keys(self, f, n_key: int) -> list[tuple]: ...           # P4
    def time_freq(self, f) -> object | None: ...                         # P5
    def iter_groups(self, f, n_key: int) -> Iterator[tuple[tuple, Any]]: ...  # P6
    def select_columns(self, f, cols) -> Any: ...                        # P7
    def concat_rows(self, parts, keys, key_names) -> Any: ...            # P8
    def concat_cols(self, parts, col_keys, flatten: bool) -> Any: ...    # P9
    def time_extreme(self, f, n_key: int, how: str) -> object: ...       # P10
    def upsert_rows(self, f, new) -> Any: ...                            # P11
    def take_rows(self, f, positions) -> Any: ...                        # P12
    def time_contains(self, a, b) -> bool: ...                           # P13
    def aligned_binary(self, a, b, op) -> Any: ...                       # P14
```

What this protocol does **not** remove:

* the per-slice conversion to the inner mtype (`y_inner_mtype="pd.Series"` for most
  forecasters, P15): as long as the inner estimator is pandas, every slice is converted to
  pandas — the protocol only moves *where* (per slice instead of once up-front);
* per-slice re-entry into the public `fit/predict` of the cloned forecaster, which re-runs
  `check_is_scitype`, `convert_to`, `_check_fh`, `get_cutoff` per instance (§5) — the dominant
  boundary cost;
* the four API contracts (§4).

## 4. (b) pandas as a value type in the PUBLIC API contract

44 of 229 touchpoints are flagged "API contract": changing them changes what users see, so they
need a separate (deprecation-level) decision from the implementation question.

| contract | touchpoints | where | what the user sees today |
|---|---:|---|---|
| **C1** estimator grid `forecasters_` is a `pd.DataFrame` | 21 | `_vectorize.py:726-819` (pivot, reindex, from_tuples, iterrows), `_base.py:1831-1832, 2459` | `f.forecasters_` is a `DataFrame` of estimators, row index = instance keys (`MultiIndex` for hierarchical), columns = variable names or `"forecasters"` |
| **C2** `cutoff`, fh, cv value types are pandas | 11 | `_base.py:2214-2215` (`pd.Index([cutoff])`), `_utilities.py:155-166, 239-279`, `_base.py:1716, 2722-2725, 2860` | `f.cutoff` is a length-1 `pd.Index` with `.freq`; `ForecastingHorizon` wraps a pandas index; `update_predict` uses splitters over pandas indices |
| **C3** probabilistic output columns | 2 | `_base.py:2797` (`MultiIndex.from_product([vars, coverage, ["lower","upper"]])`), `_base.py:2805` (`[vars, alpha]`) | `predict_interval`/`predict_quantiles` return pandas frames with 3-/2-level column MultiIndex |
| **C4** fitted-param key naming | 1 | `_base.py:1833` `f"forecasters.loc[{ix},{col}]"` | `get_fitted_params()` keys literally contain pandas `.loc` syntax, e.g. `"forecasters.loc['s0','forecasters']__sp"` and `"forecasters.loc[('h0_0', 'h1_0'),'forecasters']"` |
| defaults `y_inner_mtype="pd.Series"`, `X_inner_mtype="pd.DataFrame"` | 2 | `_base.py:115-116` | extension contract for 100s of subclasses |

Probe evidence (`scripts/a1_semantic_probes.py`, output `results/a1_semantic_probes.json`):

```
[output_types_polars_panel_input] predict: polars.dataframe.frame.DataFrame
   predict_interval: pandas.core.frame.DataFrame   predict_quantiles: pandas.core.frame.DataFrame
   predict_var: pandas.core.frame.DataFrame        cutoff: pandas.core.indexes.datetimes.DatetimeIndex
   forecasters_: pandas.core.frame.DataFrame       fh: sktime.forecasting.base._fh.ForecastingHorizon
   fitted_param_keys_sample: ["forecasters.loc['s0','forecasters']", "forecasters.loc['s0','forecasters']__sp"]
[output_types_pandas_hier_input] predict_interval_columns: ["('c0', 0.9, 'lower')", "('c0', 0.9, 'upper')"]
   pi_columns_type: pandas.core.indexes.multi.MultiIndex   forecasters_index_type: MultiIndex
```

I.e. a polars user today gets **mixed return types**: point predictions are converted back to
polars, but all probabilistic outputs, `cutoff` and `forecasters_` are pandas. C3 has no direct
polars analogue (no column MultiIndex): a container-independent contract would need a long/tidy
format (`var, coverage, bound, value`) or flattened column names (`c0__0.9__lower`), which is
a public API change.

## 5. Where the time goes (dynamic trace + exclusive timing)

### 5.1 pandas entry calls per module group (`a1_dynamic_trace.py --grid dev`)

Per-instance slope = (count at n=25 − count at n=5)/20, i.e. pandas *entry* calls (sktime → pandas,
pandas-internal calls not counted) added by each extra instance.

| scenario | total slope / instance | boundary | mtype-check | mtype-convert | fh/time | inner estimator | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| panel_univ [fit, predict, pred_int, pred_q, update, predict, get_fitted_params] | 698.6 | 117.0 (17%) | 147.6 (21%) | 136.0 (19%) | 156.0 (22%) | 85.0 (12%) | 57.0 |
| hier_multivar_2col [fit, predict, pred_int] | 671.0 | 95.0 | 140.0 | 132.0 | 158.0 | 88.0 | 58.0 |
| panel_with_X [fit, predict] | 169.8 | 56.0 | 57.8 | 25.0 | 18.0 | 13.0 | – |
| polars_panel [fit, predict] (pandas calls) | 162.2 | 35.0 | 44.2 | 34.0 | 18.0 | 17.0 | 14.0 |
| update_predict_series (slope per cv window) | 314.5 | 80.0 | 104.0 | 78.5 | 24.0 | 14.0 | 14.0 |
| series_baseline_novec (no vectorization) | 0 (315 total) | 26 total | 69 | 66 | 79 | 44 | 31 |

Key-function call slopes (panel_univ): `convert_to` 30/instance, `mtype` inference 24/instance,
`_check_X_y` 10/instance, `_check_fh` 9/instance, `VectorizedDF.items` 9, `get_cutoff` 4,
`_enforce_index_freq` 4, `update_data` 6 — every vectorized public call re-enters the public
API of the per-instance clone, which re-validates and re-converts each slice.
Polars input: `polars_panel` uses **more** pandas calls than the boundary itself would need:
only 9 polars calls/instance (mtype-convert), 162 pandas calls/instance. The polars input is
converted once to `pd-multiindex` and then everything is pandas (`conversion_count` probe:
pandas path 15 non-trivial conversions, polars path 17 = the same 15 + `polars_panel→pd-multiindex`
+ `pd-multiindex→polars_panel` for the output).

Hot/cold join with the census: 138/229 touchpoints hot; per protocol op hot/total:
`C1 21/21, C2 4/11, C3 1/2, C4 1/1, P1 21/29, P2 15/22, P3 11/12, P4 3/6, P5 24/36, P6 3/5,
P7 2/2, P8 3/5, P9 4/7, P10 8/15, P11 4/5, P12 3/4, P13 4/6, P14 0/1, P15 5/25, N 1/14`.
The 14 "N" touchpoints (legacy `nested_univ`/`df-list` cells in get_cutoff/get_time_index, the
stale `VectorizedDF.X` duck typing) are nearly all cold.

### 5.2 Exclusive wall time per boundary step (`a1_overhead_breakdown.py --grid dev`)

Exclusive timer stack around boundary functions; NaiveForecaster (y_inner_mtype pd.Series) and
a toy `LastValueX` forecaster (cheap `_fit/_predict`) on a Panel, n_time=100. Instrumentation
overhead factor (instrumented/uninstrumented) is 0.67–1.36× (noise of concurrent load).

| scenario | n_series | wall (instr.) | container layer (VectorizedDF iterate/reconstruct/pivot/prepare/freq) | per-slice re-entry (check/convert/fh/cutoff/glue/clone) | inner estimator |
|---|---:|---:|---:|---:|---:|
| naive.fit | 10 | 0.046 s | 21.7% | 77.6% | 0.2% |
| naive.fit | 100 | 0.430 s | 16.6% | 82.6% | 0.3% |
| naive.fit | 1000 | 3.684 s | 15.8% | 83.7% | 0.3% |
| naive.predict | 1000 | 9.116 s | 1.8% | 16.7% | 81.5% |
| naive.predict_interval | 100 | 1.337 s | 1.7% | 13.0% | 85.4% |
| toy.fit+predict | 10 | 0.059 s | 19.8% | 61.9% | 17.9% |
| toy.fit+predict | 100 | 0.540 s | 14.0% | 63.3% | 22.5% |
| toy.fit+predict | 1000 | 5.579 s | 13.5% | 64.7% | 21.6% |

naive.fit, 1000 series, buckets: mtype-check 39.4% (1.45 ms/instance, 4005 calls), cutoff
21.4% (0.79 ms/instance, 2002 calls), vec-iterate 11.8%, public-glue 8.9%, mtype-convert 8.3%,
vec-est-glue (pivot) 3.6%, check_X_y-glue 2.9%, clone 1.5%, fh-check 1.3%, inner 0.3%.
Uninstrumented: naive.fit 3.96 ms/instance (1000 series), naive.predict 9.5 ms/instance,
toy.fit+predict 4.8 ms/instance.

Conclusion: the **container layer is 13.5–21.7% of fit** and ~2% of predict; the pivot (C1) is
3.6–4.4%. Replacing pandas *container ops* by a polars backend can save at most that fraction
for these forecasters; the per-slice re-entry (64–84% of fit) is the larger lever and is
backend-independent (a "trusted inner call" path that skips re-validation for slices produced by
VectorizedDF, i.e. call `_fit` on the clone with pre-validated metadata, would address it).
This is *unmeasured/expected*: no prototype of a trusted path was timed in this workstream.

### 5.3 P10 `cutoff`: current vs vectorized implementations (`a1_cutoff_bench.py --grid dev`)

Command: `PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/a1_cutoff_bench.py --grid dev`

```
    10x100  sktime=0.0038s (3.8us/row)  pd_vec=0.636ms  pl=0.296ms  pl_lazy=0.973ms  ratio sk/pdvec=6x  sk/pl=13x
   100x100  sktime=0.0040s (0.4us/row)  pd_vec=0.617ms  pl=0.297ms  pl_lazy=0.863ms  ratio sk/pdvec=7x  sk/pl=14x
  1000x100  sktime=0.0207s (0.2us/row)  pd_vec=1.283ms  pl=0.536ms  pl_lazy=1.266ms  ratio sk/pdvec=16x  sk/pl=39x
 10000x50   sktime=0.0852s (0.2us/row)  pd_vec=4.983ms  pl=1.056ms  pl_lazy=1.719ms  ratio sk/pdvec=17x  sk/pl=81x
```

`get_cutoff` on a `pd-multiindex` frame (groupby/nth path) is 6–17× slower than
`idx.get_level_values(-1).max()` and 13–81× slower than polars `pl.col("time").max()`; results
are equal (asserted). Absolute cost is small at the frame level (85 ms at 500k rows); the bucket
"cutoff" in §5.2 is large because `get_cutoff` is called ~2× per **instance** (per-slice
re-entry, 2002 calls for 1000 series), not because of the frame-level call.
Latent hazard (cold path): `_get_cutoff_from_index` on a **MultiIndex** loops in Python over
`idx.droplevel(-1)` — every row, not every instance — with a `DataFrame.loc` lookup per row:

```
$ PYTHONPATH=. $EXP_PY -W ignore -c "...  _get_cutoff_from_index(make_panel_fast(n,100).index, return_index=True) ..."
100 rows 10000 1.282s 128.2us/row
1000 rows 100000 16.615s 166.1us/row
```

(only reached when a `pd.Index` is passed to `get_cutoff`; in-repo callers pass frames; the
tests in `datatypes/tests/test_utils.py` call it directly.)

## 6. (c) Container-free `forecasters_` (EstimatorGrid prototype)

`protos/a1_estimator_grid.py` provides

* `EstimatorGrid(cells, row_keys, col_keys)`: row-major list + ordered key lists + dict
  positions; `from_long(triples)` replaces `pd.DataFrame(long).pivot(...).reindex(...).reindex(...)`
  + `MultiIndex.from_tuples` (the 15 P + 7 O touchpoints of C1); `shape`, `items()`,
  `get(row, col)`, `cells()`, and a lazy `to_pandas()` that rebuilds today's DataFrame.
* `PandasCompatGrid`: + read-only shims `.loc[r,c]`, `.iloc[i,j]`, `.index`, `.columns`,
  `.values`, `.iterrows()` (via `to_pandas()`).
* patched copies of `VectorizedDF.vectorize_est`, `BaseForecaster._vectorize`,
  `BaseForecaster.get_fitted_params` that use only the grid API (key naming
  `forecasters.loc[...]` kept for compatibility — C4 is a separate decision).
* `install(mode)` monkeypatches; nothing under `sktime/` is modified.

**Cost of the pivot (C1) vs the grid** — `scripts/a1_grid_bench.py --grid dev` (placeholder
estimators, so only container cost; "build" = the post-`parallelize` pivot logic verbatim from
`vectorize_est`; "consume" = `iterrows`+`.values` cell iteration (how `vectorize_est` reads
`forecasters_` back on every predict/update) + the `get_fitted_params` `product(index,columns)`
+`.loc` key loop). Equality of cells/keys/order asserted.

```
panel n=    10 cols=1  build pd=    2.49ms grid=   0.04ms (69x)  consume pd=     0.98ms grid=   0.02ms (48x)
hier  n=    10 cols=1  build pd=    3.41ms grid=   0.04ms (82x)  consume pd=     0.90ms grid=   0.02ms (47x)
panel n=    10 cols=3  build pd=    2.10ms grid=   0.03ms (69x)  consume pd=     0.61ms grid=   0.02ms (26x)
panel n=   100 cols=1  build pd=    2.65ms grid=   0.06ms (48x)  consume pd=     2.89ms grid=   0.03ms (91x)
hier  n=   100 cols=1  build pd=    2.84ms grid=   0.07ms (43x)  consume pd=     4.65ms grid=   0.03ms (134x)
panel n=   100 cols=3  build pd=    2.39ms grid=   0.08ms (31x)  consume pd=     3.55ms grid=   0.06ms (58x)
panel n=  1000 cols=1  build pd=    4.14ms grid=   0.22ms (19x)  consume pd=    24.78ms grid=   0.14ms (179x)
hier  n=  1000 cols=1  build pd=    4.31ms grid=   0.26ms (17x)  consume pd=    45.34ms grid=   0.14ms (314x)
panel n=  1000 cols=3  build pd=    3.41ms grid=   0.40ms (9x)  consume pd=    34.22ms grid=   0.42ms (81x)
panel n= 10000 cols=1  build pd=   11.19ms grid=   2.01ms (6x)  consume pd=   231.77ms grid=   1.44ms (161x)
hier  n= 10000 cols=1  build pd=   30.17ms grid=   3.02ms (10x)  consume pd=   480.91ms grid=   1.61ms (298x)
panel n= 10000 cols=3  build pd=   17.40ms grid=   3.88ms (4x)  consume pd=   344.22ms grid=   3.98ms (87x)
```

The grid is 4–82× faster to build and 26–314× faster to consume; in absolute terms the build
is ≤30 ms at 10k series (≪1% of a vectorized fit, cf. §5.2: 3.7 s at 1k series) and the
consume side is 0.23–0.48 s at 10k series (≈1% of fit-scale cost). So C1 is a **contract /
dependency** question much more than a performance one.

Test run: `bash experiments/container_independence/scripts/a1_grid_pytest.sh` runs
`datatypes/tests/test_vectorize.py` (4378 tests), `forecasting/base/tests/test_base.py` (78),
`test_base_bugs.py` (4), `performance_metrics/forecasting/tests/test_metrics.py` (25),
`transformations/tests/test_base.py` (97) under `A1_GRID_MODE=off|strict|compat`
(pytest from a private `--target` dir on PYTHONPATH; shared env untouched).

Command: `bash experiments/container_independence/scripts/a1_grid_pytest.sh off` then
`... a1_grid_pytest.sh strict compat` (parallel). Outputs `results/a1_grid_pytest_{off,strict,compat}.txt`.

| mode | failed | passed | skipped | wall |
|---|---:|---:|---:|---:|
| off (baseline) | 701 | 3821 | 60 | 1411 s |
| strict (`EstimatorGrid`) | 2813 | 1709 | 60 | 1732 s |
| compat (`PandasCompatGrid`) | 2813 | 1709 | 60 | 1738 s |

* Baseline already has **701 failures, all in `test_vectorize.py`** (672 × `TypeError: 'NoneType'
  object is not subscriptable`, 29 × dask frame not recognised) — pre-existing breakage of the
  test module vs the stateless-VectorizedDF refactor / dask version, not caused by the grid.
* **New failures with the grid: 2112, all `test_vectorize.py::test_vectorize_est`, all the single
  assertion `assert isinstance(result, pd.DataFrame)`** (`test_vectorize.py:535`) — i.e. the
  type contract, not behaviour. 0 baseline failures were fixed; strict and compat fail the
  identical set (diff of FAILED ids: 0 lines).
* `forecasting/base/tests/test_base.py` (78), `test_base_bugs.py` (4),
  `performance_metrics/.../test_metrics.py` (25), `transformations/tests/test_base.py` (97):
  **no new failures** in either mode (fit/predict/update/predict_interval/get_fitted_params
  vectorized paths work on the grid).
* But the test selection **does not cover** the vectorized-metric path. Direct probe
  `scripts/a1_grid_metrics_probe.py` (MeanAbsoluteError on Hierarchical 2-column y, all
  multioutput × multilevel combos) → `results/a1_grid_metrics_probe.json`:

```
off|multioutput=uniform_average|multilevel=raw_values ok DataFrame (4, 1)
off|multioutput=uniform_average|multilevel=uniform_average ok float64 ()
off|multioutput=raw_values|multilevel=raw_values ok DataFrame (4, 2)
off|multioutput=raw_values|multilevel=uniform_average ok ndarray (2,)
strict|multioutput=uniform_average|multilevel=raw_values FAIL ValueError: DataFrame constructor not properly called!
strict|multioutput=uniform_average|multilevel=uniform_average FAIL AttributeError: 'EstimatorGrid' object has no attribute 'mean'
strict|multioutput=raw_values|multilevel=raw_values FAIL AttributeError: 'EstimatorGrid' object has no attribute 'iloc'
strict|multioutput=raw_values|multilevel=uniform_average FAIL AttributeError: 'EstimatorGrid' object has no attribute 'iloc'
compat|... (same 4 combos) FAIL (ValueError / no attribute 'mean' / TypeError slice*int in the .iloc shim)
```

  4/4 hierarchical metric configurations break in both modes: metrics use the
  `vectorize_est` DataFrame as a numeric result table, which the estimator grid is not.

What breaks when `forecasters_` stops being a DataFrame (code reading, confirmed by grep):

| consumer | usage | strict grid | compat grid |
|---|---|---|---|
| `BaseForecaster._vectorize` `_base.py:2459` | `len(self.forecasters_.columns)` | patched → `shape[1]` | ok |
| `get_fitted_params` `_base.py:1831-1833` | `product(index, columns)` + `.loc` | patched → `items()` | ok |
| `VectorizedDF.vectorize_est` input `_vectorize.py:766-773` | `isinstance(pd.DataFrame)`, `.shape`, `.iterrows()` | patched | ok |
| `BaseTransformer._vectorize` `transformations/base/_base.py:1537` | `self.transformers_.shape` | ok (grid has `shape`) | ok |
| performance metrics `performance_metrics/forecasting/_base.py:414-434` | `vectorize_est(...)` default return type used as a **data** table: `eval_result.iloc[:, 0].to_list()`, `.index`, `.mean(axis=0)` | breaks (measured, 4/4 configs) | breaks (measured, 4/4: `.iloc[:, 0]` slice shim, `.mean`, DataFrame ctor) |
| `cinn.py:707` | `self.forecasters_.values[0, :]` | breaks | ok (via to_pandas) |
| user code / docs | `forecasters_.loc[...]`, `.iloc`, `.iterrows()` | breaks | ok |
| `forecasters_` *name* | composites (`_fhplex.py`, `_grouped.py`, `_column_ensemble.py`, `_hierarchy_ensemble.py`, `_fallback.py`, `conformal.py`) use their **own** `forecasters_` attribute as dict/list — unaffected, but shows the attribute has no single type contract today | – | – |

Design consequence: `vectorize_est(return_type="pd.DataFrame")` currently serves two roles —
(i) estimator grid (`forecasters_`, `transformers_`), (ii) result table of scalars (metrics).
A container-free design should split them: `return_type="grid"` for (i) (pure Python,
O(1) keyed lookup, no pandas) and keep (ii) as a table built by the backend's `concat_rows`.
The pivot's only job was to restore insertion order after `DataFrame.pivot` sorts (comment at
`_vectorize.py:807-809`, issues #4683/#5108); the grid gets insertion order for free.

## 7. Behavioural findings (probes)

`scripts/a1_semantic_probes.py` → `results/a1_semantic_probes.json`:

| probe | pandas input | polars input | cause (touchpoint) |
|---|---|---|---|
| X/y time-index containment (`X-y-must-have-same-index`, disjoint dates) | **ValueError raised** ("indices of series 0 are not contained...") | **silently passes** (`mismatch_detected: False`) | `check_equal_time_index` is called on RAW inputs (`_base.py:2073`) and `get_index_for_series` duck-types `hasattr(obj, "index")` (`_utilities.py:87-90`) → polars frame treated like numpy → `RangeIndex` |
| `predict_residuals` (Series) | ok (DataFrame, len 20) | TypeError "y must be a supported Series mtype" (`_base.py:1728`) | `isinstance(y, (pd.DataFrame, pd.Series))` branch (`_base.py:1715`) |
| `update_predict` (Series) | ok, shape (9,1) | ValueError "y_preds must be a list of pd.Series or pd.DataFrame" (`_base.py:2838`) | `_predict_moving_cutoff` converts each y_pred back to the user mtype (polars) **before** the pandas-only `_format_moving_cutoff_predictions` |
| `update_predict` with `freq=None` index | freq=D: ok; freq=None: TypeError `int * NoneType` | – | cutoff shift (`_shift`, `_base.py:2722`) requires pandas `freq` on the index (C2) |
| column literally named `"X"` in a Panel | `"x"`: ok; `"X"`: TypeError "No valid mtype ..." | same TypeError | **confirmed**: `get_cutoff` duck-types the removed `VectorizedDF.X` via `hasattr(obj, "X")` (`_utilities.py:221-222`), which is True for a DataFrame with a column `X` → `obj = obj.X` (the column Series) → `convert_to` fails (`_utilities.py:235`). `check_is_scitype` on the same frame returns valid=True. |
| lazy polars Panel `y` | – | TypeError (not a recognised scitype) | no mtype for `pl.LazyFrame` panels at the forecaster boundary |
| `update_data(VectorizedDF, ...)` | AttributeError `'VectorizedDF' object has no attribute 'X'` (`_utilities.py:373`) | – | stale duck-typing after VectorizedDF became stateless (also `get_cutoff` `_utilities.py:221`) |
| conversions per Panel fit+predict | 34 convert calls, 15 non-trivial (`pd.DataFrame→pd.Series` 10, `pd.Series→pd.DataFrame` 5) | 34 calls, 17 non-trivial (+`polars_panel→pd-multiindex`, `pd-multiindex→polars_panel`) | polars input is converted once at the boundary; all per-slice conversions are pandas↔pandas |

Confirmation of the column-`"X"` cause (inline command, from WT root):

```
$ PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore -c "... get_cutoff(make_panel(3,20,n_levels=1).rename(columns={'c0':'X'}), return_index=True)"
x ok DatetimeIndex(['2000-01-20'], dtype='datetime64[ns]', freq=None)
TypeError No valid mtype could be identified for object of type <class 'pandas.core.series.Series'>. ... ['datatypes/_utilities.py:235', 'datatypes/_convert.py:264', 'datatypes/_check.py:436']
```

i.e. any pandas Panel/Hierarchical `y` with a variable called `X` cannot be fitted by a
vectorized forecaster today — a pure duck-typing bug, independent of polars.

## 8. Conclusions

1. The boundary's coupling to pandas is **mostly time-index semantics, not container ops**:
   78 time/freq/cutoff touchpoints (16 APIs) vs 42 group/slice/concat/pivot touchpoints. Any
   adapter/narwhals/backend approach must first solve freq/cutoff representation (C2/P5/P10),
   which is where pandas semantics (`Index.freq`, `pd.infer_freq`, `pd.Index` cutoff) leak most.
2. A **14-op protocol** is sufficient for VectorizedDF + BaseForecaster boundary; 4 ops are
   container ops proper. This is small enough that idea (1) (adapter per backend) is tractable *at
   the boundary* — the cost concern of idea (1) is in the inner estimators (y_inner_mtype), not here.
3. **The protocol alone buys little speed for cheap forecasters**: container layer is ≤22% of fit and
   ~2% of predict; per-slice public-API re-entry is 62–84% of fit. Gains require a trusted inner
   path (skip re-check/re-convert of slices produced by VectorizedDF) and/or native whole-panel
   implementations; both are container-independent decisions.
4. **4 API contracts** (forecasters_ DataFrame, pandas cutoff/fh, proba MultiIndex columns,
   `.loc`-syntax fitted-param keys) must be decided explicitly; today a polars user receives a mix
   of polars and pandas outputs.
5. `forecasters_` can be made container-free with a ~100-line grid; the main obstacle is that
   `vectorize_est`'s DataFrame return is also used as a numeric result table by metrics.
6. Several pandas assumptions are **silent correctness bugs** for non-pandas input (X/y index check
   passes for polars) — a protocol with explicit `backend_of` dispatch (P1) instead of
   `hasattr(..., "index")` duck typing fixes the class of bug.

## Caveats

* Static census is syntactic: dynamic attribute access via helpers outside the scoped functions
  is not counted; mtype checkers/converters (where most pandas in the check/convert buckets lives)
  are out of scope (a2).
* Dynamic counts are lower bounds (no profile events for Cython properties).
* Timings under concurrent load (other agents running); instrumentation factor 0.67–1.36×.
  Shares (percentages) within a run are more reliable than absolute seconds.
* Only DEV_GRID sizes; all scripts accept `--grid large` for the serial phase.

## Appendix A — full touchpoint table

Generated by `scripts/a1_static_touchpoints.py` (`results/a1_touchpoints_table.md`).
cat: G group/iterate, S slice/select, C concat, P pivot (estimator grid), L MultiIndex
levels/names, T time/freq/cutoff, M metadata, O output construction, D type dispatch,
K mtype pinning. Protocol op: see §3; C1–C4 = API contracts; N = legacy/non-hot path.

| # | file:line | function | token | canonical pandas API | cat | abstract op | protocol op | API-contract |
|---|---|---|---|---|---|---|---|---|
| 1 | `datatypes/_utilities.py:9` | `_get_index` | `hasattr(.., 'index')` | DataFrame/Series.index | D | duck: has index | P1 |  |
| 2 | `datatypes/_utilities.py:10` | `_get_index` | `.index` | DataFrame/Series.index | T | time_index | P5 |  |
| 3 | `datatypes/_utilities.py:13` | `_get_index` | `pd.RangeIndex` | pd.RangeIndex | T | synthesize RangeIndex for numpy | P2 |  |
| 4 | `datatypes/_utilities.py:15` | `_get_index` | `pd.RangeIndex` | pd.RangeIndex | T | synthesize RangeIndex for numpy | P2 |  |
| 5 | `datatypes/_utilities.py:17` | `_get_index` | `pd.RangeIndex` | pd.RangeIndex | T | synthesize RangeIndex for numpy | P2 |  |
| 6 | `datatypes/_utilities.py:38` | `get_time_index` | `pd.DataFrame` | pd.DataFrame | D | is pandas | P1 |  |
| 7 | `datatypes/_utilities.py:38` | `get_time_index` | `pd.Series` | pd.Series | D | is pandas | P1 |  |
| 8 | `datatypes/_utilities.py:40` | `get_time_index` | `.index` | DataFrame/Series.index | L | row keys | P2 |  |
| 9 | `datatypes/_utilities.py:40` | `get_time_index` | `pd.MultiIndex` | pd.MultiIndex | D | is multiindexed | P1 |  |
| 10 | `datatypes/_utilities.py:41` | `get_time_index` | `.index` | DataFrame/Series.index | T | time_index of first instance | P10 |  |
| 11 | `datatypes/_utilities.py:41` | `get_time_index` | `.loc` | DataFrame.loc | S | select first instance | P6 |  |
| 12 | `datatypes/_utilities.py:43` | `get_time_index` | `.iloc` | DataFrame/Series.iloc | S | first cell (nested) | N |  |
| 13 | `datatypes/_utilities.py:43` | `get_time_index` | `pd.DataFrame` | pd.DataFrame | D | is nested frame | N |  |
| 14 | `datatypes/_utilities.py:44` | `get_time_index` | `.iloc` | DataFrame/Series.iloc | S | first cell (nested) | N |  |
| 15 | `datatypes/_utilities.py:46` | `get_time_index` | `.iloc` | DataFrame/Series.iloc | S | first cell (nested) | N |  |
| 16 | `datatypes/_utilities.py:46` | `get_time_index` | `pd.DataFrame` | pd.DataFrame | D | is nested frame | N |  |
| 17 | `datatypes/_utilities.py:46` | `get_time_index` | `pd.Series` | pd.Series | D | is nested series | N |  |
| 18 | `datatypes/_utilities.py:47` | `get_time_index` | `.iloc` | DataFrame/Series.iloc | S | first cell (nested) | N |  |
| 19 | `datatypes/_utilities.py:50` | `get_time_index` | `.index` | DataFrame/Series.index | T | time_index | P5 |  |
| 20 | `datatypes/_utilities.py:55` | `get_time_index` | `pd.RangeIndex` | pd.RangeIndex | T | synthesize RangeIndex for numpy | P2 |  |
| 21 | `datatypes/_utilities.py:58` | `get_time_index` | `pd.RangeIndex` | pd.RangeIndex | T | synthesize RangeIndex for numpy | P2 |  |
| 22 | `datatypes/_utilities.py:59` | `get_time_index` | `hasattr(.., 'X')` | - | D | duck: legacy VectorizedDF.X (stale) | N |  |
| 23 | `datatypes/_utilities.py:87` | `get_index_for_series` | `hasattr(.., 'index')` | DataFrame/Series.index | D | duck: has index (else RangeIndex) | P1 |  |
| 24 | `datatypes/_utilities.py:88` | `get_index_for_series` | `.index` | DataFrame/Series.index | T | time_index | P13 |  |
| 25 | `datatypes/_utilities.py:90` | `get_index_for_series` | `pd.RangeIndex` | pd.RangeIndex | T | synthesize RangeIndex for numpy | P2 |  |
| 26 | `datatypes/_utilities.py:94` | `<module-consts>` | `'pd.DataFrame'` | - | K | get_cutoff internal mtypes | P15 |  |
| 27 | `datatypes/_utilities.py:95` | `<module-consts>` | `'pd.Series'` | - | K | get_cutoff internal mtypes | P15 |  |
| 28 | `datatypes/_utilities.py:97` | `<module-consts>` | `'pd-multiindex'` | - | K | get_cutoff internal mtypes | P15 |  |
| 29 | `datatypes/_utilities.py:99` | `<module-consts>` | `'nested_univ'` | - | K | get_cutoff internal mtypes | P15 |  |
| 30 | `datatypes/_utilities.py:100` | `<module-consts>` | `'df-list'` | - | K | get_cutoff internal mtypes | P15 |  |
| 31 | `datatypes/_utilities.py:101` | `<module-consts>` | `'pd_multiindex_hier'` | - | K | get_cutoff internal mtypes | P15 |  |
| 32 | `datatypes/_utilities.py:130` | `_get_cutoff_from_index` | `pd.Index` | pd.Index | D | is pd.Index | P1 |  |
| 33 | `datatypes/_utilities.py:141` | `_get_cutoff_from_index` | `pd.MultiIndex` | pd.MultiIndex | D | is multiindexed | P1 |  |
| 34 | `datatypes/_utilities.py:142` | `_get_cutoff_from_index` | `pd.DataFrame` | pd.DataFrame | S | frame from index for .loc lookups | P10 |  |
| 35 | `datatypes/_utilities.py:143` | `_get_cutoff_from_index` | `.droplevel` | Index.droplevel | L | instance keys | P4 |  |
| 36 | `datatypes/_utilities.py:147` | `_get_cutoff_from_index` | `.index` | DataFrame/Series.index | T | time_index of instance | P10 |  |
| 37 | `datatypes/_utilities.py:147` | `_get_cutoff_from_index` | `.loc` | DataFrame.loc | S | select instance (per instance loop) | P6 |  |
| 38 | `datatypes/_utilities.py:148` | `_get_cutoff_from_index` | `.freq` | Index.freq | T | freq | P5 |  |
| 39 | `datatypes/_utilities.py:148` | `_get_cutoff_from_index` | `hasattr(.., 'freq')` | Index.freq | T | freq | P5 |  |
| 40 | `datatypes/_utilities.py:149` | `_get_cutoff_from_index` | `.freq` | Index.freq | T | freq | P5 |  |
| 41 | `datatypes/_utilities.py:153` | `_get_cutoff_from_index` | `ss[ix]` | Index.__getitem__(int) | T | positional element of time index | P10 |  |
| 42 | `datatypes/_utilities.py:154` | `_get_cutoff_from_index` | `.get_level_values` | MultiIndex.get_level_values | T | time level values (dead store) | N |  |
| 43 | `datatypes/_utilities.py:154` | `_get_cutoff_from_index` | `.sort_values` | Index.sort_values | T | time level values (dead store) | N |  |
| 44 | `datatypes/_utilities.py:155` | `_get_cutoff_from_index` | `pd.Index` | pd.Index | T | cutoff as length-1 pd.Index | C2 | yes |
| 45 | `datatypes/_utilities.py:156` | `_get_cutoff_from_index` | `.freq` | Index.freq | T | set freq on cutoff | C2 | yes |
| 46 | `datatypes/_utilities.py:159` | `_get_cutoff_from_index` | `.freq` | Index.freq | T | freq | P5 |  |
| 47 | `datatypes/_utilities.py:159` | `_get_cutoff_from_index` | `hasattr(.., 'freq')` | Index.freq | T | freq | P5 |  |
| 48 | `datatypes/_utilities.py:160` | `_get_cutoff_from_index` | `.freq` | Index.freq | T | freq | P5 |  |
| 49 | `datatypes/_utilities.py:165` | `_get_cutoff_from_index` | `time_idx[ix]` | Index.__getitem__(int) | T | positional element of time index | P10 |  |
| 50 | `datatypes/_utilities.py:166` | `_get_cutoff_from_index` | `time_idx[[ix]]` | Index.__getitem__(list) | T | length-1 index slice (keeps type) | C2 | yes |
| 51 | `datatypes/_utilities.py:167` | `_get_cutoff_from_index` | `.freq` | Index.freq | T | freq | P5 |  |
| 52 | `datatypes/_utilities.py:167` | `_get_cutoff_from_index` | `hasattr(.., 'freq')` | Index.freq | T | freq | P5 |  |
| 53 | `datatypes/_utilities.py:168` | `_get_cutoff_from_index` | `.freq` | Index.freq | T | freq | P5 |  |
| 54 | `datatypes/_utilities.py:221` | `get_cutoff` | `hasattr(.., 'X')` | - | D | duck: legacy VectorizedDF.X (stale) | N |  |
| 55 | `datatypes/_utilities.py:224` | `get_cutoff` | `pd.Index` | pd.Index | D | is pd.Index | P1 |  |
| 56 | `datatypes/_utilities.py:235` | `get_cutoff` | `convert_to` | - | K | convert input to pandas/numpy for cutoff | P15 |  |
| 57 | `datatypes/_utilities.py:239` | `get_cutoff` | `pd.Index` | pd.Index | D | is cutoff pd.Index | P1 | yes |
| 58 | `datatypes/_utilities.py:247` | `get_cutoff` | `cutoff[0]` | Index.__getitem__(int) | T | unwrap length-1 pd.Index cutoff | C2 | yes |
| 59 | `datatypes/_utilities.py:261` | `get_cutoff` | `pd.RangeIndex` | pd.RangeIndex | T | cutoff for numpy as RangeIndex | C2 | yes |
| 60 | `datatypes/_utilities.py:278` | `get_cutoff.sub_idx` | `idx[ix]` | Index.__getitem__(int) | T | positional element of time index | P10 |  |
| 61 | `datatypes/_utilities.py:279` | `get_cutoff.sub_idx` | `idx[[ix]]` | Index.__getitem__(list) | T | length-1 index slice (keeps type) | C2 | yes |
| 62 | `datatypes/_utilities.py:280` | `get_cutoff` | `hasattr(.., 'freq')` | Index.freq | T | freq | P5 |  |
| 63 | `datatypes/_utilities.py:281` | `get_cutoff` | `.freq` | Index.freq | T | freq | P5 |  |
| 64 | `datatypes/_utilities.py:283` | `get_cutoff` | `.freq` | Index.freq | T | freq | P5 |  |
| 65 | `datatypes/_utilities.py:283` | `get_cutoff` | `pd.infer_freq` | pd.infer_freq | T | infer_freq | P5 |  |
| 66 | `datatypes/_utilities.py:290` | `get_cutoff` | `.freq` | Index.freq | T | freq | P5 |  |
| 67 | `datatypes/_utilities.py:291` | `get_cutoff` | `.freq` | Index.freq | T | freq | P5 |  |
| 68 | `datatypes/_utilities.py:294` | `get_cutoff` | `pd.Series` | pd.Series | D | is pd.Series | P1 |  |
| 69 | `datatypes/_utilities.py:295` | `get_cutoff` | `.index` | DataFrame/Series.index | T | time_index | P10 |  |
| 70 | `datatypes/_utilities.py:298` | `get_cutoff` | `.index` | DataFrame/Series.index | L | row keys | P2 |  |
| 71 | `datatypes/_utilities.py:298` | `get_cutoff` | `pd.DataFrame` | pd.DataFrame | D | is pd.DataFrame | P1 |  |
| 72 | `datatypes/_utilities.py:298` | `get_cutoff` | `pd.MultiIndex` | pd.MultiIndex | D | is multiindexed | P1 |  |
| 73 | `datatypes/_utilities.py:299` | `get_cutoff` | `.columns` | DataFrame.columns | M | column_names | P3 |  |
| 74 | `datatypes/_utilities.py:299` | `get_cutoff` | `.dtypes` | DataFrame.dtypes | M | dtypes (object -> nested) | N |  |
| 75 | `datatypes/_utilities.py:302` | `get_cutoff` | `.index` | DataFrame/Series.index | T | time_index | P10 |  |
| 76 | `datatypes/_utilities.py:306` | `get_cutoff` | `.index` | DataFrame/Series.index | T | time_index (nested) | N |  |
| 77 | `datatypes/_utilities.py:311` | `get_cutoff` | `.index` | DataFrame/Series.index | L | row keys | P2 |  |
| 78 | `datatypes/_utilities.py:311` | `get_cutoff` | `pd.DataFrame` | pd.DataFrame | D | is pd.DataFrame | P1 |  |
| 79 | `datatypes/_utilities.py:311` | `get_cutoff` | `pd.MultiIndex` | pd.MultiIndex | D | is multiindexed | P1 |  |
| 80 | `datatypes/_utilities.py:314` | `get_cutoff` | `.index` | DataFrame/Series.index | L | row keys | P2 |  |
| 81 | `datatypes/_utilities.py:314` | `get_cutoff` | `.nlevels` | Index.nlevels | L | n_index_levels | P2 |  |
| 82 | `datatypes/_utilities.py:316` | `get_cutoff` | `.agg` | Series.agg | G | global max of groupwise last | P10 |  |
| 83 | `datatypes/_utilities.py:316` | `get_cutoff` | `.groupby` | DataFrame.groupby | G | groupwise last time point | P10 |  |
| 84 | `datatypes/_utilities.py:316` | `get_cutoff` | `.iloc` | DataFrame/Series.iloc | G | groupwise last time point | P10 |  |
| 85 | `datatypes/_utilities.py:316` | `get_cutoff` | `.index` | DataFrame/Series.index | L | row keys | P2 |  |
| 86 | `datatypes/_utilities.py:316` | `get_cutoff` | `.nth` | GroupBy.nth | G | groupwise last time point | P10 |  |
| 87 | `datatypes/_utilities.py:316` | `get_cutoff` | `.to_frame` | Index.to_frame | G | groupwise last time point | P10 |  |
| 88 | `datatypes/_utilities.py:324` | `get_cutoff` | `.index` | DataFrame/Series.index | T | time level | P10 |  |
| 89 | `datatypes/_utilities.py:324` | `get_cutoff` | `.levels` | MultiIndex.levels | T | time level | P10 |  |
| 90 | `datatypes/_utilities.py:325` | `get_cutoff` | `.freq` | Index.freq | T | freq | P5 |  |
| 91 | `datatypes/_utilities.py:325` | `get_cutoff` | `hasattr(.., 'freq')` | Index.freq | T | freq | P5 |  |
| 92 | `datatypes/_utilities.py:326` | `get_cutoff` | `.freq` | Index.freq | T | freq | P5 |  |
| 93 | `datatypes/_utilities.py:327` | `get_cutoff` | `.freq` | Index.freq | T | freq | P5 |  |
| 94 | `datatypes/_utilities.py:334` | `get_cutoff` | `.index` | DataFrame/Series.index | T | time_index (df-list) | N |  |
| 95 | `datatypes/_utilities.py:339` | `<module-consts>` | `'pd.DataFrame'` | - | K | update_data internal mtypes | P15 |  |
| 96 | `datatypes/_utilities.py:340` | `<module-consts>` | `'pd.Series'` | - | K | update_data internal mtypes | P15 |  |
| 97 | `datatypes/_utilities.py:342` | `<module-consts>` | `'pd-multiindex'` | - | K | update_data internal mtypes | P15 |  |
| 98 | `datatypes/_utilities.py:344` | `<module-consts>` | `'pd_multiindex_hier'` | - | K | update_data internal mtypes | P15 |  |
| 99 | `datatypes/_utilities.py:379` | `update_data` | `convert_to` | - | K | convert to pandas/numpy for update | P15 |  |
| 100 | `datatypes/_utilities.py:400` | `update_data` | `pd.DataFrame` | pd.DataFrame | D | is pandas | P1 |  |
| 101 | `datatypes/_utilities.py:400` | `update_data` | `pd.Series` | pd.Series | D | is pandas | P1 |  |
| 102 | `datatypes/_utilities.py:405` | `update_data` | `.combine_first` | DataFrame.combine_first | C | upsert rows by index (append/overwrite) | P11 |  |
| 103 | `datatypes/_utilities.py:406` | `update_data` | `.index` | DataFrame/Series.index | T | time_index | P11 |  |
| 104 | `datatypes/_utilities.py:406` | `update_data` | `hasattr(.., 'freq')` | Index.freq | T | freq | P5 |  |
| 105 | `datatypes/_utilities.py:407` | `update_data` | `.freq` | Index.freq | T | freq | P5 |  |
| 106 | `datatypes/_utilities.py:407` | `update_data` | `.index` | DataFrame/Series.index | T | time_index | P11 |  |
| 107 | `datatypes/_utilities.py:408` | `update_data` | `.freqstr` | Index.freqstr | T | freq | P5 |  |
| 108 | `datatypes/_utilities.py:408` | `update_data` | `.index` | DataFrame/Series.index | T | time_index | P11 |  |
| 109 | `datatypes/_utilities.py:409` | `update_data` | `.freq` | Index.freq | T | freq | P5 |  |
| 110 | `datatypes/_utilities.py:409` | `update_data` | `.index` | DataFrame/Series.index | T | time_index | P11 |  |
| 111 | `datatypes/_vectorize.py:24` | `<module>` | `'pd.DataFrame'` | - | K | target mtype for Series frame | P15 |  |
| 112 | `datatypes/_vectorize.py:25` | `<module>` | `'pd-multiindex'` | - | K | target mtype for Panel frame | P15 |  |
| 113 | `datatypes/_vectorize.py:26` | `<module>` | `'pd_multiindex_hier'` | - | K | target mtype for Hierarchical frame | P15 |  |
| 114 | `datatypes/_vectorize.py:27` | `<module>` | `'pd-multiindex'` | - | K | target mtype for Panel frame | P15 |  |
| 115 | `datatypes/_vectorize.py:27` | `<module>` | `'pd.DataFrame'` | - | K | target mtype for Series frame | P15 |  |
| 116 | `datatypes/_vectorize.py:27` | `<module>` | `'pd_multiindex_hier'` | - | K | target mtype for Hierarchical frame | P15 |  |
| 117 | `datatypes/_vectorize.py:90` | `coerce_to_multiindex.infer_freq` | `len` | - | M | n_rows(frame) | P3 |  |
| 118 | `datatypes/_vectorize.py:93` | `coerce_to_multiindex` | `.index` | DataFrame/Series.index | L | row keys of frame | P2 |  |
| 119 | `datatypes/_vectorize.py:94` | `coerce_to_multiindex` | `pd.MultiIndex` | pd.MultiIndex | D | is_multiindexed(frame) | P1 |  |
| 120 | `datatypes/_vectorize.py:95` | `coerce_to_multiindex` | `.droplevel` | Index.droplevel | L | instance_keys(frame) | P4 |  |
| 121 | `datatypes/_vectorize.py:95` | `coerce_to_multiindex` | `.index` | DataFrame/Series.index | T | time_index(slice) | P5 |  |
| 122 | `datatypes/_vectorize.py:95` | `coerce_to_multiindex` | `.xs` | DataFrame.xs | S | select_instance(frame, key) -> time slice | P6 |  |
| 123 | `datatypes/_vectorize.py:99` | `coerce_to_multiindex` | `getattr(.., 'freqstr')` | Index.freqstr | T | freq(time_index) | P5 |  |
| 124 | `datatypes/_vectorize.py:100` | `coerce_to_multiindex` | `.freqstr` | Index.freqstr | T | freq(time_index) | P5 |  |
| 125 | `datatypes/_vectorize.py:102` | `coerce_to_multiindex` | `pd.infer_freq` | pd.infer_freq | T | infer_freq(time_index) | P5 |  |
| 126 | `datatypes/_vectorize.py:241` | `VectorizedDF.__init__` | `.columns` | DataFrame.columns | M | column_names(frame) | P3 |  |
| 127 | `datatypes/_vectorize.py:242` | `VectorizedDF.__init__` | `.index` | DataFrame/Series.index | L | row keys of frame | P2 |  |
| 128 | `datatypes/_vectorize.py:242` | `VectorizedDF.__init__` | `.names` | Index.names | L | index_level_names(frame) | P2 |  |
| 129 | `datatypes/_vectorize.py:243` | `VectorizedDF.__init__` | `.index` | DataFrame/Series.index | L | row keys of frame | P2 |  |
| 130 | `datatypes/_vectorize.py:243` | `VectorizedDF.__init__` | `.nlevels` | Index.nlevels | L | n_index_levels(frame) | P2 |  |
| 131 | `datatypes/_vectorize.py:244` | `VectorizedDF.__init__` | `.index` | DataFrame/Series.index | L | row keys of frame | P2 |  |
| 132 | `datatypes/_vectorize.py:292` | `VectorizedDF._iter_indices_from_index` | `.droplevel` | Index.droplevel | L | instance_keys(frame, levels) (unique) | P4 |  |
| 133 | `datatypes/_vectorize.py:292` | `VectorizedDF._iter_indices_from_index` | `.unique` | Index/Series.unique | L | instance_keys(frame, levels) (unique) | P4 |  |
| 134 | `datatypes/_vectorize.py:294` | `VectorizedDF._iter_indices_from_index` | `.droplevel` | Index.droplevel | L | instance_keys(frame, levels) (unique) | P4 |  |
| 135 | `datatypes/_vectorize.py:294` | `VectorizedDF._iter_indices_from_index` | `.unique` | Index/Series.unique | L | instance_keys(frame, levels) (unique) | P4 |  |
| 136 | `datatypes/_vectorize.py:368` | `VectorizedDF.items` | `.columns` | DataFrame.columns | M | column_names(slice) | P3 |  |
| 137 | `datatypes/_vectorize.py:369` | `VectorizedDF.items._iter_cols` | `inst[[col]]` | DataFrame.__getitem__(list) | S | select_columns(slice, [col]) | P7 |  |
| 138 | `datatypes/_vectorize.py:396` | `VectorizedDF.items` | `.groupby` | DataFrame.groupby | G | iter_groups(frame, levels) -> (key, slice) | P6 |  |
| 139 | `datatypes/_vectorize.py:397` | `VectorizedDF.items` | `.droplevel` | Index.droplevel | L | drop instance levels from slice | P6 |  |
| 140 | `datatypes/_vectorize.py:516` | `VectorizedDF.reconstruct` | `pd.DataFrame` | pd.DataFrame | D | is_native_frame(x) | P1 |  |
| 141 | `datatypes/_vectorize.py:527` | `VectorizedDF.reconstruct` | `.columns` | DataFrame.columns | M | column_names(slice) | P3 |  |
| 142 | `datatypes/_vectorize.py:529` | `VectorizedDF.reconstruct` | `.columns` | DataFrame.columns | M | column_names(slice) | P3 |  |
| 143 | `datatypes/_vectorize.py:536` | `VectorizedDF.reconstruct` | `pd.DataFrame` | pd.DataFrame | C | identity reconstruct (single slice) | P8 |  |
| 144 | `datatypes/_vectorize.py:538` | `VectorizedDF.reconstruct` | `pd.concat` | pd.concat | C | concat_rows(slices, keys=instance_keys) | P8 |  |
| 145 | `datatypes/_vectorize.py:545` | `VectorizedDF.reconstruct` | `pd.concat` | pd.concat | C | concat_cols(slices, keys=col_keys) | P9 |  |
| 146 | `datatypes/_vectorize.py:557` | `VectorizedDF.reconstruct` | `pd.concat` | pd.concat | C | concat_cols(slices, keys=col_keys) | P9 |  |
| 147 | `datatypes/_vectorize.py:559` | `VectorizedDF.reconstruct` | `pd.concat` | pd.concat | C | concat_rows(slices, keys=instance_keys) | P8 |  |
| 148 | `datatypes/_vectorize.py:561` | `VectorizedDF.reconstruct` | `.index` | DataFrame/Series.index | L | row keys of frame | P2 |  |
| 149 | `datatypes/_vectorize.py:564` | `VectorizedDF.reconstruct` | `.columns` | DataFrame.columns | M | column_names(frame) | P3 |  |
| 150 | `datatypes/_vectorize.py:564` | `VectorizedDF.reconstruct` | `pd.MultiIndex` | pd.MultiIndex | D | has_multiindex_columns(frame) | P1 |  |
| 151 | `datatypes/_vectorize.py:565` | `VectorizedDF.reconstruct` | `.columns` | DataFrame.columns | O | flatten column names 'a__b' | P9 |  |
| 152 | `datatypes/_vectorize.py:567` | `VectorizedDF.reconstruct` | `.names` | Index.names | L | index_level_names(frame) | P2 |  |
| 153 | `datatypes/_vectorize.py:568` | `VectorizedDF.reconstruct` | `.index` | DataFrame/Series.index | L | set index level names | P2 |  |
| 154 | `datatypes/_vectorize.py:568` | `VectorizedDF.reconstruct` | `.set_names` | Index.set_names | L | set index level names | P2 |  |
| 155 | `datatypes/_vectorize.py:726` | `VectorizedDF.vectorize_est` | `'pd.DataFrame'` | - | K | return_type selector of vectorize_est | P15 | yes |
| 156 | `datatypes/_vectorize.py:766` | `VectorizedDF.vectorize_est` | `pd.DataFrame` | pd.DataFrame | D | is estimator grid a DataFrame | P1 | yes |
| 157 | `datatypes/_vectorize.py:767` | `VectorizedDF.vectorize_est` | `.shape` | DataFrame.shape | P | estimator grid shape check | C1 | yes |
| 158 | `datatypes/_vectorize.py:773` | `VectorizedDF.vectorize_est` | `.iterrows` | DataFrame.iterrows | P | iterate estimator grid cells row-major | C1 | yes |
| 159 | `datatypes/_vectorize.py:773` | `VectorizedDF.vectorize_est` | `.values` | Series.values | P | iterate estimator grid cells row-major | C1 | yes |
| 160 | `datatypes/_vectorize.py:801` | `VectorizedDF.vectorize_est` | `'pd.DataFrame'` | - | K | return_type selector of vectorize_est | P15 | yes |
| 161 | `datatypes/_vectorize.py:802` | `VectorizedDF.vectorize_est` | `pd.DataFrame` | pd.DataFrame | P | long list of (row,col,est) -> table | C1 | yes |
| 162 | `datatypes/_vectorize.py:803` | `VectorizedDF.vectorize_est` | `.loc` | DataFrame.loc | P | unique col keys in order | C1 | yes |
| 163 | `datatypes/_vectorize.py:803` | `VectorizedDF.vectorize_est` | `.unique` | Index/Series.unique | P | unique col keys in order | C1 | yes |
| 164 | `datatypes/_vectorize.py:804` | `VectorizedDF.vectorize_est` | `.loc` | DataFrame.loc | P | unique row keys in order | C1 | yes |
| 165 | `datatypes/_vectorize.py:804` | `VectorizedDF.vectorize_est` | `.unique` | Index/Series.unique | P | unique row keys in order | C1 | yes |
| 166 | `datatypes/_vectorize.py:806` | `VectorizedDF.vectorize_est` | `.pivot` | DataFrame.pivot | P | pivot long -> estimator grid | C1 | yes |
| 167 | `datatypes/_vectorize.py:810` | `VectorizedDF.vectorize_est` | `.reindex` | DataFrame.reindex | P | restore column order after pivot | C1 | yes |
| 168 | `datatypes/_vectorize.py:811` | `VectorizedDF.vectorize_est` | `.reindex` | DataFrame.reindex | P | restore row order after pivot | C1 | yes |
| 169 | `datatypes/_vectorize.py:814` | `VectorizedDF.vectorize_est` | `.index` | DataFrame/Series.index | O | clear index names of grid | C1 | yes |
| 170 | `datatypes/_vectorize.py:814` | `VectorizedDF.vectorize_est` | `.names` | Index.names | O | clear index names of grid | C1 | yes |
| 171 | `datatypes/_vectorize.py:815` | `VectorizedDF.vectorize_est` | `.columns` | DataFrame.columns | O | clear column name of grid | C1 | yes |
| 172 | `datatypes/_vectorize.py:815` | `VectorizedDF.vectorize_est` | `.name` | Index.name | O | clear column name of grid | C1 | yes |
| 173 | `datatypes/_vectorize.py:819` | `VectorizedDF.vectorize_est` | `.index` | DataFrame/Series.index | O | grid row index -> MultiIndex of tuples | C1 | yes |
| 174 | `datatypes/_vectorize.py:819` | `VectorizedDF.vectorize_est` | `pd.MultiIndex.from_tuples` | pd.MultiIndex.from_tuples | O | grid row index -> MultiIndex | C1 | yes |
| 175 | `datatypes/_vectorize.py:866` | `_enforce_index_freq` | `.freq` | Index.freq | T | freq(time_index of slice) | P5 |  |
| 176 | `datatypes/_vectorize.py:866` | `_enforce_index_freq` | `.index` | DataFrame/Series.index | T | time_index(slice) | P5 |  |
| 177 | `datatypes/_vectorize.py:866` | `_enforce_index_freq` | `hasattr(.., 'freq')` | Index.freq | T | freq(time_index of slice) | P5 |  |
| 178 | `datatypes/_vectorize.py:867` | `_enforce_index_freq` | `.index` | DataFrame/Series.index | T | time_index(slice) | P5 |  |
| 179 | `datatypes/_vectorize.py:868` | `_enforce_index_freq` | `.freq` | Index.freq | T | set freq on slice time index (per slice!) | P5 |  |
| 180 | `datatypes/_vectorize.py:868` | `_enforce_index_freq` | `.index` | DataFrame/Series.index | T | time_index(slice) | P5 |  |
| 181 | `datatypes/_vectorize.py:868` | `_enforce_index_freq` | `pd.infer_freq` | pd.infer_freq | T | infer_freq per slice | P5 |  |
| 182 | `forecasting/base/_base.py:115` | `<class-attrs>` | `'pd.Series'` | - | K | default y_inner_mtype tag | P15 | yes |
| 183 | `forecasting/base/_base.py:116` | `<class-attrs>` | `'pd.DataFrame'` | - | K | default X_inner_mtype tag | P15 | yes |
| 184 | `forecasting/base/_base.py:603` | `BaseForecaster.predict` | `convert_to` | - | K | back-convert prediction to last-seen mtype | P15 |  |
| 185 | `forecasting/base/_base.py:1187` | `BaseForecaster.pretrain` | `'pd-multiindex'` | - | K | pretrain forces pandas panel mtypes | P15 |  |
| 186 | `forecasting/base/_base.py:1187` | `BaseForecaster.pretrain` | `'pd_multiindex_hier'` | - | K | pretrain forces pandas panel mtypes | P15 |  |
| 187 | `forecasting/base/_base.py:1715` | `BaseForecaster.predict_residuals` | `pd.DataFrame` | pd.DataFrame | D | is y index-carrying (pandas) | P1 |  |
| 188 | `forecasting/base/_base.py:1715` | `BaseForecaster.predict_residuals` | `pd.Series` | pd.Series | D | is y index-carrying (pandas) | P1 |  |
| 189 | `forecasting/base/_base.py:1716` | `BaseForecaster.predict_residuals` | `.index` | DataFrame/Series.index | T | fh := time index of y (absolute) | C2 | yes |
| 190 | `forecasting/base/_base.py:1735` | `BaseForecaster.predict_residuals` | `y - y_pred` | DataFrame.__sub__ (index-aligned) | C | index-ALIGNED subtraction (residuals) | P14 | yes |
| 191 | `forecasting/base/_base.py:1831` | `BaseForecaster.get_fitted_params` | `.columns` | DataFrame.columns | P | grid col keys | C1 | yes |
| 192 | `forecasting/base/_base.py:1831` | `BaseForecaster.get_fitted_params` | `.index` | DataFrame/Series.index | P | grid row keys | C1 | yes |
| 193 | `forecasting/base/_base.py:1832` | `BaseForecaster.get_fitted_params` | `.loc` | DataFrame.loc | P | grid cell lookup by (row,col) key | C1 | yes |
| 194 | `forecasting/base/_base.py:1833` | `BaseForecaster.get_fitted_params` | `'forecasters.loc[...]'` | - | O | fitted-param KEY naming embeds pandas .loc syntax | C4 | yes |
| 195 | `forecasting/base/_base.py:2073` | `BaseForecaster._check_X_y` | `check_equal_time_index` | - | T | X time index contains y's | P13 |  |
| 196 | `forecasting/base/_base.py:2214` | `BaseForecaster._set_cutoff` | `pd.Index` | pd.Index | D | is cutoff a pd.Index | P1 | yes |
| 197 | `forecasting/base/_base.py:2215` | `BaseForecaster._set_cutoff` | `pd.Index` | pd.Index | T | cutoff stored as length-1 pd.Index | C2 | yes |
| 198 | `forecasting/base/_base.py:2459` | `BaseForecaster._vectorize` | `.columns` | DataFrame.columns | P | n_cols of estimator grid | C1 | yes |
| 199 | `forecasting/base/_base.py:2466` | `BaseForecaster._vectorize` | `.columns` | DataFrame.columns | O | drop helper column level of predictions | P9 |  |
| 200 | `forecasting/base/_base.py:2466` | `BaseForecaster._vectorize` | `.droplevel` | Index.droplevel | O | drop helper column level of predictions | P9 |  |
| 201 | `forecasting/base/_base.py:2722` | `BaseForecaster._predict_moving_cutoff` | `_shift` | - | T | shift cutoff by -1 period (pandas freq arithmetic) | C2 |  |
| 202 | `forecasting/base/_base.py:2725` | `BaseForecaster._predict_moving_cutoff` | `cv.split(y)` | - | T | splitter windows over y's pandas time index | C2 |  |
| 203 | `forecasting/base/_base.py:2726` | `BaseForecaster._predict_moving_cutoff` | `.iloc` | DataFrame/Series.iloc | S | positional row slice of y (cv window) | P12 |  |
| 204 | `forecasting/base/_base.py:2797` | `BaseForecaster._get_columns` | `pd.MultiIndex.from_product` | pd.MultiIndex.from_product | O | predict_interval columns (var, coverage, lower/upper) | C3 | yes |
| 205 | `forecasting/base/_base.py:2805` | `BaseForecaster._get_columns` | `pd.MultiIndex.from_product` | pd.MultiIndex.from_product | O | predict_quantiles columns (var, alpha) | C3 | yes |
| 206 | `forecasting/base/_base.py:2836` | `_format_moving_cutoff_predictions` | `pd.DataFrame` | pd.DataFrame | O | empty moving-cutoff result | P8 | yes |
| 207 | `forecasting/base/_base.py:2837` | `_format_moving_cutoff_predictions` | `pd.DataFrame` | pd.DataFrame | D | is native frame/series | P1 |  |
| 208 | `forecasting/base/_base.py:2837` | `_format_moving_cutoff_predictions` | `pd.Series` | pd.Series | D | is native frame/series | P1 |  |
| 209 | `forecasting/base/_base.py:2841` | `_format_moving_cutoff_predictions` | `pd.DataFrame` | pd.DataFrame | D | is native frame | P1 |  |
| 210 | `forecasting/base/_base.py:2842` | `_format_moving_cutoff_predictions` | `.columns` | DataFrame.columns | M | column_names | P3 |  |
| 211 | `forecasting/base/_base.py:2851` | `_format_moving_cutoff_predictions` | `pd.DataFrame` | pd.DataFrame | D | is native frame | P1 |  |
| 212 | `forecasting/base/_base.py:2853` | `_format_moving_cutoff_predictions` | `.columns` | DataFrame.columns | M | column_names | P3 |  |
| 213 | `forecasting/base/_base.py:2853` | `_format_moving_cutoff_predictions` | `.equals` | Index.equals | M | column_names equal | P3 |  |
| 214 | `forecasting/base/_base.py:2858` | `_format_moving_cutoff_predictions` | `pd.concat` | pd.concat | C | concat_rows(predictions) | P8 | yes |
| 215 | `forecasting/base/_base.py:2860` | `_format_moving_cutoff_predictions` | `cutoff[0]` | Index.__getitem__(int) | T | unwrap length-1 pd.Index cutoff | C2 | yes |
| 216 | `forecasting/base/_base.py:2861` | `_format_moving_cutoff_predictions` | `pd.concat` | pd.concat | C | concat_cols(predictions, keys=cutoffs) | P9 | yes |
| 217 | `forecasting/base/_base.py:2863` | `_format_moving_cutoff_predictions` | `.index` | DataFrame/Series.index | T | is time index sorted | P12 |  |
| 218 | `forecasting/base/_base.py:2863` | `_format_moving_cutoff_predictions` | `.is_monotonic_increasing` | Index.is_monotonic_increasing | T | is time index sorted | P12 |  |
| 219 | `forecasting/base/_base.py:2864` | `_format_moving_cutoff_predictions` | `.sort_index` | DataFrame.sort_index | S | sort rows by index | P12 |  |
| 220 | `forecasting/base/_base.py:2866` | `_format_moving_cutoff_predictions` | `.columns` | DataFrame.columns | M | column_names | P3 |  |
| 221 | `forecasting/base/_base.py:2866` | `_format_moving_cutoff_predictions` | `hasattr(.., 'columns')` | DataFrame.columns | D | duck: has columns | P1 |  |
| 222 | `forecasting/base/_base.py:2866` | `_format_moving_cutoff_predictions` | `pd.MultiIndex` | pd.MultiIndex | D | has_multiindex_columns | P1 |  |
| 223 | `forecasting/base/_base.py:2867` | `_format_moving_cutoff_predictions` | `.columns` | DataFrame.columns | M | column_names | P3 |  |
| 224 | `forecasting/base/_base.py:2868` | `_format_moving_cutoff_predictions` | `.loc` | DataFrame.loc | S | select_columns (reorder to original order) | P7 |  |
| 225 | `utils/multiindex.py:39` | `flatten_multiindex` | `pd.Index` | pd.Index | O | flat str column names | P9 |  |
| 226 | `utils/validation/series.py:268` | `check_equal_time_index` | `.equals` | Index.equals | T | time index equality | P13 |  |
| 227 | `utils/validation/series.py:271` | `check_equal_time_index` | `.symmetric_difference` | Index.symmetric_difference | T | time index diff (msg) | P13 |  |
| 228 | `utils/validation/series.py:274` | `check_equal_time_index` | `.isin` | Index.isin | T | time index containment | P13 |  |
| 229 | `utils/validation/series.py:277` | `check_equal_time_index` | `.difference` | Index.difference | T | time index diff (msg) | P13 |  |

