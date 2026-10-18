# x4: adapter/dispatcher-based VectorizedDF, and how far it reaches into forecasters

Workstream `x4_proto_adapter`. This evaluates idea (1), "a pandas adapter when pandas
is passed, a polars adapter when polars is passed", by building it and running real
sktime forecasters through it. Every number below comes from a script in
`experiments/container_independence/scripts/x4_*.py`; the commands are in the
[reproduce](#reproduce) section. Anything not measured is marked *unmeasured/expected*.

## TL;DR

1. **At the VectorizedDF level the adapter approach works and is cheap.**
   - A 15-operation protocol covers everything `VectorizedDF` and `BaseForecaster._vectorize` do with a frame.
   - The pandas adapter is bit-identical to today's code: 78/78 configurations match under both pandas 2.3.3 and 3.0.6, at a cost ratio of 0.85 to 1.00.
   - The polars adapter iterates and reconstructs with zero pandas conversions (26/26 configurations match the pandas reference). It iterates 2x to 27x faster than today's convert-then-`groupby` path, and iterate plus reconstruct is 6x to 22x faster.
   - A new backend costs 67 to 100 logical lines of code (LOC) for the adapter. Its pandas-free sktime mtype family (check plus convert) costs more, about 370 LOC for polars.
2. **It stops at the forecaster's `y_inner_mtype`.**
   - 152 of 160 registered forecasters compute only in pandas. The other 8 are pass-through compositors, so 0 leaf forecasters compute in polars.
   - For a pandas-inner forecaster, iterating in the input's backend ("native" dispatch) does not remove the conversion. It splits it into N per-slice conversions:
     - 2N conversion calls instead of 2.
     - 1.8x to 3.2x more conversion time in isolation.
     - FORECASTER_A2_SUMMARY
   - **Dispatching on the input type is the wrong key.**
3. **The bridge is to dispatch on the forecaster's inner-mtype family instead.** VectorizedDF should iterate in the backend of `y_inner_mtype` (a tag that already exists): convert the whole frame once to that family, then use that family's adapter.
   - For pandas-inner forecasters this is today's behaviour (1+1 conversions).
   - For a polars-inner forecaster it gives **0 conversions** end to end: polars in, polars out, Panel included. FORECASTER_B_SUMMARY
   - Three small framework fixes are also needed:
     - a backend-aware cutoff (`get_cutoff` silently converts whole non-pandas frames to pandas, twice per `fit`);
     - an adapter op in place of `y_pred.columns.droplevel(1)` (`_base.py:2466`, which crashes for polars);
     - passing the already-known mtype instead of re-checking it (the polars mtype check costs 1.9 s at 10000x50, versus 0.8 ms for pandas).
4. **The adapter approach cannot carry laziness into per-instance forecasters.**
   - Schema construction needs only the key columns, and reconstruct stays lazy.
   - But per-instance forecasters need materialized slices. Per-slice lazy filters are 5x to 130x slower than one `collect()` followed by `partition_by`.
   - To keep the value of lazy or out-of-core backends, the per-instance computation must itself become a frame expression (group-wise, in-engine). That is a different approach from a Python loop over adapters.
5. **Prior art points the same way.** In scikit-learn 1.9.1:
   - The boundary adapter (`set_output`: 4 methods, 2 adapters) reaches 84/84 transformers.
   - Per-estimator internal dispatch (array API) reaches 16/184 default-instantiable estimators.
   - Boundary adapters scale to every estimator for free; rewriting estimator internals is opt-in, one estimator at a time. In sktime, the inner-mtype tag is that opt-in switch.

## 1. What was built

| file | content |
|---|---|
| `protos/x4_adapter.py` | `ContainerAdapter` protocol (15 ops), `AdapterRegistry` (dispatch), `PandasAdapter`, `PolarsAdapter`, `PolarsLazyAdapter`, `PyArrowAdapter`, `VectorizedDF2`, `prepare_VectorizedDF2(target=...)`, `family_of_inner_mtype`, `adapter_get_cutoff` |
| `protos/x4_polars_forecaster.py` | `PolarsNaiveLast`, a toy `BaseForecaster` declaring `y_inner_mtype="pl.DataFrame"` (same strategy as `NaiveForecaster("last")`) |
| `protos/x4_instrument.py` | `ConvCounter` (counts every pandas<->polars conversion by wrapping `pl.DataFrame.to_pandas` and `polars.from_pandas`, the two functions all sktime polars converters go through); `patched_base(policy, cutoff)` (swaps `prepare_VectorizedDF` and `get_cutoff` **inside the `_base` module namespace only**; no file under `sktime/` was modified) |

### 1.1 Protocol (derived from the code, not invented)

| op | current pandas code it replaces |
|---|---|
| `is_native` | `isinstance(x, pd.DataFrame)` in `reconstruct` |
| `key_names` | `X.index.names` |
| `value_columns` | `X.columns` |
| `n_rows` | `len(X)` |
| `instance_keys(k)` | `X.index.droplevel(-1 / [-1,-2]).unique()` |
| `iter_groups(k)` | `X.groupby(level=..., sort=False)` + `group.droplevel(...)` |
| `select_cols` | `inst[[col]]` |
| `finalize_slice` | `_enforce_index_freq(inst)` |
| `concat_rows_with_keys` | `pd.concat(parts, keys=row_ix)` |
| `concat_cols` | `pd.concat(parts, axis=1, keys=col_keys)` |
| `flatten_columns` | `flatten_multiindex(X.columns)` |
| `set_key_names` | `X.index.set_names(...)` |
| `time_max` | `get_cutoff(...)` (`BaseForecaster._set_cutoff_from_y`) |
| `infer_freq` | `coerce_to_multiindex.infer_freq` |
| `to_numpy` | `.values` in numpy-based `_fit` code |

That is 15 operations, against 4 in sklearn's `set_output` protocol (`create_container`, `is_supported_container`, `rename_columns`, `hstack`). All four adapters pass a `runtime_checkable` `isinstance(adapter, ContainerAdapter)` check.

### 1.2 Design decisions

- **Dispatch.** An O(1) dict lookup keyed by the exact `type(obj)`, falling back to `(module root, class name)` and then an `is_native` scan.
  - sklearn keys on the module root only (`type(X).__module__.split(".")[0]`), which cannot tell `pl.DataFrame` from `pl.LazyFrame`.
  - Measured cost is 174 to 189 ns per call, against 66 ns for a bare `isinstance`.
  - Dispatch runs once per `items()` or `reconstruct()` call, not once per slice, so it is negligible.
- **`VectorizedDF2` subclasses `VectorizedDF` only to inherit `vectorize_est` unchanged.**
  - `vectorize_est` (about 100 of the 338 LOC of `VectorizedDF`) needed **zero changes**. It only uses `items`, `shape` and keys, and the estimator grid (`forecasters_`) is a pandas object table, not data.
  - The overridden part is 128 LOC: `__init__`, `items`, `reconstruct`, `get_iter_indices`.
- **Keys are opaque per adapter.**
  - pandas keeps a `pd.Index`, so it is bit-identical to today.
  - polars and pyarrow use a list of python scalars or tuples. Single-level keys are unwrapped to scalars so the group names match pandas `groupby(level=0)`.
  - Level names are stored *logically* (`"h0"`, `"time"`, `None`) and mapped to physical names by each adapter. This lets `reconstruct` run in a different backend from the input, dispatched on the **outputs**. That is needed when polars input feeds a pandas-inner forecaster.
- **Polars key columns: the sktime `__index__<name>` convention, not plain key columns.** Justified by test T5:
  - A plain-key polars frame (`h0, time, c0`, Nixtla-style) passes `check_is_scitype` but is **silently misread** as a *Series* (`pl.DataFrame` mtype) with 3 features `['h0','time','c0']`.
  - A plain-key slice `[time, c0]` is read as a 2-variable series with `time` as a feature.
  - A stateless schema passes slices into forecasters that call `check_is_scitype` on them, so the frame must be self-describing. With the `__index__` convention, every slice (`[__index__time, c0]`) is a valid sktime `"pl.DataFrame"` Series.
  - User frames with plain keys are handled at the boundary by a rename. The rename is zero-copy: the value buffer address is unchanged (T5: `rename_shares_buffers=True`).
- **Polars column-vectorized output has no column MultiIndex.** With keys, `concat_cols` yields flat `f"{key}__{col}"` names. This equals pandas `"flat"`, but pandas `"multiindex"` mode is not representable, which is the root of the `droplevel` break in section 4.3.
- **`PolarsLazyAdapter`.** Schema ops use `collect_schema()` (no data). `instance_keys` collects only the key columns (projection). `reconstruct` stays lazy. Slices are either lazy filters (`slice_mode="lazy"`) or come from one collect plus `partition_by` (`"collect_once"`).
- **`PyArrowAdapter`.** pyarrow has no group-iteration API. Groups are found as runs of equal keys with vectorized `pc.not_equal`, and slices are zero-copy `Table.slice`. Non-contiguous input falls back to a stable sort on first-appearance codes (tested on shuffled rows).

## 2. Correctness (`scripts/x4_test_adapter.py`)

| test | result (pandas 2.3.3; identical under 3.0.6) |
|---|---|
| T1 pandas equivalence vs current `VectorizedDF`. Covers Series, Panel, Hierarchical (2 and 3 instance levels); `iterate_as` Series/Panel/Hierarchical; `iterate_cols` True/False; datetime/period/int time; 1 to 3 columns. Compares keys (`assert_index_equal(exact=True)`), every slice (`assert_frame_equal`, `check_freq=True`), freq, schema-only `items()`, `reconstruct` in all 3 `col_multiindex` modes x `convert_back` x 2 per-slice functions (one forces "flat"), and the `vectorize_est` estimator grid | **78/78 identical** |
| T2 polars native (`polars_panel` / `polars_hierarchical`, all `iterate_as` / `iterate_cols`) | **26/26**; **0** pandas<->polars conversions inside prepare + iterate + reconstruct; keys, values, time stamps and level names equal to the pandas reference; identity reconstruct `equals` the input frame |
| T3 polars lazy | LazyFrame Panel/Hierarchical is **not** a recognised sktime mtype (`check_is_scitype` -> invalid). Schema build: 1 collect (key columns only). Reconstruct: 0 collects, result is a `LazyFrame` equal to the input after collect. Slices are equal to eager in both modes |
| T4 pyarrow | slices equal to the polars adapter, reconstruct round trip equal, shuffled (non-contiguous) groups correct |
| T5 key convention | see 1.2 |

## 3. VectorizedDF-level performance (`scripts/x4_bench_vectorize.py --grid dev`)

Univariate Panel, `iterate_as="Series"`, min of 5 repeats (3 for n >= 1000), cur/new
measured back to back in one process. The machine was shared (load average 17 to 40 on 16
cores), so read ratios rather than absolute numbers.

VEC_TABLES

**Reading the tables.**

- **P: the pandas adapter adds no measurable cost** (new/cur 0.85 to 1.00 end to end). Individual stages wobble ±15%, which is load noise in both directions.
- **L: native polars iteration is where the adapter pays off.**
  - `partition_by` against pandas `groupby` + `droplevel` + `_enforce_index_freq`: 2x faster at 10 series, 19x at 1000, 27x at 10000.
  - End to end (prepare + iterate + reconstruct) the gain is only 2.0x to 3.2x, because **both** paths pay the polars mtype check in `prepare`. That check is 1897 ms at 10000x50 against 0.8 ms for pandas: `sktime/datatypes/_adapter/polars.py:is_monotonically_increasing` loops over groups in Python.
  - With the mtype passed through (`prepare new (mtype known)`), prepare drops from 1602 ms to 31 ms. This fix is independent of the adapter design.
- **C: per-slice conversion costs more than convert-once.**
  - Measured as sktime `convert_to` per slice against one whole-frame convert plus pandas slicing: 1.8x to 3.2x.
  - Raw polars `to_pandas` is 6x to 379x more expensive per slice than once on the whole frame.
  - **For pandas-inner forecasters, converting once before vectorization (today) is the right choice.**
- **Z: lazy slices.** Per-slice `filter().collect()` is 5x to 130x slower than one `collect()` + `partition_by`. `pl.collect_all(filters)` is 2x to 35x slower. Parquet scans (predicate pushdown) do not change the picture at this scale. The materialization boundary has to be "collect once at iteration".

### Memory (`scripts/x4_memory.py`, one fresh process per measurement)

A first same-process measurement suggested the native path needs 2x to 4x more peak RSS. That was an allocator-caching artifact: it was measured in one process after the pandas path had run. In fresh processes:

| size | variant | input MB | peak RSS delta MB (run1 / run2) | held MB while all slices alive (run1 / run2) |
|---|---|---|---|---|
| 1000x100 | pd_current | 1.1 | 6.7 / 7.5 | 1.2 / 1.6 |
| 1000x100 | pl_current | 2.3 | 18.1 / 18.8 | 5.2 / 4.5 |
| 1000x100 | pl_native | 2.3 | 14.9 / 13.5 | 9.3 / 6.8 |
| 10000x50 | pd_current | 6.0 | 60.2 / 60.1 | 11.9 / 10.9 |
| 10000x50 | pl_current | 11.4 | 89.1 / 87.0 | 35.5 / 28.3 |
| 10000x50 | pl_native | 11.4 | 83.0 / 83.9 | 18.6 / 20.1 |

Native polars is 5% to 25% lower at peak and holds less while all 10000 slices are alive (about 19 MB vs 28 to 35 MB). At 1000 slices it holds slightly more (polars per-frame overhead). **No material memory win at this scale.** The input frames are 2 to 11 MB, so the gains in section 3 are compute (Python-level per-group overhead), not memory.

## 4. Extension into the forecaster layer (`scripts/x4_forecaster_ext.py`)

These runs use real `BaseForecaster.fit` / `predict`, vectorized over Panel instances, with fh=[1,2,3] and univariate y. Ten scenarios. `ConvCounter` counts pandas<->polars conversion calls and rows. Predictions are compared to A0 (pure pandas) in values, instance keys and time stamps.

| key | input | forecaster (y_inner_mtype) | pipeline |
|---|---|---|---|
| A0 | pandas | NaiveForecaster (pd.Series) | current sktime |
| A1 | polars | NaiveForecaster (pd.Series) | current: convert whole frame once |
| A2 | polars | NaiveForecaster (pd.Series) | adapter-native slices (per-slice conversion) + cutoff bridge |
| A2u | polars | NaiveForecaster (pd.Series) | adapter-native, no cutoff bridge |
| A3 | polars | NaiveForecaster (pd.Series) | **inner-family bridge** -> pandas (== convert once, but via VectorizedDF2) |
| B1 | polars | PolarsNaiveLast (pl.DataFrame) | current |
| B2u | polars | PolarsNaiveLast (pl.DataFrame) | adapter-native, no cutoff bridge |
| B2 | polars | PolarsNaiveLast (pl.DataFrame) | adapter-native + cutoff bridge (== inner-family bridge -> polars) |
| B3 | pandas | PolarsNaiveLast (pl.DataFrame) | current |
| B4 | pandas | PolarsNaiveLast (pl.DataFrame) | **inner-family bridge** -> polars + cutoff bridge |

### 4.1 (b) Does sktime's check/convert machinery already allow a `pl.DataFrame`-inner forecaster? Yes, with one hidden leak

- `PolarsNaiveLast` (declares `y_inner_mtype="pl.DataFrame"`) works end to end with no framework change. On a single polars Series, `fit`/`predict` return a `pl.DataFrame`. On pandas input, output is pandas.
- **But `fit` performs 2 hidden whole-frame polars->pandas conversions** (probe: `fit 2 to_pandas calls, 40 rows` for a 20-row series):
  - `_set_cutoff_from_y` -> `get_cutoff` calls `convert_to(obj, GET_CUTOFF_SUPPORTED_MTYPES)`, and that list is pandas/numpy only;
  - `_update_y_X` calls `_set_cutoff_from_y` **twice** when `remember_data=True` (`_base.py:2169` and `:2180`).
- With `adapter_get_cutoff` (polars `max(time)`, then only the scalar is wrapped in `pd.Index`): **0 conversions**.
- `fh` / `cutoff` stay pandas `Index` objects. These are O(1) or O(h) metadata, not O(N*T) data. The toy builds its output time column from `fh.to_absolute_index(cutoff).to_numpy()`.

### 4.2 Conversion counts and timings

FORECASTER_TABLES

**Reading the forecaster results.**

- **(a) pandas-inner forecaster on polars input.**
  - Native polars slices (A2) turn 1+1 whole-frame conversions (A1) into **N+N per-slice conversions**: 1000+1000 calls at n=1000, moving the same number of rows. Without the cutoff bridge (A2u), 2 more whole-frame conversions appear.
  - FORECASTER_A_READING
  - Per-slice framework overhead dominates the totals. At n=100, the `NaiveForecaster` inner `_predict` and the per-slice `check_is_scitype` / `convert_to` / `get_cutoff` calls dwarf the conversions themselves (profile table).
  - **Native-backend iteration buys nothing for the 95% of forecasters that are pandas-inner, and costs a little.**
- **(b) polars-inner forecaster.**
  - Under today's pipeline, polars input (B1) pays the worst of both worlds:
    - 1 whole pl->pd conversion;
    - N per-slice pd->pl conversions in the clones' `_check_X_y`;
    - 2N hidden per-slice pl->pd conversions in `get_cutoff`;
    - N per-slice pl->pd conversions at predict, plus 1 whole pd->pl.
    - Total **3N+1 conversions in fit and N+1 in predict** (2001+1000 / 1000+1 at n=1000).
  - Pandas input (B3) pays 3N / N.
  - The adapter plus cutoff bridge (B2) gives **0+0**. With pandas input, the inner-family bridge (B4) gives exactly **1 in + 1 out**.
  - FORECASTER_B_READING
- **Multivariate y (column vectorization).**
  - Polars y with 2 columns, through a univariate forecaster under the adapter, fails in `BaseForecaster._vectorize` at `sktime/forecasting/base/_base.py:2466` `y_pred.columns = y_pred.columns.droplevel(1)` (`AttributeError: 'list' object has no attribute 'droplevel'`). This happens for both `NaiveForecaster` and `PolarsNaiveLast`.
  - This is the only *data-API* pandas call in `_vectorize`; the census finds 4 pandas touch points on 2 lines, and the other is `len(self.forecasters_.columns)` on the estimator grid.
  - Fix *(unmeasured/expected)*: a `relabel_cols_to_keys` adapter op, or make "multiindex" mode in `reconstruct` emit key-named columns directly.

### 4.3 Where exactly it stops, and what bridges it

`scripts/x4_loc_census.py` counts regex pandas-API touch points (with line numbers in the JSON):

| location | pandas touch points (lines) | nature |
|---|---|---|
| `BaseForecaster._check_X_y` | 0 (0) | goes entirely through `check_is_scitype` / `convert_to` / `prepare_VectorizedDF`, i.e. the mtype registry. **This is why swapping VectorizedDF needs no change here** |
| `BaseForecaster._vectorize` | 4 (2) | `len(forecasters_.columns)` (estimator grid, fine); `y_pred.columns.droplevel(1)` (**breaks for polars**) |
| `BaseForecaster._set_cutoff` | 3 (2) | `pd.Index` wrapping of the cutoff (metadata, fine) |
| `get_cutoff` (datatypes `_utilities.py`) | converts any non-pandas container to pandas | **hidden whole-frame conversion**, called 2x per `fit` |
| `_base.py` whole file | 41 (28) | regex matches, 4 of them inside string literals (tag defaults `"pd.Series"`, messages; lines 115, 116, 1181, 2838). The real sites: `predict_residuals` (1715-1716); the `get_fitted_params` estimator grid (1831-1833); cutoff (2214-2215); `_vectorize` (2459, 2466); `_predict_moving_cutoff` `y.iloc[...]` (2726); proba output column `pd.MultiIndex` builders (2797, 2805); `_format_moving_cutoff_predictions` `pd.concat` / `.loc` (2836-2868). Only the vectorized fit/predict path was exercised here; update-predict / moving-cutoff / proba paths are *unmeasured* |
| `_vectorize.py` whole file | 62 (46) | all moved behind the adapter in `VectorizedDF2` |

Registry census (`all_estimators("forecaster")`, 160 forecasters):

- `y_inner_mtype` is pandas-only for **152**.
- The **8** that list polars mtypes are all pass-through compositors that accept every mtype (`ConformalIntervals`, `DontUpdate`, `FallbackForecaster`, `ForecastByLevel`, `MultiplexForecaster`, `Permute`, `UpdateEvery`, `UpdateRefitsEvery`).
- 36 declare pandas Panel/Hierarchical inner mtypes (global forecasters that bypass VectorizedDF for Panel input).
- **Zero leaf forecasters compute natively in anything but pandas.**

**Where the adapter approach stops.** It stops at the `y_inner_mtype` boundary of each forecaster. An adapter can move the conversion to that boundary but cannot remove it: the forecaster's `_fit` is written against the pandas API. Making the subclass layer container-independent is a per-forecaster rewrite. sklearn shows how slowly that goes (16/184 estimators after the array-API effort, section 5).

**What bridges it, in order of leverage:**

1. **Inner-family dispatch.**
   - What: choose the VectorizedDF backend from `family(y_inner_mtype)`, not from the input type. The input type decides only one whole-frame `convert_to` at the boundary (or none).
   - Cost: about 8 LOC (`family_of_inner_mtype`) plus a `target=` argument to `prepare`. The tags already exist.
   - Effect: pandas-inner forecasters keep today's cost (A3 vs A1); polars-inner forecasters drop to 0 or 1+1 conversions (B2, B4).
   - This turns "rewrite 100s of forecasters" into **opt-in per forecaster via an existing tag**, with no regression for the rest.
2. **Backend-aware cutoff** (`time_max` op).
   - Removes the 2 hidden whole-frame conversions per `fit` and 2N per-slice conversions (A2u to A2, B2u to B2).
   - About 17 LOC. It also needs the duplicate `_set_cutoff_from_y` call removed.
3. **Replace `columns.droplevel(1)`** with an adapter op (1 call site).
4. **Pass the known mtype from `_check_X_y` into `prepare`** instead of re-running `mtype()`. Saves 1.6 s at 10000x50 polars (section 3). This is an independent bug fix.
5. **Complete the mtype converter graph.**
   - There is no `pl.DataFrame -> np.ndarray` edge, so a numpy-inner forecaster rejects polars input with `NotImplementedError: no conversion defined from type pl.DataFrame to np.ndarray`.
   - Polars->numpy is zero-copy for null-free float columns (probe: `np.shares_memory=True`).
   - `convert_dict` is a module-level dict, so an adapter-backed edge can be registered (`scripts/x4_slice_extract.py` does this at runtime; 0 pandas conversions).
   - A numpy family is a *creative* middle path: many forecasters already compute on `.values`, so declaring `np.ndarray` plus framework-held time metadata would make them container-agnostic without a polars rewrite *(the number of convertible forecasters is unmeasured; the cost side is measured below)*.

SLICE_TABLE

### 4.4 Minimal BaseForecaster change sketch (what the monkeypatches emulate)

The experiments did not edit `sktime/`. They emulate the following *(sketch, not run as a diff)*:

```python
# _check_X_y, vectorized branch
target = family_of_inner_mtype(y_inner_mtype)          # "pandas" for 152/160 today
y_schema, y_inner = prepare_VectorizedDF(
    y, iterate_as=iterate_as, is_scitype=y_scitype, iterate_cols=req_vec_because_cols,
    target=target, X_orig_mtype=y_metadata["mtype"],   # no second mtype() check
)
# _set_cutoff_from_y: backend-aware, called once
self._cutoff = get_adapter(y).time_max_index(y)        # scalar -> pd.Index, no frame conversion
# _vectorize, column-vectorized predict
y_pred = adapter.relabel_cols_to_keys(y_pred)          # instead of y_pred.columns.droplevel(1)
```

Things *not* exercised: exogenous `X` through the adapter (same `prepare` call, so expected to behave like y), `update`, `update_predict`, `predict_interval` / `predict_quantiles` (pandas `MultiIndex` column builders at `_base.py:2797/2805`), and `_predict_moving_cutoff` (`y.iloc`).

## 5. Prior art (`scripts/x4_sklearn_prior_art.py`)

The closest prior art for idea (1) is scikit-learn's `ADAPTERS_MANAGER` (`sklearn/utils/_set_output.py`):

- It has a 4-method `ContainerAdapterProtocol` with `pandas` and `polars` adapters, registered in a dict keyed by module root.
- It is applied at the **boundary** (wrapping transformer outputs), and **84/84** transformers get it.
- scikit-learn's other track rewrites estimator internals against a dispatched namespace (array API, opt-in via the `array_api_support` tag). It reaches **16/184** default-instantiable estimators in 1.9.1.

The same split appears here:

- An adapter at the VectorizedDF/BaseForecaster boundary is universal and cheap.
- Internal container independence is a per-estimator opt-in. sktime already has the opt-in switch: the `y_inner_mtype` tag.

## 6. Marginal cost of adding a backend

Logical LOC (no blanks, comments or docstrings), from `x4_loc_census.py`:

| component | LOC |
|---|---|
| `PandasAdapter` | 67 |
| `PolarsAdapter` (eager) | 100 |
| `PolarsLazyAdapter` | 75 |
| `PyArrowAdapter` (incl. run-detection grouping) | 93 |
| generic, paid once: `AdapterRegistry` 32 + `VectorizedDF2` 128 + `prepare_VectorizedDF2` 45 + `adapter_get_cutoff` 17 + `family_of_inner_mtype` 8 | 230 |
| current `VectorizedDF` for comparison (incl. the reused `vectorize_est`) | 338 (+ 67 prepare/coerce) |
| existing sktime polars **mtype family** (check classes + converters + `_adapter/polars.py`) | 370 |

Two different costs make up a backend:

- **VectorizedDF-only** support (iteration and reconstruction, slices handed to code that knows the backend): about 70 to 100 LOC.
- **Forecaster-consumable** support (slices recognised by `check_is_scitype`, converted to and from other inner types): an mtype family as well. For polars that is about 370 LOC already in sktime.
- pyarrow and LazyFrame have no Panel mtype, so their slices cannot reach a forecaster today (T3).

## 7. Incidental findings (independent of the design choice)

1. `get_cutoff` converts non-pandas frames to pandas wholesale. `_update_y_X` calls it twice per `fit`.
2. The polars Panel/Hierarchical mtype check loops over groups in Python. At 10000x50 it is 1897 ms against 0.8 ms for pandas, about 2400x. It runs twice on the forecaster path: in `check_is_scitype` in `_check_X_y`, and again via `mtype()` in `prepare_VectorizedDF`.
3. There is no `pl.DataFrame -> np.ndarray` converter, so numpy-inner forecasters cannot accept polars.
4. A plain-key polars frame is silently accepted as a multivariate *Series* (keys become features).
5. Current `iterate` cost is dominated by pandas per-group work (`groupby` + `droplevel` + `_enforce_index_freq`): 2.8 s for 10000 groups.

## 8. Caveats

- The machine was heavily shared (load average 17 to 40 on 16 cores) during all runs. Wall-clock per-round spread at n=1000 reached about 2x (raw per-round values are in the JSON). I report min over interleaved, order-rotated rounds, plus CPU time (`process_time`). A3 vs A1 serves as an A/A control: same conversions, both pandas iteration, differing only in the adapter indirection.
- Forecaster runs are capped at n=1000 series for the dev grid (about 7 ms of sktime overhead per series per fit). `--grid large --max-series 50000` runs the full grid serially.
- One toy forecaster (`PolarsNaiveLast`) stands in for "a polars-native forecaster". Real models would have heavier `_fit` / `_predict`, which shrinks the *relative* effect of conversions *(expected)*.
- The polars eager iteration builds N small DataFrames (`partition_by`). It is a per-instance Python loop like today's, just with cheaper slices. Expression-level, in-engine vectorization is out of scope for x4.
- Period time indices: sktime's polars mtypes convert periods to timestamps, so polars results were checked on datetime data only. Integer and period data were covered by the pandas-adapter tests.

## Reproduce

All commands run from the worktree root with
`PP=.:experiments/container_independence:experiments/container_independence/protos` and
`$EXP_PY` (pandas 2.3.3) or `$EXP_PY3` (pandas 3.0.6):

```
PYTHONPATH=$PP $EXP_PY  -W ignore experiments/container_independence/scripts/x4_test_adapter.py      # -> results/x4_test_adapter.json
PYTHONPATH=$PP $EXP_PY3 -W ignore experiments/container_independence/scripts/x4_test_adapter.py      # -> (renamed) results/x4_test_adapter_pd3.json
X4_TMP=<scratch> PYTHONPATH=$PP $EXP_PY -W ignore experiments/container_independence/scripts/x4_bench_vectorize.py --grid dev   # -> results/x4_bench_vectorize_dev.json
PYTHONPATH=$PP $EXP_PY -W ignore experiments/container_independence/scripts/x4_memory.py --grid dev --repeat 2                  # -> results/x4_memory_dev.json
PYTHONPATH=$PP $EXP_PY -W ignore experiments/container_independence/scripts/x4_forecaster_ext.py --grid dev --max-series 1000 --repeat 4 --profile-n 100   # -> results/x4_forecaster_ext_dev.json (+ .log)
PYTHONPATH=$PP $EXP_PY -W ignore experiments/container_independence/scripts/x4_slice_extract.py --grid dev                     # -> results/x4_slice_extract_dev.json
PYTHONPATH=$PP $EXP_PY -W ignore experiments/container_independence/scripts/x4_loc_census.py                                   # -> results/x4_loc_census.json
$EXP_PY -W ignore experiments/container_independence/scripts/x4_sklearn_prior_art.py                                           # -> results/x4_sklearn_prior_art.json
python3 experiments/container_independence/scripts/x4_summarize.py --grid dev                                                  # renders the tables in this report
```

For the serial large phase, use `--grid large` on `x4_bench_vectorize.py`, `x4_memory.py`, `x4_slice_extract.py` and `x4_forecaster_ext.py` (add `--max-series 50000`).

An earlier forecaster run that recorded wall clock only is kept as `results/x4_forecaster_ext_dev_run1_wallonly.json`.
