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
   - The polars adapter iterates and reconstructs with zero pandas conversions (26/26 configurations match the pandas reference). It iterates 1.4x to 32x faster than today's convert-then-`groupby` path, and iterate plus reconstruct is 4.2x to 25x faster (cloud re-run; earlier run 2x to 27x / 6x to 22x).
   - A new backend costs 67 to 100 logical lines of code (LOC) for the adapter. Its pandas-free sktime mtype family (check plus convert) costs more, about 370 LOC for polars.
2. **It stops at the forecaster's `y_inner_mtype`.**
   - 152 of 160 registered forecasters compute only in pandas. The other 8 are pass-through compositors, so 0 leaf forecasters compute in polars.
   - For a pandas-inner forecaster, iterating in the input's backend ("native" dispatch) does not remove the conversion. It splits it into N per-slice conversions:
     - 2N conversion calls instead of 2.
     - 1.5x to 4.3x more conversion time in isolation (cloud re-run; earlier 1.8x to 3.2x).
     - end-to-end fit/predict with NaiveForecaster: 0.98x to 1.31x of convert-once (A2 vs A1, cloud re-run), never faster.
   - **Dispatching on the input type is the wrong key.**
3. **The bridge is to dispatch on the forecaster's inner-mtype family instead.** VectorizedDF should iterate in the backend of `y_inner_mtype` (a tag that already exists): convert the whole frame once to that family, then use that family's adapter.
   - For pandas-inner forecasters this is today's behaviour (1+1 conversions).
   - For a polars-inner forecaster it gives **0 conversions** end to end: polars in, polars out, Panel included. Measured: fit 0.36x to 0.43x and predict 0.49x to 0.56x of today's time (B2 vs B1, n=10..1000, cloud re-run).
   - Three small framework fixes are also needed:
     - a backend-aware cutoff (`get_cutoff` silently converts whole non-pandas frames to pandas, twice per `fit`);
     - an adapter op in place of `y_pred.columns.droplevel(1)` (`_base.py:2466`, which crashes for polars);
     - passing the already-known mtype instead of re-checking it (the polars mtype check costs 0.68 s at 10000x50 on the cloud re-run, 1.9 s earlier, versus 0.6 to 0.8 ms for pandas).
4. **The adapter approach cannot carry laziness into per-instance forecasters.**
   - Schema construction needs only the key columns, and reconstruct stays lazy.
   - But per-instance forecasters need materialized slices. Per-slice lazy filters are 3x to 77x (cloud re-run; 5x to 130x earlier) slower than one `collect()` followed by `partition_by`.
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

#### P: pandas input, current VectorizedDF vs VectorizedDF2(PandasAdapter) [ms]

| n_series x T | prepare cur | prepare new | iterate cur | iterate new | reconstruct cur | reconstruct new | total cur | total new | new/cur total |
|---|---|---|---|---|---|---|---|---|---|
| 10x100 | 1.5 | 1.2 | 2.0 | 2.1 | 0.8 | 1.3 | 4.4 | 4.4 | 1.02 |
| 100x100 | 2.1 | 1.8 | 10.4 | 12.5 | 2.3 | 2.0 | 15.8 | 14.8 | 0.94 |
| 1000x100 | 8.2 | 10.0 | 141.3 | 126.5 | 16.3 | 15.5 | 141.6 | 150.0 | 1.06 |
| 10000x50 | 34.4 | 35.5 | 1179.5 | 1224.8 | 142.9 | 139.6 | 1721.4 | 1528.6 | 0.89 |

#### L: polars input, current (convert to pandas, iterate, reconstruct, convert back) vs VectorizedDF2(PolarsAdapter) native [ms]

| n_series x T | mtype check pl | mtype check pd | prepare cur | prepare new | prepare new (mtype known) | iterate cur | iterate new | iter speedup | reconstruct cur (+convert back) | reconstruct new | iter+recon cur (incl. convert) | iter+recon new | speedup | total cur | total new | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10x100 | 4.3 | 0.6 | 8.5 | 4.6 | 2.6 | 2.2 | 1.6 | 1.4x | 3.2 | 1.1 | 10.9 | 2.6 | 4.2x | 13.2 | 5.3 | 2.5x |
| 100x100 | 10.0 | 0.6 | 35.8 | 12.9 | 2.7 | 10.6 | 1.8 | 6.0x | 5.5 | 1.8 | 35.0 | 3.1 | 11.4x | 44.1 | 16.8 | 2.6x |
| 1000x100 | 104.3 | 0.7 | 232.3 | 138.9 | 6.0 | 102.8 | 8.3 | 12.3x | 27.5 | 7.4 | 240.6 | 17.6 | 13.6x | 369.8 | 118.5 | 3.1x |
| 10000x50 | 682.9 | 0.6 | 1442.9 | 696.7 | 16.0 | 1324.1 | 41.2 | 32.1x | 207.6 | 59.6 | 2732.4 | 110.7 | 24.7x | 3230.1 | 992.3 | 3.3x |

#### C: where to convert for a pd.Series-inner forecaster (polars input) [ms]

| n_series x T | convert once + pandas slices + slice->pd.Series | polars slices + convert_to(slice, pd.Series) | per-slice / once | raw whole.to_pandas() | raw sum(slice.to_pandas()) | raw ratio |
|---|---|---|---|---|---|---|
| 10x100 | 11.6 | 17.2 | 1.48x | 1.9 | 3.3 | 2x |
| 100x100 | 51.1 | 174.8 | 3.42x | 1.9 | 35.4 | 19x |
| 1000x100 | 506.6 | 1913.4 | 3.78x | 4.8 | 579.1 | 120x |
| 10000x50 | 5059.9 | 21794 | 4.31x | 25.9 | 7022.8 | 271x |

#### Z: materialising all slices of a polars LazyFrame [ms]

| n_series x T | source | schema build (keys only) | collect once + partition_by | per-slice filter+collect | collect_all(filters) | per-slice / once | collect_all / once |
|---|---|---|---|---|---|---|---|
| 10x100 | mem | 1.1 | 2.0 | 6.3 | 3.1 | 3x | 2x |
| 10x100 | parquet | 2.1 | 2.0 | 12.0 | 7.1 | 6x | 4x |
| 100x100 | mem | 1.5 | 2.7 | 34.7 | 13.4 | 13x | 5x |
| 100x100 | parquet | 2.3 | 4.8 | 170.1 | 36.7 | 36x | 8x |
| 1000x100 | mem | 4.4 | 10.4 | 803.6 | 188.0 | 77x | 18x |
| 1000x100 | parquet | 7.7 | 18.8 | 1336.1 | 295.5 | 71x | 16x |

#### D: dispatch overhead (ns per get_adapter call)

| pandas | polars | polars_lazy | pyarrow | isinstance_pd_DataFrame |
|---|---|---|---|---|
| 101 | 107 | 105 | 102 | 39 |

#### M: peak RSS delta, full polars pipeline at 10000x50 (polars frame 11.4 MB)

current (2 runs): 16.2 / 2.9 MB; native: 70.5 / 54.3 MB

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

| scenario | conv fit / pred (n=1000) | out type | preds == A0 |
|---|---|---|---|
| A0: pd in | Naive(pd.Series) | current | 0+0 calls (0 rows) / 0+0 calls (0 rows) | pandas.DataFrame | True |
| A1: pl in | Naive(pd.Series) | current convert-once | 1+0 calls (100000 rows) / 0+1 calls (3000 rows) | polars.DataFrame | True |
| A2: pl in | Naive(pd.Series) | adapter-native per-slice convert + cutoff bridge | 1000+0 calls (100000 rows) / 0+1000 calls (3000 rows) | polars.DataFrame | True |
| A2u: pl in | Naive(pd.Series) | adapter-native, no cutoff bridge | 1002+0 calls (300000 rows) / 0+1000 calls (3000 rows) | polars.DataFrame | True |
| A3: pl in | Naive(pd.Series) | inner-family bridge (->pandas) | 1+0 calls (100000 rows) / 0+1 calls (3000 rows) | polars.DataFrame | True |
| B1: pl in | PolarsNaiveLast(pl.DataFrame) | current | 2001+1000 calls (400000 rows) / 1000+1 calls (6000 rows) | polars.DataFrame | True |
| B2u: pl in | PolarsNaiveLast(pl.DataFrame) | adapter-native, no cutoff bridge | 2002+0 calls (400000 rows) / 0+0 calls (0 rows) | polars.DataFrame | True |
| B2: pl in | PolarsNaiveLast(pl.DataFrame) | adapter-native + cutoff bridge | 0+0 calls (0 rows) / 0+0 calls (0 rows) | polars.DataFrame | True |
| B3: pd in | PolarsNaiveLast(pl.DataFrame) | current | 2000+1000 calls (300000 rows) / 1000+0 calls (3000 rows) | pandas.DataFrame | True |
| B4: pd in | PolarsNaiveLast(pl.DataFrame) | inner-family bridge (->polars) + cutoff bridge | 0+1 calls (100000 rows) / 1+0 calls (3000 rows) | pandas.DataFrame | True |

Timings (s, min over interleaved rounds); ratios vs A1 (polars input, current) for A*, vs B1 for B2*, vs B3 for B4:

| scenario | fit n=10 | pred n=10 | fit n=100 | pred n=100 | fit n=1000 | pred n=1000 |
|---|---|---|---|---|---|---|
| A0 | 0.052 (0.82) | 0.080 (0.97) | 0.345 (0.76) | 0.895 (1.07) | 4.154 (0.90) | 9.699 (0.99) |
| A1 | 0.064 (1.00) | 0.082 (1.00) | 0.452 (1.00) | 0.840 (1.00) | 4.635 (1.00) | 9.844 (1.00) |
| A2 | 0.068 (1.07) | 0.088 (1.07) | 0.443 (0.98) | 0.822 (0.98) | 6.090 (1.31) | 10.172 (1.03) |
| A2u | 0.078 (1.23) | 0.097 (1.19) | 0.501 (1.11) | 0.927 (1.10) | 5.435 (1.17) | 10.763 (1.09) |
| A3 | 0.055 (0.87) | 0.083 (1.01) | 0.432 (0.95) | 0.870 (1.04) | 4.336 (0.94) | 9.233 (0.94) |
| B1 | 0.120 (1.00) | 0.047 (1.00) | 1.086 (1.00) | 0.383 (1.00) | 11.107 (1.00) | 5.071 (1.00) |
| B2u | 0.102 (0.85) | 0.024 (0.52) | 0.782 (0.72) | 0.222 (0.58) | 9.471 (0.85) | 2.460 (0.49) |
| B2 | 0.052 (0.43) | 0.023 (0.49) | 0.395 (0.36) | 0.215 (0.56) | 4.072 (0.37) | 2.804 (0.55) |
| B3 | 0.102 (1.00) | 0.039 (1.00) | 0.968 (1.00) | 0.419 (1.00) | 10.879 (1.00) | 4.131 (1.00) |
| B4 | 0.056 (0.55) | 0.028 (0.72) | 0.322 (0.33) | 0.209 (0.50) | 3.702 (0.34) | 2.752 (0.67) |

CPU time (s, process_time, min over rounds) -- less sensitive to machine contention; same ratio bases:

| scenario | fit n=10 | pred n=10 | fit n=100 | pred n=100 | fit n=1000 | pred n=1000 |
|---|---|---|---|---|---|---|
| A0 | 0.052 (0.82) | 0.080 (0.98) | 0.343 (0.78) | 0.891 (1.08) | 4.094 (0.89) | 9.618 (0.99) |
| A1 | 0.064 (1.00) | 0.081 (1.00) | 0.441 (1.00) | 0.829 (1.00) | 4.615 (1.00) | 9.759 (1.00) |
| A2 | 0.069 (1.09) | 0.087 (1.07) | 0.446 (1.01) | 0.824 (0.99) | 5.948 (1.29) | 10.123 (1.04) |
| A2u | 0.081 (1.27) | 0.098 (1.21) | 0.496 (1.12) | 0.924 (1.12) | 5.368 (1.16) | 10.654 (1.09) |
| A3 | 0.056 (0.88) | 0.081 (1.00) | 0.433 (0.98) | 0.868 (1.05) | 4.333 (0.94) | 9.102 (0.93) |
| B1 | 0.120 (1.00) | 0.047 (1.00) | 1.065 (1.00) | 0.379 (1.00) | 10.840 (1.00) | 4.992 (1.00) |
| B2u | 0.106 (0.88) | 0.025 (0.54) | 0.776 (0.73) | 0.222 (0.59) | 9.270 (0.86) | 2.456 (0.49) |
| B2 | 0.054 (0.45) | 0.023 (0.50) | 0.395 (0.37) | 0.216 (0.57) | 4.083 (0.38) | 2.795 (0.56) |
| B3 | 0.102 (1.00) | 0.038 (1.00) | 0.941 (1.00) | 0.410 (1.00) | 10.491 (1.00) | 4.036 (1.00) |
| B4 | 0.057 (0.56) | 0.028 (0.74) | 0.326 (0.35) | 0.210 (0.51) | 3.706 (0.35) | 2.746 (0.68) |

A3 vs A1 is an A/A control (identical conversions, both iterate in pandas): its ratio is the noise floor.

Multivariate polars y (2 columns, univariate forecaster => column vectorization):

- `mv_current_naive`: ok, cols=['__index__h0', '__index__time', 'c0', 'c1']
- `mv_native_naive`: FAILS `AttributeError: 'list' object has no attribute 'droplevel'` at `/home/user/sktime/sktime/forecasting/base/_base.py:2466 `y_pred.columns = y_pred.columns.droplevel(1)``
- `mv_native_toy`: FAILS `AttributeError: 'list' object has no attribute 'droplevel'` at `/home/user/sktime/sktime/forecasting/base/_base.py:2466 `y_pred.columns = y_pred.columns.droplevel(1)``

cProfile cumulative seconds, one fit+predict, n=100x100:

| scenario | total_s | convert_to | check_is_scitype | get_cutoff | adapter_get_cutoff | inner _fit | inner _predict | pl.to_pandas | pl.from_pandas | items(VectorizedDF) | items(VectorizedDF2) | reconstruct(VectorizedDF) | reconstruct(VectorizedDF2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 3.401 | 0.779 | 0.169 | 0.446 | 0.000 | 0.003 | 2.224 | 0.000 | 0.000 | 0.116 | 0.000 | 0.011 | 0.000 |
| A1 | 2.804 | 0.662 | 0.188 | 0.422 | 0.000 | 0.003 | 1.646 | 0.003 | 0.001 | 0.105 | 0.000 | 0.008 | 0.000 |
| A2 | 3.009 | 1.105 | 0.114 | 0.396 | 0.405 | 0.003 | 1.616 | 0.141 | 0.081 | 0.000 | 0.008 | 0.000 | 0.003 |
| A2u | 3.885 | 1.341 | 0.115 | 0.459 | 0.000 | 0.003 | 2.346 | 0.134 | 0.107 | 0.000 | 0.006 | 0.000 | 0.003 |
| A3 | 3.458 | 0.831 | 0.242 | 0.491 | 0.495 | 0.005 | 2.076 | 0.003 | 0.001 | 0.001 | 0.124 | 0.000 | 0.012 |
| B1 | 3.038 | 1.653 | 0.193 | 1.150 | 0.000 | 0.041 | 0.297 | 0.432 | 0.099 | 0.108 | 0.000 | 0.009 | 0.000 |
| B2u | 2.131 | 0.977 | 0.119 | 1.076 | 0.000 | 0.039 | 0.295 | 0.257 | 0.000 | 0.000 | 0.008 | 0.000 | 0.003 |
| B2 | 1.252 | 0.132 | 0.105 | 0.000 | 0.245 | 0.026 | 0.310 | 0.000 | 0.000 | 0.000 | 0.006 | 0.000 | 0.003 |
| B3 | 2.811 | 1.543 | 0.169 | 1.157 | 0.000 | 0.040 | 0.248 | 0.401 | 0.094 | 0.107 | 0.000 | 0.011 | 0.000 |
| B4 | 1.386 | 0.170 | 0.101 | 0.000 | 0.288 | 0.031 | 0.327 | 0.003 | 0.003 | 0.000 | 0.007 | 0.000 | 0.003 |

**Reading the forecaster results.**

- **(a) pandas-inner forecaster on polars input.**
  - Native polars slices (A2) turn 1+1 whole-frame conversions (A1) into **N+N per-slice conversions**: 1000+1000 calls at n=1000, moving the same number of rows. Without the cutoff bridge (A2u), 2 more whole-frame conversions appear.
  - Timing (cloud re-run, wall): A2/A1 fit 1.07 / 0.98 / 1.31 at n=10/100/1000, predict 1.07 / 0.98 / 1.03; A2u is worse (fit 1.11 to 1.23). The A/A control A3/A1 is 0.87 to 1.04, so A2's ~1.3x fit at n=1000 is above noise and the rest is within it.
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
  - Timing (cloud re-run, wall): B2/B1 fit 0.43 / 0.36 / 0.37, predict 0.49 / 0.56 / 0.55; B4/B3 fit 0.55 / 0.33 / 0.34, predict 0.72 / 0.50 / 0.67. Removing the cutoff leak alone (B2u -> B2) halves fit time at n=100..1000.
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

Slice extraction (`x4_slice_extract.py --grid dev`, cloud re-run; ms). `np_per_slice` = polars slices -> zero-copy numpy; `np_once` = current whole pl->pd then pandas slices `.to_numpy()`:

| n_series x T | pd_once | pd_per_slice | np_once | np_per_slice | pl_per_slice | np_reshape | pd_per_slice/pd_once | np_once/np_per_slice |
|---|---|---|---|---|---|---|---|---|
| 10x100 | 12.4 | 20.5 | 9.9 | 1.1 | 1.2 | 75.5 us | 1.66x | 8.8x |
| 100x100 | 52.0 | 203.4 | 30.5 | 2.5 | 2.4 | 72.5 us | 3.91x | 12.1x |
| 1000x100 | 503.5 | 2139.8 | 220.6 | 9.4 | 7.4 | 71.8 us | 4.25x | 23.4x |
| 10000x50 | 6135.1 | 21547 | 2427.8 | 69.8 | 52.0 | 94.9 us | 3.51x | 34.8x |

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

## Cloud re-run (2026-10-18, 4 cores, load avg ~2 to 4)

Re-ran `x4_test_adapter.py` (pandas 2.3.3), `x4_bench_vectorize.py --grid dev`, `x4_forecaster_ext.py --grid dev --max-series 1000 --repeat 3 --profile-n 100`, `x4_slice_extract.py --grid dev`, then `x4_summarize.py` to render the tables above (they are the cloud numbers). Not re-run: pandas 3.0.6 test, `x4_memory.py`, `x4_loc_census.py`, `x4_sklearn_prior_art.py` (LOC/census/prior-art numbers are from the earlier run).

Correctness unchanged: 10/10 tests pass; T1 78/78, T2 26/26 with 0 conversions; T3/T4/T5 as before.

Conversion counts at n=1000 are identical to the earlier run (A1 1+1, A2 1000+1000, A2u 1002+1000, B1 2001+1000 fit / 1000+1 predict, B3 2000+1000 / 1000+0, B2 0+0, B4 0+1 / 1+0), and all predictions equal A0. The `droplevel` failure at `_base.py:2466` reproduces.

Numbers that moved (the TL;DR ranges should be read with these):

| quantity | earlier (local, loaded 16-core) | cloud re-run |
|---|---|---|
| pandas adapter new/cur total | 0.85 to 1.00 | 0.89 to 1.06 (noise) |
| polars native iterate speedup | 2x to 27x | 1.4x to 32x |
| polars iterate+reconstruct speedup | 6x to 22x | 4.2x to 24.7x |
| polars end-to-end (incl. prepare) speedup | 2.0x to 3.2x | 2.5x to 3.3x |
| polars mtype check at 10000x50 | 1897 ms (vs 0.8 ms pandas) | 683 ms (vs 0.6 ms pandas), ~1100x |
| prepare at 10000x50, mtype re-checked -> passed through | 1602 -> 31 ms | 1443 -> 16 ms (cur path) / 697 -> 16 ms (new path) |
| per-slice `convert_to` vs convert-once (C) | 1.8x to 3.2x | 1.5x to 4.3x |
| raw per-slice `to_pandas` vs once | 6x to 379x | 2x to 271x |
| lazy per-slice filter+collect vs collect once | 5x to 130x | 3x to 77x (n<=1000) |
| `collect_all(filters)` vs collect once | 2x to 35x | 2x to 18x |
| dispatch per call | 174 to 189 ns (bare isinstance 66) | 101 to 107 ns (bare isinstance 39) |

The same-process RSS line "M" in the bench table (native 54 to 70 MB vs current 3 to 16 MB) is the allocator-caching artifact explained in the Memory section; the fresh-process `x4_memory.py` numbers were not re-run.
Conclusions are unchanged.
