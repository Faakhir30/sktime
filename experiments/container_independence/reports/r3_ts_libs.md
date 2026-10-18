# r3_ts_libs: how other time-series and forecasting libraries handle data containers

Workstream `r3_ts_libs`. This is the prior-art survey for making sktime forecasters and `VectorizedDF` independent of pandas. Where I could, I checked claims against installed site-packages source, and then ran small experiments that turn each pattern into a number.

---

## 0. TL;DR

1. **No mature forecasting library runs its models on a dataframe API, whether pandas, polars or narwhals.** Every library that supports more than one container does this: it converts at the boundary into a container-free core (numpy arrays, torch tensors, or a numpy-based object), runs the models there, and rebuilds the output at the exit.
   - Nixtla `statsforecast/models.py` is 7,103 lines and 41 model classes. It has **0 pandas or polars references**, and models take `fit(y: np.ndarray, X: np.ndarray)`.
   - All container logic lives in one adapter module, `utilsforecast/processing.py`. It is 1,074 lines and contains 62 `isinstance(pd|pl)` branches.
   - This is how Nixtla avoids the "adapter must reach 100s of forecasters" problem raised against idea (1): **forecasters never see a container.**
2. **Narwhals shows up only at the edges and in frame-to-frame utilities, never in model cores.**
   - darts uses it in `from_dataframe`, `from_group_dataframe` and `to_dataframe`.
   - utilsforecast uses it in `losses` and `evaluation`, and hierarchicalforecast in core, utils and evaluation. mlforecast uses it in a few joins.
   - Every one of these adopters keeps pandas-specific escape hatches. darts has 5 `is_pandas()` fast paths and uses `nw.maybe_get_index`. hierarchicalforecast uses `.to_native().sparse.to_coo()` and `nw.maybe_reset_index`.
3. **Where a library does row-wise frame work and never calls `collect`, the output stays in the input engine.** utilsforecast `mae`, which is pure narwhals, returned the input type for all **6/6** engines tried: pandas, polars, polars-lazy, pyarrow, dask and duckdb. Libraries that do call `collect` handle lazy input in one of these ways:
   - **Reject it:** darts uses `eager_only=True`, and statsforecast `process_df` raises.
   - **Materialize it, then convert back:** pytimetk collects, then calls `.lazy()` on the result. functime calls `.lazy().collect()` at the start of `fit`.
   - Order-dependent time-series operations (diff, lag, shift) **fail on narwhals LazyFrames unless they get an explicit `order_by`**. utilsforecast `mase` on a polars LazyFrame raised `InvalidOperationError: Order-dependent expressions are not supported for use in LazyFrame`.
4. **The numbers that matter for sktime** (dev grid, this machine, under shared load):
   - **Per-series iteration** over 10k series × 50 points:
     - numpy `indptr` slicing (Nixtla `GroupedArray`): 4.7 ms.
     - pandas `groupby(level=0, sort=False)`, the VectorizedDF style: **96× slower** (452 ms).
     - The same with `.droplevel`: **260× slower** (1.22 s).
     - polars `partition_by`: 7.9× slower.
   - **Forecasts that can be written as an expression** (naive) at 10k series:
     - a hand-written polars expression is **10× faster** than statsforecast's numpy hub and **570× faster** than a per-series Python loop;
     - functime, which is polars-native but carries framework overhead, is 8.2× slower than the raw expression.
   - **Forecasts that cannot be written as an expression** (per-series `np.polyfit`): polars UDFs are slower than the numpy hub, by **3.1×** for `map_batches` and **8.0×** for `map_groups`. They are slower even than a pandas `groupby` loop (2.8×). "Polars-native" pays off only for estimators that can be expressed as polars expressions.
   - **Converting to pandas at the boundary** (polars in, pandas inside, polars out), which is what sktime does today: **1.37–1.80× slower** end to end than statsforecast's native polars path.
   - **Collecting lazy input at the boundary** (parquet scan, naive forecast on a subset):
     - collect-first is **2.5–3.8× slower** than a pushed-down lazy plan;
     - collect → `to_pandas` is **9–14× slower**, with 26–72 MB peak versus 0.3–8.5 MB.
   - **Building darts `TimeSeries` objects** costs about **1.0–1.6 ms per series**, about 12.8 s for 10k series. That is 80–4,400× the bare numpy split. Polars input is 0.60–0.82× the time of pandas input.
   - **Delegating per partition** (statsforecast + fugue on dask, a stand-in for distributed back-ends): **203–610× slower** than eager pandas at 100–1,000 series. It does keep the result lazy (it returns a dask DataFrame).
5. **Recommended blend for sktime, based on this prior art:**
   - **(a)** Use a numpy "grouped" hub as the vectorization substrate, meaning keys, `indptr`, values and a time array, built once per call by a small per-backend adapter. This is the Nixtla `process_df` pattern, and it is a concrete form of idea (1) confined to `VectorizedDF` and the base class.
   - **(b)** Let forecasters opt into container-free inner types. An `"np-grouped"` or `"np.ndarray"` `y_inner_mtype` would do. A grep found **no** `np.ndarray` among sktime's 187 `y_inner_mtype` tag lines under `sktime/forecasting`.
   - **(c)** Use narwhals only for frame-level utilities that never collect: metrics, evaluation, joins, future-frame construction.
   - **(d)** Offer optional backend-native fast paths, written as polars or SQL expressions, for the few estimators that are pure aggregations: naive, mean, seasonal naive, drift and window statistics.
   - **(e)** Record the input implementation at the boundary and rebuild outputs in it. Nixtla uses `df_constructor = type(df)`; darts and hierarchicalforecast use the narwhals `backend=`.

---

## 1. Method and environment

Five private venvs, built with `uv`, Python 3.12. The shared envs were not touched.

| venv | contents | purpose |
|---|---|---|
| `scratchpad/venvs/r3` | utilsforecast 0.2.17, statsforecast 2.1.1, mlforecast 1.1.0, coreforecast 0.0.18, fugue 0.9.7, pandas 2.3.3, polars 2.0.0, narwhals 2.27.1, numpy 2.5.4, dask 2026.8.0 (+distributed), duckdb 1.5.6 | Nixtla source + experiments |
| `scratchpad/venvs/r3d` | darts (u8darts) 0.41.0, pandas 3.0.6, polars 2.0.0, narwhals 2.27.1 | darts source + experiment |
| `scratchpad/venvs/r3d48` | darts 0.48.1 (latest on PyPI, 2026-10-16), pandas 3.0.6 | darts re-run on latest |
| `scratchpad/venvs/r3f` | functime 1.0.0 (2026-05-03), statsforecast 2.1.1, polars 2.0.0, pandas 2.3.3 | functime vs hub |
| `scratchpad/venvs/r3src` | `--no-deps` source-only: aeon 1.6.0, pytimetk 2.5.3, skforecast 0.26.0, gluonts 0.17.0, tslearn 0.9.0, prophet 1.5.0, chronos-forecasting 2.3.2, timesfm 3.0.2, hierarchicalforecast 1.5.3, neuralforecast 3.3.0, kats 0.2.0 (temporian 0.9.0 failed to build, docs only) | reading source |

Latest PyPI versions and dates were fetched from `https://pypi.org/pypi/<pkg>/json`:

| package | version | date |
|---|---|---|
| darts | 0.48.1 | 2026-10-16 |
| utilsforecast | 0.2.17 | 2026-10-05 |
| statsforecast | 2.1.1 | 2026-07-16 |
| mlforecast | 1.1.0 | 2026-07-10 |
| functime | 1.0.0 | 2026-05-03 |
| aeon | 1.6.0 | 2026-09-18 |
| pytimetk | 2.5.3 | 2026-10-16 |
| skforecast | 0.26.0 | 2026-10-08 |
| gluonts | 0.17.0 | 2026-07-31 |
| temporian | 0.9.0 | 2024-04-16, inactive |
| kats | 0.2.0 | 2022-03-15, inactive |
| prophet | 1.5.0 | 2026-10-05 |
| tslearn | 0.9.0 | 2026-07-02 |

All benchmark scripts use `common.bench`, which takes the minimum of repeats after a warm-up, and `common.make_panel_fast`. Every script accepts `--grid {dev,large}`. Shared load makes absolute times noisy, so read the ratios, which are measured inside one process.

Commands, all run from the WT root. `$SP` is the scratchpad.

```
# E1 Nixtla hub / boundary / iteration
PYTHONPATH=experiments/container_independence $SP/venvs/r3/bin/python -W ignore experiments/container_independence/scripts/r3_nixtla_boundary.py --grid dev
# E2 Darts narwhals boundary (both versions)
PYTHONPATH=experiments/container_independence $SP/venvs/r3d/bin/python  -W ignore experiments/container_independence/scripts/r3_darts_boundary.py --grid dev            # darts 0.41.0
PYTHONPATH=experiments/container_independence $SP/venvs/r3d48/bin/python -W ignore experiments/container_independence/scripts/r3_darts_boundary.py --grid dev --max-series 1000   # darts 0.48.1
# E3 polars-native vs numpy hub vs loops
PYTHONPATH=experiments/container_independence $SP/venvs/r3f/bin/python -W ignore experiments/container_independence/scripts/r3_polars_native_vs_hub.py --grid dev
# E4 Nixtla partition-and-delegate (fugue) on dask / duckdb
PYTHONPATH=experiments/container_independence $SP/venvs/r3/bin/python -W ignore experiments/container_independence/scripts/r3_nixtla_distributed_probe.py --grid dev
# E5 what collect-at-the-boundary forfeits for lazy input
PYTHONPATH=.:experiments/container_independence $SP/expenv/bin/python -W ignore experiments/container_independence/scripts/r3_lazy_collect_cost.py --grid dev --tmpdir $SP/r3work/data
# E6 narwhals-based utilsforecast losses on 6 engines
PYTHONPATH=experiments/container_independence $SP/venvs/r3/bin/python -W ignore experiments/container_independence/scripts/r3_utilsforecast_losses_probe.py
# E7 functime container probe
PYTHONPATH=experiments/container_independence $SP/venvs/r3f/bin/python -W ignore experiments/container_independence/scripts/r3_functime_probe.py
```

Results are in `results/r3_*.json`:

- `r3_nixtla_boundary_dev.json`
- `r3_darts_boundary_dev_darts0.41.0.json`
- `r3_darts_boundary_dev_darts0.48.1.json`
- `r3_polars_native_vs_hub_dev.json`
- `r3_nixtla_distributed_probe_dev.json`
- `r3_lazy_collect_cost_dev.json`
- `r3_utilsforecast_losses_probe.json`
- `r3_functime_probe.json`

---

## 2. Per-library findings

### 2.1 Nixtla: utilsforecast, statsforecast, mlforecast, hierarchicalforecast

**Internal representation: the numpy grouped array, "hub and spoke".**

- `utilsforecast.processing.process_df(df, id_col, time_col, target_col)` returns `ProcessedDF(uids, last_times, data, indptr, sort_idxs)`, where:
  - `uids` is a pandas or polars Series of unique ids;
  - `data` is a 2-D numpy array (rows × [target + exog]);
  - `indptr` is an `int32` array of series boundaries, built as `np.append(0, sizes.cumsum())`;
  - `sort_idxs` is a permutation, or `None` when the input is already sorted.
- `GroupedArray(data, indptr)` slices `data[indptr[i]:indptr[i+1]]` to get each series. This is a CSR-like layout.
- statsforecast `_StatsForecast._prepare_fit` calls `process_df` and keeps `self.ga = GroupedArray(data, indptr)`. It also stores `self.df_constructor = type(df)` and `last_dates` as a `pd.Index` or `pl.Series`.
- The models in `statsforecast/models.py` take numpy only: `fit(self, y: np.ndarray, X: Optional[np.ndarray])`. Grep found 0 pandas or polars references in that file's 7,103 lines and 41 classes.
- mlforecast uses the same `process_df` → `GroupedArray`. Lag and rolling features are computed in C++ by `coreforecast`, which also works on grouped arrays.

**Boundary conversion.** One adapter module with explicit dispatch: `utilsforecast/processing.py`, 39 exported helpers including `counts_by_id`, `maybe_compute_sort_indices`, `take_rows`, `vertical_concat`, `offset_times`, `time_ranges`, `group_by_agg`, `join`, `cv_times` and `make_future_dataframe`. Each one branches on `isinstance(df, pd.DataFrame)` or the polars types. A grep counted **62** `isinstance(...pd|pl...)` branches in this file, against only 5 in `statsforecast/core.py`. The type unions live in `utilsforecast/compat.py` (`DataFrame = Union[pd.DataFrame, pl_DataFrame]`, `DFType = TypeVar(..., pd.DataFrame, polars.DataFrame)`).

**Narwhals adoption: partial and recent.**

- utilsforecast lists `narwhals>=2.0` as a hard dependency. `losses.py` (62 `nw.` calls, 0 isinstance branches), `evaluation.py` (30 `nw.` calls), `rectify.py` and `model_selection.py` use `narwhals.stable.v2`.
- The losses moved to narwhals in PR #195 ("Adds Narwhals losses", merged 2025-10-30). The release notes list it as v0.2.15, "Improved DataFrame Compatibility with Narwhals".
- `processing.py`, the hub-building module, is still hand-written pandas/polars dispatch with 0 `nw.` calls.
- mlforecast mixes the two: 21 isinstance branches and 12 `nw.` calls in `core.py`, plus 11 and 5 in `forecast.py`.
- hierarchicalforecast uses narwhals in `core.py`, `utils.py` and `evaluation.py`. Its reconciliation `methods.py` is pure numpy and scipy, with 0 `nw.` calls. It still drops to pandas for sparse S matrices (`S_nw...to_native().sparse.to_coo()`) and calls `nw.maybe_reset_index`.

**Supported containers and output type.** Measured in E1 and E4:

| input | statsforecast `forecast` output |
|---|---|
| pandas | `pandas.DataFrame` |
| polars | `polars.DataFrame` (rebuilt with `df_constructor`) |
| polars LazyFrame | `NotImplementedError: no registered dataset conversion for LazyFrame` (fugue path) |
| pyarrow Table | `SchemaError: large_string type is not supported` (fugue path) |
| dask DataFrame | `dask DataFrame` (lazy; computes to pandas) via fugue partition-and-delegate |
| duckdb relation | `ValueError: TIMESTAMP_NS is not supported` (fugue/duckdb) |

`process_df` raises `ValueError: df must be either pandas or polars dataframe` on a LazyFrame.

The narwhals-based `utilsforecast.losses.mae` returns the input type for all six engines (E6). `mase` on a LazyFrame fails: order-dependent expressions need `order_by`.

**Distributed path: partition and delegate.** `statsforecast/distributed/fugue.py` partitions the input with `partition={"by": id_col}` and calls functions typed `df: pd.DataFrame` on each partition. It declares the output schema up front (`_get_output_schema`). In E4, on dask with 4 partitions and the threads scheduler, this was **203× slower** than the eager pandas path at 100×100 (1.35 s vs 6.7 ms) and **610× slower** at 1000×100 (15.3 s vs 25 ms). That is roughly 13–15 ms per series. Each id group becomes its own pandas call, so small series are dominated by per-group overhead. The path is designed for cluster-scale data, and this benchmark does not test that regime.

**Notable issues.**

- statsforecast 2.1.1 still pins `pandas<3.0.0`, and mlforecast 1.1.0 pins `pandas<3.0`. utilsforecast only added pandas 3 support in v0.2.17 (2026-10-05, PR #259). Coupling to pandas internals blocks major-version upgrades.
- utilsforecast's `polars` extra pins `polars<=1.31`. The core paths used here still worked with polars 2.0.0.
- Nixtla does not expose lazy or streaming semantics. For polars inputs it uses only eager polars.

**Measured costs** (E1, `r3_nixtla_boundary_dev.json`; 10k × 50 unless stated):

- `process_df`: pandas 32.3 ms, polars 15.3 ms. Polars is **1.5–2.1× faster** across the grid.
- `StatsForecast(Naive).forecast`, end to end:

  | input | time |
  |---|---|
  | pandas | 123 ms |
  | polars, native | 97 ms |
  | polars → pandas → forecast → polars | 133 ms |

  The via-pandas route costs **1.37× (10k) to 1.80× (10 series)** relative to the native polars path.
- One pass over all series, the "vectorization loop" cost:

  | method | time | relative to indptr |
  |---|---|---|
  | numpy indptr slices | 4.7 ms | 1× |
  | polars `partition_by` | 37 ms | 7.9× |
  | pandas `groupby(level=0, sort=False)` | 452 ms | 96× |
  | the same + `droplevel(0)` (closest to VectorizedDF) | 1,220 ms | 260× |

  The ratios grow with the number of series: at 10 series they are 51× and 91×.

### 2.2 Darts

**Internal representation.** Since darts **0.36.0** (2025-06-29, PR #2807, "Migrated the TimeSeries backend from xarray to numpy" for "huge performance boost"), a `TimeSeries` holds:

- `_values`, a 3-D numpy array (time, component, sample);
- `_time_index`, a pandas `DatetimeIndex` or `RangeIndex`;
- `static_covariates`, a pandas DataFrame;
- `metadata`.

Verified in 0.41.0: `vars(ts)` shows `_time_index` (`pandas.RangeIndex`) and `_values` (`numpy.ndarray`, shape `(5,1,1)`). Plotting stopped going through xarray in 0.39.0 (#2932). Pandas is therefore still structurally required for time.

**Boundary, via narwhals.**

| version | change |
|---|---|
| 0.34.0 (#2661, merged 2025-02-28) | `from_dataframe` and `from_series` accept any narwhals-compatible frame; `to_dataframe` and `to_series` take `backend=`; `pd_dataframe()` deprecated, then removed in 0.35 |
| 0.38.0 (#2766, merged 2025-09-02) | `from_group_dataframe` moves to narwhals; a pandas fast path is added: "set index for pandas df for performance boost" |
| 0.41.0 | pandas 3 support; `darts.to_group_dataframe()` added |
| 0.45.0 | `pl.Date` time columns accepted |

The PR #2661 thread reports `from_narwhals_dataframe` at 9.86 s against 10.97 s for the old `from_dataframe` on their benchmark, about 10% faster. The bottleneck named there was `series_df.to_numpy()`.

Source, from darts 0.41/0.48 `timeseries.py`:

- `nw.from_native(df, eager_only=True, pass_through=False)` is used everywhere.
- `nw.maybe_get_index(df)` reads the pandas index when `time_col=None`.
- `df.implementation.is_pandas()` is used in five places. For pandas input, `from_group_dataframe` converts `time_col` to the index and goes native.
- `to_dataframe` builds `nw.from_dict(data, backend=backend).to_native()`. The default `backend` is pandas.

**Supported containers and output.** Measured in E2 probes on both 0.41.0 and 0.48.1:

- Accepted: polars and pyarrow.
- Rejected: polars LazyFrame (`TypeError: Cannot only use eager_only ... with polars.LazyFrame`).
- A pandas MultiIndex frame without `time_col` → `ValueError`. Darts has no MultiIndex or panel container; panels are `list[TimeSeries]`.
- `PeriodIndex` → `TypeError: Cannot interpret 'period[M]'`.
- After polars input, `time_index` is a pandas `DatetimeIndex` and `static_covariates` is a pandas DataFrame.
- **The output does not follow the input:** `to_dataframe()` returns pandas unless `backend="polars"` is passed explicitly.

**Measured costs** (E2):

| darts | grid point | from_group (pandas) | from_group (polars) | µs/series (polars) | polars/pandas |
|---|---|---|---|---|---|
| 0.41.0 | 10×100 | 20.9 ms | 13.4 ms | 1343 | 0.64 |
| 0.41.0 | 100×100 | 166 ms | 107 ms | 1069 | 0.64 |
| 0.41.0 | 1000×100 | 1.79 s | 1.24 s | 1243 | 0.69 |
| 0.41.0 | 10000×50 | 19.5 s | 12.8 s | 1276 | 0.65 |
| 0.48.1 | 10×100 | 19.1 ms | 12.4 ms | 1240 | 0.65 |
| 0.48.1 | 100×100 | 157 ms | 94.8 ms | 948 | 0.60 |
| 0.48.1 | 1000×100 | 1.97 s | 1.61 s | 1605 | 0.82 |

- Relative to a bare numpy split of the same values, the polars path is **80× (10 series) to 4,428× (10k series)** slower. The cost is one Python object per group.
- `to_dataframe` + concat for all series: the polars backend is 1.5–2.3× faster than the pandas backend. At 10k series that is 1.32 s vs 2.96 s on 0.41.
- Peak RSS for the 0.41 run was 506 MB. That is the 10k-series point, about 50 KB per `TimeSeries` object. The script now takes a `--max-series` cap.

**Lesson.** Narwhals at the boundary works for darts because the core is not a dataframe; it is numpy plus a pandas time index. The darts maintainers still added pandas fast paths, and the object-per-series model, not the conversion, dominates cost at scale.

### 2.3 functime: polars-native core

- **Status.** v1.0.0 on PyPI (2026-05-03). The repo (`functime-org/functime`) is not archived and has about 1.2k stars, with no GitHub releases section. Dependencies are `polars>=1.0`, sklearn and flaml. pandas is only an optional extra for plot and llm.
- **Representation.** A long polars frame whose first two columns are entity and time, followed by the target. Internally it uses `pl.LazyFrame`.
- **Estimators.** Naive and seasonal naive are pure expressions (`group_by(entity).agg(pl.col(target).last()/tail(sp))`). Global ML models build lag features with polars expressions and then call `to_numpy` for sklearn, LightGBM and similar. Feature extraction uses a `.ts` namespace backed by a Rust extension (`_functime_rust.abi3.so`).
- **Laziness is mostly nominal.** `Forecaster.fit` starts with `y = self._set_string_cache(y.lazy().collect()).lazy()`, which materializes the whole input, and cutoffs are always collected. The source carries comments such as `# BUG: Cannot run the following in lazy streaming mode? Causes internal error: entered unreachable code` and `# Cutoffs cannot be lazy`.
- **Containers (E7).**

  | input | result |
  |---|---|
  | polars eager | `pl.DataFrame` |
  | polars lazy | `pl.DataFrame`, eager, not lazy |
  | pandas | `AttributeError: 'DataFrame' object has no attribute 'lazy'` |
  | pyarrow | same `AttributeError` |

- **Measured (E3, W1 naive, 10k×50):**

  | implementation | time | vs raw polars expression |
  |---|---|---|
  | raw polars expression | 15.8 ms | 1× |
  | functime | 129 ms | 8.2× (framework overhead: string cache, collect/lazy round trips, cutoffs) |
  | statsforecast hub | 161 ms | 10.2× |
  | per-series polars loop | 9.0 s | 570× |

  At 10 series the hub is faster than the raw expression (0.55×), because polars has fixed per-query overhead.

### 2.4 aeon (sktime fork)

- **v1.0.0** removed the sktime-derived forecasting module (#2037), the forecasting metrics (#2247) and the `datatypes` module. The changelog says: "Our focus now is on writing efficient code based on array-based bespoke implementations."
- **Collections:** `aeon/utils/data_types.py` (1.6.0) lists `COLLECTIONS_DATA_TYPES` = numpy3D `(n_cases, n_channels, n_timepoints)`, np-list (a list of 2-D arrays for unequal length), df-list, numpy2D, pd-wide and pd-multiindex. numpy3D and np-list are the inner types that estimators are written against. `pd_multiindex_hier` is the only hierarchical type.
- **Series:** `VALID_SERIES_INNER_TYPES = ["np.ndarray", "pd.DataFrame"]`, with an `axis` convention: `(n_channels, n_timepoints)` by default.
- **New forecasting module:** `aeon/forecasting/base.py` sets `_tags["y_inner_type"] = "np.ndarray"` and uses an integer `horizon`.
  - `predict` and `forecast` are annotated to return `float`.
  - `direct_forecast` and `iterative_forecast` return `np.ndarray` of shape `(prediction_horizon,)`.
  - No time index, cutoff or ForecastingHorizon object exists. `NaiveForecaster._predict` is `y.squeeze()[-1]`.
- **Consequence:**
  - Gained: container independence (any array-like converts to numpy), simpler estimators, and numba-friendly code.
  - Lost: time and index semantics such as dates, PeriodIndex, irregular indices, hierarchical aggregation and exogenous alignment by index. These are left to the user.
  - aeon now targets single-series numeric forecasting rather than panel or hierarchical frame workflows. This last point is my inference from the API; no maintainer statement on rationale beyond the changelog was found.

### 2.5 pytimetk: per-function engine dispatch (the closest analogue to idea (1))

- `engine: {"auto","pandas","polars","cudf"}` on about 40 public functions; grep found `normalize_engine(` in 40 files.
- The source has **55 `_*_pandas`, 42 `_*_polars` and 6 `_*_cudf` private implementations**, one per backend per function.
- `convert_to_engine` returns a `FrameConversion(data, original_kind, group_columns, row_id_column, pandas_cache, ...)`, and `restore_output_type` converts back:
  - polars LazyFrame input is collected (`collect_lazyframe`, optionally on GPU), then `.lazy()` is called on the result;
  - polars results returned to pandas callers go through `to_pandas()`.
- The project claims 3×–3,500× speedups for the polars engine. Its PyPI table states `augment_rolling()` at 10–3,500× and `summarize_by_time()` at 13.4×. These are unmeasured here; the PyPI page gives no methodology.
- **Lesson.** Per-backend duplication scales with the number of functions times the number of backends. Here it covers about 40 feature-engineering functions, not hundreds of stateful estimators. "Lazy in, lazy out" is cosmetic: no pushdown survives the collect.

### 2.6 Brief: other libraries

| library | internal representation | boundary | containers | output follows input? | notes |
|---|---|---|---|---|---|
| **GluonTS 0.17** | dict entries `{"start": pd.Period, "target": np.ndarray, ...}`, iterated lazily | `PandasDataset` (long or wide pandas, `.from_long_dataframe`) → entries; `ArrowFile`/`ParquetFile` datasets via pyarrow | pandas, numpy, arrow/parquet files, jsonl | no (forecast objects) | no polars or narwhals imports; arrow is used for storage, not computation |
| **skforecast 0.26** | `dict[str, pd.Series]` + `dict[str, pd.Index]` (`check_preprocess_series` converts long MultiIndex frames to a dict) | pandas only; errors say "must be a pandas Series with a DatetimeIndex or a RangeIndex" | pandas | pandas | 38 files import pandas, 0 polars or narwhals |
| **tslearn 0.9** | numpy `(n_ts, sz, d)`, NaN-padded for unequal length (`to_time_series_dataset`) | array-like → numpy | numpy, lists, (pandas via numpy) | numpy | collections only, no time index |
| **Prophet 1.5** | pandas df with `ds`, `y` → numpy arrays for Stan | `fit(self, df: pd.DataFrame)` | pandas | pandas | single series; 41 `'ds'` literals in `forecaster.py` |
| **Kats 0.2.0** (2022) | `TimeSeriesData` wrapping pandas `time` + `value` | pandas | pandas | pandas | inactive since 2022; 107 files import pandas |
| **temporian 0.9.0** (2024) | `EventSet`: per-index-key numpy arrays (float64 timestamps + feature arrays), C++ ops; keys are ordinary features | `from_pandas`/`to_pandas`, CSV (per docs) | pandas, CSV | n/a | inactive since 2024-04; "index" means group keys, a MultiIndex-free design |
| **Chronos 2.3** | torch tensors (`context: torch.Tensor`) | `predict_df(df: pd.DataFrame, id_column="item_id", timestamp_column="timestamp")` → pandas; core takes tensors or lists of arrays | pandas (df API), tensors, arrays | pandas | 4 files import pandas |
| **TimesFM 3.0** | numpy, torch or JAX arrays; `forecast(horizon, inputs: Sequence[np.ndarray]) -> list[np.ndarray]` | none in core (0 pandas imports) | numpy | numpy | array-only core |
| **neuralforecast 3.3** | torch tensors from `TimeSeriesDataset(temporal, indptr)`, built by `ufp.process_df` (`tsdataset.py:362`) | pandas or polars via utilsforecast | pandas, polars | yes, through a `df_constructor` (`core.py:212-215`) | 6 pandas files, 2 polars, 0 narwhals: the same P1 hub as statsforecast |

Chronos internals were read only briefly; that row is based on imports and signatures, not a full trace. For neuralforecast, the `process_df` and `df_constructor` call sites were checked in source.

---

## 3. Summary table (main libraries)

| library | core representation | boundary strategy | containers in | output in input type | lazy input |
|---|---|---|---|---|---|
| statsforecast / mlforecast | numpy `GroupedArray(data, indptr)` | hand-written pandas/polars dispatch in `utilsforecast.processing`; fugue for distributed | pandas, polars (+ dask/spark/ray via fugue) | yes for pandas, polars and dask (`df_constructor`, fugue schema) | rejected (polars lazy); dask kept lazy via fugue |
| utilsforecast losses / evaluation | the frame itself (narwhals expressions) | narwhals `from_native` → `to_native` | pandas, polars, polars-lazy, pyarrow, dask, duckdb (6/6 for `mae`) | yes, 6/6 | kept lazy, but order-dependent ops fail without `order_by` |
| hierarchicalforecast | numpy / scipy.sparse S and Y matrices | narwhals at the edges, pandas escape hatches | pandas, polars | yes (`backend=` from the native namespace) | not tested |
| darts | numpy 3-D + pandas Index | narwhals `eager_only=True`, pandas fast paths | pandas, polars, pyarrow | **no**: default pandas, needs `backend=` | rejected |
| functime | polars (LazyFrame API, collected at fit) | none (polars only) | polars eager or lazy | polars eager | collected at fit |
| pytimetk | per-function pandas, polars and cudf implementations | `convert_to_engine` + `restore_output_type` | pandas, polars, cudf, groupbys | yes | collected, then `.lazy()` re-wrap |
| aeon | numpy (3-D collections, 2-D series) | `convert_collection` / `_convert_y` to numpy | numpy, pandas, lists | no (numpy or float) | n/a |
| sktime (today, for reference) | pandas (`y_inner_mtype`: 0 of 187 tag lines name `np.ndarray`) | `convert_to` → pandas; VectorizedDF on pandas `groupby(level)` | pandas + polars/dask/... mtypes converted to pandas | via `convert_to` back | collected |

---

## 4. Cross-cutting patterns

**P1. Hub and spoke: a numpy grouped array plus a per-backend adapter (Nixtla, neuralforecast, temporian, tslearn, aeon).**

- *Shape:* one adapter turns any frame into (keys, `indptr`, values, times, sort permutation). Estimators see numpy only. The output is rebuilt with `df_constructor = type(df)`.
- *Pros:* models are container-free, so the adapter is written once, not once per estimator. Iteration is cheap: indptr slicing was 96× faster than pandas `groupby(level)` and 260× faster with `droplevel` (E1). Adding a backend means adding about 40 small helpers, not changing models.
- *Cons:* the full data is materialized once, with no lazy pushdown past the boundary. The time index becomes a raw array, so frequency and period semantics need separate handling. There is still a Python loop over series, unless the model is vectorized or written in C++/numba as in coreforecast.
- *Fit for sktime:* this is the most direct template for `VectorizedDF` and for an opt-in container-free `y_inner_mtype`.

**P2. Narwhals at the boundary, numpy or arrays in the core (darts, hierarchicalforecast, utilsforecast losses).**

- *Shape:* `nw.from_native` → select, join or aggregate → `to_native` or `to_numpy`. The core is not narwhals.
- *Pros:* one code path for frame-level work. When nothing is collected, the type is preserved across 6/6 engines (E6).
- *Cons:* adopters keep pandas escape hatches. darts has 5 `is_pandas()` paths plus `maybe_get_index`; hierarchicalforecast has `.to_native().sparse` and `maybe_reset_index`. Lazy order-dependent operations need explicit `order_by` (E6 `mase`). darts sidesteps lazy frames entirely with `eager_only=True`.
- *Fit for sktime:* metrics, evaluation, splitters, and the future-frame and output assembly. It does not fit inside 100s of forecasters.

**P3. A polars-native core (functime, and parts of pytimetk).**

- *Pros:* for estimators that can be written as expressions, it is by far the fastest. A raw polars naive was 10× faster than the numpy hub and 570× faster than a per-series loop at 10k series (E3 W1).
- *Cons:* most statistical forecasters (ARIMA, ETS, Theta, anything with a numpy or scipy fit) cannot be written as polars expressions. Run as group UDFs, they were **3.1× (`map_batches`) and 8.0× (`map_groups`) slower than the numpy hub**, and slower than a pandas groupby loop (2.8×) (E3 W2). functime ended up polars-only (pandas input raises), and still collects lazy input at fit.
- *Fit for sktime:* optional fast paths for a handful of aggregation-type estimators, not a general substrate.

**P4. Per-function backend duplication with convert-and-restore (pytimetk; this is idea (1) applied per function).**

- *Pros:* each backend gets its own idiom, and the claimed speedups are large (unmeasured here).
- *Cons:* about 100 private implementations for about 40 functions: 55 pandas, 42 polars and 6 cudf implementations, so coverage across backends is already uneven. This does not scale to hundreds of stateful estimators. Lazy input is collected.
- *Fit for sktime:* only at the `VectorizedDF` / base-class layer, with very few hooks, so that it becomes P1.

**P5. Partition and delegate for distributed engines (statsforecast + fugue; mlforecast for dask, spark and ray).**

- *Pros:* lazy and distributed engines stay native (a dask result was returned), and the output schema is declared up front.
- *Cons:* per-group delegation overhead was **203–610×** slower than eager pandas at 100–1,000 series (E4). duckdb and pyarrow paths failed on dtypes (`TIMESTAMP_NS`, `large_string`).
- *Fit for sktime:* this is the right shape for dask, spark or ray, a `map_partitions` over coarse id-partitions plus a declared output schema. Partitions must be coarse (many series per call), not one series per call.

**P6. An array-only core and an array-only API (aeon forecasting, TimesFM, tslearn, the Chronos core).**

- *Pros:* trivially independent of containers.
- *Cons:* time-index semantics are pushed onto users. aeon dropped its frame-based forecasting module entirely.
- *Fit for sktime:* the right place for the inner layer, not for the user-facing API.

**P7. One blessed container (skforecast, Prophet, Kats).** These are pandas-only. They are simple, but heavily exposed to pandas majors: statsforecast and mlforecast still pin `pandas<3`.

---

## 5. Implications for sktime ideas (1) and (2), and the C1/C2 claims

- **Idea (1), dispatcher/adapter.** The prior art says to confine the adapter to the boundary and the vectorization layer, and make the core it feeds container-free. Nixtla is the existence proof: one adapter module of 62 branches serves 41 numpy-only models, and polars runs about 1.4–1.8× faster end to end than converting to pandas. Its weakness is laziness. Nixtla rejects LazyFrames, and P5 (fugue) is slow per group.
- **Idea (2), narwhals.** No surveyed library re-implements forecasters in narwhals. The adopters use it for edge and frame utilities and keep pandas-specific paths.
  - *C1 (collect semantics):* prior art avoids it either by never collecting (utilsforecast losses kept lazy for polars-lazy, dask and duckdb, 6/6 types preserved) or by refusing lazy input (darts). Whether `collect(backend=...)` can force the original backend was not tested here. That belongs to the narwhals workstreams.
  - *C2 (no MultiIndex):* consistent with prior art. darts reads the pandas index with `nw.maybe_get_index`, raises on MultiIndex input, and adds `is_pandas()` fast paths. hierarchicalforecast calls `nw.maybe_reset_index`. Nobody keeps a MultiIndex through narwhals; everybody moves to long format with key columns (or keys + indptr).
- **The creative synthesis from prior art:** a **grouped-array hub** (P1) for `VectorizedDF`, plus **narwhals at the edges** (P2) for frame-level utilities and output reconstruction, plus **expression fast paths** (P3) for a few estimators, plus **coarse partition-and-delegate** (P5) for dask, spark and duckdb.
  - Forecasters opt in to an `"np-grouped"`-style inner mtype. Legacy pandas forecasters keep working through the existing pandas path.
  - Time metadata (frequency, Period vs Datetime, cutoff) is kept in the stateless schema next to the arrays, rather than in a pandas index.

---

## 6. Caveats

- All timings come from the dev grid on a shared 16-core machine with about 14 concurrent agents. Absolute numbers are noisy. Ratios are measured inside one process, using the minimum of 3–5 repeats.
- **E1:** statsforecast's `n_jobs=1`. The "via pandas" variant includes `to_pandas` and `from_pandas`. The iteration benchmark reads one value per series, so it isolates loop overhead. Real models add their own per-series cost, and that cost shrinks the ratio (see E3 W2, where per-series `polyfit` brings the pandas-groupby penalty down to 2.8×).
- **E2:** darts 0.41.0 and 0.48.1 both ran on pandas 3.0.6. The 0.48.1 re-run stopped at 1,000 series to stay under the memory budget. At 10k series, the 0.41 run peaked at 506 MB RSS.
- **E3:** peak RSS was 544 MB at the 10k point, slightly over the ~450 MB guideline, because of the per-series loop variants. `--max-series` now caps it. functime was run with `freq="1d"` on polars 2.0.0. The functime overhead may differ on the polars 1.x versions it was developed against.
- **E4:**
  - dask ran on the threads scheduler with task shuffle and 4 partitions. Dask's P2P shuffle requires a distributed client and failed without one.
  - This is a local, small-data stand-in, and it does not represent cluster-scale behaviour.
  - The duckdb failure (`TIMESTAMP_NS`) may be specific to these fugue_duckdb/duckdb versions.
- **E5:**
  - The parquet file is 14.2 MB (10k series × 50 points × 4 columns), so the peak-RSS deltas are small and noisy.
  - A polars 2.0 performance cliff was found: `is_in(<python list of ≤100 strings>)` inside a scan took about 110 ms, against about 7 ms for 101 items or for an imploded Series. The script uses `pl.Series(ids).implode()`.
- Not measured: the pytimetk speedup claims, temporian (failed to build), the narwhals `collect(backend=...)` behaviour, and neuralforecast and Chronos internals beyond their signatures.
- PR and changelog facts were read with WebFetch from GitHub pages, so they are summaries of those pages. No PR benchmark images were readable, and darts #2807 gave no numeric speedups in its text.

## 7. Sources

- utilsforecast source (`processing.py`, `compat.py`, `grouped_array.py`, `losses.py`, `evaluation.py`), statsforecast (`core.py`, `models.py`, `distributed/fugue.py`) and mlforecast (`core.py`, `forecast.py`), in `scratchpad/venvs/r3/lib/python3.12/site-packages/`.
- darts `timeseries.py` and `utils/utils.py` (0.41.0, 0.48.1); functime 1.0.0 (`base/forecaster.py`, `forecasting/naive.py`, `snaive.py`, `_ar.py`); aeon 1.6.0 (`utils/data_types.py`, `forecasting/base.py`, `_naive.py`); pytimetk 2.5.3 (`utils/dataframe_ops.py`, `feature_engineering/lags.py`); skforecast, gluonts, tslearn, prophet, chronos, timesfm, kats, hierarchicalforecast and neuralforecast, in `scratchpad/venvs/r3src/...`.
- [darts CHANGELOG](https://raw.githubusercontent.com/unit8co/darts/master/CHANGELOG.md); darts PRs [#2807](https://github.com/unit8co/darts/pull/2807), [#2661](https://github.com/unit8co/darts/pull/2661) and [#2766](https://github.com/unit8co/darts/pull/2766).
- [utilsforecast releases](https://github.com/Nixtla/utilsforecast/releases); [utilsforecast PR #195](https://github.com/Nixtla/utilsforecast/pull/195).
- [statsforecast polars guide](https://nixtlaverse.nixtla.io/statsforecast/docs/getting-started/getting_started_complete_polars.html).
- [functime repo](https://github.com/functime-org/functime); [aeon v1.0 changelog](https://www.aeon-toolkit.org/en/stable/changelogs/v1.0.html); [narwhals README (Used by)](https://github.com/narwhals-dev/narwhals); [pytimetk PyPI](https://pypi.org/project/pytimetk/); [temporian user guide](https://temporian.readthedocs.io/en/stable/user_guide/).
