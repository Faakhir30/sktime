# r4_dispatch_arch: prior art for multi-backend dataframe architectures

Workstream key: `r4_dispatch_arch`. Scope: how other libraries outside time series let one
API run on pandas, polars, Arrow, Dask, DuckDB, Spark and similar backends, and what sktime's
vectorization layer (`VectorizedDF` / `BaseForecaster._vectorize`) can borrow.
The libraries covered are pandera, Fugue, Ibis, Modin, cudf.pandas, pyspark.pandas,
Great Expectations, Hamilton, Kedro, the Arrow PyCapsule interface, the data-apis DataFrame
standard, and the engine-native "apply per group" mechanisms.

Every number below comes from a script in `scripts/r4_*.py` that was re-run in this (cloud)
session. Raw logs are in `results/r4_log_*.txt` and JSON is in `results/r4_*.json`. Statements
taken from docs or source reading are marked **[doc]** or **[src]**. Statements that are
expected but were not measured are marked **[unmeasured]**.

## 0. Environment and commands

- Shared env, pandas 2.3.3: `EXP_PY=.../scratchpad/expenv/bin/python`. Versions: polars 2.0.0,
  narwhals 2.27.1, pyarrow 26, dask 2026.8, duckdb 1.5.6, ibis 12.
- Shared env, pandas 3.0.6: `EXP_PY3=.../scratchpad/expenv_pd3/bin/python`.
- Private venv `.../scratchpad/venvs/r4` with Python 3.12: fugue 0.9.7, pandera 0.34.1,
  modin 0.37.1 with dask, pandas 2.3.3, polars 2.0.0, narwhals 2.27.1, duckdb 1.5.6, dask 2026.8.
- Machine: 4 cores, about 2 agents running at once, so timings are noisy. Wherever possible
  each case is the minimum of 3 repeats, run interleaved in the same process (`common.bench`),
  and results are reported as ratios against a baseline measured in that same process.

Commands, all run from the worktree root:

```
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/r4_api_surface.py
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/r4_arrow_capsule.py --grid dev
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/r4_arrow_capsule.py --grid dev
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/r4_group_apply.py --grid dev
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/r4_canonical_arrow.py --grid dev
PYTHONPATH=experiments/container_independence $R4_PY -W ignore experiments/container_independence/scripts/r4_fugue_transform.py --grid dev
PYTHONPATH=experiments/container_independence $R4_PY -W ignore experiments/container_independence/scripts/r4_pandera_backends.py --grid dev
MODIN_ENGINE=dask MODIN_CPUS=2 PYTHONPATH=experiments/container_independence $R4_PY -W ignore experiments/container_independence/scripts/r4_modin_fidelity.py --grid dev
MODIN_ENGINE=dask MODIN_CPUS=2 $R4_PY -W ignore experiments/container_independence/scripts/r4_modin_diag.py
$EXP_PY -W ignore experiments/container_independence/scripts/r4_dask_order.py
```

`R4_PY` is `.../scratchpad/venvs/r4/bin/python`. Every benchmark script accepts `--grid {dev,large}`.
DEV_GRID (n_series × n_time) is 10×100, 100×100, 1000×100 and 10000×50, with 1 float column.

---

## 1. pandera: a backend registry behind one schema API

### Mechanism [src: pandera 0.34.1 `api/base/schema.py`, `api/base/checks.py`]

- Each API class (`DataFrameSchema`, `Column`, `Check`, `Parser`) has a class-level
  `BACKEND_REGISTRY: dict[(api_cls, frame_type) -> backend_cls]`.
  - `register_backend(type_, backend)` adds an entry to it.
  - `get_backend(check_obj)` walks `inspect.getmro(type(check_obj))` and returns the first
    registered backend. This is plain type-keyed dispatch, open to extension.
  - Registration is lazy: `register_default_backends(type)` runs on first use, which keeps
    imports light.
- The user-facing API is **not fully shared**. `pandera.pandas.DataFrameSchema` and
  `pandera.polars.DataFrameSchema` are different classes. The probe recorded
  `schema_classes_distinct = True`. Dtypes are also per-backend through "engines":
  `engines/pandas_engine.py`, `polars_engine.py` and the others.
  - What is shared: the abstract base classes, the `Check` objects with their built-in names
    (`in_range` and so on), the error-reporting model, and the class-based `DataFrameModel`
    pattern.
- Polars support was added as a **separate backend** (an independent implementation), not as an
  adapter over pandas. Lines of code per backend in 0.34.1:

  | backend | api LOC | backend LOC | dtype engine LOC |
  |---|---|---|---|
  | pandas | 1808 | 4633 | 2337 |
  | polars | 846 | 1968 | 1189 |
  | ibis | 528 | 1589 | 532 |
  | pyspark | 1336 | 2002 | 580 |
  | **narwhals** | 171 | 3801 | 400 |
  | shared base (api/base + api/dataframe, backends/base) | 3933 | 404 | — |

  Command: `wc -l` over the installed package.
- **New in 2025–26: a narwhals backend.** pandera now ships an opt-in narwhals backend,
  enabled with `PANDERA_USE_NARWHALS_BACKEND=True` or `pandera.config.set_config(use_narwhals_backend=True)`.
  When enabled, the `(DataFrameSchema, pl.DataFrame | pl.LazyFrame | ibis.Table | pd.DataFrame)`
  registry entries are replaced with one narwhals implementation. Our probe confirms that all
  four frame types then dispatch to `pandera.backends.narwhals.container.DataFrameSchemaBackend`.
  - **Key lesson for C2 (MultiIndex):** the narwhals backend does *not* model the pandas
    index. Its source says, verbatim: *"Index/MultiIndex is the one pandas concept Narwhals
    cannot express, so index coercion is delegated to the native pandas Index backend. Done on
    the native frame before wrapping so the coerced index propagates through Narwhals (which
    preserves the pandas index) to the output."* (`backends/narwhals/container.py` around line 441).
  - The pandas register docstring adds that the native pandas backends "are always registered
    — they serve SeriesSchema, Index, MultiIndex ... as well as the pandas-like frame types
    (dask, modin, geopandas, pyspark.pandas)".
  - So pandera uses a **hybrid**: the column logic goes through narwhals, and the index logic
    goes through a native pandas escape hatch.

### Measurements (`r4_pandera_backends.py --grid dev`)

The schema is the same on every backend: c0 float with `in_range`, plus h0 and time. It is
written as a MultiIndex schema for the "pandas MI" case and as columns otherwise. Times are
min-of-3 wall times per `validate`.

| n_series×n_time | native pd MI | nw pd MI | native pd flat | nw pd flat | native pl eager | nw pl eager | native pl lazy+collect | nw pl lazy+collect |
|---|---|---|---|---|---|---|---|---|
| 10×100 | 3.97 ms | 9.16 | 4.79 | 12.67 | 4.36 | 8.92 | 1.20 | 3.14 |
| 100×100 | 4.53 | 9.79 | 5.67 | 23.12 | 5.67 | 8.66 | 1.22 | 3.14 |
| 1000×100 | 5.87 | 10.88 | 19.76 | 23.24 | 12.31 | 8.34 | 3.74 | 2.85 |
| 10000×50 | 14.54 | 17.34 | 80.74 | 46.17 | 7.64 | 9.67 | 1.43 | 3.31 |

- On small data the narwhals backend costs about 2–2.6× the native backend. This is a fixed
  overhead of roughly 5 ms per validate.
- At 500k rows the narwhals backend is as fast as or faster than native: 0.57× on pandas flat,
  1.2× on pandas MI, 1.27× on polars eager.
- The MultiIndex schema validated correctly under both modes: the index type stayed
  `MultiIndex`, the names `['h0','time']` were kept, and an out-of-range value was caught.

### Lessons

1. A type-keyed registry with MRO lookup is cheap. One `get_backend` call measured
   0.77–1.1 µs (it is a dict lookup), and adding a backend means adding a package rather than
   editing the core.
2. A full per-backend reimplementation (pandas → polars → ibis → pyspark) cost pandera roughly
   2–5 kLOC per backend. pandera then **converged onto narwhals** to remove that duplication.
3. Even the narwhals convergence kept a native-pandas path for Index/MultiIndex. This is
   direct prior art that C2 is real, and that the accepted workaround is a hybrid (narwhals
   for columns, native code for the index), not reset_index everywhere.
4. sktime is in the same position: a hierarchical index plus many estimators that consume it.
   pandera's lesson argues for a **narrow registry at the container boundary**, keeping the
   "index" concept in sktime's own schema object. `VectorizedDF` already stores its schema
   separately, which is exactly what pandera does.

---

## 2. Fugue: `transform()` as an engine-agnostic per-partition map

Fugue is the closest analogue to sktime vectorization: it runs a Python function once per
group of `partition={"by": keys, "presort": time}` on any engine.

### Mechanism [src: fugue 0.9.7; doc: fugue-tutorials "transform"]

- **The function's type annotations choose the per-group container.** The supported annotations
  are `pd.DataFrame`, `List[Dict]`, `pa.Table`, `pl.DataFrame` and iterables of these.
  Fugue builds a converter from the annotation.
- **An output schema is mandatory.** The doc says "Explicit `schema` is a hard requirement in
  distributed computing frameworks". Our probe confirmed that omitting it raises
  `FugueInterfacelessError(... "schema can't be None")`.
- How each engine runs the per-group function:
  - **Native engine:** `df.as_pandas()`, then `pl_utils.safe_groupby_apply(pdf, keys, _map)`.
    Each group is sorted by the presort keys, `reset_index(drop=True)`, wrapped in
    `PandasDataFrame`, converted to the annotated type, and the result is converted back to
    pandas (`native_execution_engine.py:103-190`).
  - **DuckDB engine:** `create_default_map_engine()` returns a `PandasMapEngine(self._native_engine)`.
    So **Python per-group functions on DuckDB run through pandas**.
  - **Dask engine:** `df.native.groupby(keys, sort=False, group_keys=False).apply(_map, meta=output_dtypes)`.
    `_map` again resets the index, sorts by presort and wraps the group in a fugue
    `PandasDataFrame` (`fugue_dask/execution_engine.py`).
  - Spark and Ray use the same pattern with pandas UDFs (Arrow batches) [doc, not run here].
  - **In short, Fugue's portability for Python UDFs is "everything becomes pandas per group".**
    Engine-native execution exists only for SQL and relational ops.

### Fidelity (`r4_fugue_transform.py`, from `results/r4_fugue_transform_dev.json` → `fidelity`)

| probe | result |
|---|---|
| polars input, default engine | returns **pandas.DataFrame** (the input format is not respected) |
| arrow input, default engine | returns pandas.DataFrame |
| any input, engine=dask | returns a dask DataFrame; with `as_local=True` → pandas |
| polars/arrow input, engine=duckdb | **error** `ValueError('TIMESTAMP_NS is not supported')` |
| pandas MultiIndex input (sktime pd-multiindex) | **error** `pandas like datafame must have default index, but got MultiIndex` |
| period-typed time column | **error** `SchemaError(... pandas.period ... is not supported)` |
| frame seen by the per-group fn | `RangeIndex`, keys as columns, time as `datetime64[ns]` |

This is the same behaviour that was disputed in C1 and C2. A generic engine layer (a) returns
its own preferred format rather than the input's, (b) has no index and requires keys as
columns, and (c) does not support pandas-only dtypes such as Period.

### Overhead (min of 3; cases projected above 30 s were skipped)

| n_series×n_time | pandas groupby loop (base) | polars partition_by | fugue native fn(pd) | fugue native fn(pl), pl in | fugue native fn(pa.Table) | fugue native fn(List[Dict]) | fugue duckdb fn(pd) | fugue dask(2 thr) fn(pd) |
|---|---|---|---|---|---|---|---|---|
| 10×100 | 2.35 ms (1.0×) | 0.66× | 7.4× | 13.8× | 11.9× | 16.0× | 20.5× | 99× |
| 100×100 | 6.43 ms | 0.52× | 21.2× | 39.3× | 30.2× | 52.5× | 27.0× | 71× |
| 1000×100 | 56.3 ms | 0.54× | 22.1× | 46.5× | 31.6× | 60.0× | 22.7× | 55.6× |
| 10000×50 | 709 ms | 0.49× | 21.0× (14.9 s) | 38.6× | 27.9× | skipped | 22.0× | skipped |

- Fugue's fixed cost is **about 1.2–1.5 ms per group** on the pandas path, against about
  50–70 µs for a bare pandas groupby loop: 21–22× the bare loop at ≥100 groups.
- Asking for polars per group costs about 2.6 ms per group (39–47×), because every group is
  converted pandas → polars.
- Fugue is designed for coarse partitions (thousands of rows each) on clusters. It is not
  built for 10k small series.

**Lesson:** the *pattern* fits sktime well: per-partition UDF, schema declared up front, type
annotations choosing the per-group container. Fugue's *implementation* does not.
- Fugue normalises to pandas per group, which is today's sktime behaviour plus 20× overhead.
- Taking Fugue as a dependency would not deliver native polars or DuckDB execution of
  forecasters.
- Worth copying:
  1. Declare the output schema up front. This lets lazy engines plan without executing.
  2. Pick the per-group container from a declaration (sktime already has `y_inner_mtype`,
     which plays the role of the annotation).
  3. Use an `as_local` switch to choose whether results stay on the engine.

---

## 3. Ibis: expression IR compiled to backends

- [doc: ibis-project.org/why] Ibis "defines a Python dataframe API that executes on any query
  engine". Most backends compile through SQLGlot to SQL; polars is a dataframe-generating
  backend. The docs list "more than 20 backends", say "portability with Ibis isn't perfect",
  and give Substrait IR as only a long-term goal. Ibis has no index, so ordering must be
  written explicitly with `order_by`.
- Measured, as the expression-native forecaster baseline in `r4_group_apply.py`. The "forecaster"
  is mean + last per series, so no Python runs per group:

  | n_series×n_time | pandas loop (base) | ibis(duckdb) expr | ibis(polars) expr | polars expr (eager) | duckdb SQL | pandas groupby.agg |
  |---|---|---|---|---|---|---|
  | 10×100 | 1.62 ms | 3.2× | 1.15× | 0.50× | 2.2× | 0.82× |
  | 1000×100 | 48.2 ms | 0.244× | 0.099× | 0.058× | 0.223× | 0.173× |
  | 10000×50 | 474 ms | 0.091× | 0.066× | 0.044× | 0.095× | 0.087× |

  - At 10k series an expression-compiled forecaster is 11–23× faster than the per-instance
    Python loop.
  - Ibis adds about 1.5× over raw polars (31 ms vs 21 ms) and about 0.96× over raw DuckDB SQL
    (43 ms vs 45 ms).
  - For tiny inputs Ibis's fixed compile cost dominates: 5 ms vs 1.6 ms.
- Ibis Python UDF path: `collect(order_by)` followed by a `@ibis.udf.scalar.python` function on
  the list. It measured 1.02× the pandas loop at 1000 series and 0.65× at 10k. So even
  "Python per group" can run inside the engine at a cost comparable to the pandas loop,
  because the engine does the grouping.
- Version-skew bug seen: `ibis 12.0.0 + duckdb 1.5.6` → `con.create_table(...)` raises
  `ParserException`. The scripts work around it by registering through the raw duckdb
  connection. This is a reminder that IR/compiler layers depend on version pairs.
- **When Ibis is the right tool:** the estimator logic is relational (aggregations, window
  functions, lags, rolling means, seasonal-naive, exponential smoothing written as recursive
  windows only with difficulty), the data lives in a database or warehouse, and pushdown or
  laziness matters.
- **When it is not:** sktime-style fit/predict of arbitrary Python estimators (statsmodels,
  scipy optimisation). There, Ibis reduces to "fetch the data and loop in Python".
- Conclusion: an IR is a candidate **for a small set of reduction-style forecasters**
  (Naive, Mean, Drift, seasonal naive, window summaries). It is not a framework-wide substrate.
  Narwhals offers a lighter variant of the same idea (expressions that run natively on polars
  and DuckDB lazy frames) without the SQL compiler.

---

## 4. Keep the pandas API, swap the engine: Modin, cudf.pandas, pyspark.pandas

### Modin (measured: `r4_modin_fidelity.py`, MODIN_ENGINE=dask, 2 CPUs)

These are the operations `VectorizedDF` actually uses, compared against pandas.

Single-shot times, which include some warm-up, so read these as orders of magnitude:

| op (panel 1000×100) | modin/pandas time ratio | equal incl. names | "defaulting to pandas" warnings |
|---|---|---|---|
| `groupby(level, sort=False)` iterate + droplevel (our iteration) | **721×** (109 s vs 152 ms) | yes* | 1 |
| `concat(keys=...)` (our reconstruct) | **484×** (94 s vs 194 ms) | yes* | 1 |
| `groupby(level).mean()` | 11.5× | yes | 0 |
| `groupby(level)[c].apply(f)` | 1.4× | yes* | 0 |
| `reset_index().set_index(names)` | 8.1× | yes | 0 |
| `loc[instance key]` | 58× | yes | 0 |
| `xs(instance key)` | 9.8× | yes | 0 |
| `index.droplevel(-1).unique()` | 1.0× | yes | 0 |

\* "Equal" holds only because `make_panel_fast` keys are zero-padded, so sorted order equals
appearance order. On `make_panel(20, 30)`, whose keys `s0..s19` are not zero-padded, the
same ops returned **False**. `r4_modin_diag.py` shows the causes:

- Modin's `groupby(level=0, sort=False)` yields groups in **sorted** order
  (`['s0','s1','s10','s11']` instead of pandas' `['s0','s1','s2','s3']`), so `sort=False` is
  not honoured.
- `groupby(level)[c].apply` **loses the Series name** (`None` instead of `'c0'`).
- `isinstance(modin_df, pd.DataFrame)` is **False**, so every `isinstance(X, pd.DataFrame)` in
  sktime's mtype checks would reject it.
- MultiIndex construction kept names and levels, and none of the lookups defaulted.

Lesson: an emulated full API is never fully faithful. The parts that break are exactly the
MultiIndex groupby iteration and concat that `VectorizedDF` relies on, and Modin's per-op
latency (tens of ms) is 2–3 orders of magnitude too high for a per-instance Python loop.

### cudf.pandas [doc: docs.nvidia.com/cudf/latest/cudf_pandas/faq]

- A module proxy (`ModuleAccelerator(fast=cudf, slow=pandas)`) with automatic fallback on
  every unsupported call, at the cost of GPU↔CPU copies. The docs note that "pandas value
  mutability is not always guaranteed" because of those copies.
- Proxies pass `isinstance` checks against pandas types, which Modin's objects do not. That
  matters for libraries that type-check, like sktime.
- The FAQ page does not discuss MultiIndex. **[unmeasured]** No GPU was available here.

### pyspark.pandas [doc: Spark "best practices"]

- It attaches a "default index" when none is known. The `distributed-sequence` index "does not
  guarantee that the same row-to-index mapping is maintained across different operations ...
  apply(), groupby(), or transform() may cause the index to be regenerated".
- The docs also warn against shuffles and single-partition window ops.
- So emulating pandas index semantics on a distributed engine is expensive and only partly
  faithful. **[unmeasured]** Spark was not installed.

### Pattern verdict

"Full API emulation" makes the 100s of estimators work unchanged, which is attractive for the
breadth problem. It fails sktime on three counts:
- fidelity: MultiIndex order, names, `isinstance`;
- latency: per-op overhead multiplies by the number of instances;
- it does not deliver what the user wants from polars or DuckDB, because the user's polars
  frame is not a Modin frame.

The concept worth keeping is cudf.pandas's **fallback with a profiler**: a fast path where
one is implemented, an automatic and logged slow path to pandas elsewhere.

---

## 5. Great Expectations, Hamilton, Kedro (brief)

- **Great Expectations** [doc: GX `ExecutionEngine` class reference]
  - It uses an `ExecutionEngine` base class with `PandasExecutionEngine`,
    `SparkDFExecutionEngine` and `SqlAlchemyExecutionEngine`.
  - Each metric is implemented per engine. Spark and SQL engines support *bundled, deferred*
    metric resolution (`resolve_metric_bundle`: many metrics in one round trip); pandas does
    not, "because Pandas computations are immediate".
  - Pattern: per-engine implementations of *small primitives* (metrics), plus a batching
    planner for lazy engines. This is the GE analogue of "narrow adapter + push down as a
    batch".
- **Hamilton** [doc/src snippets: `hamilton.plugins.h_polars`, polars example]
  - Its dataflow nodes are plain functions over a concrete type.
  - Backend support comes through **result builders** and plugins per dataframe type
    (pandas, polars and others). Hamilton does not abstract the dataframe API.
  - Its polars example warns that "Polars has no index, so you must control row order
    yourself".
  - Pattern: the framework is agnostic to the container; user functions are not.
- **Kedro** [**unmeasured** / from general knowledge of kedro-datasets, not re-verified this session]
  - Abstraction exists only at the I/O boundary: `pandas.CSVDataset`, `polars.EagerPolarsDataset`,
    `ibis.TableDataset`.
  - Nodes receive the native object and are written for one backend.
  - Pattern: a "container-typed boundary", similar to sktime's `*_inner_mtype` tags today.

---

## 6. Arrow PyCapsule interface as the canonical interchange format

### Adoption matrix (`r4_arrow_capsule.py`; pandas 2.3.3 / pandas 3.0.6)

| object | `__arrow_c_stream__` | `__arrow_c_array__` | `__dataframe__` (interchange) | `__dataframe_consortium_standard__` |
|---|---|---|---|---|
| pandas.DataFrame (incl. MultiIndex) | ✔ / ✔ | – | ✔ / ✔ | **✔ (pd2) / ✘ (pd3)** |
| pandas.Series | **✘ (pd2) / ✔ (pd3)** | – | – | `__column_consortium_standard__` ✔ pd2, ✘ pd3 |
| polars.DataFrame, polars.Series | ✔ | – | ✘ | ✘ |
| polars.LazyFrame | **✘** (must collect) | – | – | – |
| pyarrow Table / RecordBatchReader / ChunkedArray | ✔ | – | Table ✔ | – |
| pyarrow RecordBatch | ✔ | ✔ | ✔ | – |
| pyarrow Array | – | ✔ | – | – |
| duckdb.DuckDBPyRelation | ✔ | – | ✘ | – |
| narwhals DataFrame / Series (pandas or polars backed) | ✔ | – | ✘ | – |
| narwhals LazyFrame | ✘ | – | – | – |
| dask.DataFrame | **✘** | – | ✔ | – |
| ibis Table (memtable) | ✔ | – | ✔ | – |

Every eager container that sktime wants to support, except dask, now exports an Arrow stream.
Lazy frames (polars LazyFrame, narwhals LazyFrame) do not export one, so materialisation
stays an explicit decision.

### Zero-copy checks (buffer addresses, n=1e6 float64)

| hand-off | zero-copy? (pd2 / pd3) |
|---|---|
| polars → pyarrow via `pa.table(pl_df)` | ✔ / ✔ |
| pyarrow → polars via `pl.DataFrame(tbl)` | ✔ / ✔ |
| pandas (numpy, 1-D column) → pyarrow | ✔ / ✔ |
| pandas built from one 2-D array (block) → pyarrow, column a | **✘ (pd2) / ✔ (pd3)** |
| pyarrow → pandas `to_pandas()` default | ✘ / ✘ (copy) |
| pyarrow → pandas `types_mapper=pd.ArrowDtype` | ✔ / ✔ |
| polars → pandas `pd.DataFrame.from_arrow` (pd3 only) | ✘ (numpy float64 result, copy) |
| polars Series → numpy (float, no nulls) | ✔ |

Scaling over 3 sizes: 100k, 1M and 3M rows × 4 float64 columns, which is 3, 31 and 92 MB.

- Capsule hand-offs polars→pyarrow and pyarrow→polars take **constant time, about 0.17–0.29 ms
  regardless of size**, with 0 new Arrow allocations.
- `polars→pandas` and `pyarrow→pandas` allocate the **full data size**: 3.05, 30.5 and 91.6 MB
  of new Arrow memory, taking 1.9, 4.1 and 13.1 ms on pd2.
- `pandas→pyarrow` allocated 0 MB of Arrow memory, but its time scaled with size
  (3.3 → 12.5 → 24 ms on pd2). This suggests a linear validation or null-scan pass rather than
  a copy; the cause was not isolated.
- DuckDB `select *` over a registered polars frame, returned as Arrow, took a constant
  2.5–2.9 ms.

### MultiIndex through the capsule

- `pa.table(df_with_MultiIndex)` exports the index levels as **ordinary columns** plus `pandas`
  schema metadata (`columns: ['y','h0','time']`, `has_pandas_metadata: True`).
- `.to_pandas()` restores `MultiIndex(['h0','time'])` with its names, on both pd2 and pd3.
- So the Arrow route keeps level names through metadata, which answers the C2 complaint about
  a RangeIndex coming back.
- Caveat: `pl.DataFrame(pandas_df_with_MI)` (the polars constructor) **drops the index**
  (`columns: ['y']`). Ingesting into polars must therefore use `pl.from_pandas(df.reset_index())`
  or `pa.table(df)`, not the plain constructor.

### Canonical-Arrow iteration prototype (`r4_canonical_arrow.py`)

Any input → `pa.Table` (capsule, or one explicit collect/compute for lazy and dask inputs) →
dictionary-encode the key → contiguous run offsets → numpy views per instance.

| n_series×n_time | status quo: pandas MI groupby loop (base) | status quo: sktime polars_panel → convert_to pd → loop | arrow: pandas flat in | arrow: polars in | arrow: polars lazy (collect once) | arrow: duckdb rel (ORDER BY) | arrow: dask (compute once) |
|---|---|---|---|---|---|---|---|
| 10×100 | 1.74 ms | 4.40× | 2.98× | 0.34× | 0.95× | 3.0× | 6.4× |
| 100×100 | 6.46 ms | 3.20× | 0.44× | 0.18× | 0.29× | 0.93× | 1.63× |
| 1000×100 | 52.6 ms | 3.16× | 0.20× | **0.113×** | 0.144× | 0.44× | 0.44× |
| 10000×50 | 516 ms | (not run: conversion too slow) | 0.17× | **0.082×** | 0.083× | 0.26× | 0.19× |

- At 1000 series, today's polars path spends 111 ms of its 166 ms in `convert_to`. The capsule
  conversion takes 0.2 ms, which is **about 550× cheaper**.
- All results matched the pandas reference.
- Caveat: these iterate numpy views. Handing each instance to a *pandas-consuming* estimator
  would add per-group frame construction again (measured in other workstreams, x3/x4).

---

## 7. The data-apis DataFrame API standard and `dataframe-api-compat`

- `dataframe-api-compat` on PyPI: last release **0.2.7, 2024-04-26**, with no releases since.
  Measured with the PyPI JSON API. Its README still describes it as the "Implementation of the
  DataFrame Standard for pandas and Polars".
- The `data-apis/dataframe-api` README [doc] still says the standard is "draft and under
  development, moving towards an RFC status", while the interchange protocol is "released and
  ... stable".
- In practice the standard has been abandoned by its implementers:
  - pandas 2.3.3 still exposes `DataFrame.__dataframe_consortium_standard__` and
    `Series.__column_consortium_standard__`; **pandas 3.0.6 removed both** (measured above).
  - pandas 3 deprecated the interchange protocol (`__dataframe__`). Its
    `pandas/core/interchange/from_dataframe.py` now uses the PyCapsule interface first and emits
    a `Pandas4Warning` pointing users to *"Arrow PyCapsule Interface"* and *"Narwhals"*
    (verified by grep of the installed pd3 source, lines 49–130; pandas PR #60739).
  - polars does not implement `__dataframe__` in 2.0.0 (measured ✘).
- What replaced it:
  - **narwhals**, the same main author, as a compatibility layer *outside* the libraries, with
    a polars-like expression API.
  - **Arrow PyCapsule** for data exchange.
- Why the standard failed [doc + inference]:
  - A standard needs every library to implement and maintain it. A shim outside the libraries
    can ship weekly, while pandas and polars release on their own schedules.
  - The hard problems were dtype and null semantics, not method names. Gorelli makes this
    point in the Posit "Test Set" episode: interop is hard in "data types, not APIs".
  - A lowest-common-denominator API with no index and no row order did not fit pandas users,
    and polars users had a better native API anyway.
  - **Takeaway for sktime:** do not wait for or design a "standard"; build on narwhals and
    PyCapsule, which are where the ecosystem converged.

---

## 8. Engine-native "apply per group" overhead (`r4_group_apply.py --grid dev`)

The per-group function is trivial (`mean + last`), so the times measure framework overhead.
Ratios are against the pandas `for k, g in groupby(level=0)` loop, which is what `VectorizedDF`
does today.

| method | 10×100 | 100×100 | 1000×100 | 10000×50 | µs/group @10k |
|---|---|---|---|---|---|
| pandas groupby(level=0) loop (**base**) | 1.62 ms | 5.81 ms | 48.2 ms | 474 ms | 47.5 |
| pandas groupby(level)[c].apply(f) | 1.02× | 0.86× | 0.71× | 0.69× | 32.6 |
| numpy offsets + np.split (**lower bound**) | 0.18× | 0.11× | 0.082× | **0.073×** | 3.5 |
| polars partition_by(as_dict) loop | 0.69× | 0.63× | 0.55× | 0.51× | 24.2 |
| polars `for k,g in group_by()` | 1.52× | 1.57× | 1.00× | 1.37× | 65.2 |
| polars `group_by().map_groups(f)` | 1.24× | 1.91× | 2.17× | **1.96×** | 93.1 |
| polars lazy `agg(col.map_batches(f, returns_scalar))` | 1.95× | 1.00× | 1.63× | 0.81× | 38.6 |
| polars `agg(list)` then Python loop | 0.59× | 0.37× | 0.28× | 0.22× | 10.5 |
| narwhals(pandas) `for k,g in group_by()` | 1.24× | 1.76× | 1.89× | 1.84× | 87.5 |
| narwhals(polars) `for k,g in group_by()` | 1.35× | 2.29× | 1.58× | 2.17× | 102.8 |
| duckdb list() agg + arrow UDF (vectorised) | 4.39× | 2.14× | 0.71× | 0.46× | 21.6 |
| duckdb list() agg + native UDF (1 call/group) | 3.81× | 1.68× | 0.95× | 0.58× | 27.3 |
| duckdb sorted fetch → arrow → numpy split | 2.54× | 1.01× | 0.54× | 0.32× | 15.1 |
| ibis(duckdb) collect(order_by) + Python UDF | 5.56× | 2.35× | 1.02× | 0.65× | 31.0 |
| dask(sync) groupby().apply(f) | 21.5× ✘ | 7.97× | 1.69× | 0.72× | 34.0 |
| dask(threads=2) groupby().apply(f) | 24.4× ✘ | 9.22× | 2.33× | 0.86× | 40.8 |

✘ marks results that were **wrong**: 2 and 1 of 10 groups mismatched.
`r4_dask_order.py` confirms why: **within a dask `groupby().apply`, rows of a group that
spans partitions are not in time order** (1 of 10 groups was non-monotonic in time at 10×100).
Any per-series forecaster run through dask must re-sort inside the UDF. Fugue does this with
`presort`.

Observations:
- polars `map_groups` is the documented "apply per group", and it is the **slowest polars
  option** (about 2× pandas). The polars docs themselves discourage it in favour of expressions.
- `partition_by` (0.51×) and "agg to list, then loop" (0.22×) are the fast Python-level
  patterns.
- Narwhals' `group_by` iteration costs 1.6–2.3× the pandas loop on either backend. Building a
  per-group narwhals frame is the dominant cost.
- The ceiling for any Python-per-instance design is the numpy run-offset loop: 3.5 µs per
  group, **13.7× faster than today's loop**. Most of today's 47 µs per group is pandas
  frame-slicing overhead, not user work.
- DuckDB and Ibis per-group UDFs have a fixed cost of about 3–8 ms. They win only above about
  1000 groups.

---

## 9. Design patterns extracted

| pattern | prior art | what it buys sktime | cost / risk (measured where possible) |
|---|---|---|---|
| **A. Narrow adapter protocol** (type-keyed registry, a handful of primitives: `iter_instances`, `reconstruct`, `schema`, `to_native`) | pandera `BACKEND_REGISTRY`; GE ExecutionEngine; sktime mtype converters | extensible (new backend = new package); the core stays container-agnostic; registry dispatch costs 0.8–1.1 µs | each backend reimplements the primitives (pandera: 2–5 kLOC per backend); estimators still need *some* container. The surface to cover is small: `VectorizedDF` + `BaseForecaster` + `_fh` + `NaiveForecaster` use **16 distinct `pd.*` names and 52 pandas-specific attributes** (`r4_api_surface.py`; `_vectorize.py` alone uses 5 `pd.*` names and 19 attributes) |
| **B. Full API emulation** (keep the pandas API, swap the engine) | Modin, cudf.pandas, pyspark.pandas | zero estimator changes | Modin: groupby-iterate 721× and concat 484× slower; `sort=False` ignored; Series name lost; `isinstance(pd.DataFrame)` False; does not help users who bring polars or DuckDB |
| **C. Expression IR / compile to engine** | Ibis (SQLGlot), narwhals expressions, polars lazy | real pushdown and laziness; 11–23× faster than the per-instance loop for reduction forecasters at 10k series | only for forecasters expressible relationally; compile overhead of about 1–5 ms; version skew (ibis 12 + duckdb 1.5.6 `create_table` bug) |
| **D. Per-partition UDF execution** (engine groups, Python fits per group) | Fugue `transform`, Spark pandas UDFs, dask `groupby.apply`, duckdb list-agg UDF, polars `map_groups` | runs on the engine where the data lives; parallel/distributed `fit` across instances | Fugue: 21–47× the pandas loop (about 1.2–2.7 ms per group); requires an output schema; returns pandas; no MultiIndex or Period; dask: group order not preserved (wrong results without presort) |
| **E. Canonical interchange format** (Arrow via PyCapsule, offsets plus numpy views) | DuckDB/polars/pyarrow consumers; pandas 3 `from_dataframe`; narwhals interchange level; pandera pyarrow engine | one ingestion path for every eager container; zero-copy (0.2 ms constant); iteration 0.08–0.11× today's loop; MultiIndex names survive via pandas metadata | lazy and dask inputs need one explicit materialisation; output must be rebuilt per container (pd needs a copy unless `ArrowDtype`); Period has no Arrow type |
| **F. Hybrid: generic layer + native escape hatch** | pandera narwhals backend + native pandas Index backend; cudf.pandas fast/slow fallback | generic code where possible, faithful native code for the irreducible pandas concepts (MultiIndex, Period) | two paths to test; needs a fallback profiler or logging so users see when the slow path ran |

### Recommendation for sktime

The prior art converges on combining **A + E + F**, with **C** reserved for a few forecasters
and **D** as a later option for distributed engines.

1. **Boundary (A + E).** Keep `VectorizedDF`'s stateless schema. Make its two container-touching
   primitives, "iterate instances" and "reconstruct", a small type-keyed registry like pandera's.
   - The default backend ingests anything with `__arrow_c_stream__` zero-copy and iterates with
     run offsets.
   - The pandas backend keeps `groupby(level=...)` and `concat(keys=...)` so MultiIndex and
     Period are exactly faithful. This mirrors pandera keeping native Index/MultiIndex.
2. **Estimator side (F).** Estimators keep declaring `*_inner_mtype`. This is Fugue's
   annotation idea, which sktime already has.
   - Convert per instance only when an estimator requests pandas, and log it, as cudf.pandas
     logs fallbacks.
   - A forecaster that declares polars or numpy gets native slices.
3. **Selected forecasters (C).** Give reduction-style forecasters (Naive, Mean, Drift, seasonal
   naive, window summaries) an expression implementation, using narwhals or polars expressions,
   that runs once over all instances with no Python loop. This is where laziness and pushdown
   actually pay off: 11–23× at 10k series.
4. **Later (D).** For dask or Spark, map `fit` per partition with a mandatory presort by time.
   Do not use Fugue as a dependency: its per-group overhead and pandas normalisation would undo
   the gains.

## Caveats

- Timings come from a shared 4-core machine with another agent running, and use DEV_GRID only.
  Ratios measured in the same process are more reliable than absolute values. The Modin
  numbers are single-shot and include warm-up, so read them as orders of magnitude. Rerun
  everything with `--grid large` in the serial phase.
- Fugue's DuckDB engine failed on polars and Arrow inputs with `TIMESTAMP_NS is not supported`.
  That may be version skew (fugue 0.9.7 with duckdb 1.5.6) rather than a design limit.
- Statements about cudf.pandas, pyspark.pandas, GX, Hamilton and Kedro come from docs, not
  runs. Kedro's dataset classes were not re-verified this session.
- The API-surface count is a static AST heuristic, so it gives orders of magnitude only.
- The canonical-Arrow prototype hands out numpy views. It does not include the cost of building
  per-instance pandas objects for pandas-only estimators.
