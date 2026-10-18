# x1: narwhals `collect()` semantics on lazy backends (claim C1)

Workstream key: `x1_nw_lazy_collect`. All numbers below come from the scripts listed in
"Commands". Anything not measured is marked *unmeasured/expected*.

## TL;DR

**Claim C1:** "Polars lazy collects to Polars by default, but Narwhals' `collect` also sends
Dask to pandas and DuckDB to Arrow."

**Verdict: true as stated, but misleading as an objection.**

* **The defaults are real and documented.** In narwhals 2.27.1 the `LazyFrame.collect` docstring
  and code give these defaults: polars.LazyFrame -> polars.DataFrame, dask -> pandas,
  duckdb -> pyarrow, pyspark -> pyarrow. ibis -> pyarrow is in code, not the docstring. We
  observed all of them (table B).
* **The defaults copy each backend's own native materialisation.** `collect()` gives the same
  dtypes as the backend's own call: dask `.compute()` returns pandas, DuckDB `.to_arrow_table()`
  returns pyarrow, ibis `.to_pyarrow()` returns pyarrow. Across 11 tricky dtypes, zero dtype or
  null/NaN changes. Time is within run-to-run noise, at 0.82–1.34x the native call (F2, F3). There
  is no "eager DuckDB" or "eager dask" type to collect into. So "collect into the original format"
  is not a meaningful request for lazy-only backends.
* **You can override the target, but only to three libraries.** `collect(backend=...)` accepts
  "pandas", "polars", "pyarrow" (as strings, `Implementation` values or modules). It raises
  `ValueError` for "dask", "duckdb", "ibis" and "modin".
* **To keep the original format, don't collect.** Every narwhals LazyFrame op that succeeded
  returned the original native type from `.to_native()`, with zero hidden dask computes:
  * polars.LazyFrame: 34/34 applicable ops
  * DuckDB relation on the default connection: 32/34
  * ibis: 31/34
  * dask: 24/34

  All results matched a polars-eager reference.
* **If you must collect, a hand-written "relift" gets back to the same lazy type** (dtype-lossless).
  It costs 0.85–1.8x a native collect on pandas 2 (up to 4.3x for dask on pandas 3).
  narwhals' own `.collect().lazy(backend=impl)` does the same, but each backend loses something:
  * dask: npartitions 8 -> 1
  * DuckDB: the relation lands on the default connection, so it can't be joined with the user's
    connection
  * ibis: you get an unbound memtable

**What actually bites, and C1 does not mention it:**
1. **Op coverage on LazyFrame varies a lot by backend** (matrix C). Most notably:
   * dask has no `over(order_by=...)`, `rank`, `last`, `cum_count(reverse)` or `tail`.
   * pyarrow has no `over(order_by)` windows and no `join_asof`.
   * DuckDB and ibis have no `tail` or `gather_every`.
2. **The default collect target decides which eager ops you get.** DuckDB and ibis collect to
   pyarrow by default, and the pyarrow narwhals backend can't `pivot`, can't do `over(order_by)`
   windows and can't do `join_asof`. Code that calls `.collect()` without `backend=` gets a
   different eager feature set depending on the input. Always pass an explicit backend.
3. **DuckDB relations from a user-owned connection break 4 of 34 ops.** `top_k`, `unpivot`,
   `join_asof` and joins with narwhals-created frames fail. The cause: narwhals runs these through
   module-level `duckdb.sql(...)` replacement scans, and `.lazy("duckdb")` goes through
   `duckdb.table(...)`, which binds to the default connection. A `DuckDBPyRelation` does not expose
   its connection, so sktime can't recover it either.
4. **Row order is not preserved** by lazy ops (polars join; DuckDB/ibis `over(order_by)`; dask
   `over`). This matters for `reconstruct`.
5. **The pandas target loses information.** Collecting polars, DuckDB or ibis to pandas gives:
   * int with nulls -> float64
   * null and NaN merged
   * bool with nulls -> object
   * decimal -> object or float
   * date -> datetime

   The polars and pyarrow targets kept every dtype family (F1).
6. **Staying lazy only pays off at scale, and only for engines that optimise.**
   * At 500k rows, the narwhals lazy plan was 1.4–1.6x faster than collect-all.
   * With an order-aware aggregation (`last(order_by=)`) it was 2.1–3.8x faster, with 8–30x
     lower peak memory.
   * At ≤10k rows it was 0.4–0.8x (slower).
   * On dask, staying lazy was 2.7–12x *slower* at every dev size.
7. **Iterating groups lazily is a dead end.** You can't iterate a `LazyGroupBy`. A per-key
   `filter().collect()` loop costs 1.9–34 ms per group, about 200–660x slower than one grouped query
   at 1000 groups. Lazily concatenating 1000 per-instance DuckDB relations fails with
   `Max expression depth limit of 1000 exceeded`.

## Environment

| lib | EXP_PY (pd2) | EXP_PY3 (pd3) |
|---|---|---|
| python | 3.12.3 | 3.12.3 |
| narwhals | 2.27.1 | 2.27.1 |
| polars | 2.0.0 | 2.0.0 |
| pandas | 2.3.3 | 3.0.6 |
| pyarrow | 26.0.0 | 26.0.0 |
| dask | 2026.8.0 (scheduler forced to `synchronous`) | 2026.8.0 |
| duckdb | 1.5.6 (`SET threads=4`) | 1.5.6 |
| ibis | 12.0.0 (duckdb backend) | not installed |
| modin | 0.37.1 (`MODIN_ENGINE=python`) | not installed |

`POLARS_MAX_THREADS=4`. The machine was shared with about 14 other agents, so absolute timings
are noisy. Ratios come from the same process with interleaved min-of-repeats runs (`common.bench`).

## Commands

All commands were run from the worktree root. `ED=experiments/container_independence`.

```
SP=/tmp/claude-1000/.../scratchpad   # session scratchpad (X1_TMP for temp parquet files)
X1_TMP=$SP/x1tmp PYTHONPATH=.:$ED $EXP_PY  -W ignore $ED/scripts/x1_matrix.py --tag pd2
X1_TMP=$SP/x1tmp PYTHONPATH=.:$ED $EXP_PY3 -W ignore $ED/scripts/x1_matrix.py --tag pd3
PYTHONPATH=.:$ED $EXP_PY  -W ignore $ED/scripts/x1_fidelity.py --tag pd2
PYTHONPATH=.:$ED $EXP_PY3 -W ignore $ED/scripts/x1_fidelity.py --tag pd3
PYTHONPATH=.:$ED $EXP_PY  -W ignore $ED/scripts/x1_fallback.py --grid dev --tag pd2
PYTHONPATH=.:$ED $EXP_PY3 -W ignore $ED/scripts/x1_fallback.py --grid dev --tag pd3
X1_TMP=$SP/x1tmp PYTHONPATH=.:$ED $EXP_PY -W ignore $ED/scripts/x1_stay_lazy.py --grid dev   # 1 subprocess per backend
PYTHONPATH=.:$ED $EXP_PY  -W ignore $ED/scripts/x1_concat.py --grid dev
```

Raw outputs:
* `results/x1_matrix_{pd2,pd3}.json`
* `results/x1_fidelity_{pd2,pd3}.json`
* `results/x1_fallback_dev_{pd2,pd3}.json`
* `results/x1_stay_lazy_dev_pd2.json`
* `results/x1_concat_dev_pd2.json`

The benchmark scripts (`x1_fallback`, `x1_stay_lazy`, `x1_concat`) accept `--grid {dev,large}`
for the serial large-scale phase.

Data:
* Matrix, fallback, stay-lazy and concat runs use `common.make_panel_fast(...).reset_index()`:
  long format, keys as columns (`h0` str, `time` datetime64[ns], `c0`, `c1` float64).
* The matrix uses 20 series x 30 time points.
* The fallback and stay-lazy runs use `DEV_GRID`.

## A/B. `from_native` and `collect(backend=...)` (script `x1_matrix.py`, part AB)

Results are identical on pd2 and pd3, except that ibis and modin are absent on pd3.

| native input | narwhals class | implementation | `eager_only=True` | `collect()` default | `collect(backend=)` accepted -> result | rejected (ValueError "Unsupported `backend` value") |
|---|---|---|---|---|---|---|
| pandas.DataFrame | DataFrame | pandas | ok | (via `.lazy()`) pandas | 'pandas'/'polars'/'pyarrow', `Implementation.{PANDAS,POLARS,PYARROW}`, modules pd/pl -> pandas / polars / pyarrow | 'dask','duckdb','ibis','modin', `Implementation.{DASK,DUCKDB,MODIN}` |
| polars.DataFrame | DataFrame | polars | ok | (via `.lazy()`) polars | same | same |
| polars.LazyFrame | LazyFrame | polars | TypeError | **polars.DataFrame** | same | same |
| pyarrow.Table | DataFrame | pyarrow | ok | (via `.lazy()`) pyarrow | same | same |
| dask DataFrame | LazyFrame | dask | TypeError | **pandas.DataFrame** (1 dask compute) | same; polars/pyarrow go through pandas first (`pl.from_pandas(compute())`) | same |
| DuckDBPyRelation | LazyFrame | duckdb | TypeError | **pyarrow.Table** | same | same |
| ibis Table (duckdb) | LazyFrame | ibis | TypeError | **pyarrow.Table** | same | same |
| modin DataFrame | DataFrame | modin | ok | (via `.lazy()`) modin | same (no modin target) | same |

* **`DataFrame.lazy(backend=...)`** accepts None (keeps the eager backend under the lazy API),
  'polars', 'duckdb', 'dask' and 'ibis'. 'pandas' and 'pyarrow' raise
  `ValueError: Not-supported backend.`
* **Collect kwargs pass through:**
  * `collect(engine="streaming")` on polars returns polars.DataFrame.
  * `collect(scheduler="threads")` on dask returns pandas.
* **Source cross-check (narwhals 2.27.1):**
  * `can_lazyframe_collect` allows exactly {PANDAS, POLARS, PYARROW} (`narwhals/_utils.py:1661`).
  * The dask compliant `collect` always calls `self.native.compute()` first, then converts to the
    requested target (`_dask/dataframe.py:127`).
  * DuckDB uses `to_arrow_table()`, `.df()` or `.pl()`.
  * ibis uses `to_pyarrow()`, `to_pandas()` or `to_polars()`.

## C. Operation matrix on `nw.LazyFrame` (script `x1_matrix.py`, part C)

Legend:
* **lazy**: op built without error, `.to_native()` is still the *original* native type (isinstance
  check), and the collected result equals the polars-eager reference.
* **eager**: the input was an eager backend wrapped by `.lazy()`. The op ran eagerly, and the
  result is still the original eager type.
* **(reordered)**: input row order was not preserved. Checked only for row-count-preserving ops.
* **ERR**: an exception was raised.
* **COLLECT-ERR**: the plan was built, but collecting it failed.

dask compute calls were counted with a dask `Callback` while each plan was built. All counts were
0, except `sink_parquet`, which computes once by design.

`duckdb` means a relation on duckdb's default connection (`duckdb.from_arrow`). `duckdb_owncon`
means a relation from `con = duckdb.connect(); con.from_arrow(...)`, which is the usual pattern in
production code.

| op | pandas | polars_eager | polars_lazy | pyarrow | dask | duckdb | duckdb_owncon | ibis_duckdb | modin |
|---|---|---|---|---|---|---|---|---|---|
| with_columns | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| filter | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| select_cols | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| sort | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| collect_schema | metadata | metadata | metadata | metadata | metadata | metadata | metadata | metadata | metadata |
| unique_keys | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| unique_keys_sorted | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| n_rows_per_key (group_by.agg(len)) | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| filter_one_key | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| group_by_agg_mean_max | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| group_by_agg `last()` (no order_by) | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR |
| `sort().group_by().agg(last())` | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR |
| group_by_agg `last(order_by="time")` | eager | eager | lazy | eager | ERR | lazy | lazy | lazy | eager |
| over_mean (no order) | eager | eager | lazy | eager | lazy (reordered) | lazy | lazy | lazy | eager |
| over_cum_sum(order_by) | eager | eager | lazy | ERR | ERR | lazy (reordered) | lazy (reordered) | lazy (reordered) | eager |
| over_shift(order_by) | eager | eager | lazy | ERR | ERR | lazy (reordered) | lazy (reordered) | lazy (reordered) | eager |
| over_shift (no order_by) | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR |
| over_diff(order_by) | eager | eager | lazy | ERR | ERR | lazy | lazy | lazy | eager |
| over_rolling_mean(order_by) | eager | eager | lazy | ERR | ERR | lazy | lazy | lazy | eager |
| over_ffill(order_by) | eager | eager | lazy | ERR | ERR | lazy | lazy | ERR | eager |
| last_row_per_key (filter `time == max().over`) | eager | eager | lazy | eager | lazy **COLLECT-ERR** | lazy | lazy | lazy | eager |
| last_row_per_key (semi-join on per-key max) | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| last_k_per_key (rank over) | eager | eager | lazy | ERR | ERR | lazy | lazy | lazy | eager |
| last_k_per_key (cum_count(reverse) over) | eager | eager | lazy | ERR | ERR | lazy | lazy | lazy | eager |
| with_row_index(order_by) | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| head | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| tail | eager | eager | lazy | eager | ERR | ERR | ERR | ERR | eager |
| top_k | eager | eager | lazy | eager | lazy | lazy | **ERR** | lazy | eager |
| gather_every | eager | eager | lazy | eager | lazy | ERR | ERR | ERR | eager |
| join_left_on_key | eager | eager (reordered) | lazy (reordered) | eager | lazy | lazy | lazy | lazy | eager |
| join_cross_horizon (lift small frame via `.lazy(backend=impl)`) | eager | eager | lazy | eager | lazy | lazy | **ERR** | lazy | eager |
| concat_vertical (`nw.concat`) | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| concat_diagonal | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| unpivot | eager | eager | lazy | eager | lazy | lazy | **ERR** | lazy | eager |
| join_asof | eager | eager | lazy | ERR | lazy | lazy | **ERR** | lazy | eager |
| map_batches | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR |
| iter groups (`iter(lf.group_by)`) | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR |
| to_numpy | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR |
| pivot | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR |
| select(nw.len()) | eager | eager | lazy | eager | lazy | lazy | lazy | lazy | eager |
| sink_parquet | written | written | written | written | written (1 compute) | written | written | written | written |

Seven of the 41 probed ops fail for *every* backend by narwhals design:
* Order-dependent expressions without `order_by` (3 rows), which raise `InvalidOperationError:
  Order-dependent expressions are not supported for use in LazyFrame.`
* `map_batches`, which narwhals treats as order-dependent in a LazyFrame, even on polars lazy,
  where polars itself allows it.
* `iter(LazyGroupBy)`, which raises `TypeError: 'LazyGroupBy' object is not iterable`.
* `LazyFrame.to_numpy` and `LazyFrame.pivot`, which don't exist (AttributeError).

Coverage of the remaining 34 applicable ops:

| backend | applicable ops OK |
|---|---|
| polars_lazy | 34/34 |
| duckdb (default connection) | 32/34 |
| ibis | 31/34 |
| duckdb_owncon | 28/34 |
| dask | 24/34 |

pd3 gives exactly the same matrix for the shared backends (`results/x1_matrix_pd3.json`).

Error causes (messages verbatim from the JSON):
* **dask:**
  * `over` with `order_by` is not yet supported in Dask (narwhals cites wrong results,
    dask/dask#11806).
  * 'rank', 'last' and `map_batches` are not implemented for dask.
  * `cum_count(reverse=True)` is not supported.
  * `LazyFrame.tail` is not supported with multiple partitions.
  * The filter `time == time.max().over("h0")` builds lazily, but `compute()` dies inside
    dask-expr: `_repartition._layer`: `TypeError: '<' not supported between instances of
    'NoneType' and 'NoneType'`. This is a dask bug. The semi-join formulation works.
* **pyarrow (eager under the lazy API):** "Only elementary aggregations are supported for `.over`
  in PyArrow backend when `partition_by` is specified". `join_asof` is not implemented.
* **duckdb:** `'DuckDBLazyFrame' object has no attribute 'tail'` / `'gather_every'`.
* **duckdb_owncon:**
  * `top_k`, `unpivot` and `join_asof` raise `InvalidInputException: Python Object "_rel" of type
    "DuckDBPyRelation" not suitable for replacement scan`. Source:
    `narwhals/_duckdb/dataframe.py:397,478,556` build SQL strings and run them with module-level
    `duckdb.sql(query)`.
  * The cross join with a frame lifted through `.lazy(backend="duckdb")` raises `Cannot combine
    LEFT and RIGHT relations of different connections!`. `_pandas_like/dataframe.py` lazy() uses
    `duckdb.table("pandas_df")`, which binds to the default connection.
  * `DuckDBPyRelation` has no public attribute for its connection (checked with `dir()`).
* **ibis:** `fill_null(strategy=...)` -> "`strategy` is not supported for the Ibis backend".
  `tail`/`gather_every` don't exist.

Row order:
* polars joins (`maintain_order` defaults to none) reordered rows in our test.
* DuckDB/ibis `over(order_by)` windows reordered rows.
* dask `over` reordered rows (it is groupby-transform based).
* narwhals documents that LazyFrame has no row-order guarantee. This is a correctness hazard for
  any VectorizedDF that relies on position (today `pd.concat(keys=)` keeps order).
* Ops where we saw order preserved at 600 rows (e.g. DuckDB join) are **not guaranteed** to keep
  it at scale or with more threads (*unmeasured*).

## D. Eager-only ops after collect (script `x1_matrix.py`, part D)

| eager frame | iter groups | to_numpy | pivot | group_by.agg(last()) |
|---|---|---|---|---|
| pandas / modin / polars / dask.collect() (pandas) | ok (20 groups, tuple keys) | ok | ok (pandas, polars); **modin: NotImplementedError (modin#7409)** | ok |
| pyarrow / duckdb.collect() / ibis.collect() (all pyarrow) | ok | ok | **NotImplementedError for pyarrow** | ok |

`partition_by` doesn't exist on narwhals DataFrame (AttributeError). Iterating groups, `to_numpy`
and `pivot` are therefore eager-only, and `pivot` is missing for the pyarrow target. That is the
*default* target for DuckDB and ibis. `vectorize_est` today depends on a pivot.

## F1. Dtype / null fidelity (script `x1_fidelity.py`)

Input: 12 rows, built in pyarrow, with these columns:
* `key` string
* `time_tz` timestamp[us, UTC]
* `time_ns` timestamp[ns]
* `date` date32
* `i_null` int64 with 3 nulls
* `f_null_nan` float64 with 1 null and 1 NaN
* `b_null` bool with 3 nulls
* `cat` dictionary
* `u8` uint8
* `dur` duration[us]
* `dec` decimal(10,2)

Each route is compared with the backend's *own* native materialisation. "Lossy" means a
dtype-family change or a null/NaN count change.

| route | polars lazy | dask | duckdb | ibis |
|---|---|---|---|---|
| `collect()` default | 0 changes | 0 | 0 | 0 |
| `collect("pyarrow")` | 0 lossy (names only, e.g. String -> large_string) | 0 lossy | 0 | 0 |
| `collect("polars")` | 0 | 0 lossy | **ComputeError: could not import `month_day_nano_interval`** (duckdb INTERVAL) | 0 lossy |
| `collect("pandas")` pd2 | **6 lossy**: string->object, date->datetime64[ms], Int64-with-null->float64, null+NaN merged (1 null + 1 NaN -> 2 NaN), bool->object, decimal->object | 0 | **6 lossy**: string->object (2 cols), date->datetime64[us], null+NaN merged, decimal->float64 (precision), interval->timedelta | **7 lossy** (same as polars, plus cat-string->object) |
| `collect("pandas")` pd3 | 5 lossy (string becomes `str`, not object; the rest is the same) | 0 | 4 lossy | n/a |
| `collect().lazy(backend=impl)` -> native collect | 0 | 0 | 0 | 0 |
| hand relift (`rt_manual`) -> native collect | 0 | 0 | 0 | 0 |

Each backend's own ingestion was already lossy before narwhals was involved:
* DuckDB:
  * converted `timestamp[us, UTC]` to the session time zone (`Asia/Karachi` on this host)
  * turned dictionary into VARCHAR
  * turned duration into INTERVAL, which then breaks `collect("polars")`
  * ibis: `to_pyarrow()` on that INTERVAL column raised `ArrowNotImplementedError`, so the script
    dropped `dur` for ibis
* dask: `dd.from_pandas` turned object columns (date, decimal) into `string`. On pd3, building the
  pandas `Float64` input from arrow also turned NaN into NA.

These are backend properties, not narwhals'.

## F2/F3. Fallback route: cost of collect and "convert back" (script `x1_fallback.py --grid dev`)

The input is a lazy native object holding `n_rows` rows: (str, datetime, 2 floats).

What each route does:
* `native_collect`:
  * polars: `L.collect()`
  * dask: `.compute()`
  * duckdb: `rel.to_arrow_table()`
  * ibis: `t.to_pyarrow()`
* `rt_manual`:
  * polars: `.collect().lazy()`
  * dask: `dd.from_pandas(compute(), npartitions=L.npartitions)`
  * duckdb: `user_con.from_arrow(to_arrow_table())`
  * ibis: `ibis.memtable(to_pyarrow())`
* `rt_nw`: `nw.from_native(L).collect().lazy(backend=impl).to_native()`

Each cell shows min wall time over interleaved repeats, with the ratio to `native_collect` in
parentheses.

pd2, 100k and 500k rows (second run, after a memory fix; `results/x1_fallback_dev_pd2.json`):

| backend | rows | native_collect | nw collect() | nw collect('pandas') | nw collect('polars') | nw collect('pyarrow') | rt_manual | rt_nw |
|---|---|---|---|---|---|---|---|---|
| polars_lazy (in-memory) | 100k | 0.5 ms | 0.6 (x1.25) | 7.0 (x13.6) | 0.5 (x1.06) | 2.1 (x4.08) | 0.6 (x1.2) | 0.6 (x1.14) |
| dask (8 partitions) | 100k | 34.0 ms | 36.2 (x1.06) | 39.3 (x1.16) | 44.6 (x1.31) | 39.5 (x1.16) | 49.6 (x1.46) | 48.4 (x1.43) |
| duckdb table | 100k | 9.8 ms | 10.4 (x1.06) | 15.8 (x1.61) | 12.3 (x1.26) | 10.6 (x1.07) | 10.1 (x1.03) | 11.8 (x1.2) |
| ibis table | 100k | 11.9 ms | 13.0 (x1.09) | 27.0 (x2.27) | 12.6 (x1.06) | 9.9 (x0.83) | 12.9 (x1.08) | 12.1 (x1.02) |
| polars_lazy (in-memory) | 500k | 0.5 ms | 0.5 (x0.91) | 30.4 (x58.5) | 0.6 (x1.13) | 7.7 (x14.8) | 0.5 (x1.04) | 0.6 (x1.24) |
| dask (8 partitions) | 500k | 86.0 ms | 73.7 (x0.86) | 73.9 (x0.86) | 115.1 (x1.34) | 103.2 (x1.2) | 156.0 (x1.81) | 151.0 (x1.76) |
| duckdb table | 500k | 14.7 ms | 17.2 (x1.16) | 105.3 (x7.14) | 19.4 (x1.31) | 17.3 (x1.17) | 19.1 (x1.3) | 18.4 (x1.25) |
| ibis table | 500k | 18.8 ms | 19.1 (x1.01) | 144.0 (x7.64) | 21.2 (x1.13) | 21.4 (x1.13) | 16.0 (x0.85) | 29.2 (x1.55) |

pd3 at 500k rows (`results/x1_fallback_dev_pd3.json`):

| backend | native_collect | nw collect('pandas') | rt_manual | rt_nw |
|---|---|---|---|---|
| polars | 0.4 ms | 10.5 (x23.6) | 0.4 (x0.99) | 0.5 (x1.22) |
| dask | 28.2 ms (about 3x faster than pd2) | 30.6 (x1.08) | 120.4 (x4.27) | 120.2 (x4.26) |
| duckdb | 21.8 ms | 161.4 (x7.39) | 22.5 (x1.03) | 24.6 (x1.13) |

Peak RSS deltas at 500k rows, pd2:
* `collect('pandas')`: duckdb 48 MB, ibis 51 MB, against 4.7–7.7 MB for the arrow default.
* dask `collect('polars')`: 27 MB, against 17.5 MB for the default. It goes pandas -> polars, so
  it is a double conversion.

The first pd2 run showed the same patterns with different noise; for example dask `rt_manual` was
x1.58 at 500k there.

Round-trip type fidelity, checked at every size:

| backend | `rt_manual` | `rt_nw` (`.lazy(backend=impl)`) |
|---|---|---|
| polars | same type, schema equal | same type, schema equal |
| dask | same type, schema equal, **npartitions 8 -> 8** | same type, schema equal, **npartitions 8 -> 1** (narwhals calls `dd.from_pandas(pandas_df)` with the default) |
| duckdb | same type, schema equal, joinable with user connection: **True** | same type, schema equal, joinable with user connection: **False** ("Cannot combine LEFT and RIGHT relations of different connections!") |
| ibis | `ibis.Table`, schema equal, but an unbound memtable ("Expression depends on no backends") | same |

The polars `native_collect` is close to free here because the LazyFrame wraps an in-memory frame.
For a `scan_parquet` source the collect cost is I/O-bound; see F4.

## F4. Stay lazy vs collect-all (script `x1_stay_lazy.py --grid dev`, pd2)

Each backend reads a parquet file lazily. The workload is per-series cutoff, last value, mean and
n, which covers the fit state of NaiveForecaster with strategies "last" and "mean". Routes:
* **R1**: `collect()` the whole frame, then run the narwhals expression eagerly.
* **R2**: the same portable narwhals expression (semi-join on max time, plus group_by mean/len,
  plus join) run lazily, collecting only N rows.
* **R2b**: lazy with `group_by.agg(c0.last(order_by="time"))`.
* **R0**: `collect('pandas')` followed by a pandas groupby, which is today's shape.
* **R_native**: hand-written native query: polars `sort_by().last()`, DuckDB `arg_max` SQL, ibis
  `argmax`, dask merge.

All routes matched the pandas reference. "match(approx)" means float noise below 1e-9 relative.

Each cell shows time, speedup vs R1 and peak RSS delta:

| backend | rows | R1 collect-all | R2 stay lazy | R2b lazy last(order_by) | R0 pandas | R_native | R3 per-key filter loop |
|---|---|---|---|---|---|---|---|
| polars | 10x100 | 9.2 ms (x1.0; 17.2 MB) | 12.1 ms (x0.77; 1.8 MB) | 3.8 ms (x2.42) | 8.2 ms (x1.13) | 3.3 ms (x2.83) | 3.1 ms/group |
| polars | 1000x100 | 13.5 ms (x1.0; 12.9 MB) | 13.1 ms (x1.03; 6.6 MB) | 9.1 ms (x1.48) | 42.4 ms (x0.32) | 7.2 ms (x1.86) | 2.6 ms/group -> 2.64 s |
| polars | 10000x50 | 54.2 ms (x1.0; 68.8 MB) | 39.4 ms (x1.38; 21.7 MB) | 25.7 ms (x2.11; 8.1 MB) | 170.7 ms (x0.32; 43.1 MB) | 31.6 ms (x1.72) | skipped |
| duckdb | 10x100 | 8.7 ms (x1.0) | 23.7 ms (x0.37) | 8.6 ms (x1.02) | 8.3 ms (x1.04) | 3.7 ms (x2.36) | 5.7 ms/group |
| duckdb | 1000x100 | 31.6 ms (x1.0; 12.3 MB) | 31.4 ms (x1.01; 0.4 MB) | 13.7 ms (x2.31) | 43.6 ms (x0.73) | 13.0 ms (x2.42) | 6.2 ms/group -> 6.22 s |
| duckdb | 10000x50 | 81.5 ms (x1.0; 39.2 MB) | 54.2 ms (x1.50; 6.2 MB) | 21.7 ms (x3.76; 1.3 MB) | 245.6 ms (x0.33; 75.2 MB) | 26.0 ms (x3.13) | skipped |
| dask | 10x100 | 24.3 ms (x1.0) | 294.9 ms (x0.08) | ERR ('last' not implemented) | 13.9 ms (x1.75) | 194.2 ms (x0.13) | 12.3 ms/group |
| dask | 1000x100 | 51.3 ms (x1.0) | 304.6 ms (x0.17) | ERR | 38.4 ms (x1.34) | 236.1 ms (x0.22) | 33.8 ms/group -> 33.8 s |
| dask | 10000x50 | 137.2 ms (x1.0; 60.6 MB) | 375.2 ms (x0.37; 29.4 MB) | ERR | 127.1 ms (x1.08) | 297.2 ms (x0.46) | skipped |
| ibis | 10x100 | 14.5 ms (x1.0) | 25.7 ms (x0.56) | 14.4 ms (x1.01) | 14.7 ms (x0.99) | 12.0 ms (x1.2) | 6.3 ms/group |
| ibis | 1000x100 | 31.0 ms (x1.0) | 26.6 ms (x1.17) | 18.2 ms (x1.71) | 50.8 ms (x0.61) | 17.7 ms (x1.75) | 9.7 ms/group -> 9.71 s |
| ibis | 10000x50 | 71.7 ms (x1.0; 43.6 MB) | 44.7 ms (x1.60; 4.4 MB) | 29.6 ms (x2.42; 2.3 MB) | 218.6 ms (x0.33; 92.1 MB) | 28.6 ms (x2.5) | skipped |

The 100x100 rows are in the JSON and follow the same pattern.

Interpretation:
* **Staying lazy wins only at ≥100k rows, and only for polars, DuckDB and ibis.** The *expression
  you choose* matters as much as laziness. The portable semi-join plan (R2) was 1.4–1.6x faster at
  500k rows. The order-aware `last(order_by=)` (R2b) was 2.1–3.8x faster, close to hand-written
  native queries (R2b/R_native = 0.81–1.03x at 500k rows, 1.03–1.26x at 100k). Dask doesn't support R2b at all.
* **Peak memory with R2b was 8–30x lower than R1** at 500k rows: 8.1 vs 68.8 MB (polars), 1.3 vs
  39.2 MB (duckdb), 2.3 vs 43.6 MB (ibis).
* **On dask, staying lazy is slower at every dev size.** The synchronous-scheduler graph overhead
  and shuffle-based groupby/merge cost about 250 ms. Collect-to-pandas (R0/R1) wins. Large-grid
  numbers will be filled in during the serial phase.
* **Today's pandas route (R0) is slower than R1** for polars/DuckDB/ibis input: 1.4–3.1x at
  100k rows and 3.0–3.2x at 500k rows, with up to 92 MB peak. That is the cost sktime pays now for non-pandas
  input.
* **"Iterate groups lazily" via per-key filter+collect costs 1.9–34 ms per group.** At 1000 groups
  that is 2.6 s (polars), 6.2 s (duckdb), 9.7 s (ibis) and 33.8 s (dask). One grouped query (R1)
  takes 13.5, 31.6, 31.0 and 51.3 ms. So the loop is about 196x, 197x, 313x and 659x slower.

## F5. Reassembling per-instance lazy results (script `x1_concat.py --grid dev`, pd2)

`nw.concat` of N lazy pieces (50 rows each), compared with collecting each piece and concatenating
eagerly:

| N | polars build+collect / eager | duckdb | dask |
|---|---|---|---|
| 10 | 1.7 ms / 2.8 ms (x0.6) | 50.2 ms / 24.5 ms (x2.05) | 54.7 ms / 45.5 ms (x1.2) |
| 100 | 10.1 ms / 28.2 ms (x0.36) | **build 670.8 ms** + collect 193.7 ms / 288.9 ms (x2.99) | 531.7 ms / 758.5 ms (x0.7) |
| 1000 | 300.3 ms / 352.0 ms (x0.85) | **BinderException: Max expression depth limit of 1000 exceeded** | 8047.6 ms / 6279.6 ms (x1.28) |

So a VectorizedDF that keeps per-instance results lazy and rebuilds them with `nw.concat` doesn't
scale on DuckDB: plan build is superlinear, and it hard-fails at about 1000 instances. Dask pays
about 8 ms per piece either way. Per-instance work has to be expressed as *set-based* grouped
operations, not loop + concat.

## Workarounds evaluated

| workaround | works? | evidence | cost / caveat |
|---|---|---|---|
| Pass an explicit `collect(backend=...)` | yes, for pandas/polars/pyarrow only | table B | Use `"polars"` or `"pyarrow"` for fidelity (F1). Avoid pandas for polars/DuckDB/ibis input (lossy, 7–58x slower than native collect at 500k). `polars` fails on DuckDB INTERVAL. |
| Stay lazy and `to_native()` at the end | yes; type preserved for every op that succeeded | matrix C | Op coverage varies by backend. Row order not preserved. Needs `order_by` everywhere. |
| Collect only per-series results (grouped lazy aggregation) | yes for polars/DuckDB/ibis | F4 | 1.4–3.8x faster and 8–30x less peak memory at 500k rows. Slower at ≤10k rows. Slower on dask at every dev size. Only fits estimators whose fit/predict is an aggregation (naive, mean, drift, window means). |
| `.collect().lazy(backend=impl)` (narwhals) | type yes, fidelity partly | F2/F3 | dask loses partitioning (8 -> 1). DuckDB moves to the default connection. ibis gets an unbound memtable. |
| sktime-side `relift(eager, like=native, con=...)` | yes | `rt_manual` in F2/F3 | 0.85–1.8x a native collect (pd2), up to 4.3x for dask on pd3. Needs the user's DuckDB connection passed in explicitly, because the relation doesn't expose it. |
| `map_batches` for arbitrary per-batch code | no on LazyFrame | matrix C | Use `nw.to_native` -> native op -> `nw.from_native`, or collect. |
| Iterate groups lazily | no (`LazyGroupBy` isn't iterable) | matrix C; F4 R3 | The per-key filter loop is about 200–660x slower at 1000 groups. Collect once and iterate eagerly instead. |
| Lazy `nw.concat` of per-instance results | polars yes; DuckDB fails at 1000 | F5 | Use set-based ops. |

## Implications for sktime (what x1 supports)

1. **C1 is not a reason to reject narwhals.** The default collect targets match what each backend
   already returns natively. Inside sktime, always call `collect(backend="polars")` (or
   "pyarrow") explicitly, so that the eager feature set and dtype fidelity don't depend on the
   input library.
2. **"Respect the original format" means two things for a VectorizedDF:**
   * the *output* must be relifted to the input's container type;
   * per-instance *compute* must be set-based (group_by/over/semi-join), not per-instance
     iteration.

   That holds for vectorized aggregations. Arbitrary per-instance Python estimators (most
   forecasters) still need eager per-group numpy slices: collect once, iterate eagerly, relift
   the result. The relift is cheap (≤1.8x a collect, ~0.5–156 ms at 500k rows on pd2).
3. **A lazy VectorizedDF must not depend on row order.** Carry explicit `order_by=time` and sort
   before reconstruct. Avoid `tail`/`gather_every` (missing on DuckDB/ibis) and `over(order_by)`
   (missing on dask). Prefer semi-join on max time, which works on all 4 lazy backends.
4. **DuckDB users on their own connection need special handling.** Either accept the connection
   as a parameter, or avoid the four narwhals ops that use module-level `duckdb.sql`. This is
   worth reporting upstream (*not done here*).

## Caveats

* Timings were taken on a shared, loaded machine. Absolute values moved by up to ~2x between runs
  (e.g. dask `native_collect` at 500k: 44 ms in run 1, 86 ms in run 2), but within-run ratios were
  stable in direction. Only DEV_GRID sizes (≤500k rows) were run. The large grid is left to the
  serial phase (`--grid large`).
* The polars "native collect" in F2/F3 wraps an in-memory frame, so it is near zero. F4 uses
  parquet scans for a realistic lazy source.
* dask always ran with `scheduler="synchronous"`. The threaded or distributed schedulers would
  change dask's lazy-vs-collect balance (*unmeasured*).
* Peak process RSS exceeded the ~450 MB guideline in the fallback runs: 589 MB in the first pd2
  run, 596 MB in the second, 453 MB on pd3. The stay-lazy DuckDB and ibis children reached
  577–662 MB. The baseline after importing pandas, polars, pyarrow, narwhals, dask, duckdb and
  ibis is already ~254 MB (measured). The large grid should run with one backend per process,
  which `x1_stay_lazy.py` already does.
* Order-preservation checks used 600 rows. "Preserved" cells are observations, not guarantees.
* modin used `MODIN_ENGINE=python` (serial). Behaviour with the ray or dask engines is
  *unmeasured*.
* ibis `create_table(..., overwrite=True)` raised a DuckDB `ParserException` with ibis 12 +
  duckdb 1.5.6. The scripts use fresh table names instead.
