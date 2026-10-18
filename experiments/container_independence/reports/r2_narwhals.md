# r2_narwhals: Narwhals deep-dive (documented semantics, adopters' experience, C1/C2 verdicts)

Workstream: **r2_narwhals** (prior art). Date: 2026-10-18.
Versions studied: narwhals **2.27.1** (source read from `$EXP_PY` site-packages), pandas 2.3.3 and 3.0.6,
polars 2.0.0, pyarrow 26, dask 2026.8.0, duckdb 1.5.6, ibis 12 (pandas-2 env only).

Evidence types used, and labelled throughout:

- **[src]**: narwhals 2.27.1 source, `file:line` relative to `site-packages/narwhals/`.
- **[doc]**: narwhals documentation (narwhals-dev.github.io/narwhals) or docstrings.
- **[gh]**: GitHub issues and PRs, read with the read-only `gh` CLI. Links are listed at the end.
- **[probe]**: my own script `scripts/r2_narwhals_probe.py`, run under both pandas versions. Raw output is in
  `results/r2_narwhals_probe_pd{2,3}_dev.{json,log}`.
- **unmeasured/expected**: reasoning only, not measured.

---

## 0. TL;DR

**C1** ("Polars lazy collects to Polars, but Dask goes to pandas and DuckDB to Arrow"): **TRUE as stated, but misleading as a criticism.**

- The defaults are documented and verified on every installed backend:
  - polars.LazyFrame → polars.DataFrame
  - dask → pandas
  - duckdb → pyarrow.Table
  - ibis → pyarrow.Table
  - pyspark → pyarrow ([src] only, not installed)
- `collect(backend=...)` can force the result to **pandas, polars or pyarrow, and nothing else**.
- dask, duckdb and ibis have no eager frame type of their own, so "collect to the original format" cannot exist for them.
- The original lazy type is **kept as long as you do not collect**: `LazyFrame.to_native()` returns the same native lazy type after operations, verified for 4 lazy backends.
- After a collect you can re-lazify explicitly with `DataFrame.lazy(backend="duckdb"|"dask"|"ibis"|"polars"|...)`.
- narwhals does **not** remember the original backend across `collect().lazy()`. Issue #3210 requests this; it is open and maintainers are lukewarm.
- Two practical gotchas:
  - dask → polars is collected *via pandas* ([src] `_dask/dataframe.py:149`, `pl.from_pandas(result)`).
  - duckdb's default collect requires pyarrow to be installed (marimo #9082).

**C2** ("No MultiIndex, columns only, must reset_index, that is a copy, the index came back as a RangeIndex and level names were gone"): **partly true.**

- *"Columns only / no MultiIndex in the API"*: **true.**
  - `nw.from_native(pd-multiindex frame).columns == ['c0']`; the index levels are invisible.
  - The docs say the index is not part of the public API, and docstrings advise new libraries not to rely on it.
- *"Must reset_index before grouping"*: **true per the documented API.**
  - Side observation: on the pandas backend, `group_by("h0")` on an *index level* happens to work, because narwhals forwards the key to `pandas.groupby`, which resolves level names.
  - That behaviour is undocumented and not portable, so sktime must not rely on it.
- *"That is a copy"*:
  - **True on pandas 2.3**: after `reset_index`, the value column no longer shares memory with the input.
  - **False on pandas 3.0**, where Copy-on-Write makes the value column share memory.
  - Either way the key columns are newly materialized from the MultiIndex codes.
  - Measured cost of `reset_index`, relative to one native `groupby(level=0)` iteration pass that does no estimator work: 0.06–0.76× (pd2) and 0.03–0.99× (pd3). The ratio falls with size.
- *"RangeIndex came back, level names were gone"*: **misleading.**
  - After `reset_index` the row index is a RangeIndex by definition, but the level names become **column names** (`h0`, `time`).
  - `set_index(names)` restores an **identical** MultiIndex: names, dtypes (datetime64 / period / int64) and `equals()` all verified on both pandas versions.
  - Names *are* lost in three cases: unnamed levels (they become `level_0`, `level_1`); narwhals' own `maybe_reset_index`, which is `reset_index(drop=True)` and **discards** the index; and the narwhals ops that return a fresh RangeIndex, namely `group_by().agg` and `join`.
  - Most other ops keep the MultiIndex and its names, including iteration sub-frames.
- An issue C2 does not mention, which matters more for sktime: **Period-typed time maps to narwhals dtype `Unknown`.**
  - On pandas it still works, because operations are delegated to pandas.
  - It is not portable: `to_polars()` gives an opaque `Extension('pandas.period')`, and `lazy('polars')` gives `Unknown`.

**Adopter pattern.** Every time-series or grouped-estimator adopter examined uses narwhals **at the boundaries** (ingest, validation, egress, relational aggregations) and keeps the per-series numeric core in numpy, pandas or xarray. This covers darts, Nixtla's utilsforecast/mlforecast, hierarchicalforecast and scikit-lego. Examples:

- Nixtla's shared `utilsforecast/processing.py` is a hand-written `isinstance(pd.DataFrame)` / polars dispatch layer: 49 functions, 39 pandas-isinstance branches, **0 narwhals**. It sits beside a numpy `GroupedArray` with `indptr`.
- mlforecast's narwhals refactor kept native code where "DatetimeIndex date-attribute access has no narwhals equivalent" and where "narwhals rewrite would regress this hot path".

**Per-call and grouped-iteration overhead [probe]** (DEV grid, noisy shared machine):

- narwhals pandas iteration: 1.0–1.3× (pd2) and 1.0–3.1× (pd3) slower than the current `groupby(level=0, sort=False)`.
- narwhals polars iteration: 1.4–2.4× (pd2 env) and 0.75–1.75× (pd3 env).
- Native polars `partition_by(as_dict=True, maintain_order=True)`: 0.66–1.17×. In other words, the narwhals wrapper costs 1.1–2.5× over the best native polars path.
- **narwhals pyarrow iteration is O(groups × rows)**: 1.4× at 10 series and **20–29× at 1,000 series** ([src] `_arrow/group_by.py:188-213`, one full-table filter per key).
- `from_native` alone costs 18–36 µs per call. A wrapped op on a 100-row frame costs 3–22× the native pandas op and 3–75× the native polars op.
- **Polars group order through narwhals is nondeterministic**: 20 distinct orders in 20 runs over 200 groups. narwhals' `group_by` has no `maintain_order` (#3309 open; plotly #5765 hit this).

**Window and grouped expressions per backend [probe].** Eight `expr.over("h0", order_by="time")` expressions were tested against a pandas reference:

| backend | passing |
|---|---|
| pandas | 8/8 |
| polars eager and lazy | 7/8 (ewm null mismatch, see below) |
| duckdb | 7/8 (no `ewm_mean`) |
| ibis | 6/8 |
| pyarrow | 1/8 |
| dask | 1/8 |

- The API-completeness doc table marks `cum_sum`, `shift`, `rolling_*` and `over` as supported on arrow and dask. The table is per-method; the **combination** `over(partition_by, order_by)` raises `NotImplementedError` on both.
- Found a cross-backend inconsistency: `ewm_mean(...).over(g, order_by=t)` returns carried-forward values at null positions on pandas, but nulls on polars. Plain `ewm_mean` matches across the two.

**Footprint and stability.**

- Zero required dependencies; about 2.4 MB installed; `import narwhals` pulls in only the standard library. Import took 0.14–0.27 s under heavy load.
- `narwhals.stable.v1` and `narwhals.stable.v2` are frozen with a "never change or remove" promise.
- 245 releases since 2024-02, 25 of them in the last 12 months.
- **scikit-learn ≥ 1.9.0 (2026-06-02) hard-requires `narwhals>=2.0.1`.** sktime allows `scikit-learn<1.10`, so fresh sktime installs already pull in narwhals transitively.

---

## 1. `LazyFrame.collect` semantics (C1)

### 1.1 Documented and source-level behaviour

- [doc][src] `dataframe.py:2496` `LazyFrame.collect(backend=None, **kwargs)`. The docstring (lines 2508-2513) lists the defaults:
  - `polars.LazyFrame -> polars.DataFrame`
  - `dask.DataFrame -> pandas.DataFrame`
  - `duckdb.PyRelation -> pyarrow.Table`
  - `pyspark.DataFrame -> pyarrow.Table`
- [src] `_utils.py:1661-1663`: `can_lazyframe_collect` allows only `{PANDAS, POLARS, PYARROW}`. Any other `backend` raises `ValueError: Unsupported backend value` (`dataframe.py:2562`).
- [src] Per-backend `collect` implementations:

| file | default target | other targets |
|---|---|---|
| `_polars/dataframe.py:758` | polars | pandas via `.to_pandas()`, arrow via `.to_arrow()` |
| `_dask/dataframe.py:127` | pandas (`native.compute()`) | polars = `pl.from_pandas(compute())` (line 149, **two hops**); arrow = `pa.Table.from_pandas` |
| `_duckdb/dataframe.py:133` | arrow (`to_arrow_table()`; needs pyarrow) | pandas `.df()`, polars `.pl()` |
| `_ibis/dataframe.py:117` | arrow (`to_pyarrow()`) | pandas, polars |
| `_spark_like/dataframe.py:243` | arrow (`toArrow`) | pandas (`toPandas`); sqlframe uses the same class |

- [src] `_utils.py:1666-1676`: `DataFrame.lazy(backend)` allows DASK, DUCKDB, IBIS, POLARS, PYSPARK, PYSPARK_CONNECT and SQLFRAME. pandas and pyarrow are **not** lazy targets.
- [doc] `kwargs` are forwarded to the native collect (`polars.LazyFrame.collect`, `dask.compute`). This is the escape hatch for engine and streaming options.

### 1.2 Verified [probe, section A]

Identical on pandas 2.3 and 3.0 (ibis exists only in the pd2 env):

| native lazy input | `from_native` → | `.to_native()` after `with_columns` | `collect()` default | `collect('pandas' / 'polars' / 'pyarrow')` | `collect('dask' / 'duckdb' / 'ibis')` | `stable.v1.from_native` |
|---|---|---|---|---|---|---|
| polars.LazyFrame | LazyFrame | polars.LazyFrame | polars.DataFrame | ok / ok / ok | ValueError | LazyFrame |
| dask.DataFrame | LazyFrame | dask.DataFrame | **pandas.DataFrame** | ok / ok / ok | ValueError | LazyFrame |
| duckdb relation | LazyFrame | DuckDBPyRelation | **pyarrow.Table** | ok / ok / ok | ValueError | **DataFrame** (interchange-level) |
| ibis.Table | LazyFrame | ibis.Table | **pyarrow.Table** | ok / ok / ok | ValueError | **DataFrame** (interchange-level) |

`pandas DataFrame.lazy(b)`:

- `None` → pandas (API restricted to lazy ops)
- `polars` → polars.LazyFrame
- `dask` → dask.DataFrame
- `duckdb` → DuckDBPyRelation
- `ibis` → ibis.Table
- `pandas` → ValueError
- `pyarrow` → ValueError

Side note: through ibis, a pandas `datetime64[ns]` column reports as `Datetime(time_unit='s')`, so time-unit fidelity varies by backend ([probe] B, ibis `collect_schema`).

### 1.3 Design discussion upstream [gh]

- **#1479** ("collect for lazy-only libraries", 2024-12, closed): maintainers chose "each lazy backend goes to its natural eager backend". For Polars and Dask that target is "obvious"; DuckDB was an opinionated choice of pyarrow. This produced the `collect(backend=...)` argument.
- **#3210** ("Track backend between lazy->collect->lazy", open since 2025-10): a user hits duckdb → collect → arrow → `.lazy()` staying on arrow. The workaround is to capture the backend up front and call `.lazy(backend=...)`. One maintainer is "not sure"; another opposes `lazy('pandas')`.
- **#2193 / #3812** (`DataFrame.with_backend`, open with an RFC PR): there is still no generic eager-to-eager conversion. The documented workaround is `.lazy().collect(backend=...)`.

### 1.4 Verdict on C1

The defaults are exactly as claimed, and they can be overridden only to pandas, polars or pyarrow. "Respecting the original format" is possible only up to the point of materialization:

- Keep the frame as a `nw.LazyFrame` and call `to_native()`, which keeps the native lazy type.
- Or re-lazify after collecting, with `.lazy(backend=original)`, which sktime would have to track itself.

This is inherent to the backends, not a narwhals decision: dask's `compute()` returns pandas, and DuckDB, Ibis and Spark have no eager frame. **Any** design that runs sktime's per-series Python fit loop on a lazy input must materialize, so sktime needs an explicit policy for "materialize to what" and "return as what". narwhals provides the mechanisms but not the policy.

Accept the defaults as a fact, and pass `backend=` explicitly. Avoid `collect('polars')` on dask, which goes through pandas.

---

## 2. pandas index policy (C2)

### 2.1 Documented policy [doc]

From `concepts/pandas_index.md`:

- The index is **not public API**.
- narwhals "will preserve your index for common dataframe operations".
- `DataFrame.group_by` does **not** preserve it, because overlapping index and column names raise.
- For binary operations, the left-hand side's index is kept and the right-hand side is inserted **positionally**, polars-style. There is no pandas alignment.

The page does **not** mention MultiIndex.

The utilities `maybe_get_index`, `maybe_set_index`, `maybe_reset_index` and `maybe_align_index` ([src] `_utils.py:811-1060`) all carry the same note: they exist for backwards compatibility, and new libraries are encouraged not to rely on the index. All are no-ops (or a length check) for non-pandas-like inputs.

- **`maybe_reset_index` = `reset_index(drop=True)`** ([src] `_utils.py:1022-1050`). It **discards** the index and does not move levels into columns. darts asked for a keep-index variant (#2056); the maintainers declined and suggested special-casing pandas.
- **`maybe_set_index(df, index=list)` is buggy:** a plain list is treated as column keys (#2125, open).
- **`maybe_get_index` / `maybe_set_index` asymmetry:** `maybe_set_index` does not accept `maybe_get_index`'s output (#3693, open). This is the "copy the index from one object to another" use case that river and BayBE hit (#3864).
- [src] `_pandas_like/group_by.py:274-278`: when an index level name collides with a column, the pandas group_by does `reset_index(drop=True)`. This is the fix for plotly #5005 / narwhals #1907.
- [src] `_pandas_like/group_by.py:316-323`: `agg` results are `reset_index()`-ed, so the keys become columns and a RangeIndex is returned.
- [src] `_pandas_like/group_by.py:384-394`: `__iter__` slices the native pandas groupby, so sub-frames keep the original index rows.

### 2.2 Verified [probe, section C]

Input: `make_panel(n_series=4, n_time=5, n_levels=1)`, a MultiIndex `(h0, time)` with time as datetime, period or int. Results were identical for all three time kinds and both pandas versions.

| op on `nw.from_native(X)` | resulting native index |
|---|---|
| `.columns` | `['c0']` (levels invisible) |
| with_columns, select (names and expressions), filter, sort, head, rename, cast, drop_nulls, with_row_index, concat (vertical), unique | **MultiIndex, names `['h0','time']` kept** |
| iterate `group_by("k")` over a column: sub-frames | **MultiIndex `['h0','time']` kept**; key yielded as tuple `('s0',)` |
| `group_by(col).agg(...)` | RangeIndex (names gone, by design) |
| `join(...)` | RangeIndex |
| `nw.maybe_reset_index` | RangeIndex, index **dropped** |
| `group_by("h0")` where `h0` is an *index level* (pandas only) | **works** (4 groups). Undocumented: pandas resolves level names. Not portable. |

The `reset_index` round trip (the C2 path):

- `X.reset_index()` gives columns `['h0','time','c0']` on a RangeIndex. The names are now **column names**.
- narwhals schema on that frame:
  - datetime → `h0: String, time: Datetime(ns), c0: Float64` on pd2; `Datetime(us)` on pd3
  - int → `time: Int64`
  - **period → `time: Unknown`**
- `nw.group_by("h0").agg(...)` works.
- `to_native().set_index(['h0','time'])` gives `index.equals(X.index)` **True**, with names restored, for all three time kinds and both pandas versions. The time level dtype is preserved: `datetime64[ns]`, `period[D]`, `int64`.
- Unnamed levels become `level_0`, `level_1` on `reset_index`.

Probing a **Period** time column on the pandas backend:

- Supported: schema `Unknown`; `sort`, `shift().over(h0, order_by=time)`, `dt.year()`, `max()`, and filter-compare all work (delegated to pandas).
- Not portable: `to_polars()` gives `Extension('pandas.period', Int64, {"freq":"D"})`; `to_arrow()` gives `extension<pandas.period>`; `lazy('polars')` gives `Unknown`.

**Copy and cost of `reset_index`.** Measured by `bench()` min-of-repeats against a native `X.groupby(level=0, sort=False)` iteration pass that does no estimator work:

| env | n_series × n_time | value col shares memory | reset_index | reset+set_index | groupby-level iteration | reset / iter | reset+set / iter |
|---|---|---|---|---|---|---|---|
| pd 2.3.3 | 10×100 | **False** | 1.0 ms | 2.8 ms | 1.2 ms | 0.76 | 2.23 |
| pd 2.3.3 | 100×100 | False | 1.2 ms | 3.0 ms | 4.2 ms | 0.30 | 0.72 |
| pd 2.3.3 | 1000×100 | False | 4.0 ms | 12.5 ms | 34.5 ms | 0.12 | 0.36 |
| pd 2.3.3 | 10000×50 | False | 20.8 ms | 107 ms | 329 ms | 0.06 | 0.33 |
| pd 3.0.6 | 10×100 | **True** (CoW) | 2.0 ms | 4.6 ms | 2.0 ms | 0.99 | 2.31 |
| pd 3.0.6 | 100×100 | True | 1.9 ms | 4.8 ms | 5.8 ms | 0.33 | 0.83 |
| pd 3.0.6 | 1000×100 | True | 3.6 ms | 12.0 ms | 42.3 ms | 0.09 | 0.28 |
| pd 3.0.6 | 10000×50 | True | 14.2 ms | 52.3 ms | 506 ms | 0.03 | 0.10 |

Relative to fitting any real forecaster per series, the one-off `reset_index` is negligible at 100+ series (unmeasured/expected). The current pipeline already pays one `convert_to` to the pandas multiindex.

### 2.3 Upstream experiences with index-reliant code [gh]

- **plotly #5005** (v6.0.0): `color=df.index` broke with "'Type' is both an index level and a column label". narwhals fixed this in 1.24.2 by dropping the index inside `group_by` (#1907).
- **plotly #4917**: scatter_matrix lost colour with a DatetimeIndex in the v6 release candidate; fixed before release.
- **darts #2056 / #2689**: going from a `TimeSeries` (time held as `pd.DatetimeIndex`) to "any backend" loses the index on non-pandas backends. The maintainers' answer: a pandas-only `time_as_index` path, and a time **column** everywhere else.
- **fairlearn #1552** (open): public outputs (`MetricFrame.by_group`, a Series with a structured MultiIndex) cannot be expressed in polars or arrow. Options discussed: (1) change outputs to "compatible dataframes" (an API break); (2) input-type-dependent outputs (rejected as causing silent failures); (3) and (4) flags or `_as_df` variants. Maintainers lean to option (1). **This is the same dilemma sktime faces with `pd-multiindex` outputs.**
- **scikit-learn #33958** (META "Use narwhals in codebase"): includes the TODO "Remove use of `iloc`", i.e. move towards an index-free design.

### 2.4 Verdict on C2

- *No MultiIndex in the API*: true.
- *Must move keys into columns*: true, for portable code.
- *It is a copy*: true on pandas < 3, and the measured cost is small relative to the work being vectorized. On pandas 3 (CoW) the value columns are not copied.
- *Level names were gone*: **not inherent.** They survive as column names and round-trip exactly. They are lost only via `maybe_reset_index` (drop), unnamed levels, or the documented index-resetting ops (`group_by().agg`, `join`). The dev's observation is consistent with having used `maybe_reset_index` or an `agg` result.

The real costs for sktime are elsewhere:

1. sktime's **public** `pd-multiindex` output contract (the fairlearn dilemma).
2. **PeriodIndex** semantics, which map to `Unknown`.
3. Frequency (`freq`) metadata, which has no narwhals representation.

The deterministic ordering problem is covered in §5.

---

## 3. Stability, footprint and overhead

### 3.1 Stable API and backwards-compatibility policy [doc] (backcompat page)

- `narwhals.stable.v1` and `narwhals.stable.v2` are frozen, with a promise never to change or remove a public function you use. Breaking changes go to main and later to v3. v1 and v2 are maintained indefinitely.
- Exceptions: unambiguous bugs, radical backend changes, more precise typing, anything marked "unstable", minimum-backend-version bumps, and dropping end-of-life Pythons.
- Notable v1 vs main/v2 differences relevant here:
  - v1 `from_native(duckdb | ibis)` returns an interchange-level `DataFrame`; v2 returns a `LazyFrame`. Verified [probe A].
  - Order-dependent expressions on a LazyFrame require `over(order_by=...)` from v2 on.
  - `LazyFrame.with_row_index` requires `order_by`.
  - `native_namespace` is replaced by `backend`.
- Release cadence (PyPI JSON, fetched 2026-10-18): **245 releases** from 0.1.6 (2024-02-20) to 2.27.1 (2026-10-10). That is 163 in 2024, 65 in 2025 and 17 in 2026, with **25 in the last 12 months**. 2.0.0 shipped on 2025-07-28.
- Churn propagates: polars 2.0 broke `cat.get_categories` in narwhals (#3895), and downstream darts (#3235) and pointblank (#428). Any abstraction layer, including a home-grown dispatcher, carries this lag risk.

### 3.2 Dependency footprint

- `Requires-Dist` (2.27.1 METADATA): **every dependency is an extra.** There are no required runtime dependencies. Python ≥ 3.10.
- Installed package directory: about 2.4 MB (`du -sh`, including `__pycache__`).
- [probe G] `import narwhals` imports **none** of pandas, polars, pyarrow, numpy, dask or duckdb; 152 modules in total, all stdlib plus narwhals.
- Import time, under heavy concurrent load (about 14 agents), so treat as an upper bound:
  - `-X importtime` cumulative: 0.141–0.158 s over 4 runs.
  - Subprocess min-of-5 wall overhead: 0.25 s (pd2 env), 0.27 s (pd3 env).
- **scikit-learn 1.9.0 (uploaded 2026-06-02) and 1.9.1 hard-require `narwhals>=2.0.1`.** sklearn 1.8.0 did not. Source: sklearn PR #31127, "ENH add narwhals as dependency", merged 2026-05-07, which replaced the `__dataframe__` interchange protocol. sktime's `pyproject.toml` allows `scikit-learn>=0.24,<1.10.0`, so any environment resolving sklearn 1.9.x already has narwhals. Making it a direct sktime dependency would add nothing new there. Users on older sklearn would gain one small pure-Python package.

### 3.3 Overhead claims [doc] vs measured [probe]

What is claimed:

- The overhead page says pandas-through-narwhals overhead on TPC-H is "negligible", sometimes negative, attributed to avoiding "unnecessary copies and index resets".
- The numbers exist only in a chart, guarded by a CI test, `tpch/tests/overhead_test.py`.
- Lazy backends pay schema introspection for some operations (joins, selectors, `nth`, vertical concat, `unique`). This is cached, but "not free" on remote data.
- The Plotly v6 blog claims 3x–14x faster plots for polars and pyarrow inputs on "large datasets" (Apple M1, sizes unspecified), and that pandas is the same or slightly faster. These are vendor numbers, not independently verified.

Measured: **per-call overhead on tiny inputs** [probe F], µs/call as min of 5 × 2000 calls on a 100-row frame or series. These are a hot-loop proxy for a forecaster that runs once per series.

| operation | pd 2.3 env native | pd 2.3 env narwhals | ratio | pd 3.0 env native | pd 3.0 env narwhals | ratio |
|---|---|---|---|---|---|---|
| `nw.from_native(pd df)` only | – | 27.7 | – | – | 35.5 | – |
| `nw.from_native(pl df)` only | – | 17.5 | – | – | 24.4 | – |
| pandas Series last value (`iloc[-1]` vs `item(-1)`) | 4.0 | 40.0 | 9.9× | 6.5 | 55.3 | 8.5× |
| pandas Series mean | 17.0 | 61.1 | 3.6× | 21.6 | 71.0 | 3.3× |
| pandas df shift (`with_columns` + `to_native`) | 40.1 | 482.8 | 12.0× | 52.0 | 1158 | 22.3× |
| polars Series mean | 0.3 | 21.7 | 74.9× | 0.3 | 20.9 | 68.1× |
| polars df `with_columns(shift)` | 118.9 | 348.5 | 2.9× | 139.2 | 476.6 | 3.4× |

Interpretation:

- A simple narwhals call (`from_native` plus a scalar reduction or item access) costs about **20–50 µs of fixed overhead**. The added cost over native is 36–49 µs for pandas and about 21 µs for polars.
- A pandas `with_columns(...).to_native()` round trip costs **0.44–1.1 ms**, against 40–52 µs natively.
- For a vectorized loop over 10k series with a few simple wrapped ops per series, that is about 0.2–1.5 s per `fit` or `predict` (unmeasured extrapolation: 10,000 × 20–50 µs × 1–3 ops).
- This matches mlforecast PR #688: its recursive predict hot loop was "sensitive to per-call narwhals overhead" and had to be restructured (local A/B "~78 ms → ~30 ms").
- Takeaway: **narwhals should be applied once per frame (VectorizedDF level), not once per series inside 100s of forecasters.**

---

## 4. Adopters and their experiences [gh]

The ecosystem page lists:

- altair, bokeh, **darts**, fairlearn, formulaic, gt-extras, **hierarchicalforecast**, lightgbm, marimo, metalearners, mosaic, pandera, panel-graphic-walker, plotly, pointblank, pymarginaleffects, pyreadr, pyreadstat, py-shiny, pysummaries, rio, **scikit-learn**, **scikit-lego**, scikit-playtime, tabmat, tea-tasting, **timebasedcv**, tubular, Validoopsie, vegafusion, wimsey.

Per-adopter evidence:

- **plotly (v6)**: narwhals throughout plotly express.
  - Claims: see §3.3.
  - Index issues: #5005 (index name collides with column) and #4917 (DatetimeIndex).
  - Ordering: #5765 (2026-09). Sunburst, treemap and icicle sector order differed on every run for **polars** input, because narwhals `group_by` gives no order guarantee and has no `maintain_order`. The fix adds a temporary row-index column, takes the min per group and sorts. #5769 will switch to `maintain_order=True` when narwhals #3309 lands. The narwhals maintainer said in that thread (2026-09-21) that `maintain_order` should be added to *eager* `group_by` only.
  - Other v6 regressions: #5099 (y-axis order), #5033 (category orders), #4999 (`px.pie` ColumnNotFoundError), #5571 (sunburst with pyarrow-backed pandas 3). All fixed or closed.
- **scikit-learn**: narwhals became a **hard dependency** in 1.9.0 (PR #31127, replacing `__dataframe__`).
  - META #33958 lists the goals: simplify internals, make code faster ("narwhals expression API instead of array operations, e.g. group_by(..).agg(..)"), possibly narwhals-in/narwhals-out, and "Remove use of `iloc`".
- **darts** (forecasting):
  - PR #2661 put `from_dataframe` on narwhals. darts' own benchmark over 10 runs: 9.86 s vs 10.97 s for the old pandas-only path on pandas inputs, i.e. **10% faster**. An early version was slow because of `to_numpy()`.
  - PR #2766 put `from_group_dataframe` on narwhals. It needed a narwhals fix for joblib pickling of narwhals-wrapped polars frames (narwhals #2450), plus "a last change to get the same efficiency as we had it before for pandas".
  - Current code (darts master `dcdde24`, `darts/timeseries.py` `from_group_dataframe`):
    - `eager_only=True`.
    - The pandas index is accepted **only** for pandas; other backends require `time_col`.
    - A pandas-specific fast path moves the time column into the index.
    - Comment: "not all backends maintain the order when grouping; need to sort the groups in the end for reproducibility".
  - narwhals is used at the boundary; the internals stay xarray with a pandas time index.
- **Nixtla**:
  - `hierarchicalforecast` added polars support via narwhals (#305).
  - `utilsforecast` (main `09792af`) has `narwhals>=2.0` **and** `pandas>=1.5` as hard dependencies. narwhals appears in `losses.py` (64 mentions) and `evaluation.py` (31), i.e. frame-level group-by aggregations. `processing.py` (49 functions, 39 `isinstance(pd.DataFrame/Series)` branches), `grouped_array.py` (numpy + `indptr`), `feature_engineering.py` and `preprocessing.py` have **0** narwhals mentions.
  - `mlforecast` PR #688 ("Narwhals Refactor", merged 2026-07): unified the pandas and polars branches via narwhals. It *intentionally kept* native code; `core.py` comments include "DatetimeIndex date-attribute access has no narwhals equivalent" and "pandas positional-map fast path; narwhals rewrite would regress this hot path". In `core.py` (main `e3f0573`) there are 29 narwhals mentions vs 73 `utilsforecast.processing` calls and 30 GroupedArray/indptr mentions.
  - `mlforecast` PR #723 (closed, superseded): an experimental narwhals-expression engine for pooled lag transforms was 7–44× faster. The speedup came from an algorithmic rewrite into dataframe expressions, not from narwhals per se. The author retracted "line count is a win": about 2,850 new lines replaced about 2,127.
- **scikit-lego** (grouped estimators, the closest analogue to sktime vectorization). `GroupedPredictor` (main `f329086`):
  - Uses `narwhals.stable.v1`, `eager_only=True`.
  - Iterates `frame.group_by(groups)` and passes `nw.to_native(X_grp...)`, the **native slice**, to each sub-estimator.
  - Restores order with an explicit `__sklego_index__` row-number column.
  - **Reassembles predictions with `pd.concat(...).sort_index()`**, so pandas is still required.
- **fairlearn**: the MetricFrame narwhals migration (#1440, #1522) is blocked on the public-API index question (#1552, open). See §2.3.
- **great-tables** (#737, RFC): the maintainer prefers narwhals **confined behind the existing small internal dispatch interface** (`_tbl_data.py`, "15-ish" generic functions) over narwhals spread through the codebase. The reason given: implementing great-tables over a custom frame needs only those ~15 functions, whereas baking narwhals in everywhere is "a net increase in complexity" even if it cuts about 500 lines.
- **altair / marimo / pointblank**:
  - marimo #9082 (open): DuckDB relations need pyarrow just to render, because of the default collect target (C1).
  - pointblank #39: schema checks used narwhals dtypes, not pandas dtypes, which caused confusing mismatches. This is the same class of problem as Period → `Unknown`.
  - altair #3921: timezone not passed through for polars. Tracked as an altair-side limitation.
- **formulaic / Hamilton / py-shiny / vegafusion / tubular**: no index or time-series-specific issues found that bear on sktime. formulaic's author (Netflix) drove the `with_backend` request (#2193).
- **sktime itself**:
  - #9680 ("support polars in vectorization", open). fkiraly endorsed a **mtype-based dispatcher layer** in VectorizedDF, with inner estimators handling conversion per slice. He also suggested making an estimator, e.g. NaiveForecaster, polars-compatible "via narwhals or directly".
  - PR #10802 (stateless VectorizedDF, merged): the author asked whether narwhals could become a core dependency (2026-08-13). There is no answer in the thread. fkiraly suggested externalizing the grouping as an argument (e.g. a polars column).
  - narwhals #62 (2024, "potential use cases") lists sktime #5423, #5370 and #6091.

---

## 5. Grouped iteration, `over()`, rolling/ewm per backend (item 5)

### 5.1 API surface [src][probe B]

- `GroupBy` (eager) public methods: `agg` and `__iter__`. **No** `map_groups` / `apply`, and **no** `maintain_order` (signature `group_by(*keys, drop_null_keys=False)`).
- `LazyGroupBy` has only `agg`, and is **not iterable** (`TypeError: 'LazyGroupBy' object is not iterable`).
- `DataFrame`-only methods (26): clone, estimated_size, from_arrow/dict/dicts/numpy, get_column, is_duplicated, is_empty, is_unique, item, iter_columns, iter_rows, null_count, pivot, row, rows, sample, shape, to_arrow/dict/numpy/pandas/polars, write_csv/parquet.
- `LazyFrame`-only: `collect`, `sink_parquet`. `lf[...]`, `len(lf)`, `lf.shape`, `lf.to_numpy()` and `lf.iter_rows()` all raise.
- Neither class has `partition_by`.
- Missing time-series ops: `Expr.interpolate`, `Expr.rolling_quantile`, `DataFrame.rolling`, `upsample` and `group_by_dynamic` do not exist. Open requests: #1254 (rolling/ewm), #2544 (`DataFrame.rolling`), #3363 (`order_by` for `rolling_*`).
- Order-dependent expressions on a LazyFrame without `over(order_by=...)` raise `InvalidOperationError: Order-dependent expressions are not supported for use in LazyFrame` ([src] `dataframe.py:2459`). Verified on polars-lazy, dask, duckdb and ibis.

### 5.2 Grouped window expressions per backend [probe D]

Setup: shuffled long panel, 3 series × 8 steps, one null. Results are compared to a pandas groupby reference after sorting by `(h0, time)`. All expressions except the first are `.over("h0", order_by="time")`.

| expression | pandas | polars | polars lazy | pyarrow | dask | duckdb | ibis |
|---|---|---|---|---|---|---|---|
| `mean().over("h0")` | ok | ok | ok | ok | ok | ok | ok |
| `cum_sum()` | ok | ok | ok | NotImpl¹ | NotImpl² | ok | ok |
| `shift(1)` | ok | ok | ok | NotImpl¹ | NotImpl² | ok | ok |
| `diff()` | ok | ok | ok | NotImpl¹ | NotImpl² | ok | ok |
| `rolling_mean(3)` | ok | ok | ok | NotImpl¹ | NotImpl² | ok | ok |
| `rolling_std(3)` | ok | ok | ok | NotImpl¹ | NotImpl² | ok | ok |
| `ewm_mean(alpha=.5)` | ok | **≠ pandas at null**³ | **≠**³ | NotImpl | NotImpl | NotImpl | NotImpl |
| `fill_null(strategy="forward")` | ok | ok | ok | NotImpl¹ | NotImpl² | ok | NotImpl⁴ |

Footnotes:

1. "Only elementary aggregations are supported for `.over` in PyArrow backend when `partition_by` is specified." ([src] `_arrow/expr.py:148`)
2. "`over` with `order_by` is not yet supported in Dask." ([src] `_dask/expr.py:578`)
3. [probe H] Plain `ewm_mean` gives identical results on pandas and polars, with null at null positions. With `.over(g, order_by=t)`:
   - pandas → `[1.0, 1.0, 2.6, 3.4615, 2.0, 2.0, 2.0, 1.27]`
   - polars → `[1.0, None, 2.6, 3.4615, 2.0, 2.0, None, 1.27]`

   So the narwhals pandas `over` path does not re-mask nulls. No upstream issue was found (searched "ewm over", "ewm null").
4. "`strategy` is not supported for the Ibis backend".

The docs' API-completeness table (`api-completeness/expr/`) shows ✅ for `over`, `cum_sum`, `shift`, `diff`, `rolling_*` and `fill_null` on **arrow and dask**. That table is per-method; it does **not** imply that `.over(partition_by, order_by)` compositions work. `ewm_mean` is ✅ only for pandas-like and polars, and `map_batches` only for arrow, pandas-like and polars.

### 5.3 Grouped iteration order and cost [probe E]

**Order.** 200 groups in a first-appearance order, iterated 20 times through narwhals:

| backend | distinct orders over 20 runs | runs in first-appearance order |
|---|---|---|
| pandas | 1 | 20/20 (narwhals uses `sort=False`) |
| pyarrow | 1 | 20/20 |
| polars | **20** | **0/20** |

Keys are yielded as tuples, e.g. `('s0',)`. sktime's `VectorizedDF` relies on `sort=False` order for `reconstruct`, so on polars it would need its own ordering. Options: sort the unique keys, as darts does; add a row-index column, as plotly and scikit-lego do; or use native `maintain_order` / `partition_by(maintain_order=True)`, as utilsforecast's `group_by(..., maintain_order)` helper does. The cost of those workarounds was not measured.

**Cost.** Time for one full pass over all groups with no work per group, as a ratio to the current `X.groupby(level=0, sort=False)` on the pandas MultiIndex:

| env | n_series × n_time | current (abs) | pandas `groupby(col)` | nw[pandas] | polars `group_by(maintain_order)` | polars `partition_by(as_dict)` | nw[polars] | nw[pyarrow] |
|---|---|---|---|---|---|---|---|---|
| pd2.3 | 10×100 | 1.4 ms | 0.83 | 1.11 | 1.55 | 0.71 | 1.42 | 1.36 |
| pd2.3 | 100×100 | 4.3 ms | 0.74 | 1.21 | 1.69 | 0.87 | 2.01 | 6.69 |
| pd2.3 | 1000×100 | 33.2 ms | 0.72 | 1.33 | 1.86 | 1.15 | 2.30 | **28.9** |
| pd2.3 | 10000×50 | 317 ms | 0.71 | 1.33 | 1.73 | 0.95 | 2.40 | skipped |
| pd3.0 | 10×100 | 2.4 ms | 0.86 | 1.00 | 0.64 | 0.67 | 0.75 | 1.47 |
| pd3.0 | 100×100 | 5.9 ms | 1.52 | 3.05 | 1.40 | 0.79 | 1.46 | 6.12 |
| pd3.0 | 1000×100 | 49.7 ms | 0.93 | 2.00 | 1.19 | 0.66 | 1.49 | **20.2** |
| pd3.0 | 10000×50 | 486 ms | 1.03 | 2.53 | 1.23 | 1.17 | 1.75 | skipped |

- nw[pyarrow] was skipped above 1,000 series because the cost is quadratic.
- In the pd3 env, the polars and pyarrow columns were measured in the same process as the pd3 baseline.
- Ratios are noisy, since about 14 processes shared the machine, but the direction is stable across both runs:
  - narwhals adds 1.0–3× over native pandas iteration.
  - The narwhals polars wrapper is 1.1–2.5× slower than native `partition_by`.
  - narwhals pyarrow iteration is unusable for panels with many series.
- All of these are tiny compared with per-series model fitting (unmeasured/expected). Iteration overhead matters mainly for cheap forecasters such as Naive and per-series transformers.

---

## 6. Implications for sktime (for the synthesis phase)

What is reasonably expected, and what is measured:

1. **Narwhals as the VectorizedDF backbone (eager only)** is feasible for pandas and polars.
   - Iteration preserves the native slice type, as scikit-lego does: inner forecasters receive a native pandas or polars slice.
   - Required additions:
     - (a) explicit group ordering for polars;
     - (b) a schema object holding time dtype/freq outside narwhals, because Period maps to `Unknown`;
     - (c) a pandas-specific egress path that rebuilds `pd-multiindex` outputs via `set_index`, which was verified to be lossless;
     - (d) avoid narwhals' pyarrow iteration, or use pyarrow-native `sort_indices` + slicing.
   - Lazy inputs cannot be iterated. They must either `collect(backend=...)` with an explicit policy (C1), or be expressed as `over()` expressions.
2. **Narwhals inside 100s of forecasters is the wrong granularity.**
   - Measured per-call overhead is 20–50 µs for simple ops and 0.4–1.1 ms for a pandas `with_columns` round trip.
   - Window-op coverage is uneven across backends (pyarrow and dask fail on grouped ordered windows; ewm is pandas/polars only).
   - The ewm null semantics differ between backends.
   - None of the time-series adopters do this. They all keep a numpy, pandas or xarray core.
3. A **hybrid architecture** that the prior art converges on:
   - narwhals at ingest and egress, plus frame-level relational ops (validation, `group_by.agg` for metrics, joins, cutoffs and filters);
   - a **small internal panel-kernel protocol**, as great-tables does with ~15 functions and Nixtla's processing.py does with ~49;
   - a **numpy grouped-array** representation (values + `indptr` + key table) for the per-series loop and for vectorizable "global" forecasters.

   This gives backend independence without every forecaster seeing a frame API. Native fast paths can be added where narwhals lacks something: `maintain_order`, Period, `partition_by`, `freq`.
4. **Expression-compilable forecasters** (Naive last/mean/drift, seasonal naive via `shift().over(...)`, rolling-mean forecasters) can run natively and lazily on polars, duckdb and ibis through narwhals `over(partition_by, order_by)`. pyarrow and dask cannot, as of 2.27.1. This is a separate opt-in tag/capability path, not a replacement for `_fit` / `_predict`.
5. **Public output contract.** narwhals cannot represent sktime's `pd-multiindex` outputs for non-pandas inputs. sktime must choose from fairlearn's options: return "compatible long frames" for non-pandas inputs, or keep pandas outputs. Option (2) in fairlearn's list, input-dependent output types, is exactly what native-container support means, and fairlearn's maintainers flagged it as a source of silent failures. This needs an explicit contract and tests.
6. **Dependency risk is low.** narwhals is already a transitive dependency via scikit-learn ≥ 1.9, has zero required dependencies, and has a frozen `stable.v2`. The risk is semantic gaps and release churn, not install weight.

---

## 7. Methods and commands

All commands were run from the worktree root. The environment paths are those given in the README.

```
# main probe (all sections A-H), pandas 2.3.3 and 3.0.6 envs
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/r2_narwhals_probe.py --grid dev
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/r2_narwhals_probe.py --grid dev
#   -> results/r2_narwhals_probe_pd2_dev.{json,log}, results/r2_narwhals_probe_pd3_dev.{json,log}
#   --grid large switches sections C/E to LARGE_GRID for the serial benchmark phase (not run here)

# import time (4 runs, last line = cumulative narwhals)
$EXP_PY -X importtime -c "import narwhals" 2>&1 | tail -1      # 140834 / 151694 / 151826 / 158368 us

# dependency metadata
grep Requires-Dist $SITE/narwhals-2.27.1.dist-info/METADATA     # all extras
$EXP_PY -c "import importlib.metadata as m; print(m.requires('scikit-learn'))"   # includes narwhals>=2.0.1
curl -s https://pypi.org/pypi/scikit-learn/{1.8.0,1.9.0}/json   # 1.9.0 adds narwhals>=2.0.1; 1.9.0 uploaded 2026-06-02
curl -s https://pypi.org/pypi/narwhals/json                     # 245 releases, 25 in last 12 months

# GitHub evidence (read-only)
gh search issues --repo narwhals-dev/narwhals "<term>" ... ; gh issue view <n> --repo ...; gh pr view <n> --repo ...
gh api -H "Accept: application/vnd.github.raw" repos/<owner>/<repo>/contents/<path>   # darts, sklego, utilsforecast, mlforecast sources
```

Probe sections:

- **A**: collect / to_native / lazy per backend
- **B**: eager vs lazy API surface and lazy-unsupported ops
- **C**: pandas index policy; reset_index round trip, copy and cost; Period column
- **D**: 8 grouped window expressions × 7 backends vs a pandas reference
- **E**: group order determinism (200 groups × 20 runs) and iteration cost over DEV_GRID
- **F**: per-call overhead, min of 5 × 2000 calls
- **G**: import footprint
- **H**: ewm null semantics, plain vs over

Timing used `common.bench` (min of repeats, after warmup). All comparisons are interleaved in the same process. Peak RSS was ≤ 420 MB.

## 8. Caveats

- Timings were taken while about 14 other agents were running, so absolute numbers are noisy. Use the ratios, and the direction of each ratio, which was consistent across two independent runs and two pandas versions. Section E ratios in the pd3 env vary more.
- Section E measures iteration only, with no estimator work. It overstates the relative impact on real forecasters.
- Section F uses 100-row inputs. The narwhals overhead is fixed per call, so it shrinks relative to longer series.
- pyspark, sqlframe, cuDF and modin were not exercised. Their collect defaults come from source only. ibis is missing from the pandas-3 env.
- GitHub statements reflect thread state on 2026-10-18. Open issues (#3309 `maintain_order`, #3210, #2193/#3557 `with_backend`, #2125, #3693) may change the picture.
- Adopter performance claims (plotly 3–14×, darts −10%, mlforecast 78→30 ms, mlforecast #723 7–44×) are the adopters' own numbers and were not reproduced.
- The `group_by("h0")`-on-index-level behaviour is an undocumented pandas passthrough. It may change without notice; per the backcompat policy, bug fixes are not breaking changes.

## 9. Sources

- narwhals docs:
  - https://narwhals-dev.github.io/narwhals/concepts/pandas_index/
  - https://narwhals-dev.github.io/narwhals/backcompat/
  - https://narwhals-dev.github.io/narwhals/overhead/
  - https://narwhals-dev.github.io/narwhals/concepts/order_dependence/
  - https://narwhals-dev.github.io/narwhals/ecosystem/
  - https://narwhals-dev.github.io/narwhals/api-completeness/expr/
  - https://narwhals-dev.github.io/narwhals/extending/
- narwhals issues (https://github.com/narwhals-dev/narwhals/issues/N): #62, #1050, #1254, #1479, #1907, #2056, #2125, #2193, #2450, #2544, #3210, #3309, #3363, #3693, #3812, #3864, #3895
- plotly (https://github.com/plotly/plotly.py/issues/N): #4917, #4999, #5005, #5033, #5099, #5571, #5765, #5769
- Plotly blog: https://plotly.com/blog/chart-smarter-not-harder-universal-dataframe-support/
- scikit-learn: PR #31127, issue #33958
- darts: PR #2661, #2701, #2766; issues #2645, #2689, #2730, #3235; source `darts/timeseries.py` at `dcdde2414f07f13b3bbb095428d9c6cc41492588`
- Nixtla:
  - utilsforecast at `09792afc64d75300f74a4e414150de31f5b992a1` (processing.py, losses.py, evaluation.py, grouped_array.py, pyproject.toml); PR #195
  - mlforecast PR #688 and #723, issue #577; core.py at `e3f057376ac3d9140069f44b3937107dc3cc4ff4`
  - hierarchicalforecast PR #305
- scikit-lego: `sklego/meta/grouped_predictor.py` and `_grouped_utils.py` at `f329086cb176a03a4e7c78055d7f6ea906987d1c`; issue #658
- fairlearn: #1440, #1522, #1552. great-tables: #737. marimo: #9082. pointblank: #39, #428. altair: #3921
- sktime: #5370, #5423, #6091, #9680, PR #10802
