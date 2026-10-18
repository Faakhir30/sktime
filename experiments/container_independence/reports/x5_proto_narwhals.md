# x5_proto_narwhals — narwhals-based VectorizedDF and panel-native narwhals forecasters

Status: complete at DEV_GRID scale (cloud rerun, 4 cores, ~2 concurrent agents → timings noisy;
ratios measured inside one process). Every number below comes from a script run in this
session; commands are listed in §8. Env: narwhals 2.27.1, polars 2.0.0, pyarrow 26, dask 2026.8,
duckdb 1.5.6, ibis 12; pandas 2.3.3 (`EXP_PY`) and 3.0.6 (`EXP_PY3`).

## 0. TL;DR

* **A narwhals VectorizedDF (iterate per instance) works on every backend, but is no speed-up for
  pandas users.** It does match current sktime exactly: 17/17 pandas equivalence cases pass on
  pandas 2 and pandas 3 (index names incl. `None`, Period/Datetime/int time, freq, order,
  non-str columns). Type round-trip holds for polars, polars-lazy, pyarrow, dask, duckdb and ibis.
  For pandas input, building pandas slices from the long frame costs **6.6–8.6× the current
  round trip** (generic path). A numpy-block fast path brings that to **1.8–2.2×**. For polars input,
  native polars slices make the round trip **3.7× faster than current pandas** at 10k series
  (and 8× faster than today's polars path, which converts to pandas).
* **The value is in panel-native forecasters, not in a narwhals loop.** In pure narwhals, naive
  last / seasonal-last / mean / drift run on all series at once and are **numerically identical to
  sktime `NaiveForecaster`** (max |diff| 0 for last/drift, ≤1.1e-13 for mean) on pandas, polars,
  polars-lazy, pyarrow, dask and duckdb. Speed-ups over sktime's per-series loop:
  ×15–160 (pandas), ×36–940 (polars), ×33–540 (pyarrow), ×6–300 (duckdb). For sktime's
  `NaiveForecaster`, cost grows with the number of series (~10 ms/series). The narwhals version stays at ≈0.1 s for 10k series on polars.
* **C1 (collect backends) is true as stated, but it can be overridden.** By default, `collect()` returns
  polars for polars-lazy, **pandas for dask, pyarrow for duckdb/ibis**. `collect(backend=...)` takes
  `"polars" | "pandas" | "pyarrow"` on all four, and `.lazy(backend=<orig>)` re-wraps the result to the input type. "Respecting
  the original format" really means never collecting. The panel-native forecasters do that: lazy in →
  lazy out, same type, for all 4 lazy backends.
* **C2 (no MultiIndex) is true. The RangeIndex / lost-names anecdote comes from
  `nw.maybe_reset_index`, which drops the index.** A native `reset_index()` keeps names. Restoring via
  `set_index(names)` is exact. Cost: reset_index takes 0.18–0.49× of today's `prepare_VectorizedDF` time and
  allocates **+6.1× (pd2, object keys) / +2.1× (pd3, str keys)** of the frame's bytes. On pandas 3 the value buffers are shared;
  on pandas 2 they are not.
* **Silent-correctness bugs in narwhals 2.27.1** (reproduced in this session):
  1. pyarrow `expr.agg().over(keys)` assigns values to the **wrong rows** once there are ≥~500 groups:
     ~60–80% of rows wrong, no error (pyarrow hash join does not keep order). This made the first benchmark run's pyarrow
     forecasts NaN/wrong. A join-based workaround fixes it.
  2. `group_by().agg(when(..).then(..).max())` ignores the condition on pandas and pyarrow.
  3. duckdb `(t - t0).dt.total_seconds()` drops whole days (`[0,0,5]` instead of `[0,172800,5]`).
  4. `LazyFrame.unique(order_by="time")` given as a string errors on duckdb and dask (list works).
* Coverage gaps: `ewm_mean` is polars/pandas-only. pyarrow `over` supports only elementary aggs (no
  shift/diff/cum/rolling). dask has no `over(order_by=)`, no `rank`, no `last`. `dt.offset_by("1mo")` is missing on
  pyarrow/dask. Horizontal concat is not available on lazy frames. `LazyGroupBy` is not iterable.

## 1. What was built

| file | purpose | SLOC* |
|---|---|---|
| `protos/x5_narwhals.py` | `NwVectorizedDF` (stateless schema) + `prepare_nw_vectorized`, `pandas_to_long`/`long_to_pandas`, `infer_freq_nw` | 381 |
| `protos/x5_nw_forecasters.py` | `NwPanelNaive` (last, last+sp, mean(+sp, window), drift(+window), ses) | 205 |
| `protos/x5_backends.py` | long panel → pandas/polars/polars-lazy/pyarrow/dask/duckdb/ibis | – |
| `scripts/x5_probe_narwhals.py` | op × backend support matrix | – |
| `scripts/x5_test_narwhals.py` | equivalence & round-trip tests | – |
| `scripts/x5_bench_vectorize.py` | VectorizedDF benchmark (`--grid dev|large`) | – |
| `scripts/x5_bench_forecasters.py` | forecaster benchmark (`--grid smoke|dev|large`) | – |
| `scripts/x5_claims.py` | C1/C2 verification, silent-bug repros, dask flakiness, LOC (`--grid`) | – |
| `scripts/x5_pa_over_bug.py` | repro + workaround check for the pyarrow `over` misalignment (`--grid`) | – |

*SLOC = no blanks/comments/docstrings (computed by `x5_claims.loc()`). For comparison: `sktime/datatypes/_vectorize.py` = 425,
`NaiveForecaster` class = 273 (plus `_BaseWindowForecaster` 70 and all of `BaseForecaster`).

### 1.1 NwVectorizedDF design
* Data model: LONG format (instance-key columns + time column + value columns). There is no index anywhere.
* pandas MI input: `reset_index` once. A `PandasIndexMeta` stores the original level names (may be `None`),
  the original column `Index` (labels may be ints → renamed `__colj__` because narwhals needs str
  names), the time kind and the Period freq (Period → timestamp, since narwhals and polars have no Period dtype).
  `long_to_pandas` restores the exact index.
* Other inputs: either the sktime `__index__<name>` convention (auto-detected; tested with sktime's
  own `convert_to(..., "polars_panel")` and back, which is exact) or explicit `keys=`/`time=`.
* Iteration: one `group_by(keys).agg(min/max row index, len)`, then contiguous zero-copy slices
  `frame[start:stop]` (sort once if instances are not contiguous). Slices are native frames of the
  (collected) input backend. For pandas they are re-indexed by time with freq inferred, which is what inner
  sktime estimators receive today.
* Reconstruct: vertical concat of results + ONE horizontal concat of the repeated key columns (pandas:
  `pd.concat(keys=...)` as today).
* Order: eager inputs use first-appearance order (= `groupby(sort=False)`). Lazy inputs have no row order, so
  keys are sorted.

### 1.2 Lazy handling (decision)
* The schema never collects. Instance keys of a lazy input are computed on demand through a keys-only
  query (projection pushdown). Freq inference costs 2 tiny queries (one key value, then its time column ≤1000 rows).
* `items()` must materialize (`LazyGroupBy` is not iterable, and filter-per-key = N queries). It collects
  ONCE, explicitly, into the eager format closest to the source: polars-lazy→polars, duckdb→pyarrow,
  ibis→pyarrow, dask→pandas (the `collect_backend` argument overrides this).
* `reconstruct(relazy=True)` re-wraps with `.lazy(backend=<input>)`, so output type == input type,
  but the data has already been materialized.
* Panel-native estimators never call `items()` and stay lazy end to end. This is the only path where
  laziness pays off (measured: §4.3).

### 1.3 Panel-native forecasters (no Python loop over series)
The only Python loop is over the fh steps (H ≤ a few dozen). Each step is a `select` on the per-series
state; the steps are combined with one vertical concat.
* rank within series: `with_row_index(order_by=keys+time)` + per-group max (instead of `rank().over`,
  which pyarrow/dask lack)
* last non-missing per (series, season): rank → `when(null)` → per-group min → pick → `group_by.max`
  (the conditional aggregation is precomputed as a column because of bug 2)
* mean / drift: `group_by.agg` after `filter(r <= window_length)`
* SES: `ewm_mean(adjust=False).over(keys, order_by=time)` (pandas/polars only)
* **Future timestamps per group:** `cutoff + lit(k*timedelta)` for Tick-like freqs (works on every
  backend; pandas 3's `Day` is no longer a `Tick` and has to be special-cased, see §6). Calendar freqs use
  `dt.offset_by("{n*k}mo|q|y")`. Month/quarter/year **end** anchors use `(+1 day).offset_by(..) − 1 day`.
  `offset_by` alone maps Feb-28 + 1mo to Mar-28. The calendar path works on pandas, polars and duckdb.
  It raises NotImplementedError on pyarrow and dask. Integer time: `cutoff + k*step`. No cross join with a
  "steps" frame: duckdb refuses joins across connections, and dask fails.
* The pyarrow over-bug workaround `_group_broadcast` uses `group_by.agg` + row join on pyarrow, and `.over` everywhere else.

## 2. Tests (scripts/x5_test_narwhals.py) — results/x5_test_pd2.json, x5_test_pd3.json

**A. pandas equivalence with current `VectorizedDF`.** Checks: names, column names, slices via
`assert_frame_equal(check_freq=True)`, reconstruct, and round trip == input.
Cases: Datetime D, Period M, int time, 3-level hierarchy (iterate_as Series and Panel), multivariate,
int column names, `None` level names, unsorted instance order; iterate_cols False/True.
**17/17 PASS on pandas 2.3.3 and 17/17 on pandas 3.0.6.** One exception: the optional `fast_pandas`
slice path gives a wrong index name for `panel_none_names` (it uses the flat name instead of `None`).
This is a known prototype gap.

**B. Round-trip fidelity (items → identity → reconstruct), 6×12 panel, 2 value columns.**

| input | slice type | output type == input | values equal | dtype changes |
|---|---|---|---|---|
| polars.DataFrame | polars.DataFrame | yes | yes | none |
| polars.LazyFrame | polars.DataFrame (collected once) | yes (re-lazified) | yes | none |
| pyarrow.Table | pyarrow.Table | yes | yes | none |
| dask.DataFrame | pandas.DataFrame | yes (re-lazified) | yes | none |
| duckdb relation | pyarrow.Table | yes (re-lazified) | yes | none |
| ibis Table (pd2 env only) | pyarrow.Table | yes | yes | **time: Datetime(s) → Datetime(us)** |
| sktime `polars_panel` (`__index__`) | polars | yes | yes, also after `convert_to` back to pd-multiindex | none |

**C. Forecasts vs sktime `NaiveForecaster`** (20 series × 60, fh=1..14). `last`, `last sp=7`, `mean`,
`mean sp=7`, `mean wl=10`, `drift`, `drift wl=15`: **ok on pandas-MI (exact `assert_frame_equal`),
pandas, polars, polars-lazy, pyarrow, dask, duckdb, ibis**, on both pandas versions.
Further cases:
* NaN (trailing NaNs in 2 series), last / last sp7 / mean: ok on all backends.
* Unequal length / unequal end: ok on polars and duckdb (280 rows, diff 0).
* Month-end, last obs 2002-02-28: future = 03-31, 04-30, 05-31 = sktime on pandas-MI, polars and duckdb.
  pyarrow and dask raise `Offsetting by mo is not yet supported`.
* SES (α=0.3) vs pandas `ewm(adjust=False)`: ok on pandas/polars/polars-lazy. NotImplemented on pyarrow, dask, duckdb, ibis.

Note: the n=20 tests did **not** catch the pyarrow `over` bug (it appears from ~500 groups). It was found
by the numeric check inside the benchmark at n=1000 (see §5).

## 3. VectorizedDF benchmark (results/x5_bench_vectorize_dev.json, pandas 2.3.3, min-of-repeats)

Ratios are relative to `cur_pd` (current sktime on pd-multiindex) in the same process.
`cur_pl` = current sktime fed `polars_panel` (convert to pandas inside, convert back).
`nw_pd` = narwhals proto on pandas MI producing pandas slices with a time index (what inner estimators expect).
`nw_pl`/`nw_pll` = polars eager/lazy producing **polars slices**, which is less work than building pandas slices,
so this is not like-for-like with cur_pd (it shows what polars-native estimators would get).

| n_series×T | phase | cur_pd | cur_pl | nw_pd | nw_pd_fast | nw_pl | nw_pll |
|---|---|---|---|---|---|---|---|
| 1000×100 | prepare | 9.85 ms | 187 ms (×19.0) | 15.9 (×1.61) | – | 4.62 (×0.47) | 6.46 (×0.66) |
| 1000×100 | items | 92.6 ms | – | 776 (×8.38) | 186 (×2.01) | 18.9 (×0.20) | 19.8 (×0.21) |
| 1000×100 | reconstruct | 13.5 ms | 25.3 (×1.88) | 17.1 (×1.27) | – | 16.2 (×1.20) | 17.5 (×1.30) |
| 1000×100 | round trip | 114 ms | 311 (×2.72) | 821 (×7.20) | 222 (×1.95) | 39.2 (×0.34) | 41.9 (×0.37) |
| 10000×50 | prepare | 43.7 ms | 1359 (×31.1) | 68.2 (×1.56) | – | 13.0 (×0.30) | 11.3 (×0.26) |
| 10000×50 | items | 1208 ms | – | 8649 (×7.16) | 2032 (×1.68) | 170 (×0.14) | 179 (×0.15) |
| 10000×50 | reconstruct | 144 ms | 202 (×1.40) | 206 (×1.43) | – | 146 (×1.01) | 149 (×1.03) |
| 10000×50 | round trip | 1310 ms | 2873 (×2.19) | 8608 (×6.57) | 2342 (×1.79) | 353 (×0.27) | 350 (×0.27) |

Smaller sizes (10, 100 series) are in the JSON: at 100 series nw_pd round trip is ×6.69, nw_pl ×0.64.
Peak RSS delta of prepare at 10k: cur_pl 26.7 MB, nw_pl 6.7 MB, nw_pll 3.0 MB, cur_pd/nw_pd ≈0 (sampling resolution).

Reading:
* For pandas users, a narwhals VectorizedDF is a slowdown: generic ×6.6–8.6, fast path ×1.8–2.2.
  Building pandas slices from long columns (set_index + infer_freq per slice) costs more than
  `groupby(level)` on the existing MultiIndex.
* For polars users it is a large win. Today they pay ×2.2–2.9 over pandas (conversion), and prepare alone
  is ×19–31. Native polars slices give ×0.27–0.37 of the pandas round trip, i.e. **≈8× faster than
  today's polars path** at 10k series (2873 → 353 ms). This only helps if the inner estimator accepts polars.
* Lazy polars costs ≈ eager polars. Laziness gives nothing in a per-instance loop (it collects once).

## 4. Forecaster-layer benchmark (results/x5_bench_forecasters_dev.json, pandas 2.3.3, fh=1..12)

Times are fit+predict. Lazy paths include the final `.collect()`/`.compute()`/fetch-to-arrow.
`sk_pd` = sktime NaiveForecaster on pd-multiindex (per-series loop). At 10k series sktime was run only
for `last` (single run, ~100 s). Rows at 10k for the other strategies have no sktime reference: unmeasured.

| n×T | strategy | sk_pd | nw_pdmi | nw_pl | nw_pll | nw_pa | nw_duckdb | nw_dask |
|---|---|---|---|---|---|---|---|---|
| 100×100 | last | 992 ms | 38.3 (×26) | 14.3 (×69) | 17.2 (×58) | 16.2 (×61) | 93.1 (×11) | 1374 (×0.7) |
| 100×100 | last sp7 | 2301 | 62.7 (×37) | 21.3 (×108) | 45.6 (×51) | 18.3 (×126) | 122 (×19) | 1548 (×1.5) |
| 100×100 | mean | 546 | 30.8 (×18) | 15.1 (×36) | 16.0 (×34) | 14.0 (×39) | 68.7 (×8) | 1016 (×0.5) |
| 100×100 | drift | 593 | 41.4 (×14) | 14.0 (×42) | 18.1 (×33) | 18.2 (×33) | 103 (×6) | 1479 (×0.4) |
| 1000×100 | last | 10257 | 122 (×84) | 28.6 (×359) | 29.4 (×349) | 46.7 (×220) | 136 (×75) | 1636 (×6) |
| 1000×100 | last sp7 | 22692 | 268 (×85) | 37.4 (×607) | 93.2 (×244) | 43.1 (×527) | 155 (×146) | 1956 (×12) |
| 1000×100 | mean | 5758 | 75.6 (×76) | 25.3 (×227) | 26.5 (×217) | 30.8 (×187) | 94.7 (×61) | 1165 (×5) |
| 1000×100 | drift | 5732 | 95.4 (×60) | 29.2 (×196) | 30.4 (×189) | 40.0 (×143) | 136 (×42) | 1667 (×3) |
| 10000×50 | last | 99296 | 675 (×147) | 105 (×943) | 126 (×791) | 185 (×536) | 333 (×298) | 2603 (×38) |
| 10000×50 | last sp7 | unmeasured | 2166 | 131 | 459 | 198 | 444 | 4592 |
| 10000×50 | mean | unmeasured | 314 | 85.9 | 104 | 133 | 238 | 1567 |
| 10000×50 | drift | unmeasured | 392 | 97.8 | 125 | 169 | 319 | 2313 |

* max |diff| vs sktime: **0.0** for last, last sp7 and drift on every backend, and ≤1.14e-13 for mean (summation
  order), at 100 and 1000 series and at 10k for `last`. Before the pyarrow workaround (first run of this session),
  pyarrow gave `nan` diffs at n≥1000 (689/1000 series NaN state).
* `sk_pl` (polars_panel into today's sktime) ≈ `sk_pd` (10374 vs 10257 ms at 1000): the conversion is noise
  next to the per-series loop. The loop is the dominant cost (~10 ms per series for fit+predict).
* Ranking at 10k: polars eager < polars lazy < pyarrow < duckdb < pandas < dask. The pandas backend through
  narwhals is 4–16× slower than polars for the same expressions (last sp7: 2166 vs 131 ms).
  dask carries ~1–1.5 s fixed graph/scheduler overhead at these sizes.

### 4.3 Laziness preserved (x5_claims, 1000×100, last sp7)
| backend | out type | fit+predict (plan only, freq given) | execution |
|---|---|---|---|
| polars lazy | polars.LazyFrame | 4.4 ms | 146 ms |
| dask | dask.DataFrame | 71 ms | 2012 ms |
| duckdb | DuckDBPyRelation | 77 ms | 121 ms |
| ibis | ibis.Table | 34 ms | 169 ms |

With freq inference added: polars 12 ms, dask 327 ms, duckdb 117 ms. The output is a deferred plan of the same type as the input, so a
user's downstream `filter`/`select` can still be pushed down.

## 5. C1 / C2 / silent bugs (results/x5_claims_{pd2,pd3}_dev.json, x5_pa_over_bug_dev.json)

### C1 — "lazy collect: polars→polars, dask→pandas, duckdb→arrow"
**True.** Default `collect()`: polars-lazy→polars.DataFrame, dask→pandas.DataFrame,
duckdb→pyarrow.Table, ibis→pyarrow.Table. **It can be overridden:** `collect("polars"|"pandas"|"pyarrow")`
works for all four, and `collect().lazy(backend=orig)` returns the original type (dask, duckdb, ibis
and polars all verified). Measured costs (1000×100, pd2): dask collect default 18.8 ms vs collect+relazy 17.8 ms; duckdb
7.2 vs 8.4 ms. Re-lazifying is cheap but **does not restore laziness semantics**: the data is
materialized, and a dask frame re-created from pandas has new partitions. Practical answer: accept that per-instance
iteration materializes (choose the backend explicitly). Where laziness matters, use panel-native
estimators that never collect (§4.3).

### C2 — "no MultiIndex; reset_index copy; came back as RangeIndex, names gone"
* `nw.from_native(MI frame).columns == ["c0"]`: the index is invisible to narwhals.
* **`nw.maybe_reset_index` drops the index** (`level_values_kept: False`, RangeIndex). That reproduces
  the "names gone" anecdote exactly. Native `reset_index()` keeps level names as columns (`None` names →
  `level_0..`). Narwhals ops return RangeIndex frames, and `set_index(names)` restores the original names exactly
  (all 17 A-cases restore index names, dtypes and freq).
* Gotcha: `nw.from_native(MI).group_by("h0")` *works* (pandas accepts level names), so code that accidentally
  groups an un-reset MI frame silently behaves differently from the other backends.
* Cost of reset_index:

| env | n×T | reset_index | prepare_VectorizedDF | ratio | new bytes / frame bytes | value buffer shared |
|---|---|---|---|---|---|---|
| pd2 | 1000×100 | 3.8 ms | 7.6 ms | 0.49 | 7.30 MB / 1.20 MB (×6.1) | no |
| pd2 | 10000×50 | 17.8 ms | 46.6 ms | 0.38 | 36.5 / 6.34 MB (×5.8) | no |
| pd3 | 1000×100 | 3.4 ms | 13.0 ms | 0.26 | 2.41 / 1.12 MB (×2.2) | yes (CoW) |
| pd3 | 10000×50 | 10.7 ms | 60.1 ms | 0.18 | 12.1 / 5.66 MB (×2.1) | yes |

  The copy is the key levels materialized per row (object strings on pd2). Iterating `groupby("h0")` on the
  column vs `groupby(level=0)`: 148 vs 197 ms (pd2), 259 vs 199 ms (pd3) at 10k. Same order of magnitude.
* Dtype drift after reset_index: pd3 gives `str` keys and `datetime64[us]`. Period columns are `Unknown` to narwhals
  (must convert to timestamp, as the proto does).

### Silent-correctness bugs (narwhals 2.27.1), all reproduced on pd2 and pd3
| # | expression | backend | got | expected |
|---|---|---|---|---|
| 1 | `col.min().over(g)` with ≥~500 groups | pyarrow | wrong row assignment in 8/10 trials at 500 groups, 10/10 at 1000 and 10000; 80%/80%/62% of rows wrong; join-workaround 0/10 | – |
| 2 | `group_by.agg(when(r==1).then(v).max())` | pandas, pyarrow | [30, 5] | [20, 1] (polars correct) |
| 3 | `last().over(g, order_by=t)` | pyarrow | [20,1,20,1,20] | [20,20,20,1,1] |
| 4 | `(t - t0).dt.total_seconds()` | duckdb | [0, 0, 5] | [0, 172800, 5] |
| 5 | `LazyFrame.unique(keys, order_by="time")` | duckdb, dask | BinderException / KeyError | works with `["time"]` |

Root cause of #1/#3: narwhals' pyarrow `over` = `group_by` + `pa.Table.join` + take the joined column.
pyarrow's hash join does not preserve left row order. Workaround (`_group_broadcast`): join the aggregates back as
whole rows. `NwVectorizedDF._offsets` also re-sorts after its join by an explicit ordinal (it passed
without the re-sort up to 5000 series, but there is no ordering guarantee). dask flakiness: 0/60 failures
(20 trials × 3 strategies) in this run.

### Op-support matrix (results/x5_probe_pd2.json; reference = polars eager)
Some ERR cells for pyarrow/dask come from the probe's helper using `rank().over` (unsupported on those two),
not from the op itself. Those are marked "dep".

| op | pandas | polars | pl-lazy | pyarrow | dask | duckdb | ibis |
|---|---|---|---|---|---|---|---|
| group_by.agg(mean), mean().over, with_row_index(order_by), left join, vertical concat | ok | ok | ok | ok | ok | ok | ok |
| rank(...).over | ok | ok | ok | ERR | ERR | ok | ok |
| group_by.agg(last) | ok | ok | ERR | ok | ERR | ERR | ERR |
| last/shift/diff/cum_*/rolling_mean `.over(keys, order_by)` | ok | ok | ok | ERR (last: WRONG) | ERR | ok | ok |
| ewm_mean `.over(keys, order_by)` | ok | ok | ok | ERR | ERR | ERR | ERR |
| dt.offset_by("1d") / ("1mo") | ok/ok | ok/ok | ok/ok | ok/ERR | ERR/ERR | ok/ok | ok/ok |
| time + lit(timedelta) | ok | ok | ok | ok | ok | ok | ok |
| dt.total_seconds() | ok | ok | ok | ok | ERR | WRONG | ERR |
| cross join with helper frame | ok | ok | ok | ok | ERR | ERR (other connection) | ok |
| concat horizontal | ok | ok | ERR | ok | ERR | ERR | ERR |
| group_by iteration | ok | ok | ERR | ok | ERR | ERR | ERR |
| when/then inside group_by.agg | WRONG | ok | ok | dep | dep | ok | ok |

## 6. Gotchas found while building
1. **Index:** narwhals sees no index. pandas inputs need reset_index + metadata, and `maybe_reset_index` drops data.
   Integer column names must be renamed (narwhals requires str), and Period has no dtype (convert to timestamp).
2. **Lazy:** row order does not exist. "First-appearance" instance order (sktime's `groupby(sort=False)`)
   becomes sorted order for lazy inputs. Order-dependent expressions need `order_by`, and polars-lazy refuses
   `agg(last)` without it.
3. **Dtypes:** time unit drift (ibis s→us; pd3 reset_index → us). NaN vs null: the proto converts NaN→null
   (`fill_nan(None)`) so that "missing" means the same thing everywhere.
4. **pandas 3:** `to_offset("D")` is no longer a `Tick`. The first pd3 run failed on all C tests with
   `freq 'D' not expressible` until `Day` was special-cased (`timedelta(days=n)`, valid for tz-naive data only).
5. **dask:** never put a window expression inside `filter`, derive columns before filtering, and `over(order_by)` is unsupported.
6. **duckdb:** frames from different connections cannot be joined (no helper "steps" frames).
7. **Silent backend bugs** (§5) mean every narwhals estimator needs a cross-backend numeric test at a
   realistic group count (≥1000), not just a toy size.

## 7. Conclusions for the design question
* **Narwhals as the VectorizedDF layer:** it is feasible (exact pandas parity, type round-trip on 6 backends,
  381 SLOC ≈ size of the current `_vectorize.py`). It is only worthwhile if the per-instance estimator accepts
  the native slice type. For pandas users it is slower (×1.8–2.2 at best). Keep a pandas MI fast path (today's code)
  and use narwhals for non-pandas inputs.
* **Narwhals for forecasters:** large wins (×100–1000 on polars at 1k–10k series), but only for estimators
  that can be written as set-based expressions: naive family, window means, drift, and SES on polars/pandas only.
  Anything recursive (ARIMA, ETS fitting, theta with optimisation) cannot be written in narwhals and still needs the
  per-instance loop or numpy. 205 SLOC reimplement 4 strategies across 6–7 backends. Expect per-backend
  conditionals (`_group_broadcast`, calendar offsets) and a test matrix of backends × strategies × sizes.
* Lazy: keep lazy only in panel-native estimators. Elsewhere collect once, explicitly, to the native eager counterpart.

## 8. Commands (from WT root; `EXP_PY`/`EXP_PY3` as in the task context)
```
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x5_probe_narwhals.py   # -> results/x5_probe.json (copied to x5_probe_pd2.json)
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x5_test_narwhals.py    # -> results/x5_test_pd2.json
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/x5_test_narwhals.py    # -> results/x5_test_pd3.json
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x5_claims.py --grid dev # -> results/x5_claims_pd2_dev.json
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/x5_claims.py --grid dev # -> results/x5_claims_pd3_dev.json
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x5_pa_over_bug.py --grid dev  # -> results/x5_pa_over_bug_dev.json
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x5_bench_vectorize.py --grid dev   # -> results/x5_bench_vectorize_dev.json
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x5_bench_forecasters.py --grid dev # -> results/x5_bench_forecasters_dev.json
```
For the serial benchmark phase: the same scripts with `--grid large`. In bench_forecasters, sktime at ≥10k
series runs only `last` by default (`--sk-big-strategies`).
The LOC in §1 come from `x5_claims.loc()` re-run after the final prototype edits. The claims JSON holds the
pre-edit LOC (376/199).

## 9. Caveats
* Timings come from a shared 4-core machine with another agent running. Ratios are within-process and min-of-repeats,
  but sktime at 10k series is a single run.
* The VectorizedDF comparison nw_pl vs cur_pd is not like-for-like (polars slices vs pandas slices with freq).
* Only pd2 for benchmarks. Correctness was checked on pd2 and pd3.
* The forecaster fh is integer out-of-sample only (no in-sample fh, no prediction intervals, no update).
* `fast_pandas` slice path: wrong index name for `None` level names (prototype gap).
* The pyarrow `over` bug threshold (~500 groups) is nondeterministic (8/10 trials at 500).
