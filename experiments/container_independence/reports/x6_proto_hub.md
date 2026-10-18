# x6_proto_hub — hub-and-spoke canonical internal representation

**Question.** Instead of (1) one adapter per library for the whole framework, or (2) rewriting
forecasters against narwhals, can sktime convert every input **once** into a small,
library-neutral, numpy-based *hub*, run vectorization and (numpy) forecaster kernels on views,
and convert back to the **exact** native container at egress? What does that cost in time,
memory and fidelity, and what is lost?

Prior art this follows (background, not measured here):
- **Nixtla `GroupedArray`** (utilsforecast / mlforecast / statsforecast): `data` 2D float array +
  `indptr` offsets; models run per group on array slices or over all groups at once. Input
  frames (pandas or polars) are sorted by (id, ds) and reduced to arrays, with ids and times
  kept beside them for the output.
- **Darts `TimeSeries`**: values array (time x component x sample) + time index + static
  covariates and metadata. It has `from_dataframe` / `to_dataframe` spokes for several
  libraries.
- **GluonTS** `ListDataset` (one dict per series: `start` + `target` array) and aeon's numpy3D /
  list-of-2D collections. Both are canonical numpy forms with metadata kept on the side.
- sktime already has numpy mtypes (`np.ndarray`, `numpy3D`), but they carry no time or instance
  metadata, so they cannot round-trip a native container.

## 1. Design (protos/x6_hub.py, 1238 lines)

`PanelData`:

| field | content | size |
|---|---|---|
| `indptr` | int64 offsets, instance i = rows `indptr[i]:indptr[i+1]` | O(n_instances) |
| `time` | per-row `datetime64[unit]` (UTC-naive if tz), or int64 (period ordinals / int) | 8 B/row |
| `values` | 2D float64 `(n_rows, n_cols)`, a **view** of the source where possible | 8 B/row/col |
| `key_names`, `key_values` | instance key table, one 1D array per level | O(n_instances) |
| `aux` | non-float columns (str/categorical), row aligned | — |
| `tmeta` | kind, unit, tz, pandas freqstr, numeric step, RangeIndex flag | — |
| `meta` | source library + layout, column order, native dtypes/schema, row permutation if input was not grouped contiguously, lazy flag, npartitions, null columns, pandas Series name, columns-index name | — |

Spokes:
- **(i) hand-written**:
  - pandas MultiIndex, plain index and Series. Reads `MultiIndex.codes` directly, with **no
    `reset_index`**.
  - pandas long, polars eager and lazy (natural layout and the sktime `__index__` convention).
  - pyarrow, including dictionary and chunked keys.
  - dask (compute once, then the pandas-long spoke).
- **(ii-a) narwhals**: one generic `nw_from_native` / `nw_to_native`.
- **(ii-b) Arrow PyCapsule**: `pa.table(obj)` on any `__arrow_c_stream__` producer, then the arrow
  spoke. Egress goes through `narwhals.from_arrow(..., backend=<source lib>)`, re-lazified for
  duckdb, dask and polars-lazy sources.

Forecaster side:
- `SeriesView`: per-instance views of `time` and `values`.
- `SeriesView.to_pandas(freq_mode=...)`: lazily materializes a pandas object on top of the
  view.
- `future_time`: fh arithmetic without pandas for int, period and fixed-step datetimes. It
  falls back to pandas offsets for calendar frequencies.
- `HubNaive`: a per-slice numpy kernel that mirrors `NaiveForecaster` last/mean/drift with
  `window_length` and `sp`.
- `hub_naive_batched`: all instances at once using `indptr` `reduceat`.

## 2. Methods / commands

All commands were run from the worktree root with `PYTHONPATH=.:experiments/container_independence`.

| | |
|---|---|
| `$EXP_PY` | pandas 2.3.3, numpy 2.2.6, polars 2.0.0, narwhals 2.27.1, pyarrow 26, dask 2026.8, duckdb 1.5.6 |
| `$EXP_PY3` | pandas 3.0.6 |
| Machine | 4 cores, shared with about 1 other agent, so timings are noisy. Each ratio below compares two measurements taken in the **same process**, interleaved, using min-of-repeats from `common.bench` / `abench`. |
| Memory | Each case runs in a fresh process. The script resets `VmHWM` through `/proc/self/clear_refs`, then reports peak = VmHWM − RSS_before. It also records the sampling helper `common.peak_rss_delta_mb` as a cross-check. |

```
$EXP_PY  -W ignore experiments/container_independence/scripts/x6_fidelity.py          # -> results/x6_fidelity_pd2.json
$EXP_PY3 -W ignore experiments/container_independence/scripts/x6_fidelity.py          # -> results/x6_fidelity_pd3.json
$EXP_PY  -W ignore experiments/container_independence/scripts/x6_zerocopy.py --grid dev
$EXP_PY  -W ignore experiments/container_independence/scripts/x6_bench.py --grid dev --parts ingress slicing
$EXP_PY  -W ignore experiments/container_independence/scripts/x6_bench.py --grid dev --parts forecaster --current-max 1000
$EXP_PY  -W ignore experiments/container_independence/scripts/x6_mem.py --grid dev
$EXP_PY  -W ignore experiments/container_independence/scripts/x6_pushdown.py --grid dev
$EXP_PY  -W ignore experiments/container_independence/scripts/x6_profile.py            # 200x50 default
python   experiments/container_independence/scripts/x6_tables.py                       # renders the tables below
```

All benchmark scripts accept `--grid {dev,large}`. Everything in this report was re-run in this
(cloud) session; the numbers below come from those runs.

## 3. Results

### 3.1 Ingress (native → hub) vs current (`prepare_VectorizedDF`, incl. convert_to pandas)

These are `x6_bench --parts ingress`, 1 float column. For non-pandas sources the "current"
column includes the conversion to a pandas MultiIndex.

| source | 100x100 hub / current ms (×) | 10000x50 hub / current ms (×) | values zero-copy | time zero-copy |
|---|---|---|---|---|
| pandas_mi | 0.54 / 2.36 (4.4×) | 2.38 / 37.2 (15.6×) | yes | no (8 B/row copy via `codes.take`) |
| polars_sktime (`__index__`) | 1.58 / 31.6 (20×) | 7.01 / 2163 (**309×**) | yes | yes |
| polars_long | 1.32 / 6.20 (4.7×) | 7.83 / 95.5 (12.2×) | yes | yes |
| polars_lazy (collect once) | 1.78 / 6.17 (3.5×) | 7.27 / 98.6 (13.6×) | n/a (materialized) | n/a |
| arrow_long | 0.46 / 5.51 (12×) | 11.7 / 96.0 (8.2×) | yes | yes |
| dask_sktime (compute) | 7.41 / 11.4 (1.5×) | 165 / 258 (1.6×) | n/a | n/a |

Notes on the table:
- The 309× for `polars_sktime` is mostly one current-path problem. `x6_profile`
  (`prepare_VectorizedDF(polars_panel)` at 2000x50) shows 93% of the time inside the mtype
  check's `polars is_monotonically_increasing`. Conversion itself is only 47%, which overlaps
  with that check.
- On the other paths the speedup is 4–15×.
- dask gains little, because `.compute()` dominates both paths.

### 3.2 Egress (per-instance outputs → reconstruct → native), h=10

| target | 10000x50 hub ms (of which concat) | current reconstruct(+convert) ms | × |
|---|---|---|---|
| pandas_mi | 13.8 (11.1) | 88.4 | 6.4× |
| polars_sktime | 13.7 (12.3) | 96.3 | 7.0× |
| dask_sktime | 28.7 (17.1) | 363 | 12.7× |
| polars_long / lazy / arrow | 17.0 / 17.9 / 17.4 | — (no current path: the current output is pandas or the sktime convention) | |

The pandas_mi egress output is checked with `assert_frame_equal` against the current
`reconstruct`.

### 3.3 Peak memory of ingress (fresh process, 10000x100, median of 3)

| source | cols | current peak MB | hub peak MB | ratio |
|---|---|---|---|---|
| pandas MI | 1 | 53.3 | 15.4 | 3.5× |
| polars `__index__` | 1 | 109.8 | 5.9 | 18.6× |
| arrow long | 1 | 111.4 | 20.0 | 5.6× |
| pandas MI | 4 | 53.6 | 15.4 | 3.5× |
| polars `__index__` | 4 | 129.5 | 5.8 | 22× |
| arrow long | 4 | 142.3 | 33.3 | 4.3× |

The hub's own allocations are as follows:
- **pandas**: time 8 MB plus key table. Values are a view. The extra transient in the 15.4 MB
  comes from MultiIndex code materialization.
- **polars**: almost nothing. Values and time are zero-copy, and only `indptr` and keys are
  allocated.
- **arrow**: dictionary-encoding the string keys, and with 4 columns one 2D stacking copy, because
  arrow columns are separate buffers.

End-to-end NaiveForecaster(last) fit+predict at 1000x100 (`x6_mem` e2e cases):

| path | pandas input: peak MB / time | polars input: peak MB / time |
|---|---|---|
| current | 24.2 / 12.3 s | 28.3 / 13.6 s |
| hub loop | 1.4 / 0.025 s | 1.3 / 0.029 s |
| hub batched | 2.0 / 0.004 s | 0.7 / 0.007 s |

### 3.4 Zero-copy contract (`x6_zerocopy`)

**Ingress and egress sharing:**
- pandas: values are a view (writeable). With one column the layout is F; with 3 columns it is C
  (single block). Time is never a view, because a MultiIndex stores levels + codes, not a
  per-row array.
- polars and arrow: values and time are views. polars values are **read-only**, and so are arrow
  values with 1 column. Arrow with 3 columns copies values, because separate buffers have to be
  stacked into 2D.
- Egress shares `values` with the output for pandas, polars and arrow.

**Writeability:**
- An in-place write on a slice from a polars or arrow source raises
  `ValueError: assignment destination is read-only`. This is a useful safety property: kernels
  cannot corrupt user data. It also means kernels that mutate in place must copy.
- Per-slice pandas objects built from hub views share memory with the hub (yes in every case).

**Unsorted input.** When rows of an instance are not contiguous, the hub permutes rows once
(and restores the original order at egress; the round-trip check passes). At 10000x50 this makes
ingress **46×** slower for pandas (2.8 → 128 ms) and **35×** slower for polars (7.2 → 253 ms),
and values are no longer zero-copy. The current path on unsorted input was not measured
(unmeasured).

### 3.5 Slicing and per-slice pandas materialization (`x6_bench --parts slicing`)

Cost per slice:

| | 100x100 µs/slice | 10000x50 µs/slice |
|---|---|---|
| current `VectorizedDF.items` (groupby level) | 125.0 | 109.4 |
| hub view (`SeriesView`) | 1.64 | 1.52 (**≈72×** cheaper) |
| hub → pandas, public `pd.DatetimeIndex(t)` | 46.2 | 44.6 |
| hub → pandas, private ctor, no freq | 25.8 | 25.2 |
| hub → pandas, private ctor, **trusted freq from hub metadata** | 27.7 | 26.7 (**4.1×** cheaper than current items) |
| hub → pandas, `DatetimeIndex(t, freq=f)` (validated) | 200.5 | 230.0 |
| hub → pandas, `infer_freq` per slice (≈ current `_enforce_index_freq`) | 267.0 | 277.4 |

So a pandas view per slice is cheap (about 26 µs) **only if** freq is set from hub metadata
without re-validating it. That requires the private `DatetimeArray._simple_new` /
`DatetimeIndex._simple_new` constructors, which are fragile across pandas versions. With the
public constructor, freq validation alone costs about 8× the view construction. All pandas-view
variants share memory with the hub.

### 3.6 Forecaster: numpy shim vs current vs "pandas view per slice"

NaiveForecaster, fh=1..10. `x6_bench --parts forecaster --current-max 1000`. All hub variants are
asserted equal to the current output (`assert_frame_equal`, or polars `.equals`) for n ≤ 100,
for both strategies and both sources.

| strategy / source, 1000x100 | current s | hub + full sktime estimator per slice (pandas view) | hub + NaiveForecaster `_fit/_predict` only (pandas view) | hub numpy loop (`HubNaive`) | hub batched (`indptr` reduceat) |
|---|---|---|---|---|---|
| last / pandas_mi | 12.38 | 12.25 (1.01×) | 10.86 (1.14×) | 0.0228 (**543×**) | 0.0026 (**4676×**) |
| last / polars_sktime | 12.91 | 12.02 (1.07×) | 10.28 (1.26×) | 0.0229 (564×) | 0.0037 (3462×) |
| drift / pandas_mi | 7.11 | 6.51 (1.09×) | 4.16 (1.71×) | 0.0194 (365×) | 0.0021 (3337×) |
| drift / polars_sktime | 7.52 | 5.90 (1.28×) | 4.71 (1.60×) | 0.0203 (371×) | 0.0032 (2339×) |

At 10000x50 the current path was skipped as too slow for the dev budget. There the hub loop took
0.15–0.19 s and the batched version 7–12 ms.

Attribution (`x6_profile`, NaiveForecaster(last), pd-multiindex 200x50, total 5.06 s ≈ 25 ms per
instance for fit+predict):

| component | share of total |
|---|---|
| `NaiveForecaster._predict_naive` (Lag + concat + ffill, pandas) | 64% |
| `convert_to` (1604 calls) | 23% |
| `mtype()` checks | 17% |
| `_check_X_y` | 10.5% |
| `ForecastingHorizon.to_absolute` | 10.3% |
| `pd.concat` | 9.5% |
| `VectorizedDF.items` | 3.3% |
| `prepare_VectorizedDF` | 0.06% |

The shares are cumulative and overlap.

**Interpretation.** Swapping the data container under an *unchanged* pandas forecaster buys
≤ 1.3×. The per-slice estimator, both its pandas internals and the BaseForecaster boilerplate,
dominates, not the container. The hub's large gains (365–4700×) come only when the forecaster's
math runs on numpy views, which means re-implementing the kernel (here about 50 lines for Naive
last/mean/drift with `sp`), ideally batched across instances.

### 3.7 Laziness / pushdown via an ingress "lookback hint" (`x6_pushdown`, 5000x200 parquet, w=10)

NaiveForecaster(mean, window_length=10) needs only the last 10 rows per instance. The hub spoke
can push "tail w per instance" into the lazy plan before materializing.

| case | rows ingested | t min s | peak MB |
|---|---|---|---|
| current: `pd.read_parquet` → set_index → prepare_VectorizedDF | 1,000,000 | 0.684 | 165.6 |
| polars scan → collect all → hub → batched predict | 1,000,000 | 0.071 | 96.1 |
| polars scan → `group_by.tail(10)` in plan → hub | 50,000 | 0.060 | 96.4 |
| same, `engine="streaming"` | 50,000 | 0.066 | 96.5 |
| duckdb read_parquet ORDER BY → capsule → hub | 1,000,000 | 0.189 | 158.7 |
| duckdb … `QUALIFY row_number() <= 10` → capsule → hub | 50,000 | 0.096 | **39.7** |

Predictions are identical in all hub cases (hash 033b5e8add, allclose rtol 1e-12).
- **duckdb** pushdown cuts peak memory 4.0× (158.7 → 39.7 MB) and time 2×.
- **polars** `group_by.tail` cuts rows 20× but not peak memory: the scan still materializes the
  file in this version (polars 2.0.0) and plan.

Either way, the hub path is 7–11× faster than today's pandas ingress. This only works because the
hub knows the estimator's lookback *before* materializing. A pandas-first design cannot express
that.

### 3.8 Round-trip fidelity and LOC: spoke styles (`x6_fidelity`, 30 fixtures)

Check: strict native → hub → native equality of values, dtypes, names, freq, row order, container
type (eager/lazy), arrow schema + metadata, and dask npartitions.

| style | pandas 2.3.3 pass | pandas 3.0.6 pass | logical LOC |
|---|---|---|---|
| (i) hand-written | **29/29** (+1 n/a: no duckdb spoke) | **29/29** | 449 total (pandas 139, polars 102, arrow 108, dask + dispatch rest) + 67 shared numpy core |
| (ii-a) narwhals generic | 22/30 | 21/30 | 120 (+ core) |
| (ii-b) Arrow PyCapsule + arrow spoke + `nw.from_arrow` | 26/30 | 26/30 | 145 incl. reused arrow spoke (+ core) |

Narwhals spoke failures:
- **Period index.** Period has no narwhals dtype, so the column becomes object and the
  freq/step inference crashes.
- **Single-series freq lost** (`pd_single_freq_D` and `pd_series_named_h`). The `reset_index`
  plus `set_index` round trip drops `DatetimeIndex.freq`.
- **RangeIndex comes back as an Int64 Index** on pandas 2 (it passes on pandas 3).
- **Arrow schema drift.** Dictionary-typed keys come back as plain string. On pandas 3 every arrow
  fixture fails: `string` comes back as `large_string`, and pandas metadata is lost.
- **dask npartitions** 2 → 1.

Capsule spoke failures: single-series freq, `pd.Series` comes back as a DataFrame, a RangeIndex
Series cannot be keyed, and dask npartitions.

Every failure is in **metadata the generic layer has no slot for** (freq, Series-ness,
RangeIndex, arrow logical types, partitioning). Each could be patched with extra fields in
`meta`, but at that point the generic spoke becomes a hand-written spoke.

Under pandas 3 the narwhals spoke gains zero-copy on a few pandas fixtures (int, 3-level, `ME`).

### 3.9 The contested narwhals claims, seen from the hub

**C1 (collect changes the backend).** In the narwhals spoke, the default `collect()` turns dask
into pandas and duckdb into Arrow. That is what makes the hub needed, but it does not break the
round trip: `nw_to_native` re-lazifies with `.lazy(backend=impl)`, and the `duckdb_relation` and
`pl_lazy` fixtures round-trip as a duckdb relation and a polars LazyFrame. The spoke also exposes
`collect_backend=`, which is passed through to narwhals. What is *not* recoverable generically:
- dask partitioning (npartitions),
- the original duckdb query plan. Egress returns a relation over materialized Arrow data.

So C1 is true for the intermediate. Under a hub design it is irrelevant, because materializing
once is the design, and the target format is restored at egress. Pushdown (§3.7) is the only way
to keep lazy-engine value.

**C2 (no MultiIndex, reset_index loses names).** The hand-written pandas spoke never calls
`reset_index`. It reads `MultiIndex.codes` and levels, so the only copy is the time column
(8 B/row), and level names are kept in `key_names` / `time_name`. All MultiIndex fixtures
round-trip in every style: hierarchical 3-level, categorical level, unsorted, unequal length, tz,
NaN. The narwhals spoke *does* need `reset_index` (a copy, see §3.8) and restores names from
`meta`, which passes. What it loses is freq and RangeIndex, not names. Under a hub, C2 is solved
by metadata, not by narwhals.

## 4. Conclusions

1. **Ingress/egress.** A hub with hand-written spokes is a cheap, high-fidelity boundary.
   - Ingress time: 4–15× faster than `prepare_VectorizedDF` incl. conversion, and up to 309× for
     `polars_panel`, mostly because of an expensive mtype monotonicity check.
   - Egress: 6–13× faster.
   - Peak ingress memory: 3.5–22× lower.
   - Fidelity: 29/29 strict round trips on pandas 2 and pandas 3.
   - Values are zero-copy from pandas (single float block), polars and single-chunk arrow. The
     one unavoidable ingress copy for pandas is the time array (8 B/row). For lazy sources the
     only copy is the one collect.
2. **Slicing.** Hub views are about 72× cheaper than `VectorizedDF.items`. A pandas view per
   slice costs about 26 µs when freq is trusted from metadata, 4× cheaper than today's `items`,
   but 200–280 µs if freq is validated or inferred.
3. **Forecasters.** The container alone is not the bottleneck: pandas-semantic forecasters gain
   ≤ 1.3×. Large gains (365× per-slice numpy loop, 2300–4700× batched) need numpy kernels, which
   can only be written per estimator. This matches the prior art: Nixtla gets its speed because
   every model is written against `GroupedArray`, not because of the container.
4. **Spoke style.**
   - **Hand-written per-library spokes** are the only style with full fidelity. They cost about
     100–140 LOC per library and are confined to two functions per library, not hundreds of
     estimators. This is the key difference from idea (1).
   - **Narwhals** halves spoke LOC but loses pandas freq, Period, RangeIndex, arrow logical types
     and npartitions.
   - **The Arrow PyCapsule** is a good *generic fallback ingress* for unknown producers (duckdb,
     ibis, ...): 26/30 fidelity.
   - Recommended mix: hand-written spokes for pandas, polars and arrow, plus a capsule fallback.
5. **Lazy value.** It is realized only by pushing estimator needs into the plan before
   materializing, through a lookback hint (duckdb: 4× less peak memory, 2× faster). A hub makes
   that expressible. Converting to pandas first never can.

## 5. What is lost / caveats

- **Pandas semantics inside estimators.** Calendar-frequency arithmetic (MS, ME, B, Q) needs
  pandas offsets (or polars `dt.offset_by`). `future_time` only handles int, period and
  fixed-step datetimes natively. Pandas nullable and extension dtypes are mapped to float NaN in
  `values` and restored from schema (`null_cols`). Non-float columns live in `aux`, so kernels see
  only floats.
- **Memory model.** pandas ingress always allocates the time array. Unsorted input forces a full
  permutation copy, with ingress up to 46× slower. polars and arrow views are read-only, so
  in-place estimator code must copy. A hub that aliases user memory means the user must not
  mutate the input during fit (pandas views are writeable).
- **Private pandas API.** The fast "trusted freq" pandas view uses `_simple_new`, which may break
  across pandas versions.
- **Single process.** The hub materializes and has no out-of-core or distributed execution. dask
  ingress = compute(). This is fine for sktime's per-series estimators; it is not a dask
  replacement.
- **Scope of tests.** Fidelity was tested on 30 fixtures (DEV sizes). Timing was measured on a
  shared 4-core machine, with ratios taken within one process. The current path was not timed at
  10000 series in the forecaster bench (too slow for the dev budget). LARGE grid not yet run
  (`--grid large`).
- **Not measured:** sktime-wide integration cost (BaseForecaster, fh, cutoff on the hub); behaviour
  of non-Naive forecasters; the current path on unsorted input.
