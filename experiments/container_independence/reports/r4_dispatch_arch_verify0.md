# r4_dispatch_arch_verify0: adversarial verification of r4_dispatch_arch

Lens: open the cited sources, especially the Fugue and pandera internals, and check the claims
against them. Re-run the original scripts and write independent checks that take a different
code path.

Environments: `EXP_PY` (pandas 2.3.3), `EXP_PY3` (pandas 3.0.6), and the r4 private venv
`.../scratchpad/venvs/r4/bin/python` (fugue 0.9.7, pandera 0.34.1, polars 2.0.0,
narwhals 2.27.1). The machine is a shared 4-core box, so all ratios were measured in the same
process with min-of-3 `common.bench`.

## Commands run (from the worktree root)

```
# re-runs of the original scripts
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/r4_canonical_arrow.py --grid dev
PYTHONPATH=experiments/container_independence $R4_PY -W ignore experiments/container_independence/scripts/r4_pandera_backends.py --grid dev
$EXP_PY -W ignore experiments/container_independence/scripts/r4_dask_order.py
# independent checks (new)
PYTHONPATH=experiments/container_independence $R4_PY -W ignore experiments/container_independence/scripts/r4_dispatch_arch_verify0_fugue.py --grid dev
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/r4_dispatch_arch_verify0_offsets.py --grid dev
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/r4_dispatch_arch_verify0_mapgroups.py --grid dev
PYTHONPATH=experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/r4_dispatch_arch_verify0_dask.py
$EXP_PY  -W ignore experiments/container_independence/scripts/r4_dispatch_arch_verify0_capsule.py
$EXP_PY3 -W ignore experiments/container_independence/scripts/r4_dispatch_arch_verify0_capsule.py
# plus inline python -c probes (narwhals group_by on a pandas MI; narwhals collect backends; pandera lazy depth)
```

The results JSON files are `results/r4_dispatch_arch_verify0_{fugue_dev,offsets_dev,mapgroups_dev,dask}.json`.

## Source checks

- **pandera `api/base/schema.py:96-135`.** `register_backend` and `get_backend` work as
  claimed: `BACKEND_REGISTRY[(cls, type_)]`, an `inspect.getmro` walk, and lazy
  `register_default_backends`.
  - Re-measured `get_backend`: 0.90 µs (native) and 0.92 µs (narwhals). **Confirmed.**
  - LOC per backend re-counted (`cat backends/<b>/*.py | wc -l`): pandas 4633, polars 1968,
    ibis 1589, pyspark 2002, narwhals 3801. The original agent omitted one point: **the
    narwhals backend (3.8 kLOC) is nearly as large as the pandas backend and about twice the
    polars backend.** Converging on narwhals removed duplication *across* backends, but the
    generic backend was not small.
- **pandera `backends/narwhals/container.py`.** The quote at line 441 ("Index/MultiIndex is the
  one pandas concept Narwhals cannot express …") is verbatim. Lines 158-164 and 195-208 confirm
  that index coercion and validation are delegated to native pandas for pandas-like frames,
  and that index validation is *skipped with a warning* for non-pandas frames.
  - The same source also says, twice, that "**Narwhals preserves the pandas index** through
    its operations". This weakens the strong form of C2 (see the missed items below).
- **Fugue 0.9.7.**
  - `native_execution_engine.py:103-190` confirmed: `df.as_pandas()` followed by
    `safe_groupby_apply`, with `_map` resetting the index.
  - `fugue_duckdb/execution_engine.py:204-205` confirmed: `create_default_map_engine` returns
    `PandasMapEngine(self._native_engine)`.
  - `fugue_dask/execution_engine.py:158,167` confirmed: `safe_groupby_apply` and
    `df.native.groupby(...)`.
  - `fugue_polars` has only a DataFrame wrapper and a registry, with no execution engine.
    **Confirmed.**
- **pandas 3.0.6 `core/interchange/from_dataframe.py`.** The `Pandas4Warning` text recommends
  the Arrow PyCapsule Interface and Narwhals. **Confirmed.** PyPI JSON shows
  dataframe-api-compat at 0.2.7 (2024-04-26), the latest release. **Confirmed.**

## Experimental re-checks

### V1. Fugue overhead (claim .4). Asymmetric baseline

The original baseline loop computes `f(numpy)` and returns a dict. The Fugue UDF `fn_pandas`,
however, **builds a 1-row `pd.DataFrame` per group**, and the result is concatenated. That
user-function cost was charged to Fugue. The new script `r4_dispatch_arch_verify0_fugue.py`
adds a fair baseline: a pandas groupby loop that runs the same `fn_pandas(g.reset_index())`
plus `concat`.

| n_series×n_time | numpy loop | fair frame loop | fugue native | fugue / numpy loop | **fugue / fair loop** | fugue extra µs/group |
|---|---|---|---|---|---|---|
| 10×100 | 1.93 ms | 11.19 | 16.75 | 8.7× | **1.50×** | 556 |
| 100×100 | 7.45 | 80.34 | 96.96 | 13.0× | **1.21×** | 166 |
| 1000×100 | 61.6 | 784.1 | 934.4 | 15.2× | **1.19×** | 150 |

Fugue's own framework overhead is **about 150 µs per group, 1.2–1.5× an equivalent pandas
loop**, not 21–47×. The 21× figure is mostly the cost of building a pandas frame per group.
Fugue is still slower and still pandas-normalising. However, the "unsuitable for many small
series" conclusion follows from the per-group pandas round-trip, which sktime *also* pays
today, rather than from Fugue's overhead.

### V2. Fugue output type (claim .3)

- Polars input with a `fn(pl.DataFrame)` annotation returns **pandas**. The fn does receive a
  polars frame, converted per group.
- Polars input without partitioning also returns pandas.
- **Confirmed and strengthened:** Fugue never respects polars input, even when the function is
  written for polars.

### V3. Canonical-Arrow iteration (claim .8)

Re-run of `r4_canonical_arrow.py`: polars input gave 0.347 / 0.181 / 0.113 / 0.094× the loop.
The status-quo polars path was 4.5 / 3.3 / 3.2×, with `convert_to` taking 140 ms at 1000
series against 0.21 ms for the capsule. **The numbers reproduce.**

The independent script `r4_dispatch_arch_verify0_offsets.py` tests the same data in the same
process and finds three problems.

| n_series×n_time | A pd MI groupby loop | **B pd MI codes + np.split (no Arrow)** | C r4 Arrow canon | D polars rle (no pyarrow) | **E = C + per-instance pd.Series** |
|---|---|---|---|---|---|
| 10×100 | 1.87 ms (1×) | 0.172× | 0.352× | 0.313× | 0.693× |
| 100×100 | 7.57 | 0.107× | 0.181× | 0.163× | 0.831× |
| 1000×100 | 71.5 | **0.070×** | 0.103× | 0.086× | 0.746× |
| 10000×50 | 628 | **0.075×** | 0.095× | 0.078× | **0.816×** |

1. **The speedup is not due to Arrow.** The same offsets-plus-numpy-views idea applied
   directly to the pandas MultiIndex codes (B) is *faster* than the Arrow path. The polars-native
   run-length variant (D) is also faster. The gain comes from removing the per-group frame
   objects. Arrow, the PyCapsule interface and dictionary encoding are not what produces it.
2. **The headline numbers assume contiguous keys.** They use `arrow_runs(assume_contiguous=True)`,
   and nothing validates that assumption. On a shuffled polars input this path silently returns
   **wrong results for 100% of keys**: 9997 of 10000 keys were wrong at 10000×50, because
   900 → 499950 "runs" are produced. The safe path (lexsort) measured 0.134–0.158× in the
   original script's own output, but was not cited.
3. **Building pandas objects per instance erases most of the gain.** Handing each instance to a
   pandas-inner estimator as `pd.Series` with a `DatetimeIndex` brings the speedup down to
   **0.75–0.83×**, only a 1.2–1.3× gain. The original report listed this as an open question.
   It is now measured.

### V4. Engine-native per-group apply (claim .9) and expression IR (claim .11)

The script is `r4_dispatch_arch_verify0_mapgroups.py`.

| n | pd loop (numpy) | pd loop building pd frames | polars partition_by building pl frames | polars map_groups building pl frames | pandas groupby.agg(mean,last) | polars expr |
|---|---|---|---|---|---|---|
| 100×100 | 7.38 ms (1×) | 3.06× | 1.07× | 1.72× | 0.289× | 0.163× |
| 1000×100 | 69.2 | 3.10× | 1.16× | 1.74× | 0.137× | 0.041× |
| 10000×50 | 634 | 4.05× | 1.12× | **2.01×** | **0.064×** | 0.017× |

- **map_groups.** The ≈2× figure reproduces, but it is again an asymmetric comparison: the
  map_groups UDF builds a `pl.DataFrame` per group while the base loop does not.
  - With a like-for-like workload, map_groups is 0.50–0.56× a pandas loop that builds pandas
    frames, and 1.6–1.8× a polars partition_by loop that builds polars frames.
  - "map_groups is the slowest polars option" holds, but its overhead over partition_by is about
    1.7×, not "2× pandas".
- **Expression IR.** The 11–23× gain at 10k series is **not specific to an IR or to polars, ibis
  or duckdb**. Plain pandas `groupby(level=0).agg(["mean","last"])` is already 0.064×, which is
  15.6×. The decisive move is reformulating the forecaster as a vectorised aggregate. The
  original table contains this pandas row (0.087×), but the impact statement credits the gain
  to expression IR.

### V5. dask within-group order (claim .10)

The independent check `r4_dispatch_arch_verify0_dask.py` uses a row-id monotonicity test.

| n | npart | groups spanning partitions | out-of-order groups after groupby.apply | `set_index(sorted=True)` + `map_partitions` |
|---|---|---|---|---|
| 10×100 | 4 | 2 | 2 / 10 | 0 |
| 10×100 | 7 | 6 | **4 / 10** | 0 |
| 100×100 | 4 | 0 | 0 | 0 |
| 1000×50 | 8 | 0 | 0 | 0 |

- **Confirmed.** Disorder happens exactly for groups that span partitions.
- The "10–20%" rate is an artifact of the setup; it reached 40% with 7 partitions.
- A partition-aligned workaround exists: sort and set the index on the key, then
  `map_partitions` with a pandas groupby. It gave 0 violations, so presort inside the UDF is not
  the only option.

### V6. Arrow PyCapsule and MultiIndex (claims .6 and .7)

The capsule matrix reproduces on both pandas versions:
- pd.Series has the capsule only on pd3.
- `__dataframe_consortium_standard__` exists on pd2 and is gone on pd3.
- polars and narwhals LazyFrame and dask have no capsule.
- The buffer addresses of the polars↔pyarrow hand-off are identical, so it is zero-copy.

Claim .7 is right for pyarrow consumers only:
- A *pure* capsule read (`pa.RecordBatchReader.from_stream(pdf)`) carries the index levels
  plus pandas metadata, and `to_pandas()` restores `MultiIndex(['h0','time'])` on pd2 and pd3.
- This works because pandas' `__arrow_c_stream__` is literally `pa.Table.from_pandas(self)`.
  It is therefore *not* a zero-copy view in general; pd2 2-D blocks are copied.
- **Other capsule-era consumers drop the index.**
  - `duckdb.sql("select * from pdf")` sees only `['y']`.
  - `pl.from_pandas(pdf)` and `pl.DataFrame(pdf)` give `['y']`. Only
    `pl.from_pandas(..., include_index=True)` keeps the levels.
  - A pandas → arrow → polars → arrow → pandas round trip loses the metadata and returns a
    **RangeIndex with no names**. That is exactly the C2 symptom.
- "The Arrow route keeps level names" holds only if every hop is pyarrow. sktime would have to
  carry the level names in its own schema, as VectorizedDF already does, rather than relying on
  Arrow metadata.

### V7. pandera timings (claim .2)

The re-run reproduces the result directionally:
- At 10×100, pandas MI took 4.26 ms native against 9.07 ms with narwhals (2.1×).
- At 10000×50, the narwhals backend was 1.16× on pandas MI, 0.50× on pandas flat and 1.47× on
  polars eager. The original reported 1.27×.

Two methodology flaws:
- (a) The "detects_bad" MultiIndex test perturbs **column c0**, so it never exercises the
  delegated *index* validation.
- (b) The "polars lazy" columns measure **schema-only validation**. pandera's polars lazy default
  skips data checks; a direct probe showed `in_range` violated by 1e12 is *not* caught on a
  LazyFrame but is caught on the eager frame. The lazy timings are therefore not comparable
  work.

## Things missed or wrong in the original

1. **C2 is weaker than the report concludes.** On narwhals 2.27.1 with a pandas backend,
   `nw.from_native(pd_MI_df)` keeps the MultiIndex through row-wise ops (with_columns, filter).
   - `group_by('h0')`, where `h0` is an index level and not a column, *works*, both as an
     aggregate and as iteration. Each iterated sub-frame keeps `MultiIndex(['h0','time'])` with
     names, in appearance order, on pd2 and pd3.
   - This is undocumented leakage of pandas' "by may name a level" behaviour. `select('h0')`
     raises `ColumnNotFoundError` and `columns` shows only `['y']`, so it is unsupported and
     non-portable.
   - The conclusion is that "reset_index is forced" is not literally true. pandera's real lesson
     is "narwhals preserves the pandas index, but cannot *express* it", and pandera relies on the
     preservation.
2. **C1 was not tested directly.** It is true, and it can be overridden.
   - `nw.LazyFrame.collect()` defaults to polars for polars lazy input, pandas for dask, and
     pyarrow for duckdb.
   - `collect(backend="pandas"|"polars"|"pyarrow")` forces any of the three for all three
     inputs.
   - Dask and duckdb have no eager frame of their "own" format, so "respecting the original
     format" is moot for them.
3. **Asymmetric accounting recurs** in the Fugue (V1) and map_groups (V4) benchmarks: UDF
   output-frame construction is charged to the framework.
4. **The canonical-Arrow headline** relies on an unchecked contiguity assumption, credits Arrow
   for a speedup that pandas codes can achieve alone, and omits the pandas-estimator hand-off
   cost (V3).
5. **The pandera narwhals backend size (3.8 kLOC)** is relevant to the "narrow adapter is cheap"
   argument and was not discussed.

## Verdict summary

| claim | verdict |
|---|---|
| .2 pandera narwhals hybrid, MI delegation, timings | confirmed (with the methodology caveats in V7) |
| .4 Fugue 21–47× overhead | partially confirmed: about 1.2–1.5× vs a like-for-like loop |
| .7 MI survives the Arrow capsule | partially confirmed: pyarrow consumers only |
| .8 canonical Arrow 0.08–0.11× | partially confirmed: correct only for contiguous input, the gain is not due to Arrow, and it drops to 0.75–0.83× with pandas instances |
| .9 engine-native apply costs | partially confirmed: map_groups overhead overstated by output-frame construction |
| .10 dask order loss | confirmed; a partition-aligned workaround exists |
| .11 expression IR 11–23× | partially confirmed: pandas vectorised agg gets about 15.6× without any IR |
| .12 DataFrame API standard abandoned | confirmed |

Also confirmed by source or re-count, though outside the 8 selected claims: .1 (registry,
LOC, 0.9 µs), .3 (Fugue pandas normalisation; also returns pandas for fn(pl.DataFrame)) and
.6 (capsule matrix, zero-copy).
