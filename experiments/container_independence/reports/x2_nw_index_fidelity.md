# x2: narwhals and the pandas MultiIndex (testing claim C2)

Workstream: `x2_nw_index_fidelity`. Environments: narwhals 2.27.1, polars 2.0.0, pyarrow 26, numpy 2.2.6, Python 3.12.

- **pd2**: `EXP_PY`, pandas 2.3.3, copy-on-write off (the default).
- **pd2+CoW**: the same env with `pd.options.mode.copy_on_write = True` (cost script only).
- **pd3**: `EXP_PY3`, pandas 3.0.6. CoW is always on, the default string dtype is `str` (arrow-backed), and the default datetime unit is `us`.

The claim being tested (C2):

> "No `MultiIndex`. Narwhals dataframes are columns only. The pandas path would have to `reset_index` into columns before grouping. That is a copy, and last time the index that came back was a `RangeIndex`, so level names were gone. The current code groups with `groupby(level=...)` and keeps those names."

## TL;DR verdicts

| Sub-claim | Verdict | Evidence (numbers) |
|---|---|---|
| (a) "Narwhals dataframes are columns only / no MultiIndex" | **True for the narwhals API. Wrong if read as "narwhals destroys the index".** | Index levels are not in `.columns` or `.schema`, and `nw.col("h0")` raises `ColumnNotFoundError`. But `nw.from_native(MI frame)` accepts the frame silently (no warning). Of 37 ops probed, 27 return the original MultiIndex, still aligned to the rows, on both pd2 and pd3. These are all the row-preserving or row-subsetting ones, 5 of them Series ops. Of the other 10, 8 change the index (aggregation, join, unpivot, `maybe_reset_index`/`maybe_set_index`, `group_by` when index names overlap columns) and 2 raise. Ops 1 |
| (b) "would have to `reset_index` before grouping" | **True for portable code.** | On the pandas backend `group_by("h0")` and `sort("h0")` silently resolve index level names. This is a pandas leak: the same call on a polars backend raises `ColumnNotFoundError`. Code that relies on it is not portable. Ops 1 |
| (c) "That is a copy" | **pd2 default: true. pd3 / CoW: false for the data. Either way it is small next to the iteration.** | pd2: data columns copied (`np.shares_memory` False). 500k rows x 10 cols: 65.8 MB traced peak for 38.1 MB of data. pd3 and pd2+CoW: data columns shared (True), and the allocation does not depend on n_cols (pd3: 11.9 MB traced + 7.7 MB arrow at 500k rows). `reset_index` costs 0.49 to 4.7% of one `VectorizedDF.items()` pass at 100k rows or more (cloud re-run: 0.7 to 8.2%; reset+set 1.9 to 18%, §8). Cost 2 |
| (d) "the index that came back was a RangeIndex, level names gone" | **Reproduced, but only through specific paths, none of them required.** | (i) `nw.maybe_reset_index` is `reset_index(drop=True)`: you get a RangeIndex and the level *values* are dropped too. (ii) `group_by().agg`, `join` and `unpivot` return a RangeIndex by design. (iii) If the index names equal column names, narwhals' pandas `group_by` calls `reset_index(drop=True)` internally. (iv) `reset_index()` -> ops -> `to_native()` with no `set_index`. `reset_index()` -> nw ops -> `set_index(names)` restored names, order and dtypes in 9/9 cases on pd2 and pd3. Fidelity 3 |
| (e) "current code groups with groupby(level=...) and keeps those names" | **True but incomplete.** | It groups by level *positions*. The final names are re-applied from the stored schema (`reconstruct`: `X_mi_index.set_names(self.X_mi_index_names)`), which is the same mechanism the narwhals recipe needs. The current code also does **not** keep `freq` through `groupby` on `from_tuples` MultiIndexes. It re-infers it for every slice, which makes `items()` 3.0 to 5.4x slower on pd2. On pd3 the re-inference also **mutates the caller's frame**. Freq 5 |

Bottom line: C2 is mostly accurate on facts but wrong on consequences. The index is not something narwhals has to carry. sktime's VectorizedDF schema already holds the instance keys, level names and freq. Store the level dtypes as well and re-apply all of them at the pandas boundary, and every case round-trips exactly. That was verified for 9 index variants x 2 narwhals backends (pandas, polars) x 2 pandas versions: 36/36 combinations, 11 checks each, 0 failures (Recipe 6). The costs that matter are not the `reset_index` copy. They are:

- rebuilding a time-indexed pandas object for each slice;
- attaching freq;
- iteration order on the polars backend.

All three are quantified below.

---

## 1. `nw.from_native` on a MultiIndex frame, and what `.to_native()` returns (sub-question 1)

Script: `scripts/x2_nw_ops_index.py`. Data: `make_panel(4, 6, n_cols=2, n_levels=1)` (index `h0, time`) plus a row-id column `rid`. "aligned" means `result.index.equals(orig.index.take(result.rid))`, i.e. the index still belongs to its row.

```
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x2_nw_ops_index.py
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/x2_nw_ops_index.py
```

Raw output: `results/x2_nw_ops_index_pd{2,3}.{txt,json}`. pd2 and pd3 give identical results except for the RangeIndex vs Index[int64] detail noted below.

- `from_native`: does not raise and emits no warnings. `columns == ['c0','c1','rid']`, so the levels are invisible. `nw.maybe_get_index` returns the MultiIndex `['h0','time']`. `to_native(nw.from_native(df)) is df` is True (zero-copy wrapper).
- **Index kept and aligned (MultiIndex, names `['h0','time']`)** for:
  - `select` (by name and by expression), `with_columns`, `rename`, `drop`, `filter`, `sort('c0')`, `head`, `tail`;
  - `df[2:7]`, `df[[0,3,5]]`, `drop_nulls`, `unique()`, `with_row_index`, `lazy().collect()`;
  - `concat` (vertical and horizontal), `c0.mean().over(g)`, and every group yielded by `group_by(col)` iteration;
  - Series ops: `filter`, `sort`, `*2`, `cum_sum`, `shift`.
- **RangeIndex, names `[None]`** for: `join` (inner and left), `unpivot`, `group_by(col).agg`, and `nw.maybe_reset_index`, which drops the levels. `maybe_set_index(['rid'])` replaces the index.
- **Silently work on pandas only** (they resolve index level names):
  - `group_by('h0').agg(...)` returns a frame with an `h0` column;
  - `group_by('h0')` iteration yields key `('s0',)`, and each group keeps its MultiIndex;
  - `sort('h0','time')` works.

  The same `group_by('h0')` on a polars-backed frame raises `ColumnNotFoundError`.
- **Raise:** `nw.col('h0')` and `filter(nw.col('time') > ...)` raise `ColumnNotFoundError`. Expressions cannot see levels.
- **Index names that overlap column names** (`df.reset_index().set_index(['h0','time'], drop=False)`): `group_by('h0')` iteration yields groups with `Index[int64]` (pd2) or **`RangeIndex`** (pd3), names `[None]`, and `aligned=False`. This comes from narwhals `_pandas_like/group_by.py`: `if set(native.index.names) & columns: native = native.reset_index(drop=True)` (narwhals issue #1907). `filter` on the same frame keeps the MultiIndex.

The narwhals docs say the same thing (concepts/pandas_index): the index is kept for most ops, `group_by` does not keep it, and the index is "not part of the Narwhals public API". MultiIndex is not mentioned.

## 2. Cost of `reset_index` / `set_index` vs the current `groupby(level=...)` iteration (sub-question 2)

Script: `scripts/x2_reset_index_cost.py --grid {dev,large} [--cow] [--ncols 1,10]`. Data: `make_panel_fast` over DEV_GRID, n_cols in {1, 10}, datetime time level. All timings come from the same process: min of 5 (`bench`). Memory is measured three ways:

- `tracemalloc` peak, which covers numpy/pandas buffers;
- the `pyarrow` pool delta, which covers pd3 `str` columns;
- an RSS sampler. RSS was unreliable here (0.0 MB for 12 MB allocations, because freed arena memory is reused), so the tracemalloc and arrow figures are the ones quoted.

```
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x2_reset_index_cost.py --grid dev
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x2_reset_index_cost.py --grid dev --cow
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/x2_reset_index_cost.py --grid dev
```

Raw: `results/x2_reset_index_cost_{pd2,pd2cow,pd3}_dev.{json,txt}`. Column key:

- `shared`: `np.shares_memory` between the original data columns and the reset frame's data columns (all columns).
- `tm`: tracemalloc peak in MB.
- `pa`: arrow pool delta in MB.
- `itSk`: `list(vec.items(X=X_mi))` with the current VectorizedDF.

| config | series x time | cols | rows | shared | data MB | tm reset MB | pa reset MB | reset ms | set_index ms | itSk ms | reset/itSk | (reset+set)/itSk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pd2 | 1000x100 | 1 | 100k | F | 0.76 | 6.30 | 0 | 3.45 | 8.76 | 208.3 | 1.7% | 5.9% |
| pd2 | 1000x100 | 10 | 100k | F | 7.63 | 13.17 | 0 | 11.06 | 9.86 | 237.1 | 4.7% | 8.8% |
| pd2 | 10000x50 | 1 | 500k | F | 3.81 | 31.48 | 0 | 22.40 | 71.17 | 1985.4 | 1.1% | 4.7% |
| pd2 | 10000x50 | 10 | 500k | F | 38.15 | 65.81 | 0 | 55.34 | 54.85 | 1569.9 | 3.5% | 7.0% |
| pd2+CoW | 1000x100 | 10 | 100k | **T** | 7.63 | 5.54 | 0 | 2.41 | 5.01 | 98.3 | 2.5% | 7.5% |
| pd2+CoW | 10000x50 | 10 | 500k | **T** | 38.15 | 27.67 | 0 | 15.76 | 34.03 | 1604.9 | 1.0% | 3.1% |
| pd3 | 1000x100 | 10 | 100k | **T** | 7.63 | 2.39 | 1.54 | 2.79 | 5.12 | 181.8 | 1.5% | 4.4% |
| pd3 | 10000x50 | 1 | 500k | **T** | 3.81 | 11.93 | 7.69 | 10.95 | 20.73 | 2049.0 | 0.53% | 1.6% |
| pd3 | 10000x50 | 10 | 500k | **T** | 38.15 | 11.93 | 7.69 | 12.01 | 29.08 | 2444.6 | 0.49% | 1.7% |
| pd3 | 10x100 | 1 | 1k | T | 0.01 | 0.03 | 0.02 | 1.09 | 1.56 | 2.28 | 48% | 116% |

Findings:

- **pd2 (CoW off): `reset_index` copies all data columns.** The traced peak grows with n_cols: 31.5 MB at 1 col, 65.8 MB at 10 cols (500k rows).
- **pd3 and pd2+CoW: data columns are not copied.** `shares_memory` is True. The peak does not depend on n_cols: 11.9 MB at both 1 and 10 cols on pd3, 27.7 MB on pd2+CoW. Only the index levels are materialised as new columns. At 500k rows on pd3 that is `h0` as an arrow `str` column (7.7 MB in the arrow pool) and `time` as `datetime64[us]` (~4 MB). `set_index` back to a MultiIndex is also zero-copy for data under CoW (`c0_shared_after_set_index_back=True`).
- Writing into the reset frame never changed the original in any config (`write_to_reset_frame_leaks_into_orig=False`).
- **In proportion**, from 100k rows up:
  - `reset_index` alone is 0.5 to 4.7% of one current `items()` pass;
  - `reset_index` + `set_index` is 1.6 to 8.8%.

  Only for tiny frames (1k rows) is it comparable (40 to 50%, ~1 ms absolute), because the current iteration is also only ~2 to 3 ms there.
- **`set_index` costs more than `reset_index`** (1.2 to 3.2x at 500k rows), because it factorizes the key columns again to build the MultiIndex.

## 3. Index fidelity round trip (sub-question 3)

Script: `scripts/x2_nw_roundtrip_fidelity.py`. Raw: `results/x2_nw_roundtrip_fidelity_pd{2,3}.{json,txt}`.

Cases. All are Panel `(h0, time)` with 4 series x 8 points unless noted:

- `dt_freqD`: built with `from_tuples`, so the level has no freq;
- `dt_freqD_product`: built with `from_product` over a `date_range`, so the level has freq D;
- `dt_freqMS`;
- `period_M` (PeriodIndex M);
- `int_range` (int64);
- `dt_tz_Berlin_h` (tz-aware, hourly);
- `categorical_inst` (instance level `CategoricalIndex`);
- `hier3_dt` (`h0, h1, time`, 8 series);
- `unsorted_inst` (instance order s2, s0, s3, s1).

Each path gets a **fresh** frame. `df.copy()` shares Index objects, and on pd3 path A mutates them (see §5).

Paths:

- **A**: current sktime: `prepare_VectorizedDF -> items -> reconstruct`.
- **A0**: plain `groupby(level, sort=False)` without freq inference.
- **B**: `reset_index -> nw group_by(inst) iteration -> per-group set_index(time) -> pd.concat(keys=schema row_index)`.
- **B2**: B, but `nw.concat` the groups, then `set_index(names)`.
- **C**: `reset_index -> nw filter + sort(names) -> to_native -> set_index(names)`.
- **D**: `nw.maybe_reset_index`.
- **D2**: `reset_index(drop=True)`.
- **D3**: `nw group_by.agg`.
- **D4**: `reset_index -> nw filter -> to_native` with no `set_index`.
- **E**: polars via narwhals (`pl.from_pandas(flat)` -> group_by -> concat -> sort -> `to_pandas` -> `set_index(names)`).
- **F**: sktime's own `convert_pandas_to_polars` -> `convert_polars_to_pandas`.

```
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x2_nw_roundtrip_fidelity.py
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/x2_nw_roundtrip_fidelity.py
```

Results. These are identical on pd2 and pd3 except for the freq column; see §5.

| path | names | level order | level dtypes | row order | values | notes |
|---|---|---|---|---|---|---|
| A current | ok 9/9 | ok | ok 9/9 | ok | ok | Slices get freq by `pd.infer_freq`. On pd3 this mutates the caller's index (§5). |
| A0 groupby(level) | ok | ok | ok | ok | ok | Slice freq is `None` for `from_tuples` MIs on pd2. |
| B nw iter + set_index + concat(keys=schema) | ok 9/9 | ok | ok 9/9 | ok 9/9 | ok | Slice freq `None`, except period. |
| B2 nw iter + nw.concat + set_index | ok | ok | ok | ok 9/9 | ok | pandas group order = first appearance. |
| C nw filter+sort + set_index(names) | ok | ok | ok | 8/9 | ok | `unsorted_inst` comes back sorted, as requested by `sort`. |
| **D `nw.maybe_reset_index`** | **[None]** | - | **int64 RangeIndex** | - | data ok | **The level values are gone too.** Columns are `['c0','c1']`. |
| D2 `reset_index(drop=True)` | [None] | - | RangeIndex | - | data ok | Identical to D. |
| D3 `group_by.agg` | [None] | - | RangeIndex | - | - | By design (an aggregation). |
| D4 no `set_index` | [None] | - | pd3: RangeIndex; pd2: Index[int64] | - | ok | The names live on as columns `h0, time`. |
| E polars via nw + set_index | ok 9/9 | ok | ok 9/9 | 8/9 | ok | Period survives (polars 2.0 Extension type). tz survives. Freq lost. Category order changes (see §5). |
| F sktime polars adapter | ok | ok | **8/9** | ok | ok | **`period[M]` comes back as `datetime64`.** The adapter calls `to_timestamp()` and never reverses it. Freq is re-inferred. |

**Where "RangeIndex came back, level names were gone" comes from:** D, D2, D3, D4 and the overlap case in §1 all produce it. The most likely real-world cause is `nw.maybe_reset_index`, whose name suggests "move the index into columns" but which actually drops it (narwhals `_utils.py`: `native_obj.reset_index(drop=True)`). The second most likely is calling `to_native()` after a `reset_index()` and forgetting the `set_index`. Neither is needed. `reset_index()` -> narwhals -> `set_index(names)` restores names, order and dtypes in every case, and the B path also restores row order through the schema's row index.

`nw.maybe_set_index` / `maybe_reset_index` / `maybe_get_index` do not solve this. `maybe_get_index` returns the native MultiIndex, which is useful for reading the schema. `maybe_set_index(df, names)` is just `native.set_index(names)` on pandas and a no-op on other backends. `maybe_reset_index` destroys the levels.

## 4. narwhals `group_by` iteration on a pandas-backed frame (sub-question 4)

Script: `scripts/x2_nw_groupby_iter_bench.py --grid {dev,large} [--ncols N]`. Input: `make_panel_fast` with the **instance order permuted**, so first-appearance order differs from sorted order. Methods are interleaved, min of 5 (min of 3 at 500k rows). Raw: `results/x2_nw_groupby_iter_bench_pd{2,3}_dev_c1.{json,txt}`.

```
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x2_nw_groupby_iter_bench.py --grid dev
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/x2_nw_groupby_iter_bench.py --grid dev
```

Methods (see the script docstring): `pd_level` = `groupby(level=0, sort=False)` + droplevel; `sk_items` = current `VectorizedDF.items`; `pd_col` = pandas groupby on the reset (flat) frame; `nw_pd` = narwhals `group_by("h0")` iteration on the flat pandas frame (`nw_pd_native` + `to_native`); `nw_pd_recipe[_infer|_ctor]` = reset_index -> nw group_by -> rebuild time-indexed pandas slice (schema freq / per-slice infer_freq / direct DatetimeIndex ctor); `nw_pd_recipe_fast` = `protos/x2_index_recipe.iter_slices`; `nw_offsets_iloc` = narwhals computes group boundaries only, slices are `iloc` on the original MI frame; `pl_iter_mo` / `pl_partition` = polars `group_by(maintain_order=True)` / `partition_by(maintain_order=True)`; `nw_pl` = narwhals `group_by` on a polars frame. Cloud re-run (4 cores), n_cols=1, ms, min of repeats:

| pandas | size | pd_level | sk_items | pd_col | nw_pd | nw_pd_recipe | nw_pd_recipe_fast | nw_offsets_iloc | pl_iter_mo | pl_partition | nw_pl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.3.3 | 10x100 | 2.09 | 2.14 | 1.35 | 1.27 | 9.78 | 4.77 | 4.47 | 1.22 | 0.93 | 1.54 |
| 2.3.3 | 100x100 | 12.20 | 11.07 | 4.25 | 4.32 | 80.33 | 24.45 | 13.71 | 5.98 | 3.27 | 5.75 |
| 2.3.3 | 1000x100 | 100.66 | 104.28 | 31.52 | 33.48 | 753.58 | 210.41 | 110.41 | 37.99 | 23.78 | 29.23 |
| 2.3.3 | 10000x50 | 1328.38 | 1175.82 | 493.02 | 518.42 | 9331.30 | 2672.75 | 1522.97 | 361.45 | 241.72 | 456.24 |
| 3.0.6 | 10x100 | 3.03 | 2.92 | 1.43 | 2.02 | 17.77 | 8.10 | 7.78 | 1.39 | 0.86 | 1.61 |
| 3.0.6 | 100x100 | 12.19 | 13.30 | 4.17 | 7.08 | 142.07 | 46.88 | 17.84 | 3.00 | 3.08 | 5.40 |
| 3.0.6 | 1000x100 | 136.09 | 125.39 | 38.74 | 77.96 | 1566.07 | 416.63 | 131.55 | 23.45 | 24.57 | 35.12 |
| 3.0.6 | 10000x50 | 1535.22 | 1767.42 | 694.30 | 1151.61 | 17424.86 | 5413.41 | 1798.50 | 451.10 | 213.96 | 488.48 |

Key ratios (from 100 series up):

- **Raw narwhals iteration is cheap**: `nw_pd/pd_col` = 1.02-1.06 (pd2), 1.66-2.01 (pd3); `nw_pd/pd_level` = 0.33-0.39 (pd2), 0.57-0.75 (pd3). Grouping a flat frame by a column is *faster* than `groupby(level=...)` + droplevel.
- **The cost is rebuilding time-indexed pandas slices**: naive recipe (`drop` + `set_index` + freq) is 7.2-7.9x `sk_items` (pd2), 9.9-12.5x (pd3). Per-column numpy ctor + global-grid freq (`nw_pd_recipe_fast`) cuts this to 2.0-2.3x (pd2), 3.1-3.5x (pd3). Using narwhals only for boundaries and `iloc` on the original MI frame (`nw_offsets_iloc`) is 1.02-1.34x `sk_items` (parity at >=1000 series).
- **Polars**: `partition_by(maintain_order=True)` is 0.18-0.27x (pd2 run) / 0.14-0.25x (pd3 run) of `pd_level`.
- **Order**: all pandas-backed methods (incl. narwhals) yield groups in first-appearance order (= `sort=False`); keys are 1-tuples `('s0',)` in narwhals vs scalars in pandas. **`nw_pl` (narwhals on polars) order is neither first-appearance nor stable over 5 runs** — narwhals' `group_by` iteration does not request `maintain_order`, so R5 (iterate in schema order) is required.

## 5. freq and Period (sub-question 5, plus a side finding)

### freq

- **pandas does not keep freq through a MultiIndex unless the level has it.** In the `from_tuples` cases the time level, the `xs` slices and the `groupby` slices all have `freq=None` (A0, pd2). With `from_product(date_range)`, the level carries `freq=D` and so do the slices (`results/x2_freq_mutation_probe.txt`).
- **Every columns-based path (B, B2, C, E) loses freq**, because `set_index` from a column gives `freq=None`. The same is true after a polars round trip (E). sktime's adapter (F) gets it back by calling `pd.infer_freq` on the level.
- **The current sktime path re-infers freq for every slice** (`_enforce_index_freq`). Cost, from `scripts/x2_freq_attach_cost.py` (raw `results/x2_freq_attach_cost_pd{2,3}_dev.{json,txt}`):

| per slice (us), pd2 | 10x100 | 100x100 | 1000x100 | 10000x50 |
|---|---|---|---|---|
| `items()` on an MI whose level has freq | 374.7 | 328.3 | 184.6 | 320.9 |
| `items()` on the same data built with from_arrays, so infer runs per slice | 1396.1 | 1031.1 | 997.2 | 960.2 |
| **slowdown** | **3.7x** | **3.1x** | **5.4x** | **3.0x** |
| attach freq: `DatetimeIndex` + `pd.infer_freq` | 817.7 | 522.1 | 495.2 | 654.4 |
| attach freq: `DatetimeIndex(values, freq=f)` (validates) | 393.0 | 287.0 | 279.3 | 302.0 |
| attach freq: `date_range(start, periods, f)` + `array_equal` | 243.0 | 162.2 | 121.3 | 152.6 |
| **attach freq: slice of one global grid + `array_equal`** (recipe) | **118.3** | **74.9** | **74.6** | **82.6** |

The global-grid attach is 6.6 to 7.9x cheaper than `infer_freq` per slice on pd2, and 4.5 to 5.8x cheaper on pd3. On pd3 the `items()` "no freq" row is about the same as the "level freq" row (ratios 0.95 to 1.24). That is because of the mutation described next: after the first slice's inference, the shared level already has freq.

### Side finding: on pandas 3, `VectorizedDF.items()` mutates the caller's frame

Script `scripts/x2_freq_mutation_probe.py`, output `results/x2_freq_mutation_probe.txt`:

```
pandas 3.0.6
make_panel: ... fresh groupby slice freq=None; slice index shares DatetimeArray with MI level: True
  after `slice.index.freq = 'D'`: MI levels[-1].freq=<Day>; xs-slice freq=<Day>
prepare_VectorizedDF returned the caller's object (X_mi is df): True
caller df (time-level freq, xs freq) before items(): (None, None)  after: (<Day>, <Day>)
caller frame mutated by VectorizedDF.items(): True
```

On pd2 the slice does not share the array, and the caller is not mutated.

Mechanism: on pd3, `group.droplevel(...)` returns an index whose `DatetimeArray` is the MultiIndex level's own array. `_enforce_index_freq` then sets `item.index.freq = ...`, which writes `_freq` into that shared array. The "stateless" VectorizedDF therefore leaves state behind on the user's object under pandas 3. It is usually a correct freq, but it is still a side effect, and it also makes timing comparisons depend on the order they run in. The fidelity script uses fresh frames for this reason. Not fixed here: the workstream does not modify `sktime/`.

### Period

- narwhals has **no Period dtype**.
  - On a pandas backend the column's narwhals dtype is `Unknown`. Ops pass straight to pandas and work: `sort`, `filter ==`, `diff`, `dt.year`, `group_by.agg(max)` (which returns `period[M]`).
  - `to_polars()` and `pl.from_pandas` produce `Extension('pandas.period', Int64, {"freq": "M"})`, and `to_pandas()` turns that back into `period[M]`. This was not the case in older polars.
  - **On a polars backend narwhals reports `Unknown`.** `sort` and `group_by.agg(max)` work, but `filter(==)` raises `NarwhalsError`, and `diff()` and `.dt.year()` raise `InvalidOperationError`.
- sktime's polars adapter turns Period into `datetime64` and returns `datetime64` (path F, dtype mismatch).
- Recipe rule R6: keep `period_freq` in the schema, convert to timestamps before leaving pandas, and convert back with `to_period(freq)` at the boundary.

### Other dtypes

- **tz-aware:** preserved on every path and backend. narwhals reports `Datetime(time_zone='Europe/Berlin')`.
- **int time:** preserved.
- **pd3:** `datetime64[us]` and `str` round-trip unchanged. polars returns `us` and `str` on pd3, and `ns` and `object` on pd2.
- **categorical instance level:** the dtype survives, but after a polars round trip the **category order** becomes appearance order: original `['s0','s1','s2','s3']`, after polars sort + `to_pandas` `['s1','s0','s2','s3']`. pandas reports `dtype == original` as True, because unordered categoricals compare as sets, so this is silent. The recipe re-codes to the original categories.

## 6. Minimal correct recipe

Prototype: `protos/x2_index_recipe.py`. Verifier: `scripts/x2_recipe_fidelity.py`. It runs every §3 case on a pandas backend and on a polars backend. The polars run sorts by a data column first, to scramble rows on purpose. It compares:

- `long_to_mi(...)` against the original: names, level dtypes, row order, values, `DataFrame.equals`;
- `iter_slices(...)` against `VectorizedDF.items`: order, values, index values, name, dtype, freq.

```
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x2_recipe_fidelity.py
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/x2_recipe_fidelity.py
```

Result: **18/18 combinations on pd2 and 18/18 on pd3, 11 checks each, 0 failures** (`results/x2_recipe_fidelity_pd{2,3}.{json,txt}`). A separate check, `scripts/x2_verify_recipe.py`, shows the per-slice constructor yields slices identical to `items()` (values, freq, name, dtype) on both pandas versions.

```python
# schema = what VectorizedDF already stores + level dtypes (+ period freq, + grid)
names  = list(df.index.names)                         # R3
cols   = [f"__index__{n if n is not None else i}" ...]  # R2: never collide with data cols
dtypes = [df.index.get_level_values(i).dtype ...]     # R3
freq   = level freq | xs(first).freq | infer once      # R4 (VectorizedDF.freq)
flat   = df.rename_axis(cols).reset_index()           # R1: native reset_index, NOT nw.maybe_reset_index
nf     = nw.from_native(flat)                         # (polars: period -> timestamp first, R6)
... narwhals ops ...
# back: MultiIndex.from_arrays([restore_dtype(col, dtype) ...], names=names)
#       categorical -> CategoricalIndex(categories=orig.categories); period -> to_period(freq)
# slices: group_by(inst_cols) -> dict -> yield in schema.row_index order (R5: polars order is arbitrary)
#         values: np.column_stack([g[c].to_numpy() ...]) -> DataFrame(copy=False)
#         time index: cut from one global date_range(freq) grid, keep only if array_equal (R4)
```

Rules:

- **R1.** Use the native `reset_index()`, never `nw.maybe_reset_index`.
- **R2.** Prefix the level columns (`__index__`, the convention sktime's polars mtypes already use) so index names can never overlap data columns. Overlap triggers narwhals' internal `reset_index(drop=True)` in `group_by`.
- **R3.** Restore names, level order and dtypes from the schema. Aggregation, join and unpivot results always come back with a RangeIndex.
- **R4.** Restore freq from the schema. Do not infer it per slice.
- **R5.** Iterate in schema order.
- **R6.** Period is pandas-only.

## 7. Caveats

- Timings were taken on a shared 16-core machine with about 14 other agents running, so absolute ms are noisy. Every ratio compares methods measured in the same process, interleaved, min-of-repeats. DEV_GRID only, at most 500k rows. The scripts accept `--grid large` for the serial benchmark phase.
- RSS deltas were unreliable for allocations of 12 MB or less (freed arena memory is reused), so memory claims use tracemalloc (numpy/pandas) plus the pyarrow pool (pd3 strings). Python object overhead of `object` string columns on pd2 is shared with the level and not counted. `memory_usage(deep=True)` would over-count it.
- Only Panel data was used for cost and iteration; hierarchical data was covered only in the fidelity tests (`hier3_dt`). Only datetime time levels were benchmarked. The per-slice cost for Period and int indexes is unmeasured; it is expected to be similar or lower, since no freq attach is needed.
- narwhals 2.27.1 and polars 2.0.0 only. Polars' Period extension-type round trip and narwhals' internal `reset_index(drop=True)` on name overlap may differ in other versions.
- The recipe prototype is not wired into sktime. Plugging it into `VectorizedDF`/`BaseForecaster` is a separate workstream.
- `iter_slices` builds per-slice pandas objects. It does **not** show what a natively polars-consuming forecaster would cost. On polars, `partition_by(maintain_order=True)` is 0.15 to 0.2x of `pd_level` (see §4).

## 8. Cloud re-run (2026-10-18, 4 cores, ~2 concurrent agents)

Re-ran the three headline scripts with the commands in §2, §4 and §6 (both EXP_PY and EXP_PY3).

- `x2_recipe_fidelity.py`: unchanged. 18 combinations x 11 checks = 198/198 True, 0 False, on both pd2 and pd3.
- `x2_nw_groupby_iter_bench.py --grid dev`: this run produced the §4 table. In the earlier run that section was a placeholder.
- `x2_reset_index_cost.py --grid dev` (pd2 default and pd3):
  - Memory is identical to §2. pd2 is not shared: 65.8 MB traced at 500k x 10 and 31.5 MB at 500k x 1. pd3 is shared: 11.9 MB traced + 7.7 MB arrow, regardless of n_cols.
  - The ratios moved a little:

| config | rows x cols | reset ms | set ms | itSk ms | reset/itSk | (reset+set)/itSk |
|---|---|---|---|---|---|---|
| pd2 | 100k x 1 | 4.96 | 10.31 | 104.99 | 4.7% | 14.5% |
| pd2 | 100k x 10 | 8.80 | 10.47 | 107.11 | 8.2% | 18.0% |
| pd2 | 500k x 1 | 18.25 | 57.35 | 1397.15 | 1.3% | 5.4% |
| pd2 | 500k x 10 | 53.68 | 54.04 | 1308.03 | 4.1% | 8.2% |
| pd3 | 100k x 1 | 3.33 | 6.61 | 124.97 | 2.7% | 8.0% |
| pd3 | 100k x 10 | 3.90 | 6.32 | 127.39 | 3.1% | 8.0% |
| pd3 | 500k x 1 | 11.19 | 18.48 | 1575.42 | 0.71% | 1.9% |
| pd3 | 500k x 10 | 14.57 | 24.28 | 1665.44 | 0.87% | 2.3% |

  Updated range, for frames of 100k rows or more: `reset_index` is **0.7 to 8.2%** of one `items()` pass (earlier: 0.49 to 4.7%), and `reset_index` + `set_index` is **1.9 to 18%** (earlier: 1.6 to 8.8%). The qualitative conclusion does not change: the copy is small compared with the iteration, and on pd3 it does not grow with n_cols.
  - The pd2+CoW configuration was not re-run.
