# x9_cheap_paths — two low-cost options that need no forecaster rewrite

Workstream key: `x9`. Everything below was re-run in the cloud rerun (4 cores, 15 GB RAM, other agents
running at the same time). Absolute timings are noisy, so read the **ratios** (interleaved, min-of-repeats,
same process). Peak RSS comes from a fresh subprocess per (size, variant), measured with
`common.peak_rss_delta_mb` while the result is kept alive.

Environments: `EXP_PY` = pandas 2.3.3 / polars 2.0.0 / pyarrow 26 / duckdb 1.5.6. `EXP_PY3` = pandas 3.0.6
with the same libraries. statsmodels 0.15.0 and patsy 1.0.3 were made importable for the Theta and
ExponentialSmoothing tests. They were installed with `uv pip install --target $SCRATCH/venvs/x9_smtarget --no-deps`
and that folder was prepended to PYTHONPATH, so nothing was installed into the shared envs.

Scripts (all in `experiments/container_independence/scripts/`):

| script | what |
|---|---|
| `x9_arrow_convert.py --grid {dev,large}` | A1: polars→pandas and pandas→polars conversion time and peak RSS; zero-copy evidence |
| `x9_profile_convert.py --grid {dev,large}` | cProfile of `convert_to(polars_panel→pd-multiindex)` and a vectorized replacement for the monotonic check |
| `x9_arrow_compat.py` | A2: does sktime accept ArrowDtype frames (check / forecasters / pipelines / VectorizedDF), pd2 + pd3 |
| `x9_lazy_mtype_probe.py` | does sktime accept a polars LazyFrame as forecasting `y` |
| `x9_lazy_pushdown.py --grid {dev,large,tiny}` | B: eager load-all versus lazy/duckdb pushdown on parquet |

Results: `results/x9_arrow_convert_dev_{pd2,pd3}.json`, `results/x9_arrow_compat_{pd2,pd3}.{json,txt}`,
`results/x9_lazy_mtype_probe_{pd2,pd3}.json`, `results/x9_lazy_pushdown_dev_{pd2,pd3}.json`.

Commands (from the WT root, `SM=$SCRATCH/venvs/x9_smtarget`):

```
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x9_arrow_convert.py --grid dev
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/x9_arrow_convert.py --grid dev
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x9_profile_convert.py
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/x9_profile_convert.py
PYTHONPATH=.:experiments/container_independence:$SM $EXP_PY  -W ignore experiments/container_independence/scripts/x9_arrow_compat.py
PYTHONPATH=.:experiments/container_independence:$SM $EXP_PY3 -W ignore experiments/container_independence/scripts/x9_arrow_compat.py
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x9_lazy_mtype_probe.py   (and $EXP_PY3)
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/x9_lazy_pushdown.py --grid dev   (and $EXP_PY3)
```

---

## Headline

1. **The conversion copy is not where the time goes. The mtype check is.** `convert_to(polars_panel → pd-multiindex)` on 500k rows (10k series × 50) takes about 0.9–1.0 s. 96% of that is `sktime/datatypes/_adapter/polars.py::is_monotonically_increasing`, which loops over series in Python, builds one `pl.Series` per series and runs one polars query per series. The real polars→pandas conversion (`convert_polars_to_pandas`) takes 29–69 ms. A one-expression vectorized check gives the same answer in 33–35 ms, which is **25–28× faster** (measured). This fix needs no change to the architecture.
2. **"pandas API, Arrow memory" really is zero-copy for float values and the timestamp column.** Buffer addresses are identical, and `np.shares_memory` returns True. At 500k rows the flat conversion takes **0.05–0.13×** the time of sktime's numpy adapter (8–20× faster) and 0.29–0.73× the time of the plain numpy `to_pandas()`. However, **sktime rejects any frame whose index is Arrow-backed** (`Index[timestamp[ns][pyarrow]]`, or MultiIndex levels backed by Arrow). Raw `polars.to_pandas(use_pyarrow_extension_array=True).set_index(...)` therefore fails `check_is_scitype` on both pandas 2.3 and 3.0. The **hybrid** layout works: a numpy-backed (Multi)Index plus Arrow value columns. It passes 13/14 Series tests and 14/15 Panel tests, gives values identical to numpy, and the Arrow buffers stay zero-copy. Only Theta fails, because `y.values.reshape` is not available on ArrowExtensionArray.
3. **Arrow dtypes do not survive the pipeline.** Only 5–7 out of 12–13 outputs stay Arrow. NaiveForecaster(mean/drift), PolynomialTrend, reduction and ExponentialSmoothing all return float64, because numpy runs inside them. The gain is limited to the input boundary.
4. **Laziness is where the big win is, and sktime has no route for it today.** For the slice a forecaster needs (column `y`, 1000 of 10k series, last 50 of 200 points, from a 2M-row parquet):
   - sktime's current route takes 2.1–2.6 s and about 370–460 MB peak RSS. That route is: eager `pl.read_parquet`, then `convert_to(pd-multiindex)`, then select.
   - A polars LazyFrame with predicate and projection pushdown takes **6 ms and 30 MB**. That is about **350–440× faster and 12–14× less memory**.
   - duckdb takes 13–14 ms and 11 MB.
   - Even with the mtype-check cost removed (B1c, the raw adapter), eager load-all takes 200–450 ms and 380–415 MB. A lazy scan followed by conversion of only the slice (B5b) takes 15–26 ms and 44–52 MB.
5. **sktime does not accept a polars LazyFrame as forecasting `y`.** `polars_lazy_table` is a *Table* mtype only. There is no lazy Series, Panel or Hierarchical mtype. `NaiveForecaster().fit(lazy)` raises `TypeError: Unsupported input data type` for all three shapes, and so does `convert_to(lazy, 'pd-multiindex')`. `_convert.py` has orphan converter keys `('pl.LazyFrame', 'pd.DataFrame', 'Series')` that point to an mtype name that is not registered. Eager polars works end to end and *returns polars*: `NaiveForecaster.fit(pl_df).predict()` gives a `polars.DataFrame`.

---

## A1. Conversion cost and zero-copy (`x9_arrow_convert.py`)

The input is built directly in polars in the sktime `polars_panel` layout: `__index__h0` (String), `__index__time` (Datetime[ns]), `c0..` (Float64).

### Zero-copy evidence (1000 × 100, pd2; pd3 is identical)

Columns are compared by buffer address in the polars source versus the pandas result.

| column | `to_pandas()` (numpy) | `to_pandas(use_pyarrow_extension_array=True)` | then `pl.from_pandas` back |
|---|---|---|---|
| `c0` Float64 | copy | **same address (zero-copy)** | **zero-copy** (from both numpy and Arrow) |
| `__index__time` Datetime[ns] | copy | **zero-copy** | **zero-copy** (Arrow route only) |
| `__index__h0` String | object array (one Python str per row) | copy (polars string_view → Arrow large_string) | copy |

`np.shares_memory(pl_df["c0"].to_numpy(allow_copy=False), arrow_flat["c0"]…to_numpy(zero_copy_only=True))` returns **True**. The same check against the numpy route returns False. sktime's adapter output `c0` is at a different address, so it is a copy.

Hybrid layout (numpy MultiIndex + Arrow values): `c0` is zero-copy. The index is factorized into levels and codes, so building it is a copy by construction.

### Timings (dev grid; ratio = time / `sktime_adapter` for forward variants and / `ret_sktime_convert_to` for return variants)

pandas 2.3.3 (`results/x9_arrow_convert_dev_pd2.json`):

| rows (series×time×cols) | sktime convert_to | sktime adapter | np_flat | **arrow_flat** | np_mi | arrow_mi | **hybrid_mi** | ret sktime | ret np_flat | **ret arrow_flat** |
|---|---|---|---|---|---|---|---|---|---|---|
| 1k (10×100×1) | 5.9 ms (1.94) | 3.0 ms | 0.41 | 0.34 | 0.86 | 1.01 | 1.02 | 2.4 ms | 0.30 | 0.33 |
| 10k (100×100×1) | 16.6 ms (4.25) | 3.9 ms | 0.37 | 0.26 | 0.88 | 0.81 | 0.95 | 3.3 ms | 0.42 | 0.26 |
| 100k (1000×100×1) | 132 ms (8.3) | 15.9 ms | 0.22 | 0.10 | 0.90 | 0.49 | 0.65 | 11.3 ms | 0.57 | 0.13 |
| 500k (10000×50×1) | 1002 ms (14.5) | 69 ms | 0.19 | **0.05** | 1.02 | 0.45 | 0.73 | 46 ms | 0.63 | **0.085** |
| 100k ×10 cols | 133 ms (8.3) | 16.1 ms | 0.25 | 0.12 | 0.95 | 0.54 | 0.69 | 20.6 ms | 0.44 | 0.084 |

pandas 3.0.6 (`results/x9_arrow_convert_dev_pd3.json`):

| rows | sktime convert_to | sktime adapter | np_flat | arrow_flat | np_mi | arrow_mi | hybrid_mi | ret sktime | ret np_flat | ret arrow_flat |
|---|---|---|---|---|---|---|---|---|---|---|
| 1k | 6.9 ms (1.67) | 4.1 ms | 0.34 | 0.30 | 0.83 | 0.78 | 0.86 | 3.2 ms | 0.38 | 0.32 |
| 10k | 18.1 ms (4.2) | 4.4 ms | 0.33 | 0.29 | 0.89 | 0.85 | 0.95 | 3.8 ms | 0.33 | 0.29 |
| 100k | 128 ms (14.0) | 9.2 ms | 0.24 | 0.19 | 0.86 | 0.74 | 0.95 | 7.4 ms | 0.32 | 0.23 |
| 500k | 974 ms (33.2) | 29 ms | 0.17 | 0.13 | 1.05 | 0.86 | 1.07 | 18 ms | 0.38 | 0.23 |
| 100k ×10 cols | 133 ms (14.9) | 9.0 ms | 0.35 | 0.23 | 0.93 | 0.87 | 0.88 | 10.5 ms | 0.55 | 0.22 |

Peak RSS delta (MB, fresh process; source polars frame size in brackets):

| case | sktime convert_to | sktime adapter | np_flat | arrow_flat | np_mi | arrow_mi | hybrid_mi |
|---|---|---|---|---|---|---|---|
| 500k×1 [11.4 MB], pd2 | 39.7 | 48.3 | 18.6 | **4.0** | 53.6 | 34.8 | 36.9 |
| 500k×1, pd3 | 22.1 | 27.3 | 15.8 | **0.1** | 28.3 | 22.1 | 22.3 |
| 100k×10 [9.2 MB], pd2 | 26.7 | 22.0 | 11.0 | **1.7** | 24.0 | 4.6 | 6.6 |
| 100k×10, pd3 | 20.4 | 13.5 | 8.1 | **0.2** | 13.8 | 4.4 | 7.4 |

Reading:
- **Flat** Arrow conversion is close to free. It takes 2–4 ms at 500k rows and allocates almost no new memory, because the float and timestamp buffers are shared. Only the string key is copied.
- Once a pandas **MultiIndex** is needed, building the index (factorizing the keys) dominates. The hybrid layout is then only 0.65–1.07× the time of sktime's adapter. The memory saving appears only when value columns dominate: with 10 value columns the hybrid layout's peak RSS is 6.6/7.4 MB, against 22.0/13.5 MB for the adapter (about 2–3× less).
- The return direction (pandas Arrow → polars) is zero-copy for values and the timestamp column, at 0.08–0.23× of sktime's `convert_to(…, "polars_panel")`.

### Where `convert_to` spends its time (`x9_profile_convert.py`)

```
cProfile, convert_to(10k×50 polars_panel → pd-multiindex), pd2: 1.62 s total under profiler
  check_is_mtype → _check_polars_panel → check_polars_frame → is_monotonically_increasing   1.479 s
      10000 × pl.Series.__init__ 1.27 s, 10000 × DataFrame.select/collect ...
  convert → convert_polars_to_pandas                                                       0.140 s
```

`is_monotonically_increasing` (sktime/datatypes/_adapter/polars.py:14-40) runs `group_by(...).agg(time)` and then runs `pl.Series(val[0]).is_sorted()` **once per series in Python**. The vectorized replacement measured is
`obj.select((t >= t.shift(1).over(keys)).fill_null(True).all())`:

| env | input | current | vectorized | speedup |
|---|---|---|---|---|
| pd2 | 500k rows, sorted | 921.6 ms (True) | 33.0 ms (True) | ×28 |
| pd2 | unsorted at row 7 | 87.2 ms (False) | 35.3 ms (False) | ×2 |
| pd3 | sorted | 876.6 ms | 35.1 ms | ×25 |
| pd3 | unsorted at row 7 | 586.2 ms | 37.6 ms | ×16 |

The current check takes 949 / 885 ms, which is **96%** of `convert_to` (993 / 923 ms). The current check short-circuits on the first unsorted group in iteration order, so the "unsorted" timings vary between runs. This check runs on **every** fit, predict and update call for polars input, so it is a free win whatever architecture is chosen.

---

## A2. Does sktime accept ArrowDtype frames? (`x9_arrow_compat.py`)

### Setup

- Series: 60 points, daily. Panel: 4 series × 40.
- Inputs:
  - `np`: numpy baseline.
  - `arrow_vals`: `double[pyarrow]` values with a numpy DatetimeIndex.
  - `arrow_vals+arrow_ts_idx`: `Index[timestamp[ns][pyarrow]]`.
  - `np_vals+arrow_ts_idx`.
  - Integer index in numpy and Arrow variants.
  - `polars_to_pandas_arrow`: the literal output of `pl.to_pandas(use_pyarrow_extension_array=True).set_index(...)`.
  - `hybrid`: numpy index + Arrow values.
  - Nulls: NaN versus `pd.NA`.
- Panel inputs:
  - `P_arrow_vals`.
  - `P_arrow_vals+arrow_levels`: MultiIndex levels `large_string[pyarrow]` and `timestamp[ns][pyarrow]`.
  - `P_polars_to_pandas_arrow`.
  - `P_hybrid`.
- Tests:
  - `check_is_scitype`.
  - NaiveForecaster with strategies last, mean (w=7) and drift, plus `predict_interval`.
  - PolynomialTrend(deg=2).
  - `TTF[Differencer, Naive(mean)]` and `TTF[Detrender, Naive(last)]`.
  - `Differencer.fit_transform` plus `inverse_transform` round-trip.
  - `Detrender.fit_transform` and `Naive.update`.
  - `make_reduction(LinearRegression)`, ThetaForecaster(sp=1) and ExponentialSmoothing(trend=add).
  - For panels only: `prepare_VectorizedDF → as_list → reconstruct`.
- Every successful output is compared with the numpy baseline output (values allclose, index equal), to catch results that are silently wrong.

### Counts (pd2; pd3 is identical except where noted)

| input | OK | ERROR | OK but differs from numpy | outputs still Arrow |
|---|---|---|---|---|
| series / np | 14 | 0 | 0 | 0/13 |
| series / arrow_vals (numpy DatetimeIndex) | 13 | 1 (Theta) | 0 | 5/12 (pd3: 6/12) |
| series / arrow_vals + arrow **int** index | 13 | 1 (Theta) | 0 | 5/12 (pd3: 6/12) |
| series / hybrid (DataFrame) | 13 | 1 (Theta) | 0 | 5/12 (pd3: 6/12) |
| series / **arrow ts index** (values numpy or Arrow) | 1 (check → invalid) | 13 | – | – |
| series / literal `polars→pandas(Arrow).set_index` | 1 (check → invalid) | 13 | – | – |
| series_nulls / arrow_null (pd.NA) | 8 | 6 | 0 | 3/7 (pd3: 4/7) |
| series_nulls / np_nan | 8 | 6 | 0 | 0/7 |
| panel / P_np | 15 | 0 | 0 | 0/14 |
| panel / P_arrow_vals | 14 | 1 (Theta) | 0 | 6/13 (pd3: 7/13) |
| panel / P_hybrid | 14 | 1 (Theta) | 0 | 6/13 (pd3: 7/13) |
| panel / **P_arrow_levels** | 1 (check → invalid) | 14 | – | – |
| panel / literal P_polars_to_pandas_arrow | 1 (check → invalid) | 14 | – | – |

No cell returned a wrong value. Every OK result matched the numpy baseline in both values and index.

### What breaks and why

1. **Arrow-backed time index: rejected at the boundary.** `sktime/datatypes/_series/_check.py:51` and `_panel/_check.py:59` hard-code `VALID_INDEX_TYPES = (pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex)`, plus a check for integer dtype. `pd.Index(dtype=timestamp[ns][pyarrow])` is a plain `pd.Index`, so the message is `<class 'pandas.Index'> is not supported for obj, use one of (RangeIndex, PeriodIndex, DatetimeIndex) or integer index instead`. An Arrow *integer* index passes, because the integer-dtype check accepts it. This happens on pandas 2.3 and on 3.0. Fixing it is not just a matter of loosening the check: freq inference, `cutoff`, and the ForecastingHorizon arithmetic all assume DatetimeIndex or PeriodIndex. In practice, the timestamp level has to be turned into a numpy-backed DatetimeIndex. For the time level this is cheap, because Arrow `timestamp[ns]` and numpy `datetime64[ns]` use the same memory layout.
2. **VectorizedDF on invalid input crashes with an unhelpful error.** `prepare_VectorizedDF` → `coerce_to_multiindex` at `sktime/datatypes/_vectorize.py:75` does `metadata["scitype"]` on the `None` metadata returned when no mtype matches, which raises `TypeError: 'NoneType' object is not subscriptable`. It should raise the usual "unsupported input" message. This is a small bug, independent of Arrow.
3. **ThetaForecaster** fails at `sktime/forecasting/theta.py:230`: `_fit_trend(y.values.reshape(1, -1))` raises `NotImplementedError: ArrowExtensionArray does not support reshape`. This is a typical failure for code written against numpy. `y.to_numpy()` would fix it, but any of the hundreds of estimators may contain the same pattern.
4. **Arrow does not survive forecaster internals.** Naive(mean/drift), predict_interval, PolynomialTrend, reduction, ExponentialSmoothing and VectorizedDF on pandas 3 with a numpy panel all return `float64`. Pass-through paths keep `double[pyarrow]`: Naive(last), Differencer and its inverse, Detrender, update, and VectorizedDF reconstruct with Arrow values. On pandas 3.0, `TTF[Differencer, Naive(mean)]` also keeps Arrow, while on 2.3 it does not. Output dtype therefore varies by estimator and pandas version. A container-preserving design would need an explicit output conversion rather than rely on this.
5. **Nulls:** `pd.NA` in `double[pyarrow]` is detected as missing exactly like NaN. The set of errors is identical ("cannot handle missing data" in PolynomialTrend, Theta and ETS; sklearn "Input X contains NaN"), and the successful outputs match.
6. **VectorizedDF with hybrid / Arrow values:** `X_mi` keeps `double[pyarrow]` and slices keep a DatetimeIndex with `freq=<Day>`. The schema freq is `D`, and reconstruct returns Arrow with the index matching numpy. On pandas 3, string levels of a numpy panel come out as `str` and the time level as `datetime64[us]`. That is the pandas 3 default, not a regression.

### Return to the input container

`pl.from_pandas(arrow_flat)` is zero-copy for the float and timestamp columns (address check above) and takes 0.08–0.23× the time of sktime's `convert_to(…, "polars_panel")`. From a hybrid MultiIndex frame you first need `reset_index()`, which copies the levels. That route is 0.52–1.18× sktime's return conversion, so it is no faster. sktime already returns polars for polars input: `NaiveForecaster.fit(pl_df).predict()` → `polars.DataFrame`.

---

## B. Laziness and pushdown value (`x9_lazy_pushdown.py`)

### Setup

- Data: `results/x9_data.parquet`, 10,000 series × 200 days = 2M rows, 43.2 MB. Columns: `series_id` (str), `time` (timestamp[ns]), `y`, `x1`, `x2` (float64). The file is sorted, with 20 row groups of 100k rows each and statistics on. It is written by the script and deleted at the end.
- Task: column `y` for 1000 series, last 50 time points each. That is 50,000 rows, 2.5% of rows and 3 of 5 columns.
- Two selections:
  - **contiguous** ids: 2 of 20 row groups overlap, so min/max statistics can prune.
  - **random** ids: 20 of 20 row groups overlap.
- All 11 variants return the same 50,000 rows, the same sum(y) and the same first/last keys (`all_agree` = True for every variant).
- Threads are pinned to 4. The page cache is **warm**, so cold-cache I/O savings were not measured. On disk or object storage, pushdown would save more (unmeasured/expected).

### Results

Ratio = speedup versus B1, the sktime-today route. The polars plan shows `PROJECT 3/5 COLUMNS` and `SELECTION: series_id.is_in(...) & time >= cut` pushed into the parquet scan.

pandas 2.3.3 (`results/x9_lazy_pushdown_dev_pd2.json`):

| variant | contiguous: time | ×B1 | peak RSS Δ | random: time | ×B1 | peak RSS Δ |
|---|---|---|---|---|---|---|
| B0 pandas read_parquet all → set_index → select | 381 ms | 6.5 | 443 MB | 393 ms | 6.4 | 441 MB |
| **B1 pl.read_parquet all → sktime convert_to → select (today)** | **2467 ms** | 1 | **417 MB** | **2506 ms** | 1 | **461 MB** |
| B1b eager read all → polars filter → convert_to | 143 ms | 17 | 147 MB | 145 ms | 17 | 149 MB |
| B1c eager read all → raw adapter (no check) → select | 446 ms | 5.5 | 392 MB | 379 ms | 6.6 | 415 MB |
| **B2 polars scan_parquet + filter + select (pushdown)** | **5.6 ms** | **442** | **29.6 MB** | 17.6 ms | 142 | 34.8 MB |
| B2w polars lazy, per-series last-50 via rank().over() | 11.1 ms | 222 | 47.4 MB | 22.8 ms | 110 | 50.5 MB |
| **B3 duckdb WHERE id IN … AND time ≥ cut** | 12.6 ms | 195 | **10.9 MB** | 30.0 ms | 84 | 19.4 MB |
| B3w duckdb QUALIFY row_number() ≤ 50 | 29.6 ms | 83 | 24.7 MB | 44.5 ms | 56 | 31.6 MB |
| B4 pyarrow.dataset to_table(filter, columns) | 40.8 ms | 60 | 151 MB | 39.0 ms | 64 | 153 MB |
| B5 lazy pushdown → sktime convert_to | 106 ms | 23 | 50.3 MB | 117 ms | 21 | 54.8 MB |
| **B5b lazy pushdown → raw adapter** | **15.5 ms** | **159** | 43.8 MB | 26.0 ms | 97 | 49.2 MB |

pandas 3.0.6 (`results/x9_lazy_pushdown_dev_pd3.json`):

| variant | contiguous | ×B1 | RSS Δ | random | ×B1 | RSS Δ |
|---|---|---|---|---|---|---|
| B0 | 206 ms | 10.2 | 296 MB | 192 ms | 11.0 | 304 MB |
| B1 (today) | 2111 ms | 1 | 371 MB | 2101 ms | 1 | 367 MB |
| B1b | 155 ms | 13.6 | 151 MB | 147 ms | 14.3 | 152 MB |
| B1c | 203 ms | 10.4 | 383 MB | 202 ms | 10.4 | 383 MB |
| B2 | 6.0 ms | 350 | 29.7 MB | 17.7 ms | 119 | 34.9 MB |
| B2w | 12.7 ms | 166 | 47.0 MB | 25.9 ms | 81 | 53.4 MB |
| B3 | 14.0 ms | 151 | 10.9 MB | 27.7 ms | 76 | 18.8 MB |
| B3w | 30.4 ms | 70 | 24.7 MB | 45.5 ms | 46 | 29.9 MB |
| B4 | 39.4 ms | 54 | 147 MB | 41.1 ms | 51 | 153 MB |
| B5 | 104 ms | 20 | 51.0 MB | 118 ms | 18 | 55.8 MB |
| B5b | 14.6 ms | 144 | 46.8 MB | 24.8 ms | 85 | 52.2 MB |

### Breakdown of what sktime loses

- B1 versus B1c: the mtype check (monotonic loop) costs about 1.9–2.1 s of B1's 2.1–2.5 s.
- B1c versus B5b: collecting and converting everything up front, then selecting, costs **200–450 ms and 380–415 MB**, compared with **15–26 ms and 44–52 MB** when the selection is pushed into the scan. That is **8–29× time and about 7–9× peak memory**, with no change to forecaster internals.
- Row-group pruning (contiguous versus random ids) is worth another 3× for polars and about 2× for duckdb.
- Per-series "last N" without a global time cutoff (B2w/B3w, which also covers unequal lengths) stays at 11–45 ms.
- B4 pyarrow.dataset prunes columns but holds about 150 MB. It shows no contiguous/random difference, so it does not prune at row-group level here.

### Can sktime use this today? (`x9_lazy_mtype_probe.py`, identical on pd2 and pd3)

Registered lazy or out-of-core mtypes are: `dask_series`, `dask_panel` and `dask_hierarchical` (python type dask.dataframe), and **`polars_lazy_table` (scitype Table only)**.

For a LazyFrame in the `__index__` layout, with Series, Panel or Hierarchical shape:

| check | result |
|---|---|
| `check_is_scitype(lazy, ["Series","Panel","Hierarchical"])` | **invalid** (no mtype matches `polars.LazyFrame`) |
| `check_is_scitype(lazy, "Table")` | valid, `polars_lazy_table` |
| `check_is_mtype(lazy, "pl.DataFrame" / "polars_panel" / "polars_hierarchical")` | False |
| `convert_to(lazy, "pd.DataFrame" / "pd-multiindex" / "pd_multiindex_hier")` | `TypeError: No valid mtype could be identified for object of type LazyFrame` |
| `NaiveForecaster().fit(lazy, fh=[1,2])` | `TypeError: Unsupported input data type in NaiveForecaster` |
| `NaiveForecaster().fit(eager)` | OK, and **predict returns `polars.DataFrame`** |

`sktime/datatypes/_convert` contains converter keys `('pl.LazyFrame','pd.DataFrame','Series')` and `('pd.DataFrame','pl.LazyFrame','Series')`. The mtype `pl.LazyFrame` they refer to is **not registered**, so these are dead entries.

So for forecasting, sktime currently offers no lazy input at all. A user must `.collect()` first, which means the full B1/B1c cost. Making a lazy-aware boundary would look like this (design sketch, not measured):

1. Register lazy Series/Panel/Hierarchical mtypes. Their check only inspects the schema (`collect_schema()`) and must not collect.
2. Have `_check_X_y` and VectorizedDF push the known selections into the plan before collecting: instance subset, time window (`cutoff`, a `window_length` tag, or the `fh` range for X in predict) and columns (y plus the X columns used).
3. Collect, then convert only the slice with the zero-copy or hybrid path.

B5b measures the effect of steps 2 and 3.

---

## Conclusions (per approach)

| approach | effect (measured) | cost / limits |
|---|---|---|
| Vectorize `is_monotonically_increasing` for polars input | `convert_to` from polars about 25–28× cheaper at 500k rows (check 920 ms → 33 ms; check = 96% of convert_to). B1 would fall from about 2.1–2.5 s toward about 0.2–0.45 s (B1c; expected) | a few lines; no API change |
| "pandas API, Arrow memory", flat frame | 8–20× faster than sktime's numpy adapter at 500k rows (1.4–3.6× vs plain numpy to_pandas), near-zero extra RSS, true zero-copy for float and timestamp | **rejected by sktime** as soon as the index is Arrow-backed (pd2 and pd3) |
| Hybrid: numpy (Multi)Index + Arrow values | accepted by 13/14 (Series) and 14/15 (Panel) tests, values identical, VectorizedDF OK; 2–3× less peak RSS when value columns dominate (10 cols) | index factorization dominates time (0.65–1.07× adapter); Theta breaks (`.values.reshape`); Arrow survives in only about half of outputs; index still copied |
| Lazy-aware boundary (pushdown into scan, then convert the slice; B5b) | 8–29× faster and 7–9× less peak memory than eager load-all + raw convert (B1c). 85–159× faster and 7–10× less memory than sktime today (B1). Pure polars-lazy result (B2) is 119–442× faster and 11–15× less memory than B1 | needs new lazy mtypes and selection plumbing in `_check_X_y` / VectorizedDF; forecasters unchanged |
| duckdb as the pushdown engine | lowest memory (11–19 MB), 13–30 ms | SQL generation; Arrow result then needs conversion like polars |

## Caveats

- Timings come from a shared, loaded machine. The pd2 and pd3 absolute numbers for the same variant differ by up to about 2× (for example the adapter at 500k: 69 versus 29 ms), so ratios are the robust quantity. Peak RSS is sampled every 5 ms and may miss very short spikes.
- Part B used a warm page cache on a local file, so I/O savings under cold-cache or object-storage conditions are not measured. They are expected to be larger in favour of pushdown.
- B5b assumes the boundary knows the time window and instance subset. In fit, a forecaster usually needs all history unless a `window_length`-style tag bounds it, so the gain applies to predict/update with X, to subset fitting, and to forecasters with bounded memory.
- The compatibility matrix covers 13–15 estimator calls, not the whole estimator suite. Failures like Theta's `.values.reshape` are probably widespread (unmeasured).
- Statsmodels came from a `--target` dir (statsmodels 0.15.0, patsy 1.0.3, no deps).
