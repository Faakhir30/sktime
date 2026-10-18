# r3_ts_libs_verify0: adversarial verification of the `r3_ts_libs` prior-art survey

Verifier lens: open the cited sources and installed package code, and re-run or independently re-implement the quantified claims. Date: 2026-10-18, cloud machine (4 cores, about 2 concurrent agents).

## Environment

- `venvs/r3v0` is my own venv: utilsforecast 0.2.17, statsforecast 2.1.1, pandas 2.3.3, polars 2.0.0, narwhals 2.27.1, numpy 2.5.4, dask 2026.8.0, duckdb 1.5.6. The original `venvs/r3` lacks dask and duckdb, so E6 could not be re-run there.
- `r3v0src` holds source-only installs (`--no-deps --target`): darts 0.48.1, aeon 1.6.0, pytimetk 2.5.3.
- The shared `expenv` (pandas 2.3.3) and `expenv_pd3` (pandas 3.0.6) were used for the sktime-side scripts.
- PyPI latest versions were re-checked with `https://pypi.org/pypi/<pkg>/json` and match the report: darts 0.48.1, aeon 1.6.0 (2026-09-18), pytimetk 2.5.3 (2026-10-16), utilsforecast 0.2.17 (2026-10-05), statsforecast 2.1.1, functime 1.0.0. Note that `u8darts` on PyPI stops at 0.41.0; `darts` is 0.48.1.

All commands run from the WT root, with `SP=/tmp/claude-0/-home-user-sktime/c9329e73-89f5-514d-ae5a-9180bf0e4a71/scratchpad`. Scripts that carry the `r3_ts_libs_verify0_` prefix and already existed from an interrupted earlier verifier pass were reviewed and re-run, not trusted.

```
PYTHONPATH=experiments/container_independence $SP/venvs/r3v0/bin/python -W ignore experiments/container_independence/scripts/r3_ts_libs_verify0_losses.py
PYTHONPATH=experiments/container_independence $SP/venvs/r3v0/bin/python -W ignore experiments/container_independence/scripts/r3_ts_libs_verify0_sfvia.py --grid dev
PYTHONPATH=experiments/container_independence $SP/venvs/r3v0/bin/python -W ignore experiments/container_independence/scripts/r3_ts_libs_verify0_hubvec.py --grid dev
PYTHONPATH=.:experiments/container_independence $SP/expenv/bin/python -W ignore experiments/container_independence/scripts/r3_ts_libs_verify0_iter.py --grid dev
PYTHONPATH=.:experiments/container_independence $SP/expenv/bin/python -W ignore experiments/container_independence/scripts/r3_ts_libs_verify0_rebuild.py --grid dev     # NEW
PYTHONPATH=.:experiments/container_independence $SP/expenv/bin/python     -W ignore experiments/container_independence/scripts/r3_ts_libs_verify0_c2.py   # NEW
PYTHONPATH=.:experiments/container_independence $SP/expenv_pd3/bin/python -W ignore experiments/container_independence/scripts/r3_ts_libs_verify0_c2.py   # NEW
```

The results are in `results/r3_ts_libs_verify0_{losses.json, sfvia_dev.json, hubvec_dev.json, iter_dev.json, rebuild_dev.json, c2.json}`.

---

## Verdicts

### r3_ts_libs.1: Nixtla keeps models container-free, with one adapter module. **Partially confirmed.**

Every count reproduces in `venvs/r3v0` site-packages:

| check | result |
|---|---|
| `utilsforecast/processing.py` length | 1,074 lines |
| `isinstance(` branches on pd or pl in that file | **62** (79 `isinstance(` in total) |
| `ProcessedDF` fields | `(uids, last_times, data, indptr, sort_idxs)`, at line 759 |
| `nw.` calls in `processing.py` | 0 |
| `statsforecast/core.py` pd/pl `isinstance` branches | 5 |
| `statsforecast/models.py` length | 7,103 lines |
| pandas or polars references in `models.py` | 0 |
| `^class` count in `models.py` | 41 |

Caveats the report does not mention:

- The 41 classes include the `_TS` base class, so there are 40 models.
- "Container-free" holds only at the model API. The helper modules that models.py imports do use pandas, and statsmodels, as numerical utilities:
  - `arima.py` uses `import pandas as pd` and `statsmodels.api`, and builds `pd.DataFrame` for intervals and coefficients (lines 1515, 2407).
  - `mstl.py` returns `pd.DataFrame(output)`.
  - `tbats.py` uses `pd.Series(y).rolling(...)`.
- So statsforecast models are not pandas-free; they are container-agnostic at the interface. For sktime this matters because a numpy inner mtype does not by itself remove the pandas dependency.

### r3_ts_libs.2: indptr iteration is about 100× faster than `groupby(level)` and about 300× faster than VectorizedDF. **Confirmed.** The numbers are about 20% lower on this re-run.

`r3_ts_libs_verify0_iter.py --grid dev`, interleaved in one process. Every variant is checked to give the same sum.

| 10k × 50 | time | ratio vs hub iteration |
|---|---|---|
| hub iteration only | 5.6 ms | 1 |
| build hub from MultiIndex codes, then iterate | 7.7 ms | 1.38 |
| `groupby(level=0)` | 511 ms | **92×** (claimed 106×) |
| `groupby` + `droplevel` | 1,327 ms | **238×** (claimed 300×) |
| real `VectorizedDF.items()` | 1,386 ms | **249×** (claimed 313×); **180×** vs build+iterate (claimed 179×) |
| `prepare_VectorizedDF` on polars_panel | 1,966 ms | 353× |

At 1k × 100 the ratios are 98×, 255× and 257×.

Methodology caveat, already stated in the report: the benchmark reads one value per series, so it measures only loop overhead.

New observation: sktime's own boundary, `prepare_VectorizedDF` on polars_panel, costs **1.97 s**. That is more than the whole `items()` pass of 1.39 s. On polars input the boundary conversion, not the loop, is the larger sktime cost.

### r3_ts_libs.3: a hub that rebuilds per-series pandas frames is 2.2× slower than VectorizedDF. **Refuted.**

The 2.2× slowdown is an artifact of how the original script rebuilt each frame: `pd.DatetimeIndex(slice, freq="D")`. That call makes pandas validate the freq by generating a `date_range` and comparing it, once per series.

My new script, `r3_ts_libs_verify0_rebuild.py`, tries other ways to rebuild. Every variant yields a per-series `pd.DataFrame` whose index **equals** the groupby slice, with `freqstr == "D"` and equal values. This is checked for every series against `VectorizedDF.items()` output.

| variant (ratio = time / `VectorizedDF.items()` time) | 100×100 | 1k×100 | 10k×50 |
|---|---|---|---|
| `VectorizedDF.items()`, absolute | 15.5 ms | 148 ms | 1,320 ms |
| original: `DatetimeIndex(slice, freq=)` (validates) | 1.57 | 1.60 | **1.96** (claimed 2.2) |
| `pd.date_range(t0, periods=n, freq)`: public API, freq taken from the schema | **0.64** | **0.64** | **0.73** |
| slice + set `_data._freq` from the schema: private API, no validation | **0.33** | **0.32** | **0.37** |
| slice + `pd.infer_freq` per series (today's `_enforce_index_freq` behaviour) | 1.33 | 1.30 | 1.55 |

What this means:

- An indptr hub that feeds **unchanged pandas-inner forecasters** is 1.4–1.6× faster than today's VectorizedDF using only public pandas API. With a trusted schema freq it is 2.7–3.1× faster.
- The cost that remains is per-series freq inference or validation, which the stateless schema already makes unnecessary because it stores `freq`.
- The report's conclusion that the hub "argues against putting a hub under unchanged pandas forecasters" is therefore wrong. The hub helps legacy forecasters too, modestly; it helps numpy-inner ones by far more.
- Caveat: the `date_range` variant assumes regular series. For irregular series the slice and `infer_freq` path applies, and that path is 1.3–1.55× slower.

### r3_ts_libs.4: going polars → pandas → polars is 1.24–1.77× slower than statsforecast's native polars path. **Partially confirmed.** The ratio reproduces, but its attribution and its relevance are overstated.

`r3_ts_libs_verify0_sfvia.py --grid dev` decomposes the gap. Results agree with native polars (`np.allclose`).

| grid | via-pandas / native polars | native pandas / native polars | conversion share of gap | absolute gap |
|---|---|---|---|---|
| 10×100 | 1.62 | 1.18 | 99% | 2.1 ms |
| 100×100 | 1.57 | 1.14 | 90% | 2.7 ms |
| 1k×100 | 1.45 | 1.20 | 68% | 7.1 ms |
| 10k×50 | 1.32 | 1.11 | 50% | 36 ms |

- **Range: 1.32–1.62×**, inside the claimed 1.24–1.77×.
- At 10k series, half of the gap is statsforecast's pandas code path being slower than its polars path (1.11×), not the conversion.
- Naive is a near-zero-cost model, so these ratios are upper bounds. With any real model the ratio tends to 1 (expected, unmeasured).
- This does not measure sktime's `convert_to`. That conversion is much more expensive: 1.97 s for `prepare_VectorizedDF` from polars at 10k×50, against 18 ms for `to_pandas` + `from_pandas` here.

### r3_ts_libs.6: narwhals `mae` keeps the input type on 6/6 engines; `mase` fails on a polars LazyFrame. **Partially confirmed.** The `mase` part understates the failure.

`r3_ts_libs_verify0_losses.py` changes the original probe in two ways. It uses a random forecast error and shuffles the rows; the original's constant error made `mae` equal 1 everywhere, so a wrong per-id result could not be detected. It also checks results against a numpy reference.

| engine | `mae` output type | `mae` values correct | `mase` | lag with `over(id, order_by=ds)` |
|---|---|---|---|---|
| pandas | DataFrame | ✓ | ok | ok |
| polars | DataFrame | ✓ | ok | ok |
| polars-lazy | LazyFrame | ✓ | **fails**: order-dependent, needs `order_by` | ok, LazyFrame kept |
| pyarrow | Table | ✓ | **fails**: only elementary aggregations in `.over` | **fails** |
| dask | dask DataFrame | ✓ | **fails**: `shift` unsupported in `over` | **fails**: "`over` with `order_by` is not yet supported in Dask" |
| duckdb | DuckDBPyRelation | ✓ | **fails**: order-dependent | ok, relation kept |

- `mae` is confirmed at 6/6 for type and values.
- `mase` fails on **4 of 6** engines, not only the polars LazyFrame.
- Even written correctly with `order_by`, per-series lag and diff through narwhals 2.27.1 fails on dask and pyarrow. Narwhals cannot portably express the order-dependent per-series operations that forecasting needs (lags, diff, last value by time) for dask or pyarrow. This weighs against idea (2) more than the report says.

**C1 side check** (same script, part D), `nw.from_native(x).collect(backend)`:

| input | default `collect()` | `backend="polars"` | `"pandas"` | `"pyarrow"` |
|---|---|---|---|---|
| polars LazyFrame | polars DataFrame | polars | pandas | pyarrow |
| dask | **pandas** | polars | pandas | pyarrow |
| duckdb | **pyarrow** | polars | pandas | pyarrow |

- The C1 claim is **true**: dask collects to pandas and duckdb to pyarrow by default.
- `backend=` can force any eager target, but no `collect` returns a dask or duckdb object. The only way to preserve those types is never to collect.

### r3_ts_libs.7: darts uses narwhals only at the boundary. **Partially confirmed.** The source facts check out; the timing figures are unverifiable.

The darts 0.48.1 source (`timeseries.py`, 6,294 lines) shows:

- `nw.from_native(df, eager_only=True, pass_through=False)` at two sites, lines 791 and 954. The report says "everywhere"; it is the two `from_*dataframe` entry points.
- `is_pandas()` appears 5 times, matching the report.
- `nw.maybe_get_index` appears at lines 797 and 962.
- `to_dataframe(..., backend=Implementation.PANDAS)` is the default, at line 1785.
- `self._values` and `self._time_index` are set at lines 348–352.

The darts timing figures (1–1.6 ms per series, 80–4,400×) were not re-run: the darts venv is gone, and the timings are not decision-critical.

### r3_ts_libs.8: a polars-native core wins only for estimators that can be written as expressions; per-series UDFs are 3.1–8.0× slower than the hub. **Refuted** as the basis for an architectural conclusion.

`r3_ts_libs_verify0_hubvec.py --grid dev`. All variants are checked equal: W1 to the statsforecast output (values and dates), W2 to `np.polyfit` within 1e-8.

| 10k × 50 | time | ratio |
|---|---|---|
| **W1 naive** | | |
| polars expression | 9.0 ms | 1 |
| StatsForecast(Naive), the report's "numpy hub" | 113 ms | 12.6× |
| numpy hub with **no per-series loop**, built by `utilsforecast.process_df` | 22 ms | 2.46× |
| numpy hub with no loop, hand-built indptr | 6.2 ms | **0.69×** (faster than polars) |
| **W2 linear trend** | | |
| hub loop + `np.polyfit` | 472 ms | 1 |
| polars `map_batches(polyfit)` | 2,480 ms | **5.3×** (claimed 3.1×) |
| polars closed-form OLS **expression** | 41 ms | **0.087×** (11.5× faster) |
| numpy `np.add.reduceat` closed-form OLS on the hub | 7.5 ms | **0.016×** (63× faster; 5.5× faster than polars) |

Why the claim fails:

- The "10× faster than the hub" compared a polars expression with the *whole StatsForecast pipeline*, which includes validation, a per-series Python model loop and output assembly. It did not compare against a numpy hub.
- A vectorized numpy hub matches or beats the polars expression.
- The "cannot be written as an expression" example is wrong: linear trend has a closed form, which runs 11.5× faster as a polars expression and 63× faster as a numpy segmented reduction.
- The real lesson is **vectorize across series**: segmented kernels such as `reduceat` over indptr. Which container holds the data is secondary.
- This undercuts recommendation (d), "polars or SQL expression fast paths". On an eager in-memory hub, numpy segmented kernels deliver the speed without per-backend code. Expression pushdown still matters for lazy or out-of-core engines (E5, not re-verified).

### r3_ts_libs.12: no sktime `y_inner_mtype` names `np.ndarray`. **Confirmed.**

Command: `grep -rn '"y_inner_mtype"' sktime/forecasting --include=*.py` finds **191** lines; the report says 187. None of them, nor the 3 lines after each (`-A3`), contains `np.ndarray`.

The report misses a nuance: sktime already defines `np.ndarray` (Series) and `numpy3D` (Panel) mtypes in `sktime/datatypes`. Recommendation (b) therefore needs no new datatype for single series, only a grouped (indptr) panel mtype plus time metadata.

### Extra spot check, not counted among the 8

- **r3_ts_libs.11, aeon and pytimetk: confirmed.**
  - aeon 1.6.0 `utils/data_types.py`: `VALID_SERIES_INNER_TYPES = ["np.ndarray", "pd.DataFrame"]`; `COLLECTIONS_DATA_TYPES` begins with numpy3D and np-list.
  - aeon `forecasting/base.py`: `"y_inner_type": "np.ndarray"`, `horizon: int`.
  - pytimetk 2.5.3 has **56**/42/6 unique `_*_pandas`/`_*_polars`/`_*_cudf` definitions (the report says 55 pandas), and 40 files call `normalize_engine(`.

---

## C2 side check (`r3_ts_libs_verify0_c2.py`; pandas 2.3.3 and 3.0.6, narwhals 2.27.1)

| check | result |
|---|---|
| `nw.from_native(mi_frame).to_native()` | MultiIndex intact (passthrough); the narwhals view shows only `["c0"]` |
| `nw.maybe_reset_index(nw_frame)` | **drops** the index levels: columns `["c0"]`, RangeIndex; `h0` and `time` are gone. This is likely the "names were gone" experience behind C2 |
| native `reset_index()`, then a narwhals op, then `set_index(["h0","time"])` | MultiIndex round-trips exactly, names included |
| cost of `reset_index` at 1k×100 | 3.6 ms (pandas 2) and 4.3 ms (pandas 3), which is **5–6%** of one `groupby(level=0)` iteration pass (57 ms and 94 ms) |

C2 as stated is directionally true: narwhals has no index. But the copy is cheap relative to the loop, and names survive if the reset is done natively or the level names are kept in the schema. `maybe_reset_index` is the wrong tool for this. The report cites hierarchicalforecast's use of `nw.maybe_reset_index` without noting that it discards the levels.

---

## Things the original missed or got wrong

1. **Claim 3 is a benchmark artifact.** Per-series freq validation causes it. With `date_range` and the schema freq, a hub rebuilding pandas frames is 1.4–1.6× faster than VectorizedDF, so the hub helps legacy pandas forecasters too.
2. **Claim 8 compares against the wrong baseline.** It uses the full StatsForecast pipeline in place of a vectorized hub, and its "non-expressible" example is expressible. Vectorized numpy segmented kernels on the hub beat polars expressions by 5.5× for linear trend. Recommendation (d) should be reframed as segmented numpy kernels over indptr, with backend expressions only for lazy engines.
3. **Order-dependent per-series operations fail on dask and pyarrow even with `order_by`** in narwhals 2.27.1, and `mase` fails on 4/6 engines, not 1/6. This is a hard limit for narwhals-based forecaster code.
4. **C1 is confirmed by measurement**, but the report deferred it. Default collect sends dask to pandas and duckdb to pyarrow; `backend=` reaches only eager types; preserving a lazy engine requires never collecting.
5. **`nw.maybe_reset_index` silently drops MultiIndex levels.** A native `reset_index` then `set_index` round-trips names, at about 5% of the iteration cost.
6. **sktime's own polars boundary, `prepare_VectorizedDF` from polars_panel, takes 1.97 s at 10k×50.** That is more than the iteration itself and about 100× the bare `to_pandas` + `from_pandas` cost. statsforecast's 1.3× ratio understates what sktime pays.
7. **statsforecast model helpers (`arima`, `mstl`, `tbats`) import pandas and statsmodels.** Container-free means container-agnostic at the interface, not free of pandas.
8. **Minor count differences:** 41 classes includes the `_TS` base; there are 191 tag lines, not 187; pytimetk has 56 pandas implementations, not 55; `eager_only` appears at 2 sites, not "everywhere".
9. **Not re-verified because their venvs are gone:** darts timings (E2), functime (E3/E7), fugue on dask (E4), lazy collect cost (E5). Claims .9 and .10 remain earlier-run numbers.
