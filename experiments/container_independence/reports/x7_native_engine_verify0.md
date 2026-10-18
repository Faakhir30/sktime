# x7_native_engine — adversarial verification (verify0)

Lens: fairness and correctness. I checked numeric equality across implementations with an independent
reference, checked that the lazy and duckdb paths really execute, and re-measured the speedups.
Environment: `EXP_PY` (pandas 2.3.3, polars 2.0.0, narwhals 2.27.1, duckdb 1.5.6, sklearn 1.9.1),
1 thread, on a shared 4-core box. No files under `sktime/` were modified.

## Commands run (from the worktree root, `PYTHONPATH=.:experiments/container_independence`)

| # | command | output |
|---|---|---|
| R1 | `$EXP_PY -W ignore .../scripts/x7_native.py --sizes 100x100,1000x100,10000x100 --baseline-max 1000 --threads 1 --tag _verify0` | `results/x7_native_engine_verify0_rerun.{log,json}` |
| R2 | `$EXP_PY -W ignore .../scripts/x7_native_engine_verify0_indep.py --grid dev` (new) | `results/x7_native_engine_verify0_indep_dev.{log,json}` |
| R3 | `$EXP_PY -W ignore .../scripts/x7_native_engine_verify0_nw.py` (new) | `results/x7_native_engine_verify0_nw.log` |
| R4 | `X7_SCRATCH=<scratch> $EXP_PY -W ignore .../scripts/x7_hook_demo.py --grid dev --threads 1` | `results/x7_native_engine_verify0_hook_rerun.log` (the original json was restored afterwards) |
| R5 | `.../x7_native.py --parquet --threads 1 --repeat-parquet 2 --engines <all except sktime_today>` | `results/x7_native_engine_verify0_parquet.{log,json}` |
| R6 | inline warm repeated parquet timing in one process | `results/x7_native_engine_verify0_parquet_warm.log` |
| R7 | `.../x7_native.py --threads 1 --fitted-state last 2000 100` | `results/x7_native_engine_verify0_fitted_state_2k.log` |
| R8 | `.../x7_family_count.py` re-run, plus source spot checks | counts identical |

## Verdicts

### x7.1: pandas groupby is 382–1895× faster than the sktime loop at 10k (and 182–876× at 100). PARTIALLY CONFIRMED

R1 at 1k series: a vs f = 960 (last), 1705 (snaive7), 424 (mean10), 447 (drift), 376 (ols). At 100
series: 160–949×. These match the original figures (1k: 356–1824×). I did not re-run the 10k
baseline, which takes about 10 minutes. The per-series cost is constant (about 13 ms per series for
`last`: 13.3 s at 1k against 124 s at 10k in the original), so 10k extrapolates the same way.

**Equality was confirmed independently (R2).** I wrote a separate reference: a per-series Python loop
on raw arrays, with `np.polyfit` on centred x for OLS. It ran on a fresh seed (42, 300 series). Every
engine a/b/c/c_s/d/e/e2 and the f baseline matched for all 5 methods (max_abs ≤ 4.9e-11). The f
baseline here uses tol=1e-12.

Caveat on the 100-series figure: the original range is "182–876", but my R1 values are 160–949 and
the original's own table gives 182–1218 depending on method. These are noise-level differences.

**Caveat: shuffled input.** The vectorized engines a/b/c/e2 all rely on positional "last"/tail. On
input whose rows are not sorted by time, they silently return wrong answers (R2 part5: time mismatch,
max_abs 9.8). Only duckdb's `arg_max(c0, time)` is order-robust. Stock sktime raises a TypeError on
such input, so it does not give a wrong answer.

### x7.2: engines add 1.5–11× over pandas groupby at 10k. PARTIALLY CONFIRMED (methodology asymmetry)

R1 speedup vs (a) at 10k, compared with the original:

| engine | R1 | original |
|---|---|---|
| polars eager (b) | 2.3–6.0 | 2.1–6.0 |
| polars lazy (c) | 1.8–5.3 | 1.6–5.5 |
| streaming (c_s) | 1.9–5.7 | 1.9–5.6 |
| duckdb (d) | 0.73–2.15 | 0.7–2.0 |
| numpy reshape (e) | 6.9–10.7 | 7.2–10.7 |
| GroupedArray (e2) | 7.5–11.7 | 8.5–11.2 |
| numpy kernel (ek) | 41–506 | 37–448 |

The numbers reproduce. **The flaw is asymmetric accounting.** Engines b/c/c_s/d are timed on a
polars/arrow frame that already exists. Engines a/e/e2 start from pandas. The cost of converting a
pandas panel to polars is 89 ms at 10k and 11 ms at 1k (R2 part3_4). For OLS that is about the same
as the polars compute itself (97 ms). Once conversion is included, b/c/c_s/d are only
**1.07–1.33× faster than pandas groupby at 10k** (0.91–1.07× at 1k). Converting back to a pandas
MultiIndex costs another 120 ms. For a pandas user, then, switching engine is worth about 1×, and the
gains in this claim exist only when the data is natively polars or duckdb.

A different shortcut, the e/e2 numpy layout, keeps its 7–11× gain with pandas in and pandas out.

**The lazy and duckdb paths really execute.** R2 part3_4 shows the expected 140,000 output rows, and
time scales 7.8–12.1× from 1k to 10k for c, c_s and d.

### x7.4: parquet end to end, lazy scan 123 ms / +31 MB against 1,458 ms / +377 MB for pandas and 145 s for sktime. PARTIALLY CONFIRMED

R5, fresh process per variant, 2 repeats:

| variant | last | ols |
|---|---|---|
| polars lazy | 128/147 ms, +31.1 MB | 270/271 ms, +131 MB |
| duckdb | 255–279 ms, +56 MB | 241–255 ms, +56–57 MB |
| polars eager read-all | 335–382 ms, +238 MB | — |
| pandas read-cols | 451–737 ms, +191 MB | — |
| pandas read-all | 2,214–2,447 ms, +377 MB | — |

The peak-memory figures reproduce exactly.

**Flaw: cold timing.** The one-shot-per-process timing includes the lazy import of `pyarrow.parquet`
inside the pandas variants. polars is imported before timing starts. R6 checked this: in a process
where `pyarrow.parquet` had not been loaded yet, the first `pd.read_parquet` took 2,255 ms and the
next ones took 470–642 ms.

Warm ratios from R6: lazy vs pandas read-all is 10.5–12× (the original reported 12×, but from a
cold-inflated pandas number). Lazy vs pandas read-cols is 5–7×, and lazy vs polars eager is 4–5×.

Two more limits:

* The memory benefit of laziness only appears for `last`. For OLS, lazy peaks at +131 MB, against
  +255 MB eager and +56 MB for duckdb.
* The "1,180× vs sktime" figure is almost entirely the per-series loop (x7.1), not laziness. I did
  not re-run sktime_today (145 s). At about 13 ms per series it is consistent.

The direction holds, but the pandas absolute numbers are inflated about 3× by the import cost.

### x7.6: stock PolynomialTrendForecaster with sklearn 1.9.1 is not OLS (max_abs 19.7). CONFIRMED

The source supports it: `sklearn/linear_model/_base.py:752` calls `linalg.lstsq(X, y, cond=self.tol)`
with `tol=1e-6` as the default (line 643). sktime uses `LinearRegression(fit_intercept=False)`
(`_polynomial_trend_forecaster.py:137`) on days since epoch.

R2 part2 reproduces it on a different seed: stock vs polyfit gives **max_abs 17.8 and max_rel 6.0**.
tol=1e-12 vs polyfit gives max_abs 1.0e-12. The magnitude depends on the data, but the defect is
real.

### x7.7: the hook passes 19/19 equality checks vs stock across pandas/polars/lazy with no conversion. PARTIALLY CONFIRMED

R4 reproduced 19/19 [OK]. But not all 19 are equality checks against stock on the native path:

* 12 compare the native path with stock.
* 1 (`polars_lazy_equals_eager`) compares native with native.
* 3 cover fallback/materialization (`fallback_mean`, `fallback_sp7`, `materialize_predict_interval`).
  These compare stock with stock, so they pass by construction.
* 4 are type, plan or info assertions.

The claim is therefore "12 native-vs-stock equalities plus 7 other checks".

**Missed hazard.** The hook replaces sktime's validation with `detect_panel`, a structural check
only. On input with rows shuffled, stock sktime raises a TypeError. The hook (pandas, polars and
polars-lazy) instead **silently returns wrong forecasts** (R2 part5: wrong timestamps, max_abs
7.6–9.8). A native path needs at least a sortedness check, or a sort in the plan (for example
`arg_max` by time, or `sort_by` inside the agg).

### x7.8: hook at 1k series is 21.5 / 5.4 / 6.0 ms against 12,281 ms (570× / 2,266×), and check_is_scitype on polars adds about 1.27 s at 10k. CONFIRMED

R4 results:

* 1k: native pandas 21.37 ms, native polars 6.07 ms, lazy 6.88 ms, stock loop 12,179 ms. That is
  **570× (pandas) and 2,006× (polars)**, against 2,266× originally (noise in the 5 ms term).
* 10k: polars + check_is_scitype 1,266.5 ms vs native 24.4 ms.

Note the asymmetry: the native timings skip all input validation, while stock includes it. With
pandas `check_is_scitype` added back the speedup is still 542×. So the conclusion holds, but polars
validation must be cheap (see the hazard under x7.7).

### x7.10: about 9 of 160 are closed-form candidates; 41 composite; 33 iterative; 52 global/DL. PARTIALLY CONFIRMED

The re-run gives identical counts, and the 160 total is confirmed independently via `all_estimators`.
Spot checks of the source contradict some members:

* **`ProphetPiecewiseLinearTrendForecaster` is a `_ProphetAdapter`** with `python_dependencies:
  prophet`. That is a Stan optimisation, not closed form.
* **`ThetaForecaster` subclasses the statsmodels `ExponentialSmoothing`** and optimises alpha.
* **`NaiveVariance`** wraps another forecaster and refits it in a backtest loop. It is
  composite-like.
* **`TSB`** (a Croston variant, plain `BaseForecaster`) is counted as composite only because the
  regex `TSB` is in `COMPOSITE_PAT`. It is actually closed form.
* **Darts*RegressionModel / XGBModel** are counted as global_dl although they are tabular regressors.
* The statsforecast ADIDA model is simple aggregation but is counted as iterative.

A strict closed-form count is about 6–8 (Naive, PolynomialTrend, Trend with a linear regressor,
SplineTrend, Croston, TSB, ForecastKnownValues). The conclusion ("a fast path for a handful of
baselines only") is unchanged.

### x7.11: narwhals duckdb `dt.total_seconds()` ignores days and hours; `collect()` on duckdb returns pyarrow (C1). CONFIRMED

The source is as cited: `narwhals/_duckdb/expr_dt.py:59-65` computes
`60*datepart('minute') + datepart('second')`.

R3 used an independent probe built on **timestamp subtraction**, which is what the OLS plan does,
rather than a literal interval:

* (2000-01-04 00:00:05 − 2000-01-01) gives 259205 on pandas, polars, polars-lazy and pyarrow, but
  **5.0 on duckdb**.
* `collect()` on duckdb returns a `pyarrow.lib.Table`, while polars lazy returns a polars DataFrame.

On C1 ("can it be forced?"):

* `collect(backend=...)` accepts only polars, pandas or pyarrow. `backend="duckdb"` raises
  ValueError.
* `to_native()` on the lazy frame returns the `DuckDBPyRelation`.

So a duckdb-native result means staying lazy and calling `to_native()`. It cannot be produced by
`collect()`.

### x7.5 (bonus): fitted state is about 175× smaller. CONFIRMED (scaled)

R7 at 2k series: forecasters_ pickles to 8.13 MB against a 0.046 MB params table (178×), with +37.7 MB
retained RSS. This scales linearly to the 40.7 MB reported at 10k.

Minor caveat: a pickle size is being compared with polars `estimated_size()` in memory.

## Missed or wrong (summary)

1. **Conversion cost left out of the engine-vs-pandas ratios.** pandas→polars costs 89 ms at 10k,
   about equal to the polars compute. Including it shrinks polars/duckdb gains to 1.07–1.33× (x7.2).
   Engine gains need native-in data; numpy e/e2 keeps 7–11× from pandas.
2. **Sortedness not checked.** Shuffled rows make the hook and the a/b/c/e2 kernels silently wrong,
   where stock sktime rejects the input. The hook's cheap `detect_panel` drops a correctness
   guarantee that `check_is_scitype` provided.
3. **Cold parquet timing penalises pandas.** The pyarrow.parquet import (~1.7 s) falls inside the
   pandas variant's timing. Warm, pandas read-all takes 0.47–0.64 s, not 1.46–2.4 s.
4. **"19/19 equality checks" is overstated.** 12 are native-vs-stock equalities; the rest are
   fallbacks (stock vs stock) or type/plan assertions.
5. **Family census misclassifications:** Prophet PWL trend, Theta, NaiveVariance, TSB, Darts
   regressors.
6. The 10k f baseline was not re-measured here, because of time. The per-series cost is linear, so
   it extrapolates.
