# x8_e2e_profile: where does time go in the current vectorized forecasting path?

**Question.** When a single-series forecaster is fit/predicted on Panel or Hierarchical data (so
sktime vectorizes, one forecaster per instance), what share of the time is
(a) polars/dask to pandas conversion and other container work done once per call,
(b) per-instance framework overhead (a full `BaseForecaster.fit/predict` per slice, clone, slicing), and
(c) the forecaster's own per-series pandas algorithm?
From that: which decoupling approach can actually make things faster?

**Short answer.** Container work is **0.3–4.8 % of wall time at n ≥ 1000**, and polars input is
**0.92–1.11x the pandas wall time** (no consistent difference). Making the container independent
*on its own* can therefore save **at most ~5 %** on this path. The time goes into per-instance Python work:
the per-instance framework takes 33–61 %, and the inner per-series pandas code takes 40–66 %. That inner
code includes sktime's own nested boilerplate: NaiveForecaster "last" runs a full `Lag` transformer
`fit_transform` per series. The same results computed vectorized (pandas groupby, or polars
eager/lazy expressions) run **130–1800x faster** and match the output. Removing only the BaseForecaster
boilerplate while keeping the per-series pandas `_fit/_predict` gives **1.1–2.4x**.

## 1. Setup

- Environment: `EXP_PY` (pandas 2.3.3, numpy 2.2.6, polars 2.0.0, dask 2026.8, sklearn 1.9.1), sktime
  from worktree source, cloud box with 4 cores and other agents running concurrently.
- Data: `make_panel_fast(n, 100)` (Panel, 2-level MI) and `make_panel(n, 100, n_levels=2)` (Hierarchical,
  3-level MI), with n_series in {10, 100, 1000, 5000}, T=100, 1 column, daily DatetimeIndex.
- Containers:
  - `pd`: pandas MI.
  - `pl`: sktime `polars_panel` / `polars_hierarchical`, made by `convert_to` from pandas.
  - `dask`: `dask_panel` / `dask_hierarchical`, n ≤ 1000. **The dask mtypes are rejected by
    `check_is_scitype` as shipped.** Their `python_type="dask.dataframe"` tag never matches the class name
    `DataFrame`. For profiling only, this tag was patched in-process (`patch_dask_mtype_tags`); no file
    was modified.
- Forecasters:
  - `NaiveForecaster(strategy="last")`.
  - `PolynomialTrendForecaster(degree=1)` as the "heavier" one. statsmodels, statsforecast and pmdarima
    are not installed, so Theta and ExponentialSmoothing could not be run.
  - Workload: `fit(y, fh=[1,2,3]); predict()`.
- Every output was validated against the pandas run (`valid_vs_pd`). All runs were valid.

### Method

1. **Uninstrumented wall clock**: containers interleaved in one process, min over reps (3/3/2/1 for
   n=10/100/1000/5000).
2. **Exclusive-time buckets** come from context-aware wrappers, monkeypatched in-process, around
   check_is_scitype/check_is_mtype/mtype, convert_to/convert, prepare_VectorizedDF/coerce_to_multiindex,
   parallelize (slicing loop), `VectorizedDF.vectorize_est/_vectorize_est_single/reconstruct`, BaseForecaster
   fit/predict/_check_X_y/_vectorize/_check_fh/_update_y_X/_set_cutoff_from_y, the estimator's
   `_fit/_predict`, BaseTransformer public methods and `Lag._fit/_transform`, and `sktime.utils.warnings.warn`.
   - The same function is charged to a different bucket depending on context. For example, `convert_to`
     at top level counts as `top_convert_in/back`; inside a per-instance fit it is `pi_convert`; inside the
     forecaster's `_fit` it is `inner_nested_boiler`.
   - Instrumentation overhead (instrumented / uninstrumented wall time) is 0.92–1.24, within noise; at
     n ≥ 1000 it is 0.92–1.10.
3. **cProfile cross-check** for n ≤ 1000: cumtime of key functions. cProfile inflates totals by about 2x,
   so only use it for shares and call counts.
4. **Ceiling tiers**: the same computation done by hand, validated against sktime. For poly the reference
   is exact OLS; see caveat 4.
   - T1 *lite loop*: the per-series loop calls only the forecaster's own pandas `_fit/_predict`, with
     state set by hand. No BaseForecaster checks, conversions or clone.
   - T2 *numpy loop*: a per-series Python loop over numpy slices, no pandas per series. Input is pandas,
     or polars via numpy views.
   - T3 *vectorized*: pandas `groupby(level=...)`, and polars `group_by(...).agg(...)` expressions, both
     eager and lazy.
5. Side checks:
   - `x8_side_checks.py`: input-check cost, conversion cost and slicing cost vs n, plus warning spam.
   - `x8_callers.py`: per-instance call counts of checks and conversions, with their callers.

### Commands (run from the WT root)

```
export EXP_PY=/tmp/claude-0/-home-user-sktime/c9329e73-89f5-514d-ae5a-9180bf0e4a71/scratchpad/expenv/bin/python
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/x8_e2e_profile.py --grid dev
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/x8_side_checks.py --grid dev
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/x8_callers.py --n 300
PYTHONPATH=.:experiments/container_independence $EXP_PY experiments/container_independence/scripts/x8_summarize.py --grid dev
```

- Outputs:
  - `results/x8_e2e_profile_dev.json` (raw)
  - `results/x8_e2e_profile.json` (headline rows)
  - `results/x8_e2e_profile_dev_tables.md` (all tables, incl. fine buckets and cProfile)
  - `results/x8_side_checks_dev.json`
  - `results/x8_callers.json`
- `--grid large` = [(10,500),(100,500),(1000,500),(5000,100),(20000,100)] is available for the serial
  benchmark phase. `--jobs` runs jobs as parallel subprocesses; this run used `--jobs 1`.

## 2. Results

### 2.1 Wall clock: polars input is no faster or slower than pandas

| est | scitype | n | pd s | pl s | dask s | pl/pd | ms/instance (pd) |
|---|---|---|---|---|---|---|---|
| naive | Panel | 10 | 0.123 | 0.121 | 0.159 | 0.98 | 12.3 |
| naive | Panel | 100 | 1.068 | 1.184 | 1.279 | 1.11 | 10.7 |
| naive | Panel | 1000 | 13.82 | 12.72 | 13.72 | 0.92 | 13.8 |
| naive | Panel | 5000 | 57.77 | 64.26 | - | 1.11 | 11.6 |
| naive | Hier | 1000 | 13.18 | 14.51 | 15.55 | 1.10 | 13.2 |
| naive | Hier | 5000 | 55.98 | 54.08 | - | 0.97 | 11.2 |
| poly | Panel | 10 | 0.094 | 0.112 | 0.179 | 1.20 | 9.4 |
| poly | Panel | 1000 | 9.36 | 9.68 | 12.02 | 1.03 | 9.4 |
| poly | Panel | 5000 | 41.99 | 41.58 | - | 0.99 | 8.4 |
| poly | Hier | 1000 | 9.05 | 8.41 | 10.70 | 0.93 | 9.1 |
| poly | Hier | 5000 | 42.49 | 39.99 | - | 0.94 | 8.5 |

The full table is in `results/x8_e2e_profile_dev_tables.md`.

- The cost is linear in n at about **8.4–14 ms per instance**, whatever the container.
- The pl/pd ratio scatters around 1.0 (range 0.92–1.20), which is the size of the run-to-run noise.
- dask costs 1.0–1.3x pandas at n ≤ 1000, from the extra compute/convert work once per call.
- Peak RSS stayed at or below 642 MB per process.

### 2.2 Bucket rollups (percent of instrumented exclusive time; seconds in brackets)

| est | sc | n | cont | container-once | of which conversion proper | per-instance framework | of which BaseForecaster boilerplate | inner algorithm | inner incl. nested sktime boilerplate |
|---|---|---|---|---|---|---|---|---|---|
| naive | Panel | 1000 | pd | 0.3% (0.04) | 0.0% | 34.6% (4.60) | 27.9% (3.70) | 47.2% (6.28) | 64.9% (8.62) |
| naive | Panel | 1000 | pl | 3.3% (0.44) | 0.2% (0.02) | 34.6% (4.59) | 28.2% (3.75) | 44.9% (5.96) | 61.8% (8.20) |
| naive | Panel | 1000 | dask | 2.4% (0.35) | 0.6% (0.09) | 44.0% (6.61) | 26.0% (3.91) | 38.8% (5.83) | 53.5% (8.03) |
| naive | Panel | 5000 | pd | 0.3% (0.19) | 0.0% | 33.1% (21.1) | 26.9% (17.1) | 48.1% (30.7) | 66.3% (42.3) |
| naive | Panel | 5000 | pl | 2.7% (1.89) | 0.1% (0.10) | 33.0% (23.4) | 26.9% (19.1) | 46.6% (33.1) | 64.1% (45.5) |
| naive | Hier | 5000 | pl | 2.9% (1.56) | 0.2% (0.13) | 34.5% (18.7) | 27.5% (14.9) | 44.8% (24.3) | 62.2% (33.8) |
| poly | Panel | 1000 | pl | 4.3% (0.45) | 0.3% (0.03) | 55.2% (5.85) | 46.2% (4.90) | 40.2% (4.26) | 40.2% (4.26) |
| poly | Panel | 1000 | dask | 4.7% (0.63) | 1.2% (0.16) | 59.5% (7.84) | 38.5% (5.08) | 35.5% (4.68) | 35.5% (4.68) |
| poly | Panel | 5000 | pd | 0.3% (0.16) | 0.0% | 57.2% (26.5) | 48.3% (22.4) | 42.1% (19.5) | 42.1% (19.5) |
| poly | Panel | 5000 | pl | 3.7% (1.59) | 0.2% (0.10) | 55.1% (23.7) | 46.6% (20.0) | 40.8% (17.5) | 40.8% (17.5) |
| poly | Hier | 5000 | pl | 4.8% (1.91) | 0.3% (0.11) | 53.1% (21.2) | 44.2% (17.6) | 41.6% (16.6) | 41.6% (16.6) |

Definitions:

- *container-once* = top-level checks + conversion in + VectorizedDF construction + reconstruct +
  conversion back.
- *per-instance framework* = slicing/groupby iteration + clone + per-instance checks, conversions,
  cutoff/fh handling, other BaseForecaster code + warnings + estimator-frame assembly.
- At n=10, container-once is 4–6 % for pandas, 12–17 % for polars and 22–31 % for dask. This is fixed
  cost, and it is irrelevant above about 100 series.

**Why polars container cost is mostly checks, not conversion.**
- The biggest polars cost is `top_checks`: 0.29–0.44 s at n=1000 and 1.29–1.59 s at n=5000, against
  0.005 s for pandas.
- Almost all of it is `sktime.datatypes._adapter.polars.is_monotonically_increasing`, a Python loop over
  groups (side checks below).
- `convert_to(..., from_type=None)` also re-infers the mtype, which is charged to checks.
- The actual polars→pandas and pandas→polars transforms are only 0.02–0.13 s.

### 2.3 Fine per-instance costs (n=5000, Panel, NaiveForecaster, pandas input; µs per instance)

| bucket | s | µs/instance |
|---|---|---|
| slicing_iter (groupby + items) | 3.40 | 680 |
| pi_clone | 0.28 | 56 |
| pi_checks (check_is_scitype/mtype inside each slice's fit/predict) | 8.19 | 1638 |
| pi_convert (convert_to inside each slice) | 2.36 | 472 |
| pi_cutoff_fh (_check_fh, _set_cutoff_from_y, _update_y_X) | 4.52 | 903 |
| pi_other (remaining BaseForecaster.fit/predict code) | 2.06 | 413 |
| warnings (sktime warn per instance) | 0.27 | 55 |
| inner (NaiveForecaster's own _fit/_predict, exclusive) | 18.19 | 3638 |
| inner_nested_core (Lag._fit/_transform) | 12.48 | 2497 |
| inner_nested_boiler (BaseTransformer checks/converts around Lag) | 11.59 | 2319 |
| reconstruct (pd.concat), vdf_construct, top_* | 0.36 | 72 total |

- The Hierarchical pandas input (built by `make_panel` from tuples) iterates about 2x slower than its
  polars round-trip: slicing_iter is 6.58 s vs 3.28 s at n=5000. **Index construction matters for
  groupby slicing cost.**
- `x8_side_checks` measures slicing alone at **85–96 µs per group**, flat from n=100 to n=20000 on the
  Panel frame. The rest of `slicing_iter` is per-slice `_enforce_index_freq`/infer_freq and building the
  argument dicts.

**Per-instance call counts** (`x8_callers.py --n 300`; identical for pd and pl input):

| forecaster | `convert_to` | `mtype` inferences | `check_is_mtype` | `infer_freq` | `get_cutoff` | `check_is_scitype` |
|---|---|---|---|---|---|---|
| naive | 8.0 | 6.0 | 60.2 | 3.0 | 2.0 | 1.0 |
| poly | 5.0 | 4.0 | 38.2 | 3.0 | 2.0 | 1.0 |

- Callers of `convert_to`, per instance: `get_cutoff` (2), `update_data` (2, naive), `_check_X_y` (2),
  `BaseForecaster.predict` (1), `BaseTransformer._convert_output` (1, naive).
- Every `mtype` inference comes from a `convert_to` without `from_type`.

**Warnings.** Each per-instance forecaster emits `remember_data` FutureWarnings even when the user's filter
is `"ignore"`: `_SuppressWarningPattern.__exit__` resets the filter to `default`. The result is
**about 3 warnings per instance** (147 for n=50, 597 for n=200) and 141 kB of stderr for n=200. This
is a usability bug; the time cost is small (55 µs per instance).

### 2.4 Ceilings: the same result computed without per-instance BaseForecaster

All tier times are best-of-reps seconds. The last three columns are speedups.

| est | sc | n | sktime pd | T1 lite loop | T2 numpy loop pd | T3 pandas groupby | T3 polars eager | T3 polars lazy | sktime_pd / T1 | sktime_pd / T3pd | sktime_pl / T3pl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | Panel | 1000 | 13.82 | 11.02 | 0.0064 | 0.0185 | 0.0080 | 0.0102 | 1.25 | 746 | 1585 |
| naive | Panel | 5000 | 57.77 | 53.06 | 0.0237 | 0.0872 | 0.0360 | 0.0345 | 1.09 | 662 | 1785 |
| naive | Hier | 5000 | 55.98 | 39.84 | 0.0295 | 0.1606 | 0.0431 | 0.0498 | 1.41 | 349 | 1255 |
| poly | Panel | 1000 | 9.36 | 4.99 | 0.0415 | 0.0513 | 0.0147 | 0.0149 | 1.88 | 182 | 657 |
| poly | Panel | 5000 | 41.99 | 20.38 | 0.1701 | 0.1914 | 0.0599 | 0.0588 | 2.06 | 219 | 694 |
| poly | Hier | 5000 | 42.49 | 19.39 | 0.1632 | 0.3189 | 0.0588 | 0.0697 | 2.19 | 133 | 680 |

- All tiers matched the sktime output (`all valid = True`, max abs diff < 1e-6, identical future time
  index).
- For n ≤ 100, T3 speedups are 16–420x; there fixed costs dominate.
- At n=5000, a numpy per-series loop (T2) is still **1900–2400x faster for naive and about 250x for
  poly** than sktime. **A Python loop over instances is not the bottleneck. The ~10 ms of sktime
  machinery per iteration is.**
- Polars expressions (T3) beat pandas groupby by 2.3–5.4x at n ≥ 1000. Polars lazy ≈ eager on this
  in-memory workload.

### 2.5 Conversion micro-bench (sktime `convert_to`, side checks)

| n (T=100) | check pd | check pl | of which pl monotonic loop | pl→pd convert_to | pd→pl convert_to | prepare_VectorizedDF |
|---|---|---|---|---|---|---|
| 100 | 0.001 | 0.010 | 0.010 | 0.014 | 0.003 | 0.002 |
| 1000 | 0.001 | 0.083 | 0.083 | 0.100 | 0.013 | 0.010 |
| 5000 | 0.002 | 0.398 | 0.419 | 0.489 | 0.044 | 0.045 |
| 20000 | 0.005 | 1.748 | 1.799 | 2.370 | 0.204 | 0.168 |

- The polars mtype check is **about 400x slower than the pandas check** at n=20000.
- The polars check is ~75 % of the pl→pd `convert_to` cost, because `convert_to` re-infers the mtype.
- Even so, pl→pd plus output-sized pd→pl is **0.8–1.7 % of the sktime polars wall time** at n ≥ 1000
  (`conversion_microbench` column in the tables file).

## 3. Conclusions and impact statements

1. **Decoupling the container alone gives about nothing on this path.**
   - At n ≥ 1000, polars-specific overhead (checks + conversion + VDF) is 2.7–4.8 % of wall time, and
     pl/pd wall is 0.92–1.11.
   - Upper bound: a perfectly native polars container path that keeps the per-instance
     BaseForecaster loop is **≤ 1.05x faster**.
   - That holds for the adapter/dispatcher approach (1) and for a Narwhals rewrite (2) alike, as long
     as they keep "one BaseForecaster per instance".
   - Laziness, pushdown and memory only become real benefits when the per-instance loop goes away.
2. **Per-instance BaseForecaster overhead is large but not the only cost.**
   - Removing it (T1) gives 1.09–1.67x for naive and 1.66–2.42x for poly (1.09–1.48x and 1.88–2.19x at n ≥ 1000).
   - For naive, the forecaster's `_predict` itself re-enters the framework: `Lag.fit_transform` with full
     BaseTransformer checks and conversions, ≈ 4.8 ms per instance out of 11.6.
   - Per instance there are 8 `convert_to`, 6 mtype inferences, 60 `check_is_mtype`, 3 `infer_freq` and
     2 `get_cutoff` calls.
   - A "lighter inner loop" that skips checks for already-validated slices, plus passing `from_type` to
     avoid mtype inference, is expected to land between T1 and the current speed. This is a framework-only
     change. **unmeasured/expected: ~1.5–2x.**
3. **The decisive lever is panel-native (vectorized) implementations.**
   - Per-instance Python cost must be about 10 µs or less to compete. The numpy loop (T2) shows that a
     per-series loop over raw arrays costs 5–35 µs per series.
   - Vectorized pandas groupby is 130–750x faster than sktime at n ≥ 1000. Polars expressions are
     650–1800x faster than sktime on the same polars input.
   - This is where container choice starts to matter: polars T3 is 2.3–5.4x faster than pandas T3, and
     in-memory lazy ≈ eager.
   - Implication for design: support a `capability:panel_native` / container-native `_fit_panel/_predict_panel`
     path for simple forecasters (naive, mean, drift, trend, seasonal naive). Those could be written once
     against a dataframe-agnostic expression layer (narwhals or polars) and stay in the user's container.
     Keep the per-instance loop as the fallback. Container independence then pays off.
4. **Cheap fixes that matter for polars users regardless of approach:**
   - Vectorize `is_monotonically_increasing` in the polars adapter. It is 1.8 s at n=20000, about 75 % of
     polars container cost.
   - Pass `from_type` in `convert_to` calls.
   - Fix the dask `python_type` tag. As shipped, dask input is rejected outright.
   - Stop the FutureWarning filter reset, which floods stderr with about 3 warnings per instance.

## 4. Caveats

1. Timings were taken on a shared 4-core box with other agents running, so absolute values have
   ~±15 % noise. Ratios within a process are reliable. pl/pd differences under about 15 % are noise.
2. The buckets are exclusive times from wrappers.
   - Unwrapped helpers are charged to the nearest wrapped ancestor. For example, `_enforce_index_freq`
     inside the slicing loop counts as `slicing_iter`, and code in `BaseForecaster.fit` outside wrapped
     helpers counts as `pi_other`.
   - Wrapper overhead is about ≤ 1 µs per call (≈ 30–100 calls per instance), well under 1 % of the
     ~10 ms per instance.
   - The instrumented/wall ratio of 0.92–1.24 reflects noise.
3. Only NaiveForecaster("last") and PolynomialTrendForecaster were profiled, because statsmodels is not
   installed. For forecasters whose inner fit costs ≥ 10 ms per series (ARIMA, ETS), the framework share
   will shrink and the vectorized-rewrite ceiling does not apply in the same way. *(unmeasured/expected)*
4. Poly ceiling tiers were validated against exact OLS, not against sktime's default.
   - With sklearn 1.9.1, `LinearRegression(tol=1e-6)` passes `cond` to lstsq. This truncates the
     intercept direction for the [1, days-since-epoch] design, so sktime's default trend is not exact OLS.
   - The difference is recorded in the raw json as `poly_sktime_default_vs_exact_ols`.
5. dask was profiled only via the in-process tag patch and for n ≤ 1000. The dask conversion goes to
   pandas through `.compute()`, so no dask laziness survives.
6. T1 at n=5000 used a single rep. Times for n=5000 are single-rep in general (min of 1).
