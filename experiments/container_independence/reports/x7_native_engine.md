# x7_native_engine — what pushing forecasting into the engine is worth, and a panel-native hook for it

Workstream key: `x7`. Environment: `EXP_PY` (pandas 2.3.3, numpy 2.2.6, polars 2.0.0, narwhals 2.27.1,
duckdb 1.5.6, pyarrow 26, scikit-learn 1.9.1). Cloud box: 4 cores, about 2 agents running at the same time,
so absolute times are noisy. Ratios measured in the same process are the reliable numbers.
Every number below comes from a run made in this session. The raw JSON/logs are in `results/x7_*`.

## TL;DR

1. **The cost today is the per-series Python loop, not pandas.** For five simple forecasters on a
   10,000×100 panel, the current sktime loop (`f`) takes **70–277 s**. The same results computed with
   **pandas groupby ops** (no per-group Python, still pandas in and out) take **0.12–0.25 s**. That is
   **382×–1,895×** faster. At 100 series it is already **182×–876×**. Any engine (pandas included) gives
   2–3 orders of magnitude, provided the computation is done once for the whole panel and not once per series.
2. **On top of that, the engine choice adds roughly 1.5×–11×** (vs pandas groupby, 10k series, 1 thread):
   polars eager 2.1–6.0×, polars lazy/streaming 1.6–5.6×, numpy on an (n, T) reshape or a GroupedArray
   7–11×. duckdb is about 1× (0.7–2.0×) here, because of its ORDER BY and the result conversion. The pure
   numpy kernel (data already (n, T)) is 37–448× faster than pandas groupby. Container round-trips, not
   arithmetic, are what remains.
3. **Laziness pays end to end**, from a 63 MB parquet file with 8 unused columns (10k×100, `last`):
   polars `scan_parquet` + plan **0.12 s, +31 MB peak RSS**. `pd.read_parquet` (all columns) + groupby:
   1.46 s, +377 MB. Today's sktime path (collect → pandas → loop): **145 s, +483 MB**. Projection pushdown
   reads 3 of 10 columns.
4. **Fitted state shrinks by about 175× (pickle)**. Per-instance `forecasters_` for 10k series: 41 MB pickled /
   +179 MB RSS. A native params table (one row per series): **0.23 MB**.
5. **The hook prototype works** (`protos/x7_hook.py`). The tag is `capability:panel_native` plus
   `_fit_panel/_predict_panel`, added in a subclass with no edits to sktime/. NaiveForecaster(last) runs
   on pandas, polars and polars-lazy **with no conversion**. The output stays in the input container
   (LazyFrame in → LazyFrame out). The output matches stock sktime on 19/19 checks: panel, multivariate,
   hierarchical, NaN, int/period time, ragged ends, polars, lazy, fallbacks, materialized
   predict_interval, native update. Fit+predict at 1,000 series: **12.3 s → 21.5 ms pandas (570×) /
   5.4 ms polars (2,270×)**.
6. **The hook covers few forecasters directly.** About **9 of 160** registered forecasters are
   closed-form or an O(T) recursion. About **41 are composites** (pipelines, reducers, ensembles,
   tuning) that would only need to *delegate* the hook. **33 are iterative 3rd-party fits**
   (statsmodels, pmdarima, statsforecast, …) and **52 are global/DL/foundation models**: neither group
   gains from expression pushdown. The hook is a fast path for the baseline/benchmark families. It does
   not replace the general vectorization layer. The general layer gains most from making the
   per-instance loop cheaper, which the profile shows is mostly framework overhead.
7. **Narwhals as the single expression layer** (side experiment): the same plan written once in
   narwhals ran on pandas/polars/polars-lazy/duckdb. For OLS it ran at 0.76–1.3× the cost of hand-written
   code. For `last` it ran at 2.2–3.0×, because order-dependent `last()` had to be written as
   join+filter. It also showed two hazards. **narwhals' duckdb `dt.total_seconds()` silently returns 5
   for a 3-day+5-second duration** (only minute and second parts are used), which made OLS on duckdb
   wrong. And `collect()` on duckdb returned a **pyarrow Table**, not a duckdb relation. This confirms
   claim C1 for duckdb.

## 1. Methods

### 1.1 Benchmark `scripts/x7_native.py`

Data: `make_panel_fast(n, T)`, a single column `c0` (random walk), daily `DatetimeIndex`, instance
level `h0`. Polars input is `to_polars_long` (keys as plain columns). fh = 1..14.

| method | sktime reference (f) | native formulation |
|---|---|---|
| last | `NaiveForecaster("last")` | last value per group |
| snaive7 | `NaiveForecaster("last", sp=7)` | tail(7) per group, index `(h-1)%7` |
| mean10 | `NaiveForecaster("mean", window_length=10)` | mean of tail(10) |
| drift | `NaiveForecaster("drift")` | last + h·(last−first)/(n−1) |
| ols | `PolynomialTrendForecaster(degree=1)` | closed-form OLS on days since epoch (sufficient statistics n, Σx, Σy, Σxy, Σxx) |

Engines: (a) pandas `groupby(level=0, sort=False)` tail/first/sum, rebuilding a pandas MultiIndex;
(b) polars eager `group_by(maintain_order).agg` + cross join with the horizon;
(c) the same plan lazy with `collect()`; (c_s) `collect(engine="streaming")`;
(d) duckdb SQL (`arg_max`, `QUALIFY row_number()`, `regr_slope/regr_intercept`, `range()` cross join,
`ORDER BY`), fetched as arrow; (e) numpy reshape to (n, T); (e2) numpy GroupedArray (indptr +
`np.add.reduceat`, which also works for ragged panels); (ek) the numpy kernel alone (input already
(n,T), output (n,H)), the floor; (f) the current sktime per-series loop with pandas in;
(f_pl) the same with sktime `polars_panel` in.

**Correctness**: every engine's output (keys, future timestamps, values) is sorted and compared with
(f), using `rtol=1e-7, atol=1e-9`. All 15 (size × method) cells pass for every engine
(`x7_native_dev_t1.json`, `check`). The largest absolute deviation is 1.9e-8 (mean10, 10k series,
caused by summation order).

**OLS reference caveat (found, measured)**: scikit-learn 1.9's `LinearRegression` passes `tol=1e-6` as
`cond` to `lstsq`. With x ≈ 1.1e4 (days since epoch) the [1, x] design is truncated, so **the stock
`PolynomialTrendForecaster(degree=1)` result is not OLS**. It differs from closed-form OLS by up to
**19.7 in absolute terms (max_rel 27)** on this data (`--ols-default` run below). The baseline
therefore uses `LinearRegression(fit_intercept=False, tol=1e-12)`. Runtime is unchanged (946 ms vs
922 ms, 100 series). This is an sktime/sklearn interaction that should be reported separately.

Timing: the methods are run interleaved and the minimum over 5 repeats is taken. (f)/(f_pl) are
timed once more after the reference run, so they are min-of-2. `POLARS_MAX_THREADS` and duckdb
threads are set by `--threads`. BLAS is pinned to 1 thread. The sktime warning-suppression
context resets FutureWarning filters and floods stderr with warnings. The script patches that, which
makes (f) faster than out of the box, so the speedups are conservative.

### 1.2 Commands

```
cd /home/user/sktime; export PYTHONPATH=.:experiments/container_independence
$EXP_PY -W ignore experiments/container_independence/scripts/x7_native.py --grid dev --threads 1
$EXP_PY -W ignore .../x7_native.py --grid dev --threads 4 --baseline-max 1000
$EXP_PY -W ignore .../x7_native.py --sizes 100x100 --methods ols --engines a,f --ols-default --threads 1 --tag _olsdefault
$EXP_PY -W ignore .../x7_native.py --grid dev --threads 1 --mem --engines a,b,c,c_s,d,e,e2,f
X7_SCRATCH=<scratchpad> $EXP_PY -W ignore .../x7_native.py --grid dev --threads 1 --parquet --repeat-parquet 1
X7_SCRATCH=<scratchpad> $EXP_PY -W ignore .../x7_hook_demo.py --grid dev --threads 1
$EXP_PY -W ignore .../x7_nw.py --grid dev
$EXP_PY -W ignore .../x7_family_count.py
$EXP_PY -W ignore .../x7_profile_loop.py --n 200
```
`--grid large` exists in x7_native.py, x7_hook_demo.py and x7_nw.py for the serial benchmark phase.
x7_native large grid: 1k×100, 10k×100, 100k×100, 10k×1000. Use `--baseline-max` to cap the
slow sktime loop.

## 2. Results: engine vs per-series loop (1 thread)

Raw best times in ms (`results/x7_native_dev_t1.log`):

```
== n_series=100 T=100
  last     a=2.30  b=1.29  c=1.07  c_s=0.86  d=5.13   e=1.13  e2=0.83  ek=0.01  f=1299.55   f_pl=1556.35
  snaive7  a=3.00  b=2.08  c=1.23  c_s=0.95  d=7.20   e=1.30  e2=0.79  ek=0.02  f=2625.54   f_pl=2793.54
  mean10   a=3.20  b=1.42  c=0.96  c_s=0.76  d=6.14   e=1.17  e2=0.82  ek=0.04  f=666.08    f_pl=657.64
  drift    a=3.00  b=1.18  c=1.10  c_s=0.96  d=5.01   e=0.91  e2=0.72  ek=0.02  f=792.18    f_pl=791.29
  ols      a=5.06  b=2.32  c=2.12  c_s=1.77  d=5.07   e=1.35  e2=0.96  ek=0.10  f=922.42    f_pl=1131.79
== n_series=1000 T=100
  last     a=13.96 b=3.27  c=2.82  c_s=2.72  d=14.41  e=2.35  e2=1.97  ek=0.03  f=12041.66  f_pl=12964.37
  snaive7  a=15.05 b=6.99  c=4.09  c_s=3.57  d=27.30  e=2.83  e2=2.21  ek=0.06  f=27439.20  f_pl=31764.21
  mean10   a=20.41 b=4.09  c=3.66  c_s=3.18  d=22.05  e=2.81  e2=3.17  ek=0.18  f=7119.61   f_pl=7729.77
  drift    a=13.86 b=3.50  c=4.05  c_s=3.42  d=17.70  e=2.51  e2=2.03  ek=0.08  f=7973.26   f_pl=8048.42
  ols      a=25.69 b=11.76 c=11.38 c_s=11.29 d=16.66  e=4.75  e2=3.17  ek=0.60  f=9156.14   f_pl=9384.32
== n_series=10000 T=100
  last     a=124.64 b=37.38  c=25.32  c_s=23.54  d=116.09 e=14.13 e2=13.33 ek=0.28 f=124310.13
  snaive7  a=146.10 b=63.74  c=57.29  c_s=41.05  d=215.58 e=14.56 e2=14.89 ek=0.36 f=276886.02
  mean10   a=166.19 b=27.59  c=37.47  c_s=34.30  d=164.09 e=15.50 e2=19.33 ek=1.19 f=69804.12
  drift    a=156.66 b=28.17  c=28.57  c_s=28.06  d=141.36 e=14.87 e2=14.00 ek=0.98 f=73146.44
  ols      a=246.07 b=115.47 c=158.43 c_s=129.30 d=120.41 e=33.95 e2=28.80 ek=6.57 f=93913.06
```

### Speedup vs (f) (sktime loop)

| n_series | method | a pandas | b polars | c lazy | c_s stream | d duckdb | e numpy | e2 grouped | f/f_pl (polars input is slower) |
|---|---|---|---|---|---|---|---|---|---|
| 100 | last | 565 | 1004 | 1218 | 1516 | 254 | 1154 | 1574 | 0.83 |
| 100 | ols | 182 | 397 | 435 | 521 | 182 | 683 | 957 | 0.81 |
| 1000 | last | 863 | 3684 | 4269 | 4433 | 836 | 5118 | 6103 | 0.93 |
| 1000 | snaive7 | 1824 | 3924 | 6702 | 7690 | 1005 | 9689 | 12390 | 0.86 |
| 1000 | ols | 356 | 779 | 805 | 811 | 550 | 1930 | 2891 | 0.98 |
| 10000 | last | 997 | 3325 | 4910 | 5282 | 1071 | 8796 | 9323 | – |
| 10000 | snaive7 | 1895 | 4344 | 4833 | 6745 | 1284 | 19022 | 18590 | – |
| 10000 | mean10 | 420 | 2530 | 1863 | 2035 | 425 | 4505 | 3612 | – |
| 10000 | drift | 467 | 2597 | 2561 | 2607 | 517 | 4918 | 5223 | – |
| 10000 | ols | 382 | 813 | 593 | 726 | 780 | 2766 | 3261 | – |

### Speedup vs (a) (pandas groupby, same algorithm)

10k series: b 2.1–6.0×, c 1.6–5.5×, c_s 1.9–5.6×, d 0.7–2.0×, e 7.2–10.7×, e2 8.5–11.2×,
ek 37–448×. 100 series: b 1.4–2.5×, d 0.4–1.0×, e2 2.8–5.3×.

### 4 threads (`results/x7_native_dev_t4.log`, 10k series, ms)

```
  last     a=118.52 b=21.31 c=12.25 c_s=11.11 d=141.98 e=13.10 e2=12.89
  snaive7  a=163.30 b=38.62 c=30.73 c_s=17.75 d=195.59 e=14.15 e2=14.60
  ols      a=254.45 b=53.85 c=63.72 c_s=48.03 d=134.91 e=37.36 e2=28.79
```
Polars benefits from threads (ols b 115→54 ms, last c_s 23.5→11.1 ms; the box is shared, so this is
noisy). pandas and numpy are single-threaded and do not change. With 4 threads polars lazy/streaming
approaches the numpy GroupedArray path (last 11.1 vs 12.9 ms).

**Interpretation.** (f) costs **about 7–28 ms per series** (124 s / 10k for `last`). Engine
arithmetic is microseconds. The cProfile of the loop (`results/x7_profile_loop_n200.txt`, 200
series, `last`: fit 4.2 ms/series, predict 8.6 ms/series) shows where the time goes: 12,046
`check_is_mtype` calls (1.2 s cumulative of 5.4 s profiled), 1,604 `convert_to`, `get_cutoff`,
`ForecastingHorizon.to_absolute` (0.6 s), and above all **NaiveForecaster `_predict_naive` running a
`Lag` transformer per series** (fit_transform 2.0 s of the 3.5 s predict). The pandas→polars
container choice (f_pl/f = 0.81–0.98) is noise next to this.

## 3. Memory and fitted state (10k×100, 1 thread, one process per cell; `x7_native_mem_dev_t1.log`)

Input: pandas 11.3 MB.

| engine | last: peak Δ / retained (MB) | ols: peak Δ / retained (MB) |
|---|---|---|
| a pandas | 72.6 / 11.7 | 117.9 / 28.8 |
| b polars | 47.2 / 47.2 | 55.9 / 54.8 |
| c polars lazy | 21.5 / 21.5 | 98.1 / 98.1 |
| c_s streaming | 21.8 / 21.8 | 97.9 / 97.9 |
| d duckdb | 26.0 / 23.6 | 30.1 / 26.0 |
| e numpy | 10.5 / 10.5 | 35.2 / 35.2 |
| e2 grouped | 11.2 / 11.2 | 27.8 / 27.8 |
| **f sktime** | **265.3 / 219.3** | **275.2 / 234.2** |

Fitted state (`fitted_state` rows): sktime `forecasters_` after fit: **+179 MB RSS (last) / +203 MB
(ols); pickled 40.7 MB / 34.8 MB**. A native params table (key, tlast, value; one row per series):
**0.23 MB**, which is about 175× smaller than the pickle.
Caveat: "retained" for polars includes allocator caching (RSS is not returned to the OS), so it is
an upper bound and not a leak.

## 4. End to end from parquet: laziness and pushdown (`x7_native_parquet_dev_t1.log`)

File: 10k×100 rows, `c0` + 8 unused float columns, 63.2 MB. One fresh process per variant, 1 thread,
one repeat (so noisy).

| variant | last: wall / peak Δ | ols: wall / peak Δ |
|---|---|---|
| sktime today (scan→collect→pandas→loop) | 145,122 ms / +483 MB | 108,083 ms / +496 MB |
| pandas read all + groupby | 1,458 ms / +377 MB | 1,799 ms / +380 MB |
| pandas read 3 cols + groupby | 480 ms / +190 MB | 1,086 ms / +235 MB |
| polars read all (eager) | 331 ms / +237 MB | 415 ms / +255 MB |
| **polars scan_parquet (lazy)** | **123 ms / +31 MB** | 275 ms / +131 MB |
| polars lazy, streaming | 119 ms / +31 MB | 295 ms / +131 MB |
| duckdb read_parquet | 233 ms / +55 MB | **222 ms / +56 MB** |

So the lazy plan beats the eager polars read 2.7× and pandas read-all 12×, at a 7.7–12× smaller peak.
It beats today's sktime by about 1,180× with about 16× less memory. This is the value of non-pandas
engines that a "convert to pandas first" design cannot capture: **it only exists if the whole
computation stays in one plan**. `x7_hook_demo` confirms the plan has `PROJECT 3/10 COLUMNS`.

## 5. Panel-native hook (`protos/x7_hook.py`, `scripts/x7_hook_demo.py`)

Design (opt-in, no edits to sktime/):

* Tags `capability:panel_native: True`, `panel_native:backends: ["pandas","polars","polars_lazy"]`,
  and a parameter-dependent predicate `_panel_native_supported()` (for example only
  `strategy="last", sp=1`).
* `PanelNativeMixin.fit` routes **before** `_check_X_y`/`prepare_VectorizedDF`. It routes when the
  tag is set, the predicate is true, X is None, and a cheap structural `detect_panel` (0.04–0.06 ms,
  no conversion) recognises a pandas MultiIndex (≥2 levels) or a polars Data/LazyFrame using the
  `__index__` convention.
* `_fit_panel(pf) → params table` in the input engine (pandas groupby / polars `group_by().agg`).
  Lazy input gets one `collect()` of the small params table. The cutoff is the max instance end, kept
  as a `pd.Index` so `cutoff`/`fh` semantics are unchanged.
* `_predict_panel(params, steps)`: params × horizon. pandas builds the MultiIndex. polars uses a
  cross join and returns a `LazyFrame` when the input was lazy.
* `_update_panel`: upserts the params table natively (new instances are appended).
* **Everything else materializes**: `predict_interval/quantiles/var` (and update on another backend)
  call stock `fit` on the remembered input and delegate. The full API therefore keeps working and
  only the fast path is new. `get_fitted_params` returns the native params table.
* Pitfall found and fixed in this session: `BaseForecaster.fit` → `_reset_at("pretrained")` →
  `reset()` wipes private attributes. Routing flags must be set *after* calling the stock fit, or
  fallbacks raise `AttributeError`.

Checks (`results/x7_hook_demo_dev_t1.json`): **19/19 OK.** pandas panel, multivariate, hierarchical
(3-level), NaN (trailing + interior), int time, period[M] (calendar offset), ragged ends, polars
panel (output polars), polars lazy from parquet (output lazy, equal to eager, projection pushdown
3/10 columns), fallback mean/sp=7 equal to stock, materialized predict_interval equal to stock,
native update pandas and polars equal to stock fit+update, fitted params as a polars table. Also
recorded: stock sktime **rejects** a polars LazyFrame panel (TypeError, no lazy Panel mtype).

Timing, fit+predict, 1 thread (ms):

| n | native pandas | native polars | native lazy | +check_is_scitype pandas | +check_is_scitype polars | stock loop pandas | stock loop polars |
|---|---|---|---|---|---|---|---|
| 100 | 5.57 | 3.49 | 3.90 | 6.78 | 16.41 | 1,340 | 1,219 |
| 1000 | 21.54 | 5.42 | 5.97 | 22.93 | 129.04 | 12,281 | 12,675 |
| 10000 | 156.5 | 29.35 | 38.10 | 159.8 | 1,268.2 | – | – |

Speedup vs the stock loop at 1k: 570× (pandas), 2,266× (polars). **Note**: if the native path still
ran sktime's `check_is_scitype`, the polars check alone (1.27 s at 10k) would cost 43× the forecast.
Native paths need a cheap structural check and must not run the full mtype validator on non-pandas
inputs.

## 6. Which forecasters could implement the hook (`scripts/x7_family_count.py`)

Heuristic over `all_estimators("forecaster")` (160 classes): a name list for closed-form models,
then dependency-tag regexes and name patterns. Counts:

| class | n | examples | hook? |
|---|---|---|---|
| closed_form | 9 | NaiveForecaster, PolynomialTrendForecaster, TrendForecaster, ThetaForecaster, Croston, SplineTrend, ProphetPiecewiseLinearTrend, NaiveVariance, ForecastKnownValues | yes. Group-by aggregates / window / cumulative ops. Theta needs an SES alpha search (feasible as an (n,T) numpy kernel, not as SQL) |
| composite | 41 | TransformedTargetForecaster, ForecastingPipeline, Reduction*, Ensemble*, *SearchCV, UpdateEvery | only by **delegation** (native if every component is). Reducers could build lag tables natively (window functions), but the regressor needs numpy |
| iterative 3rd-party | 33 | ARIMA/AutoARIMA/SARIMAX, AutoETS, VAR, Prophet, TBATS, StatsForecast* | no expression pushdown. At best a batch (n,T) numpy/numba kernel. statsforecast already does this internally (GroupedArray) |
| global / DL / foundation | 52 | Chronos*, TimesFM*, MOIRAI, MIRAForecaster, NeuralForecast*, PytorchForecasting*, LTSF*, TTM | already one model over all series. They need tensors: the "native" path is panel→(n,T,…) tensor, i.e. the `e` layout, not frame ops |
| other | 25 | ARAR, Arps*, CurveFit, ForecastX, … | case by case (mostly per-series nonlinear fits) |

36 of the 160 already declare a Panel/Hierarchical `y_inner_mtype`.

**Conclusion**: an engine-native expression hook is worth it for **about 5–10 baseline families**.
These are the ones used most at scale (benchmarks, fallbacks, hierarchical bottom level), and the
gain there is about 1,000×. For most of the remaining ~150 classes, the general levers are:
(i) a cheap per-instance loop (cut the validation/convert/Lag overhead the profile shows), and
(ii) for global/DL models, a direct panel→(n,T) array export (the `e`/`e2` layouts, 7–11× faster than
pandas groupby and engine-agnostic: any container that exports contiguous key-sorted columns works).

## 7. Narwhals single-plan side experiment (`scripts/x7_nw.py`, `results/x7_nw_dev.json`)

The same narwhals plan was run on 4 backends. For portability `last` is written as join on
`max(time)` + filter, because narwhals group_by does not portably guarantee order-dependent `last()`.

| n | method | a pandas hand | b polars hand | nw pandas (from MI, incl. reset_index) | nw pandas flat | nw polars | nw polars lazy | nw duckdb |
|---|---|---|---|---|---|---|---|---|
| 1000 | last | 14.9 | 3.6 | 44.7 | 41.6 | 10.4 | 7.8 | 17.1 |
| 1000 | ols | 25.1 | 10.5 | 32.8 | 29.4 | 12.5 | 10.9 | 54.4 (**wrong**) |
| 10000 | last | 145.1 | 23.5 | 350.1 | 324.7 | 66.4 | 41.4 | 59.0 |
| 10000 | ols | 246.1 | 104.7 | 227.9 | 187.6 | 95.1 | 54.3 | 419.1 (**wrong**) |

Ratios: nw_polars/b = 2.8–2.9 (last), 0.9–1.2 (ols). nw_pandas_flat/a = 2.2–2.8 (last), 0.76–1.17
(ols). The `reset_index` copy adds 1.07–1.2× on top of the flat pandas plan (C2: it is a copy, but a
cheap one compared with what we save).
**Bug**: narwhals 2.27.1 `_duckdb/expr_dt.py::total_seconds` = `60*datepart('minute') +
datepart('second')`. Probe: a duration of 3 days + 5 s gives polars 259205, pandas 259205,
**duckdb 5.0**. So OLS on duckdb via narwhals is silently wrong.
**C1**: `nw.from_native(duckdb_rel).….collect()` returned a `pyarrow.lib.Table`. polars lazy
collected to polars. A duckdb-native result requires staying lazy (do not collect) or calling
`to_native` before collecting. A layer that must "respect the original format" has to enforce this.

## 8. Prior art

* **Nixtla utilsforecast/statsforecast/mlforecast** (read utilsforecast 0.2.17 source in its own
  venv): `processing.py` dispatches with `isinstance(df, pd.DataFrame)` / polars branches (23
  pandas-isinstance sites). `losses.py`/`evaluation.py` use **narwhals** for group-by aggregates
  (engine-native metrics). Model fitting uses `GroupedArray(data, indptr)` (our `e2`) with numba
  kernels. That is the same split this experiment points to: **aggregates/baselines in the engine,
  models on a contiguous grouped array**.
* polars/duckdb themselves: per-group closed-form models as `group_by().agg` / SQL `regr_*`. This
  is what `b/c/d` measure.
* Unmeasured/expected, not verified in this session: darts (per-series loop over `TimeSeries`
  objects), skforecast (pandas only), Fugue (used by statsforecast to push per-group *Python* UDFs to
  Spark/Dask/Ray: parallelism, not vectorization).

## 9. Caveats

* Shared 4-core box with concurrent agents. Absolute ms are noisy and baseline times are min of 2.
  The speedups are 2–4 orders of magnitude, far above the noise.
* Equal-length, aligned, daily, univariate, no-NaN panels in x7_native (the hook demo covers NaN,
  ragged ends, multivariate, hierarchical, period/int time). `e` (reshape) requires equal length;
  `e2` does not.
* The native implementations mirror the specific semantics of each sktime method (drift uses
  first/last; snaive picks the last full season). Edge cases such as NaN handling in drift/mean or
  sp>T are not covered for the engines other than the hook's `last`.
* The hook sketch does not handle X, absolute fh with differing instance cutoffs, or
  hierarchical aggregation levels. predict_interval goes through materialization (pandas).
* duckdb times include `ORDER BY` and the arrow fetch. Without sorting, duckdb would likely be faster
  (unmeasured).
* Family classification is a heuristic (regex on dependency tags/names). The counts are ±a few.
