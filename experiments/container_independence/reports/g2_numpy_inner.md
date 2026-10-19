# g2_numpy_inner — opt-in "numpy values + time axis" inner mtype, wired into BaseForecaster

**Gap (critic G2, HIGH).** Prototype a per-estimator opt-in inner mtype made of numpy values plus a
time axis, and route `BaseForecaster` to it when a forecaster declares it in `y_inner_mtype`. Then:
- port 3 forecasters,
- measure end-to-end time against stock for pandas, polars and pyarrow input,
- count conversions,
- measure LOC,
- check equality, including `predict_interval`,
- explain how cutoff, fh and the output index are produced.

**Short answer.**
- **It works without editing `sktime/`.** The opt-in is one tag:
  `y_inner_mtype = ["np_grouped_series", "np_grouped", "np_grouped_hier"]`. The registration from
  outside does three things:
  - patches the cached mtype class lists,
  - adds entries to `convert_dict`,
  - wraps `get_cutoff` and `update_data` in `forecasting.base._base`.
- **Equality.** The 3 ported forecasters (11 configurations) match stock on pandas, polars and
  pyarrow input, under both pandas 2.3.3 and 3.0.6:
  - **315/315** predict comparisons
  - **233/233** `predict_interval` comparisons
  - 6/6 absolute-fh checks and 9/9 update-then-predict checks
  - relative error ≤ 1.3e-14 for predict and ≤ 2.7e-10 for intervals
- **Conversions per fit+predict drop from about 500 + 300 identity to exactly 2** (one in, one
  out). Public `fit`/`predict` re-entries drop from 101/101 to 1/1. mtype checks drop from
  7,484–12,084 to 36 (100x100 panel).
- **End to end at 1000x100:**

  | forecaster | pandas input | pyarrow input | polars input |
  |---|---|---|---|
  | Naive | 1,229–2,735× | 585–1,486× | 35–105× |
  | PolynomialTrend | 498–1,308× | 498–845× | 34–70× |
  | per-series recursive reducer (sklearn fit per instance) | 7.3× | 7.3× | 6.4× |

  On polars, 99% of the remaining g2 time is the two **stock** `polars_panel` mtype checks in
  `_check_X_y` / `convert_to`, not the container or the kernel.
- **pandas is left only on small objects:** the 1-element cutoff index, the H-element
  `ForecastingHorizon`, and the egress target when the target itself is pandas. That includes
  `predict_interval`, whose public return type is pandas.

## 1. What was built

| file | content |
|---|---|
| `protos/g2_np_grouped.py` (443 logical LOC) | `NpGrouped` container, numpy step inference, numpy time ops, converters, 6 mtype classes, `register_np_grouped()` |
| `protos/g2_forecasters.py` | `_G2Base`, `G2NaiveForecaster`, `G2PolynomialTrendForecaster`, `G2RecursiveReducer` |
| `scripts/g2_equality.py` | equality vs stock → `results/g2_equality_pd2.json`, `results/g2_equality_pd3.json` |
| `scripts/g2_bench.py` | e2e timing (`results/g2_bench.json`); `--counts` → `results/g2_counts.json` |
| `scripts/g2_profile.py` | cProfile attribution of the g2 path → `results/g2_profile.json` |
| `scripts/g2_loc.py` | logical-LOC counts → `results/g2_loc.json` |

**Container `NpGrouped`.** A dataclass subclass of the x6 hub `PanelData`, so it reuses x6's
hand-written spokes, which round-trip 29/29 fixtures. Its fields:
- `values`: 2D float64, a view of the source where possible.
- `indptr`: instance offsets.
- `key_names` / `key_values`: the key table.
- `time`: datetime64 (naive UTC), or int64 (period ordinals or ints).
- `step`: an a3 `Step`, the library-neutral step description.
- `scitype`: `Series`, `Panel` or `Hierarchical`.
- `meta`: egress metadata, so the exact source container is rebuilt.

There is one python class but three mtype names, one per scitype. Two reasons:
- `mtype_to_scitype` raises if one mtype name maps to two scitypes.
- `_check_X_y` needs an inner mtype for every scitype it may receive. With a Panel-only inner
  type, Series input hits `_most_complex_scitype` → `ValueError`.

**Step inference runs over *all* instances, in numpy.** It finds fixed steps, month start or end
(n months), and business days.
- Within-instance diffs are masked at instance boundaries.
- An inconsistent step gives `step=None`, and the first time-arithmetic call raises.
- This fixes x6-verify's silent wrong answer on mixed-frequency panels. g2 now raises
  `ValueError('time axis has no regular step …')`, and stock raises too ("Frequencies from two
  sources do not coincide").

**pyarrow.** sktime has no pyarrow mtype. The proto adds `pa_series`, `pa_panel` and `pa_hier`:
a `pyarrow.Table` with `__index__`-prefixed index columns, the same convention as sktime's polars
mtypes.
- They are registered with converters to and from np_grouped, and to and from the pandas mtypes.
- Because of the pandas converters, *stock* forecasters also accept pyarrow input once the proto
  is registered.

### 1.1 Can sktime's mtype registry be extended from outside? Yes, but only by mutating private caches

These are the facts the registration relies on, read from the code:
- **The registry is discovered by crawling.** `generate_check_dict` and `mtype_to_scitype` both
  start from `_generate_mtype_cls_list(soft_deps)`, which is
  `_all_classes("sktime.datatypes")`, `lru_cache(maxsize=3)`.
  - **There is no public plugin hook**, so a class defined outside the package is never found.
- **Workaround used here:**
  1. Append the 6 classes to the *cached list objects* for `soft_deps` "present", "all" and
     "exclude". The cache keys are keyword calls, so this must call with `soft_deps=` exactly as
     the callers do.
  2. Call `cache_clear()` on `_generate_mtype_register`, `_generate_mtype_register_subset` and
     `generate_check_dict`.
- **Converters are easy.** `sktime.datatypes._convert.convert_dict` is a plain module dict and is
  not copied by its users. Registration adds 28 converter entries, plus 6 identity entries.
- **Fragility, measured.** Any later `cache_clear()` of `_generate_mtype_cls_list`, which a test
  harness might do, silently unregisters the mtype. Run in this session: `mtype_to_scitype('np_grouped')`
  returns `Panel` before the clear and raises
  `ValueError('np_grouped is not a supported mtype')` after it.
  - **Recommendation for a real PR:** a `register_mtype(cls, converters)` API, or an entry-point
    group.
- **Two dispatch points needed patching beyond the registry:**
  - `get_cutoff`, which converts any unknown mtype to pandas before taking the max. Here it is
    replaced by an O(1)-after-max numpy path that returns the 1-element pandas Index
    `BaseForecaster` expects.
  - `update_data`, replaced by `npg_append`: per-instance append with a monotonic-time check.

  Both are wrapped in the `sktime.forecasting.base._base` namespace. A real change would add an
  `NpGrouped` branch inside `get_cutoff` and `update_data`, about 20 lines.
- **No other BaseForecaster code was touched.** Untouched: `fit`, `predict`, `predict_interval`,
  `update`, `_check_X_y`, `_check_fh`, the cutoff property and output conversion. Vectorization is
  never entered, because every input scitype is in `y_inner_scitype`.

### 1.2 The three ports

| port | algorithmic form | panel strategy | intervals |
|---|---|---|---|
| `G2NaiveForecaster` (last / mean / drift, `window_length`, `sp`) | indptr `reduceat` kernels, x6 `hub_naive_batched` generalized to C columns | **batched** over all instances for sp=1; per-instance loop on views for sp>1 | sp=1, batched residual MSE through `bincount`, same formulas and quirks as `NaiveForecaster._predict_var` (drift slope `y[-wl or 0]` over `T-1`) |
| `G2PolynomialTrendForecaster` | default regressor: batched SVD `lstsq(cond=tol)` over the `(U, T, p)` stack; ragged input: per-instance on views; custom regressor: per-instance sklearn pipeline clone on views | batched or per-instance | batched `w^T (XᵀX)⁻¹ w = ‖R⁻ᵀw‖²` from a batched QR, residual variance as stock |
| `G2RecursiveReducer` (`make_reduction(est, strategy="recursive")`, local pooling) | per-series `clone(est).fit` on a zero-copy `sliding_window_view`, recursive predict | per-instance loop (generic sklearn estimator) | not supported (stock: not supported either) |

No statsmodels-based port was done: statsmodels is not installed in `$EXP_PY`. The reducer stands
in for the "generic per-series third-party fit on views" pattern.

**Side finding (stock bug, environment-specific, measured).**
- In sklearn 1.9.1, `LinearRegression.fit` solves the dense case with
  `scipy.linalg.lstsq(X, y, cond=self.tol)`, where `tol=1e-6`.
- `PolynomialTrendForecaster` regresses on days since 1970 (x ≈ 1.1e4). For the 50-day test series
  the design matrix has cond = 8.4e6 > 1/tol. The smallest singular value is dropped, and the
  fitted trend is **not** the least-squares fit:
  - predicted next value: −1.6458 (stock) vs −1.8076 (exact polyfit)
  - residual sum of squares: 385.82 (stock) vs 385.40 (optimum)
- This is independent of the container. g2 reproduces it on purpose with the same truncation, so
  that equality holds. With exact least squares the predictions differ from stock by up to 30%.
- The sklearn version in which `tol` became the lstsq `cond` was not checked (unmeasured).

## 2. How cutoff, fh and the output index are produced

| object | g2 path | pandas? |
|---|---|---|
| per-instance cutoffs | `time[indptr[1:]-1]` | no |
| public `cutoff` | global max of `time` → **1-element** `pd.DatetimeIndex` / `PeriodIndex` / `Index`, with freq from the `Step`, or the source freqstr if pandas gave one | yes, 1 element |
| `fh` | stock `ForecastingHorizon`, unchanged. A relative fh is read as an H-element int array. An absolute fh is converted once to numpy times, then per-instance steps via `steps_between` | yes, H elements |
| future times (U×H) | `add_steps(cut, k)` with a3 `NumpyTimeOps`: fixed, month start/end, business day, int, period ordinals. tz-aware data only with fixed steps; otherwise `NotImplementedError` | no |
| relative fh semantics | every instance steps from **its own** cutoff, which is what stock vectorization over per-instance clones does. Verified on `ragged_diff_end` | — |
| predict output | `NpGrouped` with `indptr = arange(U+1)·H`, then `convert_to(y_pred, input mtype)` → hand-written egress spoke. polars and pyarrow targets: **no pandas object constructed** (counted, §4) | only if the target is pandas |
| `predict_interval` output | pandas DataFrame (public API type: MultiIndex columns `(var, coverage, lower/upper)`), built once from numpy plus codes. The row index comes from the key table and time via `pd.MultiIndex(levels, codes)` | yes, output only |
| `update` | `_update_y_X` → patched `update_data` → `npg_append`. Naive and reducer keep fitted parameters and the window length frozen at fit, as stock `_BaseWindowForecaster` does. The trend refits | no |

**Cutoff `freq` differences.**
- Cutoff *values* are equal in 315/315 comparisons.
- The cutoff *freq* attribute differs in 81 (pandas 2) and 9 (pandas 3) comparisons. In every case
  g2 carries the freq (`D`, `MS`, `W-SUN`) where stock's `get_cutoff` on a MultiIndex time level
  returns `freq=None`.
- Stock is itself inconsistent here: it returns `freq='D'` for `panel_D` but `None` for `hier_D`,
  `panel_MS` and `panel_W`.

## 3. Equality (`scripts/g2_equality.py`, both pandas majors)

```
cd /home/user/sktime
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/g2_equality.py   # 51.4 s
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/g2_equality.py   # 56.2 s
```

**Fixtures (12):**
- panel_D (20×50, daily)
- hierarchical 3-level
- pd.Series with freq D
- pd.DataFrame single series
- Period[M] panel, int panel
- MS panel, W-SUN panel
- ragged with the same end, ragged with different ends
- panel with NaNs (25 random plus a trailing NaN)
- 2-column panel

**Configurations (11):**
- Naive last, mean, mean w=5, drift, drift w=10, last sp=7, mean sp=7 w=14
- Poly d1 with PI, d2 without intercept with PI, d2 with a Ridge regressor (generic path)
- Recursive reducer, LinearRegression, w=5

fh is [1, 2, 3, 5]; coverage is [0.8, 0.95].

**Inputs and references.** Each fixture is fed as pandas, polars and pyarrow; the same g2 class
handles all three.
- The reference is stock on the pandas fixture.
- polars and pyarrow outputs are converted back to pandas *independently* of the proto code, via
  `to_pandas` + `set_index`.
- Tolerances: `assert_frame_equal` / `assert_series_equal` with rtol 1e-9 for predict and 1e-7 for
  intervals.

| | pandas 2.3.3 | pandas 3.0.6 |
|---|---|---|
| predict | **315/315** | **315/315** |
| predict_interval | **233/233** | **233/233** |
| cutoff value | 315/315 | 315/315 |
| errors | 0 | 0 |
| n/a | 9 (Period panel as pyarrow: no arrow type for pandas Period) | 9 |
| max rel err, predict / PI | 1.3e-14 / 2.7e-10 | — |
| absolute fh (last, drift × 3 inputs) | 6/6 | 6/6 |
| update then predict (last/drift/mean × 3 inputs) | 9/9 | 9/9 |
| mixed-frequency panel | raises (stock raises) | raises |

**Notes:**
- **polars vs stock-on-polars, native comparison:**
  - 51/108 results are exactly equal.
  - 39 are equal to rel 1e-9.
  - 18 differ only because **stock's** polars output for Series input loses names: columns
    `['__index__0', '0']`. g2 keeps `['__index__time', 'c0']`.
- **Period fixture as polars.** polars has no Period type, so the reference is stock run on the
  polars input.
- **Stock refuses `panel_nan` drift** ("first and last elements … must not be missing"), so that
  case is not compared.

## 4. Conversions and re-entry (`g2_bench.py --counts`, 100×100 panel, one fit + predict, fh=1..10)

```
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/g2_bench.py --counts
```

| forecaster / input | path | `convert_dict` calls (non-identity + identity) | mtype `check` calls | public fit / predict calls | `pd.DataFrame()` / `pd.Series()` constructed |
|---|---|---|---|---|---|
| Naive last / pandas | stock | 500 + 304 | 12,084 | 101 / 101 | 606 / 2,014 |
| Naive last / pandas | **g2** | **2** + 0 | **36** | **1 / 1** | 1 / 5 (egress to pandas) |
| Naive last / polars | stock | 502 + 302 | 12,084 | 101 / 101 | 606 / 2,008 |
| Naive last / polars | **g2** | **2** | 36 | 1 / 1 | **0 / 0** |
| Naive last / pyarrow (user converts) | stock | 500 + 304 | 12,084 | 101 / 101 | 606 / 2,014, plus the user's `to_pandas` |
| Naive last / pyarrow | **g2** | **2** | 36 | 1 / 1 | **0 / 0** |
| Naive drift, Poly d1, reducer / any input | stock | 0–202 + 202–404 | 7,484 | 101 / 101 | 106 / 1,108–1,516 |
| same | g2 | 2 | 36 | 1 / 1 | 0–1 / 0–5 |

The 2 conversions are always `<input mtype>→np_grouped` and `np_grouped→<input mtype>`.

The 36 checks are:
- `check_is_scitype` over all registered Series/Panel/Hierarchical mtypes on the input, about 30
  checkers, most rejected by type pre-check;
- plus `convert_to`'s second inference;
- plus the inference on the output.

The pandas `Index` objects for cutoff and fh are not included in the construction counts.

## 5. End to end (`scripts/g2_bench.py`, interleaved in one process, min of repeats)

```
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/g2_bench.py --repeats 3
```

**Setup:**
- Each run is a fresh estimator, `fit(y, fh=1..10)` + `predict()`, and with "PI" also
  `predict_interval(coverage=[0.9])`.
- Repeats: 3 for 100 series, 2 for 1000 series. Machine: 4 cores, shared.
- For stock with pyarrow input, the user converts: `to_pandas` + `set_index`, then stock, then
  `pa.Table.from_pandas` on the output. Stock sktime has no pyarrow mtype.

Seconds, stock / g2 (×):

| forecaster | input | 100×100 | 1000×100 | g2 only, 10000×50 |
|---|---|---|---|---|
| Naive last | pandas | 1.442 / 0.0058 (248×) | 14.66 / 0.0072 (**2,043×**) | 0.017 |
| | polars | 1.284 / 0.042 (31×) | 15.39 / 0.330 (47×) | 2.01 |
| | pyarrow | 1.381 / 0.0059 (235×) | 15.14 / 0.016 (947×) | 0.056 |
| Naive last + PI | pandas | 2.849 / 0.0074 (386×) | 32.31 / 0.0118 (**2,735×**) | 0.043 |
| | polars | 2.987 / 0.041 (72×) | 31.95 / 0.304 (105×) | 2.07 |
| | pyarrow | 3.051 / 0.0078 (390×) | 32.82 / 0.022 (1,486×) | 0.084 |
| Naive drift | pandas | 0.743 / 0.0068 (109×) | 8.27 / 0.0067 (1,229×) | 0.022 |
| | polars | 0.870 / 0.046 (19×) | 8.92 / 0.257 (35×) | 1.94 |
| | pyarrow | 0.889 / 0.0054 (166×) | 8.77 / 0.015 (585×) | 0.058 |
| Poly d1 | pandas | 1.116 / 0.0066 (170×) | 11.40 / 0.023 (498×) | 0.070 |
| | polars | 1.330 / 0.033 (41×) | 10.74 / 0.313 (34×) | 2.01 |
| | pyarrow | 1.180 / 0.0063 (186×) | 11.53 / 0.023 (498×) | 0.108 |
| Poly d1 + PI | pandas | 2.261 / 0.0078 (290×) | 24.68 / 0.019 (1,308×) | 0.093 |
| | polars | 2.494 / 0.037 (67×) | 24.08 / 0.342 (71×) | 2.25 |
| | pyarrow | 2.077 / 0.0097 (214×) | 23.49 / 0.028 (845×) | 0.130 |
| Recursive reducer (LR, w=10) | pandas | 1.486 / 0.192 (7.7×) | 11.84 / 1.62 (7.3×) | 16.4 |
| | polars | 1.298 / 0.221 (5.9×) | 12.32 / 1.92 (6.4×) | 19.0 |
| | pyarrow | 1.368 / 0.158 (8.7×) | 12.42 / 1.70 (7.3×) | 17.2 |

Stock at 10000×50 was not run; it is too slow for the budget. Extrapolating stock linearly in
instances gives about 70–330 s; this is unmeasured.

**Attribution of the g2 path** (`scripts/g2_profile.py`, Naive last, 10000×50, cProfile cumulative
shares, which overlap):

```
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/g2_profile.py
```

| | pandas (0.025 s) | polars (1.87 s) | pyarrow (0.055 s) |
|---|---|---|---|
| `check_is_scitype` on input (`_check_X_y`) | 6.5% | **49.8%** | 27.7% |
| `mtype()` re-inference inside `convert_to` | 7.9% | **49.4%** | 26.9% |
| ingress spoke | 16.9% | 0.3% | 60.5% (includes the pa-checker's metadata pass) |
| `infer_step` (numpy, all rows) | 15.4% | 0.1% | 15.1% |
| `_fit` + `_predict` (kernel + time grid) | 18% | 0.1% | 6% |
| `get_cutoff` (1-element pandas index) | 12.5% | 0.1% | 6.9% |
| egress spoke | 12.5% | 0.1% | 3.4% |

**Interpretation:**
- **The container and kernels are now cheap.** On pandas and pyarrow input, g2 is 18–60 ms for 10,000
  series.
- **What is left on polars is the stock input validation.** It runs twice: `_check_X_y`, then
  `convert_to` re-infers the mtype it was just told. The stock `polars_panel` checker does a
  `group_by` plus a monotonicity check (x6 found the same 93% share).
  - Expected from these shares (unmeasured): passing `from_type=y_metadata["mtype"]` instead of
    re-inferring halves this. A cheaper polars checker removes most of the rest.
- **The reducer gains only 6–9×.** Removing the per-slice framework re-entry (101 fit/predict →
  1/1, 7,484 checks → 36) is all g2 can do. Each instance still runs 1 sklearn `fit` and 10 sklearn
  `predict` calls (11,000 calls at 1000 series). This is the "per-series third-party fit on views"
  ceiling. It is consistent with x8: the re-entry bucket is 33–63% of vectorized fit, with the rest
  in the estimator's own per-series work.

## 6. LOC (`scripts/g2_loc.py`, logical lines: no blank, comment or docstring lines; `get_test_params` excluded)

| | stock | g2 port |
|---|---|---|
| NaiveForecaster (stock: plus `_pivot_sp`/`_unpivot_sp`) | 358 (+70 shared `_BaseWindowForecaster`) | **116** |
| PolynomialTrendForecaster (stock: plus `_get_X_numpy_int_from_pandas`) | 78 | **74** |
| Recursive reducer | 314 (`_RecursiveReducer` + `_Reducer` + sliding-window helpers; these also cover global pooling, transformers and X, which the port does not) | **40** |
| shared port base `_G2Base` (grid, output hub, proba frame, `_update`) | — | 36 |

**Framework, one-off:**
- `g2_np_grouped.py`: 443 LOC.
- Reused x6 spokes and core: 506 LOC.
- Reused a3 `Step` + `NumpyTimeOps`: 130 LOC.

**Each port is a rewrite of the inner methods, not a diff.** A real opt-in would either replace a
forecaster's `_fit`/`_predict` (one class, about 40–120 LOC each) or add a sibling class. Nothing
in `BaseForecaster` changes beyond the registry and the 2 dispatch points (§1.1).

**Scope gaps in the ports** (all raise `NotImplementedError` rather than returning wrong results):
- Naive intervals for sp>1
- in-sample fh
- exogenous X
- `predict_quantiles` / `predict_var` as separate methods
- calendar steps on tz-aware data

## 7. What is lost or open

- **Public-API parity.**
  - No per-instance `forecasters_` DataFrame of fitted clones: fitted state is arrays such as
    `coef_ (U, p, C)` and `estimators_` lists.
  - `get_fitted_params` and any code that reads `forecasters_` would see a different shape.
- **Inner type lock-in per estimator.** Ports must honour stock quirks to stay equal, for example:
  - the drift-variance slope,
  - the frozen window on update,
  - sklearn's `cond=tol` truncation.

  Each quirk was found only by the equality harness.
- **Registry extension is private-cache surgery** (§1.1). It needs a public registration API before
  this can ship.
- **Not measured:**
  - peak memory of the g2 e2e path (x6 measured the hub at 1.3–2 MB vs 24–28 MB for stock Naive at
    1000×100);
  - lazy inputs (polars LazyFrame, duckdb) through the BaseForecaster route;
  - a statsmodels-based port;
  - stock at 10000 series;
  - g2 timing under pandas 3, where only equality was run.
