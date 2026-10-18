# wrap_critic — completeness critique of the container-independence study

Role: completeness critic. **No new benchmarks were run** (a serial benchmark agent was busy). Every number below is
quoted from a workstream report or its verifier (`reports/<key>.md`, `reports/<key>_verify0.md`), with the source given.
Proposed experiments are specs only and are labelled *unmeasured*. Claim ids: `wrap_critic.<n>`.

Decision under study: how to decouple sktime forecasters and VectorizedDF from pandas. The candidates are an
adapter/dispatcher, narwhals, a hub (canonical representation), per-estimator opt-in through inner-mtype tags, an
engine-native panel hook, pandas with Arrow memory, a lazy-aware boundary, and combinations of these.

---

## 0. What the evidence already settles (the baseline any gap is measured against)

| Fact | Source (verified) |
|---|---|
| The vectorized cost is about 8–14 ms per instance and linear in n, in every container. The VectorizedDF layer itself costs nothing measurable: vectorized sktime takes 0.93–1.01x the time of a user loop of clone().fit().predict(). | x8.1, x8 verifier |
| The container share of vectorized fit is about 15% for pandas and 2.7–6% polars-specific. Iteration is 1–2% of fit+predict. | a1 verifier, x8.2, x3 calib |
| Per-instance framework re-entry (check, convert, fh, cutoff, clone) is the largest bucket: 52–63% (a1 verifier) or 33–61% (x8). | a1.5, x8.4 |
| Vectorizing the *algorithm* (groupby, segmented numpy, expressions) gives 10^2–10^3x. The engine adds only 1.07–3x on top once conversion is charged. | x7 + verifier, x8.8/9, x5 verifier, r3 verifier |
| No leaf forecaster computes in anything except pandas. 29 of the 152 pandas-inner forecasters take pd-multiindex natively and bypass VectorizedDF. | a2.6, x4 verifier |

So every approach that only changes the container under unchanged per-instance BaseForecaster calls is capped at about
1.05x. The large wins come from (i) removing per-instance re-entry, which helps all 160 forecasters, and (ii)
panel-native kernels, which help a few forecasters a lot. The open gaps below are ranked by how much they block a
quantified impact statement for each candidate.

---

## 1. Approaches not evaluated or under-evaluated

### G1 (HIGH). Trusted inner path / framework fast lane: no integrated prototype
- Status: a1 left it as "unmeasured/expected". The a1 verifier measured clone+_fit per slice at 5.0–6.1x faster than public fit, about 2.1–2.7x end to end. x8's "lite loop" gives 1.09–1.67x (Naive) and 1.66–2.42x (Poly). x8's own open question about it has no prototype in `protos/`.
- Why it matters: it is the only lever that helps all 160 forecasters regardless of container, and it is larger than any container change. Its size decides whether the "adapter vs narwhals" question is worth asking at all.
- Missing: (a) a BaseForecaster subclass or plugin (in `protos/`, not `sktime/`) where VectorizedDF hands already-validated slices to `_fit/_predict` through a trusted entry point. It would skip `check_is_scitype`, `convert_to`, `_check_fh` and `_set_cutoff` re-parsing, and use `convert(obj, from_type, to_type)`. (b) Running sktime's `check_estimator` and forecasting base tests under it. (c) The same timing, done once, for update, update_predict and predict_interval, not only fit+predict.
- Experiment `wrap_critic_g1_trusted_path.py --grid {dev,large}`: subclass the forecasting base in a proto that overrides `_vectorize` so it calls a cloned estimator's `_fit(y_slice, X_slice, fh_rel)` after setting `_cutoff`, `_fh`, `_y`, `_X`, `_is_fitted` from the schema. Interleave with stock for NaiveForecaster(last), PolynomialTrendForecaster, ExponentialSmoothing (if available), make_reduction(LinearRegression) and a TransformedTargetForecaster. Assert equal predictions, intervals and cutoff. Report e2e ratios and the test pass delta. **Output:** "trusted path: X–Yx fit, A–Bx predict, k/N tests pass".

### G2 (HIGH). Per-estimator opt-in numpy / grouped-array inner mtype integrated with BaseForecaster
- Status: x4 ran a toy `pl.DataFrame`-inner forecaster. x6 and r3 built the hub but did not integrate cutoff, ForecastingHorizon or `_check_X_y` (x6 open question 1). a2 shows 74% of forecasters need no core math change under "numpy values + time metadata", but no forecaster was actually ported. sktime already has `np.ndarray` and `numpy3D` mtypes (r3 verifier) that carry no time metadata.
- Missing: a "numpy + Step/TimeAxis" inner mtype family (values plus indptr plus the a3 Step) with converters to and from every outer mtype. Then port 3–5 representative V1/V2 forecasters to it (a V1 statsmodels adapter, NaiveForecaster as V3, PolynomialTrend, a reducer after the `_sktime.py:151` last-window fix) and measure them.
- Experiment `wrap_critic_g2_numpy_inner.py`: a proto mtype `np_grouped` (values float64 2D, indptr, keys table, a3 `Step`, time array). Register the converters in the proto only. Port the forecasters through `y_inner_mtype="np_grouped"` and `capability:panel_native`-style batched `_fit`. Measure (i) e2e vs stock on pandas, polars and pyarrow input, (ii) conversions counted, (iii) LOC changed per forecaster, (iv) equality incl. intervals. **Output:** "port cost ~L LOC/forecaster; e2e speedup X; conversions 2N→2".

### G3 (HIGH). Combined architecture never measured end to end
- Each component was measured alone: hub ingress (x6), Step time axis once per panel (a3), trusted path (G1), vectorized cutoff (a1), backend-aware get_cutoff (x4), the monotonic-check fix (x9), the panel-native hook (x7). They do not add up linearly. For example, x8's verifier notes that once re-entry is gone, previously hidden costs (the Lag transformer) dominate.
- Experiment `wrap_critic_g3_stack.py --grid {dev,large}`: build cumulative configurations C0 (stock) → C1 (+monotonic fix and known-mtype pass-through) → C2 (+trusted path) → C3 (+once-per-panel time axis and vectorized cutoff) → C4 (+hub iteration with trusted pandas view) → C5 (+panel-native kernel where available). Run them for NaiveForecaster and PolynomialTrend on pandas, polars and polars-lazy input. Report a waterfall of the marginal gain of each step, and equality at each step. This is the single most decision-relevant missing table.

### G4 (HIGH). Probabilistic outputs and the second vectorization path
- `_base_proba.py` (`_vectorize_predict_proba`, `_concat_distributions_hierarchical`) and `base/_proba/_mixin.py` are unmapped (a1 verifier). No approach (adapter, narwhals, hub, hook) was tested on predict_interval, predict_quantiles, predict_var or predict_proba. The x4 open question asks whether proba needs a column-level adapter op. predict_interval returns pandas MultiIndex columns even for polars input (a1.4).
- Experiment `wrap_critic_g4_proba.py`: for each prototype (x4 adapter, x5 narwhals, x6 hub, x7 hook), call predict_interval(coverage=[.8,.9]), predict_quantiles and predict_var on Panel and Hierarchical input. Record (a) output type, (b) equality with stock, (c) time. Prototype one long/tidy proba schema (key cols, time, var, coverage, lower, upper) and one flattened-column schema, and measure the conversion back to MultiIndex columns. **Output:** "proba path: k/4 prototypes work; long-format egress cost X ms at 10k".

### G5 (MEDIUM-HIGH). Exogenous X, update, update_predict, predict_residuals, and composites
- Almost all benchmarks are y-only fit+predict. Known breakages: the polars X/y containment check passes silently on disjoint dates (a1.10), and update_predict and predict_residuals fail for polars (a1.12). x4's cutoff bridge ignores `reverse_order` (wrong in 11/11 cases). The composite verdicts are not transitive (a2 verifier): tuners run on the pandas-based `evaluate`, and HierarchyEnsemble/Reconciler need named levels.
- Experiment `wrap_critic_g5_api_surface.py`: a matrix of {prototype} × {fit(y,X), update(y_new,X_new), update_predict(cv), predict_residuals, evaluate(), TransformedTargetForecaster([Deseasonalizer, Naive]), ForecastingGridSearchCV, ReconcilerForecaster} × {pandas, polars}. Record pass/fail and equality with stock. **Output:** "API surface coverage k/M per approach".

### G6 (MEDIUM). Native-multiindex forecasters (29 classes) and global/DL models
- The 29 pd-multiindex-inner forecasters (reducers, pipelines, NeuralForecast, PytorchForecasting, MOIRAI, LagLlama) never touch VectorizedDF, so no VectorizedDF-level approach reaches them (x4 verifier). The 47 torch and foundation wrappers were not run (a2 open question). darts and neuralforecast accept polars natively (a2 verifier): 8 classes where a native path pays off directly, never measured.
- Experiment `wrap_critic_g6_native_mi.py` (own venv with neuralforecast or darts): time NeuralForecast or darts with pandas-converted vs native polars input on 1k series, with conversion charged. Separately, time the "(n,T) array export" layout from the x7 open question for global models. **Output:** "polars-native passthrough to darts/neuralforecast saves X% of fit".

### G7 (MEDIUM). Distributed and out-of-core engines are barely evaluated
- dask: shipped sktime rejects dask input (x8 used a tag patch). Fugue/partition-and-delegate ran at 1.2–1.5x in the like-for-like comparison (r4 verifier), not 21–47x. Stay-lazy on dask was 2.6–10x slower, but only with the synchronous scheduler (x1). Partition granularity is open (r3). Spark and ibis-on-backends were not run.
- Experiment `wrap_critic_g7_partition.py --grid large`: dask with the threaded scheduler, `set_index(sorted=True)` + `map_partitions` running a vectorized pandas kernel and the stock sktime loop per partition, with partitions in {1,4,16,64} and 10k–100k series. Report the speedup vs single-process stock and the point where dask breaks even. **Output:** "partition-and-delegate break-even at N series / P partitions".

### G8 (MEDIUM). pandas-with-Arrow memory under pandas 3 defaults; modin and cudf
- x9 tested Arrow value columns plus a numpy index. The cheap fix of coercing the time level at the boundary was costed in isolation (0.25 ms zero-copy on pd3) but not end to end through VectorizedDF. The hybrid survives in only 25% of outputs, and 7/71 forecasters fail (x9 verifier). pandas 3's default pyarrow-backed `str` dtype for keys has no impact on iteration and groupby cost measured for it. modin was tested only on dask (r4). cudf.pandas was not tested (no GPU; label as out of scope).
- Experiment `wrap_critic_g8_arrow_pd3.py`: on EXP_PY3, polars → `to_pandas(use_pyarrow_extension_array=True)` → coerce the time level to numpy → `VectorizedDF.items()` → NaiveForecaster. Measure on single-chunk and multi-chunk (parquet-read) input. Report e2e time and memray peak vs the numpy adapter.

### G9 (MEDIUM). Creative alternatives that were mentioned but not prototyped
- **Segmented-kernel library** (numpy `reduceat`/indptr, optionally numba): r3 verifier and x7 show it beats polars expressions (linear trend 5.5x faster than polars, 63x faster than the hub loop). No one estimated how many forecasters it covers beyond the about 6–9 closed-form ones (x7 verifier), or tested numba for recursive models (ETS, Theta, ARIMA state recursions). The x5 open question asks about hybrid narwhals state extraction plus a numpy kernel.
  - Experiment `wrap_critic_g9_numba_ses.py` (own venv with numba): a segmented SES/Holt recursion over indptr, fixed and optimised alpha, vs sktime ExponentialSmoothing per series. Report speedup and max abs diff.
- **Array-API (`xp`) inner compute** for numpy-inner forecasters, so torch, cupy or jax work (r1 template). Unmeasured.
- **Arrow ListArray / large_list "one row per series"** as the hub layout, the native nested form in polars, duckdb and arrow. It avoids the indptr↔keys bookkeeping and pushes down to duckdb `list_*` functions. Not prototyped.
  - Experiment `wrap_critic_g9_listarray.py`: build `pa.large_list` from the polars panel (group_by.agg) and compare build, iterate and egress cost and fidelity with the x6 hub.
- **Selection-aware eager boundary** (x9 verifier): pushdown adds only 1.9x time and 2.8x RSS over "select before convert". This is a cheaper alternative to new lazy mtypes and was not designed as an approach (which selections — instances, window_length tail, columns — and which estimator tag declares the lookback). x6 open question 3 is the same idea.
  - Experiment `wrap_critic_g9_lookback_tag.py`: a proto tag `lookback_window` (Naive last = 1, sp-naive = sp, Poly = all). The boundary applies `tail(w).over(key)` before conversion. Measure the time and memray peak of fit on a 2M-row parquet for eager-select, lazy-pushdown and stock.

### G10 (MEDIUM). Maintenance and extension cost is not quantified on a common scale
- LOC figures exist but are not comparable: adapter 67–100 LOC plus a 370-LOC mtype family (x4), narwhals VectorizedDF 381 SLOC (x5), hub spokes 449 hand-written / 120 narwhals / 145 capsule (x6), pandera's narwhals backend 3,801 LOC (r4 verifier), skrub 72+17 dispatch functions (r1). Missing: for each approach, (a) LOC to add one backend, (b) LOC to port one forecaster, (c) the number of test-matrix cells (backends × forecasters × methods), (d) the number of known silent-divergence sites.
- Experiment `wrap_critic_g10_cost_table.py`: a static script (no benchmark) that counts SLOC per proto module with `ast`, counts test parametrisations, and tabulates them. Cheap, and it gives the "cost" axis of the decision.

---

## 2. Claims that remain unverified, refuted or contradicted

| id | Claim | Status | Source |
|---|---|---|---|
| x3 (all) | Every iteration-strategy ratio | **Report never written.** Only a 10x100 smoke run plus the verifier's 10x100/1000x100 re-runs exist. S4 0.94x refuted (1.24–2.16x). The ratios depend heavily on scale. | x3 verifier |
| r3.3 | A hub under unchanged pandas forecasters is 2.2x slower | **Refuted**: 1.4–1.6x faster with `date_range` and the schema freq. Contradicts x6.5's verifier ("safe public view is 2.1–2.3x slower"), see §4 K3 | r3 verifier |
| r3.8 | Polars expressions beat the numpy hub by 10x | **Refuted**: segmented numpy beats polars | r3 verifier |
| x4.6 headline | Polars-inner fit 0.36–0.43x | Strawman baseline. vs today's A0 it is 0.96–1.17x, so **no win** | x4 verifier |
| x6.1 | 309x ingress | Measures sktime's slow check. vs an unchecked baseline 6–10x, and the hub validates nothing | x6 verifier |
| x6 hub | Correctness | NaN keys merge instances, mixed freq gives wrong timestamps, multivariate uses column 0 only, drift NaN is silent. **Unfixed, and the cost of fixing is unmeasured** | x6 verifier |
| x7 hook / x2 nw_offsets_iloc / r4 offsets | Speedups | All assume sorted, contiguous rows and are **silently wrong** on shuffled input. The cost of the guard (sort or arg_max) is unmeasured | x7, x2, r4 verifiers |
| x1.8 | 6–35x+ lower memory | Refuted → 1.3–7x (fresh-process ru_maxrss) | x1 verifier |
| x9.3 | convert_to 25–28x cheaper | → about 10x e2e. The vectorized check is **not equivalent with nulls** (58/75 mismatches) | x9 verifier |
| x9 laziness | 119–442x | Mostly the monotonic bug plus select-before-convert. True pushdown is 1.9x | x9 verifier |
| x5/x7 | narwhals/polars forecasters 10^2–10^3x | Credit belongs to vectorization. Plain pandas groupby is about 10x faster than narwhals-on-pandas | x5 verifier |
| x8.9 | polars 2.3–5.4x faster than pandas panel-native | → 2.1–3.0x vs a lean pandas groupby | x8 verifier |
| x8 | Keeping slices in polars with pandas-inner forecasters gives +50–80% regression | **Unmeasured** (verifier extrapolation), and contradicted by x4 verifier's numpy slice path (0.08–0.12x) | x8 / x4 verifiers |
| a3.1 | Step exact | Only for midnight, tz-free data. SME, tz W, numpy BMS/BME and int step 5 fail. Off-grid absolute fh unmeasured | a3 verifier |
| a3.8 | 17–27% per-instance fh share | Not re-run (verifier measured 8–12%) | a3 verifier |
| a2.4 | 83.5% / 67% enforcement pass | Upper bounds (taint lost through Lag and concat). The number of false OKs is unmeasured | a2 verifier |
| r1.7 | "wrap once per batch makes narwhals cheap" | Holds for `.shape` only. Compute ops cost +13–25 µs | r1 verifier |
| r3.9/.10, darts/functime timings, r4 Fugue DuckDB | Lazy collect cost, fugue-on-dask | **Not re-verified** (venvs gone) | r3 verifier |
| x6.6 | 1.01–1.28x pandas-view sub-claim | Not re-run | x6 verifier |
| x7 | 10k sktime baseline | Extrapolated, not re-measured | x7 verifier |
| x8 | n=5000 single-rep numbers; vectorized monotonic correctness on unsorted or duplicate input | Not verified | x8 verifier |
| All | Large grid | Pending the serial phase; every ratio above is DEV-grid | — |

---

## 3. Missing experiments, per approach, to state a quantified impact

| Approach | Has (verified) | Missing for a full impact statement | Experiment |
|---|---|---|---|
| Adapter/dispatcher (x4) | VectorizedDF-level 2.2–3.7x on polars; pandas neutral; 0 conversions | e2e vs A0 with the **trusted path** (otherwise ≈1x); proba/update coverage; fixed cutoff bridge (reverse_order); unnamed-level fix | G1+G3, G4, G5 |
| Narwhals VectorizedDF (x5) | pandas 1.9–6.3x slower, polars 0.26–0.32x; type round trip on 6 backends | Cost of the **dual path** (pandas fast path + narwhals); behaviour under stable.v1/v2 (x1 verifier: last(order_by) missing); pyarrow over() bug workaround cost; e2e with a forecaster | G3, G10, `wrap_critic_k_nwstable.py` |
| Narwhals inside forecasters | +10–35 µs/call, +150–340 µs with_columns | e2e on one real ported forecaster vs a plain-polars port (the extrapolation +0.2–1.5 s is unmeasured) | `wrap_critic_g11_nw_port.py`: port PolynomialTrend to narwhals and to plain polars, time at 1k/10k |
| Hub (x6, r3) | ingress 6–15x, slicing 70x, numpy kernel 10^2–10^3x | Hub with **validation and guards** added (NaN key, sortedness, per-series freq); memray-based memory; BaseForecaster integration (G2) | `wrap_critic_g12_hub_guarded.py` |
| Per-estimator inner-mtype opt-in | toy polars-inner (x4); census 74% | real ports + porting LOC (G2) | G2 |
| Engine-native panel hook (x7) | 570–2006x at 1k for naive | sortedness guard cost; how many forecasters (strict 6–8); fallback cost when an unsupported method is called after native fit; proba from the hook | `wrap_critic_g13_hook_guard.py`: add `is_sorted` check / arg_max-by-time variant, re-time; call predict_interval after native fit and time the fallback materialisation |
| pandas + Arrow memory (x9) | conversion 1.2–1.4x like-for-like; 64/71 forecasters OK | e2e with time-level coercion on multi-chunk input; pd3 defaults | G8 |
| Lazy-aware boundary (x9, x1, x6) | pushdown 1.9x / 2.8x RSS over eager select; duckdb 4x RSS | the lookback tag design and the share of forecasters that can declare a finite lookback (static census of `window_length` / `sp` / `strategy`) | G9 lookback, `wrap_critic_g14_lookback_census.py` |
| Time axis (a3) | once per panel 10^3x for the component (≈1.1–1.4x e2e) | freq × time-of-day × tz test matrix; int-step semantics; off-grid absolute fh; lru_cache `__eq__` fix gain | `wrap_critic_g15_step_matrix.py` (reuse a3 verifier semantics script with an extended grid) |
| Output-contract change (forecasters_, cutoff, proba columns) | EstimatorGrid: 0 behavioural failures in 204 tests; 4/4 metric configs break | deprecation-shim cost; the impact on downstream users (how many sktime modules or tests index `forecasters_` with `.loc`/`.iloc`) | `wrap_critic_g16_contract_census.py`: grep plus a test run with a shim that warns on DataFrame access |

Cross-cutting measurement gaps:
- **M1 Memory methodology is inconsistent** (in-process sampler in x1, VmHWM delta in x6, which under-counts allocations, RSS deltas in x9 as lower bounds). Re-run the memory claims for x1, x6 and x9 under one harness: a fresh subprocess per route, `ru_maxrss` plus `memray --peak`, and the input read from parquet (not built from one pandas block). Script `wrap_critic_m1_mem_harness.py --grid {dev,large}`.
- **M2 Baseline asymmetry recurs** (bare groupby vs real items(); UDF output frame charged to the framework; polars inputs pre-existing). Define two canonical baselines in common.py (`items_real`, `groupby_bare`) and require every ratio to cite one. Re-express the r2, x2 and x5 iteration ratios against `items_real`.
- **M3 Scale dependence**: x3 shows ratios flip between 10 and 1000 series. All the "small n" headlines need the large grid. Also, staggered start dates make S1 3.5x slower (x3 verifier), and the realistic panel layout (unequal, staggered, gappy, time-major) is used in almost no benchmark. Add `make_panel(layout="staggered|gappy|time_major")` and re-run x4, x5 and x6 bench at dev scale.
- **M4 Silent-correctness regression suite**: one adversarial fixture set (NaN keys, unsorted rows, time-major, mixed freq, unnamed levels, int time with step ≠ 1, Period M/Q on pd3, mixed dtypes, int64 > 2^53, empty frame, series shorter than sp) run against **every** prototype. This turns the many per-workstream silent bugs into one comparable "k/M correct" number per approach. Script `wrap_critic_m4_adversarial.py`.

---

## 4. Contradictions between workstreams and verifiers

- **K1. Ceiling for container decoupling.** x8 says ≤1.05x. x4 says polars is 2.5–3.3x faster, and x5 says 8x faster than today's polars path. Resolution: x4 and x5 measure prepare+iterate+reconstruct only, without the forecaster. With a forecaster, x4's verifier finds 0.96–1.17x vs A0. Both are right about different scopes, and the final report must say which scope a number applies to.
- **K2. Size of the re-entry share.** a1 says 62–84%, the a1 verifier 52–63% (2.1–2.7x e2e), and x8 33–61% (lite loop 1.09–2.42x). The definitions differ in whether clone and fh construction count. G1 resolves this with one prototype.
- **K3. Hub + unchanged pandas forecasters.** r3 says 2.2x slower, but its verifier says 1.4–1.6x faster (date_range with the schema freq). x6 says 4.1x cheaper per slice, but its verifier says a safe public DataFrame view is 2.1–2.3x slower, and a trusted view 1.9–2.3x faster. These are consistent only if `date_range` is "safe". `date_range` regenerates rather than slices, so it is wrong for gappy or irregular series and needs a per-series regularity check, at a cost nobody measured. Experiment `wrap_critic_k3_view.py`: the four constructors (DatetimeIndex(validate), date_range(schema), `_simple_new` trusted, iloc on the original frame) on regular, gappy and mixed-freq panels, with equality to items().
- **K4. Narwhals iteration on pandas.** r2 says 0.91–2.98x of current, x2 0.33–0.75x, the x2 verifier 1.7–2.3x *slower* on pd3 when accounting is symmetric, the r2 verifier 0.36–1.54x vs the real items(), and x3 S7a 1.69x. The sign flips with the baseline and the pandas version. Resolution needs M2.
- **K5. Cost of C2.** r2 says reset_index is 0.05–0.97x of a bare groupby, x2 0.7–8.2% of items(), the x2 verifier 3–26% of a bare groupby, x5 0.18–0.49x of prepare, and r3's verifier 5–6%. The r2 verifier's point dominates: the real cost is the **per-group set_index** (1.1–9.8x of items()), not the one-time reset. So C2's cost depends on whether forecasters receive pandas-with-index slices.
- **K6. Whether C2 holds at all.** "No MultiIndex in narwhals" (C2, pandera's comment) vs "27/37 ops keep the MI" (x2) and "group_by on an index-level name works" (r4 verifier). The latter rely on **undocumented pandas leakage**, and a column/level name collision silently loses names (r2 verifier). Not resolved: does narwhals intend to keep this behaviour? Experiment `wrap_critic_k6_nw_mi_contract.py`: run the x2 37-op probe under `narwhals.stable.v1`, `stable.v2` and main, and check the narwhals test suite and docs for an index guarantee.
- **K7. C1 nuance.** One verifier says "collect(backend=) can force any of the three eager types for any input" (r4), another "can never return dask/duckdb" (x5/x6). These are consistent. But **under narwhals.stable.v1, DuckDB and ibis come back as DataFrame with no collect()** (x1 verifier), so the answer to C1 depends on the narwhals namespace sktime pins, and no workstream picked one.
- **K8. Per-slice conversion penalty.** x4 says dispatching on input type is structurally wrong (per-slice conversions 1.5–4.3x), and the x8 verifier expects a +50–80% regression. The x4 verifier says a numpy-based polars-slice → pd.Series path is 0.08–0.12x of today's convert-once. The penalty belongs to sktime's `convert_to`, not to the design. Experiment `wrap_critic_k8_slice_convert.py`: e2e NaiveForecaster fit with per-slice conversion via (a) convert_to, (b) `convert(from_type)`, (c) the raw numpy path.
- **K9. Size of the polars mtype-check gap.** x4 says about 1000x (warm pandas cache), its verifier 71–263x, x8's verifier 654x (under load), and x9 857 vs 38 ms. The common conclusion holds (vectorize or skip the check). Only the ratio is in dispute.
- **K10. Value of laziness.** r3 says 9–14x, x9 119–442x, the x9 verifier 1.9x over selection-aware eager, x1 1.1–3x for grouped aggregation (memory 1.3–7x), x6 4x RSS on duckdb but none on polars, and x4 says laziness cannot survive per-instance forecasters (3–77x slower). There is a consensus that laziness only pays for panel-native or lookback-pushdown forecasters. The headline numbers mix this up with the monotonic bug.
- **K11. pd3 caller mutation.** x2 and x3 both report that `_enforce_index_freq` mutates the user's frame on pandas 3. The x2 verifier scopes it to equal-length panels. It also invalidates the x3 pd3 "no-freq" control. This is an independent bug fix that no workstream filed as a task.
- **K12. Small factual disagreements**: one shared last-window line covers 7 vs 8 classes (a2 vs verifier); cutoff reliance is 80% vs 93% (a2 vs verifier); Arrow survives in ~50% vs 25% of outputs (x9 vs verifier); pyarrow narwhals iteration is 20–28x vs 13.8x slower (r2 vs verifier); a3's "hybrid faster than polars" is an artifact (non-deduplicated P1); polars group order is [c,a,b] vs nondeterministic (r1 vs verifier).
- **K13. Reports contradict their own cloud re-runs** (r1 body, x1 TL;DR): the old local numbers sit next to the new ones. The final synthesis must use only the cloud or verifier numbers.

---

## 5. Independent bug fixes surfaced (low-risk, decision-independent; impact partly measured)
1. Vectorize polars `is_monotonically_increasing`, **with a null guard** (x9; ~10x convert_to e2e).
2. Pass the known mtype or use `convert(from_type)` instead of re-inferring it (x4: prepare 90x; x8: 10x per convert).
3. `_HashIndex` needs `__eq__` so the `_to_absolute` lru_cache can hit (a3 verifier: 0 hits today; gain unmeasured).
4. Stale `hasattr(X, "X")` duck typing at `_utilities.py:59` and in get_cutoff (column named "X" breaks fit; a1).
5. polars X/y index containment check is a no-op (a1.10).
6. pd3 `_enforce_index_freq` mutates the caller's frame (x2, x3).
7. update_predict and predict_residuals on polars fail (a1.12).
8. CustomBusinessDay holidays dropped (1,095/3,600; a3.2).
9. NaiveForecaster's FutureWarning (remember_data) is printed once per clone despite -W ignore (x4, x8).
10. PolynomialTrendForecaster is not OLS under sklearn 1.9's tol (max abs diff 17.8–19.7; x7.6).
None of these needs an architecture decision. Missing: a single measurement of their **combined** e2e effect (this is C1 of G3).

---

## 6. Conclusions (critic)
- wrap_critic.1: The study answers "which container layer" well, but it has **not measured the lever it identified as dominant**. A trusted per-slice path (G1) and its combination with the other components (G3) are needed before any approach can be given an honest end-to-end number.
- wrap_critic.2: The per-estimator opt-in (numpy/grouped inner mtype with a framework time axis), the proba path, X/update paths and composites are under-evaluated for **every** approach (G2, G4, G5).
- wrap_critic.3: Several headline speedups rest on unguarded assumptions (sortedness, contiguity, NaN keys, global freq) that make the fast paths silently wrong where sktime raises. A shared adversarial suite (M4) and guarded re-timings are required before those speedups can be cited.
- wrap_critic.4: Memory claims across x1, x6 and x9 use incompatible and partly invalid methods. They should be re-run under one fresh-process memray harness (M1).
- wrap_critic.5: The narwhals answers to C1 and C2 depend on the narwhals namespace (main vs stable.v1/v2) and on undocumented pandas-index leakage. The pinning decision is unmade (K6, K7).
- wrap_critic.6: x3 never produced its report. Its smoke-scale conclusions are unreliable, and several of them flip at 1000 series.
