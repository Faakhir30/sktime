# design_redteam: red-team review of the synthesis draft (design_synthesis_draft)

Role: a skeptical sktime maintainer and performance engineer. Every number below comes from a cited report
or a script I ran, and the command and output are recorded in §0. The overall direction holds up: trusted
lane first, conversion at the edge, and kernels opt-in. Several headline numbers are wrong, double-counted
or mis-attributed, though. One central mechanism (the explicit-context nested entry) cannot deliver its
measured gain as designed. "pandas users see no API change" is false from P3 onward.

## 0. Re-runs done for this review

1. **Nested trusted entry (design_minimal §9.1), re-run at 300x100**
   - Command (WT, EXP_PY, load average 0.02 at start):
     `PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos $EXP_PY -W ignore experiments/container_independence/scripts/design_minimal_nested.py --sizes 300x100 --reps 2 --out <scratch>/design_redteam_nested_300.json`
   - Output: copied to `results/design_redteam_nested_300.json`.

   | input | C1 | C2 | C3 | C3+N (cum.) | N marginal | predict-only C3+N | equal |
   |---|---|---|---|---|---|---|---|
   | pandas | 0.96 | 1.29 | 1.90 | **2.23** (report: 2.16) | **1.18** (report: 1.32) | 1.98 | 4/4 |
   | polars | 1.18 | 1.69 | 2.18 | **3.00** (report: 2.36) | 1.38 (report: 1.54) | 1.78 | 4/4 |

   - The direction is reproduced, and the cumulative numbers come out at or above the report's.
   - Marginals move by ±0.15–0.2x between runs, and C3 on polars (2.18) lies outside g3's Naive C3 range
     (1.55–2.02).
   - **Conclusion:** single-run ratios at n≤1000 carry about ±20–25% run-to-run noise. Quote "about
     2–3x", not "2.16–2.48x".

2. **P2-only marginal (C3/C1)**, computed from the g3 §3.1 table with an inline parser:

   | estimator | C3/C1 | C0/C1 (P0 share) |
   |---|---|---|
   | Naive | 1.35–1.69 | 1.05–1.21 |
   | Poly | 2.04–2.46 | 1.09–1.35 |

## 1. Numbers that are wrong, double-counted or mis-attributed

| # | Draft claim | Problem | Evidence | Fix |
|---|---|---|---|---|
| N1 | P0 expected "[M] final_bench T3 2.62 s→0.019 s at 10k" | 0.0185 s is **x4 adapter[polars] prepare with known mtype**: it keeps the data in polars and never builds pandas. The draft's design converts polars to the pandas lane, which costs about **0.11 s** at 10k (x9 adapter, no check; T3), or **0.011 s** for hub ingress (T3), and hub ingress is P4, not P0. | final_bench T3 rows | Cite 2.62 s→~0.11 s (about 24x) for P0. Move 0.011 s to P4. |
| N2 | P0 lists g3 C1 1.05–1.34x **and** x9 921→33 ms **and** x4 1443→16 ms | Both component numbers are already inside C1, so listing them reads as additive. The x4 verifier says 1443→16 comes from *skipping a redundant check*. Once that check is skipped, the x9 vectorized check only matters for the first (outer) check. | x4_verify0 l.185; g3 §3.3 | State C1 as the single e2e figure and the others as component timings. |
| N3 | P0 "known-mtype / from_type pass-through" credited with C1 | C1(b) was measured as an **identity memo** (weakref). That memo also answered nested `Lag` re-inference (1,005 hits remain after C2). An explicit `from_type` at prepare/convert does not reach `convert_to` on objects created inside `Lag`. g3 says the pandas C1 gain "is mostly per-slice/nested re-inference". | g3 §3.3, caveat 3; re-run C1 pandas = 0.96 (noise) | Label the C1 pandas share of 1.05–1.22 as "identity-memo upper bound; from_type version unmeasured". |
| N4 | P2 "C2 1.33–2.19x then C3 1.55–3.03x cumulative" and scorecard "with fh memo 1.55–3.03x" | These are **cumulative vs stock, including P0 (C1)**. Placed next to P0's 1.05–1.34x they double-count C1. P2's own marginal is **Naive 1.35–1.69x, Poly 2.04–2.46x** (§0.2). | §0.2 | Report P2 as its marginal over P0. Keep "cumulative" only in one waterfall table. |
| N5 | P2 lists g1 fit+predict 1.15–1.88x, g3 C2/C3 and §9.1 together | These are three overlapping measurements of the same lever. g1 is standalone, g3 C2 is the marginal 1.17–1.73x, and §9.1 includes C3. | g1, g3 | One waterfall: P0 → +trusted → +fh memo → +nested. |
| N6 | "Naive reaches 2.16–2.48x with the nested entry [measured]" | Measured with a `TrustedLag` **subclass swapped in by module patch, triggered by input type**, with no context argument (see D1). 4 rows, n≤1000, 2 reps, Lag state not compared. The re-run gives 2.23/3.00, with marginals 1.18/1.38. | design_minimal §9.1 caveats; §0.1 | "~2–3x (prototype mechanism differs from the proposed one)". |
| N7 | P2 proba "[M] 7.9 s _get_columns + 205 ms → 14–85 ms at 10k" | The 10k stock figures are **extrapolated/synthetic** (g4 caveats: 10k stock extrapolated; §4 table uses synthetic frames). Per-instance `_get_columns` is **~4% of the stock interval time** (704 ms of 18.1 s at 1k). Interval time is dominated by quantiles re-entering public predict + predict_var per instance. So "build proba once" is worth about **≤1.05x** on predict_interval, which the draft's own headline (~1.0x) concedes. | g4 impact bullets 3–4 | Label it [E/synthetic]. State the e2e predict_interval effect (≤~1.05x) next to it. |
| N8 | "distribution 826→3.2 ms at 1k" | Stock **Panel predict_proba raises NotImplementedError** (_base.py:1061-1064), and the BaseProbaForecaster path is broken (`_yvec`). The 826 ms baseline was fed by bypassing the bug. This is new functionality, not a speed-up of anything users run, and 11.75 s of per-instance `predict_proba` still dominates. | g4 l.84–98, §5 | Move it to "new capability". Remove it from the speed column. |
| N9 | P3 "polars 34–105x today, [E] ~pandas-level after P0" | g3 C5 already **measured** this with C1 installed. polars is 453–1,363x, but still **~2x below pandas** (1000x100 Naive pd 1,518 vs pl 660; Poly 797 vs 453), because the polars check is still ~45% of a 16 ms run. | g3 §3.1, §3.4 | Use [M] g3 C5: polars about 0.4–0.6x of the pandas-input gain. Drop "~pandas-level". |
| N10 | P4 "[M] T6 selective reads 190–875x, 15.6–25x less memory" | (a) The baseline B1 "sktime eager all" includes the **polars monotonic bug** fixed in P0. wrap_critic: x9 laziness was "mostly the monotonic bug + select-before-convert"; true pushdown over eager select is 1.9x/2.8x. (b) T6 selects **1000 of 20,000 series × last 50 points**. Forecaster `fit` has no instance-selection parameter, and the draft's lookback defaults to None. **No public API in the draft reaches this workload.** | wrap_critic §2; final_bench T6; DEL-1/2 | Quote 1.9x time / 2.8x RSS over select-before-convert. Say which API passes the selection (or add `instances=`/ReadSpec). Otherwise drop the number. |
| N11 | P1 "[E] polars strict-equal API cells 9/24 → ≥20/24" | There is no basis for this. The best measured prototype (x6 boundary) reached 7/24, and x4 reached 5/24. | g5 | Make it an exit target, not an expected impact. |
| N12 | Benchmark CI gate "≥1.5x on Naive/Poly 1000x100" | Naive C3 at 1000x100 is 1.55–1.69x (g3), and run-to-run noise is ±20% (§0.1). On shared CI runners this gate will flap or have to be loosened. | §0.1, g3 | Gate on **counts** (public re-entries, mtype checks, conversions per call: g2/g3 STATS), plus a loose ≥1.2x timing. |
| N13 | Scorecard narwhals "on pandas-long 70–620x slower than the stock check" | This applies only to pandas-long, not to pd-multiindex. The narwhals section words it correctly. | DN-B1 | Keep the scope qualifier. |

## 2. Design defects

**D1 (major). An explicit context argument cannot reach nested transformers inside unchanged forecasters.**
- `Lag.fit_transform(...)` is called from inside `NaiveForecaster._predict_naive`, and the same holds for
  `_lag`/Detrender in reducers, TTF and others. The trusted lane only controls the outer call.
- To pass `ctx` explicitly, every forecaster that builds a nested transformer must be edited, or
  `fit_transform` must gain a public `ctx` parameter that forecasters forward. That breaks the "0 estimator
  changes" claim and covers only forecasters that are actually edited.
- The stated reason for rejecting the ContextVar, "does not reach loky workers", is wrong for this use: the
  ContextVar would be **set inside the per-slice function**, which runs in the worker. Only a var set in
  the parent before dispatch would be lost.
- The prototype that produced 2.16–2.48x used neither design. It was a type-triggered subclass swap (N6).
- **Fix:** a ContextVar set and reset inside the trusted slice runner (worker-side). Keep the explicit
  argument only for the BaseForecaster→`_*_trusted` hop. Re-measure with that mechanism.

**D2 (major). "pandas users see no API change" is false from P3 on.** Ported np_grouped forecasters change
observable pandas-input behaviour in four ways:
- (a) no per-instance `forecasters_` grid, only an unbuilt lazy view that warns on access;
- (b) cutoff `freq` populated where stock returns None (81/315 on pd2);
- (c) the pandas output time level loses `freq` (`<Day>`→None; g3 C5, which `assert_frame_equal` does not
  catch);
- (d) NotImplementedError for Naive intervals with sp>1, in-sample fh, X, and tz calendar steps (g2).

`get_fitted_params` keys and the reconciler/HierarchyEnsemble/ForecastByLevel consumers of `forecasters_`
are affected.

- **Fix:**
  - (a) automatic per-call fallback to the trusted lane for every unsupported method or argument. This is
    required, not "old impl selectable", and the cost of fallback after a batched fit must be stated (x7:
    3.58 s vs 5.6 ms).
  - (b) build the forecasters_ view before P3 ships, or keep the port opt-in by parameter until it exists.
  - (c) add a level-freq check to the equality gate.

**D3 (major). Trusted lane on by default for third-party estimators.**
- g1's guard list covers overridden public methods and the private hooks, which is good. The gate, though,
  is in-tree tests only (7710, run with `-o addopts=""`; `test_vectorize.py` excluded).
- Out-of-tree subclasses that read state set by the public path in a different order, or rely on side
  effects of public `fit` on slices (e.g. logging, mlflow wrappers, `_is_vectorized` probes), get no
  coverage.
- **Fix:** default on for sktime-owned classes. For third-party classes, on only when `check_estimator`'s
  new trusted==stock test has been run (via a tag), or off by default for one release behind a
  FutureWarning.

**D4 (minor-major). The shared `_core` refactor (PR2.1) is the largest and least measured change.**
- g1 copied ~700 LOC. The equality and speed numbers are for the copy, not for a refactored public path.
- The refactor touches every public method of BaseForecaster for all 160 forecasters before any speed
  lands. It cannot ship independently of PR2.2 without a perf/behaviour regression check of its own.
- **Fix:** gate PR2.1 on the pd2xpd3 suite plus the counts benchmark. Allow PR2.2 to land first with the
  copy behind a config, then dedupe.

**D5 (minor). Shared FH mutation plus the fh memo under thread-based parallel backends.**
- Stock `check_fh` mutates the shared FH's freq per slice, and g1 reproduces this on purpose.
- With `backend="threading"` or `dask` threaded, concurrent slices mutate one FH. The new memo keys on
  freq.
- Only loky was checked (g1 caveat).
- **Fix:** give each slice its own FH copy (cheap) and test the threading backend.

**D6 (minor). Dask egress is undefined.** "Output is eager in the same engine" is impossible for dask. Stock
rejects dask input today.
- **Fix:** specify that dask input returns pandas (or dask via `from_pandas`), and label it a new
  capability.

**D7 (minor). Return-type changes are back-compat breaks for current polars users.** This covers
`evaluate()` returning pandas `y_pred` for polars, and the polars y→"0" rename. Users today get pandas or
"0" and may depend on it.
- **Fix:** route these through the same deprecation/config as `output:proba_format`, not as silent bug
  fixes.

**D8 (minor). Soft dependencies and entry points.**
- `backend_of` must identify containers without importing polars/duckdb/dask (check the module name of the
  type first).
- An entry-point group for `register_mtype` loads third-party code on `import sktime.datatypes`, which costs
  import time and runs code on import.
- **Fix:** load lazily, on the first unknown container only. Document it.

**D9 (minor). The memo's best case is the only one measured.** U=1 shared cutoff. With ragged ends a3 gives
1.6–3.8 ms per unique cutoff, so at U=n the memo saves ~nothing, and C3 (1.05–1.53x marginal) disappears.
The draft lists this as an experiment, but the headline expected results should carry the U=1 qualifier
inline, as the risks section already does.

**D10 (minor). Kernel equality is tied to the sklearn version.** The Poly port reproduces sklearn 1.9.1's
`cond=tol` truncation. If sklearn changes, stock and port diverge, and the "equal to trusted-lane run" gate
flips without any sktime change.
- **Fix:** pin the gate to an explicit lstsq reference, and treat the sklearn behaviour as a known deviation
  with a tolerance.

## 3. Shippability of phases

- **P0:** independent, low risk. Keep the null-guard requirement: the unguarded check gave 58/75
  mismatches.
- **P1:** "registrations survive cache_clear" needs the registry refactor first. The conformance gate
  (target 0 SILENT) cannot be green before the P0 bug fixes it tests, so order PR1.5 after P0.
- **P2:** PR2.6/PR2.7 (container-following return types, proba_format) are API changes and should not be in
  the same phase gate as the perf work (D7).
- **P3:** depends on P1 register_mtype and on D2's fallback and forecasters_ view. "~9 closed-form ports" is
  [E]: only Naive and Poly were ported and measured (g2), and the x7 verifier gives a strict 6–8.
- **P4:** the main value claim depends on an instance-selection API the draft does not define (N10).

## 4. What survives

- The trusted lane is the right first lever: g1 7710/7710 tests, 159/160 eligible, and fit/update
  1.2–5.5x/1.3–3.2x.
- The fh memo is a few lines and gives measured 1.05–1.53x marginal at U=1.
- Rejecting narwhals, Arrow internals and per-engine kernels as the core is well supported.
- Kernels are the only 10^2–10^3x lever (g2/g3 C5, measured).
- The expected-results headline must be restated:
  - P0 ≈1.05–1.3x;
  - P2 marginal Naive 1.35–1.7x / Poly 2.0–2.5x (U=1);
  - nested entry "~1.2–1.5x on Naive, mechanism to re-measure";
  - predict_interval / statsmodels ≈1.0x.
