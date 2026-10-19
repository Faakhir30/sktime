# judge_maintainer: the six designs reviewed by an sktime core maintainer

This review looks at back-compatibility and deprecation policy, how simple the extension contract stays for
third-party authors, the effect on the test framework (`check_estimator` and the datatypes suites), the soft-dependency
footprint, and consistency with skbase and the other scitypes that share `datatypes`/`VectorizedDF` (transformers,
classifiers, metrics).

No new benchmarks were run. Every number below comes from a report or a results file. I checked citations by grepping
the reports and the `results/design_*.json` outputs listed in §3.

## 1. Scores

| rank | design | score | one-line verdict |
|---|---|---|---|
| 1 | design_minimal | **8.0** | Changes the public contract for pandas users the least, needs no new dependency, and requires 0 estimator changes. It also measured the next lever (nested trusted entry). Its weak spots are implicit ContextVar state and no public mtype registry. |
| 2 | design_tiered | **6.5** | Uses sktime's own idiom (inner-mtype tags) and has the best extensibility story (public `register_mtype`, entry point, versioned caches). Against it: tier sprawl, a global `inner_dispatch` flip, `forecasters_`→FittedGrid for everyone, and deleting ported pandas implementations that third parties subclass. |
| 3 | design_adapter | **6.0** | Moves the pandas family "as is", makes the conformance suite a CI gate, and adds public registries. Against it: it puts a FutureWarning on `forecasters_.loc/.iloc` for pandas users too, adds four new registries/abstractions, and its migration census is regex-narrow. |
| 4 | design_hub | **5.0** | Replaces `VectorizedDF`, which transformers, classifiers and metrics also use, so the blast radius crosses scitypes. The float64 hub loses NaN/null and int64 fidelity, the fast path relies on private pandas `_simple_new`, and `forecasters_`→EstimatorGrid at N+2 affects all users. The guards are measured, which counts in its favour. |
| 5 | design_narwhals | **4.5** | Adds a new pinned core dependency (`narwhals.stable.v2`), two PanelOps implementations that must stay equivalent, per-backend kernel allow-lists and known narwhals silent bugs. It also deprecates `forecasters.loc[...]` fitted-param keys for every user, which breaks code for no gain. |
| 6 | design_engine_lazy | **4.0** | Adds a second estimator API (`_fit_panel/_predict_panel/_predict_interval_panel/_update_panel`) per engine family (polars, duckdb, numpy), new lazy mtypes, kernels that depend on duckdb/dask soft deps, and a `lookback` tag that silently changes forecasts when mis-declared. It is the most honest about that hazard (DEL-1). |

All six agree on the parts a maintainer cares about most:
- pandas input keeps the proba MultiIndex permanently;
- cutoff/fh stay pandas;
- the trusted per-slice core (g1) is the main lever, with class-tag/config opt-out;
- P0 bug fixes come first.

The scores therefore turn on the contract each design adds on top of that common core.

## 2. Per-design assessment

### design_minimal: 8.0
- **Back-compat.** C1–C4 are unchanged for pandas. The only planned flip is `output:proba_format` for non-pandas input,
  with a FutureWarning in N and the flip in N+2, which follows sktime's deprecation practice. Kernel forecasters
  document "no `forecasters_`", the same precedent the 29 existing pd-multiindex natives set.
- **Extension contract.** No new estimator-facing concept. A kernel uses the existing
  `y_inner_mtype="pd-multiindex"` tag, so the extension templates do not change.
- **Tests.** The g1 7710-test gate runs in both modes. No new mtypes, so the datatypes conversion matrix stays as it is.
- **Dependencies.** None added.
- **Cross-scitype.** `VectorizedDF` is untouched apart from taking `from_type`. `_fit_transform_trusted` is added to
  `BaseTransformer`.
- **Concerns:**
  - The `TRUSTED` ContextVar is implicit global state, which is unusual for skbase. It does not cross process
    boundaries: with `backend:parallel` on loky, nested calls inside workers silently take the slow path. That is
    safe, but it should be documented or replaced with an explicit argument.
  - `register_spoke` is public, but there is no public `register_mtype`, so the g2 private-lru_cache problem stays
    for third-party mtypes.
  - §9.1 measured only `Lag` in Naive (n≤1000, 2 reps), and Lag state was not checked against stock.
  - It does not address the memory loss from not having a hub. It discloses this (§7).
- Citations check out (see §3).

### design_tiered: 6.5
- **Strengths.**
  - Public `register_mtype(cls, converters, cost)` plus an entry-point group plus versioned caches. This is the right
    fix for g2's finding that `cache_clear` silently unregisters an mtype, and it helps third-party packages.
  - Opt-in goes through the existing tag.
  - The census of implementations is real: 96 distinct vectorized implementations; top 10 cover 28.2% and top 50
    cover 62.9% (`results/design_tiered_census.json`).
- **Concerns.**
  - Five tiers (T0, T1, T2, T3, T3p) plus `inner_dispatch` is a lot of new vocabulary for contributors.
  - Flipping `inner_dispatch` to `"auto"` in N+2 changes which inner mtype existing multi-mtype estimators receive.
    Every estimator that lists several mtypes has, in practice, only been tested on the first one.
  - `forecasters_` becomes a `FittedGrid` at N+2 for all users, including pandas users (a1: 4/4 hierarchical metric
    configs break).
  - "Once the harness passes, the old pandas implementation is deleted" (§ migration) breaks external subclasses of
    `_BaseWindowForecaster`, `_Reducer` and `_GeneralisedStatsForecastAdapter`. Third-party code subclasses these
    bases.
  - The test matrix (tiers × backends × methods) is admitted to be unsized (G10).
  - The `cutoff.freq` "always populated" change is a behaviour change. g2 found it differs from stock in 81/315 cases
    on pandas 2, so tests that compare cutoffs exactly may change.

### design_adapter: 6.0
- **Strengths.**
  - Today's pandas code moves into `InnerFamily('pandas')` unchanged. The pandas adapter was shown bit-identical:
    x4 78/78 and x4 verifier 28/28.
  - A per-adapter conformance suite (g5 matrix plus M4) is a CI gate. This is the most check_estimator-like
    proposal.
  - Public `register_adapter`/`register_mtype`.
  - Dynamic composite inner-mtype tags, following sklearn `array_api_support`.
- **Concerns.**
  - The `PandasCompatGrid` emits a FutureWarning on `.loc/.iloc/.values/...`, which hits pandas users who never asked
    for a container change. That is a deprecation with no benefit to them; a1 §6 puts the grid at ≲1% of fit.
  - Four registries (FrameAdapter, InnerFamily, FrameSchema/TimeAxis, OutputSpec) is a large new framework surface
    for maintainers.
  - Per-backend adapters cost 67–100 LOC each and each needs soft-dep CI.
  - The composite tag derivation has no prototype.
- The census (`results/design_adapter_contract_census.json`) is accurate as reported: 2 non-test
  pandas-attribute sites. The same file shows 89 `forecasters_` references in 19 files, and the regex misses aliased
  access (the design's own caveat). "Small blast radius" is therefore a lower bound, not a measurement.

### design_hub: 5.0
- **Strengths.**
  - Guards that cannot be turned off, with measured cost: 1.4–3.5x of unguarded ingress, still 29–71x cheaper than a
    bare stock check on polars (design_hub §9). This mitigates silent corruption, which matters a lot to a
    maintainer.
  - Honest E1–E12 list.
  - Public `register_spoke`/`register_mtype`.
- **Concerns.**
  - An Executor replaces `VectorizedDF`/`_vectorize`. `VectorizedDF` is shared with `BaseTransformer`,
    `_base_panel` and the metrics (`vectorize_est`: 15 non-test sites in 6 files, adapter census), so every scitype
    inherits the hub's fidelity limits: x6 verifier 25–26/33 adversarial fixtures; NaN/null distinction and
    int64>2^53 lost.
  - The fast pandas view relies on private `_simple_new`; g3 hit 2 pandas-3 bugs there.
  - Guarded pandas ingress costs 10–25x the bare stock check (E2).
  - `forecasters_`→EstimatorGrid at N+2, plus new tags `fitted_state` and `proba_inner_format`.

### design_narwhals: 4.5
- **Strengths.**
  - It treats C1/C2 as hard constraints: explicit `materialize`, the x2 schema recipe at 198/198.
  - The pandas lane stays native.
  - `forecasters_` is unchanged on the per-instance lane.
  - The backend × strategy conformance suite at ≥1000 groups is a good rule, since the x5 pyarrow bug is invisible at
    n=20.
- **Concerns.**
  - It makes `narwhals.stable.v2` a core dependency. It claims "sklearn>=1.9 hard-depends on narwhals, so it is
    already a transitive dependency". That holds only for users on sklearn 1.9.x; sktime's bound is
    `scikit-learn>=0.24,<1.10` (r1 lines 34 and 254 say exactly this).
  - It takes on narwhals silent bugs: pyarrow `over()`, when/then-in-agg on pandas, duckdb `total_seconds`.
  - Two PanelOps implementations must stay equivalent, with the cost unmeasured (G10).
  - Per-kernel backend allow-lists.
  - It deprecates the `forecasters.loc[...]` fitted-param keys for everyone in P4. That breaks
    `get_fitted_params` users with no speed gain, and a1 contract 4 needs no change.
  - The new `nw_long` eager+lazy mtype family adds to the datatypes test matrix.

### design_engine_lazy: 4.0
- **Strengths.**
  - The `lookback` tag is per method, defaults to None and is "declared by construction, never inferred". This is
    backed by DEL-1: AutoETS, ExponentialSmoothing and UnobservedComponents looked truncation-safe by coincidence,
    and only 1/39 is safe for intervals.
  - The order-robust kernel contract is measured: 1.11–1.38x the cost of positional, correct on shuffled input.
  - `forecasters_` is unchanged for R2. `params_` and `cutoffs_` are additive.
- **Concerns.**
  - A second estimator API family (4 hook methods × up to 3 engine families × K forecasters). Third-party authors
    face two ways to write a forecaster, and check_estimator would need engine-parametrised scenarios.
  - Kernels written per engine in duckdb SQL and polars make duckdb and polars effective test dependencies of core
    estimators.
  - New lazy Panel/Hierarchical mtypes.
  - Dask output changes key dtype from object to `string[pyarrow]` (DEL-4).
  - A mis-declared lookback silently changes forecasts, and check_estimator cannot detect that in general.
  - The gain is limited to 6–8 strict kernel candidates (x7 verifier).

## 3. Citation verification (factual errors and overstatements)

| design | claim | finding | source checked |
|---|---|---|---|
| adapter | open experiment: "expected polars gain ≈ pandas-input gain (98–2,735x)" | **Typo/error**: g2 gives 498–2,735x | g2_numpy_inner.md:270 / summary |
| adapter | "only 2 non-test pandas-attribute reads of forecasters_" | Correct for the regex, but the same census has 89 `forecasters_` refs in 19 files, aliased access is not caught, and metrics reach the grid through `vectorize_est` (a1: 4/4 configs break). **Understates** the surface | results/design_adapter_contract_census.json; a1 L36-41 |
| narwhals | "sklearn>=1.9 hard-depends on narwhals … so it is already a transitive dependency" | **Overstated**: true only for sklearn 1.9.x users; sktime allows sklearn>=0.24 | r1_sklearn_skrub.md:34, 254 |
| engine_lazy | "polars validation 29–75x cheaper than the stock check (design_hub)" | **Imprecise**: design_hub gives 29–71x vs the bare check (S0) and 34–133x vs the metadata check (S); "75" is one S/G cell | design_hub.md §9 table |
| engine_lazy | "DN-B4: lookback pushdown slower than collect-all on 1M-row polars (0.23–0.57x)" | design_narwhals reports 0.26–0.64x (polars/duckdb), with 0.23 in one column. The ranges disagree slightly; the conclusion holds | design_narwhals.md:214, 300 |
| engine_lazy / adapter | g3 C5 "106–4,560x" vs others' "105–4,560x" | Both appear in g3 (header 106, lazy row 105). Not an error | g3_stack.md:17, 119 |
| minimal | "382–1895x vs the loop (x7 verifier)" | The number is in the verifier but marked **PARTIALLY CONFIRMED**, and the 10k baseline is extrapolated (wrap_critic §2). Should carry that label | x7_native_engine_verify0.md:23 |
| tiered | "cutoff.freq always populated is a bug fix (g2: stock inconsistent 81/315)" | g2 reports g2 *differs from stock*, not that stock is inconsistent. This is a behaviour change and should be listed as one | g2 caveats |
| all | g1 7710/7710 tests, 159/160 eligible (123/124 vectorized), ChronosForecaster fallback | Correct | g1_trusted_path.md:17-18, 196, 212-214 |
| all | final_bench T6 875x / 25x memory; B5b 332x; duckdb 360x | Correct (B2 0.0059 s/30.9 MB vs 5.15 s/786 MB) | final_bench.md:41, 237-243 |
| minimal/hub | polars prepare 2.62 s→0.019 s (known mtype) / 2.84 s→11 ms (hub) at 10k | Both correct; they are different rows of T3 | final_bench.md:134-143 |
| minimal | §9.1 nested entry +1.32–1.54x, Naive 2.16–2.48x, equality 4/4 | Correct per results/design_minimal_nested.log | log |
| x4-derived (adapter P0) | 1443→16 ms prepare | Correct, but the x4 verifier says the gain comes from skipping a redundant check, not from the adapter | x4_proto_adapter_verify0.md:18, 185 |
| tiered | top 10 of 96 implementations cover 28% | Correct (10→35 classes, 0.282) | results/design_tiered_census.json |

None of the six designs has a factual error that would change a decision. Two claims, the adapter's blast-radius
claim and the narwhals transitive-dependency claim, overstate exactly the maintainer-relevant facts my lens weighs,
so I counted them against those designs.

## 4. Ideas worth grafting (maintainer-safe composite)

1. **Trusted per-slice core as validate-once plus one shared `_core`, never copied (all designs, from g1).** CI runs
   the whole forecasting suite in both modes. Add a `check_estimator` test that asserts trusted output equals stock
   output, so third-party estimators get coverage automatically. Opt-out by class tag `capability:trusted_vectorize`
   and by config.
2. **Public `register_mtype` plus an entry-point group plus versioned caches (tiered).** This is a prerequisite for any
   third-party inner type and removes the g2 silent-unregister hazard.
3. **Conformance suite as a CI gate per backend (adapter, narwhals):** the g5 240-cell matrix plus the M4 adversarial
   fixtures, comparing labels as well as values. Expose it through `check_estimator`/`datatypes` test parametrisation
   so new backends and mtypes cannot regress silently.
4. **Trusted nested-transformer entry (minimal §9.1, +1.32–1.54x measured).** Implement it as an explicit private
   argument or skbase-level context, not a module ContextVar, so it behaves predictably under joblib workers. Apply it
   to `BaseTransformer` as well, for consistency across scitypes.
5. **Keep the `forecasters_` DataFrame and the `forecasters.loc[...]` keys for the per-instance path (minimal,
   engine_lazy).** Make batched-kernel state additive (`params_`/`fitted_state_`) instead of deprecating the grid
   for everyone (against adapter, hub, tiered and the narwhals key deprecation).
6. **`output:proba_format` flips only for non-pandas input; the pandas MultiIndex stays permanently (g4, all).** Also
   fix or fold `BaseProbaForecaster` (`_yvec` is never set).
7. **`backend_of` registry replacing `hasattr(obj,"X"/"index")`, plus a top-level (key, time) X/y anti-join for every
   scitype (narwhals, minimal, adapter; g5 rule 4).**
8. **Guards on ingress that cannot be turned off, with measured cost (hub §9), and an order-robust kernel contract
   (engine DEL-3).** Prefer an order-robust kernel to a "check sortedness, then positional" kernel.
9. **`lookback` tag that is per method, defaults to None, and is declared rather than inferred (engine_lazy DEL-1).**
   Apply it only to lazy spokes (minimal).
10. **Panel-native kernels through the existing `pd-multiindex` inner tag first (minimal).** Add a numpy grouped
    mtype (g2) only once a public registry exists and a guarded pandas kernel has been shown too slow.
11. **Dynamic composite inner-mtype tags (adapter, tiered).** These are worth prototyping, but only after the above,
    and with a census over the 61 composites.

## 5. Ranking rationale

Under this lens the best design is the one that delivers the measured container-independent levers (g1 and g3
C1–C3, 1.55–3.03x) while adding the fewest contracts that third-party authors and `check_estimator` must learn.

- **design_minimal** wins because it requires 0 estimator changes, adds no dependency and no new mtype, and its only
  deprecation affects non-pandas proba.
- **design_tiered** ranks next for its extensibility API and use of the native tag idiom, but its global flips and its
  plan to delete implementations cost it.
- **design_adapter** is sound but warns pandas users about `forecasters_`.
- **design_hub** alters the substrate that all scitypes share.
- **design_narwhals** and **design_engine_lazy** add a core dependency or a parallel estimator API whose test and
  maintenance surface grows multiplicatively for a small number of kernel forecasters.
