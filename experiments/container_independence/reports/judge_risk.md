# judge_risk: the six designs under a risk and incrementality lens

The lens has four questions:
1. Can the design ship as small, reviewable PRs, each of which is worth having on its own and can be reverted on its own?
2. How much correctness risk does it take on? This covers time semantics, index and label fidelity, and the silent bug
   classes already found in a1, g5, x4, x6 and x7.
3. How much does it depend on third-party moving targets, such as the narwhals stable API, polars versions and pandas 3?
4. Are its effort estimates realistic?

**I ran no new benchmarks.** Every number here is quoted from a report or a results file. I checked citations by
grepping the reports and the `results/design_*.json|log` outputs (commands in §5). PR sizes in §3 are my own
estimates from prototype line counts (`wc -l protos/*.py`), and are labelled *estimate*.

## 1. Scores

| rank | design | score | verdict under this lens |
|---|---|---|---|
| 1 | design_minimal | **8.5** | Every phase is a short list of independent PRs. The first three phases change no public type, add no dependency and require no estimator change. The biggest lever (the trusted core) is reached in PR ~8, not after a substrate rewrite. Its risk is two code paths per method and an implicit ContextVar. Both are bounded by the g1 dual-mode test gate. |
| 2 | design_adapter | **6.5** | It has the best risk controls of the six: a conformance suite as a CI gate, an explicit `time_extreme(min\|max)` op that removes the reverse_order bug class, and a bit-identical pandas adapter (x4 78/78 + 28/28). Against it: its P1 (four registries plus moving VectorizedDF into an InnerFamily) comes before the lever, and P1 alone is worth ≈1.0x. It also puts a FutureWarning on `forecasters_` for pandas users. |
| 3 | design_tiered | **6.0** | Its port PRs (one forecaster each, gated by an equality harness against its own T1 run) are the most incremental unit any design offers. Its public `register_mtype` with versioned caches removes a real silent-failure mode (g2). Against it: T1, the default lane, is built on a GroupedArray plus trusted pandas views (P1 substrate), so the main lever waits on the riskiest code. "Delete the old implementation after the port" removes the revert path. The global `inner_dispatch` flip changes routing for every estimator at once. |
| 4 | design_engine_lazy | **5.0** | Its ordering is good: P0 fixes, then the shared trusted lane in P1, before any engine work. It is also the most rigorous about hazards: lookback is declared and never inferred, with default None (DEL-1), and kernels must be order-robust (DEL-3). Everything after P1 is high-risk and low-reach: three engine families × 6–8 kernels × 4 methods, new lazy mtypes, and dask delegation that gives no single-box gain (DEL-4). |
| 5 | design_hub | **5.0** | Its guards are measured and cannot be disabled, which is good. But its P1 replaces VectorizedDF internals with a float64 hub that loses NaN/null and int64>2^53 fidelity (x6 verifier, 25–26/33). The fast pandas view depends on private `_simple_new`, where g3 already hit 2 pandas-3 bugs. VectorizedDF is shared with transformers, classifiers and metrics. Its own risk rating for P2 is "medium-high" and for P3 "high". |
| 6 | design_narwhals | **4.5** | P0/P1 (the trusted lane) are as safe as in the other designs. But the design depends on a pinned third-party API with known silent bugs in the version tested: pyarrow `over()` mis-assigns rows at ≥~500 groups, when/then inside agg is wrong on pandas, and duckdb total_seconds drops days. It also needs two PanelOps implementations that must stay equivalent (G10, unmeasured). It rates its own kernel phase "high per kernel". One fidelity citation is overstated (§4). |

All six agree on a safe core: P0 bug fixes, the g1 trusted core with class-tag and config opt-out, an fh memo per
unique cutoff, cutoff and fh stay pandas, and pandas input keeps the proba MultiIndex. **The scores therefore mostly
reflect (a) whether the design reaches that core before or after its own new substrate, and (b) how much new
silent-divergence surface it adds on top.**

## 2. Per-design assessment

### design_minimal: 8.5
- **Incrementality.** P0 is about 10 independent bug-fix PRs. Each one is worth having alone, reverts alone and needs
  no architecture decision (wrap_critic §5). P1 is tiny: `from_type` pass-through and a max over codes for the outer
  cutoff. P2 is split into separate PRs for the trusted core, the fh memo, the trusted nested transformer entry and
  ProbaSpec. The g1 class-tag/config opt-out makes each of these revertible at run time as well as in git. P3 (spokes
  at the edges) and P4 (pd-multiindex kernels) are opt-in.
- **Correctness risk.** Lowest of the six. Time logic stays pandas. It is computed once per panel instead of per slice,
  but it is not re-implemented in a numpy Step, so it avoids the a3 Step failures (SME, tz W, BMS/BME, int step 5).
  pandas input keeps stock's 22/22 g5 cells. Kernels in P4 run on the existing pd-multiindex tag behind guards.
  Residual risks:
  - The g1 core copies `_check_fh`, `update_data` and the default `_update` refit. The design says this logic must be
    shared, not copied.
  - g3 found 2 pandas-3 bugs in trusted time code.
  - The TRUSTED ContextVar does not cross loky process boundaries. That fails safe (slow path), but it is implicit
    state.
- **Third-party exposure.** None added.
- **Effort.** Realistic. The g1 prototype is 711 LOC, the nested-transformer proto is 141 LOC and the boundary is
  ~300 LOC (expected).
- **Gaps.**
  - There is no public `register_mtype`, so the g2 cache_clear hazard stays for third-party mtypes.
  - §9.1 measured only Lag inside Naive, at n ≤ 1000 with 2 reps.
  - It forgoes the hub's 1.4–2.2x RSS gain, which it discloses.

### design_adapter: 6.5
- **Incrementality.** P0 matches minimal. P1 is one large refactor: FrameAdapter, InnerFamily, FrameSchema/TimeAxis,
  OutputSpec, public registries and the conformance suite. By its own numbers it gives pandas 0.89–1.06x and e2e
  ≤~1.2x. Only after P1 lands does P2 deliver the 1.55–3.03x.
  - Nothing in g1 or g3 requires P1 first. g1 ran on stock VectorizedDF. The design should run P2-core before P1.
- **Correctness controls (best of the six).**
  - The conformance suite is a CI gate: g5 matrix plus M4, with a target of 0 SILENT cells.
  - `time_extreme(how='min'|'max')` turns the x4 reverse_order bug into an API-level requirement.
  - `time_contains` anti-join for every scitype restores the Panel X/y check that x4 lost (g5 A1).
  - "No adapter op inside the per-instance loop" is enforced by construction.
- **Residual risk.**
  - The x4 prototype already had 3 silent per-backend bugs (x4 verifier: reverse_order, the unnamed level read as
    `"1"`, `key__col` naming), plus g5 A1/A2.
  - Every new adapter re-opens that surface: 67–100 LOC per adapter plus ~370 LOC per forecaster-consumable mtype
    family (x4 §6).
- **Contract risk.** `forecasters_` becomes an EstimatorGrid behind a FutureWarning shim for all users, including
  pandas users, while the speed value of the grid is ≲1% (a1 §6). The census behind "small surface" is narrower than
  stated (§4).
- **Third-party exposure.** Low. narwhals appears only as an optional "hybrid" experiment.

### design_tiered: 6.0
- **Incrementality.** Strongest at the leaf level and weakest at the start.
  - Each T2/T3 port is one forecaster per PR, behind its tag, gated by an equality harness. That is the ideal
    reviewable unit.
  - But T1, the default lane every forecaster uses, is defined on the GroupedArray and trusted pandas views. So P1
    (GroupedArray, spokes for pandas codes, polars and pyarrow, `np_*`/`pa_*` mtypes, FittedGrid) must land before
    P2 can turn on the lever. P1 is worth ~1.0–1.14x in time.
- **Revertibility.** "After that, the old pandas implementation is deleted" gives up the cheapest rollback. Keep the
  old implementation for at least one release, selectable by tag or config.
  - `inner_dispatch` "legacy"→"auto" at N+2 is a global routing switch for all 160 estimators. It cannot be reverted
    per estimator.
- **Correctness risk.**
  - Step is inexact for some aliases (a3 verifier); the stock fallback is required and must be tested per alias.
  - `cutoff.freq` "always populated from Step" is a pandas-visible behaviour change (§4).
  - Composite tag derivation is unprototyped. g5 inv 8 shows composites silently stay on pandas today.
- **Third-party exposure.** Low.
- **Extensibility.** Public `register_mtype` with versioned caches is the right fix and should be grafted everywhere.
- **Effort honesty.** Good. It cites r1 (19/210 estimators in 3.5 years) and its own census: top 10 of 96
  implementations cover 28% of classes, and top 50 cover 63% (`results/design_tiered_census.json`).

### design_engine_lazy: 5.0
- **Incrementality.** P0 and P1 (the shared trusted lane, hub-style positional iteration) come first and stand alone.
  This ordering is good.
  - P2 adds a SourceAdapter family plus new lazy Panel/Hier mtypes for six backends in one phase.
  - P3 multiplies the work: per-engine kernels in polars, duckdb and numpy for 6–8 forecasters × fit, predict,
    interval and update.
- **Correctness discipline (best in class on two hazards).**
  - DEL-1: lookback is declared by construction, per method, with default None. 3/26 "finite" lookbacks were data
    coincidences.
  - DEL-3: kernels are order-robust (1.11–1.38x of positional), which closes the x7/r4 shuffled-rows silent bug more
    cheaply than a sortedness check (3.7–5.3x the kernel).
- **Risk/value mismatch.** The phases after P1 add the most new surface for the least reach.
  - Lazy input gives ±15% for unchanged forecasters (g3).
  - Dask delegation is 0.70–1.31x serial on one box (DEL-4).
  - Cross-engine fidelity hazards are listed by the design itself: dask key dtype changes, duckdb connection binding,
    and C5 losing freq.
- **Third-party exposure.** High: duckdb SQL, polars lazy and dask semantics, each with its own versioning. It avoids
  narwhals on purpose.

### design_hub: 5.0
- **Incrementality.** P1 ("hub core + spokes behind VectorizedDF") is a rewrite of a module that every scitype uses,
  with no time gain (≈1.0x). Its value is RSS, 1.4–2.2x lower on polars input (g3 C4). P2 is rated
  "medium-high" and P3 "high" by the design itself.
- **Correctness risk (highest data-model risk of the six).**
  - The float64 hub loses NaN versus null and int64>2^53 precision, flattens MultiIndex columns, and fails a
    timedelta index on pd3. The x6 verifier records 25–26/33 adversarial passes.
  - The trusted pandas view depends on private `_simple_new`. g3 hit 2 pandas-3 bugs there (Day vs 24h; datetime
    unit).
  - g5 H1: the x6 round trip dropped the time-level freq and broke `update_predict` for pandas input. This is a
    regression for pandas users, the group every design promises to protect.
- **In its favour.**
  - The guards are measured: 1.4–3.5x of unguarded ingress, and they catch NaN key, mixed freq and duplicate time,
    which stock accepts.
  - The gate is explicit: x6 29/29, plus 33 adversarial fixtures, plus the g5 240-cell matrix on pd2 and pd3.
- **Third-party exposure.** Medium: hand-written spokes per library, plus private pandas API.

### design_narwhals: 4.5
- **Incrementality.** P0/P1 are as in the other designs, and the trusted lane does not depend on narwhals.
- **Dependency risk.** Everything narwhals-specific adds a pinned dependency on a fast-moving API.
  - Known silent bugs in 2.27.1: pyarrow `over` at ≥~500 groups, when/then inside agg on pandas, duckdb
    total_seconds (x5, x7).
  - Missing ops in the stable namespaces: `last(order_by)`; dask lacks `over(order_by)` (x1 verifier, DN-API).
  - The C1 answer differs between stable.v1 and v2 (wrap_critic K7).
- **Correctness risk.**
  - Two PanelOps implementations must stay equivalent; the cost is unmeasured (G10).
  - Per-kernel backend allow-lists are needed, and conformance must run at ≥1000 groups, because the pyarrow bug is
    invisible at n=20.
  - Lazy input has no row order, so output order differs from stock's first-appearance order.
- **Mitigations.** These are sound: the explicit `materialize(purpose)` policy (C1), the long-only frame with
  `maybe_reset_index` banned (C2), and the stable.v2 pin. But they add process (pin audits, allow-lists) on top of
  the code.
- **Effort.** The design is honest that narwhals is not the speed lever for the 160 per-instance forecasters.

## 3. A shippable PR sequence (graft across designs)

PR sizes are *estimates*, scaled from prototype line counts. Each row is independently valuable and revertible.
Rows 1–9 need no architecture decision.

| # | PR | source | value (cited) | revert | size (est.) |
|---|---|---|---|---|---|
| 1 | Vectorized polars monotonic check **with null guard + fallback** | all P0 | x9: ~10x convert_to e2e; g3 C1 part of 1.05–1.34x | git | S |
| 2 | Pass known mtype / `convert(from_type)` instead of re-inferring | all P0 | x4: prepare 1443→16 ms (verifier 1962→11.9 ms) at 10000x50 | git | S |
| 3 | `_HashIndex.__eq__`/`__hash__` → fh `lru_cache` hits | all P0 | g3 C3 1.05–1.53x (U=1 best case) | git | XS |
| 4–9 | g5 and a1 stock bugs, one PR each: polars update with 1–2 rows, Panel predict_residuals fh, y→"0" rename, polars X/y containment, `hasattr(X,'X')`, pd3 caller mutation, BaseProbaForecaster `_yvec`, predict_var name and cov | all P0 | correctness | git | XS–S each |
| 10 | Shared validate-once `_core` for each public method (refactor only, no behaviour change) | minimal/tiered | none alone; prerequisite for avoiding drift | git | M |
| 11 | Trusted per-slice entries on stock VectorizedDF, behind `capability:trusted_vectorize` + config; **CI runs the forecasting suite in both modes** | g1 | fit+predict 1.15–1.88x; 7710/7710 | config/tag | M (g1 proto 711 LOC) |
| 12 | Trusted nested BaseTransformer entry (**explicit argument preferred over a ContextVar**) | minimal §9.1 | +1.32–1.54x on Naive (design_minimal_nested.log) | config | S–M |
| 13 | ProbaSpec + long→MI assembly once per panel (pandas output unchanged) | g4 | 7.9 s `_get_columns` + 205 ms concat → 14–85 ms at 10k | git | M (g4 proto 369 LOC) |
| 14 | Public `register_mtype` + versioned caches + entry point | tiered | removes g2's silent unregister | git | S–M |
| 15 | Conformance suite (g5 matrix + M4 adversarial) as a required CI job | adapter | gate for 16+ | n/a | M |
| 16 | `backend_of` registry + output-follows-input egress + `time_extreme(min\|max)` + `time_contains` anti-join | adapter/minimal | polars cells 9/24 → 20+/24 (*expected*) | git | M |
| 17 | `output:proba_format` config (default pandas, FutureWarning only for non-pandas input) | g4/all | contract consistency | config | S |
| 18 | `capability:lookback` (declared, per method, default None) + Selection pushdown for lazy spokes only | engine/minimal | T6 B5b 190–330x on selective reads | tag default | M |
| 19+ | One-forecaster-per-PR kernel ports behind the inner-mtype tag, with an equality harness against T1; **old path kept ≥1 release**; order-robust contract | tiered/engine | g2 498–2,735x; g3 C5 105–4,560x | tag | S each (g2: 74–116 LOC) |
| later | Hub / positional iteration for RSS (only if a fresh-process memray study at ≥50k series shows it matters); narwhals only inside lazy spokes for select/collect | hub/adapter | g3 C4 1.4–2.2x RSS | config | L |

Effort realism.
- No design gives person-time. LOC claims are consistent with the prototypes:
  - g1 711, g3 659, g2 621+368, g4 369, x4 979, x6 1,237, x5 541+299.
  - Production code typically needs 2–3x for tests and docs (*estimate*).
- Under that lens, minimal's and adapter's framework totals (~1–2k LOC) are believable.
- Hub, narwhals and engine add multi-k-LOC substrates plus per-backend or per-engine matrices that no design sized
  (G10).

## 4. Citation checks and factual errors

Checked against the source reports. Confirmed citations are not listed one by one. The checked set includes:
- g1: 7710/7710, 159/160, 123/124.
- g3: C3 1.55–3.03, C4 1.59–3.31, RSS 161/121/54 MB.
- x4: 78/78 + 28/28, 1443→16 and 1962→11.9 ms.
- x7 verifier: 382–1895x, 1.07–1.33x.
- final_bench: T3 2.84 s→11 ms, T6 875x/332x.
- design_minimal nested log: 2.16–2.48x cumulative, 1.32–1.54x marginal, predict 1.60–1.93x, outputs equal 4/4,
  0 fallbacks.
- design_hub guard table, design_tiered census and the adapter census JSON.

| design | issue | severity | evidence |
|---|---|---|---|
| narwhals | Cites "198/198 fidelity checks on pd2 and pd3 (x2)" as settled. The x2 verifier rates it **partially confirmed**: it holds only for the tested input class and fails on mixed-dtype frames and per-series freq. | medium (it underpins the C2 mitigation) | x2_nw_index_fidelity_verify0.md:88 |
| narwhals | Says narwhals is "already a transitive dependency" via sklearn ≥1.9. r1 says sktime's own bounds (`scikit-learn<1.10`) do **not** guarantee sklearn 1.9, so it is not guaranteed. | low | r1_sklearn_skrub.md:34 |
| adapter | Thesis and P2 cite g3 **C4** (1.59–3.31x, RSS 1.4–2.2x lower), which includes hub ingress and positional iteration. The adapter makes positional iteration optional. Without it, the supported figure is **C3, 1.55–3.03x** with no RSS gain. | low–medium | g3_stack.md:15–16 |
| adapter | "Only 2 non-test pandas-attribute reads of forecasters_" understates the migration surface. The same census JSON finds **89 non-test `forecasters_` sites in 19 files** (hierarchy_ensemble 15, fhplex 11, _base 9, …). Many are composites' own attribute, but they were not triaged, and the design flags its census as regex-only. | medium for the N+2 flip risk | results/design_adapter_contract_census.json |
| tiered | Describes stock `cutoff.freq` as "inconsistent in 81 of 315 cases" and plans to populate it "always". g2 says stock's MultiIndex `get_cutoff` systematically returns None where g2 carries freq, and values are equal. So this is a deliberate pandas-visible behaviour change, which conflicts with "pandas users see the same types". It is probably beneficial (g5 H1 shows freq loss breaks update_predict) but needs a changelog entry and tests. | low | g2 caveats |
| engine_lazy | "polars validation 29–75x cheaper (design_hub guard bench)". design_hub reports 29–71x against the bare check and 34–133x against the metadata check. | trivial | design_hub.md §9 table |
| engine_lazy vs narwhals | DN-B4 is quoted as 0.23–0.57x (engine) and 0.26–0.64x (narwhals). They are different columns of the same table. Both show "pushdown slower than collect-all", so the conclusion holds. | trivial | design_narwhals.md:299–302 |
| hub | "Guarded ingress 10–25x more than stock's bare check on pandas". The table gives 9.7–22.7x. | trivial | design_hub.md §9 control table |
| minimal | "16–18x less RSS" for T6 B5b vs B1. Contiguous ids give 17.9x and random ids 15.6x. | trivial | final_bench T6 |
| minimal | 382–1895x is from x7.1, which the verifier rates **partially confirmed**, and the 10k sktime baseline is extrapolated (wrap_critic). This is not disclosed next to the number. | low | x7_native_engine_verify0.md:23 |

No design has a headline number that I could not trace to a source.

## 5. Commands run (all read-only)
- `grep -n -E "1\.59|...|0\.93" reports/g3_stack.md`: C3/C4/C5 ranges and RSS rows.
- `grep -rn "123/124|123 of 124" reports/*.md`, then `sed -n 200,220p reports/g1_trusted_path.md`: 124 vectorized, 123 eligible.
- `grep -n "78/78|28/28|1443|1962|11.9" reports/x4_proto_adapter*.md`
- `grep -n "382|1895|1.07|1.33" reports/x7_native_engine_verify0.md`
- `sed -n 310,343p reports/design_hub.md`: guard table.
- `tail results/design_minimal_nested.log`: fp_cum 2.16/2.36/2.26/2.48, fp_marginal at C4 1.32/1.54/1.33/1.39, pred_cum 1.76/1.89/1.60/1.93, equal 4/4, fallbacks 0.
- Python one-liners reading `results/design_adapter_contract_census.json` and `results/design_tiered_census.json`.
- `grep -n 198 reports/x2_nw_index_fidelity*.md`, `grep -n narwhals reports/r1_sklearn_skrub*.md`.
- `wc -l protos/*.py` for the effort scaling.
