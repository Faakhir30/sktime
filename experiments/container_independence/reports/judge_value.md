# judge_value: realized performance value of the 6 design proposals

**Lens.** The question for each design is whether it actually delivers the value of polars, lazy, dask and duckdb (speed, memory, pushdown) to real users at realistic sizes, judged on measured evidence. That evidence is final_bench, the g3 waterfall, x7/x8/x9 and their verifiers, g1/g2/g4/g5, and each design's own §9 measurements. A design is penalized when its headline gain is capped by costs it leaves unchanged, or when it cites a number its source does not support.

**Method.** I checked every headline number against the cited report or result JSON. For this I read final_bench.md, g3_stack.md, the x7/x8/x9/x4/x5/x6/r4 verifiers, and each design's §9 measurement section with its result file (`results/design_*.json`). I ran no new benchmarks. All numbers below are cited.

## 0. The value ceiling every design hits (measured facts)

| lever | measured gain | applies to | source |
|---|---|---|---|
| container-only change, per-instance calls unchanged | ≤1.05x e2e | all | x8, x8 verifier |
| C1 known mtype + vectorized monotonic check | 1.05–1.34x | all | g3 |
| C2 trusted per-slice path | 1.17–1.73x marginal | 159/160 | g3, g1 |
| C3 fh memo per unique cutoff | 1.05–1.53x marginal, **U=1 best case only** (ragged ends unmeasured) | all | g3 §3.3, caveat 2 |
| C4 hub/positional iteration | **0.93–1.14x time**; peak RSS 1.4–2.2x lower than C3 (5000x100 Poly polars 121→54 MB) | all | g3 §3.5 |
| trusted nested Lag entry | +1.32–1.54x on C3; Naive 2.16–2.48x vs stock; predict 1.60–1.93x | Naive-like (forecasters with nested transformers) | design_minimal §9.1 (verified against results/design_minimal_nested.json) |
| panel-native kernel | 105–4,560x (g3 C5); g2 498–2,735x | **6–8 strict closed-form classes** | x7 verifier |
| engine vs pandas groupby, conversion charged | **1.07–1.33x at 10k, 0.91–1.07x at 1k** | pandas-in users | x7 verifier x7.2 |
| numpy GroupedArray vs pandas groupby | 7–11x, keeps the gain with pandas in and out | kernels | x7 verifier |
| selective lazy read (instances × last 50) | 875x / 25x less memory (B2), 332x (B5b) vs eager sktime | selective workloads | final_bench T6 |
| true pushdown vs filter-then-convert, eager | **only 1.9x time / 2.8x RSS** | selective workloads | x9 verifier |
| laziness, full-history workloads | ±15% at C0–C4; 1.1–2.6x **slower** at C5 | all | g3 |
| lookback-only pushdown | 0.23–0.64x (slower) to polars at 1M rows; finite lookback is real in only 26/70 runnable forecasters (most of them Naive and its wrappers), with 3 data-coincidence false positives; for intervals 1/39 | — | DN-B4, DEL-1 |
| dask partition delegation, single 4-core box | 0.70–1.31x serial | — | DEL-4 |

Two consequences shape every score:

1. For the ~150 forecasters that are not kernels, the realistic gain from any design is **~1.1–2.5x**. That is the same C1–C3 plus the nested entry, whatever the container. It is about **1.0x for statsmodels-bound forecasters** (g1 ETS ≈1.0) and for predict_interval (g1 ≈1.0). The 1.6–3.3x headline comes from g3 C4 and was measured only on Naive and PolynomialTrend fit+predict, with U=1 aligned cutoffs. Several designs state it as the gain "for unchanged forecasters" or "for all 160". That is an overgeneralization.
2. The container choice decides only three things: **memory** (hub/positional iteration), **kernel speed for 6–8 classes** (numpy > polars > pandas groupby > narwhals), and **selective-read pushdown** (an edge op every design can add).

## 1. Scores

| design | score /10 | one-line verdict |
|---|---|---|
| design_tiered | **7.5** | Captures every measured lever: T1 trusted, numpy T3 kernels (the fastest tier, which also wins for pandas input), GroupedArray memory, selection at the edge. It is honest about the long porting tail. It has one cited-number error (dask). |
| design_minimal | **7.5** | Adds the only new measured universal lever (nested trusted entry, 2.16–2.48x Naive). It captures selective-read value at the edge, and its numbers are clean. It deliberately leaves memory (1.4–2.2x RSS, +451 vs +30 MB polars ingress at 50k) and 7–11x kernel speed on the table. |
| design_hub | **7.0** | Honest: "substrate, not a speed lever". It is the only design built around the memory win, and it measured guard cost. It has one cited-number error (dask), and fidelity risk on the realized path (x6 adversarial 25–26/33). |
| design_adapter | **6.5** | Correct: the adapter adds nothing to speed, and gains ride on g1/g3/g2. It gets memory only through *optional* positional iteration, and it schedules the nested-transformer lever late (2b). One typo. |
| design_engine_lazy | **5.5** | Its own experiments (DEL-1..4) are the best evidence in the set on lazy value, and they mostly refute its thesis. Engine kernels serve 6–8 classes, lookback is rare and fragile, dask gives no single-box gain, and polars lookback gives no memory beyond projection. It also over-attributes x7.4's 1,180x to the plan. |
| design_narwhals | **5.0** | narwhals is neither the speed nor the memory lever. Its kernel tier is 3.6–4.2x slower than a plain pandas groupby even when routed to polars (DN-R), and 7–11x over hand-written code. Its own lookback pushdown is measured as slower (DN-B4). The universal gains are the shared trusted lane. |

## 2. Per-design assessment

### design_tiered (7.5)
- **Realized value.** T1 delivers C1–C3 for 159/160 classes. T3 `np_grouped` is the fastest kernel tier, and it works directly on pandas input with no conversion penalty. The x7 verifier shows numpy keeps 7–11x over pandas groupby while polars engines keep only 1.07–1.33x once conversion is charged. g2 measured 498–2,735x on pandas and 498–1,486x on pyarrow. GroupedArray ingress gets the memory win, and Selection pushdown captures T6.
- **Honesty.** Its own census is checked against `results/design_tiered_census.json`: the top 10 of 96 implementations cover 28.2% of vectorized classes, and 50 cover 62.9%. Its P4 gives T2 ports ≈4–5x over T1, labelled "indicative". Its P5 says "Full-panel workloads: none (g3)".
- **Errors and overclaims.**
  - (a) "dask 1.2–1.5x (r4 verifier)" (P5, and the dask row of §2.3) is a **misreading**. r4 verifier V1 measures *Fugue's per-group overhead*, which is 1.2–1.5x *slower* than an equivalent pandas loop. It is not a dask delegation speedup. The only measured delegation number is DEL-4: 0.70–1.31x serial.
  - (b) The thesis "T1 … gives 1.6–3.3x (g3 C4)" attributes the C1+C3+C4 stack to T1. T1 alone (C2) is 1.17–1.73x marginal, and the figure was measured only for Naive and Poly.
  - (c) The cutoff.freq "bug fix" (81/315) is a behaviour change, not a measured value.
- **Capped by unchanged costs.** It acknowledges that the statsmodels family stays ≈1.0x. The nested-transformer lever is in P2b, which is correct.

### design_minimal (7.5)
- **Realized value.** It measured a new lever, and the numbers check out against `results/design_minimal_nested.json`. The levels are 300x100 and 1000x100, pandas and polars, with cumulative fit+predict 2.16 / 2.36 / 2.26 / 2.48x, outputs equal in 4/4 rows, and 0 fallbacks. Its value is that it lifts predict, which g1 left at ≈1.0–1.45x, to 1.60–1.93x. It is the largest measured gain for an unchanged forecaster in the whole investigation.
  - Edge Selection gives the T6 value (190–330x, 16–18x less RSS; checked: 786/43.9≈18, 784/50.3≈16).
  - pandas-groupby kernels (382–1,895x vs the loop) keep the 10²–10³x lever without a new container.
  - The Arrow measurement (§9.2: 1.07x and 1.19x slower; Arrow lost in Poly output) correctly kills an option.
- **What it leaves on the table** (it says so itself):
  - memory: C4 1.4–2.2x RSS; polars ingress +451 MB vs hub +30 MB at 50000x100 (T5b);
  - kernel speed: numpy is 7–11x faster than pandas groupby, and polars-native data is 2.1–3.0x faster (x8 verifier);
  - for polars users, every call still pays polars→pandas MI ingress: 0.11 s at 10k without the check (T3), against a kernel run of ~0.13 s.
  For kernel forecasters on polars data, conversion therefore dominates.
- **Numbers check:** "polars prepare 2.62 s → 0.019 s at 10k known mtype" matches T3 (2.62 → 0.0185). "x7 verifier 382–1895x at 10k" is cited correctly, but the verifier only *partially confirmed* it and did not re-run 10k. No factual errors found.
- **Risk to realized value.** Unguarded pandas kernels are silently wrong on shuffled rows (x7 verifier). The design notes that guard cost is unmeasured (G13). DEL-3 bounds that cost.

### design_hub (7.0)
- **Realized value.** Same speed levers as tiered. It is the only design whose core component *is* the memory win: C4 roughly halves RSS, and polars ingress drops 2.84 s → 11 ms and +451 → +30 MB (T3/T5b, checked). Its §9 guard bench is new and checks out against `design_hub_guard_bench*.json`. The guarded hub is 34–496x cheaper than stock with metadata, and 29–71x cheaper on polars without it. On pandas, though, it is **10–25x more expensive** than the bare stock check, an honest negative.
- **Errors.**
  - (a) P5 "Dask partition delegate: 1.2–1.5x like-for-like (r4 verifier)" is the same misreading as tiered. That number is Fugue overhead (slower), not a delegation gain.
  - (b) "slice view 54–236x cheaper than items" (T1) is true but does not reach e2e. g1's ablation shows slicing once vs twice is 1.18 vs 1.19, and g3 C4 is 0.93–1.14x. It is evidence of substrate cost, not of user value, and the design mostly frames it that way.
- **Risks to realized value.** The float64 hub loses NaN/null and int64>2^53 (x6 verifier 25–26/33). The trusted pandas view needs private `_simple_new`, and pandas 3 already broke it twice (g3). A memory win that ships with silent fidelity loss is not realized value until the guard and adversarial gate passes.

### design_adapter (6.5)
- **Realized value.** The thesis states correctly that the adapter alone is ≤1.05x and that x4 B2/A0 is 0.96–1.17x (verified). It puts g1, the C3 fh memo and g2 on top, so speed matches the others.
  - Memory is only an "optional positional pandas-family iteration", and the polars→pandas whole-frame ingress keeps the +451 MB class cost unless np_grouped is used.
  - The nested-transformer lever is deferred to "2b". After C3 that is the largest remaining measured lever, so the realized gain for Naive-like forecasters arrives late.
  - The lazy plan is sound: select+collect once at the boundary, never lazy per slice. x4 Z showed 3–77x worse for lazy per-slice iteration.
- **Numbers.**
  - "x4 verifier: 1962 to 11.9 ms" is verified (x4 verifier table, L_prepare_cur vs new_known_mtype at 10k).
  - The open-experiments list says "≈ pandas-input gain (98–2,735x)". This is a **typo for 498–2,735x**.
  - "15–70x less memory" (T6) reads loosely: B2 is 25x, B5b 18x, duckdb 70x. That is acceptable.
  - The phrase "Fugue-style 1.2–1.5x only" is ambiguous but does not claim a speedup.

### design_engine_lazy (5.5)
- **Realized value is narrow, and its own data shows it.**
  - DEL-1: finite lookback is mostly Naive and its wrappers. AutoETS, ExponentialSmoothing and UnobservedComponents were coincidences, and intervals are 1/39.
  - DEL-2: polars lazy+lookback gives 749–761 MB vs projection's ~803 MB, so no meaningful memory gain beyond projection. duckdb in-SQL gives 12–24x RSS only on series-major files. On time-major files it is 0.71–1.05x time.
  - DEL-4: dask delegation on one box is 0.70–1.31x serial. "2.9–3.7x faster than stock per-instance dask" uses only the best P; at P=4 it is ≈2.1–2.3x. Stock dask per-instance is itself a 0.34–0.35x strawman.
  - The kernel group is 6–8 classes in **three** engine families. For pandas users the engine route is worth 1.07–1.33x over pandas groupby (x7 verifier), so the 10²–10³x is the vectorization, not the engine.
- **Over-attribution.** Evidence_for leads with x7.4 "123 ms vs 145 s" as value "reachable only if the engine runs the forecaster". The x7 verifier says that figure "is almost entirely the per-series loop, not laziness". It also says the pandas comparison was cold-inflated about 3x by the pyarrow.parquet import. The design's own "against" section concedes most of this, so the thesis and the evidence disagree.
- **Strengths worth keeping.**
  - DEL-3 is the most actionable kernel-safety result in the set. An order-robust kernel costs 1.11–1.38x of the positional one and is correct on shuffled rows; a separate sortedness check costs 3.7–5.3x the whole kernel.
  - The R2 lane gives every forecaster the shared C1–C4 gains.
  - Column projection always on (1.25–1.68x time, 1.6x RSS) is cheap, real value.
- **Number checks:**
  - "x7.8 570x / 2,006x" matches.
  - "Polars validation 29–75x cheaper" matches design_hub S0/G and S/G.
  - DN-B4 "0.23–0.57x" omits the 1.13x at 10k to pandas. This is minor.

### design_narwhals (5.0)
- **Realized value.** Its own evidence says narwhals removes at most the ≤1.18x non-pandas penalty (T4). It is not the speed lever: the 1.5–3x comes from the shared trusted lane.
  - The kernel tier is weak. x5 verifier: plain polars is 7–11.5x faster than narwhals-on-polars, and plain pandas groupby is 9.8x faster than narwhals-on-pandas. DN-R: even routed to polars, the narwhals kernel is **3.62–4.19x slower than plain `pandas groupby.last`**.
  - So for pandas users, who are most users, a narwhals kernel delivers less than minimal's pandas-groupby kernel and 30–40x less than numpy.
  - Lazy end-to-end kernels exist, but g3 shows laziness pays nothing on full history. DN-B4 shows lookback pushdown is slower to polars (0.23–0.64x), and on dask it is 0.09–0.12x.
  - The memory gain, +6.8–29 vs +27–46 MB ingest (T5), is real but small, and the hub gets more.
- **Overclaim.** The thesis says "the 1.5–3x gain for all 160 forecasters". It was measured only on Naive and Poly; ETS and predict_interval are ≈1.0x (g1).
- **Numbers.**
  - DN-B1 (1.23–1.56 s → 15–16 ms at 10k), DN-B3 (10000/10000 flagged) and DN-R ratios all check against the §8 tables.
  - x5 "36–943x" is the original claim. The verifier gives x304/x88 at 1k and attributes the gain to vectorization, not narwhals. The design does cite the verifier.
- **Useful parts.** Validation and containment as one portable expression (DN-B1, DN-B3), and the C1/C2 handling (explicit materialize, schema recipe), make a good *implementation detail inside a polars/duckdb spoke*. They do not justify making narwhals the architecture.

## 3. Factual errors and overclaims found

| design | claim | finding |
|---|---|---|
| hub, tiered | "dask partition delegate 1.2–1.5x like-for-like (r4 verifier)" listed as expected impact | **Wrong.** r4 verifier V1 is Fugue per-group *overhead*, 1.2–1.5x slower than an equivalent loop. The only delegation measurement is DEL-4: 0.70–1.31x serial on one box. |
| adapter | "expected ≈ pandas-input gain (98–2,735x)" | Typo: g2 reports 498–2,735x. |
| tiered, adapter, hub, narwhals, engine_lazy | "1.6–3.3x for unchanged forecasters" / "1.5–3x for all 160" / "applies to all 160 forecasters" | Overgeneralized. Measured only for Naive and Poly fit+predict with U=1 aligned cutoffs (g3 caveat 2). g1: ETS ≈1.0x, predict_interval ≈1.0x, full workload 1.07–1.72x. |
| tiered | "T1 … gives 1.6–3.3x (g3 C4)" | T1 is the trusted path (C2): 1.17–1.73x marginal. 1.6–3.3x is the full C1–C4 stack. |
| engine_lazy | x7.4 "123 ms vs 145 s" as engine/plan value | x7 verifier: almost entirely the per-series loop, with pandas cold-inflated about 3x. |
| engine_lazy | DEL-4 "2.9–3.7x faster than stock per-instance dask" | Best-P only (P=16). At P=4 the ratio is ≈2.1–2.3x, and the baseline itself is 0.35x of serial. |
| engine_lazy | "DN-B4 … 0.23–0.57x" | Omits the 1.13x (polars to pandas, 10k) and 2.10x (duckdb to pandas, 10k) cells. |
| several | "g3 C5 106–4,560x" vs "105–4,560x" | g3 is inconsistent with itself (its table says 106, its impact section 105; the min row is poly pl_lazy 100x100 = 105). Harmless. |
| several | "peak RSS 1.4–2.2x lower (161 → 54 MB)" | 1.4–2.2x is C4 vs C3. 161→54 MB is C0→C4, which is 3.0x. The pair mixes baselines, but the direction is right. |

No fabricated numbers were found. All design-specific new measurements (DEL-1..4, DN-API/B1–B5/R, design_hub §9, design_minimal §9.1–9.2, both censuses) have matching result files and the stated values match.

## 4. Best ideas to graft

1. **Trusted nested-transformer entry in the main boundary phase, not "2b"/P3** (design_minimal §9.1). This is the largest measured lever for unchanged forecasters after C2. It brings predict from ≈1.0x to 1.60–1.93x. It goes into adapter, hub, tiered and engine_lazy.
2. **numpy `np_grouped` as the kernel tier** (tiered T3 / hub / g2), not narwhals or per-engine kernels. It is 7–11x faster than pandas groupby with pandas in and out, while engines add only 1.07–1.33x once conversion is charged. It is the only kernel tier that pays for pandas users.
3. **Order-robust kernels** (engine_lazy DEL-3): sort inside the aggregation, at 1.11–1.38x, instead of a separate sortedness check at 3.7–5.3x the kernel. This is required for every kernel tier; x7 kernels are silently wrong on shuffled rows.
4. **Edge selection, always on; lookback pushdown off by default** (engine_lazy DEL-1/2 + minimal §2.5). Projection and instance selection deliver the T6 value (190–875x). Lookback is rare, fragile and memory-neutral on polars, so `capability:lookback` defaults to None and is declared only structurally.
5. **Hub/GroupedArray ingress for polars input** (hub, tiered). This is the one memory lever, +30 vs +451 MB at 50k and 1.4–2.2x lower peak RSS. Minimal should graft at least this, so polars users stop paying a pandas MI copy.
6. **Cheap guarded top-level validation and X/y containment as one vectorized pass** (hub §9 guards / narwhals DN-B1, DN-B3). It costs 3.6–30 ms vs 0.12–3.9 s stock on polars. Keep the stock check on pandas, where the guarded hub is 10–25x more expensive.
7. **Census-driven port order** (tiered): the top 10 implementations cover 28% of classes, so port only where the framework share is above 50%. **Long-table proba + one distribution from a parameter table** (g4: 3.2 ms vs 826 ms) belongs in every design's boundary phase.
8. **duckdb in-SQL kernel for cold, series-major parquet** (DEL-2: 12–24x lower RSS). Make it an optional spoke feature, not a core path.

## 5. Measurements still needed before any design's value claim is final

- The C1–C3 stack plus the nested entry with **ragged cutoffs** (U = n/10 and U = n). The fh-memo share of C3 is a U=1 best case.
- The **full workload** (update, predict_interval, X) and **statsmodels or reducer forecasters** under the stack. Today the 1.6–3.3x headline rests on two cheap forecasters.
- The **10k–50k-series e2e** stack with fresh-process memory: minimal vs hub ingress at the size where +451 MB matters.
- **Cold or remote storage** pushdown (all T6/DEL-2 runs were warm and local).
- The **guard cost** for pandas-groupby and numpy kernels (G13), against DEL-3.
