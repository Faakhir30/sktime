# design_final: decoupling sktime forecasters and VectorizedDF from pandas

**Recommended architecture and phased roadmap**

Audience: sktime core developers.

Sources:
- `reports/design_{minimal,tiered,adapter,hub,narwhals,engine_lazy}.md` (six proposals)
- `reports/judge_{maintainer,value,risk}.md` (three judges)
- `reports/design_redteam.md` (red-team review)
- the gap experiments g1–g5 and the investigation digest (`results/investigation_digest.json`)

Labels used in this report:
- **[M]** measured by a script in this investigation. The source report is cited.
- **[E]** expected or unmeasured.
- **[S]** synthetic or extrapolated input.

No number in this report was produced fresh by the synthesizer. Every number is cited from a report that ran its script. The red-team re-runs are recorded in `design_redteam.md` §0.

---

## 0. TL;DR

1. **Base design: design_minimal** (judge scores 8.0 / 7.5 / 8.5, the highest on every lens). Everything below is grafted onto it.
   - pandas stays the inner container for the roughly 150 per-instance forecasters.
   - The work goes into removing **framework re-entry**: validate once, a trusted per-slice lane, fh/cutoff once per unique cutoff, and a trusted nested-transformer entry.
   - Containers convert **exactly once** at the edge, through a small spoke registry that returns the user's container.
2. **The speed is not in the container.**
   - Container-only changes are capped at about 1.05x [M x8].
   - Hub iteration gives 0.93–1.14x in time [M g3 C4].
   - All levers that pay for unchanged forecasters are container-independent [M g3 §3.5].
3. **Expected realized gain for unchanged forecasters: about 1.5–3x on fit+predict** [M g3 C3, cumulative 1.55–3.03x].
   - This is Naive/Poly only, with one shared cutoff (U=1, the best case for the fh memo).
   - ETS/statsmodels-bound forecasters and `predict_interval` stay at about 1.0x [M g1].
   - With the nested-transformer entry, Naive reaches **about 2–3x** [M design_minimal §9.1 and the red-team re-run; ±20–25% noise; prototype mechanism differs].
4. **10²–10³x applies only to about 9 closed-form forecasters that are ported to an opt-in numpy `np_grouped` inner mtype** [M g2, g3 C5].
5. **pandas users see no contract change** for per-instance forecasters. Kernel ports stay opt-in until `forecasters_` parity and a level-freq gate exist. Non-pandas users get output-type changes only through config and deprecation cycles.
6. **narwhals verdicts:**
   - **C1 is TRUE.** It is a constraint handled by explicit `collect(backend=...)`.
   - **C2 is MOSTLY TRUE.** The schema recipe repairs it only for the tested input class.
   - narwhals is **not adopted as the core**.

---

## 1. Why this design: what the evidence settles

| Fact | Evidence |
|---|---|
| Container-only change under unchanged per-instance calls caps at ~1.05x e2e | [M] x8 + verifier |
| Per-slice public-API re-entry (check/convert/fh/cutoff/clone) is 33–63% of vectorized fit | [M] a1, x8 |
| Trusted per-slice path: fit 1.2–5.5x, update 1.3–3.2x, predict 0.9–1.45x, interval ~1.0x; 7710/7710 tests, pass delta 0; 159/160 forecasters eligible | [M] g1 |
| Levers multiply up to C3, then saturate; C4 (hub) is a memory step, not a speed step | [M] g3 |
| Panel-native kernels: 10²–10³x, only for closed-form forecasters (6–9 strict) | [M] g2, g3 C5, x7 verifier |
| Engines beat pandas groupby by only 1.07–1.33x once conversion is charged; numpy segments are 7–11x faster than pandas groupby | [M] x7 verifier |
| Lazy input pays nothing on full-history fit+predict (±15%); lazy is 1.1–2.6x slower at C5 | [M] g3 |
| Proba cost is per-slice re-entry; the MultiIndex layout is reconstructible from a long schema | [M] g4 |
| Public API returns pandas types (forecasters_, cutoff, fh, proba MultiIndex) and 4/4 hierarchical metric configs depend on the grid | [M] a1 |
| Prototypes regress silently on non-pandas paths (x4: lost the disjoint-X check, y renamed to "0", reverse_order cutoff; x6: freq loss broke pandas update_predict) | [M] g5 |

Two conclusions follow:
- Build the **container-independent levers first**, with no new substrate in front of them.
- Add **narrow, gated** non-pandas edges and an opt-in kernel tier.

---

## 2. Recommended architecture

```
 user container ──► Spoke.check (vectorized, null-guarded, guards cannot be turned off)
 (pandas | polars | polars-lazy | duckdb | pyarrow | dask)
        │  lazy: Spoke.select(instances, columns[, lookback if declared]) → collect(backend=explicit)  (exactly once)
        ▼
 Spoke.to_inner(from_type=known)     ── once per public call, never inside a per-instance loop
        │
        ├─ inner = pandas (default, ~150 forecasters)
        │     VectorizedDF (stateless pandas (frame, schema), shared with transformers/metrics)
        │     └─ trusted slice runner: SliceCtx → _fit/_predict/_update/..._trusted
        │           sets the worker-side TRUSTED ContextVar → nested Lag/Detrender skip check/convert
        │     fh/cutoff: FhAbsCache per unique cutoff; outer cutoff = max/min over codes
        │
        ├─ inner = pd-multiindex (29 global forecasters): unchanged, one conversion
        │
        └─ inner = np_grouped (opt-in ports, ~9): values + indptr + KeyTable + TimeAxis/Step
              order-robust batched kernels; any unsupported call falls back per call to the trusted lane
        ▼
 OutputSpec / ProbaSpec (long table built once) ──► Spoke.egress(like=input)
```

### 2.1 Components

1. **Spoke registry** (`sktime/datatypes/_boundary.py`, about 300 LOC [E]).
   - **Ops:** `backend_of`, `check`, `select`, `collect`, `to_inner`, `egress`, `proba_out`, `time_extreme(how=min|max, per_instance)`, `time_contains` ((key, time) anti-join).
   - **Detection** (red-team D8): `backend_of` dispatches on `type(obj).__module__` prefix and exact type, **without importing** soft dependencies. Third-party spokes and mtypes load through an entry-point group **lazily**, on the first registry miss.
   - **Guards that cannot be turned off** (from design_hub §9): NaN or null key, strictly increasing time within each instance, and regular step where required. Measured at 1.4–3.5x of unguarded ingress, and still 29–71x cheaper than the stock polars check [M design_hub §9].
2. **Public `register_mtype` / `register_spoke` with versioned caches** (from tiered). This fixes g2's finding that `cache_clear` silently unregisters externally added mtypes [M g2].
3. **VectorizedDF stays a stateless pandas (frame, schema).** It is shared with BaseTransformer, `_base_panel` and the metrics (15 non-test `vectorize_est` sites).
   - It takes `from_type`.
   - The outer cutoff becomes a max/min over the time-level codes.
   - `InnerFamily.concat` replaces `droplevel(1)`.
   - Positional slicing is optional and comes later (P4) as a memory lever.
4. **BaseForecaster:** validate once, then **one shared `_core` per public method**.
   - Trusted entries `_fit_trusted`, `_predict_trusted`, `_update_trusted` and `_predict_{interval,quantiles,var}_trusted(..., SliceCtx)`. They set exactly the state the public path sets (g1 list: `_y_metadata`, `_cutoff`, `_fh`, `_y`/`_X`, `_is_vectorized`, `_state`).
   - `_HashIndex.__eq__`/`__hash__` makes the fh `lru_cache` hit, so `fh.to_absolute` runs once per unique cutoff.
   - **Each slice gets its own ForecastingHorizon copy** (red-team D5). Stock `check_fh` mutates the shared FH, and the memo makes that unsafe under threaded backends.
5. **Trusted nested-transformer entry** (red-team D1, which corrects the draft).
   - The trusted slice runner **sets and resets a `ContextVar` inside the per-slice function**, so it runs in the worker: loky, threading or dask. `BaseTransformer.fit_transform`/`transform` checks the var and enters `_fit_transform_trusted`, which skips check/convert.
   - The explicit `SliceCtx` is used only for the BaseForecaster → `_*_trusted` hop. An explicit argument cannot reach `Lag` inside an unchanged forecaster's `_predict` without editing every forecaster.
   - The same entry is mirrored for BaseTransformer's own vectorization, so behaviour is consistent across scitypes.
6. **ProbaSpec + long table** built once.
   - `long_to_multiindex` for pandas input.
   - One skpro distribution built from a parameter table.
   - Fix `BaseProbaForecaster` (`_yvec` is never set) or fold it into `_vectorize`.
   - A known-type proba mixin: no `convert_to`/`infer_freq` per instance.
7. **Opt-in numpy inner mtypes** `np_series`, `np_grouped`, `np_grouped_hier` and `pa_*` (g2).
   - Kernels must be **order-robust**: sort inside the aggregation, which costs 1.11–1.38x vs 3.7–5.3x for a separate sortedness check [M DEL-3].
   - Kernels must keep stock NaN semantics.
   - Kernels must be equality-gated against their own trusted-lane run, **including the output time-level freq**.
8. **TimeAxis/Step** (a3) is used internally, with a pandas-offset fallback for aliases where Step is inexact (SME, tz W, BMS/BME [M a3 verifier]). The public cutoff and fh stay pandas.
9. **`capability:lookback`**: set per method, default `None`, **declared only, never inferred**. It is used only by lazy spokes.
10. **`instances=` / `columns=` selection** for lazy inputs (red-team N10).
    - It is carried by a `ReadSpec` passed to `fit(..., read_spec=...)` or set via config.
    - This is the only API through which the selective-read value in §4 is reachable.

### 2.2 Data flow per input

| Input | Path | Output |
|---|---|---|
| pandas | identity spoke → pandas lane (or np_grouped port) | pandas, bit-identical for the per-instance lane |
| polars eager | guarded vectorized check → one polars→pandas conversion (≈0.11 s at 10k [M T3]), or np_grouped with no pandas object built [M g2] | polars, with names restored (behind config `output:native_container` during migration, see §5) |
| polars-lazy / duckdb | `select` with ReadSpec pushed into the plan/SQL → one explicit `collect(backend=polars|pyarrow)` → eager path | eager frame of the same engine (polars / pyarrow Table) |
| pyarrow | `pa_*` spoke | pyarrow |
| dask | `select` → `compute()` once → pandas path | **pandas** (red-team D6: "eager in the same engine" is undefined for dask); partition-and-delegate deferred to a G7 study |

**Where pandas remains:**
- the pandas lane and per-instance inner objects;
- the public cutoff (1 element) and fh (H elements);
- proba MultiIndex output for pandas input;
- skpro distributions;
- `forecasters_` for per-instance forecasters;
- the Step fallback for exotic offsets;
- V4 third-party internals.

### 2.3 Mapping of the a1 touchpoint categories

| a1 category | Destination |
|---|---|
| container ops (42) | pandas lane, unchanged; np_grouped for ports |
| time/freq/cutoff (78) | once per panel: codes max/min, FhAbsCache, TimeAxis for ports |
| metadata / validation | checked once at the top level by the spoke, with guards and anti-join containment |
| type dispatch (34 hasattr sites) | `backend_of` registry |
| output construction | OutputSpec/ProbaSpec, built once |
| forecasters_ pivot | pandas, unchanged (≲1% of fit [M a1 §6]) |
| mtype pinning | unchanged inner-mtype tags, plus registered np_* mtypes |

---

## 3. Single waterfall (marginal per phase; cumulative shown once)

Source: the g3 §3.1 table (24 rows; Naive(last) and PolynomialTrend; pandas, polars and polars-lazy; 100–5000 series; fit+predict; **U=1 shared cutoff**). Nested-entry rows come from design_minimal §9.1 and the red-team re-run.

| Step (phase) | Naive marginal | Poly marginal | Status / caveat |
|---|---|---|---|
| P0 known-mtype + vectorized polars check (g3 C1) | 1.05–1.20 | 1.09–1.34 | [M] as an **identity-memo upper bound**. The memo also answered nested Lag re-inference. The explicit `from_type` version is unmeasured, and the red-team re-run of C1 on pandas gave 0.96 (noise). |
| P2 trusted per-slice lane (C2) | 1.17–1.39 | 1.45–1.73 | [M] g3, consistent with g1 |
| P2 fh memo per unique cutoff + outer cutoff over codes (C3) | 1.05–1.28 | 1.26–1.53 | [M] at U=1. [E] about 1.0 at U=n (ragged ends) |
| **Cumulative C0→C3** | **1.55–2.02** | **2.23–3.03** | [M] the only cumulative figure used in this report |
| P2 trusted nested-transformer entry | 1.18–1.54 (Naive only) | n/a (Poly's residual is sklearn fit, 63%) | [M] two runs at n≤1000 with ±20–25% noise. The prototype used a type-triggered `TrustedLag` swap, not the ContextVar mechanism → re-measure (§8) |
| Naive cumulative incl. nested entry | **~2–3x** (2.16–2.48 report; 2.23 pd / 3.00 pl re-run) | — | [M] noisy |
| P4 positional/hub ingress (C4) | 0.95–1.13 | 0.93–1.14 | [M] time is noise level. **Peak RSS 1.4–2.2x lower vs C3** (fresh process) |
| P3 np_grouped kernel port (C5 over C4) | 76–2,330 | 32–461 | [M] e2e vs stock 106–4,560x. Ported forecasters only |

**Not on the waterfall:**
- predict-only: 0.9–1.45x with the trusted lane alone [M g1]; 1.60–1.93x with the nested entry [M design_minimal; red-team 1.78–1.98].
- `predict_interval`: ~1.0x [M g1]. Proba-once removes `_get_columns` (~4% of interval time [M g4]), so the expected e2e interval gain from P2.5 is ≤~1.05x [E]. The real interval lever is the known-type proba mixin [E] plus kernels (~1,100x for Naive [M g4]).
- ETS/statsmodels: ~1.0x under any framework change [M g1].
- Full workload (fit, predict, interval, update, predict): 1.07–1.72x [M g1, trusted lane only].

---

## 4. Roadmap (PR-sized)

PRs are listed in merge order. Each PR is independently valuable and can be reverted. All performance PRs are gated in CI on **counts** (public re-entries, mtype checks, conversions) plus a loose timing threshold. Hard timing gates would flap at ±20–25% noise (red-team N12).

### P0: decision-independent fixes (about 10 PRs)

**Scope:**
- **PR0.1** null-guarded vectorized polars monotonic check
- **PR0.2** `from_type` pass-through from `_check_X_y` to prepare/convert
- **PR0.3** `get_cutoff` without a whole-frame `convert_to`
- **PR0.4** `_HashIndex.__eq__`/`__hash__`
- **PR0.5** guard `pd.infer_freq` for <3 points (polars update with 1–2 rows)
- **PR0.6** a real polars X/y containment check, and removal of `hasattr(obj, 'X')`
- **PR0.7** Panel `predict_residuals` fh
- **PR0.8** no pd3 caller mutation
- **PR0.9** `predict_var` column name, and `cov=True` NaN (`np.fill_diagonal`)
- **PR0.10** `BaseProbaForecaster._yvec` / `_check_X`

**Expected impact:**
- [M] 1.05–1.34x e2e fit+predict (g3 C1; identity-memo upper bound, see §3).
- [M] Monotonic check 921→33 ms (x9).
- [M] polars→pandas ingress at 10k 2.62 s → ~0.11 s, about 24x (final_bench T3, x9 adapter without check). The 0.019 s figure is x4's stay-in-polars path and is **not** claimed here (red-team N1).
- Fixes at least 7 stock bugs (g5 S6/S7/S8, a1, g4).

**Exit criteria:**
- One regression test per bug.
- Suite green on pandas 2 and pandas 3.
- The vectorized check equals stock on the x9 null fixtures.

**Risk:** low. Without the null guard the check diverges (58/75 [M x9 verifier]).

### P1: VectorizedDF + datatypes substrate (5 PRs)

**Scope:**
- **PR1.1** public `register_mtype`/`register_spoke`, an entry-point group loaded lazily, and versioned caches.
- **PR1.2** `backend_of` by module name (no imports), replacing hasattr duck typing.
- **PR1.3** `time_extreme(min|max)` and the `time_contains` anti-join on the pandas and polars spokes.
- **PR1.4** VectorizedDF: `from_type`, outer cutoff over codes, concat instead of `droplevel(1)`.
- **PR1.5** conformance CI gate: the g5 240-cell matrix with label checks, plus the M4 adversarial fixtures, on pandas 2 and pandas 3.

**Expected impact:**
- [M] Time ~1.0x by design. Iteration is 1–2% of fit+predict (x8/x3); slicing once vs twice is 1.18 vs 1.19 (g1).
- Unblocks third-party inner mtypes.

**Exit criteria:**
- 0 SILENT cells.
- pandas outputs bit-identical.
- Registrations survive `cache_clear`.
- **Target** (not an expectation; red-team N11): ≥20/24 polars strict-equal cells. Today stock has 9/24, x4 5/24 and x6 7/24 [M g5].

**Risk:** low.

### P2: BaseForecaster boundary, the main lever (8 PRs)

**Scope:**
- **PR2.1** trusted lane behind config `trusted_vectorize` (**default off**), using g1's logic. Includes per-slice FH copies and a threading-backend test (D5).
- **PR2.2** dedup refactor: validate-once plus a shared `_core`, so public and trusted paths call the same code. It has its own pd2×pd3 gate and counts gate (red-team D4).
- **PR2.3** default on **for sktime-owned classes**. Third-party classes opt in with the tag `capability:trusted_vectorize=True` or by passing the new `check_estimator` trusted==stock test. They get a FutureWarning in N, and the default flips for them in N+2 (D3).
- **PR2.4** fh memo per unique cutoff.
- **PR2.5** worker-side TRUSTED ContextVar plus `BaseTransformer._fit_transform_trusted`.
- **PR2.6** ProbaSpec/long table built once, plus the known-type proba mixin.
- **PR2.7** edges: top-level X/y anti-join, and `evaluate`/`update_predict` convert once.
- **PR2.8** `output:proba_format` and `output:native_container` configs (FutureWarning for non-pandas input).

**Expected impact** (marginal over P0, U=1, Naive/Poly; red-team N4/N5):
- [M] Trusted lane plus fh memo: C3/C1 is **Naive 1.35–1.69x, Poly 2.04–2.46x**.
- [M] Nested entry: a further 1.18–1.54x for Naive.
- [M] update 1.3–3.2x and full workload 1.07–1.72x (g1).
- [E] interval ≤~1.05x from proba-once.
- [E] ETS ~1.0x.
- [E] Gain shrinks toward 1.0 for the memo part as unique cutoffs approach n.
- **New capability, not speed:** vectorized Panel `predict_proba`, which raises today. Building it from a long parameter table takes 3.2 ms vs 826 ms for today's concat path [M g4 at 1k].

**Exit criteria:**
- The forecasting suite passes in both modes with delta 0 (g1's 7710-test gate).
- The `check_estimator` trusted==stock test passes.
- Sampled per-slice state is equal (n up to 1000, including Lag state).
- Counts on the 1000x100 fixture: public re-entries ≤ 1 per call, plus a mtype-check budget.
- Timing ≥1.2x.

**Risk:** medium.
- Public/trusted drift. Mitigated by PR2.2 and the both-mode CI.
- pandas-3 fragility in time code (g3 found 2 bugs).
- The ContextVar entry is untested beyond Lag.

### P3: estimators, opt-in np_grouped kernels (1 + about 9 + 2 PRs)

**Scope:**
- **PR3.0** `np_*`/`pa_*` mtypes and TimeAxis/Step converters registered through PR1.1.
- **PR3.0** also includes the per-call fallback: if a port's `_*_supported(method, fh, X, data)` says no (sp>1 intervals, in-sample fh, X, tz calendars), BaseForecaster routes that call through the trusted pandas lane. Cost: one predicate call plus one conversion [E]. The port then performs at trusted-lane speed for that call (D2).
- **PR3.x** one PR per port: Naive (with intervals), PolynomialTrend, Trend, Croston, TSB, ForecastKnownValues, seasonal naive, `_BaseWindowForecaster` (including the `_sktime.py:151` fix).
- **PR3.y** lazily materialized `forecasters_` view for ports.
- **PR3.z** composite inner-mtype tag derivation.

**Shipping rule:**
- Ports are **opt-in** via config `forecasting:panel_kernels=True` until PR3.y and the level-freq gate land. Only then do they become the default.
- The old implementation stays selectable for at least 1 release.

**Expected impact:**
- [M] g2 at 1000x100: 498–2,735x e2e (pandas input) and 498–1,486x (pyarrow).
- [M] polars input: g3 C5 **116–1,363x** (≈0.4–0.6x of the pandas-input gain; the polars check is still ~45% of the time).
- [M] Conversions ~804→2; mtype checks 7,484–12,084→36.
- [M] Naive interval kernel ~1,100x [g4].
- [M] Reducer ports only 6–9x.
- Coverage is about 9 of 160 leaf forecasters, plus derived composites.

**Exit criteria:**
- Each port is equal to its own trusted-lane run on pandas 2 and pandas 3 (predict, interval, update, absolute fh, ragged, shuffled rows) **including output time-level freq and cutoff freq**.
- Poly is checked against an explicit lstsq reference with a documented deviation from sklearn's version-dependent `cond=tol` truncation (D10).
- Per-backend conformance passes.

**Risk:** medium per port.
- Stock quirks must be replicated (cond=tol, drift slope, frozen window).
- `forecasters_` parity depends on PR3.y.

### P4: memory + lazy (3–4 PRs)

**Scope:**
- **PR4.1** `ReadSpec` (`instances`, `columns`, `lookback` where declared) pushed into the polars-lazy/duckdb plan, with one explicit collect.
- **PR4.2** positional pandas-lane iteration and polars→np_grouped guarded ingress (hub C4).
- **PR4.3** lookback declarations, only on structurally safe forecasters.
- **PR4.4** (gated on a G7 study) dask partition-and-delegate.

**Expected impact:**
- [M] Selective read of 1000 of 20,000 series × last 50, against the **P0-fixed** eager baseline B1c (0.742 s, 659 MB):
  - lazy then adapter B5b: 15.5 ms / 44 MB, **~48x faster, ~15x less memory**;
  - polars lazy B2: 5.9 ms / 31 MB, ~126x / ~21x [final_bench T6].
  - This is reachable only through `ReadSpec` `instances=`.
- [M] Against user-side select-before-convert, true pushdown is **1.9x time / 2.8x RSS** [x9 verifier].
- The draft's 190–875x figures used the pre-P0 baseline and are withdrawn (N10).
- [M] Full-history workloads: ±15% (g3).
- [M] C4: peak RSS 1.4–2.2x lower vs C3, time 0.93–1.14x. polars ingress +30 vs +451 MB at 50k (T5b), with hub ingress 11 ms at 10k (T3).
- [M] dask delegation 0.70–1.31x on one 4-core box (DEL-4).

**Exit criteria:**
- A fresh-process memory harness confirms the RSS gain.
- x6 29/29 and the 33 adversarial fixtures pass.
- The pd3 matrix is green.
- Lazy-input conformance passes.

**Risk:** medium.
- A mis-declared lookback silently changes forecasts. DEL-1 found 3/26 coincidental lookbacks, hence default None.
- Lazy row order.
- Private pandas constructors.

### P5: contract flips, N+2 (1–2 PRs)

**Scope:**
- `output:proba_format` `auto` → long for non-pandas input.
- `output:native_container` default on.
- Trusted lane default on for third-party classes.
- `params_`-first API for ports.

**Expected impact:** no speed change. Types become consistent for non-pandas users.

**Exit criteria:**
- A full deprecation cycle has completed.
- A downstream usage survey has been done.

**Risk:** downstream code that parses proba MultiIndex output for non-pandas input (unmeasured).

---

## 5. Public API decisions

| Contract | Decision |
|---|---|
| Return type | pandas in → pandas out, unchanged. Non-pandas in: config `output:native_container`. In N the default is the current behaviour plus a FutureWarning; in N+2 the default is native (predict*, update_predict, predict_residuals, evaluate y_pred, and labels restored, which fixes the y→"0" rename). This is **a back-compat change, not a silent bug fix** (D7). dask → pandas. Lazy input → eager frame of the same engine. |
| cutoff / fh | Stay pandas `Index` / `ForecastingHorizon` permanently (O(1)/O(H) metadata). New `cutoffs_` per-instance frame (min and max). Ports carry `cutoff.freq` where stock MI `get_cutoff` returns None (81/315 pd2, 9/315 pd3 [M g2]). This is a documented behaviour change that applies only on the opt-in port path. |
| Proba | pandas input keeps the MultiIndex columns **indefinitely**. Non-pandas input: `output:proba_format ∈ {auto, multiindex, long}`, with a FutureWarning in N and auto→long in N+2. `predict_var` keeps the predict layout. `predict_proba` stays skpro, plus `to_long()`. |
| `forecasters_` | A DataFrame of clones for per-instance forecasters, **no deprecation**. Ports: `params_` plus a lazily materialized `forecasters_` view that warns on access. Ports are not on by default before this view exists. |
| `get_fitted_params` | `forecasters.loc[...]` keys unchanged. Ports add `params__<col>` and `style="table"`. |
| Trusted lane opt-out | Class tag `capability:trusted_vectorize=False` or config `trusted_vectorize=False`. Instance `set_tags` does not work, because `reset()` wipes it [M g1]. |
| Third-party extensions | Trusted lane off by default for out-of-tree classes in N. Opt in by tag or by passing the `check_estimator` trusted==stock test. |
| New tags/config | `capability:trusted_vectorize`, `capability:lookback` (per method, default None), `np_*`/`pa_*` inner mtypes, `forecasting:panel_kernels`, `output:proba_format`, `output:native_container`, `ReadSpec`. |
| Not adopted | No tier vocabulary, no global `inner_dispatch` flip, no deletion of ported implementations within one release. |

---

## 6. Approach scorecard

| Approach | Verdict | Quantified impact (evidence) | Role here |
|---|---|---|---|
| Adapter / dispatcher (x4) | Adopt narrowly as edge spokes, not as its own phase | [M] pandas adapter bit-identical 78/78 + 28/28 at 0.89–1.06x; polars e2e 0.96–1.17x (x4 verifier); container-only ≤1.05x (x8); per-slice input dispatch 1.07–1.31x **slower** (x4 §4.2) | spoke registry, `time_extreme`/`time_contains`, conformance gate |
| narwhals | Reject as core; possibly inside lazy spokes only | [M] validation 0.010–0.025x of the stock polars check (DN-B1); kernels 3.6–4.2x slower than pandas `groupby.last` even routed to polars (DN-R); 7–11x overhead vs hand-written (x5 verifier); 70–620x slower than the stock check on pandas-long; silent bugs in 2.27.1 | none in core; optional hybrid experiment |
| Hub / canonical GroupedArray (x6) | Adopt as the `np_grouped` kernel container and the P4 memory ingress; do not replace VectorizedDF | [M] time 0.93–1.14x (g3 C4); peak RSS 1.4–2.2x lower; polars ingress 2.84 s→11 ms at 10k (T3); fidelity 25–26/33 adversarial (x6 verifier) | P3 container, P4 memory lever |
| Per-estimator opt-in via inner-mtype tags | Adopt for the estimator phase | [M] g2 498–2,735x e2e; 315/315 + 233/233 equal; conversions ~804→2; reducer only 6–9x; ~9 closed-form candidates; top 10 of 96 implementations cover 28% (tiered census) | P3 mechanism, one gated port per PR |
| Trusted fast lane | **Adopt first** (core lever) | [M] g1 fit 1.2–5.5x, update 1.3–3.2x, workload 1.07–1.72x, 7710/7710 tests, 159/160 eligible; marginal C2 1.17–1.73x; C0→C3 1.55–3.03x (U=1); plus nested entry, Naive ~2–3x | P2 core |
| Engine-native panel hook (polars/duckdb kernels) | Reject for core; use the numpy kernel tier | [M] engine over pandas groupby with conversion 1.07–1.33x (x7 verifier); numpy 7–11x over pandas groupby; order-robust contract 1.11–1.38x vs 3.7–5.3x (DEL-3); duckdb in-SQL 12–24x lower RSS on series-major parquet (DEL-2) | graft the order-robust contract and `params_` |
| Arrow-backed pandas | Reject as inner representation | [M] e2e 1.07x (Naive) / 1.19x (Poly) slower (design_minimal §9.2); Arrow survives in 25% of outputs, 7/71 forecasters fail, zero-copy only for single-chunk data (x9 verifier) | none |
| Lazy-aware boundary | Adopt at the edge only (P4) | [M] selective read vs P0-fixed eager ~48x / ~15x memory (B5b) to ~126x (B2); true pushdown over select-before-convert 1.9x / 2.8x RSS (x9 verifier); full history ±15%, C5 1.1–2.6x slower (g3); finite lookback rare, 1/39 safe for intervals (DEL-1) | `ReadSpec` pushdown + one explicit collect |

---

## 7. narwhals claims, final verdicts

**C1: "collect() materializes to a default backend that may not be the input's". Verdict: TRUE.** It is a design constraint, not a blocker.
- [M x1 + verifier] Defaults were reproduced on pandas 2 and pandas 3: dask→pandas, duckdb→pyarrow.
- Under `stable.v1`, duckdb and ibis are not LazyFrames at all, so any use must pin `stable.v2`.
- Workaround [M DN-API]: `collect(backend=...)`, then relift with `.lazy(backend=orig)`.
- Staying lazy pays 1.5–3.9x at 500k rows, with a memory gain of only 1.3–7x, not the 6–35x first claimed (x1 verifier).
- A lazy per-key loop is about 15x worse than eager, so per-instance forecasters must collect once.
- Neither stable API has `last(order_by=)`.
- Rule: only explicit `collect(backend=...)`, never a bare `collect()`.

**C2: "No index; MultiIndex panels need reset_index and can lose names". Verdict: MOSTLY TRUE.**
- narwhals is columns-only, so `reset_index` is mandatory.
- Names are lost only on a naive path.
- The x2 schema recipe restores names, dtypes, freq, Period and order in 198/198 checks [M x2]. The x2 verifier rates this **partially confirmed**: the recipe fails on mixed-dtype frames and on per-series freq.
- x5 matches stock 17/17 on pandas.
- Cost on pandas: iteration 1.69–2.26x slower on pd3 under fair accounting (x2 verifier); validation on pandas-long 70–620x slower than the stock pd-multiindex check (DN-B1).
- Consequence: the pandas MultiIndex lane must stay native. That would force a dual pandas/narwhals path, which is a main reason narwhals is not the core.

---

## 8. Rejected options and why

- **narwhals as core.** It is not a speed lever: it removes at most the ≤1.18x non-pandas penalty (T4). Its kernels are slower than plain pandas groupby (DN-R). It has silent bugs in 2.27.1. It forces a dual path (C2). It would be a pinned core dependency that is transitive only for sklearn 1.9.x users.
- **Hub replacing VectorizedDF.** VectorizedDF is shared across scitypes. The float64 hub reaches 25–26/33 fidelity (NaN/null, int64>2^53). Private `_simple_new` was broken twice by pandas 3. g5 H1 freq loss broke pandas `update_predict`. Time gain is ≈1.0x.
- **Adapter registry as a first phase.** It is worth ≈1.0x and would delay g1, which needs no substrate.
- **Per-engine native kernels.** They would need a second estimator API × 3 engines × K forecasters × 4 methods, for 6–8 forecasters, and gain only 1.07–1.33x over pandas groupby once conversion is charged.
- **Arrow-backed internals.** 1.07–1.19x slower e2e; Arrow survives in 25% of outputs; 7/71 forecasters fail.
- **Lookback inference or default pushdown.** It gives silently wrong forecasts (3/26 coincidental lookbacks). Polars lookback saves nothing beyond projection (DEL-2).
- **Deprecating `forecasters_` or the `.loc` keys for everyone.** ≲1% speed value, and it breaks 4/4 hierarchical metric configs.
- **Tiered's global `inner_dispatch=auto` flip and delete-on-port.** Not revertible per estimator, removes the rollback, and breaks external subclasses.
- **An explicit context argument for nested transformers** (the draft's choice). It cannot reach nested calls in unchanged forecasters. Replaced by a worker-side ContextVar.
- **dask partition-and-delegate in v1.** 0.70–1.31x on one box (DEL-4).

---

## 9. Risks

1. **Public/trusted drift.**
   - g1 copied ~700 LOC of stock logic.
   - Mitigation: PR2.2 shared `_core`, both-mode CI, and the `check_estimator` equality test.
2. **Third-party subclasses that rely on side effects of the public path.**
   - Mitigation: off by default for out-of-tree classes in N.
3. **pandas-3 fragility in time code.**
   - Known cases: Day vs 24h, datetime unit, caller mutation.
   - Mitigation: pd2×pd3 CI and a Step → pandas-offset fallback.
4. **Headline numbers generalize poorly.**
   - They are Naive/Poly fit+predict at U=1.
   - ETS and interval are ~1.0x.
   - Ragged cutoffs remove the memo share.
5. **Noise.**
   - Single-run ratios at n≤1000 vary by ±20–25%.
   - This is why CI gates on counts, not hard timings.
6. **Kernel ports replicate stock quirks.**
   - Fixing them later (e.g. exact lstsq) is a behaviour change.
   - Equality is tied to the sklearn version.
7. **Batched ports lack a real per-instance `forecasters_`.**
   - Hence the opt-in until PR3.y.
8. **Silent per-backend divergence at the edges.**
   - x4 had 3 such bugs; g5 found A1/A2.
   - Mitigation: the conformance gate is required before any non-pandas path merges.
9. **Composite tag derivation is unprototyped.**
   - Composites stayed silently on pandas under x4.
10. **The nested-entry mechanism is not yet measured.**
    - The proposed ContextVar has not been timed.
    - Only a subclass-swap prototype for Lag in Naive was timed.
11. **A mis-declared lookback corrupts forecasts.**
    - Hence default None and declared-only.
12. **The N+2 flips may break downstream parsing of proba MultiIndex or pandas outputs for non-pandas input.**
    - The extent is unmeasured.

---

## 10. Remaining experiments (in priority order)

1. A worker-side ContextVar trusted nested entry. Test across Naive/Lag, TTF[Detrender/Deseasonalizer], reducers and STLForecaster; run the g1 suite in both modes; check Lag state equality; use 1k–5k series; include loky and threading backends.
2. An explicit `from_type` version of C1 (no identity memo) on pandas and polars, to replace the upper bound.
3. A known-type proba mixin: `predict_interval`/quantiles speed (≈1.0x today).
4. fh memo with ragged cutoffs (U=n/10, U=n).
5. The full P0+P2 stack on pandas 3.0.6 with equality checks; g2/g3 timings under pd3.
6. A conformance run (g5 240 cells + M4 + labels) against the P1/P2 spokes. Target ≥20/24 polars strict-equal cells and 0 SILENT.
7. Cost of PR2.2 (shared `_core`) vs the g1 copy: counts and timing on pandas 2 and pandas 3.
8. A pandas-groupby kernel vs the np_grouped port for Naive/Poly, including guard cost (G13).
9. Lazy `forecasters_` view for ports: access cost at 1k/10k, and the a1 grid and metric tests.
10. Per-call fallback cost for port-unsupported calls (sp>1, X, in-sample fh).
11. Composite tag derivation: does TTF([Detrender, NaiveNp]) stay on np_grouped?
12. Fresh-process memory of P2 vs P2 + positional/np_grouped ingress at 50k series.
13. A lookback census in a full-dependency environment, and cold/remote parquet pushdown through `ReadSpec` (G9/G14).
14. A narwhals-in-lazy-spoke hybrid under `stable.v2`: LOC saved vs per-call overhead.
15. dask partition-and-delegate break-even on a multi-worker cluster (G7).

---

## 11. Review response (red team, `reports/design_redteam.md`)

| # | Sev | Issue | Response |
|---|---|---|---|
| D1 | major | An explicit context argument cannot reach nested Lag/Detrender; the reason given for rejecting the ContextVar was wrong; 2.16–2.48x came from a subclass swap | **Fixed.** A worker-side ContextVar is set and reset inside the trusted slice runner (§2.1 item 5). SliceCtx is kept only for the BaseForecaster → `_*_trusted` hop. The figure is relabelled "~2–3x, prototype mechanism differs". Re-measurement is experiment #1. I did not re-measure in this synthesis step. |
| D2 | major | "No API change" is false from P3 on | **Fixed.** Ports are opt-in (`forecasting:panel_kernels`) until the `forecasters_` view and the level-freq gate exist. There is a per-call fallback to the trusted lane for unsupported calls, at trusted-lane cost plus one predicate call [E] (experiment #10). The equality gate includes the output time-level freq and the cutoff freq. The cutoff-freq difference is documented as a behaviour change on the port path only. |
| N4/N5 | major | Gains double-counted | **Fixed.** There is one waterfall (§3) with per-phase marginals and a single cumulative figure. P2 is quoted as C3/C1: Naive 1.35–1.69x, Poly 2.04–2.46x. |
| N1 | major | The P0 figure 2.62→0.019 s belongs to x4's stay-in-polars path | **Fixed.** P0 quotes 2.62 s→~0.11 s (~24x). The 11 ms hub ingress figure is moved to P4. |
| N10 | major | The 190–875x selective-read figures use a pre-P0 baseline and no API reaches the workload | **Fixed.** `ReadSpec` with `instances=`/`columns=` is defined (§2.1 item 10). Numbers are re-based on the P0-fixed B1c baseline (~48x / ~15x for B5b; ~126x for B2). True pushdown over select-before-convert (1.9x / 2.8x) is quoted alongside. |
| D3 | major | Trusted lane on by default for third-party code | **Fixed.** Default on only for sktime-owned classes. Third-party classes opt in via the tag or the `check_estimator` trusted==stock test. FutureWarning in N; flip in N+2. |
| N7/N8 | minor | Proba-once presented as speed | **Fixed.** The interval effect is labelled [E] ≤~1.05x (`_get_columns` is ~4%). The 10k `_get_columns` and concat figures are no longer used as speed claims (they were [S]). The distribution figure is moved to "new capability", because stock Panel `predict_proba` raises. |
| N3 | minor | C1 measured with an identity memo | **Fixed.** Labelled "identity-memo upper bound; from_type version unmeasured" (§3), and added as experiment #2. |
| N6/N12 | minor | Noise of ±20–25%; a ≥1.5x CI gate would flap | **Fixed.** "~2–3x" is quoted. CI gates on counts plus ≥1.2x timing. |
| N9 | minor | "polars ~pandas-level after P0" contradicts g3 C5 | **Fixed.** The measured 116–1,363x (≈0.4–0.6x of the pandas gain) is used. |
| N11 | minor | ≥20/24 polars cells has no basis | **Fixed.** It is now an exit target, not an expectation. |
| D4 | minor | Shared `_core` refactor is unmeasured and touches every public method first | **Fixed.** PR2.1 lands the trusted lane behind a config (default off). PR2.2 deduplicates under its own pd2×pd3 and counts gate. Measurement of the refactor is experiment #7. |
| D5 | minor | Shared FH mutated per slice plus the memo; threading untested | **Fixed.** Each slice gets its own FH copy, and a threading-backend test is added in PR2.1. |
| D7 | minor | Return-type changes for polars users presented as a bug fix | **Fixed.** They go behind config `output:native_container`, with FutureWarning in N and a flip in N+2, separate from the performance PRs. |
| D6/D8 | minor | Dask egress undefined; imports at registry time | **Fixed.** dask → pandas output. `backend_of` checks the module name without importing. Entry points load lazily on the first registry miss. |
| D9/D10 | minor | fh memo measured only at U=1; Poly equality tied to sklearn | **Fixed.** The U=1 qualifier is attached to every headline number, with U=n experiment #4. Poly equality is gated against an explicit lstsq reference with a documented deviation. |
