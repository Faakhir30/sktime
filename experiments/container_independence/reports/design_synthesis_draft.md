# design_synthesis (draft): one recommended architecture for decoupling sktime forecasters and VectorizedDF from pandas

Audience: sktime core developers. Status: draft synthesis. **No new benchmarks were run for this report.** Every number
is quoted from a workstream report or its verifier under `experiments/container_independence/reports/`. Numbers are
labelled **[M]** when measured (with the source report) and **[E]** when expected or unmeasured (an extrapolation or a
design target). "Stock" means this branch's sktime.

Inputs: 6 design proposals (`design_{minimal,tiered,adapter,hub,narwhals,engine_lazy}.md`), 3 judges
(`judge_maintainer.md`, `judge_value.md`, `judge_risk.md`), gap experiments g1–g5, `wrap_critic.md`, `final_bench.md`,
`results/investigation_digest.json`.

---

## 0. Decision in one paragraph

Build on **design_minimal**, the highest-scoring proposal (judges 8.0 / 7.5 / 8.5, mean 8.0; next is tiered at 6.67).
Keep pandas as the inner container for the 150-odd per-instance forecasters. Remove the per-slice framework re-entry
with a **trusted fast lane** that shares one core with the public methods, compute fh/cutoff **once per unique cutoff**,
add a **trusted nested-transformer entry**, and **build proba output once**. Convert exactly once at the edge through a
small **spoke registry** that returns the user's container, with **selection pushdown** for lazy inputs. Graft from the
other designs:
- tiered's **public `register_mtype`** (entry point, versioned caches);
- adapter's **conformance CI gate** and explicit **`time_extreme(min|max)` / `time_contains`** ops;
- the g2 **numpy `np_grouped` inner mtype** as the single opt-in kernel tier, with engine_lazy's **order-robust kernel
  contract**;
- hub's **non-disableable ingress guards**;
- engine_lazy's **declared-only, default-None `lookback`** tag.

Container-only changes are worth ≤1.05x [M, x8]. The plan delivers about **1.6–2.5x for unchanged forecasters**
on fit+predict [M for Naive/Poly; E elsewhere] and **10²–10³x for a handful of ported closed-form forecasters** [M, g2/g3
C5]. pandas users see no API change. Do **not** adopt narwhals, a hub, an in-loop adapter or Arrow-backed internals as
the core architecture.

---

## 1. Why this base, and what the evidence settles

| Settled fact | Value | Source |
|---|---|---|
| Container-only change under unchanged per-instance BaseForecaster calls | ≤ ~1.05x e2e | [M] x8 + verifier |
| Per-slice public re-entry (check/convert/fh/cutoff/clone) share of vectorized fit | 33–63% | [M] a1 verifier, x8 |
| Trusted per-slice lane (g1) | fit 1.2–5.5x, update 1.3–3.2x, predict 0.9–1.45x, interval ~1.0x; fit+predict 1.15–1.88x; full workload 1.07–1.72x; 7710/7710 tests, delta 0; 159/160 eligible | [M] g1 |
| Cumulative container-independent stack g3 C1→C3 | Naive 1.55–2.02x, Poly 2.23–3.03x (overall 1.55–3.03x) fit+predict | [M] g3 |
| Hub/positional iteration (g3 C4) on top of C3 | time 0.93–1.14x (noise); peak RSS 1.4–2.2x lower vs C3 | [M] g3 |
| Trusted nested Lag entry on top of C3 | +1.32–1.54x marginal; Naive fit+predict 2.16–2.48x vs stock, predict-only 1.60–1.93x; outputs equal 4/4 | [M] design_minimal §9.1 (n≤1000, 2 reps, Lag only) |
| Panel-native kernels | g2 498–2,735x (pandas in), 498–1,486x (pyarrow), 34–105x (polars, 99% stock checks); g3 C5 105–4,560x | [M] g2, g3 |
| Engine over pandas groupby, conversion charged | 1.07–1.33x | [M] x7 verifier |
| numpy segment kernels vs pandas groupby | 7–11x | [M] x7 verifier |
| Lazy selective read | 875x / ~25x memory (B2), 332x with conversion (B5b); pushdown over select-before-convert 1.9x time / 2.8x RSS | [M] final_bench T6, x9 verifier |
| Laziness on full-history fit+predict | within ±15% of eager; 1.1–2.6x slower at C5 | [M] g3 |
| Arrow-valued pandas e2e | 1.07x (Naive) / 1.19x (Poly) **slower**; Arrow survives in 25% of outputs; 7/71 forecasters fail | [M] design_minimal §9.2, x9 verifier |

So the speed levers are container-independent (g1, g3 C1–C3, nested entry) or estimator-specific (kernels). The
container decision matters for **correctness at the boundary, memory, pushdown and return types**, not for per-instance
speed. design_minimal is the only proposal that delivers all the measured levers without a new substrate first: the
judge_risk lens has it reaching the main lever by about PR 8. It also adds no dependency, no mandatory mtype and no
estimator API. The grafts fix its weak points: no public registry, the implicit ContextVar, the memory and kernel speed
it forgoes, and its weak conformance gating.

---

## 2. Recommended architecture

```
user y/X ──► Spoke.check (vectorized, null-guarded, non-disableable guards)          ◄── once per public call
   (pandas | polars | polars-lazy | pyarrow | duckdb | dask)
         └─► [lazy only] Spoke.select(Selection{instances, columns, lookback}) ─► collect(backend=explicit)  ◄── once
         └─► Spoke.to_inner(frame, inner_mtype, from_type=known)                         ◄── once, whole frame
                ├─ inner = pd-multiindex / pd.Series family (default, 152/160)
                │     VectorizedDF (stateless pandas (frame, schema)) ─► per-slice TRUSTED core (no re-entry)
                │        state written from SliceCtx; fh.to_absolute memo per unique cutoff;
                │        nested transformers called with trusted=SliceCtx (explicit arg)
                └─ inner = np_grouped (opt-in, values+indptr+KeyTable+TimeAxis)  ─► batched kernel
   outputs ─► OutputSpec / ProbaSpec (long table, built once) ─► Spoke.egress(like=input)     ◄── once
```

### 2.1 Components

1. **Spoke registry** (`sktime/datatypes/_boundary.py`, about 300 LOC + 70–140 per backend [E; x4 §6, x6 §3.8]).
   - `backend_of(obj)` replaces `hasattr` duck typing.
   - `register_spoke(spoke)` is public.
   - Spoke ops: `owns`, `check -> Meta`, `select`, `collect(backend=)`, `to_inner(from_type=known)`, `egress(like=)`,
     `proba_out(long, ProbaSpec, fmt)`, `time_extreme(how='min'|'max', per_instance)`, `time_contains(x, y)`.
   - Rule: **no spoke op runs inside a per-instance loop** (x4: per-slice dispatch on the input type gives N+N
     conversions and is never faster, A2/A1 1.07–1.31x [M]).
   - Guards run inside `check` and cannot be turned off: null/NaN key, strictly increasing time per instance,
     duplicate (key, time), regular step (hub §9: guarded ingress costs 1.4–3.5x unguarded and is still 29–71x cheaper
     than the stock polars check [M]). On pandas the stock `check_is_mtype` stays, because guarded hub ingress is
     10–25x more expensive than the bare stock check there [M].
2. **Public mtype registry** (tiered): `register_mtype(...)` plus an entry-point group, with versioned caches so that
   `cache_clear` cannot silently unregister (g2 measured exactly that failure [M]). This is a prerequisite for
   `np_grouped` and any third-party inner type.
3. **VectorizedDF stays stateless pandas** `(frame, schema)`. Changes:
   - it accepts `from_type` (no re-inference);
   - the outer cutoff is a min/max over level codes via `time_extreme`;
   - `concat` replaces the `droplevel(1)` (adapter `InnerFamily.concat`);
   - optional positional (codes-based) slicing comes later, as a memory lever (§4 P4).
   VectorizedDF is shared with BaseTransformer, `_base_panel` and the metrics (`vectorize_est`, 15 non-test sites,
   judge_maintainer), so it is **not** replaced (the reason hub was rejected).
4. **BaseForecaster: validate-once plus ONE shared `_core`** per public method.
   - Trusted entries `_fit_trusted / _predict_trusted / _update_trusted / _predict_{interval,quantiles,var}_trusted
     (…, ctx: SliceCtx)`.
   - `SliceCtx` is frozen per call and holds the direct converters, y-metadata template, tags, an `FhAbsCache` and the
     in-sample flag.
   - `_HashIndex.__eq__/__hash__` lets the fh memo hit once per unique cutoff (g3 C3: 999/1000 hits [M]).
   - The public methods call the same `_core`. **No copied logic**; g1 copied `_check_fh`, `update_data` and the
     default `_update` refit, which is the main drift risk.
   - Opt-out: class tag `capability:trusted_vectorize=False` and config `trusted_vectorize=False`. Instance `set_tags`
     does not work, because `reset()` wipes it (g1 [M]).
   - Fallbacks: forecasters that override public methods (ChronosForecaster), and `update_predict_single`.
5. **Trusted nested transformers.** `BaseTransformer._fit_transform_trusted(X_inner, y_inner, ctx)`, reached through an
   **explicit private argument / skbase-level context, not a module ContextVar**. A ContextVar is implicit state and does
   not cross loky process boundaries (judge_maintainer, judge_risk). The same pattern is mirrored in BaseTransformer's
   own vectorization.
6. **Proba built once** (g4).
   - `ProbaSpec` + long table (`keys…, time, variable, coverage|alpha, bound, value`), with `long_to_multiindex` for
     pandas output.
   - The proba mixin gets a known-type `convert_to` / `infer_freq` path.
   - `predict_proba` is built as one skpro distribution from a long parameter table.
   - `BaseProbaForecaster` vectorization is fixed (`_yvec` is never set) or folded into `_vectorize`.
7. **Opt-in kernel tier: `np_grouped` / `np_grouped_hier` / `np_series`** (g2), registered through (2).
   - Kernel contract (engine_lazy DEL-3): **order-robust** (sort inside the aggregation, which costs 1.11–1.38x, rather
     than check-then-positional, where a separate sortedness check costs 3.7–5.3x the kernel [M]). NaN/null semantics
     must equal stock. Output times are `cutoff + h*Step` from TimeAxis.
   - Each port is equality-gated against its own trusted-lane run.
   - The `pd-multiindex` inner tag stays available for pandas-groupby kernels. The numpy tier is the default for new
     kernel ports because numpy is 7–11x faster than pandas groupby [M x7 verifier] and g2 already proved 315/315 +
     233/233 equality on pd2 and pd3 [M].
8. **TimeAxis / Step** (a3), used inside the boundary and np_grouped only. `Step(kind="pandas")` is the fallback for the
   aliases where Step is inexact (SME, tz W, BMS/BME, int step 5; a3 verifier). Public cutoff/fh stay pandas.
9. **Lookback** tag `capability:lookback` (per method, default `None`, declared, never inferred: DEL-1 found 3/26
   coincidental truncation-equal models and only 1/39 safe for intervals [M]). It is applied only in lazy spokes.

### 2.2 Data flow per input

| Input | Ingress | Inner | Egress |
|---|---|---|---|
| pandas | identity; stock check, known mtype | pandas lane (trusted) or np_grouped (codes, zero copy) | pandas, unchanged layout |
| polars eager | vectorized null-guarded check; one `to_inner(from_type)` | pandas lane, or np_grouped with no pandas object built (g2 [M]) | polars, original names (fixes the y→"0" rename) |
| polars lazy / duckdb | `select` pushdown (instances, columns, declared lookback) then **one** `collect(backend=…)`; never iterate lazily (x4: 3–77x worse [M]) | as above | eager frame of the same engine (polars / pyarrow Table) |
| pyarrow | `pa_*` mtypes (g2) | as above | pyarrow |
| dask | `compute` once (v1) | as above | pandas, or dask on config; partition-and-delegate only after a G7 break-even (DEL-4: 0.70–1.31x on one box [M]) |

### 2.3 Where pandas remains (deliberately)

- The pandas lane for about 152 pandas-inner forecasters.
- The public 1-element cutoff and the H-element ForecastingHorizon (O(1)/O(H) metadata).
- Proba MultiIndex output for pandas input.
- The skpro distributions.
- `forecasters_` for per-instance forecasters.
- The Step fallback for exotic offsets.
- Third-party V4 internals.

pandas remains a hard dependency. That is acceptable: sklearn needs no pandas (r1), but sktime's public types are pandas.

---

## 3. Public API decisions

1. **Return type follows the input container** for `predict*`, `update_predict`, `predict_residuals` and
   `evaluate(return_data=True)`. Lazy input returns the eager frame of the same engine. Labels are restored from the
   schema. This fixes the polars y→"0" rename and evaluate returning pandas (g5).
2. **cutoff / fh stay pandas `Index` / `ForecastingHorizon` permanently.** freq comes from Step and survives fewer than
   3 points.
   - It is announced as a **behaviour change**, not a bug fix: stock MultiIndex `get_cutoff` returns `freq=None` where g2
     carries freq, in 81/315 (pd2) and 9/315 (pd3) cases (judge_maintainer correction of tiered).
   - Additive `cutoffs_`: a native per-instance (keys, cutoff_min, cutoff_max) frame. Both min and max are exposed,
     which removes the x4 reverse_order bug class.
3. **Proba**:
   - pandas input keeps the MultiIndex columns **indefinitely**.
   - Non-pandas input gets config `output:proba_format ∈ {auto, multiindex, long}`. In release N, `auto` still returns
     pandas MultiIndex output, with a FutureWarning. In N+2, `auto` returns long format in the input backend. `multiindex`
     stays available.
   - `predict_var(cov=False)` keeps the predict layout.
   - `predict_proba` stays a skpro object, plus `to_long()`.
4. **`forecasters_` stays a pandas DataFrame of clones for every per-instance forecaster, with no deprecation.**
   - The grid is ≲1% of fit (a1 §6), and changing it breaks 4/4 hierarchical metric configs (a1).
   - Batched np_grouped kernels add `params_` (a long params table; x7: 175x smaller, 0.23 vs 40.7 MB [M]). Their
     `forecasters_` is a lazily materialized view that warns on access, so the cost is paid only on access [E; unbuilt].
5. **`get_fitted_params` `forecasters.loc[...]` keys are unchanged.** Kernels add `params__<col>` plus
   `get_fitted_params(style="table")`.
6. **New tags/config**:
   - `capability:trusted_vectorize` (class tag) and config `trusted_vectorize` (opt-out);
   - `capability:lookback` (default None);
   - the inner mtypes `np_grouped*`, `pa_*`;
   - `output:proba_format`.
   - No `inner_dispatch` global flip and no tier vocabulary (judge_maintainer: tier sprawl).
7. **Extension API:** public `register_mtype` and `register_spoke`, plus an entry-point group. check_estimator gains a
   test that the trusted output equals the stock output, so third-party estimators are covered automatically.

Net: zero visible change for pandas users (except freq on MultiIndex cutoffs, which is documented). Non-pandas users get
consistent types, plus one deprecation cycle for proba.

---

## 4. Phased roadmap (VectorizedDF → BaseForecaster boundary → estimators)

Cumulative impact figures are for **fit+predict of NaiveForecaster/PolynomialTrendForecaster, univariate Panel, daily,
one shared cutoff (U=1)**, the best case for the fh memo (judge_value: "1.6–3.3x" overgeneralizes this). ETS and other
statsmodels-bound forecasters gain about 1.0x under any framework change (g1 [M]).

### P0: decision-independent fixes (≈10 independent PRs, each revertible)

- PR0.1 vectorized polars monotonic check **with null guard** (x9: 921→33 ms, 28x [M]; x9 verifier: 58/75
  mismatches without the guard).
- PR0.2 pass the known mtype from `_check_X_y` into `prepare` / `convert(from_type)` (x4: polars prepare 1443→16 ms
  at 10000x50 [M]; final_bench T3: 2.62 s→0.019 s at 10k [M]).
- PR0.3 `get_cutoff` without a whole-frame `convert_to`.
- PR0.4 `_HashIndex.__eq__/__hash__` (the fh memo hits).
- PR0.5 guard `pd.infer_freq` for <3 points (1–2-row polars update, g5).
- PR0.6 real polars X/y containment check; remove the stale `hasattr(obj,'X')`.
- PR0.7 Panel `predict_residuals` fh fix (fails even for pandas, g5).
- PR0.8 polars egress keeps the y name.
- PR0.9 no caller mutation on pd3 (x2 verifier: conditional on equal-length series).
- PR0.10 `NaiveForecaster.predict_var` variable name; `predict_var(cov=True)` NaN fix; `BaseProbaForecaster._yvec`.

**Impact:** g3 C1 1.05–1.34x e2e (pd 1.05–1.22, polars 1.10–1.34) [M]; fixes ≥7 stock bugs [M g5, a1, g4].
**Exit:** existing test suite green on pd2 and pd3; new regression tests per bug; null-guard equivalence test
against stock on the x9 null fixtures. **Risk:** low.

### P1: VectorizedDF and datatypes substrate (small; no new container)

- PR1.1 public `register_mtype`, entry point, versioned lru caches; test that `cache_clear` keeps registrations.
- PR1.2 `backend_of` registry replacing hasattr duck typing (34 type-dispatch touchpoints, a1).
- PR1.3 `time_extreme(how, per_instance)` and `time_contains` (key, time) anti-join on pandas and polars spokes.
- PR1.4 VectorizedDF `from_type`, outer cutoff over codes, `concat` replacing `droplevel(1)` (logical names for unnamed
  levels).
- PR1.5 **conformance suite as a required CI job**: the g5 240-cell matrix (plus labels, not only values) and the M4
  adversarial fixtures, on pd2 and pd3. Target 0 SILENT cells.

**Impact:** time ~1.0x by design (iteration is 1–2% of fit+predict [M x8/x3]; slicing once vs twice 1.18 vs 1.19
[M g1]). The value is correctness and extensibility. Expected polars strict-equal cells rise from 9/24 toward ≥20/24
[E; g5 baseline M]. **Exit:** conformance suite green; pandas outputs bit-identical. **Risk:** very low to low.

### P2: BaseForecaster boundary (the main lever; 6–7 PRs)

- PR2.1 refactor public fit/predict/update into validate-once + shared `_core` (no behaviour change; full suite).
- PR2.2 trusted per-slice entries + `SliceCtx`; opt-out tag/config; CI runs the forecasting suite **in both modes**
  (the g1 7710-test gate); check_estimator trusted==stock test.
  - **Impact:** g3 C2 cumulative 1.33–2.19x [M]; g1 update 1.3–3.2x, full workload 1.07–1.72x [M].
- PR2.3 fh/cutoff memo per unique cutoff (`FhAbsCache`).
  - **Impact:** g3 C3 cumulative 1.55–3.03x (marginal 1.05–1.53x at U=1) [M]; ragged cutoffs (U≈n) expected near 1.0x
    marginal [E].
- PR2.4 trusted nested BaseTransformer entry (explicit context arg); Lag, Detrender, Deseasonalizer first.
  - **Impact:** Naive fit+predict 2.16–2.48x cumulative, predict-only 1.60–1.93x [M design_minimal §9.1, Lag only,
    n≤1000]; TTF and reducer gains [E].
- PR2.5 proba once: ProbaSpec + long table, known-type proba mixin, one distribution from a parameter table.
  - **Impact:** at 10k series, egress 14–85 ms vs 205 ms concat plus 7.9 s per-instance `_get_columns` [M g4];
    distribution concat 826→3.2 ms at 1k [M g4]; `predict_interval` speed beyond that [E] (the stock interval path is
    dominated by quantiles re-entering public predict, g4).
- PR2.6 edges: return the input container everywhere; top-level X/y anti-join for every scitype; evaluate/update_predict
  convert once (g5 S9).
- PR2.7 `output:proba_format` config with a FutureWarning (non-pandas only).

**Exit:** both-mode suite pass delta 0; per-slice state equality on sampled slices; conformance suite green; benchmark
CI shows ≥1.5x on the Naive/Poly 1000x100 fixture. **Risk:** medium.
- Drift between public and trusted paths (mitigated by one shared core).
- pandas-3 fragility in trusted time code (g3 hit 2 bugs).
- The fh memo gain shrinks with ragged cutoffs.

### P3: estimators (opt-in, one forecaster per PR)

- PR3.1 `np_grouped*` and `pa_*` mtypes + TimeAxis/Step + converters, registered through PR1.1 (g2: 443 LOC framework,
  506 spokes, 130 time ops [M]).
- PR3.2 NaiveForecaster on np_grouped, including intervals (g2: 116 LOC + 70 shared). PR3.3 PolynomialTrendForecaster
  (74 LOC). Both replicate stock quirks (sklearn `cond=tol` truncation, drift slope, frozen window).
- PR3.4 composite tag derivation (TTF, delegators): native only if every component is native.
- PR3.5+ `_BaseWindowForecaster` (fix `_sktime.py:151`, 7–9 classes), then about 5 further closed-form models (Trend,
  Croston, TSB, ForecastKnownValues, seasonal naive; x7 verifier: 6–9 strict candidates). Reducers on numpy views (6–9x,
  g2 [M]) only where the profile shows framework share >50%.
- Keep the old implementation selectable for ≥1 release (no delete-on-port; judge_risk).

**Impact:**
- Ported forecasters: 498–2,735x on pandas input, 498–1,486x on pyarrow at 1000x100 [M g2].
- Polars: 34–105x before P0 [M], expected ≈ pandas-input gain after P0/PR0.2 [E].
- Conversions per 100x100 fit+predict go from ~804 to 2, and mtype checks from 7,484–12,084 to 36 [M g2].
- Native Naive interval about 1,100x [M g4].
- Coverage: about 9/160 leaves plus derived composites.

**Exit:** each port equal to its trusted-lane run on predict, interval, update, absolute fh and ragged data; conformance
suite per backend; M4 fixtures; pd2 and pd3. **Risk:** medium per port (quirk replication; `forecasters_` parity via the
lazy view, unbuilt).

### P4: memory and lazy (only after P2)

- PR4.1 `Selection` pushdown in lazy spokes (instances and columns always; lookback only where declared).
  - **Impact:** selective reads 190–875x faster and 15.6–25x less memory [M final_bench T6]; real pushdown over eager
    select is 1.9x time / 2.8x RSS [M x9 verifier]; full-history ±15% [M g3]; polars lookback pushdown is slower than
    collect-all (0.23–0.64x) and saves nothing beyond projection [M DN-B4, DEL-2].
- PR4.2 positional (codes) pandas-lane iteration and polars→np_grouped→pandas-view ingress (hub C4).
  - **Impact:** peak RSS 1.4–2.2x lower vs C3 (5000x100 Poly polars 121→54 MB) [M g3]; polars ingress +30 vs +451 MB
    at 50k [M final_bench T5b]; time 0.93–1.14x [M].
  - Gate: x6 29/29 and the 33 adversarial fixtures. Uses no private `_simple_new` unless the pd3 matrix is green.
- PR4.3 dask partition-and-delegate behind a G7 break-even study.

**Exit:** fresh-process memory harness confirms the RSS gain; lazy-input conformance. **Risk:** medium (a mis-declared
lookback silently changes forecasts, hence default None and declared-only; lazy row order).

### P5: contract flips (N+2)

- `output:proba_format=auto` gives long format for non-pandas input.
- The batched-kernel `forecasters_` view becomes a documented `params_`-first API.

**Impact:** no speed. **Risk:** downstream code parsing proba MultiIndex columns for non-pandas input (count
unmeasured).

---

## 5. Approach scorecard

| Approach | Verdict | Quantified impact (evidence) | Role in recommendation |
|---|---|---|---|
| Adapter / dispatcher (frame-level, x4) | Adopt narrowly as spokes at the edge only | pandas adapter bit-identical 78/78 + 28/28 at 0.89–1.06x [M x4]; e2e polars 0.96–1.17x [M x4 verifier]; ≤1.05x container cap [M x8]; per-slice dispatch 1.07–1.31x slower [M] | Spoke registry, `time_extreme`/`time_contains`, conformance gate; never inside the loop |
| narwhals | Reject as core; revisit only inside lazy spokes | validation 0.010–0.025x of the stock polars check [M DN-B1]; kernel 3.6–4.2x slower than pandas `groupby.last` even routed to polars [M DN-R]; 7–11x overhead vs hand-written [M x5 verifier]; silent bugs in 2.27.1 (pyarrow over, pandas when/then in agg, duckdb total_seconds) [M x5/x7] | Not a dependency. Optional pinned `stable.v2` select/collect inside lazy spokes after an LOC/overhead experiment |
| Hub / canonical GroupedArray (x6) | Adopt only as the np_grouped inner mtype and an optional ingress path (P4); do not replace VectorizedDF | time 0.93–1.14x [M g3 C4]; RSS 1.4–2.2x lower [M g3]; polars ingress 2.84 s→11 ms at 10k [M T3]; fidelity 25–26/33 adversarial [M x6 verifier] | g2 container = np_grouped; guarded ingress as a memory lever |
| Per-estimator opt-in via inner-mtype tags | Adopt (estimator phase) | g2: 498–2,735x e2e, 315/315 + 233/233 equal, 2 vs ~804 conversions [M]; reducer only 6–9x [M]; ~9 closed-form candidates [M x7 verifier]; top-10 of 96 impls cover 28% [M tiered census] | P3 mechanism; equality-gated, one port per PR |
| Trusted fast lane | Adopt first (core lever) | g1: fit+predict 1.15–1.88x, workload 1.07–1.72x, 7710/7710 tests, 159/160 eligible [M]; with fh memo 1.55–3.03x [M g3 C3]; + nested entry Naive 2.16–2.48x [M design_minimal §9.1] | P2 core, shared with public methods |
| Engine-native panel hook (polars/duckdb kernels) | Reject for core; numpy tier instead | 10²–10³x but only over the per-instance loop; engine over pandas groupby with conversion 1.07–1.33x [M x7 verifier]; numpy 7–11x over pandas groupby [M]; order-robust contract 1.11–1.38x [M DEL-3]; duckdb in-SQL 12–24x lower RSS on series-major parquet [M DEL-2] | Graft the order-robust kernel contract and the params_ table; optional duckdb kernel only for cold series-major data [E] |
| Arrow-backed pandas | Reject as inner representation | e2e 1.07x / 1.19x slower [M design_minimal §9.2]; Arrow survives in 25% of outputs, 7/71 forecasters fail, zero-copy only for single chunks [M x9 verifier]; 1.7x less memory on parquet ingress [M x9] | None (at most an ingress memory option) |
| Lazy-aware boundary | Adopt at the edge only (P4) | selective reads 875x / ~25x memory, 332x with conversion [M T6]; 1.9x / 2.8x over select-before-convert [M x9 verifier]; full-history ±15%, C5 1.1–2.6x slower [M g3]; lookback rare (1/39 safe for intervals) [M DEL-1] | `Selection` + one explicit `collect`; declared lookback, default None |

---

## 6. Final verdicts on the narwhals claims

**C1 ("collect() materializes to a default backend that may not be the input's"): TRUE, but a constraint, not a
blocker.**
- The defaults are measured on pd2 and pd3 and match the source: dask→pandas, duckdb→pyarrow (x1 + verifier .1,
  "confirmed").
- Under `stable.v1`, duckdb and ibis inputs are `DataFrame`s with no `collect` at all, so any adoption must pin
  `stable.v2` (x1 verifier §5).
- Workaround: `collect(backend=...)`, then relift with `.lazy(backend=orig)` (DN-API, measured).
- Staying lazy pays 1.5–3.9x at 500k rows on polars/duckdb/ibis. Memory gain is 1.3–7x, not the 6–35x first reported
  (x1 verifier .8).
- Against eager iteration a lazy per-key loop is about 15x worse, so per-instance estimators must collect once.
- Neither stable API offers `last(order_by=)`.
- Consequence for this design: every materialization is an explicit `collect(backend=…)` in a spoke, never a bare
  `collect()`.

**C2 ("narwhals has no index; MultiIndex panels must be reset_index'ed and names can be lost"): MOSTLY TRUE.**
- narwhals is columns-only; `reset_index` is required.
- Names are lost only on a naive path. The x2 schema recipe restores names, dtypes, freq, Period and order in 198/198
  checks (18 per pandas version × 11), but **only for the tested input class**. The x2 verifier rates it "partially
  confirmed": it fails on mixed-dtype frames and per-series freq.
- The x5 narwhals VectorizedDF matches stock 17/17 on pandas.
- Cost: narwhals iteration on pandas is 1.69–2.26x slower than stock on pd3 once the asymmetric `droplevel` accounting is
  removed (x2 verifier). Validation on pandas-long is 70–620x slower than the stock pd-multiindex check (DN-B1).
- Consequence: the pandas MultiIndex lane must stay native. This alone forces a dual path if narwhals were adopted, which
  is a main reason it is rejected as the core.

---

## 7. Rejected options and why

- **narwhals as the core boundary/kernels.** It is not a speed lever (it removes at most the ≤1.18x non-pandas penalty,
  T4). It is slower than plain pandas groupby for kernels (DN-R). It has silent bugs in the tested version, needs a dual
  pandas path (C2), and adds a pinned core dependency: it is a transitive dependency only for sklearn 1.9.x users, since
  sktime allows sklearn ≥0.24,<1.10 (r1).
- **Hub replacing VectorizedDF.** VectorizedDF is shared with transformers, classifiers and metrics. The float64 hub
  loses the NaN/null distinction and int64>2^53 precision (25–26/33). The fast pandas view needs private `_simple_new`,
  which pandas 3 broke twice (g3). g5 H1: the hub round trip broke pandas update_predict. Its time gain is ≈1.0x.
- **An adapter registry as a first phase.** It is worth ≈1.0x (x4/x8) yet would delay the g1 lever, which needs no
  substrate. It is kept, narrowed, as spokes.
- **Engine-native per-engine kernels (polars expressions + duckdb SQL).** A second estimator API × 3 engines × K
  forecasters × 4 methods. Only 6–8 strict candidates. For pandas users the engine adds 1.07–1.33x over groupby.
- **Arrow-backed pandas internals:** slower e2e and do not survive the forecasters.
- **Lookback inference / default pushdown:** silently wrong forecasts (DEL-1); polars lookback saves nothing beyond
  projection (DEL-2).
- **Deprecating `forecasters_` / `.loc` fitted-param keys for all users:** ≲1% speed value, breaks 4/4 metric configs.
- **Tiered's global `inner_dispatch=auto` flip and delete-on-port:** non-revertible routing change; removes the
  rollback and breaks external subclasses.
- **Module-level TRUSTED ContextVar:** implicit state that does not reach loky workers. Replaced by an explicit
  context argument.
- **dask partition-and-delegate in v1:** 0.70–1.31x on one 4-core box (DEL-4). Deferred to a G7 study.

---

## 8. Risks

1. **Public/trusted drift.** g1 copied about 700 LOC of stock logic. Mitigation: one shared `_core`, both-mode CI,
   check_estimator equality test.
2. **pandas-3 fragility in time code** (g3: Day vs 24h, datetime unit; x2: caller mutation). Mitigation: pd2×pd3 CI for
   the trusted and np_grouped paths; Step fallback to pandas offsets.
3. **Headline overgeneralization.** 1.55–3.03x / 2.16–2.48x were measured on Naive/Poly fit+predict with U=1.
   Statsmodels-bound forecasters and predict_interval are about 1.0x (g1); ragged cutoffs weaken the fh memo [E].
4. **Kernel ports must replicate stock quirks** (sklearn 1.9.1 `cond=tol` truncation means stock Poly is not exact least
   squares; drift slope; frozen window). Fixing the quirks later is a behaviour change.
5. **Batched kernels lack a real per-instance `forecasters_`.** The lazy-view mitigation is unbuilt.
6. **Per-backend silent divergence at the edge** (x4 had 3 such bugs; g5 A1/A2). Mitigation: conformance suite as a
   required gate before any non-pandas path merges.
7. **Composite tag derivation is unprototyped.** Composites stayed silently on pandas under x4 (g5 inv 8).
8. **Downstream use of proba MultiIndex columns by non-pandas users** (N+2 flip) is unmeasured.
9. **The nested-transformer gain is measured only for Lag in Naive** (n≤1000, 2 reps, Lag state not compared).

---

## 9. Remaining experiments (ranked)

1. Generic trusted nested-transformer entry across TTF[Detrender/Deseasonalizer], reducers and STLForecaster, with the
   g1 test suite in both modes and Lag-state equality; at 1k–5k series.
2. Known-type proba mixin: predict_interval/quantiles speed, which is ≈1.0x under g1 today.
3. fh memo with ragged cutoffs (U = n/10, U = n) under the trusted lane.
4. The full minimal stack (P0+P2) on pandas 3.0.6 with equality; g2/g3 timings under pd3.
5. Conformance suite run (g5 240 cells + M4 + label checks) against PR1.3/PR2.6 spokes. Target ≥20/24 polars
   strict-equal cells and 0 SILENT.
6. Re-run g2 on polars input with PR0.1/PR0.2 to confirm polars ≈ pandas-input gain.
7. pandas-groupby kernel vs np_grouped port for Naive/Poly with guard cost (G13), to confirm the numpy-tier default.
8. Lazy `forecasters_` view for np_grouped ports: access cost at 1k/10k and the a1 grid and metric tests.
9. Composite tag derivation: does TTF([Detrender, NaiveNp]) stay in np_grouped end to end?
10. Fresh-process memory of P2 vs P2+PR4.2 at 50k series (to size the memory lever).
11. Lookback census in a full-dependency env, and cold/remote parquet pushdown (G9/G14).
12. narwhals-in-lazy-spoke hybrid: LOC saved vs per-call overhead under `stable.v2`.
13. dask partition-and-delegate break-even on a multi-worker cluster (G7).

---

## 10. Sources

`reports/`: g1_trusted_path, g2_numpy_inner, g3_stack, g4_proba, g5_api_surface, design_minimal (§9.1, §9.2),
design_tiered, design_adapter, design_hub (§9), design_narwhals (DN-B1..B4, DN-R, DN-API), design_engine_lazy (DEL-1..4),
judge_maintainer, judge_value, judge_risk, wrap_critic, final_bench (T1–T6), x1/x2/x4/x5/x6/x7/x8/x9 and their verifiers,
a1/a2/a3, r1. Structured results: `results/investigation_digest.json`, `results/final_bench.json`,
`results/design_minimal_nested.json`.
