export const meta = {
  name: 'pandas-decoupling-design',
  description: 'Fill high-priority evidence gaps, then 6 independent architecture proposals, 3 judges, synthesis + red-team revision',
  phases: [
    { title: 'Gaps', detail: 'trusted fast path, numpy inner mtype, proba path, API surface, combined waterfall' },
    { title: 'Propose', detail: '6 independent architects from different angles' },
    { title: 'Judge', detail: 'maintainer/back-compat, realized value, risk/incrementality' },
    { title: 'Synthesize', detail: 'recommendation + phased roadmap, red-team, revise' },
  ],
}

// args: { wt, sp }
const WT = args.wt
const SP = args.sp
const ED = WT + '/experiments/container_independence'

const CTX = `
CONTEXT (shared by all agents)
Repo: sktime at WT=${WT}. Project: decouple sktime forecasters + the vectorization layer (sktime/datatypes/_vectorize.py VectorizedDF, now stateless: (frame, schema) tuples; sktime/forecasting/base/_base.py BaseForecaster) from pandas as the sole data container, so polars (eager+lazy), pyarrow, dask, duckdb are used natively and their value (speed, laziness, memory, pushdown) is realized.
A 34-agent investigation already ran. Its evidence lives in ${ED}:
 - reports/<key>.md and reports/<key>_verify0.md (adversarial verification) for: a1_core_coupling, a2_census, a3_time_index, r1_sklearn_skrub, r2_narwhals, r3_ts_libs, r4_dispatch_arch, x1_nw_lazy_collect, x2_nw_index_fidelity, x3_iter_bench (verify only), x4_proto_adapter, x5_proto_narwhals, x6_proto_hub, x7_native_engine, x8_e2e_profile, x9_cheap_paths
 - reports/wrap_critic.md (gaps G1..G15 + contradictions K1..K15), reports/final_bench.md + results/final_bench.json (serial low-noise benchmarks)
 - results/investigation_digest.json (all structured results + verdicts)
 - protos/ (prototypes: x4_adapter, x5_narwhals, x6_hub, x7_hook, a1_estimator_grid, a3_timeindex, ...), scripts/, common.py (data generators + bench helpers)
Settled headline facts (verified): container-only changes under unchanged per-instance BaseForecaster calls are capped at ~1.05x end to end; per-slice public-API re-entry (check/convert/fh/cutoff/clone) is the biggest bucket (33-63% of vectorized fit); panel-native kernels give 10^2-10^3x but only for simple/closed-form forecasters; the engine itself adds 1.07-3x once conversion is charged; lazy pushdown gives ~875x / 25x memory on selective reads; no leaf forecaster computes in anything but pandas; 29 forecasters are pd-multiindex-native and bypass VectorizedDF; public API returns pandas types (forecasters_, cutoff, fh, proba MultiIndex columns). Verified narwhals facts: C1 true (collect default: dask->pandas, duckdb->pyarrow; workaround collect(backend=...) or stay lazy); C2 mostly true (columns-only; reset_index needed; names lost only in a naive path; a recipe restores them).
Python envs (NEVER pip install into them): EXP_PY=${SP}/expenv/bin/python (pandas 2.3.3, polars 2.0.0, narwhals 2.27.1, pyarrow, dask, duckdb, statsmodels? check), EXP_PY3=${SP}/expenv_pd3/bin/python (pandas 3.0.6). Run from WT: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore <script>. Own extra libs: uv venv ${SP}/venvs/<key>.
Rules: do NOT modify WT/sktime/ (prototypes in ED/protos/). No state-changing git commands. Prefix files with your key. Machine: 4 cores, 15GB RAM, ~2 agents concurrently: keep processes < 2GB; prefer interleaved same-process relative timings (bench() from common.py) and report ratios. Every quantitative claim must come from a script you ran (record command + output) or cite a report; label others "unmeasured/expected".
`

const RESULT = {
  type: 'object',
  properties: {
    key: { type: 'string' },
    summary: { type: 'string', description: '<= 250 words' },
    report_path: { type: 'string' },
    impact_statements: { type: 'array', items: { type: 'string' } },
    caveats: { type: 'array', items: { type: 'string' } },
  },
  required: ['key', 'summary', 'report_path', 'impact_statements', 'caveats'],
}

const GAPS = [
  { key: 'g1_trusted_path', prompt: `
GAP g1_trusted_path (critic G1, HIGH): prototype the framework "fast lane": when VectorizedDF (or BaseForecaster._vectorize) hands already-validated slices to per-instance estimators, skip public-API re-entry (check_is_scitype, convert_to re-dispatch, _check_fh, cutoff re-parsing) and call clone + _fit/_predict (and _update, _predict_interval/_quantiles/_var) through a trusted internal entry that sets _cutoff/_fh/_y/_X/_is_fitted from the schema.
Implement in protos/g1_trusted.py (subclass/plugin, no edits to sktime/). Read reports/a1_core_coupling*.md and x8_e2e_profile*.md first (a1's lite loop and x8's lite loop exist).
Measure (scripts/g1_*.py --grid {dev,large}) vs stock for NaiveForecaster(last), PolynomialTrendForecaster, ExponentialSmoothing/ThetaForecaster if statsmodels available, make_reduction(LinearRegression), TransformedTargetForecaster([Deseasonalizer or Detrender, Naive]) on pandas and polars Panel/Hierarchical: fit, predict, update, predict_interval. Assert equality of predictions, intervals, cutoff. Run sktime's forecasting base tests (sktime/forecasting/base/tests, a reasonable subset with pytest -x -q, timebox 20 min) with the trusted path monkeypatched in, report pass delta. Identify which invariants the trusted path must preserve (e.g. estimators that override fit/predict publicly, tags like fit_is_empty, requires-fh-in-fit, X handling) and an opt-out tag. Write reports/g1_trusted_path.md.` },
  { key: 'g2_numpy_inner', prompt: `
GAP g2_numpy_inner (critic G2, HIGH): a per-estimator opt-in "numpy values + time axis" inner mtype integrated with BaseForecaster.
Read reports/a2_census*.md, a3_time_index*.md (Step/TimeIndex abstraction), x6_proto_hub*.md, protos/x6_hub.py, protos/a3_timeindex.py.
Prototype in protos/g2_*.py a proto mtype (e.g. "np_grouped": 2D float values, indptr, keys table, time axis via a3 Step, column names) with converters from/to pandas MI, polars, pyarrow; make BaseForecaster route y to it when a forecaster declares it as y_inner_mtype (monkeypatch/registration in proto only — check whether sktime's mtype registry can be extended from outside; report how). Port 3 representative forecasters: NaiveForecaster (last/mean/drift) as a batched panel implementation, PolynomialTrendForecaster, and one statsmodels-based or reducer one per-series on views. Measure e2e vs stock on pandas, polars, pyarrow input; count conversions; LOC changed per forecaster; equality incl. predict_interval where supported; how cutoff/fh/output index are produced without pandas (or with pandas only on the small index). Write reports/g2_numpy_inner.md.` },
  { key: 'g4_proba', prompt: `
GAP g4_proba (critic G4, HIGH): probabilistic outputs and the second vectorization path.
Map sktime/forecasting/base/_base_proba.py (_vectorize_predict_proba, _concat_distributions_hierarchical) and sktime/forecasting/base/_proba/_mixin.py (interval/quantile/var defaults) pandas touchpoints (file:line). For prototypes x4 adapter, x5 narwhals, x6 hub, x7 hook (protos/): what does predict_interval/predict_quantiles/predict_var need from each (a column-level op? a long format?). Prototype (protos/g4_*.py) (a) a long/tidy proba schema (instance keys, time, variable, coverage/alpha, lower/upper or value) that any backend can hold, (b) conversion to today's pandas MultiIndex columns; measure cost at 1k/10k series for pandas and polars, and correctness vs stock NaiveForecaster/ThetaForecaster outputs. Also note distribution (predict_proba) concat requirements. Recommend the public-API contract for non-pandas proba outputs with a deprecation path. Write reports/g4_proba.md.` },
  { key: 'g5_api_surface', prompt: `
GAP g5_api_surface (critic G5): API surface beyond y-only fit+predict.
Matrix: {current sktime, x4 adapter proto, x6 hub proto (where wired)} x {fit(y,X) with exogenous, update(y_new,X_new), update_predict, predict_residuals, evaluate() with ExpandingWindowSplitter, TransformedTargetForecaster([Deseasonalizer, Naive]), ForecastingGridSearchCV(Naive, strategy grid), ReconcilerForecaster on hierarchical} x {pandas, polars}. Record pass/fail, output types, equality vs pandas reference, and root causes of failures (file:line). Include known bugs from a1 (polars X/y index check passing on disjoint dates; column named "X"; update_predict/predict_residuals polars failures) and x4's cutoff reverse_order issue. Write reports/g5_api_surface.md with the matrix and a list of boundary invariants any new architecture must keep.` },
]

const G3 = { key: 'g3_stack', prompt: `
GAP g3_stack (critic G3, HIGH): combined-architecture waterfall. Build cumulative configurations and measure end to end:
 C0 stock sktime; C1 + known-mtype pass-through / skip redundant mtype checks (x4/x9 monotonic + check findings); C2 + trusted per-slice path (protos/g1_trusted.py from g1); C3 + time axis once per panel and vectorized cutoff (a1 P10, a3 Step); C4 + hub/positional iteration (x6) feeding a trusted pandas view to unchanged forecasters; C5 + panel-native kernel where available (x7 hook / g2 batched Naive).
Run for NaiveForecaster(last) and PolynomialTrendForecaster on pandas, polars, polars-lazy input at dev and large grids (e.g. 100x100, 1000x100, 5000x100, 1000x500), equality checked at each step. Read reports/g1_trusted_path.md and g2_numpy_inner.md and reuse their protos. Report the waterfall of marginal gains (ratios vs C0 and vs previous step), peak RSS, and which steps are container-dependent vs container-independent. Write results/g3_stack.json + reports/g3_stack.md.` }

phase('Gaps')
const gapRes = (await parallel(GAPS.map((g) => () =>
  agent(CTX + '\n' + g.prompt, { label: g.key, phase: 'Gaps', schema: RESULT })))).filter(Boolean)
const g3 = await agent(CTX + '\n' + G3.prompt, { label: G3.key, phase: 'Gaps', schema: RESULT })
const gaps = [...gapRes, g3].filter(Boolean)
log(`gap experiments done: ${gaps.length}/${GAPS.length + 1}`)
const gapDigest = JSON.stringify(gaps.map((g) => ({ key: g.key, report: g.report_path, summary: g.summary, impact: g.impact_statements, caveats: g.caveats })), null, 1)

const EVID = `
EVIDENCE: read reports/wrap_critic.md, reports/final_bench.md, and skim results/investigation_digest.json; dig into individual reports as your design needs. New gap experiments (also read their reports):
${gapDigest}
`

const PROPOSAL = {
  type: 'object',
  properties: {
    angle: { type: 'string' },
    title: { type: 'string' },
    thesis: { type: 'string', description: '<= 120 words' },
    architecture: { type: 'string', description: 'components, interfaces/protocols with signatures, where pandas remains, data flow for pandas/polars/lazy/dask input' },
    public_api: { type: 'string', description: 'what changes for users: return types, cutoff/fh, forecasters_, proba outputs, deprecations' },
    estimator_migration: { type: 'string', description: 'what happens to the ~160 forecasters; tags; who must change what' },
    roadmap: { type: 'array', items: { type: 'object', properties: { phase: { type: 'string' }, scope: { type: 'string' }, expected_impact: { type: 'string', description: 'quantified, cite evidence ids' }, risk: { type: 'string' } }, required: ['phase', 'scope', 'expected_impact', 'risk'] } },
    evidence_for: { type: 'array', items: { type: 'string' } },
    evidence_against: { type: 'array', items: { type: 'string' } },
    open_experiments: { type: 'array', items: { type: 'string' } },
    report_path: { type: 'string' },
  },
  required: ['angle', 'title', 'thesis', 'architecture', 'public_api', 'estimator_migration', 'roadmap', 'evidence_for', 'evidence_against', 'open_experiments', 'report_path'],
}

const ANGLES = [
  ['adapter', 'ADAPTER-FIRST: a narrow ContainerAdapter protocol + registry (sklearn set_output / skrub style) for VectorizedDF and the BaseForecaster boundary; pandas adapter behaviour-identical.'],
  ['narwhals', 'NARWHALS-FIRST: narwhals as the abstraction for the boundary and panel-native forecasters; take C1/C2 evidence seriously and design around it.'],
  ['hub', 'HUB / CANONICAL: an internal canonical panel representation (GroupedArray-like values+indptr+keys+time axis, Nixtla/Darts style) with spokes per backend; forecasters receive views or lazily materialized pandas.'],
  ['tiered', 'TIERED PER-ESTIMATOR OPT-IN: keep the inner-mtype tag system as the mechanism; add new inner mtypes (numpy grouped, polars) plus a trusted fast lane; estimators migrate one by one (sklearn Array API style).'],
  ['engine_lazy', 'ENGINE-NATIVE / LAZY-FIRST: lazy-aware boundary with selection pushdown, engine-native panel hooks for forecasters that can express themselves in expressions/SQL, delegation to dask/duckdb partitions.'],
  ['minimal', 'MINIMAL-CHANGE BASELINE: keep pandas internally, fix the framework overheads (trusted path, cutoff, checks), Arrow-backed pandas where accepted, convert at the edges and return the input container; argue what is NOT worth doing.'],
]

phase('Propose')
const proposals = (await parallel(ANGLES.map(([k, a]) => () => agent(`${CTX}
${EVID}
You are an independent ARCHITECT. Your assigned angle: ${a}
Design the BEST architecture you can from this angle (you may borrow from others where the evidence demands, but stay recognizably on your angle). Be concrete: protocols with method signatures, where each pandas touchpoint category (a1: container ops, time/freq/cutoff, metadata, output construction, public-API value types) goes, how the 4 public-API contracts (forecasters_, cutoff/fh, proba MultiIndex columns, get_fitted_params keys) evolve with deprecation, how lazy and distributed inputs are handled, and how the ~160 forecasters (a2 buckets) are affected. Give a phased roadmap VectorizedDF -> BaseForecaster boundary -> estimators with QUANTIFIED expected impact per phase citing evidence (report key + claim). List the strongest evidence against your own design honestly. Write ${ED}/reports/design_${k}.md.`,
  { label: `architect:${k}`, phase: 'Propose', schema: PROPOSAL })))).filter(Boolean)
log(`proposals: ${proposals.length}/${ANGLES.length}`)

const SCORE = {
  type: 'object',
  properties: {
    lens: { type: 'string' },
    scores: { type: 'array', items: { type: 'object', properties: { title: { type: 'string' }, score: { type: 'number', description: '0-10' }, strengths: { type: 'string' }, weaknesses: { type: 'string' }, factual_errors: { type: 'string' } }, required: ['title', 'score', 'strengths', 'weaknesses', 'factual_errors'] } },
    best_ideas_to_graft: { type: 'array', items: { type: 'string' } },
    ranking_rationale: { type: 'string' },
    report_path: { type: 'string' },
  },
  required: ['lens', 'scores', 'best_ideas_to_graft', 'ranking_rationale', 'report_path'],
}
const LENSES = [
  ['maintainer', 'SKTIME CORE MAINTAINER: back-compatibility and deprecation policy, extension-contract simplicity for third-party estimator authors (sktime has many external contributors and 160 forecasters + transformers/classifiers sharing the datatypes layer), test-framework impact (check_estimator), dependency footprint (soft deps), consistency with skbase / other scitypes.'],
  ['value', 'REALIZED PERFORMANCE VALUE: does the design actually deliver the value of polars/lazy/dask (speed, memory, pushdown) for real users at realistic sizes, based on the measured evidence (final_bench, g3 waterfall, x7, x9)? Penalize designs whose claimed gains are capped by unchanged costs; check every number against the reports.'],
  ['risk', 'RISK & INCREMENTALITY: can it ship in small reviewable PRs, each independently valuable and revertible? Correctness risk (time semantics, index fidelity, silent bugs found in a1/g5), reliance on third-party moving targets (narwhals stable API, polars versions), effort estimate realism.'],
]
const propText = JSON.stringify(proposals, null, 1)

phase('Judge')
const judgements = (await parallel(LENSES.map(([k, l]) => () => agent(`${CTX}
${EVID}
You are a JUDGE with lens: ${l}
Score each of these ${proposals.length} architecture proposals 0-10 under your lens (full proposals also at reports/design_*.md). Verify their evidence citations against the reports and flag factual errors. Name the best ideas to graft across proposals. Write ${ED}/reports/judge_${k}.md.
PROPOSALS:
${propText}`, { label: `judge:${k}`, phase: 'Judge', schema: SCORE })))).filter(Boolean)

const FINAL = {
  type: 'object',
  properties: {
    recommendation: { type: 'string', description: '<= 200 words, the decision' },
    architecture: { type: 'string' },
    roadmap: { type: 'array', items: { type: 'object', properties: { phase: { type: 'string' }, scope: { type: 'string' }, prs: { type: 'string' }, expected_impact: { type: 'string' }, exit_criteria: { type: 'string' }, risk: { type: 'string' } }, required: ['phase', 'scope', 'prs', 'expected_impact', 'exit_criteria', 'risk'] } },
    approach_scorecard: { type: 'array', items: { type: 'object', properties: { approach: { type: 'string' }, verdict: { type: 'string' }, quantified_impact: { type: 'string' }, role_in_recommendation: { type: 'string' } }, required: ['approach', 'verdict', 'quantified_impact', 'role_in_recommendation'] } },
    narwhals_claims: { type: 'string', description: 'final verdict on C1 and C2 with numbers' },
    public_api_decisions: { type: 'array', items: { type: 'string' } },
    rejected_and_why: { type: 'array', items: { type: 'string' } },
    remaining_experiments: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
    report_path: { type: 'string' },
  },
  required: ['recommendation', 'architecture', 'roadmap', 'approach_scorecard', 'narwhals_claims', 'public_api_decisions', 'rejected_and_why', 'remaining_experiments', 'risks', 'report_path'],
}

phase('Synthesize')
const synthPrompt = (extra) => `${CTX}
${EVID}
You are the SYNTHESIZER. Produce ONE recommended architecture for decoupling sktime forecasters + VectorizedDF from pandas, starting from the highest-scoring proposal and grafting the best ideas from the others, and a phased roadmap (VectorizedDF -> BaseForecaster boundary -> estimators) of concrete PR-sized steps with expected quantified impact (citing evidence), exit criteria and risk. Include an approach scorecard covering: adapter/dispatcher, narwhals, hub/canonical GroupedArray, per-estimator opt-in via inner-mtype tags, trusted fast lane, engine-native panel hook, Arrow-backed pandas, lazy-aware boundary — each with a quantified impact statement from the evidence. Give final verdicts on narwhals claims C1 and C2. Be explicit about which numbers are measured vs expected. Audience: sktime core developers.
PROPOSALS:
${propText}
JUDGEMENTS:
${JSON.stringify(judgements, null, 1)}
${extra}`

const draft = await agent(synthPrompt(`Write ${ED}/reports/design_synthesis_draft.md.`), { label: 'synthesizer', phase: 'Synthesize', schema: FINAL })

const RED = { type: 'object', properties: { issues: { type: 'array', items: { type: 'object', properties: { severity: { type: 'string', enum: ['blocker', 'major', 'minor'] }, issue: { type: 'string' }, fix: { type: 'string' } }, required: ['severity', 'issue', 'fix'] } }, report_path: { type: 'string' } }, required: ['issues', 'report_path'] }
const red = await agent(`${CTX}
${EVID}
You are a RED TEAM reviewer (skeptical senior sktime maintainer + performance engineer). Attack this draft recommendation: wrong or unsupported numbers (check against reports/results), hidden back-compat breaks, steps that cannot ship independently, missing failure modes (time semantics, index fidelity, proba, update/evaluate, composites, soft deps, third-party estimators), gains that are double-counted. Re-run a key script if a headline number looks suspicious. Write ${ED}/reports/design_redteam.md.
DRAFT:
${JSON.stringify(draft, null, 1)}`, { label: 'red-team', phase: 'Synthesize', schema: RED })

const final = await agent(synthPrompt(`Your earlier draft:
${JSON.stringify(draft, null, 1)}
Red-team issues to address (fix each or explain why not, in a "Review response" section):
${JSON.stringify(red, null, 1)}
Write the FINAL report to ${ED}/reports/design_final.md (self-contained, readable by sktime devs, with tables), and return it structured.`), { label: 'synthesizer:final', phase: 'Synthesize', schema: FINAL })

return { gaps, proposals, judgements, draft, red, final }
