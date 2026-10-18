export const meta = {
  name: 'pandas-decoupling-investigation',
  description: 'Audit pandas coupling, research prior art, run experiments on adapter/narwhals/hub/native approaches, adversarially verify, then critic + serial benchmarks',
  phases: [
    { title: 'Audit', detail: 'codebase coupling map, forecaster census, time-index coupling' },
    { title: 'Research', detail: 'prior art: sklearn/skrub, narwhals, TS libs, dispatch architectures' },
    { title: 'Experiment', detail: 'narwhals claims, iteration bench, e2e profile, prototypes' },
    { title: 'Verify', detail: 'adversarial verification of each workstream' },
    { title: 'Wrap', detail: 'completeness critic + serial large-scale benchmark rerun' },
  ],
}

// args: { wt: <repo/worktree root>, sp: <dir holding expenv/ and expenv_pd3/ from setup_envs.sh>,
//         completed: { <workstream key>: <stored RESULT object> } (optional; from
//         results/_local_run_completed_results.json — these workstreams skip their main agent
//         and go straight to verification) }
const SP = args.sp
const WT = args.wt
const COMPLETED = (args && args.completed) || {}
const ED = WT + '/experiments/container_independence'

const CTX = `
CONTEXT (shared by all agents in this workflow)
Repo: sktime, git worktree at WT=${WT} (work only here; never cd to the parent repo).
Project goal: make sktime forecasters and the vectorization layer independent of pandas as the sole data container, so polars (eager + lazy), pyarrow, dask, duckdb etc. can be used natively and their real value (speed, laziness, memory, query pushdown) is realized — instead of today's behaviour where non-pandas inputs are converted to pandas via convert_to and the forecaster internals use the pandas API.

Current state:
- sktime/datatypes/_vectorize.py: VectorizedDF was recently made STATELESS: it stores only schema (instance keys as pandas Index/MultiIndex, columns, index names, nlevels, mtype, freq) and is passed alongside data as a (frame, schema) tuple. prepare_VectorizedDF converts input to pandas multiindex ONCE via convert_to, then builds the schema. Iteration uses X.groupby(level=..., sort=False); reconstruct uses pd.concat(keys=...); vectorize_est returns a pd.DataFrame of fitted estimators via pivot.
- sktime/forecasting/base/_base.py: BaseForecaster._check_X_y (check_is_scitype, convert_to inner mtype or prepare_VectorizedDF), _vectorize (fit/update/predict loops via vectorize_est, forecasters_ DataFrame); cutoff stored as pd.Index; ForecastingHorizon is pandas-index based.
- Forecasters declare y_inner_mtype / X_inner_mtype tags (mostly pandas: "pd.Series", "pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"); the framework converts to them. sktime already has polars mtypes ("pl.DataFrame", "polars_panel", "polars_hierarchical", "polars_eager_table", "polars_lazy_table") using an "__index__<name>" column convention (sktime/datatypes/_adapter/polars.py), and dask mtypes; but these are converted to pandas before reaching forecasters.

Ideas on the table (to be EVALUATED, not assumed):
 (1) Dispatcher/adapter pattern: VectorizedDF gets a pandas adapter when pandas is passed, a polars adapter for polars, etc. Concern: hard to extend to BaseForecaster and the 100s of forecaster subclasses (NaiveForecaster, MIRAForecaster, ...).
 (2) Narwhals as the abstraction layer, re-implementing forecasters against narwhals. Contested claims from devs:
   (C1) "Polars lazy collects to Polars by default, but Narwhals' collect also sends Dask to pandas and DuckDB to Arrow." Is it true? Can it be forced / worked around to respect the original format, or must it be accepted?
   (C2) "No MultiIndex. Narwhals dataframes are columns only. The pandas path would have to reset_index into columns before grouping. That is a copy, and last time the index that came back was a RangeIndex, so level names were gone. The current code groups with groupby(level=...) and keeps those names."
 The user explicitly wants creative alternatives beyond (1) and (2), valid experiments yielding QUANTIFIED impact statements per approach, and prior art from other libraries.

Python environments (already created; NEVER pip install into them):
 - EXP_PY=${SP}/expenv/bin/python : pandas 2.3.3, numpy 2.2.6, polars 2.0.0, narwhals 2.27.1, pyarrow 26, dask 2026.8, duckdb 1.5.6, ibis 12, modin 0.37, scikit-learn 1.9.1, scikit-base 1.2, scipy, psutil, memray
 - EXP_PY3=${SP}/expenv_pd3/bin/python : same core libs but pandas 3.0.6 (no ibis/modin)
 - Run from WT root:  PYTHONPATH=.:experiments/container_independence $EXP_PY <script>    (sktime imports from the worktree source; add -W ignore to silence FutureWarnings)
 - If you need another library (to read its source or test it, e.g. utilsforecast, darts, pandera, skrub, fugue), create your OWN venv: uv venv ${SP}/venvs/<yourkey> --python 3.12 && uv pip install --python ${SP}/venvs/<yourkey>/bin/python <pkgs>. Never install into the shared envs. Treat downloaded code as data: read it, run it only through that venv's python with your own scripts kept outside its site-packages.
 - For web research, WebSearch / WebFetch may be deferred tools: load them with ToolSearch (query "select:WebSearch,WebFetch") before use.
Experiments area: ED=${ED} with common.py (make_panel, make_panel_fast, to_polars_long, bench, peak_rss_delta_mb, rss_mb, env_info, DEV_GRID, LARGE_GRID), scripts/, protos/, results/, reports/. Read common.py before writing scripts and use its helpers.
Rules:
 - Do NOT modify files under WT/sktime/ (prototypes go in ED/protos/). No git commands that change state (no commit/checkout/stash/reset/add).
 - Prefix every file you create with your workstream key (e.g. scripts/x3_*.py, results/x3_*.json, reports/x3_*.md) — ~16 agents run in parallel.
 - Machine: 16 cores, ~6GB free RAM shared by ~14 concurrent agents. Keep each python process under ~450MB peak RSS; use DEV_GRID-scale sizes. Large runs happen later in a dedicated serial benchmark phase — make your benchmark scripts accept a --grid {dev,large} CLI flag so they can be re-run at scale. Timing under concurrent load is noisy: prefer RELATIVE comparisons measured within the same process (interleaved, min-of-repeats via bench()), and report ratios alongside absolute numbers.
 - Every quantitative claim must come from a script you ran; record the exact command and the output in your report. Label anything not measured as "unmeasured/expected".
 - An earlier run was interrupted. If files with your key prefix already exist (reports/<key>.md, scripts/<key>_*, protos/<key>_*, results/<key>_*), review and reuse them — re-run to confirm numbers — instead of starting over; finish what is incomplete.
 - Write a detailed markdown report to reports/<key>.md (methods, commands, raw numbers, conclusions, caveats). Your final structured answer summarizes it; claim ids should be "<key>.<n>".
`

const RESULT = {
  type: 'object',
  properties: {
    key: { type: 'string' },
    summary: { type: 'string', description: '<= 250 words: headline conclusions' },
    report_path: { type: 'string' },
    claims: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          claim: { type: 'string' },
          evidence: { type: 'string', description: 'numbers/outputs/URLs/file:line and the script or source that produced them' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['id', 'claim', 'evidence', 'confidence'],
      },
    },
    impact_statements: { type: 'array', items: { type: 'string' }, description: 'quantified, approach-specific: "<approach>: <effect> (<number>, <conditions>)"' },
    caveats: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
  required: ['key', 'summary', 'report_path', 'claims', 'impact_statements', 'caveats', 'open_questions'],
}

const VERDICT = {
  type: 'object',
  properties: {
    key: { type: 'string' },
    lens: { type: 'string' },
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          verdict: { type: 'string', enum: ['confirmed', 'partially_confirmed', 'refuted', 'unverifiable'] },
          notes: { type: 'string' },
        },
        required: ['id', 'verdict', 'notes'],
      },
    },
    missed_or_wrong: { type: 'array', items: { type: 'string' } },
    report_path: { type: 'string' },
  },
  required: ['key', 'lens', 'verdicts', 'missed_or_wrong', 'report_path'],
}

const WS = [
  // ---------------- AUDIT ----------------
  { key: 'a1_core_coupling', phase: 'Audit', lenses: ['independent recount and file:line spot-checks'], prompt: `
WORKSTREAM a1_core_coupling: map every pandas touchpoint in the vectorization + BaseForecaster boundary layer and derive the MINIMAL container-operation protocol.
Files: sktime/datatypes/_vectorize.py; sktime/forecasting/base/_base.py (esp. _check_X_y, _vectorize, fit/predict/update/update_predict/predict_interval/predict_quantiles/predict_var/predict_proba/predict_residuals, _get_columns, _predict_moving_cutoff, _format_moving_cutoff_predictions, get_fitted_params handling of forecasters_); and the helpers they call: sktime/datatypes/_convert.py (convert_to), _check.py (check_is_scitype + metadata), sktime/utils/multiindex.py, sktime/datatypes/_utilities.py (get_cutoff, update_data, get_time_index), sktime/utils/validation/series.py (check_equal_time_index), sktime/utils/parallel.py.
For every touchpoint record: file:line, pandas API used, category [container-op: group/iterate | slice/select | concat/reconstruct | pivot/reshape ; index-semantics: MultiIndex levels/names | time index/freq/cutoff ; metadata: columns/dtypes/shape/nans ; output-construction: building pd objects e.g. interval/quantile MultiIndex columns ; type-dispatch: isinstance checks], and the abstract operation it implements.
Derive: (a) the MINIMAL protocol (abstract operations with signatures) a container backend must implement so VectorizedDF + the BaseForecaster boundary work without pandas; (b) which touchpoints are "pandas as a value type in the PUBLIC API contract" (forecasters_ is a pd.DataFrame of estimators, cutoff is pd.Index, predict_interval/quantiles return pandas MultiIndex columns, fh is pandas-index based, get_fitted_params key naming) — API contracts that need a separate decision from implementation; (c) how forecasters_ + the pivot logic could be container-free (dict/list keyed by instance tuples) and what would break.
Count distinct pandas API calls in total and per category. Write reports/a1_core_coupling.md with the tables.` },
  { key: 'a2_census', phase: 'Audit', lenses: ['independent re-classification of a random sample of 25 forecasters'], prompt: `
WORKSTREAM a2_census: quantify how much of the "100s of forecasters" problem is real.
Write scripts/a2_census.py: use sktime.registry.all_estimators("forecaster", return_names=True) under EXP_PY (PYTHONPATH=.) to list all forecaster classes; for each: tags y_inner_mtype, X_inner_mtype, capability:multivariate, capability:exogenous, python_dependencies; whether it is composite/meta (estimator-valued params, _HeterogenousMetaEstimator, wrappers, reducers, pipelines, ensembles, reconcilers).
AST-scan the class's own methods (_fit, _predict, _update, _predict_interval, _predict_quantiles, _predict_var, _predict_proba + private helpers in the class body, + same-module helper functions they call where feasible) for pandas API usage: pd.* calls, .loc/.iloc/.xs/.groupby/.index/.columns/.resample/.shift/.rolling/.asfreq/.to_period/.freq, vs .values/.to_numpy()/np.asarray(.
Buckets: (B1) numpy-core — converts to numpy immediately, pandas only to wrap outputs/index; (B2) pandas-semantic — uses pandas index/time semantics or dataframe ops in core logic; (B3) third-party delegating — wraps an external lib (statsmodels, pmdarima, statsforecast, prophet, darts, neuralforecast, gluonts, torch/HF foundation models), whose input type then matters; (B4) composite/meta; (B5) other.
Report counts per bucket and per inner mtype (incl. any non-pandas inner mtypes); examples per bucket (incl. NaiveForecaster and MIRAForecaster — find it). Manually spot-check >=15 classifications by reading code and report heuristic accuracy.
Also count reliance on fh.to_absolute / to_absolute_index / self._cutoff / self.cutoff for constructing output indexes.
Key deliverable: an evidence-based estimate of what fraction of forecasters could be served by a "numpy values + time-index metadata" inner representation with ZERO changes to core math (boundary changes only) vs needing rewrites, and how many are bound to a third-party library's own input format. Write results/a2_census.csv and reports/a2_census.md.` },
  { key: 'a3_time_index', phase: 'Audit', lenses: ['re-run experiments and check each pandas-vs-polars semantic difference independently'], prompt: `
WORKSTREAM a3_time_index: map time-index / forecasting-horizon coupling and its non-pandas equivalents, with experiments.
Read sktime/forecasting/base/_fh.py (ForecastingHorizon), sktime/utils/datetime.py, sktime/utils/validation/forecasting.py (check_fh), sktime/datatypes/_utilities.py (get_cutoff, get_time_index), index helpers, sktime/datatypes/_adapter/polars.py, polars converters in sktime/datatypes/_series/_convert.py, _panel/_convert.py, _hierarchical/_convert.py, and the dask adapter.
Questions: (1) which semantics are pandas-only: PeriodIndex & period arithmetic, freq/DateOffset (business days, month-end/start), pd.infer_freq, allowed index types (RangeIndex, Int, Datetime, Period), absolute vs relative fh conversions. (2) what polars / pyarrow / numpy can represent: polars Datetime/Date/Duration, dt.offset_by("1mo"), pl.datetime_range(interval=...), no Period dtype, no freq attribute; arrow timestamps; numpy datetime64/timedelta64 (no calendar months).
Experiments (scripts/a3_*.py): (a) round trip PeriodIndex(M), DatetimeIndex(freq in MS, ME, B, h), RangeIndex, integer index through sktime's current polars mtypes (convert_to pandas -> polars_hierarchical/polars_panel/pl.DataFrame -> pandas): what is lost (period->timestamp? freq dropped? dtype changes?); (b) fit NaiveForecaster on polars monthly input and inspect predict output: container type, time dtype, the predicted timestamps (month starts? ends?); (c) can fh.to_absolute be expressed with polars dt.offset_by for each pandas freq type (test B, MS, ME, W-SUN, h, Q).
Propose a minimal "TimeIndex" abstraction (cutoff = last value, add k steps, steps-between, step/freq representation, generate range) with pandas and polars/arrow/numpy implementations, and pinpoint semantic mismatches (calendar months, business days, irregular timestamps, integer/range indexes).
Deliverable: table of pandas time semantics vs non-pandas equivalents, and a feasibility verdict on decoupling ForecastingHorizon/cutoff from pandas — including the option "keep pandas only as the time-arithmetic engine on the (small) index while data lives in polars" and its measured cost. Write reports/a3_time_index.md.` },

  // ---------------- RESEARCH ----------------
  { key: 'r1_sklearn_skrub', phase: 'Research', lenses: ['open every cited URL / source file and check the claim and its currency'], prompt: `
WORKSTREAM r1_sklearn_skrub: prior art — scikit-learn and skrub approaches to container independence. Verify by reading actual code, not memory.
Read source in EXP_PY site-packages (scikit-learn 1.9.1: sklearn/utils/_set_output.py, sklearn/utils/_array_api.py, any dataframe/narwhals helpers, validation.py dataframe handling). Create your own venv to install and read skrub (skrub/_dataframe/_common.py and its dispatch).
Cover: (1) sklearn set_output ContainerAdapterProtocol + _PandasAdapter/_PolarsAdapter + ADAPTERS_MANAGER (dispatcher for OUTPUT containers): protocol size, ops, extension story; (2) sklearn Array API support (array-api-compat get_namespace, xp pattern): how estimators were ported INCREMENTALLY (opt-in per estimator, tags such as array_api_support), the test strategy, lessons for sktime's per-estimator inner-mtype tags; (3) sklearn discussions about dataframe-agnosticism / narwhals / the dataframe interchange protocol (search GitHub issues/PRs; e.g. "narwhals" in the scikit-learn repo) — what they decided and why; (4) skrub's own dispatch mechanism for pandas + polars: how it works, number of ops, why they did/didn't use narwhals; (5) Dataframe Interchange Protocol (__dataframe__) vs Arrow PyCapsule Interface (__arrow_c_stream__) vs the DataFrame API Standard — status in 2026 (deprecations in pandas/polars?) and what that means for sktime ingress/egress.
For each: approach category, how it works, what went well, what went wrong, relevance to sktime, sources (URLs or file paths). Write reports/r1_sklearn_skrub.md.` },
  { key: 'r2_narwhals', phase: 'Research', lenses: ['open every cited URL / narwhals source line and check the claim against narwhals 2.27.1'], prompt: `
WORKSTREAM r2_narwhals: prior art — Narwhals deep-dive: documented semantics + adopters' experiences.
Use the installed narwhals 2.27.1 source in EXP_PY site-packages (read code), its docs (narwhals-dev.github.io/narwhals) and GitHub issues via WebSearch/WebFetch. Answer with citations:
(1) LazyFrame.collect(backend=...) semantics: default eager target per lazy backend (polars lazy, dask, duckdb, ibis, pyspark, sqlframe), allowed backend values, whether the original type can be preserved, to_native() on lazy frames; the eager/lazy split and lazy-unavailable ops (group_by iteration, __getitem__, iter_rows, to_numpy, ...).
(2) pandas index policy: what narwhals does with a non-default index / MultiIndex on from_native; which ops reset vs preserve index; maybe_set_index / maybe_get_index / maybe_reset_index / maybe_align_index; documented guidance for index-reliant libraries.
(3) stable API (narwhals.stable.v1/v2) and backward-compat promise; dependency footprint; overhead claims/benchmarks.
(4) adopters and their experiences: plotly (v6 migration and perf claims), altair, marimo, bokeh, scikit-lego, fairlearn, formulaic, Hamilton, great-tables, py-shiny, vegafusion, darts, pointblank, tubular, any time-series/forecasting library. Find CONCRETE issues/PRs about index/MultiIndex handling, datetime/period dtypes, performance regressions, missing ops, lazy limitations, pandas-specific behaviour lost.
(5) any narwhals support/discussion for grouped iteration over panel data, over() window expressions across backends, rolling/ewm availability per backend; any sktime-related discussion ("sktime narwhals").
Give an evidence-based verdict on C1 and C2 from documentation (experiments test them separately). Write reports/r2_narwhals.md.` },
  { key: 'r3_ts_libs', phase: 'Research', lenses: ['open cited sources / installed package code and check each library description'], prompt: `
WORKSTREAM r3_ts_libs: prior art — how time-series/forecasting libraries handle data containers. Verify with source where possible (own venv: try installing utilsforecast statsforecast mlforecast darts — skip any that take too long — and read site-packages source) plus docs/GitHub via WebSearch/WebFetch.
Cover: (1) Nixtla: utilsforecast (processing.py dispatch over pandas/polars, compat.py unions, process_df / DataFrameProcessor -> numpy GroupedArray with indptr, whether/where they adopted narwhals), statsforecast/mlforecast polars support (convert polars to pandas? or to GroupedArray?), how outputs return in the input container, perf claims. (2) Darts: TimeSeries internal representation (xarray/numpy), from_dataframe / from_group_dataframe / to_dataframe via narwhals (since which version, what issues: index, datetime, perf). (3) functime: polars-native forecasting design (LazyFrame, group_by/agg, map_batches, over), what worked, project status. (4) aeon (sktime fork): moving away from pandas MultiIndex for collections (numpy3D / list of 2D arrays) and series — rationale and consequences. (5) briefly: GluonTS, skforecast, pytimetk (polars engine), tslearn, Prophet, Kats, temporian, Chronos/TimesFM pipelines.
For each: internal representation, boundary conversion strategy, supported containers, whether output preserves input container type, notable issues. Extract cross-cutting patterns (hub-and-spoke numpy GroupedArray, narwhals-at-the-boundary, polars-native core, ...). Write reports/r3_ts_libs.md.` },
  { key: 'r4_dispatch_arch', phase: 'Research', lenses: ['open cited sources and check claims, especially Fugue and pandera internals'], prompt: `
WORKSTREAM r4_dispatch_arch: prior art — multi-backend architectures outside time series, and standardization efforts. Verify via docs/source (WebSearch/WebFetch; optionally own venv to read source).
Cover: (1) pandera backend registry (pandas, polars, ibis, pyspark): how one schema API is shared while backends implement checks; how polars support was added (rewrite vs adapter); lessons. (2) Fugue: abstract DataFrame + execution engines (pandas/Spark/Dask/Ray/DuckDB) and transform() running a per-partition python function on any engine — directly analogous to sktime vectorization (apply a function per instance group on any engine); how it maps per-partition functions over engines, handles schema, and its overheads. (3) Ibis: expression IR compiled to backends — when it is the right tool. (4) Modin / cudf.pandas / pyspark.pandas: keep-pandas-API-swap-engine; fidelity (MultiIndex!), performance. (5) Great Expectations execution engines, Hamilton, Kedro — briefly. (6) Arrow as interchange: Arrow PyCapsule Interface adoption (__arrow_c_stream__, __arrow_c_array__) across polars/pandas/duckdb/pyarrow/narwhals; zero-copy guarantees. (7) DataFrame API standard (data-apis consortium) and dataframe-api-compat — why abandoned, what replaced it. Plus: engine-native "apply per group" mechanisms (dask groupby().apply, polars group_by().map_groups, duckdb UDFs) and their overhead characteristics.
Extract design patterns with pros/cons (narrow adapter protocol, full API emulation, expression IR, per-partition UDF execution, canonical interchange format). Write reports/r4_dispatch_arch.md.` },

  // ---------------- EXPERIMENTS ----------------
  { key: 'x1_nw_lazy_collect', phase: 'Experiment', lenses: ['methodology: re-run on EXP_PY and EXP_PY3, try to find a counterexample to each matrix cell', 'read narwhals 2.27.1 source for collect()/to_native() and check the matrix against the code paths'], prompt: `
WORKSTREAM x1_nw_lazy_collect: empirically test claim C1 about narwhals collect().
Scripts scripts/x1_*.py (EXP_PY; repeat key parts on EXP_PY3). For each native input: polars.LazyFrame, polars.DataFrame, dask DataFrame, duckdb DuckDBPyRelation, ibis Table (duckdb backend), pyarrow.Table, pandas.DataFrame, modin (if quick):
- nw.from_native(x) -> narwhals type (DataFrame vs LazyFrame), implementation;
- for lazy ones: .collect() default -> native result type; .collect(backend=...) for each allowed value ("pandas", "polars", "pyarrow", Implementation enums) -> type or error;
- can we keep it lazy and get the SAME native lazy type back via .to_native() after ops (with_columns, filter, group_by().agg, sort, over() windows, join, concat) — i.e. "respect the original format" by not collecting.
- which ops a VectorizedDF needs work on LazyFrame per backend: group_by().agg, unique keys, over() windows, sort, per-group last-k rows (e.g. filter on rank/over), iterating groups (expected eager-only), to_numpy (eager-only), nw.concat lazy.
- fallback route: collect(backend=<eager equivalent>) then convert back (dask -> pandas -> dd.from_pandas; duckdb -> arrow -> duckdb.from_arrow): cost and type fidelity.
Produce a matrix backend x operation -> lazy-ok / needs-collect / error. Verdict on C1 (true / partially / misleading) with precise semantics, and workarounds (explicit backend param, stay lazy and collect only per-group results, map_batches, ...). Record versions. Write reports/x1_nw_lazy_collect.md.` },
  { key: 'x2_nw_index_fidelity', phase: 'Experiment', lenses: ['methodology: independently re-measure reset_index copy/cost with a different method (e.g. memray or np.shares_memory on every column) on pandas 2 and 3', 'try to reproduce the "RangeIndex came back, names gone" scenario and the minimal correct recipe; look for cases the recipe fails (period, categorical, tz-aware, duplicate names, unsorted)'], prompt: `
WORKSTREAM x2_nw_index_fidelity: empirically test claim C2 about narwhals and pandas MultiIndex. Run scripts/x2_*.py under BOTH EXP_PY (pandas 2.3.3) and EXP_PY3 (pandas 3.0.6, copy-on-write).
(1) nw.from_native(df with MultiIndex): raise / warn / silently keep? What do .to_native() results look like after select, with_columns, filter, sort, group_by().agg, join, concat?
(2) reset_index-before-grouping path: time + peak RSS of df.reset_index() for make_panel_fast sizes (10k x 50, 1k x 100, ...) and n_cols in {1, 10}; under pandas 3 CoW are data columns actually copied (np.shares_memory between original values and reset frame columns)? Also the inverse set_index cost. Compare to the cost of the current groupby(level=...) iteration itself so the copy is put in proportion.
(3) index fidelity round trip: MI (names h0,h1,time; time as DatetimeIndex with freq, PeriodIndex(M), RangeIndex/int, categorical level, tz-aware) -> reset_index -> narwhals ops (group_by iteration, filter, sort) -> to_native -> set_index(names): are level names, dtypes (period[M], datetime64 + freq attr), level order, row order restored? Reproduce the dev's "RangeIndex came back, level names gone" — which exact code path produces it, and whether nw.maybe_set_index / maybe_reset_index / maybe_get_index help.
(4) narwhals group_by iteration on a pandas-backed frame: order preservation (sort=False semantics), key dtypes, cost vs df.groupby(level=..., sort=False) iteration (bench, same process).
(5) freq after polars round trip; Period dtype support in narwhals (nw dtype for period?).
Verdict on each sub-claim of C2 ("columns only", "reset_index is a copy", "RangeIndex came back, names gone", "current code keeps names") with numbers, and the minimal correct recipe to preserve names/dtypes when using narwhals. Write reports/x2_nw_index_fidelity.md.` },
  { key: 'x3_iter_bench', phase: 'Experiment', lenses: ['methodology & fairness: check that every strategy produces equivalent slices/outputs and that conversion costs are accounted symmetrically; re-run'], prompt: `
WORKSTREAM x3_iter_bench: microbenchmark the core VectorizedDF operations across implementation strategies.
Script scripts/x3_iter_bench.py with --grid {dev,large}. Operations: (A) list instance keys; (B) iterate all instance slices, materialized as what a forecaster would receive; (C) reconstruct: concatenate per-instance outputs (e.g. the last 2 rows of each slice, mimicking a fh=2 predict) with instance keys back into one frame.
Strategies: S1 pandas groupby(level=..., sort=False) iteration + pd.concat(keys=) (current sktime); S2 pandas precomputed positions (codes/factorize -> stable argsort -> indptr -> df.iloc[start:stop]); S3 polars partition_by(keys, maintain_order=True, as_dict=True); S4 polars group_by(keys, maintain_order=True) iteration; S5 polars sorted + offsets -> df.slice(offset, len) (zero-copy); S6 numpy GroupedArray (2D values + indptr; slice = view; output via np.concatenate + key repetition); S7 narwhals group_by iteration over (a) pandas reset_index frame (b) polars frame; S8 pyarrow Table offset slicing.
Also measure the conversion cost to reach each representation from a pandas MI frame and from a polars frame (pandas->polars, polars->pandas via sktime convert_to("pd-multiindex"), pandas->GroupedArray, polars->GroupedArray), with peak RSS deltas.
Report per-slice overhead (us), totals, ratios vs S1 per grid size. For pandas strategies the slice must carry the time index as its index (what a pd.DataFrame-inner forecaster gets). Explain WHY each is fast/slow. Write results/x3_iter_bench.json + reports/x3_iter_bench.md.` },
  { key: 'x8_e2e_profile', phase: 'Experiment', lenses: ['methodology: re-profile with a different profiler/attribution method (e.g. manual timers or pyinstrument-style sampling via cProfile callees) and check bucket attribution'], prompt: `
WORKSTREAM x8_e2e_profile: end-to-end profile of the CURRENT sktime vectorized forecasting path — where does time actually go, and therefore which decoupling approach can deliver real gains?
Script scripts/x8_e2e_profile.py (--grid flag): Panel and Hierarchical inputs (make_panel / make_panel_fast; n_series in {10, 100, 1000, 5000}, T=100); containers: pandas MI, sktime polars_panel / polars_hierarchical (via sktime convert_to from pandas), and dask if a dask mtype works. Run NaiveForecaster(strategy="last") fit(y, fh=[1,2,3]) + predict(), and one heavier forecaster without missing soft deps (check what's installed: e.g. ThetaForecaster, PolynomialTrendForecaster, ExponentialSmoothing if statsmodels present).
Use cProfile and aggregate into buckets: input checks (check_is_scitype + metadata), conversion (convert_to), VectorizedDF construction, slicing/groupby iteration, clone, per-instance inner _fit/_predict, per-instance BaseForecaster boilerplate (fit/predict re-running checks/conversions on each slice!), reconstruct (pd.concat), back-conversion to input mtype, other. Report % and absolute per size.
KEY QUESTION: for polars input, what fraction is polars<->pandas conversion vs per-instance Python overhead of running a full BaseForecaster.fit/predict per instance? Does decoupling the container alone give speedups, or does per-instance BaseForecaster overhead dominate (so a lighter inner loop / panel-native implementations matter more)? Also measure the same work done "by hand" (e.g. groupby().last() for naive-last) as the ceiling. Write results/x8_e2e_profile.json + reports/x8_e2e_profile.md.` },
  { key: 'x4_proto_adapter', phase: 'Experiment', lenses: ['correctness: run the equivalence tests yourself, add adversarial cases (unsorted index, unequal lengths, iterate_cols=True, Hierarchical->Panel iteration, string vs int keys, empty frame)'], prompt: `
WORKSTREAM x4_proto_adapter: prototype idea (1) — adapter/dispatcher-based VectorizedDF — and test how far it extends into the forecaster layer.
Implement protos/x4_adapter.py: a minimal ContainerAdapter protocol (derive ops from sktime/datatypes/_vectorize.py and BaseForecaster._vectorize: e.g. instance_keys(levels), iter_instances(levels, cols), select_cols, concat_with_keys(parts, keys, names), time index access, n_rows, columns, to_numpy, freq) + a registry dispatching on native type (look at sklearn's ADAPTERS_MANAGER in EXP_PY site-packages sklearn/utils/_set_output.py), with PandasAdapter (MultiIndex-based, behaviour-identical to current), PolarsAdapter (key columns: sktime "__index__" convention vs plain key columns — decide and justify), PolarsLazyAdapter (what can be done without collect?), optionally PyArrowAdapter. Build VectorizedDF2 on top (items / as_list / reconstruct / vectorize_est-like apply).
Tests (scripts/x4_test_adapter.py): equivalence with current sktime VectorizedDF for pandas Panel & Hierarchical, iterate_cols True/False, iterate_as Series/Panel (compare slices + reconstruct via pandas.testing); polars round trip with no pandas conversion.
Crucial extension experiment: run sktime NaiveForecaster per slice when slices are polars: (a) forecaster with y_inner_mtype pd.Series -> needs per-slice conversion: measure per-slice convert vs whole-frame convert-once (current); (b) a toy BaseForecaster subclass in protos declaring y_inner_mtype "pl.DataFrame" (test whether sktime's check/convert machinery already allows it end to end!) -> zero conversion.
Measure LOC per adapter, number of protocol methods, perf vs current at DEV_GRID. Conclude: what the adapter approach buys at VectorizedDF level, marginal cost of adding a backend, exactly where it stops (forecaster inner-mtype boundary) and which mechanism bridges it. Write reports/x4_proto_adapter.md.` },
  { key: 'x5_proto_narwhals', phase: 'Experiment', lenses: ['correctness & fidelity: run its tests, add adversarial inputs (period index, unsorted, unequal lengths, multi-column, lazy backends) and check numeric equality vs sktime'], prompt: `
WORKSTREAM x5_proto_narwhals: prototype idea (2) — narwhals-based VectorizedDF and narwhals-native forecasters.
Implement protos/x5_narwhals.py: same functional surface as current VectorizedDF (schema from input, iterate slices, reconstruct) written with narwhals, accepting pandas MI (reset_index + metadata to restore the index), polars eager, polars lazy, pyarrow, dask, duckdb; decide and document lazy handling (collect where? with which backend?).
Tests (scripts/x5_test_narwhals.py): equivalence with current sktime VectorizedDF for pandas (index names, dtypes, order, freq); round-trip fidelity for polars/pyarrow (output type == input type?). Benchmark vs current at DEV_GRID.
Forecaster-layer experiment: implement 2-4 panel-native forecasters in pure narwhals operating on ALL instances at once without Python loops (naive last, naive mean, seasonal naive with sp, drift; optionally SES via ewm_mean if supported across backends) producing the forecast frame for fh steps on pandas / polars / polars-lazy / duckdb / dask; compare speed vs sktime NaiveForecaster vectorized (per-series loop) at n_series 100/1000/10000 and check numeric equality with sktime outputs.
Report: what narwhals covers well, what is missing (how did you generate future timestamps per group? rolling/ewm availability per backend), LOC, maintainability, gotchas (index, dtypes, lazy). Write reports/x5_proto_narwhals.md.` },
  { key: 'x6_proto_hub', phase: 'Experiment', lenses: ['fidelity & zero-copy claims: independently check round trips (dtypes, names, freq, period) and np.shares_memory claims; re-measure costs'], prompt: `
WORKSTREAM x6_proto_hub: prototype a CREATIVE alternative — hub-and-spoke canonical internal representation (cf. Nixtla GroupedArray, Darts TimeSeries).
Implement protos/x6_hub.py: PanelData = (instance keys as small table / codes+categories, per-row time array (datetime64/int64), 2D float values, indptr offsets, column names, + metadata needed to rebuild the exact native container: source library/type, index names, time dtype incl. pandas freq / period freq, column dtypes, original order). Spokes: from_native/to_native for pandas MI, polars eager, polars lazy (collect once), pyarrow (zero-copy where possible), dask (compute). Try two spoke styles: (i) hand-written, (ii) narwhals-based or Arrow-PyCapsule-based — compare LOC and fidelity.
Measure: conversion into hub from each source and back (time + peak RSS), zero-copy behaviour (np.shares_memory where applicable), slicing (views), reconstruct, round-trip fidelity (assert_frame_equal / polars equals incl. dtypes, names, freq).
Forecaster side: implement a thin "numpy values + time metadata" inner shim so NaiveForecaster-like math runs on views with no pandas; measure vs current. Then measure the cost of lazily materializing a pandas view per slice (pd.DataFrame from a view with the right time index) for pandas-semantic forecasters. Discuss memory (one copy at ingress vs none) and what is lost. Write reports/x6_proto_hub.md.` },
  { key: 'x7_native_engine', phase: 'Experiment', lenses: ['fairness & correctness: check numeric equality across implementations and that lazy/duckdb paths actually execute; re-measure speedups'], prompt: `
WORKSTREAM x7_native_engine: quantify the "true value" of non-pandas engines for forecasting workloads — the gain from pushing computation into the engine instead of per-series Python loops — and evaluate an engine-native vectorization hook.
Script scripts/x7_native.py (--grid flag): implement naive last, seasonal naive (sp=7), window mean (w), drift, and closed-form OLS linear trend per series, in: (a) pandas groupby vectorized ops (no per-group python), (b) polars eager expressions (over, group_by().agg), (c) polars lazy (streaming / new engine if available), (d) duckdb SQL window functions, (e) numpy on equal-length series reshaped to (n_series, T) or GroupedArray, (f) baseline: per-series loop via current sktime vectorization (NaiveForecaster, PolynomialTrendForecaster). Sizes n_series in {100, 1000, 10000} x T=100 (RSS < 450MB). Check numeric equality vs (f). Report speedups vs (f) and vs (a).
Then sketch protos/x7_hook.py: a forecaster may optionally implement a panel-level _fit_predict_panel(frame, fh) (capability tag e.g. "capability:panel_native") that BaseForecaster uses INSTEAD of the per-instance VectorizedDF loop when the input container supports it; show it working for naive-last on polars and pandas without conversion (subclassing, no edits to sktime/). Note which forecaster families could have such implementations (closed-form/simple statistical) vs not (statsmodels ARIMA, deep learning, foundation models); grep the codebase to estimate counts. Write reports/x7_native_engine.md.` },
  { key: 'x9_cheap_paths', phase: 'Experiment', lenses: ['correctness: re-run, verify zero-copy claims with np.shares_memory / pyarrow buffer addresses, and check sktime compatibility claims on more forecasters'], prompt: `
WORKSTREAM x9_cheap_paths: test two low-cost creative options that do not require rewriting forecasters.
(A) "pandas API, Arrow memory": polars -> pandas with pyarrow-backed dtypes (polars.to_pandas(use_pyarrow_extension_array=True) / pd.ArrowDtype types_mapper) is (near) zero-copy. Measure conversion time + peak RSS vs numpy-backed conversion (current sktime polars->pandas path) at DEV_GRID sizes, verify zero-copy (buffer addresses / shares_memory where meaningful). Then: does sktime accept ArrowDtype frames? Run check_is_scitype, NaiveForecaster, ThetaForecaster/PolynomialTrendForecaster, a transformer pipeline (e.g. Differencer / Detrender) and VectorizedDF on ArrowDtype float + timestamp[ns] index levels, on pandas 2.3 and 3.0 (EXP_PY, EXP_PY3). Record what breaks and why (numpy ops on ArrowDtype, index types, freq). Also test returning results in the input container (pandas Arrow -> polars) zero-copy.
(B) laziness / pushdown value: with a parquet file of 10k series x 200 time points (write it to ED/results/x9_data.parquet, delete after), compare: current-style eager load all + convert to pandas + select last 50 timepoints of 1000 series, vs polars LazyFrame / duckdb with predicate+projection pushdown producing only the needed slice. Measure time + peak RSS. This quantifies what sktime loses by collecting/converting lazy inputs up front, and what a lazy-aware boundary (pushing instance/time/column selection into scan) would gain.
Also check: does sktime currently accept polars LazyFrame for forecasting y (polars_lazy_table is a Table mtype — is there a lazy Series/Panel mtype)? Write reports/x9_cheap_paths.md.` },
]

phase('Audit')
log(`launching ${WS.length} workstreams; each gets adversarial verification as soon as it completes`)

const results = await pipeline(
  WS,
  (ws) => COMPLETED[ws.key]
    ? Promise.resolve(COMPLETED[ws.key])
    : agent(CTX + '\n' + ws.prompt, { label: ws.key, phase: ws.phase, schema: RESULT }),
  (res, ws) => {
    if (!res) return { ws: ws.key, res: null, verdicts: [] }
    return parallel(ws.lenses.map((lens, i) => () => agent(`${CTX}
You are an ADVERSARIAL VERIFIER for workstream "${ws.key}". The original task was:
${ws.prompt}

The original agent's structured result:
${JSON.stringify(res, null, 1)}

Read its detailed report (${res.report_path}) and the scripts/protos/sources it references.
Lens: ${lens}
Task: select up to 8 of the most decision-relevant claims (prefer quantified ones, ones about C1/C2, and ones that would change the recommended architecture). Try to REFUTE each:
 - experimental claims: re-run the referenced script (reduced size is fine) and, where feasible, write an independent check via a DIFFERENT method/code path in scripts/${ws.key}_verify${i}_*.py; hunt for methodology flaws (asymmetric accounting of conversion costs, missing warm-up, caching, lazy query never executed, weak equality checks, sizes too small to matter, measurements taken under load without ratios).
 - research claims: open the cited URLs / source files and check they say what is claimed; check currency vs installed versions (narwhals 2.27.1, polars 2.0.0, pandas 2.3.3/3.0.6, sklearn 1.9.1).
 - audit/census claims: re-count independently or spot-check samples.
Verdicts: "unverifiable" if you cannot check; "refuted" if evidence contradicts; "partially_confirmed" if directionally right but numbers/conditions differ (say how). Also list important things the agent missed or got wrong. Write reports/${ws.key}_verify${i}.md.`,
      { label: `verify:${ws.key}#${i}`, phase: 'Verify', schema: VERDICT })))
      .then((vs) => ({ ws: ws.key, res, verdicts: vs.filter(Boolean) }))
  },
)

const ok = results.filter(Boolean)
log(`workstreams done: ${ok.filter((r) => r.res).length}/${WS.length} produced results`)

const digest = ok.map((r) => ({
  key: r.ws,
  summary: r.res ? r.res.summary : 'FAILED',
  report: r.res ? r.res.report_path : null,
  impact_statements: r.res ? r.res.impact_statements : [],
  open_questions: r.res ? r.res.open_questions : [],
  verdicts: r.verdicts.map((v) => ({ lens: v.lens, verdicts: v.verdicts.map((x) => `${x.id}: ${x.verdict} — ${x.notes}`), missed: v.missed_or_wrong })),
}))

phase('Wrap')
const GAPS = {
  type: 'object',
  properties: {
    gaps: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          why_it_matters: { type: 'string' },
          proposed_experiment: { type: 'string' },
          priority: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['title', 'why_it_matters', 'proposed_experiment', 'priority'],
      },
    },
    contradictions: { type: 'array', items: { type: 'string' }, description: 'places where workstreams or verifiers disagree' },
  },
  required: ['gaps', 'contradictions'],
}

const [critic, bench] = await parallel([
  () => agent(`${CTX}
You are the COMPLETENESS CRITIC. Below is the digest of all workstreams (summaries, impact statements, open questions, verifier verdicts). Read the reports in ${ED}/reports/ as needed (do NOT run heavy benchmarks — another agent is benchmarking right now).
Decision the user must make: how to decouple sktime forecasters + VectorizedDF from pandas (candidate approaches: adapter/dispatcher, narwhals, hub/canonical representation, per-estimator opt-in via inner-mtype tags, engine-native panel hook, pandas-with-Arrow-memory, lazy-aware boundary, combinations).
Identify: (1) approaches not evaluated or under-evaluated; (2) claims that remain unverified or were refuted/contradicted; (3) missing experiments needed to state a quantified impact for each approach; (4) contradictions between workstreams/verifiers. For each gap propose a concrete, runnable experiment. Write ${ED}/reports/wrap_critic.md.

DIGEST:
${JSON.stringify(digest, null, 1)}`, { label: 'critic', phase: 'Wrap', schema: GAPS }),
  () => agent(`${CTX}
You are the SERIAL BENCHMARK runner. All parallel experiment agents have FINISHED, so the machine is now (mostly) quiet. Your job: produce final, low-noise numbers.
1. List ${ED}/scripts/ and identify timing-sensitive benchmark scripts (x3_iter_bench, x8_e2e_profile, x4/x5/x6/x7/x9 benchmark parts, any script using bench()). Skip *_verify* scripts unless they are the better benchmark.
2. Re-run them ONE AT A TIME (never concurrently) with --grid large where supported (otherwise copy to scripts/final_<name>.py and raise sizes; do not edit originals). Keep peak RSS <= 3GB; drop sizes that exceed it or take > 10 min each; total budget ~60 min. Note the machine load (uptime) before each run.
3. Consolidate into ${ED}/results/final_bench.json and ${ED}/reports/final_bench.md: one comparison table across approaches — current sktime pandas path; current sktime polars-input path; adapter proto (x4); narwhals proto (x5); hub/grouped-array proto (x6); engine-native implementations (x7); arrow-backed pandas (x9) — for iterate/reconstruct per-slice overhead, conversion costs, end-to-end NaiveForecaster fit+predict, and peak memory, at each size. Mark anything you could not run and why.
Digest of workstreams for orientation:
${JSON.stringify(digest.map((d) => ({ key: d.key, report: d.report, summary: d.summary })), null, 1)}`, { label: 'final-bench', phase: 'Wrap', schema: RESULT }),
])

return { digest, critic, bench }
