# design_engine_lazy: plan-first boundary, engine-native panel kernels, partition delegation

Architect angle: **ENGINE-NATIVE / LAZY-FIRST.** The forecaster call is treated as a query against the user's
engine. The boundary first works out *what the estimator will read* (columns, instances, lookback). It pushes that
into the user's plan (polars-lazy, duckdb, pyarrow dataset, dask) and only then materializes, **once**, into the
form the estimator consumes. Forecasters that can be written as group-wise expressions or SQL run **inside** the engine
through a panel-native hook, and their output stays in that engine. Distributed inputs are partitioned by key, and each
partition is handed to the ordinary vectorized path.

Citations use the form `report-key claim`, for example `x7.4` or `g3 C5`. Numbers tagged **[DEL-n]** come from scripts I
ran for this report (§9). Anything else is labelled *unmeasured/expected*. Nothing under `sktime/` was modified.

---

## 0. Thesis (≤120 words)

The engine's value only appears when the computation stays in its plan. x7 measured 10²–10³x for panel-native kernels and
≈1,180x vs today from parquet, but it also showed that per-instance pandas forecasters destroy that value (g3: laziness
is worth ±15% at C0–C4). So:
- **Plan first.** Build one `ReadSpec` from the estimator (columns, instances, structural lookback) and push it down before
  any collect.
- **Engine-native kernels** for the roughly 6–25 forecasters that are group-wise closed form or their composites. Kernels
  are order-robust by construction and return a params table plus long output in the input engine.
- **Partition delegation** for dask.
- **Every other forecaster** (about 135) takes one explicit materialization into the shared trusted per-instance lane
  (g1 + g3 C3).

---

## 1. What the evidence forces on a lazy-first design

| constraint | evidence | consequence |
|---|---|---|
| The value of laziness exists only if the whole computation stays in one plan | x7.4: polars scan + kernel 123 ms / +31 MB vs today's 145 s / +483 MB. x7 verifier: lazy is 4–5x faster than polars eager when warm. g3: lazy input under unchanged forecasters is within ±15% of eager, and 1.1–2.6x **slower** at C5 because of the boundary collect | Laziness is a property of the **kernel** route, not the container route. Non-native forecasters collect once, and nothing pretends otherwise |
| Most of the "lazy" gain is projection and selection, which eager code can also do | x9 verifier: true pushdown is 1.9x time / 2.8x RSS over eager filter-before-convert. **[DEL-2]**: polars lazy lookback is 1.12–1.60x over eager read-all, ≈ eager projection; duckdb is the outlier with 12–24x less RSS on series-major files | The boundary's job is a **ReadSpec**. A lazy source applies it in-plan, an eager source applies it before conversion. Same code path, same semantics |
| Projection is the dominant lever. Lookback adds little in polars 2.0 | **[DEL-2]** series-major, `last`: projection 1.68x / 1.58x RSS; projection + lookback (lazy) 1.60x / 1.69x. x6 §3.7: polars `group_by.tail` cuts rows 20x but not peak memory | Columns are pushed down first. Lookback is pushed down only where the engine benefits (duckdb, pyarrow dataset with row-group stats), or when the user's file is huge |
| Finite structural lookback is rare and covers point forecasts only | **[DEL-1]**: 26/70 runnable forecasters are equal on a truncated history, but ≥3 of those are data coincidences (AutoETS, ExponentialSmoothing, UnobservedComponents). Only 1/39 is truncation-safe for intervals | The `lookback` tag is **declared by construction**, never inferred. It is per method (`predict` vs `predict_interval`), and it defaults to `None` (= all history) |
| Positional kernels are silently wrong on shuffled rows | x7 verifier R2 (max_abs 7.6–9.8); r4 verifier V3 (9,997/10,000 keys wrong). **[DEL-3]**: positional is wrong and order-robust is right on shuffled 8M rows | Kernels must be **order-robust by construction** (`sort_by` inside the agg, `ORDER BY` in the window). **[DEL-3]**: that costs 1.11–1.38x the positional kernel. A separate sortedness check costs 0.70–0.77 s, **3.7–5.3x the whole kernel**, so "check, then positional" is the wrong guard |
| `collect()` targets differ by backend; duckdb cannot be collected into duckdb | x1.1–.3, x7.11, x1 verifier (stable.v1 has no collect for duckdb) | One explicit `materialize(target=...)` policy. A duckdb result is returned as a relation or Arrow, never via a bare collect |
| narwhals' duckdb `dt.total_seconds()` is silently wrong | x7.11 (5 vs 259205) | Engine kernels are written **natively per engine family** (polars expressions, duckdb SQL, numpy arrays). narwhals is used at most for the boundary plumbing, never for kernel arithmetic |
| The engine adds only 1.07–1.33x over pandas groupby once conversion is charged | x7 verifier x7.2 | Kernels run **in the input's own engine**. Crossing engines to "get polars speed" is not allowed. The numpy grouped kernel (g2) is the universal fallback: 7–11x over pandas groupby with pandas in and out |
| Per-instance framework re-entry dominates for unchanged forecasters | g1 (fit+predict 1.15–1.88x), g3 C2–C3 (1.6–3.3x) | The lane for about 135 forecasters is the trusted path. This design adopts it and does not compete with it |
| Distributed engines give little by distribution alone | r4 verifier V1 (Fugue +150 µs/group); r3 E4 (203–610x slower per-group delegation). **[DEL-4]**: on 4 shared cores, partition-delegate is 0.70–1.31x serial, per-instance dask is 0.34–0.35x, loky is 0.93–0.96x | Delegation is **partition-coarse** (one vectorized call per partition) and exists for data that does not fit or lives on a cluster. It is not a speed lever on one box |

---

## 2. Architecture

### 2.1 Components

```
 user input (pandas | polars | pl.LazyFrame | duckdb rel | pa.Table/Dataset | dask)
        │
        ▼
 ┌──────────────────────── Boundary (once per public call) ────────────────────────────┐
 │ SourceAdapter.describe(obj) → SourceInfo(backend, lazy, keys, time, cols, est_rows) │
 │ est.read_spec(method, fh) → ReadSpec(columns, instances, lookback, time_range)      │
 │ SourceAdapter.push(obj, spec)  (in-plan for lazy, filter-before-convert for eager)  │
 │ Guards (null keys, X/y containment, step) as engine expressions; no positional       │
 │   assumptions                                                                       │
 │ PanelSchema + TimeAxis (a3 Step, per-instance cutoff)                               │
 └──────────────┬───────────────────────────────┬──────────────────────────┬───────────┘
                │ route = choose_route(est, src)│                          │
   R1 engine kernel                R2 materialize once                R3 partition delegate
   (est has kernel for             → inner family (pandas MI/          (dask/spark: key-aligned
    src.backend family)              np_grouped)                       partitions, map_partitions
   _fit_panel/_predict_panel        → trusted per-instance lane        → R2 per partition,
   in the engine; params table       (g1 + g3 C3) or panel kernel       declared output schema)
   output long, same engine          on np_grouped (g2)
                │                               │                          │
                └───────────────► Egress: OutputSpec → user's backend ◄────┘
                    (pandas input → today's pandas layouts; others → long schema, g4)
```

### 2.2 Protocols (signatures)

```python
# sktime/datatypes/_source/_base.py  (new)
@dataclass(frozen=True)
class SourceInfo:
    backend: str                  # "pandas" | "polars" | "polars_lazy" | "duckdb" | "pyarrow" | "pa_dataset" | "dask"
    lazy: bool
    keys: tuple[str, ...]         # instance key columns (pandas: MI level names, unnamed levels get synthetic names)
    time: str
    columns: tuple[str, ...]
    scitype: str                  # "Series" | "Panel" | "Hierarchical"

@dataclass(frozen=True)
class ReadSpec:
    columns: tuple[str, ...] | None = None    # projection (y vars + X vars the estimator uses)
    instances: object | None = None           # key filter (user arg, or update() delta)
    lookback: int | None = None               # rows per instance, by time, from the END; None = all
    time_range: tuple | None = None           # [start, end] e.g. evaluate() windows

class SourceAdapter(Protocol):                # registry keyed by backend via public register_source()
    def describe(self, obj) -> SourceInfo: ...
    def push(self, obj, spec: ReadSpec, info: SourceInfo): ...        # returns the SAME backend type
    def guard(self, obj, info, *, X=None) -> GuardReport: ...          # expressions; lazy stays lazy
    def cutoffs(self, obj, info) -> "Cutoffs": ...                      # per-key max/min time, O(n_keys) out
    def materialize(self, obj, info, target: str): ...                  # target in {"pandas_mi","np_grouped","polars","arrow"}
    def emit(self, long_result, info, *, like) -> object: ...           # long result to the user's backend/layout
    def partitions(self, obj, info, n: int | None) -> "Partitioned | None": ...  # dask/spark only

# estimator side (BaseForecaster gains one method, overridable, default = conservative)
class BaseForecaster:
    def read_spec(self, method: str, fh) -> ReadSpec:
        # default: columns = all, lookback = None. Uses tag "lookback" (see §5)
        ...
    # engine-native hook (opt-in, x7 design + order-robust contract)
    _tags = {"capability:panel_native": False,
             "panel_native:engines": (),          # subset of ("polars", "duckdb", "numpy")
             "lookback": None}                     # int | "sp" | "window_length" | callable(self, method)->int|None

    def _panel_native_supported(self, method: str, info: SourceInfo) -> bool: ...  # param predicate (e.g. strategy=="last")
    def _fit_panel(self, data, schema: "PanelSchema", engine: str, fh) -> "ParamsTable": ...
    def _predict_panel(self, params: "ParamsTable", steps: "HorizonSteps", engine: str): ...   # long output
    def _predict_interval_panel(self, params, steps, coverage, engine): ...       # long proba schema (g4)
    def _update_panel(self, params, data_delta, schema, engine) -> "ParamsTable": ...

@dataclass
class ParamsTable:          # fitted state of a native kernel; replaces per-instance clones
    engine: str
    table: object           # pl.DataFrame | pl.LazyFrame | duckdb rel | np record arrays; one row per key
    key_names: tuple[str, ...]
    cutoff_col: str
    def to_pandas(self) -> "pd.DataFrame": ...  # for get_fitted_params / forecasters_ shim
```

Kernel contract (enforced by a conformance suite, §6):
1. **Order-robust.** Each kernel orders by time inside its own plan (`pl.col(v).sort_by(t)`, `ORDER BY t` / `arg_max(v, t)`). It
   never relies on row order. **[DEL-3]** shows this is cheaper than checking sortedness.
2. **Null-key and NaN semantics equal stock**, and checked by `guard`.
3. **Equality with the stock per-instance result** at rtol 1e-9 on the M4 adversarial fixture set (wrap_critic M4), on every
   engine the kernel declares.
4. **Output** is long `(keys…, time, var, value)` with time computed from the TimeAxis (`cutoff + h·step`), never inferred.

### 2.3 Routing

```python
def choose_route(est, info, method) -> str:
    if info.backend in ("dask", "spark") and not est.get_tag("capability:panel_native"):
        return "R3"                                            # coarse partition delegation
    eng = ENGINE_FAMILY[info.backend]                          # polars_lazy/polars -> "polars"; duckdb -> "duckdb";
                                                               # pandas/pyarrow -> "numpy" (g2 np_grouped)
    if est.get_tag("capability:panel_native") and eng in est.get_tag("panel_native:engines") \
       and est._panel_native_supported(method, info):
        return "R1"
    if est.get_tag("capability:panel_native") and "numpy" in est.get_tag("panel_native:engines") \
       and est._panel_native_supported(method, info):
        return "R1-numpy"                                      # materialize to np_grouped once, kernel there
    return "R2"
```

R1 never crosses engines. If the input is pandas and the kernel exists only in polars, the route is R1-numpy or R2, not a
pandas→polars hop. Reason: the x7 verifier found that the conversion eats the engine gain (1.07–1.33x).

### 2.4 Data flow per input container

| input | boundary | R1 (native kernel) | R2 (all other forecasters) | output |
|---|---|---|---|---|
| pandas MI | `describe` from index levels; `push` = column projection only; guards on MI codes (design_hub G cost) | numpy kernel over MI codes (g2), no pandas built per slice | today's path + trusted lane (g1) + fh/cutoff once (g3 C3) | today's pandas layouts, bit-identical |
| polars eager | `push` = select + optional lookback filter (eager filter-before-convert, x9 verifier) | polars expression kernel in place | one `materialize("pandas_mi")` (vectorized monotonic check, x9.3) → trusted lane; or `np_grouped` for g2 ports | polars (predict); long-schema polars for proba |
| polars LazyFrame (new mtype family) | `push` in-plan (projection, predicate, lookback); guards as lazy exprs; cutoffs via a lazy agg (DN-B4 lazy cutoffs 9–15 ms at 10k) | **the whole fit+predict is one plan**, `predict` returns a `LazyFrame` (x7 hook demo) | one collect with explicit `engine`, then as polars eager | LazyFrame for R1; eager polars for R2 (`output:lazy=True` re-wraps) |
| duckdb relation | `push` as SQL (`SELECT cols … QUALIFY row_number() … <= w`) | SQL kernel; result stays a relation (`to_native`, never `collect`, x7 verifier C1) | `fetch_arrow_table` → arrow spoke → pandas or np_grouped | relation for R1; Arrow/pyarrow for R2 |
| pyarrow Table / Dataset | Dataset: `scanner(columns, filter)` (row-group pushdown, T6 B4 88x) | numpy kernel on arrow buffers (zero-copy single chunk only, x9 verifier) | arrow → pandas MI with numpy time level (x9.4: Arrow time index rejected) | pyarrow |
| dask | `push` = column projection + key filter, lazy; `partitions` = `set_index(key, sorted=True)` key-aligned (r4 V5: 0 order violations) | dask-native kernel only for aggregates (x5 dask 4.7x at 1000x500) | **R3**: `map_partitions` → each partition runs R2 with trusted lane; output meta declared; result stays dask | dask DataFrame, long |

### 2.5 Where pandas remains (by design)

- **R2 inner lane.** About 135 forecasters keep `y_inner_mtype` pandas. pandas is created once per call, or once per
  partition, never per slice from a non-pandas source (x4 verifier S5p: per-slice conversion costs 2.9–5.3x).
- **Public O(1) metadata.** `cutoff` and `fh` stay pandas `Index`/`ForecastingHorizon`. g5 rule 1 says they are acceptable
  as stated contract.
- **pandas input.** Every output layout stays as today, indefinitely (g4 recommendation).

---

## 3. a1 touchpoint categories → destination

a1 found 229 touchpoints: time/freq/cutoff 78, container ops 42, metadata, output construction, and 44 public-API contract
touchpoints.

| a1 category | today | this design |
|---|---|---|
| container ops (iterate, slice, concat, pivot; 42) | pandas groupby/concat in VectorizedDF | R1: none. Group-by happens inside the engine kernel. R2: positional iteration over materialized pandas/np_grouped (x6/g3 C4, which also cut RSS 1.4–2.2x). R3: dask partitions. Reconstruct = one long concat, then `emit` |
| time / freq / cutoff (78) | pandas index attributes, per slice | `SourceAdapter.cutoffs` as one engine aggregate (DN-B2: 16 ms polars at 10k) → `TimeAxis` (a3 Step) once per panel. fh→absolute once per unique cutoff (g3 C3). Kernels compute output times as `cutoff + h·step` in-engine |
| metadata / validation | `check_is_scitype` per slice | `guard` once, as engine expressions. polars validation is 29–75x cheaper than the stock check (design_hub guard bench; DN-B1 40–100x). Lazy guards run inside the plan. X/y containment is a key/time anti-join (g5 rule 4) |
| output construction | per-slice pandas outputs + concat + MultiIndex columns | kernels emit the long schema. `emit` builds pandas layouts only for pandas input (g4: long→MI 14–85 ms at 10k vs 205 ms concat today) |
| public-API value types (44) | DataFrame `forecasters_`, Index cutoff, MI proba columns, `"forecasters.loc[...]"` keys | §4 |

---

## 4. Public API: the four contracts

| contract | pandas input | non-pandas input | deprecation |
|---|---|---|---|
| **C1 `forecasters_`** | unchanged DataFrame of clones for R2. For R1 (native kernel), a lazily built **shim**: a DataFrame of lightweight per-key views over the `ParamsTable`, materialized on first access, with a `FutureWarning` that points to `params_` | R2: a pandas DataFrame indexed by keys (as today, O(n_keys) objects). R1: `params_` (a ParamsTable in the input engine; x7.5: 175x smaller than pickled clones) plus the same shim | N: `params_` added, shim warns on R1 only. N+2: the R1 shim's `.loc` access is the only supported legacy path. R2 is never changed (a1.5: metrics consume `forecasters_` as data) |
| **C2 `cutoff` / `fh`** | unchanged (`pd.Index` with freq; g5 rule 10) | unchanged type. Values come from the engine `cutoffs` aggregate. New `cutoffs_` (per-key table in the input engine) for ragged panels | additive only |
| **C3 proba layout** | MultiIndex columns kept indefinitely | config `output:proba_format` ∈ {"pandas_mi","long"}. Default "pandas_mi" in N with a FutureWarning for non-pandas input, flipped to "long" in N+2 (g4). R1 proba kernels produce long natively (g4 §3c: 16 ms vs 18.1 s at 1k) | as g4 |
| **C4 `get_fitted_params` keys** | `"forecasters.loc[i,c]__param"` kept for R2 | R1: `params_` columns as `"params__<col>"`, plus a long table via `get_fitted_params(style="table")` | N: new style. Old keys emitted via the shim for R1 with a warning |

New public surface (additive): `ReadSpec` arguments on `fit/predict/update` (`instances=`, `columns=` for the user-side filter),
the `output:lazy` config, `params_`, `cutoffs_`, and the `register_source` / `register_mtype` entry points. A public registry is
required: g2 found that private lru_cache mutation silently unregisters an mtype.

---

## 5. Estimator migration (≈160 forecasters)

| group (a2 buckets / x7 families) | n | what happens | who changes what |
|---|---|---|---|
| **K: engine-kernel candidates.** Naive (last/mean/drift/sp), PolynomialTrend, Trend(linear), SplineTrend?, Croston, TSB, ForecastKnownValues (x7 verifier strict 6–8) | 6–8 | add `_fit_panel`/`_predict_panel` per engine family (polars expr + duckdb SQL + numpy). Declare `lookback` structurally (Naive last=1, sp-naive=sp, mean=window_length, drift/trend=None) | maintainers of ~6 files. Per-kernel LOC ~74–116 per family (g2.8). Kernel conformance suite |
| **D: delegating composites.** TTF, ForecastingPipeline, Ensemble, Multiplex, ColumnEnsemble, ForecastByLevel, Fallback, FhPlex, IgnoreX, tuners | ~41 (x7) | `capability:panel_native` and `lookback` are **derived**: native iff every component is native and every transformer has an engine form. Otherwise R2 | composite base classes (~5) compute derived tags. **[DEL-1]**: 15+ composites wrapping Naive were truncation-equal at k=1, so derivation is feasible in practice |
| **V1/V2 per-instance, numpy core** | ~119 (a2: 74%) | **R2** unchanged: trusted lane (g1, 159/160 eligible). Optional later port to `np_grouped` (g2), which is reachable from every engine after one materialize | nobody at first. Ports are opt-in |
| **V3 pandas-algorithm** (reducers, HierarchyEnsemble) | 13 | R2. Reducers could build lag tables natively (window functions, x7 family table), but the regressor needs numpy, so it is deferred | nobody |
| **V4 third-party-bound / global DL** | 28 (+ the 29 pd-multiindex-native) | R2, or `materialize("polars")` passthrough for libraries that accept polars (darts/neuralforecast, a2 verifier: 8 classes; G6 unmeasured). Dask input goes via R3 partitions | adapter owners, opt-in |

Tags added: `capability:panel_native`, `panel_native:engines`, `lookback` (per-method dict allowed, e.g.
`{"predict": 1, "predict_interval": None}`). **Default lookback is None.** A wrong lookback gives wrong forecasts, so the tag
is set only by construction. **[DEL-1]** shows why it must not be inferred: AutoETS and ExponentialSmoothing looked "finite
(k=7)" on test data.

---

## 6. Phased roadmap with quantified expected impact

| phase | scope | expected impact (evidence) | risk |
|---|---|---|---|
| **P0. Decision-independent fixes** | vectorized polars monotonic check with null guard (x9.3); known-mtype pass-through; `_HashIndex.__eq__`; X/y containment for polars; pd3 `_enforce_index_freq` mutation | polars `convert_to` ≈10x (x9 verifier). End to end 1.05–1.34x (g3 C1). Correctness fixes for a1.10/a1.12/g5 S8 | low; the null-guard regression is known (x9 verifier 58/75) |
| **P1. VectorizedDF: positional iteration + trusted lane** (shared with other designs) | g1 trusted per-slice path; fh/cutoff once per unique cutoff (g3 C3); hub-style positional iteration (g3 C4) | unchanged forecasters: **1.6–1.9x Naive, 2.4–3.3x Poly** fit+predict (g3 C4). Whole workload 1.07–1.72x (g1). Peak RSS down 1.4–2.2x (g3 C4). All 160 forecasters, every container | drift between trusted and public logic (g1 caveat). Must share code |
| **P2. BaseForecaster boundary: SourceAdapter + ReadSpec + guards** | `describe/push/guard/cutoffs/materialize/emit` for pandas, polars, polars-lazy (new lazy Panel/Hier mtypes), duckdb, pyarrow(+Dataset), dask; column projection always; `instances=` filter | non-pandas ingest: polars validation 29–75x cheaper than the stock check (design_hub); stock rejects LazyFrame today (x9.10), so lazy becomes **accepted**. Selective reads: polars lazy 875x / duckdb 360x vs eager sktime (T6). Projection alone: **1.25–1.68x time, 1.6x RSS** vs read-all **[DEL-2]**. For R2 forecasters, e2e stays ≈ P1 numbers (g3: laziness ±15%) | new mtype family. narwhals pinning avoided (native adapters). Duckdb connection binding (x1.3) |
| **P3. Engine-native kernels (K group) + derived tags (D group)** | `_fit_panel/_predict_panel/_predict_interval_panel/_update_panel` in polars/duckdb/numpy for Naive, PolynomialTrend, Trend, Croston, TSB, ForecastKnownValues; composites derive the capability | K forecasters: **10²–10³x** (g3 C5 106–4,560x; x7.8 570x pandas/2,006x polars at 1k; g4 interval 1,100x). Order-robust guard costs only **1.11–1.38x** of the unguarded kernel **[DEL-3]**. Fitted state 175x smaller (x7.5). Lazy R1 from parquet: 123 ms vs 145 s (x7.4) | small coverage (6–8 strict, x7 verifier). Per-engine kernels triple the test matrix. Quirk parity (sklearn cond=tol truncation, g2) |
| **P4. Lookback pushdown + partition delegation** | `lookback` tag on the K group plus derived composites (≈22 runnable classes, **[DEL-1]**); R3 for dask (key-aligned `map_partitions` of the R2 path, declared output schema) | Lookback: **duckdb 12–24x lower peak RSS and 1.9–2.6x faster** on series-major files, but only **1.6–3x RSS / 0.7–1.05x time on time-major** files **[DEL-2]**. polars lookback ≈ projection (1.1–1.6x) **[DEL-2]**, and x6 §3.7 finds no RSS gain. Dask: partition delegation 0.70–1.31x serial on 4 shared cores, 2.9–3.7x faster than today's per-instance dask backend **[DEL-4]**. Main value is out-of-core and cluster scale (*unmeasured*) | lookback misdeclaration means wrong results. Single-box dask gives no speed (**[DEL-4]**). Key dtype changes (object → string[pyarrow]) **[DEL-4]** |
| **P5. Contract flips (N+2)** | `output:proba_format="long"` default for non-pandas; R1 `forecasters_` shim to `params_`; `get_fitted_params(style="table")` | proba egress 14–85 ms vs 205 ms concat at 10k (g4). User-visible break only for non-pandas or R1 users | downstream `.loc` users (a1.5: 4/4 hierarchical metric configs break without the shim) |

Cumulative expected outcome:
- pandas users with unchanged forecasters: 1.6–3.3x (P1).
- non-pandas users: the same, plus the stock polars penalty removed (x8: polars 1.04–1.18x slower than pandas today), plus
  lazy/duckdb input accepted.
- K-group forecasters on any engine: 10²–10³x, with outputs that stay lazy.
- Selective reads from files: 10²x (T6) through ReadSpec, regardless of route.

---

## 7. Evidence

### For
1. x7.1/.8 and g3 C5: kernels give 10²–10³x. x7.4: one lazy plan from parquet gives 123 ms / +31 MB vs 145 s / +483 MB. This
   is only reachable if the engine runs the forecaster.
2. T6 (final_bench): selective lazy read 875x / 25x memory vs eager sktime. ReadSpec captures it for **every** route.
3. g4 §3c: proba kernels give 1,100x and emit the long schema natively. Distribution concat drops from 826 → 3.2 ms.
4. **[DEL-3]**: the order-robust contract is cheap (1.11–1.38x) and correct on shuffled input. It closes the x7/r4 silent-wrong
   hazard without an expensive sortedness pass.
5. **[DEL-2]**: duckdb pushdown with an in-SQL kernel used 52–105 MB peak vs 1.27 GB eager (12–24x) on 8M rows, with
   identical results.
6. x7.5: the params table is 175x smaller than per-instance clones, which matters for memory and pickling at 10⁴+ series.
7. r3: the Nixtla split (aggregates/baselines in the engine, models on a grouped array, coarse partitions for distributed)
   is the same split as R1 / R1-numpy+R2 / R3.

### Against (strongest first)
1. **Coverage is small.** Only 6–8 forecasters are strict kernel candidates (x7 verifier). Everything else gets ≤1.05x from
   the engine (x8). Its real gains come from the trusted lane (P1), which is container-agnostic and not lazy-first at all
   (g1, g3 C2–C3). The design's headline lever helps ~5% of classes.
2. **Laziness gives nothing for R2 forecasters.** g3: lazy input is within ±15% of eager polars, and 1.1–2.6x slower at C5.
   DN-B4: lookback pushdown is *slower* than collect-all on 1M-row polars (0.23–0.57x).
3. **Polars lookback pushdown does not cut memory.** **[DEL-2]**: lazy+lookback peaked at 749–761 MB vs 792–803 MB for
   projection only. x6 §3.7: no RSS gain either. Only duckdb benefits, and only on series-major files. On time-major files
   duckdb was 0.71–1.05x eager polars in time **[DEL-2]**.
4. **Finite lookback is rare and fragile.** **[DEL-1]**: 1/39 for intervals. 3 of 26 "finite" point lookbacks were data
   coincidences, so a mis-declared tag silently changes forecasts.
5. **Dask delegation is not a speed lever on one box.** **[DEL-4]**: best 1.31x on 4 cores, and P=4 was *slower* than serial
   (0.70–0.80x, worker start-up and sktime import). r3 E4: per-group Fugue 203–610x slower. Benefits for out-of-core data
   are *unmeasured*.
6. **Per-engine kernels multiply maintenance**: 3 engine families × K forecasters × methods (fit/predict/interval/update).
   The narwhals alternative is unsafe (x7.11 duckdb `total_seconds` bug) and slower (2.2–3.0x for order-dependent `last`).
7. **Fidelity across engines is a recurring hazard**: dask key dtype object → `string[pyarrow]` **[DEL-4]**, duckdb results
   unbound from the user's connection (x1.3), multi-chunk Arrow copies (x9 verifier), and C5 output time-level freq=None (g3).
8. **The engine gain shrinks to 1.07–1.33x** when the data is not already native (x7 verifier). pandas users gain only through
   R1-numpy (g2), which is a hub/numpy design, not an engine one.

---

## 8. Open experiments
1. **Cold or remote storage pushdown** (S3/HTTP parquet, page cache dropped). DEL-2 and x9 are warm-local, so I/O savings from
   row-group skipping are *unmeasured*. That is where lookback and instance filters should pay most.
2. **Out-of-core dask/duckdb**: data larger than RAM (e.g. 200M rows) through R3 vs R2. Measure the break-even and the peak
   memory on a multi-worker `distributed` cluster (G7). DEL-4 covers only small, in-memory, single-box data.
3. **Kernel conformance suite** (M4 fixtures × {polars, polars-lazy, duckdb, numpy} × K forecasters × 4 methods), including
   ragged ends, NaN, Period/int time, tz, unnamed levels.
4. **Derived capability for composites**: implement the `panel_native` and `lookback` derivation for TTF + Deseasonalizer +
   Naive. Measure the share of real pipelines that become native.
5. **Polars streaming + lookback memory** on a 50–100M row file. Does `engine="streaming"` start cutting RSS at a scale where
   in-memory polars does not? (DEL-2 streaming: 1.3–2.1x time, no RSS gain.)
6. **Lazy update path**: `_update_panel` upserting the params table from a lazy delta (`instances`/`time_range` ReadSpec) vs
   refit.
7. **Lookback census in a full-dependency env**: DEL-1 covered 70/160 classes, missing statsforecast, pmdarima, torch.

---

## 9. Measurements performed for this report

Env: `EXP_PY` (pandas 2.3.3, polars 2.0.0, duckdb 1.5.6, dask 2026.8) and, for DEL-1, the g1 venv
(`scratchpad/venvs/g1_trusted_path`: same versions + statsmodels 0.15.0). Shared 4-core box with another agent, load ≈0.4–1.
Scripts are in `experiments/container_independence/scripts/`.

**[DEL-1] Empirical lookback census.**
- Command: `PYTHONPATH=.:experiments/container_independence $G1PY -W ignore scripts/design_engine_lazy_lookback_census.py`
- Output: `results/design_engine_lazy_lookback_census.{json,log}`. Wall 124 s.
- Method: fit on full y (monthly, T=96, trend + season + noise) vs `y.tail(k)`, k ∈ {1,2,3,4,7,8,12,13,14,24,25,36,48,72}.
  2 seeds, fh=1..3, `allclose(rtol=1e-9)`. Same for `predict_interval(0.8)`.
- Results:
  - 160 registered, 70 runnable, 90 errors (89 missing soft dependencies, 1 freq error), 0 timeouts.
  - **26/70 finite for predict**: k=1: 17, k=2: 3, k=7: 5, k=12: 1.
  - Finite: Naive(last) and composites wrapping it (ColumnEnsemble, Ensemble, Fallback, FhPlex, ForecastByLevel, IgnoreX,
    Multiplex, OnlineEnsemble, Permute, ThetaModular, TTF, ForecastingPipeline, GroupbyCategory, TransformSelect,
    NaiveVariance), ForecastKnownValues 1, PluginParams(sp=12) 12, AutoEnsemble 2, ConformalIntervals 2, Stacking 7,
    GridSearch/RandomizedSearch 7.
  - Data coincidences (statistical fits with no structural finite lookback): AutoETS 7, ExponentialSmoothing 7,
    UnobservedComponents 2.
  - Intervals: 1/39 finite (BoxCoxBiasAdjusted, k=2, also suspicious).
  - All reducers, trend, Theta, ARIMA-family, Croston/TSB (with default params) need the full history.

**[DEL-2] Fresh-process pushdown + order-robust kernel.**
- Command: `$EXP_PY -W ignore scripts/design_engine_lazy_pushdown.py --make 20000 400`, then `--sweep 20000 400`.
- Output: `results/design_engine_lazy_pushdown_20000x400.{json,log}`. Tables: `python3 scripts/design_engine_lazy_tables.py`.
- Data: 8M rows; key, time, y, 4 unused floats; 291 MB parquet; series-major and time-major layouts.
- Workload: panel-native Naive(last) / Naive(mean, w=10), fh=1..14, 280k output rows.
- Method: one process per cell, imports done before the baseline, `ru_maxrss` delta, min of 2 reps. **All 64 cells have
  checksums identical to the reference.**

| layout, strategy | pl eager all | pl eager proj | eager proj+select | pl lazy plan | pl lazy+lookback | pl lazy streaming | duckdb SQL (+QUALIFY) | pandas read+groupby |
|---|---|---|---|---|---|---|---|---|
| series-major, last: s / MB | 0.681 / 1268 | 0.405 / 803 | 0.517 / 758 | 0.479 / 792 | 0.426 / 749 | 0.351 / 793 | **0.261 / 52** | 3.92 / 766 |
| series-major, mean10 | 0.830 / 1272 | 0.664 / 802 | 0.626 / 725 | 0.618 / 794 | 0.656 / 760 | 0.388 / 794 | **0.427 / 105** | 6.65 / 1166 |
| time-major, last | 0.621 / 1301 | 0.444 / 807 | 0.496 / 767 | 0.503 / 798 | 0.552 / 759 | 0.471 / 797 | 0.590 / 442 | 4.52 / 858 |
| time-major, mean10 | 0.619 / 1305 | 0.442 / 810 | 0.550 / 765 | 0.506 / 799 | 0.505 / 761 | 0.447 / 799 | 0.871 / 826 | 7.49 / 1222 |

- Ratios vs polars eager read-all:
  - projection: 1.25–1.68x time, 1.58–1.61x RSS;
  - lazy+lookback: 1.12–1.60x time, 1.67–1.72x RSS;
  - streaming: 1.32–2.14x time;
  - duckdb: 1.94–2.61x time and 12–24x RSS on series-major, but 0.71–1.05x time and 1.6–3.0x RSS on time-major;
  - pandas read+groupby: 0.08–0.17x.
- Caveat: warm page cache, local disk.

**[DEL-3] Guard cost and shuffled correctness.**
- Command: `$EXP_PY -W ignore scripts/design_engine_lazy_pushdown.py --guard 20000 400`
- Output: `results/design_engine_lazy_guard_20000x400.json`.
- In-process, min of 3, 8M rows eager polars:

| strategy | positional (x7-style) | order-robust | robust / positional | separate sortedness check | positional on shuffled | robust on shuffled |
|---|---|---|---|---|---|---|
| last | 0.146 s | 0.200 s | 1.38x | 0.773 s | **wrong** | equal |
| mean10 | 0.173 s | 0.192 s | 1.11x | 0.701 s | **wrong** | equal |

**[DEL-4] Partition delegation vs stock parallel backends.**
- Command: `$EXP_PY -W ignore scripts/design_engine_lazy_dask_delegate.py 400 100 {naive,poly}`
- Output: `results/design_engine_lazy_dask_{naive,poly}_400x100.json`, `results/design_engine_lazy_dask.log`.
- Single run each, 4 workers, processes scheduler, shared box, so noisy.

| est (serial s) | loky n_jobs=4 (stock per-instance) | dask per-instance (stock) | delegate P=4 | P=16 | P=64 |
|---|---|---|---|---|---|
| Naive last (7.68) | 0.93x | 0.35x | 0.80x | **1.31x** | 1.04x |
| PolyTrend (5.03) | 0.96x | 0.34x | 0.70x | **1.03x** | 0.75x |

- Values and index are equal to serial in all cells after casting keys back.
- Delegate outputs come back with keys as `string[pyarrow]` instead of object, a fidelity change that `emit` must undo.
