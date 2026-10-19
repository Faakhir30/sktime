# design_tiered — tiered per-estimator opt-in on the inner-mtype tag system

Architect angle: **TIERED PER-ESTIMATOR OPT-IN.** The existing `y_inner_mtype` / `X_inner_mtype` tags stay the
one mechanism by which a forecaster says what it computes on. The framework adds new inner mtypes (numpy grouped,
numpy series, and frame-native passthrough) and a trusted fast lane that every unchanged forecaster gets for free.
Estimators then migrate one by one, as sklearn does with Array API, gated by a tag plus an equality harness.

Every number is cited as `report-key: claim`, or comes from the one script I ran for this report
(`scripts/design_tiered_census.py`, §8). Anything else is labelled *unmeasured/expected*. Nothing under `sktime/`
was modified.

---

## 0. Thesis

The evidence says that speed comes from removing per-instance framework work and from batched kernels. It does not
come from the container (x8: ≤1.05x container-only; g3: C4 adds 0.93–1.14x). So the architecture should be built
around **what each estimator computes on**, which is exactly what `y_inner_mtype` already expresses.

The design has four tiers:
- **T1 (trusted pandas slices):** the default for all 159 eligible classes. It needs no estimator change and gives
  1.6–3.3x (g3 C4).
- **T2 (`np_series`):** per-series numpy views, about 6–9x for per-series fitters (g2 reducer).
- **T3 (`np_grouped`):** batched panel kernels, 10²–10³x (g2, g3 C5).
- **T3p (frame passthrough):** for wrapped libraries that natively take polars or arrow.

Containers enter and leave once, through spokes. Time lives in a framework `TimeAxis`.

---

## 1. What the evidence forces on a tiered design

| evidence | consequence |
|---|---|
| g1: the trusted per-slice path gives fit+predict 1.15–1.88x and workload 1.07–1.72x, and is 100% state-identical. 7710/7710 tests pass, and 159/160 classes are eligible | **T1 is the default floor**, not an opt-in. It is the only lever that reaches every class, so it ships before any port. |
| g3: C1→C3 multiply (Naive 1.11×1.31×1.12, Poly 1.22×1.45×1.53). After C3 the time is inner code (nested Lag 63%, sklearn fit 63%). C4 (hub iteration) adds ≈1.0x in time but 1.4–2.2x lower RSS | The framework's iteration substrate can change under T1 without touching estimators. To go faster than that, the estimator itself must change, so a tier opt-in is needed. |
| g2: an inner `np_grouped` with **only a tag change** in the forecaster, plus 2 framework dispatch points (`get_cutoff`, `update_data`), matches stock 315/315 and 233/233. It does 2 conversions instead of ~800 and 36 checks instead of 7,484–12,084 | The tag system **already works as the opt-in mechanism**. The missing pieces are a public mtype registry and backend-aware `get_cutoff`/`update_data`. |
| g2: the reducer gains only 6–9x. g1: ETS gains ≈1.0x (165–300 ms/series inside statsmodels) | Tiers above T1 pay only where the estimator's own per-series work is cheap relative to framework work. Prioritise ports by that ratio, not by class count. |
| g2: on polars input, 99% of g2 time is the two stock `polars_panel` checks; on pandas/pyarrow, 18–60 ms for 10k series | Ingress must be fixed (P0/P1) **before** ports. Otherwise polars users see 34–105x where pandas users see 498–2,735x. |
| a2: V1+V2 = 74% need no core-math change; V3 = 13 classes; V4 = 28 bound to third-party formats | Most classes can reach T2 by mechanical glue edits. V4 classes stay at T1, or move to T3p where their library takes polars or arrow. |
| r1: sklearn Array API is a global switch, a dynamic per-estimator tag derived for composites, and a test matrix keyed on the tag. It reached 19/210 estimators in ~3.5 years | Copy the governance (dynamic tag, conformance harness, global escape hatch). Do **not** count on wide coverage from ports: T1 must carry the population. |
| g4: proba is dominated by per-slice public re-entry (Naive interval 18.1 s vs var 3.1 s at 1k). The x7-style native kernel takes 16 ms vs 18.1 s; the long schema round-trips exactly | T3 forecasters must provide native proba (long + `ProbaSpec`). Otherwise proba falls back and erases the gain (x7 hook: 3.58 s vs 5.6 ms). |
| g5: x4 lost the Panel disjoint-X check and the y name. The x6 hub round trip dropped the time-level freq and broke pandas `update_predict` | Validation runs once at top level on the canonical form, and freq is carried as `Step` metadata (g5 invariants 3, 4). The g5 conformance matrix gates every tier. |

---

## 2. Architecture

### 2.1 Tiers (what the estimator declares → what the framework does)

| tier | declared by | estimator receives | framework loop | conversions per call | who |
|---|---|---|---|---|---|
| **T0 legacy** | `capability:trusted_vectorize=False` (class), config `trusted_vectorize=False` (user), or a guarded override of a public method | pandas slices via public `fit/predict` per slice (today) | stock | ~N+N (g2: 500+304 at 100 series) | ChronosForecaster (g1 census), `update_predict_single`, `predict_proba` until P2 |
| **T1 trusted pandas** (default) | nothing: any pandas `y_inner_mtype` | a pandas Series/DataFrame view with trusted freq, built from the GroupedArray (24–27 µs/slice, final_bench T1) | framework iterates positionally over the GroupedArray and calls `_fit_trusted/_predict_trusted/_update_trusted` per slice | 1 ingress + 1 egress, plus a pandas view per slice | 123 of the 124 vectorized classes |
| **T2 numpy series** | `y_inner_mtype=["np_series", ...]` | `SeriesArray` (values (T,C) view + `TimeAxis` view), O(1) per slice (0.6–5 µs, final_bench T1 hub view) | same positional loop, but no pandas object per slice | 1 + 1 | V1/V2 per-series fitters (reducers, window forecasters, statsforecast models) |
| **T3 numpy panel** | `y_inner_mtype=["np_grouped", "np_grouped_hier", "np_series"]` | the whole `GroupedArray`: batched `_fit/_predict` | none; no vectorization (all scitypes are inner) | 1 + 1 (g2 counts) | closed-form: Naive, PolynomialTrend, about 9 candidates (x7) |
| **T3p frame passthrough** | `y_inner_mtype=["polars_panel", "pa_panel", "pd-multiindex", ...]` | the user's container if it is listed, otherwise the first listed | none | 0 if listed (identity), else 1 + 1 | global and third-party models that take polars or arrow (darts, neuralforecast, statsforecast batched; G6 unmeasured) |

The pd-multiindex-native classes (36 by my census rule, 29 by a2's rule) are **T3p with pandas**. They benefit from
P0/P1 ingress fixes only.

**Inner selection rule (single function, replaces the implicit "first in list" logic in `convert_to`):**
1. If the input mtype is in the list, use identity (no conversion).
2. Otherwise, under config `inner_dispatch="auto"`, choose the cheapest entry by converter cost class
   (`view` < `vectorized` < `python_loop`), taken from the registry. Example: polars input with
   `["pd.DataFrame", "np_series"]` → `np_series`.
3. Otherwise (`inner_dispatch="legacy"`, the default in release N), use the first listed entry of the needed
   scitype, which is today's behaviour.

This is the sklearn `array_api_dispatch` pattern: a global switch, so users can turn the new inner path off for one
release.

### 2.2 Components and protocols (signatures)

```python
# sktime/datatypes/_registry.py  (new, public)  -- g2 §1.1: today only private lru_cache surgery works
def register_mtype(cls: type["BaseDatatype"], *, converters: Mapping[tuple[str, str], Callable],
                   cost: Mapping[tuple[str, str], Literal["view", "vectorized", "python_loop"]] = {}) -> None: ...
def unregister_mtype(name: str) -> None: ...
# also discovered from an entry-point group "sktime.mtypes"; caches are keyed on a registry version counter,
# so cache_clear() can never silently unregister (g2 measured that failure)

# sktime/datatypes/_time/_axis.py  (a3 Step + NumpyTimeOps, promoted)
@dataclass(frozen=True)
class Step:                       # kind in {"fixed","month","bday","cbday","int","period"}, n, anchor, holidays, tz
    ...
@dataclass(frozen=True)
class TimeAxis:
    time: np.ndarray              # datetime64 (unit kept from source; g3 pd3 bug) or int64 ordinals, len n_rows
    step: Step | None             # None => irregular: any step arithmetic raises (g2 mixed-freq behaviour)
    kind: Literal["datetime", "period", "int", "range"]
    freqstr: str | None           # source alias kept verbatim (g3 pd3 'Day' vs '24h'; a3 QS-JAN)
    def cutoffs(self, indptr: np.ndarray, *, how: Literal["max", "min"] = "max") -> np.ndarray   # g5 A4
    def add_steps(self, base: np.ndarray, k: np.ndarray) -> np.ndarray     # (U,)x(H,) -> (U,H)
    def steps_between(self, a: np.ndarray, b: np.ndarray) -> np.ndarray
    def to_pandas_index(self, values: np.ndarray) -> pd.Index            # egress and public cutoff only
    @classmethod
    def infer(cls, time: np.ndarray, indptr: np.ndarray) -> "TimeAxis"   # all instances, numpy, masked diffs

# sktime/datatypes/_grouped/_container.py  (g2 NpGrouped, promoted)
@dataclass(frozen=True)
class KeyTable:
    names: tuple[Hashable | None, ...]       # unnamed levels allowed (g4 3a)
    codes: tuple[np.ndarray, ...]            # one per level, len U
    levels: tuple[Sequence, ...]
@dataclass(frozen=True)
class GroupedArray:                          # mtypes "np_grouped" (Panel), "np_grouped_hier" (Hierarchical)
    values: np.ndarray                       # (n_rows, C), view of source where possible
    indptr: np.ndarray                       # (U+1,)
    keys: KeyTable
    axis: TimeAxis
    columns: tuple[Hashable, ...]            # original labels (fixes g5 S7 / A2 y -> "0")
    egress: "EgressMeta"                     # enough to rebuild the source container exactly
    def series(self, i: int) -> "SeriesArray": ...          # O(1) view
    def take(self, idx: np.ndarray) -> "GroupedArray": ...
@dataclass(frozen=True)
class SeriesArray:                           # mtype "np_series"
    values: np.ndarray; axis: TimeAxis; columns: tuple[Hashable, ...]; key: tuple | None

# sktime/datatypes/_spokes/_base.py  -- one spoke per outer container; frame-level only, never per slice
class ContainerSpoke(Protocol):
    mtypes: tuple[str, ...]                 # e.g. ("polars_panel", "polars_hierarchical", "pl.DataFrame")
    lazy: bool
    def check(self, obj, scitype: str) -> "SpokeMeta": ...        # vectorized; null-guarded (x9 verifier)
    def ingress(self, obj, meta: "SpokeMeta") -> GroupedArray: ...
    def egress(self, out: GroupedArray | "LongTable", like: "EgressMeta") -> Any: ...
    def select(self, obj, sel: "Selection") -> Any: ...           # lazy only: push instances/columns/lookback
    def collect(self, obj) -> Any: ...                            # lazy only: exactly once, backend-native
@dataclass(frozen=True)
class Selection:
    instances: np.ndarray | None; columns: Sequence | None; lookback: int | None   # from estimator tag

# sktime/forecasting/base/_boundary.py  -- public methods become validate-once + core
@dataclass
class CallContext:
    method: str
    y_inner: Any; X_inner: Any | None        # in the selected inner mtype
    ga: GroupedArray | None                  # canonical form when y is Panel/Hierarchical
    y_meta_template: dict                    # one check per distinct column set (g1)
    cutoffs: np.ndarray                      # per instance, numpy
    cutoff: pd.Index                         # 1 element, freq from Step (public type unchanged)
    fh: "ForecastingHorizon"; fh_cache: dict # to_absolute once per unique cutoff (g3 C3)
    tier: Literal["T0", "T1", "T2", "T3", "T3p"]
    egress: "EgressMeta"

class BaseForecaster:
    # trusted entries: same state writes as the public method (g1 invariant 1), shared code, not copies
    def _fit_trusted(self, y_inner, X_inner, fh, ctx: CallContext, i: int) -> "BaseForecaster": ...
    def _predict_trusted(self, fh, X_inner, ctx: CallContext, i: int, method: str = "predict", **kw): ...
    def _update_trusted(self, y_inner, X_inner, ctx: CallContext, i: int, update_params: bool = True): ...
    # helpers for T2/T3 authors (replace hand-built pandas output indexes; a2: 128/160 use fh/cutoff)
    def _fh_grid(self, fh, ctx) -> np.ndarray: ...                 # (U,H) absolute times, numpy
    def _make_output(self, values: np.ndarray, ctx) -> GroupedArray: ...
    def _make_proba_output(self, table: "LongTable", spec: "ProbaSpec") -> "LongTable": ...

# public fitted-state container (a1 EstimatorGrid, promoted)
class FittedGrid:
    keys: KeyTable; columns: tuple | None
    def __getitem__(self, ij) -> "BaseForecaster | InstanceView": ...
    def to_pandas(self) -> pd.DataFrame: ...                       # today's forecasters_ layout
    def params_table(self, backend: str = "pandas"): ...           # T3: parameter arrays as a table
```

The **T3 author contract** keeps the signatures unchanged: `_fit(self, y, X, fh)` and `_predict(self, fh, X)`. Only
the inner types differ. Tags:
- `y_inner_mtype` (the opt-in);
- `proba_inner_format: "pandas" | "long"` (g4);
- `lookback_window` (int, `"sp"`, or None = all; used for pushdown, see §2.4);
- `capability:trusted_vectorize` (default True).

### 2.3 Where each a1 pandas-touchpoint category goes (229 touchpoints, a1 §2)

| a1 category (touchpoints) | today | in this design |
|---|---|---|
| container ops: group/iterate 6, slice/select 12, concat/reconstruct 9 | pandas groupby/xs/concat in `_vectorize.py` | `GroupedArray.indptr` views (T1/T2/T3) plus spoke `ingress/egress`. pandas ingress reads MultiIndex codes zero-copy (x6, g3 C4). |
| pivot/reshape, estimator grid (15) | `forecasters_` pivot | `FittedGrid` (a1: 4–82x faster build, ≲1% of fit) |
| time index / freq / cutoff (78, 55 of them in `_utilities.py`) | pandas Index, `freq`, `infer_freq` per slice | `TimeAxis` + `Step` once per panel. `cutoffs()` in numpy (min and max). `fh_cache` per unique cutoff. pandas only for the 1-element public cutoff and for exotic steps (fallback). |
| MultiIndex levels/names (23) | `droplevel`, `set_names` | `KeyTable` (names may be None). Egress rebuilds exact names, with no `__index__` prefix leaking (g5 H2). |
| metadata: columns/dtypes/nans (13) | per-slice `check_is_scitype` | `SpokeMeta` from one vectorized check; per-slice `has_nans` from numpy (g1) |
| output construction (14) | per-instance `pd.MultiIndex.from_product` (`_get_columns`, 0.7–0.8 ms/instance, g4) | `_make_output` / long + `ProbaSpec`, built once at egress (g4: 14–85 ms at 10k vs today's concat 205 ms + `_get_columns` 7.9 s) |
| type dispatch, isinstance/duck typing (34) | `hasattr(obj, "X")`, `hasattr(obj, "index")` | an explicit spoke registry keyed on python type. Duck typing removed (g5 S4, S6; a1.10/11). |
| mtype pinning (25) | `convert_to(..., "pd-multiindex")` | the inner selection rule (§2.1). The target comes from the estimator's tag, not a hard-coded string. |
| public API value types (44 contract touchpoints) | pandas | see §3 |

### 2.4 Data flow by input container

| input | ingress | T1 (unchanged estimator) | T3 (ported) | output |
|---|---|---|---|---|
| pandas MultiIndex | MultiIndex codes → GroupedArray (x6 ingress 0.06–0.28x of prepare, final_bench T3). Validation once, including Panel X/y containment by (key, time) anti-join (g5 inv 4) | pandas views with trusted freq (24–27 µs) → trusted entries | kernel on GroupedArray | pandas, bit-identical layout (time-level freq restored from `Step`, unlike g3 C5/x6 H1) |
| polars eager | vectorized, null-guarded check (x9: 921→33 ms) → GroupedArray. No pandas MultiIndex is built (g3 C4) | pandas views per slice (only T1 pays pandas) | kernel; no pandas objects (g2 counts: 0 DataFrame/Series) | polars, original names |
| polars lazy / duckdb | `select(Selection)` with `lookback_window` and instance and column filters pushed into the plan, then `collect()` **once** in the backend's native eager type (x1 C1: duckdb→pyarrow), then the arrow or polars spoke | as eager | as eager. Optional later: single-plan expression kernels for the x7 subset | eager in the collect backend (outputs are O(n·h)); docs state that lazy-in → eager-out |
| pyarrow | `pa_*` mtypes (g2) → GroupedArray | as eager | kernel | pyarrow |
| dask | partition-and-delegate: `set_index(sorted)` + `map_partitions`, running the **same tier** per partition on a pandas spoke. Instances never straddle partitions | per partition | per partition | dask (r4 verifier: 1.2–1.5x like-for-like; break-even is G7, unmeasured) |

Rule: **no spoke op runs inside a per-instance loop, and no lazy object is iterated** (x4 Z / x1: 3–77x and
170–540x penalties).

---

## 3. Public API: the four contracts (a1 C1–C4) and how they evolve

| contract | release N (opt-in tiers ship) | N+1 | N+2 |
|---|---|---|---|
| **C1 `forecasters_`** (pd.DataFrame of fitted clones) | T0/T1/T2: unchanged DataFrame, built lazily from `FittedGrid` on first access. T3: `forecasters_` is a DataFrame of `InstanceView` objects (`get_fitted_params()`, `cutoff`, `predict` for its key, backed by parameter arrays). It raises on `fit`/`update`. New `fitted_grid_` everywhere | `FutureWarning` when `forecasters_` is accessed through DataFrame-only API (`.iloc`, `.values`, `.pivot`); metrics move to `fitted_grid_` (a1: 4/4 hierarchical metric configs depend on the DataFrame) | `forecasters_` returns `FittedGrid` (supports `.loc[key, col]`, `to_pandas()`) |
| **C2 `cutoff` / `fh`** | Public type stays a 1-element pandas Index and `ForecastingHorizon`. **freq is now always populated from `Step`**: g2 saw freq differ from stock in 81/315 (pd2) and 9/315 (pd3) cases, where stock returns None inconsistently. This ships as a documented bug fix. New `cutoffs_`: per-instance cutoffs as a frame in the input backend | — | No change planned. Cutoff and fh are O(1)/O(H) metadata, and pandas there costs nothing measurable (g2 profile: `get_cutoff` 12.5% of a 25 ms run) |
| **C3 proba columns** (MultiIndex var/coverage/lower-upper) | g4 plan: pandas input keeps the MultiIndex **indefinitely**. Config `output:proba_format="pandas"\|"native"`, default pandas; `FutureWarning` for non-pandas input. T3 leaves may return long + `ProbaSpec` (`proba_inner_format="long"`). `proba_to_long` / `long_to_proba` utilities ship | metrics and `evaluate` read the long form | default `"native"` for non-pandas input only. `predict_var` keeps the predict layout. `predict_proba` stays a skpro object, plus `to_long()` |
| **C4 `get_fitted_params` keys** (`"forecasters.loc[...]"`) | unchanged for T0–T2. T3 returns batched arrays (`coef_` (U,p,C)) plus a `keys` entry. Config `fitted_params:per_instance=True` expands them into today's keys on demand | warn when per-instance keys are expanded for more than 1k instances | default False for T3 only |

Also fixed at the boundary, since every tier needs them (g5 invariants 1–7):
- `evaluate` returns the input backend (S9);
- `update` works with 1–2 rows (S8);
- Panel `predict_residuals` becomes an aligned (key, time) join (S3);
- `update_predict` slices windows on the GroupedArray and converts once after formatting (S1, A3).

---

## 4. Estimator migration (160 classes; a2 buckets; census §8)

**Nobody has to change anything for T1.** 123/124 vectorized classes and all composites get the trusted path through
the class tag default. ChronosForecaster falls back to T0, because it overrides public `predict` (g1 census).
Third-party extensions get T1 automatically unless they override a guarded method (g1 invariant 2).

| bucket (a2) | n | target tier | change required | expected gain over T1 |
|---|---|---|---|---|
| V3 core rewrite: Naive, reducers, ForecastKnownValues, … | 13 | Naive → T3; reducers → T2 (after the `_sktime.py:151` last-window fix, which covers 7–8 classes) | rewrite `_fit/_predict` (g2 LOC: Naive 116 vs 358, reducer 40 vs 314 but narrower scope) | Naive 498–2,735x vs stock (g2); reducer 6–9x vs stock (g2) ≈ 4–5x over T1 (*indicative*: g1 reduce fit+predict 1.61–1.68x is a cross-report comparison) |
| V1 boundary-only (numpy core) | 63 | T2, or T3 where batched maths exists | swap `y.index` / `.values` glue for `SeriesArray` + `_fh_grid` / `_make_output` | high only for cheap per-series fits. statsmodels-family classes stay ≈1.0x over T1 (g1 ets) |
| V2 glue rewrite | 56 | T2 | positional replacements of `.loc` / `reindex` | as V1 |
| V4 third-party pandas-bound | 28 | stay at T1 (vectorized) or T3p (native panel) | none; optionally list `polars_panel` where the library takes polars (darts, neuralforecast: 8 classes, a2 verifier) | T3p passthrough unmeasured (G6) |
| pd-multiindex-native (global/DL) | 29–36 | T3p pandas | none | P0/P1 ingress only |
| composites (61 by a2; 27 flagged by my MRO rule) | — | **derived tag** (r1 sklearn pattern): delegators pass the child's inner mtypes through; pipelines take the intersection of the transformer and forecaster capabilities; tuners run T1 per fold on the GroupedArray | framework code in `_DelegatedForecaster`, `_Pipeline`, `BaseGridSearch` | transitive. TTF stays T1 until transformers have np inner types (g5 inv 8) |

**Porting leverage is a long tail** (my census, §8). The 124 vectorized classes have 96 distinct `(_fit owner,
_predict owner)` implementations:

| implementations ported | 1 | 5 | 10 | 20 | 50 | 96 |
|---|---|---|---|---|---|---|
| classes covered | 8 (6.5%) | 24 (19%) | 35 (28%) | 48 (39%) | 78 (63%) | 124 (100%) |

The best shared units are:
- `_GeneralisedStatsForecastAdapter`: 8 classes as a single implementation (9 in the MRO). Its library is
  numpy-native and has its own GroupedArray (r3), so it is the strongest T2/T3 candidate after Naive.
- `_StatsModelsAdapter`: 12 classes, but low gain, because the cost is inside statsmodels.
- `BaseDeepNetworkPyTorch`: 9.
- `_BaseWindowForecaster`: 9.
- `_Reducer`: 8.

**Port order**, by expected gain × classes covered:
1. Naive and PolynomialTrend (T3; done as g2 protos);
2. the `_BaseWindowForecaster` last-window fix + reducers (T2);
3. the statsforecast adapter (T2, then T3 through StatsForecast's own batched API, *unmeasured*);
4. Theta, Croston and the other closed-form models (T3, about 9 per x7);
5. everything else stays T1 unless a profile shows framework share > 50%.

**Gate for each port** (the Array-API-style test matrix):
- A generalised `g2_equality` harness compares the T2/T3 forecaster with its own T1 run on pandas input
  (predict, interval, update, absolute fh, ragged). In g2 this found every quirk: `cond=tol`, the drift slope, the
  frozen window.
- The g5 conformance matrix runs per backend.
- The adversarial suite M4 (wrap_critic) must pass: NaN keys, unsorted rows, mixed freq, unnamed levels.
- Once the harness passes, the old pandas implementation is **deleted**. Two maintained implementations is the
  failure mode to avoid. The global `inner_dispatch="legacy"` switch is a one-release escape hatch, implemented by
  converting to pandas and running the T1 route of the same new code where possible.

---

## 5. Phased roadmap (VectorizedDF → BaseForecaster boundary → estimators)

| phase | scope | expected impact (cited) | risk |
|---|---|---|---|
| **P0 decision-independent fixes** (1 release) | Vectorized, null-guarded polars monotonic check. Pass the known `from_type` instead of re-inferring. `_HashIndex.__eq__` (fh memo). Duck-typing removal. Fixes for: polars X/y containment, `update` with < 3 rows, the polars y name, Panel residuals, `predict_var` column names and `cov`, `BaseProbaForecaster` `_yvec`/`_check_X`, pd3 caller mutation | C1 1.05–1.34x e2e (g3); polars `convert_to` ~10x e2e (x9 verifier); fh memo 1.05–1.53x on top of trusted (g3 C3, best case U=1). Correctness: 7+ g5/g4 stock bugs removed | low. The null-guard divergence (x9 verifier 58/75 mismatches without the guard) must be tested |
| **P1 VectorizedDF substrate** | public `register_mtype` + entry point; `TimeAxis`/`Step`; `GroupedArray` + spokes (pandas, polars, pyarrow); `np_series` / `np_grouped` / `np_grouped_hier` / `pa_*` registered; schema iteration becomes positional; `FittedGrid` behind `forecasters_` | time ≈1.0–1.14x (g3 C4 marginal); peak RSS 1.4–2.2x lower (g3: 5000x100 Poly 129/161 → 50/54 MB); polars ingress 2.84 s → 11 ms at 10k (final_bench T3) | medium. pandas-version fragility of trusted time code (g3 found 2 pd3 bugs). Step is exact only for some aliases (a3 verifier: SME, tz W, BMS/BME fail) → unsupported steps fall back to the stock path |
| **P2 BaseForecaster boundary + T1** | public methods refactored into validate-once + shared core; `_fit/_predict/_update_trusted`; top-level Panel X/y validation; inner selection rule; `update_predict` / `evaluate` on GroupedArray; proba concat from long parameter tables; vectorized `predict_proba` folded into `_vectorize` | g1: fit+predict 1.15–1.88x, workload 1.07–1.72x. Cumulative P0–P2: Naive 1.59–1.91x, Poly 2.44–3.31x (g3 C4), for all 159 eligible classes. Distribution concat 826 → 3.2 ms at 1k (g4) | medium. Drift between public and trusted code (g1 caveat 5) is avoided only by the shared-core refactor (LOC unmeasured). Opt-out must be by class tag or config, not instance `set_tags` (g1) |
| **P2b trusted nested transformers** | the same trusted entry for `BaseTransformer` (Lag, Detrender, Deseasonalizer inside forecasters); mtype-aware proba mixin | *expected*: Naive predict and interval are capped today by nested Lag (63% of C4 time, g3) and mixin `convert_to`/`infer_freq` (g1 §3.3). Upper bound ≈ x8's nested share of 4.8 / 11.6 ms per instance, i.e. ≤1.7x on Naive (unmeasured) | medium; the same invariants as g1 for transformers |
| **P3 T3 ports + native proba** | Naive, PolynomialTrend, then ~7 closed-form candidates (x7); `proba_inner_format="long"`; `InstanceView` for `forecasters_` | 498–2,735x on pandas input and 498–1,486x on pyarrow at 1000x100 (g2); 105–4,560x vs stock with P0 fixes (g3 C5); interval kernel ~1100x (g4); conversions ~800 → 2 (g2) | medium. API parity for `forecasters_`/`get_fitted_params` (§3); quirk replication (g2 §7) |
| **P4 T2 ports by gain** | `_BaseWindowForecaster` + reducers, statsforecast adapter, other V1/V2 classes where profiling shows framework share > 50% | reducer 6–9x vs stock (g2); statsmodels family ≈ T1 only (g1 ets ≈1.0) | low per port; the total program is long (r1: 19/210 in 3.5 y) |
| **P5 lazy + distributed** | `lookback_window` tag census; `Selection` pushdown in lazy spokes; dask partition-and-delegate; T3p passthrough for darts/neuralforecast | selective reads: lazy-then-adapter 15 ms vs 5.15 s eager sktime (332x, final_bench T6); pushdown itself only 1.9x time / 2.8x RSS over eager select-before-convert (x9 verifier). Full-panel workloads: none (g3: lazy ±15% of eager; C5 lazy 1.1–2.6x slower). dask 1.2–1.5x (r4 verifier) | medium. Row-order nondeterminism of lazy backends (x1); op coverage per backend |
| **P6 contract flips (N+2)** | `forecasters_` → `FittedGrid`; native proba for non-pandas input; `inner_dispatch="auto"` default | no speed impact; it removes compat views | user breakage on `forecasters_` access patterns (G16 census unmeasured) |

---

## 6. Evidence

### For
1. The tag system is enough as the opt-in mechanism. g2 moved three forecasters to a non-pandas inner type by changing
   only `y_inner_mtype`, with 315/315 + 233/233 equality on pd2 and pd3, and touched no BaseForecaster code beyond the
   registry and 2 dispatch points (g2 §1.1).
2. The lever that reaches every class is container-independent and needs no estimator change: T1 = g1 (159/160
   eligible, 7710/7710 tests, 1.15–1.88x), and stacked with C1/C3 it reaches 1.59–3.31x (g3).
3. The large wins are per-estimator by nature. Batched kernels give 10²–10³x (g2, g3 C5, x7), and only about 9
   forecasters are closed-form (x7). An opt-in per estimator is the natural granularity.
4. numpy beats dataframe engines inside kernels: segmented numpy beats polars expressions (r3 verifier), and pandas
   groupby is ~10x faster than narwhals-on-pandas (x5 verifier). So the inner tier should be numpy rather than a
   dataframe API.
5. Prior art agrees on the shape: Nixtla converts once to a GroupedArray and keeps models numpy-only (r3); sklearn
   keeps a narrow boundary and opts compute in per estimator (r1).
6. Proba fits the design: a long schema + `ProbaSpec` round-trips exactly, and native kernels are ~1100x (g4).

### Against (strongest first)
1. **Opt-in migration is slow and long-tailed.** sklearn Array API reached 19/210 estimators in about 3.5 years
   (r1). My census: porting the top 10 implementations covers only 28% of vectorized classes, and 50 implementations
   cover 63%. Most of the population will stay at T1 for years, so headline T3 numbers apply to a handful of classes.
2. **Ports are rewrites, not diffs, and must replicate quirks.** g2: Naive is 116 new LOC; equality needed sklearn
   `cond=tol` truncation, the drift-variance slope and the frozen window. Exact least squares would differ from stock
   by up to 30%. Every port needs the equality harness, and keeping old and new implementations side by side is a
   maintenance hazard.
3. **T2/T3 gain only for cheap estimators.** Reducer 6–9x (g2); ETS/statsmodels ≈1.0x (g1); after C3 the time is
   sklearn or statsmodels inside the estimator (g3: 63%). A tier does not help expensive per-series models.
4. **Public-API breakage for T3.** There is no per-instance `forecasters_` grid, `get_fitted_params` has a different
   shape, and the pandas time-level freq was lost in the g2 egress (g2 §7, g3 C5). `InstanceView` is designed but
   unmeasured.
5. **Polars users see much less unless P0/P1 land first.** g2 on polars was 34–105x vs 498–2,735x on pandas, because
   99% of the time is stock polars checks.
6. **Trusted time code is fragile across pandas versions.** g3 found 2 pd3 bugs (Day vs 24h, datetime unit); the x6
   hub dropped freq and broke pandas `update_predict` (g5 H1); Step is not exact for every alias (a3 verifier).
7. **Composite transitivity is easy to get wrong.** Under x4, composites silently stayed on pandas (g5 inv 8), and
   a2's verdicts are not transitive. Derived tags must be tested per composite.
8. **The registry today is private-cache surgery.** A later `cache_clear` silently unregisters the mtype (g2). P1 must
   ship a real registry before any third party can opt in.
9. **Tiers do not express laziness.** Pushdown needs a boundary-level `lookback_window`, not an inner mtype. Lazy pays
   nothing on full-panel workloads (g3) and only 1.9x over eager select (x9 verifier).
10. **Several tiers multiply the test matrix** (tiers × backends × methods). The cell count was not estimated (G10).

---

## 7. Open experiments (needed to firm up the numbers)

1. `design_tiered_t2_window`: a T2 port of `_BaseWindowForecaster` + `make_reduction` (recursive and direct) on
   `SeriesArray`, timed against T1 (g1 path) in one interleaved process. This replaces the cross-report "4–5x over T1"
   estimate.
2. **Trusted BaseTransformer (P2b)**: g1-style entries for Lag, Detrender and Deseasonalizer. Measure the Naive
   predict and interval ratio and the TTF workload. This tests the ≤1.7x expectation.
3. **statsforecast batched T3**: route `_GeneralisedStatsForecastAdapter` to StatsForecast's own grouped path. Measure
   on 1k–10k series against stock and T1, with conversion charged.
4. **InstanceView fidelity**: build `forecasters_` for T3 from parameter arrays. Run the a1 grid tests and the
   metrics probe, and time `get_fitted_params(per_instance=True)` at 10k.
5. **Ragged ends**: the fh memo with U≈n unique cutoffs (g3 caveat; a3 P3 measured 1.6–3.8 ms per unique cutoff) under
   T1 and T3.
6. **M4 adversarial suite** against `GroupedArray` ingress and each ported T3 forecaster, plus the cost of the
   sortedness and NaN-key guards (G13).
7. **Composite tag derivation census**: for the 61 composites, compute the derived tier and run g5 cells for TTF,
   GridCV and Reconciler with a T3 child.
8. **`lookback_window` census + pushdown** (G9/G14): the share of classes with a finite lookback, and the memray peak
   for fit on a 2M-row parquet.
9. **pd3 timing of g2 and g3** (only equality ran under pd3), and a fresh-process memray peak for the T3 e2e path.
10. **Test-matrix cost** (G10): count the cells tiers × backends × methods, and the CI minutes for check_estimator
    with the T1 path on.

---

## 8. Measurement performed for this report

```
cd /home/user/sktime
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
  experiments/container_independence/scripts/design_tiered_census.py \
  experiments/container_independence/results/design_tiered_census.json
```

Output (static, 2.8 s):

```
n_forecasters 160   n_vectorized 124   n_panel_native 36   (pd-multiindex anywhere in y_inner_mtype)
n_distinct_impl 120   n_distinct_impl_vectorized 96
coverage_curve_vectorized [(1, 8, 0.065), (3, 17, 0.137), (5, 24, 0.194), (10, 35, 0.282),
                           (20, 48, 0.387), (30, 58, 0.468), (50, 78, 0.629), (96, 124, 1.0)]
top impl pairs: _GeneralisedStatsForecastAdapter 8, BaseDeepNetworkPyTorch 5, _StatsModelsAdapter 4,
                _DelegatedForecaster 4, _ArpsDcaBase 3, _DartsRegressionAdapter 3, _PmdArimaAdapter 2, _TbatsAdapter 2
shared bases in MRO (vectorized): _StatsModelsAdapter 12, BaseDeepNetworkPyTorch 9, _BaseWindowForecaster 9,
                _DelegatedForecaster 9, _GeneralisedStatsForecastAdapter 9, _Reducer 8, BaseGridSearch 4, ...
n_pred_int 84   n_composite_flag (MRO rule) 27
```

How these counts relate to earlier ones:
- 124 vectorized classes matches g1's census (124).
- 120 distinct implementations matches a2.
- a2 counts 29 panel-native classes with its own rule; mine (36) includes the 8 "all mtypes" passthrough
  composites, minus overlap.
- The composite count (27) is lower than a2's 61, because I flag only the MRO meta bases, not estimator-valued
  parameters.
