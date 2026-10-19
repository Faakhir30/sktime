# design_minimal — pandas inside, a lean framework, native containers at the edges

Architect angle: **MINIMAL-CHANGE BASELINE.** pandas stays the single inner compute container for all 160
forecasters. Speed comes from removing framework overhead: validate once, a trusted per-slice core, known-mtype
conversion, an fh/cutoff memo, and a trusted nested-transformer entry. Other containers are converted once at
the boundary, and the result goes back in the input's own container. This document also states what is **not**
worth building.

Every number is cited as `report-key: claim` or comes from the two scripts I ran for this report (§9). Anything
else is marked *unmeasured/expected*. Nothing under `sktime/` was modified.

---

## 0. Thesis (≤120 words)

The study measured where the time goes, and it is not the container:
- A container-only change is worth ≤1.05x (x8).
- A hub changes time by 0.93–1.14x (g3 C4).
- Engines beat a pandas groupby by only 1.07–1.33x once conversion is charged (x7 verifier).

The levers that do pay work with *any* container and keep pandas inside:
- the trusted path, the fh memo and known-mtype conversion: 1.55–3.03x (g3 C1–C3);
- a trusted nested-transformer entry: a further 1.32–1.54x (§9, new);
- panel-native **pandas** kernels: 382–1895x vs the loop (x7 verifier).

So the plan is: keep pandas, make the framework lean, convert once, and return the user's container. Two
things are added only at the edges: a selection-aware lazy boundary, and a long proba format for non-pandas
users. A hub, narwhals, adapters inside the loop and Arrow-backed internals are not built. Each of those
decisions comes with the condition that would reverse it.

---

## 1. What the evidence forces on a minimal design

| evidence | consequence for this design |
|---|---|
| x8 / wrap_critic §0: the vectorized cost is 8–14 ms per instance in every container; a container-only change is ≤1.05x | The inner container does not need to change to get speed. |
| g1: trusted per-slice path, fit 1.2–5.5x, update 1.3–3.2x, fit+predict 1.15–1.88x; 7710/7710 tests pass; 159/160 classes eligible | The core of the design. It is container-agnostic: the slices are pandas either way. |
| g3: C1 1.05–1.34x → C2 1.33–2.19x → C3 1.55–3.03x cumulative; **C4 (hub) 0.93–1.14x in time** | Ship C1–C3. Do not ship a hub for speed. C4's only value is memory (1.4–2.2x lower RSS), which is discussed in §7. |
| g3 §3.4 + g1 §3.3: after C3, nested `Lag` is 63% of Naive time; inner `convert_to`/`infer_freq` dominate the proba mixin | The trusted path must also cover **nested transformers and the proba mixin**. Measured in §9: +1.32–1.54x. |
| x7 verifier: pandas groupby is 382–1895x faster than the sktime loop at 10k; polars/duckdb add only 1.07–1.33x with conversion charged; numpy layout 7–11x over pandas groupby with pandas in and out | The 10²–10³x lever does **not** need a new container. A kernel can take `pd-multiindex` (which already exists) and use numpy internally. |
| x8 verifier: polars over a lean pandas groupby is 2.1–3.0x on native-polars data | This is the one real cost of keeping pandas: polars users of kernel forecasters lose up to 3x. That is accepted, see §7. |
| x9 + verifier: Arrow-backed pandas is zero-copy only for single-chunk data, 1.7x less memory than numpy on parquet; 64/71 forecasters pass; Arrow survives in 25% of outputs | Arrow is an ingress memory option at most. It is not an inner representation. |
| **§9 (new):** Arrow-valued hybrid MI end to end is **1.07x (Naive) and 1.19x (Poly) slower** than numpy pandas; Arrow survives only in Naive's output | This confirms that Arrow-backed internals are not worth doing for speed. |
| final_bench T6 / x9 verifier: selective lazy read 875x, 25x memory; lazy+convert-slice-only (B5b) 190–330x; real pushdown over select-before-convert is only 1.9x | Laziness pays only through **selection before conversion**. A selection-aware boundary captures most of it. Lazy mtypes inside BaseForecaster do not add anything. |
| g3 §4.4: laziness gives nothing for full-panel fit+predict; every level collects | No lazy Series/Panel mtype in the inner path. |
| g5: x4 *lost* checks (disjoint X, y name) and x6 *broke* pandas `update_predict` (freq); stock has 10 boundary rules | Keeping pandas inside keeps every behaviour that stock gets for free. The pandas conformance cells stay 22/22 by construction. |
| g4: stock proba is dominated by per-slice re-entry (interval 18.1 s vs var 3.1 s at 1k); final concat 205 ms vs `_get_columns` 7.9 s at 10k; `BaseProbaForecaster` vectorization is broken | Proba needs the trusted path, plus one output builder that runs once per panel. The MultiIndex layout stays for pandas. |
| a1: 229 touchpoints; 78 are time/freq/cutoff; only 42 are container ops; per-slice re-entry is 52–84% of fit | The time and validation code is what gets reworked. The container ops do not move. |

---

## 2. Architecture

### 2.1 Components (new or changed; everything else unchanged)

```
user y/X (pandas | polars eager | polars lazy | pyarrow | duckdb | dask)
   │
   ▼  Boundary (sktime/datatypes/_boundary.py, NEW, ~300 LOC expected)
   │    spoke = backend_of(obj)            # explicit registry, no hasattr duck typing (g5 rules 4/5)
   │    meta  = spoke.check(obj, scitype)  # vectorized checks (x9 mono fix with null guard)
   │    sel   = Selection.from_estimator(est, fh)   # instances, columns, lookback (lazy inputs only)
   │    pdf   = spoke.to_pandas(obj, scitype, select=sel)   # ONE conversion, known from_type
   ▼
pandas pd.Series / pd-multiindex / pd_multiindex_hier  (the only inner container)
   │
   ▼  BaseForecaster public method = validate-once + _core (sktime/forecasting/base/_base.py, refactor)
   │    if panel-native inner (pd-multiindex tag): _fit/_predict on the whole frame
   │    else: VectorizedDF (stateless, pandas) → per slice _fit_trusted / _predict_trusted / _update_trusted
   │          with SliceCtx (per-call constants), fh memo per unique cutoff,
   │          nested transformers via trusted entry (TrustedCtx contextvar)
   ▼
pandas output (point: pd frame; proba: built once per panel by ProbaSpec)
   │
   ▼  spoke.from_pandas(out, store) / spoke.proba_out(long, spec, fmt)  → user's container
```

### 2.2 Protocols (signatures)

```python
# sktime/datatypes/_boundary.py  (NEW)
class Spoke(Protocol):
    name: str                                   # "pandas", "polars", "polars_lazy", "pyarrow", "duckdb", "dask"
    def owns(self, obj) -> bool: ...
    def check(self, obj, scitype: str) -> "Meta": ...          # vectorized; null-guarded (x9 verifier)
    def to_pandas(self, obj, scitype: str, *, select: "Selection | None" = None
                  ) -> tuple[pd.DataFrame | pd.Series, str, "Store"]: ...  # returns (frame, mtype, store)
    def from_pandas(self, obj, mtype: str, store: "Store"): ...            # point outputs
    def proba_out(self, long: pd.DataFrame, spec: "ProbaSpec", fmt: str, store: "Store"): ...

def backend_of(obj) -> Spoke: ...               # registry lookup; replaces hasattr(obj, "index"/"X")
def register_spoke(spoke: Spoke) -> None: ...   # public; no lru_cache surgery (g2 §1.1 failure)

@dataclass(frozen=True)
class Selection:                                 # only applied when spoke is lazy (polars_lazy/duckdb/dask)
    instances: Sequence | None = None
    columns: Sequence[str] | None = None
    lookback: int | None = None                  # rows per instance, from estimator tag (see §2.5)

@dataclass(frozen=True)
class Store:                                     # what egress needs to restore the user's container
    spoke: str; key_names: tuple; time_name: str | None; col_names: tuple; native_schema: Any
```

```python
# sktime/forecasting/base/_base.py  (refactor of the existing public methods; g1 caveat 5: one shared core)
class BaseForecaster:
    def fit(self, y, X=None, fh=None):
        b = self._ingress(y, X)                       # Boundary + checks, ONCE, on the whole panel
        return self._fit_core(b, fh)                  # public and trusted paths share this core

    # trusted per-slice entry, used only by _vectorize; never public
    def _fit_trusted(self, y_inner, X_inner, fh, ctx: "SliceCtx") -> Self: ...
    def _update_trusted(self, y_inner, X_inner, update_params: bool, ctx: "SliceCtx") -> Self: ...
    def _predict_trusted(self, method: str, X_inner, fh, ctx: "SliceCtx", **kw): ...

@dataclass(frozen=True)
class SliceCtx:                                   # computed once per public call (g1 §1)
    y_conv: Callable; X_conv: Callable | None; back_conv: Callable
    y_meta_template: dict; tags: Mapping[str, Any]
    fh_cache: "FhAbsCache"                        # to_absolute once per unique cutoff (g3 C3)
    insample_ok: bool
```

```python
# sktime/transformations/base/_base.py  (small addition)
class BaseTransformer:
    def _fit_transform_trusted(self, X_inner, y_inner=None):
        """Sets the state that fit sets (_X if remember_data, _is_fitted) and calls _fit and _transform.
        Only used when TrustedCtx is active and X is already of X_inner_mtype; otherwise fit_transform."""

# sktime/utils/_trusted.py  (NEW, ~40 LOC)
TRUSTED: ContextVar[bool]      # set by _vectorize's trusted loop; read by BaseTransformer.fit_transform
                               # → nested transformers inside a forecaster's _predict skip check/convert
```

```python
# sktime/forecasting/base/_fh.py  (bug fix = feature)
class _HashIndex:  __eq__/__hash__ on (values bytes, dtype, freqstr)  # lru_cache hits (critic bug #3)

# sktime/forecasting/base/_proba_out.py  (NEW; g4 ProbaSpec, promoted)
@dataclass(frozen=True)
class ProbaSpec: kind: Literal["interval","quantiles","var"]; vars: tuple; levels: tuple
def assemble_long(parts: list[np.ndarray], keys, times, spec) -> pd.DataFrame   # once per panel
def long_to_multiindex(long: pd.DataFrame, spec) -> pd.DataFrame   # pandas users: today's layout
```

### 2.3 Where each a1 pandas-touchpoint category goes (229 touchpoints)

| a1 category (count) | in this design |
|---|---|
| container ops: group/iterate (6), slice/select (12), concat/reconstruct (9) | **Stay pandas.** VectorizedDF stays as is (stateless `(frame, schema)`). The only change is that conversion uses a known `from_type` and is not re-inferred (g3 C1b). |
| pivot/reshape `forecasters_` (15) | Stays pandas. EstimatorGrid is ≲1% of fit (a1 §6), so it is not worth an API change. |
| MultiIndex levels/names (23) | Stays pandas. The spokes restore names at egress. The g5 rule 2 tests compare labels as well as values. |
| time index/freq/cutoff (78) | Stays pandas, but runs **once per panel** instead of once per slice: an fh memo per unique cutoff, the outer cutoff as a max over codes (a1 P10), the slice freq trusted from the level. pandas 3 fragility is handled by the CI matrix (g3 found 2 bugs). |
| metadata columns/dtypes/nans (13) | Checked once at the top level. Per-slice `has_nans` uses numpy (g1). |
| output construction (14) | pandas. Proba is built once per panel through `ProbaSpec` (g4: 14–85 ms vs 205 ms concat + 7.9 s `_get_columns` at 10k). |
| type dispatch / duck typing (34) | **Replaced** by `backend_of`. This fixes the disjoint-X silent pass, the column-named-"X" bug and the `get_cutoff` `VectorizedDF.X` duck typing (a1.10, g5). |
| mtype pinning (25) | Stays. pandas mtypes remain the inner targets. |
| public-API value types (44 = C1–C4) | See §3. Kept, apart from proba for non-pandas input. |

### 2.4 Data flow by input container

| input | ingress | inner | egress | evidence |
|---|---|---|---|---|
| pandas | none (identity spoke) | pandas | none | g1/g3: pandas cells stay 22/22 (g5 stock) |
| polars eager | vectorized check + adapter **without** the per-series monotonic loop → pd-multiindex (numpy-backed) | pandas | `from_pandas` → polars, including proba long frame | T3: adapter with no check is 1.4–1.7x pandas prepare, vs 7–36x today; x9: check 25–28x faster |
| polars lazy | apply `Selection` (instances, columns, `tail(lookback).over(keys)`) in the plan, then `collect()` → polars spoke | pandas | **eager polars** (documented: `predict` on a LazyFrame returns a DataFrame) | T6 B5b: 15–27 ms / +44–50 MB vs 5.15 s / +786 MB; g3: lazy without selection is ±15% of eager |
| duckdb / ibis | Selection pushed into SQL, then `.arrow()` → pandas | pandas | pyarrow Table (C1 default, x1/r2) or `to_duckdb` if requested | T6 B3: 14 ms / +11 MB for the selective read |
| pyarrow | `Table.to_pandas()` (numpy) | pandas | pyarrow Table | — |
| dask | Selection → `compute()` → pandas (sktime rejects dask today, x8) | pandas | pandas, or `dd.from_pandas` if config `output:dask=True` | x6: dask ingress 4–7x pandas prepare; distributed fitting is **not** offered (see §7) |

Scale-out for dask users goes through the **existing** `backend:parallel` config (loky/dask/ray). This works
with the trusted path (g1: loky verified). Partitioning the data stays the user's job.

### 2.5 Two additions at the edges (the only new semantics)

1. **`capability:lookback` tag (int | None).** It states how many trailing rows per instance `fit` reads.
   Examples: Naive last = 1, seasonal naive = sp, window forecasters = `window_length`, None = all. The
   Boundary uses it only for lazy spokes, and only when `update` is not expected (`remember_data=False`).
   Otherwise it selects instances and columns only. The design rests on T6, where the selective lazy read
   costs 1/190–1/875 of eager. *What share of forecasters can declare a finite lookback is unmeasured*
   (wrap_critic G9 census open).
2. **`output:proba_format` config (`"auto"|"multiindex"|"long"`).** `"auto"` gives MultiIndex for pandas
   input and long for any other backend, in that backend (g4 recommendation).

---

## 3. Public API: the four a1 contracts

| contract | decision | deprecation |
|---|---|---|
| **C1 `forecasters_`** is a `pd.DataFrame` | **Unchanged.** A container-free grid saves ≲1% of fit (a1 §6) and breaks 4/4 hierarchical metric configs, so it fails cost/benefit. Panel-native kernel forecasters (§4) have no `forecasters_` and say so in their docstring, as the 29 pd-multiindex-native classes already do. | none |
| **C2 `cutoff` / `fh`** are pandas `Index` + `ForecastingHorizon` | **Unchanged.** These are O(1) metadata (g5 rule 1 accepts pandas here). Fixed: cutoff `freq` is always present (g5 rule 10), and `get_cutoff` is backend-aware through `backend_of`, not duck typing. | none |
| **C3 proba columns** (3-/2-level MultiIndex) | Unchanged for pandas input, permanently. Non-pandas input: long schema `(keys…, time, var, coverage/alpha, bound, value)` in the input backend, via `output:proba_format`. Today polars users get pandas MultiIndex (a1.4). | N: `"auto"` still returns pandas MultiIndex for non-pandas input, with a `FutureWarning`. N+2: `"auto"` → long. `"multiindex"` keeps the old behaviour forever. |
| **C4 `get_fitted_params` keys** `forecasters.loc[...]` | **Unchanged.** One touchpoint; renaming breaks users' code that parses the keys, for no speed gain. | none |
| return type | `predict*`, `update_predict`, `predict_residuals` and `evaluate(return_data=True)` return the input container (g5 rule 1; fixes `evaluate`'s pandas `y_pred` for polars input, g5). lazy → eager of the same engine. | Behaviour fix, noted in the changelog. |
| `predict_proba` | Stays a skpro distribution. `_concat_distributions_hierarchical` is replaced by one distribution built from the long parameter table (g4: 3.2 ms vs 826 ms at 1k, identical). | none |
| opt-out | class tag `capability:trusted_vectorize=False`; config `trusted_vectorize=False` (g1: instance `set_tags` does not work, because `reset()` wipes it) | — |

For pandas users the visible API change is zero. For polars, pyarrow and duckdb users, the changes are: proba
output in their own container (with a deprecation window), and `evaluate`, `update_predict` and
`predict_residuals` working and returning their container.

---

## 4. Estimator migration (160 classes; a2 buckets)

| group | count | what changes | who changes it |
|---|---|---|---|
| per-instance pandas-inner, vectorized | 123/124 eligible (g1 census) | **Nothing.** They get the trusted path automatically. | framework |
| overrides a guarded public method | 1 (ChronosForecaster) | falls back to stock automatically (g1 GUARDED) | optional: move the override to `_predict` |
| pd-multiindex-native (bypass VectorizedDF) | 29 (a2/x4 verifier; reducers, pipelines, NeuralForecast, PytorchForecasting, MOIRAI, LagLlama) | Nothing. They gain from ingress (one conversion with known mtype) and from the nested-transformer entry. | framework |
| forecasters with nested sktime transformers (Naive's `Lag`, TTF's Detrender/Deseasonalizer, reducers' `Lag`) | not counted (*census open*) | Nothing: the `TRUSTED` contextvar makes `BaseTransformer.fit_transform` use `_fit_transform_trusted` when the input already has the inner type. §9: +1.32–1.54x fit+predict on Naive. | framework |
| V1 boundary-only (63) / V2 glue (56) | 119 | Nothing required. Fixing `_sktime.py:151` (last-window `.loc`) is a pure bug/speed fix and optional. | — |
| V3 closed-form, "panel-kernel candidates" (Naive variants, PolynomialTrend, mean/drift/seasonal, Croston-like; about 6–9 per x7 verifier) | ~6–9 | **Opt-in**: `y_inner_mtype=["pd-multiindex","pd_multiindex_hier","pd.Series"]` plus a groupby/numpy kernel on `df.index.codes` + `df.to_numpy()`. This is the existing tag mechanism, with no new mtype or registry. Each kernel has a guard (sorted, unique, regular freq), and falls back to the per-instance path if the guard fails (x7 verifier: unguarded kernels are silently wrong on shuffled input). | maintainers, one at a time, behind a parity test vs the per-instance path |
| V4 third-party (28) | 28 | Nothing. darts/neuralforecast take pandas, which they already get. Native polars passthrough for the 8 polars-capable classes is *not* offered (G6 unmeasured) | — |
| proba-heavy defaults (`_predict_interval` from `_predict_var` in the mixin) | all with `capability:pred_int` | Nothing: the mixin's `convert_to`/`infer_freq` calls use known types inside `TRUSTED` (*unmeasured/expected*; g1 §3.3 names it as the residual) | framework |

**Total estimator-side work required: 0 classes.** The optional work is ~6–9 kernel ports, which pays
10²–10³x. All of it is pandas plus numpy inside the kernel.

---

## 5. Roadmap (VectorizedDF → BaseForecaster boundary → estimators)

| phase | scope | expected impact (cited) | risk |
|---|---|---|---|
| **P0: decision-independent fixes** (1 release) | wrap_critic §5 list. Vectorized polars monotonic check with a null guard; known `from_type` instead of re-inference; `_HashIndex.__eq__`; `backend_of` replacing `hasattr(…,"X"/"index")`; polars X/y containment through a real join; `pd.infer_freq` guard for <3 rows in polars update; Panel `predict_residuals` fh from the time level; pd3 `_enforce_index_freq` must not mutate the caller's data; one FutureWarning per fit, not per clone; fix `BaseProbaForecaster._yvec` | **1.05–1.34x e2e** (g3 C1); polars `convert_to` ~10x faster e2e (x9 verifier), check 25–28x (x9); polars prepare 2.62 s → 0.019 s with known mtype at 10k (final_bench T3); fixes 4 g5 stock bugs and 3 a1 bugs | Low. The vectorized check is not equivalent with nulls (x9 verifier: 58/75), so the null guard must fall back to stock. |
| **P1: VectorizedDF stays pandas; conversion once** | `prepare_VectorizedDF` takes `from_type`; the outer cutoff is a max over level codes (a1 P10); slice freq is trusted from the level. **No hub, no adapter dispatch in the loop.** | included in C1/C3. Iteration is 1–2% of fit+predict (x8, x3), and slicing once vs twice makes no difference (g1 ablation 1.18 vs 1.19) | Very low |
| **P2: BaseForecaster boundary (the main phase)** | public = validate-once + shared `_core`; `_fit/_update/_predict_trusted` with `SliceCtx`; fh memo per unique cutoff; `TRUSTED` contextvar plus `BaseTransformer._fit_transform_trusted`; known-type helpers in the proba mixin; `ProbaSpec` output builder | g1: fit 1.2–5.5x, update 1.3–3.2x, workload 1.07–1.72x. g3: **C3 cumulative 1.55–3.03x** fit+predict (Naive 1.55–2.02, Poly 2.23–3.03). **§9: + nested entry 1.32–1.54x marginal → Naive 2.16–2.48x cumulative** (300–1000 series, pd and polars, outputs identical). Proba output: 7.9 s `_get_columns` + 205 ms concat → 14–85 ms at 10k (g4). | Medium. Drift between the public and trusted paths (g1 caveat 5), so the code must be shared, not copied. The fh memo gains shrink with ragged cutoffs (U→n: 1.6–3.8 ms per unique cutoff, a3 via g3). The 7710-test gate (g1) must run in CI under both modes. pandas-3 internals (g3 found 2 bugs). |
| **P3: edges** | spoke registry; return the input container for all frame outputs; `output:proba_format` (FutureWarning in N, default flip in N+2); `capability:lookback` + Selection for lazy spokes; accept dask through compute | Selective lazy reads 190–330x faster, 16–18x less RSS (T6 B5b vs B1). Non-selective lazy ≈ eager ±15% (g3). Polars proba egress: long → MI 14–85 ms at 10k (g4). Polars users get consistent types (a1.4: 5 of 6 outputs are pandas today). | Low–medium. The share of forecasters that can declare a lookback is unknown. `update` after a lookback-trimmed fit needs `remember_data` semantics (a lookback fit must not remember the full history). |
| **P4: opt-in pandas panel kernels** | ~6–9 closed-form forecasters get `pd-multiindex` inner + guarded groupby/numpy kernels + native proba (long) | 382–1895x vs the loop at 10k for pandas groupby (x7 verifier); numpy-on-codes kernels another 7–11x (x7 verifier); polars engine would add only 2.1–3.0x on native-polars data (x8 verifier) and 1.07–1.33x with conversion charged (x7 verifier) | Medium. Guards (sortedness, regular freq) cost unmeasured (wrap_critic G13). Kernels lose `forecasters_`, as the existing 29 already do. |
| **N+2: contract flip** | `output:proba_format="auto"` → long for non-pandas | — | Users who parse MI columns on polars output (*few, unmeasured*) |

Order rationale: P0+P2 deliver ~all of the e2e gain that is reachable without rewriting estimators, for
every container, with zero API change. P3 is the only place where non-pandas containers add real value
(selective laziness and type consistency). P4 is where the 10²–10³x lives, and it needs no new container.

---

## 6. What is NOT worth doing (and the evidence that would reverse each verdict)

| not doing | why (evidence) | reverse if… |
|---|---|---|
| **Narwhals VectorizedDF or narwhals inside forecasters** | 1.7–2.5x slower on pandas with a fast path, 3.8–5.7x generic (final_bench T3b); plain pandas groupby ~10x faster than narwhals-on-pandas (x5 verifier); the answers to C1/C2 depend on the narwhals namespace and on undocumented index leakage (wrap_critic K6/K7) | narwhals stable gains an index contract, *and* a ported forecaster shows parity on pandas |
| **Backend adapter dispatching inside the loop (x4)** | e2e 0.96–1.17x vs today (x4 verifier); polars→pandas per slice costs 2.9–5.3x (T1 S5p); g5: polars coverage strict 5/24 vs stock 9/24, lost the disjoint-X check and the y name | a material share of leaf forecasters computes natively in polars (today: 0, a2.6) |
| **Hub / grouped array as the canonical inner representation** | g3 C4 0.93–1.14x time; x6 hub had unfixed correctness issues (NaN keys, mixed freq, multivariate col 0; x6 verifier); its round trip broke pandas `update_predict` (g5) | memory is the binding constraint (see §7, the strongest counter-argument) |
| **New numpy inner mtype family + public mtype registry (g2)** | equal results, but the 10²–10³x comes from batching, which pd-multiindex kernels also give (x7 verifier); polars g2 is held back by stock checks (g2: 99% of g2 time); g2 needs `get_cutoff`/`update_data` dispatch and lru_cache surgery | pandas-groupby kernels prove too slow (x7 verifier numpy 7–11x faster); then add `np_grouped` *for kernels only* |
| **Arrow-backed pandas as the inner representation** | §9: e2e 1.07–1.19x **slower**; Arrow survives in 25% of outputs (x9 verifier); 7/71 forecasters fail; zero-copy only for single-chunk data | pandas makes Arrow the float default *and* numpy-free paths exist in the forecasters |
| **Lazy mtypes through BaseForecaster** | g3: no gain without selection (every level collects); x4: per-instance laziness 3–77x slower; x1/r2: per-group lazy iteration 170–543x slower | lookback cannot be declared for most forecasters and the full-scan memory matters |
| **Container-free `forecasters_` / renamed fitted-param keys** | ≲1% of fit (a1 §6); 4/4 hierarchical metric configs break | — |
| **Fugue/partition-and-delegate for dask** | like-for-like 1.2–1.5x (r4 verifier); break-even unmeasured (G7) | G7 shows a break-even at realistic panel sizes |

---

## 7. Evidence against this design (honest, strongest first)

1. **Polars and lazy users lose engine speed on kernel forecasters.** On native-polars data, a polars kernel is
   2.1–3.0x faster than a lean pandas groupby (x8 verifier), and numpy kernels 7–11x (x7 verifier). Keeping
   pandas means a polars user pays the conversion plus a slower kernel. At 50000x50 this is 24 ms (polars
   lazy) vs 450 ms (pandas groupby) (final_bench T4 x7 rows): **~19x**, *when the data is already polars*.
   For the ~6–9 kernel forecasters this is the largest cost of my angle.
2. **Memory.** The hub (g3 C4) cuts fresh-process peak RSS 1.4–2.2x (5000x100 Poly polars: 121 MB at C3 → 54 MB
   at C4). Polars ingress through the pandas MI costs +451 MB vs +30 MB via a hub at 50000x100 (final_bench
   T5b). My design keeps the C3 memory profile. For out-of-core or 50k+ series panels this matters more than
   1.1x of time.
3. **Conversion is still paid for every non-pandas call.** Even with the check fixed, polars→pandas costs
   1.4–1.7x of pandas prepare (T3, adapter without check), and egress back costs 1.8–4.2x of pandas
   reconstruct today (T2). These costs are small vs per-instance fitting (≤2% at 11 ms/instance, x8), but
   they are the whole cost for kernel forecasters (g3 C5: polars check ≈45% of a 16 ms run).
4. **pandas-version fragility moves into the trusted core.** g3 found 2 pandas-3 bugs (Day vs 24h, datetime
   unit) in trusted time code; K11 shows pd3 caller mutation. A trusted path that skips validation can turn
   pandas changes into silent errors, where today they raise.
5. **Predict-side gains stay modest.** g1: predict 0.9–1.45x and predict_interval ≈1.0x. §9 lifts Naive
   predict to 1.60–1.93x cumulative, but only for nested `Lag`. Statsmodels-bound forecasters (ETS 165–300
   ms per series, g1) gain ≈1.0x under any framework change.
6. **The "no leaf computes outside pandas" fact becomes permanent.** The 8 darts/neuralforecast classes that
   take polars natively (a2 verifier) still get pandas. The ecosystem may move faster than this design
   (sklearn uses narwhals at its boundary, r1).
7. **Scale-out is not addressed.** dask input is only computed to pandas. A user with 10⁶ series on a cluster
   gets nothing beyond `backend:parallel` (G7 unmeasured).
8. **Two code paths per method** (public and trusted) remain a maintenance cost. g1's prototype is ~700 LOC
   that copies stock logic (caveat 5).

---

## 8. Open experiments (to firm up this design)

1. **Generic trusted nested transformer**: run the `TRUSTED` contextvar version on TTF[Detrender/Deseasonalizer,
   Naive], the reducers and STLForecaster, and run the g1 test suite under it (here only `Lag` in Naive was
   measured).
2. **Proba mixin with known types**: time predict_interval/quantiles after replacing the mixin's
   `convert_to`/`infer_freq` with known-type calls (g1 §3.3 residual; expected to be the main lever for
   predict_interval ≈1.0x).
3. **pandas-groupby kernel ports with guards**: port Naive(last/mean/drift, sp) and PolynomialTrend to
   `pd-multiindex` inner with sortedness/regularity guards. Measure the guard cost and the gap to g2 numpy
   ports (decides P4 vs a g2-style numpy mtype).
4. **Lookback census + lazy fit**: count forecasters that can declare a finite `capability:lookback`.
   Time fit on a 4M-row parquet with Selection vs eager (wrap_critic G9 spec).
5. **Ragged cutoffs**: g3 C3 at U = n/10 and U = n (staggered ends), to bound the fh-memo gain on realistic panels.
6. **pandas 3 run of the full minimal stack (C1–C3 + nested)** on EXP_PY3, with equality.
7. **Memory of the minimal stack at 50k series**, fresh process (M1 harness), to quantify counter-argument 2.

---

## 9. Measurements performed for this report

Both scripts ran from WT with `EXP_PY` (pandas 2.3.3, polars 2.0.0), interleaved in one process, min over
2 reps, on a shared 4-core box. Ratios within ±10% of 1 are noise.

### 9.1 Trusted nested-transformer entry on top of g3 C3 (`scripts/design_minimal_nested.py`)

```
PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
  $EXP_PY -W ignore experiments/container_independence/scripts/design_minimal_nested.py --sizes 300x100,1000x100 --reps 2
# -> results/design_minimal_nested.json, results/design_minimal_nested.log
```

Levels: C0..C3 = `g3_stack.configure(level)` (stock; +mono/known mtype; +g1 trusted; +fh memo/cutoff).
C3+N adds a `TrustedLag` (a subclass of `Lag` swapped in as `sktime.transformations.lag.Lag`). When the input
is a plain pandas Series/DataFrame, its `fit_transform` skips the BaseTransformer check/convert, sets `_X`,
and calls `_fit` + `_transform` directly. Workload: `NaiveForecaster("last")`, `fit(y, fh=[1,2,3])` +
`predict()`. Equality vs C0: `assert_frame_equal(check_exact=True)` (pandas), `DataFrame.equals` + type
(polars), and cutoff + freq. All levels were **equal in 4/4 rows**. Trusted Lag calls: 300 / 1000 per predict,
0 fallbacks.

| size | input | fit+predict C0 s | C3 (×C0) | **C3+N (×C0)** | C3+N marginal over C3 | predict-only C3 (×C0) | predict-only C3+N (×C0) |
|---|---|---|---|---|---|---|---|
| 300x100 | pandas | 3.64 | 1.63 | **2.16** | 1.32 | 1.11 | 1.76 |
| 300x100 | polars | 3.90 | 1.54 | **2.36** | 1.54 | 1.31 | 1.89 |
| 1000x100 | pandas | 12.75 | 1.70 | **2.26** | 1.33 | 1.21 | 1.60 |
| 1000x100 | polars | 13.64 | 1.78 | **2.48** | 1.39 | 1.21 | 1.93 |

The C0→C3 values (1.54–1.78x) reproduce g3's Naive range (1.55–2.02x). The trusted nested entry is the
largest single marginal step for Naive after the outer trusted path. It brings predict, which g1 found
≈1.0–1.15x, to 1.60–1.93x.

Caveats:
- `TrustedLag` covers only `Lag` called with no `y` on non-MultiIndex pandas.
- The transformer instance is discarded after use, so only the output was checked. Lag state was not
  checked against stock.
- n ≤ 1000 and 2 reps.

### 9.2 Arrow-valued pandas end to end (`scripts/design_minimal_arrow.py`)

```
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
  experiments/container_independence/scripts/design_minimal_arrow.py --sizes 300x100 --reps 2
# -> results/design_minimal_arrow.json
```

Input: `make_panel_fast(300,100)` with numpy values vs the same frame `.astype("double[pyarrow]")` (the hybrid
MI: numpy index, Arrow values), through stock sktime.

| estimator | numpy s | Arrow s | Arrow/numpy | values equal | output dtype for Arrow input |
|---|---|---|---|---|---|
| NaiveForecaster(last) | 4.03 | 4.33 | **1.07** (slower) | yes | double[pyarrow] |
| PolynomialTrendForecaster(1) | 2.54 | 3.03 | **1.19** (slower) | yes | float64 (Arrow lost) |

Only one size was run, and ingress conversion cost is not included; x9 already measured it. The result
supports §6: Arrow-backed internals cost time and do not survive the forecasters.
