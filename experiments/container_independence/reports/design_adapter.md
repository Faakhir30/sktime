# design_adapter: a narrow adapter registry at the frame boundary, with iteration dispatched on the inner family

Architect angle: **ADAPTER-FIRST.** This is a narrow `ContainerAdapter` protocol plus a registry, in the style of
sklearn `set_output` and skrub, used by `VectorizedDF` and at the `BaseForecaster` boundary. The pandas
adapter behaves exactly like today's code.

Every number below is either cited as `report-key claim` or comes from the one script I ran
(`scripts/design_adapter_contract_census.py`). Anything not measured is labelled *unmeasured/expected*.
Nothing under `sktime/` was modified.

---

## 0. Thesis (short)

Put **one frame-level adapter per input backend** at the public boundary. It handles ingress,
validation, cutoff, selection/pushdown and egress, all **once per call and never per slice**.
**Per-slice iteration** goes through a small *inner-family* adapter chosen by the forecaster's `y_inner_mtype`
(pandas today; `np_grouped` as an opt-in), not by the input type. Time semantics live in a schema-owned
`TimeAxis`, not in any adapter.

The adapter is plumbing that buys correctness and backend reach. On its own it gives about 1.0x speed
(x8: container-only ≤1.05x; x4 verifier: B2/A0 0.96–1.17x). The phases are therefore ordered so that the
container-independent levers it enables come first: the trusted per-slice path (g1), fh/cutoff once per
panel (g3 C3), and the opt-in panel-native numpy family (g2). Expected result: **1.6–3.3x** end to end for
unchanged forecasters (g3 C4) and **10²–10³x** for ported closed-form ones (g3 C5, g2). pandas input keeps
bit-identical behaviour (x4 78/78 + 28/28 adversarial).

---

## 1. Why adapter-first, and what the evidence forces on it

| Evidence | Consequence for the design |
|---|---|
| x4: pandas adapter bit-identical (78/78 on pd2 and pd3, 28/28 adversarial in x4 verifier), cost 0.89–1.06x | A pandas adapter can wrap today's code with no behaviour change. This is the angle's non-negotiable. |
| x4 §4.2 / x4 verifier §4: dispatching per slice on the *input* type turns 1+1 conversions into N+N (A2/A1 fit 1.07–1.31x, never faster) | Adapter ops are **frame-level only**. Per-slice work is dispatched on the **inner family** (x4's "bridge": A3 = today, B2/B4 = 0 or 1+1 conversions). |
| a1 §3: 14 abstract ops at the boundary, **only 4 are container ops**; 78/229 touchpoints are time/freq/cutoff | Split the protocol: a *frame adapter* (ingress, metadata, set-level ops), an *inner-family iterator* (the 4 container ops), and a separate `TimeAxis` (a3 `Step`). Time semantics must never sit behind per-backend adapters. |
| a1 §5.2, g1, g3: per-slice re-entry is 33–84% of fit. Trusted path: fit+predict 1.15–1.88x (g1), and C2+C3 give a cumulative 1.55–3.03x (g3) | The adapter's real job at the boundary is to make validation **canonical and once**. That is what lets per-slice re-validation be skipped safely (g5 invariant 4). |
| g5: x4 *lost* the Panel disjoint-X check (A1), renamed y→"0" (A2), and its cutoff ignored `reverse_order` (A4) | Validation and cutoff are adapter-level set operations (`time_contains` anti-join, `time_extreme(how)`), not per-slice duck typing. There is a conformance suite per adapter. |
| r1: sklearn `set_output` has 4 methods and reaches 84/84 transformers; skrub's 72+ generic ops still had silent divergence (#2335) | Keep the *required* surface small (8 ops) and push everything else into shared code. Add one conformance suite that every adapter must pass. |
| x4 Z, g3: lazy slices cost 3–77x more than collect-once; laziness pays nothing for per-instance forecasters | A lazy adapter offers `select` (pushdown) and `collect` **once**, at the boundary. It never iterates lazily. |
| x9 / final_bench T6: selective read 875x faster, 25x less memory (lazy scan vs sktime eager); x9 verifier: true pushdown is 1.9x time / 2.8x RSS better than select-before-convert | Pushdown is an adapter op, driven by a forecaster `lookback` tag and by user-requested instances/columns. |
| g2: the registry can only be extended by mutating private `lru_cache` lists; `cache_clear` silently unregisters | Ship a public `register_adapter` / `register_mtype` API (or entry points) **before** any third-party backend. |

---

## 2. Architecture

### 2.1 Components

```
user frame ──► [FrameAdapter for type(user frame)]  (registry, exact-type dispatch)
                 │  validate once • schema • key table • TimeAxis • cutoff • select/pushdown • collect
                 ▼
            FrameSchema + TimeAxis  (container-free metadata; pandas only for 1-elt cutoff / H-elt fh)
                 │
                 ├─ forecaster is panel-native for this scitype (29 pd-multiindex classes, np_grouped ports):
                 │      adapter.to_family(frame, family)  ── one whole-frame conversion ──► _fit/_predict
                 │
                 └─ else vectorize:  family = family_of(y_inner_mtype)  (pandas for 152/160 today)
                        inner = adapter.to_family(frame, family)        (1 conversion, 0 for pandas→pandas)
                        for key, view in FAMILIES[family].iter_slices(inner, schema):  (positional, no re-check)
                            TrustedCore.fit_slice / predict_slice / update_slice (g1)
                        parts → FAMILIES[family].concat(parts, schema) → OutputSpec
                 ▼
            OutputSpec (point / long-proba / grid)  ──► adapter.egress(spec, like=user frame)  (once)
```

Four registries, each small:

1. `FrameAdapter`: one per *user* backend (pandas, polars, polars-lazy, pyarrow, duckdb, dask). Frame-level only.
2. `InnerFamily`: one per *inner* representation forecasters compute on (`pandas`, `np_grouped`, later
   `polars`). Implements the 4 container ops of a1 (P6–P9).
3. `TimeAxis` / `Step`: a single implementation (a3), with `TimeOps` backends only for vectorized arithmetic.
4. `OutputSpec` builders: point predictions, long proba (g4), `EstimatorGrid` (a1).

### 2.2 Protocols (signatures)

```python
# sktime/datatypes/_adapters/_base.py  (new)
@dataclass(frozen=True)
class FrameSchema:                       # container-free; replaces the pandas parts of VectorizedDF's schema
    scitype: Literal["Series", "Panel", "Hierarchical"]
    key_names: tuple[str | None, ...]    # logical names; None = unnamed level (fixes x4-verifier unnamed-level bug)
    time_name: str | None
    value_cols: tuple[Hashable, ...]     # original labels (fixes g5 S7/A2 y→"0")
    dtypes: tuple[str, ...]
    keys: KeyTable                       # unique keys in first-appearance order + int codes (+ indptr if contiguous)
    time: TimeAxis                       # a3 Step + per-instance (first, last) times, int64/datetime64 numpy
    source: AdapterRef                   # which adapter, plus egress metadata (column naming convention, tz, unit)
    is_sorted: bool                      # (key, time) contiguity; guards every positional fast path (wrap_critic x7/x2 verifiers)

@runtime_checkable
class FrameAdapter(Protocol):
    name: str                            # "pandas", "polars", "polars-lazy", "pyarrow", "duckdb", "dask"
    is_lazy: bool
    def matches(self, obj) -> bool: ...                                    # a1 P1; exact type, no hasattr duck typing
    # --- REQUIRED (8) ---
    def infer_schema(self, obj, *, scitype_hint=None, known_mtype=None) -> FrameSchema: ...
        # ONE vectorized validation pass (x9 vectorized monotonic check with null guard; x4 known-mtype pass-through)
    def select(self, obj, schema, *, instances=None, cols=None, lookback: int | None = None) -> Any: ...
        # pushdown: tail(lookback).over(keys), instance filter, projection; lazy backends stay lazy
    def collect(self, obj, *, backend: str | None = None) -> Any: ...      # identity for eager; ONE collect for lazy
    def to_family(self, obj, schema, family: str) -> Any: ...             # whole-frame conversion, ONCE
    def from_family(self, inner, schema, *, like) -> Any: ...              # whole-frame egress, ONCE
    def time_extreme(self, obj, schema, how: Literal["max", "min"], per_instance=False) -> np.ndarray: ...
        # a1 P10; replaces get_cutoff's hidden whole-frame convert_to; how="min" fixes x4 A4 reverse_order
    def time_contains(self, x, y, x_schema, y_schema) -> ContainmentResult: ...
        # a1 P13; anti-join on (keys, time) for every scitype (g5 inv 4: S4/S5/A1)
    def egress_long(self, table: LongTable, spec: OutputSpec, *, like) -> Any: ...
        # build the user-facing frame from numpy columns + codes (point preds, long proba, residuals)
    # --- OPTIONAL (default implementations go through to_family("pandas")) ---
    def upsert_rows(self, obj, new, schema) -> Any: ...                    # a1 P11 (update_data)
    def take_window(self, obj, schema, start, stop) -> Any: ...            # a1 P12 (update_predict windows, no .iloc)
    def aligned_binary(self, a, b, schema, op) -> Any: ...                 # a1 P14 (predict_residuals)

class InnerFamily(Protocol):             # per inner representation, NOT per user backend
    name: str                            # "pandas" | "np_grouped" | "polars"
    mtypes: dict[str, str]               # scitype -> mtype string, e.g. {"Series": "pd.Series", "Panel": "pd-multiindex"}
    def iter_slices(self, inner, schema, *, iterate_as, iterate_cols) -> Iterator[tuple[Key, Any]]: ...  # a1 P6/P7
    def concat(self, parts, keys, schema, *, col_mode) -> Any: ...         # a1 P8/P9 (incl. the _base.py:2466 droplevel)
    def to_long(self, inner, schema) -> LongTable: ...                     # numpy columns + key codes (for egress_long)

class AdapterRegistry:
    def register(self, adapter: FrameAdapter, types: tuple[type, ...]) -> None: ...   # public; plus entry point
    def get(self, obj) -> FrameAdapter: ...      # exact type(obj) dict hit → module-root fallback → matches() scan
                                                 # (x4: 101–107 ns per call; called once per public call)
```

Nine required callables (`matches` plus the 8 listed). sklearn has 4, but sktime has index semantics that sklearn
lacks (r1 TL;DR 2). There are three optional ops with a pandas fallback. The pandas adapter is
`infer_schema = today's check_is_scitype metadata`, `to_family("pandas") = identity`, `time_extreme =
get_level_values(-1).max()` (a1 §5.3: 6–17x faster than `get_cutoff`, equal results), and
`egress_long = today's MultiIndex constructors`.

### 2.3 Where each pandas touchpoint category goes (a1 categories)

| a1 category (touchpoints) | Today | In this design |
|---|---|---|
| container ops: group/iterate, slice/select, concat (27) | `VectorizedDF` groupby/xs/concat | `InnerFamily.iter_slices/concat`. The pandas family is today's code moved, verified identical (x4 T1). Optional: positional slicing with codes (g3 C4 hub iteration: 0.93–1.14x time, 1.4–2.2x lower peak RSS) |
| pivot / estimator grid (15) | `DataFrame.pivot` | `EstimatorGrid` (a1 §6: 4–82x build, 26–314x consume; 0 new failures in 204 tests), plus a pandas-compat shim (§3) |
| MultiIndex levels/names (23) | `.index.names`, `droplevel` | `FrameSchema.key_names/time_name` (logical). The adapter maps them to physical names at egress (g5 inv 2) |
| time index / freq / cutoff (78) | `pd.infer_freq`, `Index.freq`, `get_cutoff` | `TimeAxis` (a3 `Step`, inferred once per panel in numpy; g2 §1). `time_extreme` gives per-instance and global cutoffs. pandas is materialised only for the public 1-element cutoff and H-element fh (g2 §2). The fh cache is keyed by (fh, Step, cutoff value) (g3 C3: 1.05–1.53x) |
| metadata: columns/dtypes (13) | `X.columns`, `dtypes` | `FrameSchema.value_cols/dtypes` |
| type dispatch (34) | `isinstance` / `hasattr(.,"index"/"X")` | `AdapterRegistry.get` only. My census: 21 non-test `hasattr(.,'index')` sites remain in the codebase (`design_adapter_contract_census.json`). The 2 in `_utilities.py` and the stale `hasattr(obj,"X")` go in phase 0 (g5 S4/S6) |
| mtype pinning (25) | `"pd.Series"` defaults, `convert_to` targets | `family_of(y_inner_mtype)`. Defaults stay `pd.Series` / `pd.DataFrame` (no estimator change) |
| output construction (14) | `pd.MultiIndex.from_product` (proba), `pd.concat` | `OutputSpec` + `egress_long` (g4 long schema). Pandas egress rebuilds today's layout exactly (g4 §3a) |
| public-API value types (44) | `forecasters_`, cutoff, proba columns, param keys | §3 |

### 2.4 Data flow by input container

| input | ingress (once) | iteration | egress | expected vs stock |
|---|---|---|---|---|
| **pandas** | `infer_schema` = today's checks, with known-mtype pass-through | pandas family (today's slices) + trusted core | today's constructors | C1–C3 levers: 1.55–3.03x fit+predict (g3), identical outputs |
| **polars eager** | vectorized monotonic check (x9 fix), then `to_family("pandas")` once (x9: adapter conversion without the check is 1.4–1.7x of pandas prepare vs 7–36x today) | pandas family + trusted core | `egress_long` straight to polars (no pandas round-trip of the output: x4 T2 reconstruct 0.41–0.75x of pandas) | same as pandas ±20% (g3 C4 pl rows: 1.80–3.06x vs C0) |
| **polars eager, np_grouped forecaster** | zero-copy numpy (x4 slice-extract: polars→numpy shares memory for null-free floats) | none (batched kernel) | `egress_long` → polars, no pandas object (g2 counts) | 34–105x on polars, limited by 2 stock polars checks, which phase 0 removes (g2 profile: 99%). After the fix *expected* ≈ pandas-input 498–2,735x (unmeasured) |
| **polars lazy / duckdb** | `select(lookback, instances, cols)` pushed into the plan, then `collect(backend="polars"|"pyarrow")` **once**. Narwhals C1: duckdb defaults to pyarrow, so the backend is passed explicitly | as eager | eager frame in the collect backend (g4: proba outputs are O(n·h), eager is fine) | **unchanged-forecaster** workload: ±15% of eager (g3 "laziness pays nothing"). **Selective read**: up to 875x / 25x memory (final_bench T6, B2 vs B1); B5b lazy-then-adapter is 332x |
| **pyarrow** | x4 `PyArrowAdapter` (93 LOC, run detection with a sort fallback for non-contiguous rows) + g2 `pa_*` mtypes | pandas or np family | pyarrow table | g2: 498–1,486x for ported kernels |
| **dask** | v1: `collect(backend="pandas")` at the boundary (today's behaviour once the tag bug is fixed; x8: 1.2–1.3x of pandas) | pandas | dask `from_pandas` | ≈ pandas. Partition-and-delegate is phase 4 and **unmeasured** (r4 verifier: Fugue-style 1.2–1.5x only) |

**Rule:** no adapter op runs inside the per-instance loop. The loop sees only inner-family views, so
per-slice conversion (x4 C: 1.5–4.3x worse) cannot happen by construction.

---

## 3. Public API: what users see

| contract (a1 C1–C4) | phase-1 behaviour | target | deprecation |
|---|---|---|---|
| **`predict` / `update_predict` / `predict_residuals` / `evaluate` return type** | follows the input container. Fix: `update_predict` formats internally and converts **once after** (g5 S1); residuals = `aligned_binary` with fh from the time level (g5 S2/S3, which also fixes Panel residuals for pandas); `evaluate` uses adapter ingress/egress instead of `PANDAS_MTYPES` at entry (g5 S9) | same | none (these are bug fixes; outputs are pandas for pandas input) |
| **C2 `cutoff`, `fh`** | stay pandas (`pd.Index` length 1, `ForecastingHorizon`). They are O(1)/O(h) metadata (x4 §4.1; g5 inv 1). freq comes from `TimeAxis`, so it survives every re-layout (g5 H1) and works with <3 points (g5 S8) | add `get_cutoff(per_instance=True)`, which returns a native frame (keys, cutoff) in the input backend. **pandas stays a hard dependency** for these value types | none |
| **C3 proba columns** | pandas input: today's MultiIndex, unchanged forever. Non-pandas input: config `output:proba_format="pandas"` (default) + `FutureWarning` | default `"native"` for non-pandas input in N+2 (long schema `keys…, time, variable, coverage, lower, upper`; g4 §6) | N: warn; N+2: flip for non-pandas only. `predict_var(cov=False)` keeps the predict layout; `predict_proba` stays a skpro object + `to_long()` |
| **C1 `forecasters_`** | becomes an `EstimatorGrid` behind `PandasCompatGrid` (`.loc/.iloc/.values/.iterrows/.index/.columns` via a lazy `to_pandas`). The compat attributes emit `FutureWarning` | pure `EstimatorGrid` (`get(key)`, `items()`, `shape`, `to_pandas()`) in N+2 | My census: only **2 non-test pandas-attribute reads** of `forecasters_` in sktime (`cinn.py`, `_base.py`), so the internal blast radius is tiny. `vectorize_est` has **15 non-test call sites in 6 files**. It splits into `return_type="grid"` (estimators) and `"table"` (metrics keep a backend table; a1: 4/4 metric configs break otherwise) |
| **C4 `get_fitted_params` keys** | keep `"forecasters.loc[<key>,<col>]__param"` strings. They are strings and need no pandas at runtime (census: 2 sites, 1 file) | unchanged. Add alias keys `"forecasters[<key>]__param"` only if users ask | none (cheapest contract; changing it breaks user `set_params` code for no gain) |
| **Opt-outs** | config `trusted_vectorize=False`, class tag `capability:trusted_vectorize=False` (g1 §4; instance `set_tags` does not work because of `_reset_at`) | same | — |

Net effect for users: a pandas user sees nothing change apart from speed. A polars/pyarrow user gets
consistent native outputs for point predictions straight away, and for proba after N+2. `cutoff`/`fh` stay pandas by design.

---

## 4. Estimator migration (~160 forecasters, a2 buckets)

| group (source) | count | what changes | who does it |
|---|---|---|---|
| **all registered forecasters** | 160 | nothing. They get the trusted path automatically (159/160 eligible; ChronosForecaster falls back because it overrides `predict`; g1 §4) | framework |
| pandas-inner leaves under VectorizedDF (g1 census) | 123 eligible | none. Inner family = pandas. Same outputs and state (g1: 7710/7710 tests) | framework |
| **pd-multiindex-native** (x4 verifier) | 29 | none. One whole-frame `to_family("pandas")`. They gain only from the phase-0 fixes (no VectorizedDF on their path) | framework |
| **V1 boundary-only** (a2) | 63 (39%) | *optional*: declare `y_inner_mtype="np_grouped"` and rewrite `_fit/_predict` on `values/indptr/TimeAxis`. Port LOC: Poly 74 vs 78 stock (g2 §6). Output index from `TimeAxis.add_steps(cutoffs, fh)` | estimator owners, opt-in, prioritised by usage |
| **V2 glue rewrite** (a2) | 56 (35%) | as V1, plus replacing label ops with positional ones. The shared `_BaseWindowForecaster._get_last_window` line (`_sktime.py:151`) unlocks 7 classes in one go (a2) | core team for the shared bases, then owners |
| **V3 core rewrite** (a2) | 13 (8%) | Naive: g2 port (116 LOC + 70 shared). Reducers: per-series fit on `sliding_window_view` (g2 reducer 40 LOC, but only 6–9x because sklearn per-series fit dominates) | core team |
| **V4 third-party bound** (a2) | 28 (17.5%) | stay pandas. For darts/neuralforecast (8 classes accept polars natively, a2 verifier), a `polars` inner family could pass frames straight through (**unmeasured**, G6) | later, opt-in |
| **composites** (TTF, pipelines, tuners, Reconciler) | ~31 (a2: classes using neither fh nor cutoff) | inner-mtype tags become **dynamic**, derived from components (sklearn `array_api_support` pattern, r1 TL;DR 3), so a composite of np_grouped leaves gets np_grouped. Today they silently convert to pandas (g5 §1 / inv 8) | framework |
| nested transformers inside forecasters (Lag in Naive, Detrender) | — | trusted `BaseTransformer` entry, the same design as g1 (Lag is 63% of Naive's residual time, g3) | framework, phase 2b, *unmeasured/expected* |

New tags: `capability:trusted_vectorize` (bool, default True), `y_inner_mtype` accepts `np_grouped*`, and
`capability:lookback` (int | "sp" | None, used by `select` pushdown; G9, *unmeasured*). Composite tag
derivation is in `_HeterogenousMetaEstimator`.

---

## 5. Phased roadmap

### Phase 0: decision-independent fixes (pandas/polars, ~1 release)
- Vectorized polars monotonic check with a null guard. Pass the known mtype from `_check_X_y` into `prepare`.
  `get_cutoff` stops converting whole frames (x4 §7.1). `_HashIndex.__eq__` (fh cache). Guard `pd.infer_freq` (<3 points,
  g5 S8). Remove the stale `hasattr(obj,"X")` at `_utilities.py:59/221`. Fix the polars `y`→`"0"` egress (g5 S7).
- **Impact:** g3 C1 1.05–1.34x e2e fit+predict. Polars prepare at 10000x50: 1443 → 16 ms (x4 cloud re-run;
  x4 verifier 1962 → 11.9 ms). Fixes g5 S6/S8/S7 and a1 column-"X".
- **Risk:** low. The vectorized check is not equivalent with nulls unless guarded (x9 verifier: 58/75
  mismatches without the guard).

### Phase 1: adapter registry + VectorizedDF on inner families (no estimator change)
- `FrameAdapter` (pandas, polars, polars-lazy, pyarrow), `InnerFamily("pandas")` (today's VectorizedDF code
  moved), `AdapterRegistry` with public `register_adapter`/`register_mtype` (g2 §1.1), `FrameSchema`, `TimeAxis`.
  `time_extreme(how="min"|"max")`. Unnamed-level logical names. Replace `droplevel(1)` with `InnerFamily.concat(col_mode=…)`.
  Adapter conformance suite = g5 matrix + wrap_critic M4 adversarial fixtures, run per adapter.
- **Impact:** pandas: 0.89–1.06x (x4 P), so **neutral by design**, with outputs bit-identical (x4 78/78, 28/28).
  Polars input: prepare+iterate+reconstruct 2.2–3.7x faster (x4 verifier L_total). End to end only up to
  ~1.2x (x4 verifier B2/A0; x8 container-only ≤1.05x). Correctness: g5 polars strict-equal cells should rise
  from 9/24 (stock) toward 20+/24 (*expected*: S1, S2, S4, S7, S8, A4 are fixed by named ops). LOC: ~230
  generic + 67–100 per adapter (x4 §6).
- **Risk:** silent per-backend divergence (skrub #2335 class; x4 already had 3 such bugs). Mitigated by the
  conformance suite as a CI gate.

### Phase 2: BaseForecaster boundary: validate once, trusted core, panel-level time
- (a) The g1 trusted per-slice path, refactored so the public methods = `validate` + shared `trusted_core`
  (prevents drift; g1 caveat). (b) Top-level `time_contains` anti-join for every scitype (g5 inv 4).
  (c) Outer cutoff via `time_extreme`; `Step` inferred once per panel; fh cache per unique cutoff (g3 C3).
  (d) Proba: `OutputSpec`-based egress, removing the per-instance `_get_columns` MultiIndex build (g4: 0.7–0.8
  ms/instance) and `pd.concat` (long→MI 0.07–0.42x of today's concat). (e) `forecasters_` → `EstimatorGrid`
  + compat shim. (f) Optional: positional pandas-family iteration (g3 C4).
- **Impact:** cumulative C0→C3 **1.55–3.03x** fit+predict, C4 1.59–3.31x (g3, Naive and Poly, pandas/polars/lazy, 100–5000
  series). Whole fit/predict/interval/update/predict workload **1.07–1.72x** (g1). Update 1.3–3.2x (g1).
  Peak RSS 1.4–2.2x lower with (f) (g3 C4: 5000x100 Poly 161 → 54 MB on polars). Tests: 7710/7710 pass in g1.
- **Risk:** medium. Invariants (g1 §4: guarded overrides, per-slice containment, fresh-clone semantics,
  default `_update` refit) must be encoded as tests. Predict/predict_interval stay ≈1.0x until (g) a trusted
  nested-transformer entry exists (g1 §3.3; *unmeasured/expected*).

### Phase 3: estimators opt in to `np_grouped` (and later `polars`) inner families
- Ship `InnerFamily("np_grouped")` + `TimeAxis` arithmetic (g2). Port the closed-form/batched forecasters first (Naive,
  PolynomialTrend, mean/drift/seasonal naive, Croston-like; x7 verifier: strictly 6–9 classes batchable), then the shared
  window base (7 classes via `_sktime.py:151`), then per-series V1 wrappers on numpy views.
- **Impact:** batched ports **498–2,735x** e2e on pandas input, 498–1,486x on pyarrow, 34–105x on polars (g2
  at 1000x100; polars limited by stock checks that phase 0 removes). g3 C5: 106–4,560x vs stock. Per-series
  third-party ports (reducer): **6–9x** only (g2). Conversions per 100x100 fit+predict: ~804 → 2; mtype checks
  7,484–12,084 → 36 (g2 counts).
- **Risk:** each port must replicate stock quirks to stay equal (sklearn `cond=tol` truncation, drift variance;
  g2 caveats). There is no per-instance `forecasters_` for batched ports, so `forecasters_` must become optional for
  `np_grouped` estimators (contract §3).

### Phase 4: lazy and distributed
- `select(lookback=…)` driven by `capability:lookback`, plus user `instances=`/`cols=`. `collect(backend=…)` explicit
  (narwhals C1 lesson). dask partition-and-delegate behind an adapter op `map_partitions(fn, schema)` *only if* the
  G7 experiment shows a break-even.
- **Impact:** selective reads 190–875x faster and 15–70x less memory (final_bench T6: B5b 332x / 44 MB, B2 875x / 31
  MB vs 5.15 s / 786 MB). True pushdown over select-before-convert: 1.9x time / 2.8x RSS (x9 verifier).
  Full-history workloads: ±15% (g3). dask: **unmeasured**.
- **Risk:** the lookback census is unknown (G9). Wrong lookback means wrong forecasts, so the tag must be
  conservative (`None` = full history).

### Phase 5: contract flips (N+2)
- `output:proba_format` default `native` for non-pandas input. `forecasters_` compat shim removed. `evaluate`/tuners
  run natively.
- **Impact:** no speed (C1 grid is ≲1% of fit, a1 §6). Removes mixed return types for polars users (a1 TL;DR 3).
- **Risk:** downstream code that indexes `forecasters_`. Low inside sktime (2 sites, my census), unknown outside.

---

## 6. Evidence against this design (honest)

1. **The adapter itself is not the lever.** Container-only changes are capped at ~1.05x (x8). The adapter's
   zero-conversion polars path is 0.96–1.17x of today's pandas path (x4 verifier). The container layer is 13.5–21.7% of
   fit (a1 §5.2). Every big number in §5 comes from g1/g3/g2, which would work equally well behind narwhals or a hub.
2. **Adapters multiply surface and divergence.** a1 needs 14 ops against sklearn's 4. skrub's own layer grew to
   72+ generic ops and still diverged silently (r1). x4's prototype already had 3 silent divergences: wrong
   `reverse_order` cutoff, unnamed level names, `key__col` naming (x4 verifier §7). On top of that, g5 A1 (lost
   disjoint-X check) and A2 (y→"0").
3. **Per-backend cost is not small.** An adapter is 67–100 LOC, but a forecaster-consumable backend also needs
   an mtype family (~370 LOC for polars; x4 §6). Narwhals covers 6 backends for one code path (x5: type
   round trip on 6 backends).
4. **Narwhals beats per-backend hand-written ops on polars round trips** (x5 0.24–0.5x of today's pandas round
   trip vs x4 adapter[polars] 0.7–1.9x; final_bench T3b). My design keeps narwhals out of the hot path, but could
   use it *inside* the polars/duckdb adapters for `select`/`collect`, a hybrid this report does not measure.
5. **The hub (x6) iterates 54–236x cheaper per slice** (final_bench T1). g3 shows that at the e2e level this is
   worth only 0.93–1.14x in time but 1.4–2.2x in memory. A hub-first design would get the memory win structurally.
   Mine gets it only through the optional positional pandas-family iteration.
6. **pandas stays a hard dependency** (cutoff/fh value types, proba for pandas input, skpro). A "no pandas installed"
   goal (as in sklearn r1: 4/4 with pandas absent) is not achieved and is not a goal here.
7. **Composites are the weak spot for any boundary design.** Tuners and `evaluate` currently convert to pandas on entry
   (g5 S9). Dynamic tag derivation for composites is required and unprototyped.
8. **Lazy value is narrow.** Laziness cannot survive per-instance forecasters (x4 Z 3–77x; g3). The 875x
   headline applies only to selective reads, and most of the older x9 headline was the monotonic bug (wrap_critic §2).

---

## 7. Open experiments (to firm up the numbers above)

1. **Adapter conformance:** implement phase-1 PandasAdapter/PolarsAdapter with `time_extreme(min|max)`,
   `time_contains` anti-join and label-preserving egress. Run the g5 matrix + M4 adversarial suite. Target ≥20/24
   polars strict-equal with 0 SILENT cells.
2. **Phase-0 combined on polars np_grouped:** re-run g2 at 1000x100 with the vectorized polars check + known mtype, to
   confirm the expected polars ≈ pandas-input gain (g2: 99% of time is in the two stock checks).
3. **Positional pandas family under the trusted core** vs stock `items()` at 1k–10k series, staggered/gappy layouts
   (wrap_critic M3, K3). Expect ≤1.1x time and 1.4–2.2x RSS (g3 C4).
4. **Trusted nested BaseTransformer** (Lag, Detrender) to unlock predict/predict_interval gains (g1 §3.3; Lag = 63% of C4 Naive).
5. **Lookback census + pushdown through BaseForecaster** (G9): what share of forecasters can declare a finite lookback,
   and e2e Naive(last) fit on 4M-row parquet vs final_bench B5b.
6. **dask partition-and-delegate break-even** with the threaded scheduler (G7).
7. **Hybrid: narwhals inside polars/duckdb adapters** for `select`/`collect` only. Measure LOC saved and overhead
   (r1 verifier: +13–25 µs per compute op, paid once per call here).
8. **Composite dynamic inner-mtype tags:** TTF([Detrender, G2Naive]) staying in `np_grouped` end to end (g5 inv 8).
9. **External `forecasters_` usage:** run a downstream corpus (or GitHub code search) with the compat grid warning, to size
   the N+2 risk.

---

## 8. Measurement performed for this report

```
cd /home/user/sktime
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/design_adapter_contract_census.py
# -> results/design_adapter_contract_census.json  (static regex census over sktime/*.py; counts are syntactic sites)
forecasters_ pandas-attr access (.loc/.iloc/.values/.iterrows/.index/.columns/.shape): 2 sites, 2 files, 0 in tests (cinn.py, forecasting/base/_base.py)
forecasters_ any reference: 89 sites, 19 files
vectorize_est( : 18 sites, 15 non-test, 6 non-test files (transformations/base, base/_base_panel, forecasting/base, performance_metrics, datatypes/_vectorize)
'forecasters.loc[' fitted-param keys: 2 sites, 1 file (forecasting/base/_base.py)
MultiIndex.from_product/from_tuples under sktime/forecasting (non-test files): 40 files (upper bound on proba-layout builders; g4 counted 31 leaf files)
hasattr(., 'index') duck typing: 22 sites, 21 non-test
cutoff.freq / cutoff[0]: 39 sites, 28 non-test (stream/_update.py 10, moirai 4, _fh 4, lagllama 3)
get_cutoff( : 36 sites, 8 non-test
```
Caveat: regex census only. Aliased access (e.g. `est = f.forecasters_; est.loc[...]`) is not caught, so
these counts are lower bounds for consumers.
