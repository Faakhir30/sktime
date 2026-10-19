# design_narwhals: a narwhals-first architecture for container-independent sktime forecasting

Angle: **narwhals as the abstraction for the boundary and for panel-native forecasters**, taking
C1 (lazy `collect()` targets) and C2 (no index in narwhals) as hard design constraints instead of
arguing them away. Where the evidence shows narwhals is the wrong tool (pandas-MI iteration, hot
numeric kernels, the per-slice framework cost), this design uses something else and says so.

Evidence keys: `<report>.<claim>` refers to `reports/<report>.md` (and its `_verify0`); "T#" are tables in
`reports/final_bench.md`. New numbers in this report come from two scripts I ran, listed in §8
(`DN-B*`, `DN-R*`). Anything else is labelled *unmeasured/expected*.

---

## 0. Thesis (≤120 words)

Use narwhals (pinned to `narwhals.stable.v2`) as **the single implementation of the panel boundary
for every non-pandas backend**: ingest, validation, cutoffs, X/y checks, lazy selection, egress and
proba long-format. Use it also as **the authoring language for expression-shaped panel-native
kernels**. Keep a hand-written **pandas-MultiIndex lane** for pandas input, because narwhals on pandas
is 1.7–8x slower and cannot see the index (C2). Never call a bare `collect()`; materialize through one
explicit policy (C1). The 1.5–3x gain that holds for all 160 forecasters comes from a container-independent
trusted per-slice lane, not from narwhals. What narwhals adds is that non-pandas users stop paying a penalty, plus lazy
end-to-end kernels and one portable kernel per algorithm.

---

## 1. What the evidence forces on a narwhals-first design

| constraint | evidence | design consequence |
|---|---|---|
| Container-only change under per-instance BaseForecaster is capped at ≈1.05x | x8.1, wrap_critic §0 | The narwhals layer is not sold as a speed lever for the 160 per-instance forecasters. The speed for them comes from the trusted lane (g1, g3 C2–C3). |
| narwhals VectorizedDF on pandas: 1.7–2.3x (fast path) to 5.7–8.3x (generic) slower per slice; round trip 1.8–5.7x | T1, T3b, x5.3 (verifier ×5.86–6.30 / ×1.88–2.0) | **pandas lane stays native** (today's `groupby(level)` code plus fixes). narwhals is not on the pandas per-slice path. |
| narwhals validation on pandas-long is 70–620x slower than sktime's pd-multiindex check | **DN-B1** (new) | Validation of pandas input stays native. narwhals validates only non-pandas input. |
| C2: index invisible. `maybe_reset_index` drops levels. Native `reset_index` + schema recipe restores names, dtypes, freq, Period and order in 198/198 checks | x2.TL;DR, x2 recipe, x5.7 | The `PanelSchema` carries key/time names, level dtypes, freq/Step, Period flag and column labels. The narwhals frame is always **long** (keys and time as columns). `maybe_reset_index` is banned. Group-by on index level names (a pandas leak) is banned. |
| C1: `collect()` defaults to polars→polars, dask→pandas, duckdb/ibis→pyarrow. `collect(backend=)` only accepts pandas, polars or pyarrow | x1.TL;DR, x5.6, r2.C1; **DN-API** confirms this under stable.v2 | One `materialize(frame, purpose)` function with an explicit target. Lazy type is kept only by never collecting (kernels). Outputs are relifted with `.lazy(backend=orig)` (0.82–1.92x a native collect, x1). |
| pyarrow narwhals backend silently mis-assigns `over()` results at ≥~500 groups (60–100% of rows wrong) | x5 §5 bug 1, x5 verifier K3 | **pyarrow input is never computed on the narwhals pyarrow backend.** It is rerouted to polars when polars is installed (0.4–0.7x the cost of `to_pandas`, DN-B5) or to pandas otherwise. |
| `Expr.last(order_by=)` is absent in stable.v1/v2. dask has no `over(order_by)`. Under stable.v1, duckdb/ibis are not LazyFrames | x1 verifier §2a, K7; **DN-API** reproduces `last(order_by)` TypeError and dask `over(order_by)` NotImplementedError under v2. Under v2, duckdb and dask are `LazyFrame` | Pin `stable.v2`. Kernels use the rank/semi-join formulations (`with_row_index(order_by=)` + per-group max, or a join on `max(time)`). Both work in v2 (DN-API). |
| Real gains are vectorized algorithms (10²–10³x). The engine adds 1.07–3x | x7, x8.8/9, g3 C5 | narwhals expression kernels exist for the naive/window/drift family. Numeric kernels (OLS, recursions) extract numpy segments through the boundary instead of being written in narwhals. |
| narwhals costs 7–11x over hand-written native code on the same backend (x5 verifier). narwhals-on-pandas is 9.8–11.5x slower than plain pandas groupby | x5 verifier §speed attribution | Kernels on pandas input are **routed to polars** when it is installed: 2.0–11.8x faster than narwhals-on-pandas and identical output (**DN-R**). A per-backend native override hook stays available for hot kernels. |

---

## 2. Architecture

```
            user frame (pandas MI | polars | pl.LazyFrame | pyarrow | duckdb | ibis | dask)
                                  │  backend_of()  (explicit registry; no hasattr duck typing)
          ┌───────────────────────┴───────────────────────────┐
   pandas lane (native)                                narwhals lane (nw.stable.v2)
   PandasMIOps: today's groupby(level),                NwPanelOps: long frame + PanelSchema,
   known-mtype pass-through, freq fixes                eager or lazy, validation via expressions
          └───────────────────────┬───────────────────────────┘
                         PanelFrame(schema, native, lane)
                                  │
     ┌────────────────────────────┼─────────────────────────────────────┐
  TimeAxis (Step, per-panel     Execution dispatch on the ESTIMATOR, not the input
  cutoffs, fh memo)               ├─ PanelNativeLane: est has capability:panel_native
                                  │     kernel(PanelFrame) -> state frame -> forecast long frame
                                  │     (nw expressions, or numpy segments via to_numpy)
                                  ├─ GlobalLane: y_inner_mtype in {pd-multiindex, "nw_long"}
                                  │     one conversion of the whole frame
                                  └─ PerInstanceLane (default, 160/160): ONE whole-frame conversion
                                        to the inner mtype + trusted per-slice _fit/_predict (g1)
                                  │
                          Egress: output follows input (relift for lazy), proba long schema
```

### 2.1 Core types and protocols (signatures)

```python
# sktime/datatypes/_panel/_schema.py
@dataclass(frozen=True)
class PanelSchema:
    key_names: tuple[str | None, ...]       # original level names (may be None), C2 recipe
    key_cols: tuple[str, ...]               # physical column names in the long frame (__index__k)
    time_name: str | None; time_col: str
    time_kind: Literal["datetime", "period", "int"]
    time_dtype: str                         # e.g. "datetime64[ns]"; egress restores it (DN-R found ns->us drift)
    step: Step | None                       # a3 Step (freq + anchor + holidays); None = irregular
    value_labels: tuple[Hashable, ...]      # original column labels (ints allowed); physical str names
    value_cols: tuple[str, ...]
    scitype: Literal["Series", "Panel", "Hierarchical"]
    n_instances: int | None                 # None for lazy until needed
    order: Literal["first_appearance", "sorted"]   # lazy => sorted (no row order, x5 §6.2)
    backend: str                            # nw.Implementation name of the USER input
    is_lazy: bool
    validated: frozenset[str]               # {"nonnull_keys","sorted","unique","contiguous"} already proven

@dataclass
class PanelFrame:
    schema: PanelSchema
    native: Any                             # pandas MI frame (pandas lane) or native long frame
    lane: Literal["pandas", "narwhals"]
    def nw(self) -> "nw.DataFrame | nw.LazyFrame": ...   # narwhals lane: wrap once (r1: 11–18 µs per wrap)

class PanelOps(Protocol):                   # a1 P1–P14 collapsed; two implementations
    def ingest(self, obj, known_mtype: str | None) -> PanelFrame: ...          # P1,P2,P3,P4,P5
    def validate(self, pf: PanelFrame, checks: set[str]) -> PanelFrame: ...  # top level, once (g5 inv. 4)
    def cutoffs(self, pf) -> "nw.DataFrame":                                # P10: keys, cut_max, cut_min
    def time_contains(self, y: PanelFrame, X: PanelFrame) -> int: ...        # P13: #violating keys
    def upsert(self, old: PanelFrame, new: PanelFrame) -> PanelFrame: ...    # P11 (update)
    def window(self, pf, lookback: int | None, keys=None) -> PanelFrame: ... # P12 + lazy selection
    def aligned_binary(self, a, b, op) -> PanelFrame: ...                    # P14 (residuals, g5 inv. 7)
    def materialize(self, pf, purpose: Literal["iterate","kernel","output"]) -> PanelFrame: ...
    def to_inner(self, pf, inner_mtype: str) -> Any: ...                     # ONE whole-frame conversion
    def iter_slices(self, inner_obj, schema) -> Iterator[tuple[tuple, Any]]: # P6/P7 (pandas only)
    def egress(self, long_out: "nw.DataFrame | pd.DataFrame", schema, kind="point"|"proba") -> Any: ... # P8,P9

class PandasMIOps(PanelOps): ...            # today's code + fixes; never touches narwhals
class NwPanelOps(PanelOps): ...             # one implementation for polars/pl-lazy/duckdb/ibis/dask/modin
```

`materialize` is the only place that collects (C1):

```python
COLLECT_TARGET = {  # purpose="iterate" / "kernel"; always explicit, never collect()
    "polars": "polars", "pyarrow": "polars*", "duckdb": "polars*", "ibis": "polars*",
    "dask": "pandas", "modin": "pandas"}       # * = "pyarrow" only if polars missing -> then reroute to pandas, never compute on nw-pyarrow
```

### 2.2 Panel-native estimators (the narwhals compute surface)

```python
class BaseForecaster:
    _tags = {"capability:panel_native": False,         # x7 hook, generalised
             "panel_kernel:engine": "narwhals",          # "narwhals" | "numpy" | "native"
             "panel_kernel:backends": ("polars", "pandas", "duckdb", "ibis"),  # conformance-tested set
             "lookback_window": None}                    # int | "sp" | None (= all history)

    # narwhals kernel: inputs are ALWAYS long, columns named by schema
    def _fit_panel(self, Y: "nw.LazyFrame | nw.DataFrame", X, schema: PanelSchema,
                   fh: ForecastingHorizon) -> "nw.LazyFrame | nw.DataFrame": ...   # state: one row per key(×season)
    def _predict_panel(self, state, grid: "FutureGrid", X=None) -> "nw.LazyFrame | nw.DataFrame": ...  # long: keys,time,values
    def _predict_interval_panel(self, state, grid, coverage) -> "nw.LazyFrame": ...  # g4 long schema
    def _update_panel(self, state, Y_new, schema) -> state: ...                      # optional, else refit

@dataclass
class FutureGrid:          # computed ONCE per panel from unique cutoffs (a3, g3 C3)
    table: "nw.DataFrame"  # columns: cut (cutoff value), k (step index), time (absolute); |unique cutoffs|×H rows
    rel: np.ndarray        # relative fh steps
```

* **numpy kernels** (`panel_kernel:engine="numpy"`) receive `(values: ndarray[n_rows, n_vars], indptr,
  keys_table, time: ndarray, Step)`. They are built by `NwPanelOps.materialize(..., "kernel")` + sort
  + `to_numpy()`, which is the g2/x6 hub layout reached through narwhals. This is how OLS (Poly),
  Theta closed forms and later numba recursions get the g2 speed, which narwhals expressions cannot reach
  (x7: narwhals plan 2.2–3.0x hand-written cost for naive; recursive models cannot be expressed, x5 §7).
* **Future timestamps** come from the `FutureGrid` table, which a3 `TimeOps` builds in Python/pandas on the
  *unique* cutoffs (a3: 12–17 ms at 10k series). It is joined to the state on `cut`, so no backend needs
  `offset_by("mo")` (missing on pyarrow/dask, x5 §1.3) and business-day calendars stay exact (a3:
  narwhals has no business-day support). For a single unique cutoff the join becomes a `lit` per step
  (x5's form, which avoids the duckdb cross-connection join failure, x5 §6.6).
* **Engine routing for kernels**: pandas input with a narwhals kernel is routed pandas→polars→pandas
  when polars is installed. This is 2.0–11.8x faster than narwhals-on-pandas, with identical output once the
  schema time dtype is restored (DN-R).

### 2.3 Where each a1 pandas-touchpoint category goes

| a1 category (touchpoints) | pandas lane | narwhals lane | stays pandas? |
|---|---|---|---|
| container ops: group/iterate, slice/select, concat (27) | unchanged code (`groupby(level)`, `pd.concat(keys)`); trusted slices | not iterated in narwhals. PerInstanceLane converts the whole frame once (`to_inner`) and iterates the inner pandas object. Kernels never iterate | per-instance lane: yes (inner mtype is pandas) |
| pivot/reshape estimator grid (15) | replaced by `EstimatorGrid` (a1: build 4–82x, consume 26–314x faster) with a DataFrame view for compatibility | same | only in the compatibility view |
| MultiIndex level names (23) | schema-driven, as today | `PanelSchema.key_names/key_cols`; x2 recipe at egress | no |
| time index / freq / cutoff (78) | `Step` + per-panel cutoffs + fh memo (`_HashIndex.__eq__`, g3 C3) | `ops.cutoffs()` = `group_by(keys).agg(max, min)` (DN-B2: 16 ms at 10k polars); `FutureGrid` | public `cutoff` and `fh` values stay pandas Index (O(1)) |
| metadata: columns/dtypes/nans (13) | unchanged | `schema` + `nw` schema; NaN→null normalisation recorded (x5 §6.3) | no |
| output construction (14) | unchanged | `egress()` builds native long output; pandas MI only for pandas input | pandas only for pandas users |
| type dispatch / duck typing (34) | `backend_of` registry (g5 inv. 5: removes the `hasattr(obj,"X")` bug) | same | no |
| mtype pinning (25) | inner-mtype tags (dispatch on estimator, x4) | adds the `nw_long` mtype family incl. lazy (closes x9 "no lazy Panel mtype") | no |
| public-API value types (44) | see §3 | see §3 | partly (by contract) |

---

## 3. Public API: the four contracts and how they evolve

| contract (a1) | today | phase N (no break) | phase N+2 | evidence |
|---|---|---|---|---|
| **Return type of `predict`/`update_predict`/`predict_residuals`/`evaluate` y_pred** | follows input for predict; `evaluate` returns pandas for polars (g5 S9); polars Series column renamed y→"0" (g5) | follows input for all frame outputs. Lazy input → output of the same lazy type (kernel lane: never collected; per-instance lane: relift of the small output). Labels restored from schema | same | g5 inv. 1–2; x5 verifier K4: 0 UDF calls before the user's collect |
| **`cutoff`, `fh`** | pandas `Index` / `ForecastingHorizon` (pandas inside) | unchanged type. Cutoff computed per panel with **both** max and min (fixes x4 `reverse_order`). New `cutoffs_` attribute: per-instance cutoffs as a native frame | unchanged; ForecastingHorizon internals move to `Step` (a3) with a pandas view | g5 inv. 3, 10; a3 |
| **`forecasters_`** (DataFrame of fitted clones) | always | PerInstanceLane: unchanged DataFrame (g1/g3 keep it identical). PanelNativeLane: `fitted_state_` (native params table; x7: 0.23 MB vs 40.7 MB) plus a `forecasters_` **property** that materialises clones on access with a `FutureWarning` (cost = stock loop, x7/g4: 3.58 s at 200 series) | `forecasters_` access on panel-native estimators returns an `EstimatorGrid` view; DataFrame only via `.to_frame()` | a1 EstimatorGrid (0 new behavioural failures in 204 tests; 4/4 metric configs break → metrics must move first) |
| **proba MultiIndex columns** | always pandas MI, even for polars input | config `output:proba_format="pandas"|"native"`, default "pandas". FutureWarning for non-pandas input. narwhals long schema `keys..., time, variable, coverage, lower, upper` (g4) is backend-portable (scalar columns only) | default "native" **for non-pandas input only**. Pandas keeps the MI indefinitely | g4 §6 (long→MI 14–85 ms vs 205 ms concat at 10k) |
| **`get_fitted_params` keys `forecasters.loc[..]`** | per instance | unchanged for PerInstanceLane. Panel-native: `fitted_state` (table) + `fitted_state__<col>` arrays | deprecate the `.loc[...]` string keys in favour of `get_fitted_params(style="table")` | a1 contract 4 |

Deprecations ride on sktime's usual 2-minor-release cycle. Nothing changes for pandas users except
bug fixes, plus identical-output speed (g1/g3).

---

## 4. Estimator migration (a2 buckets, 160 classes)

| bucket (a2) | count | what changes | who does it |
|---|---|---|---|
| all | 160 | Nothing is required. PerInstanceLane + trusted per-slice path: 159/160 eligible (g1; Chronos falls back) | framework |
| V1 boundary-only | 63 (39%) | none. They receive pandas from `to_inner`, as today | — |
| V2 glue edits | 56 (35%) | optional; fixing `_sktime.py:151` last-window `.loc` removes 7/14 numpy-like failures (a2) | maintainers, as needed |
| V3 core rewrite (naive, reducers, ...) | 13 (8%) | candidates for `capability:panel_native`: NaiveForecaster (narwhals kernel: last/sp/mean/window/drift, x5 identical to stock), PolynomialTrend (numpy kernel, g2/g3 C5), Theta/SES closed forms (numpy; SES as narwhals `ewm_mean` only on pandas/polars, x5) | core team, ~9 classes (x7 count) |
| V4 third-party-locked | 28 (17.5%) | unchanged (pandas). darts/neuralforecast/Nixtla wrappers that accept polars (8 classes, a2 verifier) may declare `y_inner_mtype="nw_long"` and receive the native long frame (*unmeasured/expected*, G6) | wrapper owners |
| pd-multiindex-native globals | 29 | unchanged by default; may opt into `nw_long` | wrapper owners |
| composites (TTF, pipelines, tuners, reconciler) | ~41 delegate-capable (x7) | forward `capability:panel_native` when all steps support it, and go through the same `PanelOps` dispatch (g5 inv. 8). Tuners/`evaluate` get a native window splitter (`ops.window`) instead of convert-to-pandas-at-entry (g5 S9) | core team |

Port cost reference: x5 NwPanelNaive 205 SLOC covers 4 strategies × 6–7 backends. g2 Naive port 116
logical LOC vs 358 stock. Every narwhals kernel must pass a conformance suite of backends × strategies at
**≥1000 groups** (x5 §6.7: the pyarrow bug does not appear at n=20) plus the M4 adversarial fixtures.

---

## 5. Roadmap (VectorizedDF → BaseForecaster boundary → estimators)

Phase ratios are end-to-end fit+predict vs stock unless stated. "Cumulative" means on top of the previous phases.

| phase | scope | expected impact (cited) | risk |
|---|---|---|---|
| **P0 bug fixes** (container-independent, ~weeks) | vectorized polars monotonic check with a null guard; pass the known mtype / `convert(from_type)`; `_HashIndex.__eq__` for the fh memo; stock bugs: polars `update` with 1–2 rows, Panel `predict_residuals`, `hasattr(X,"X")`, polars X/y containment no-op, pd3 caller mutation, y→"0" rename | g3 C1: **1.05–1.34x** e2e (pandas 1.05–1.22, polars 1.10–1.34). x9: monotonic check 921→33 ms (28x). x4: polars prepare 1443→16 ms at 10000x50 | low; x9 verifier: vectorized check not equivalent with nulls (58/75) unless null-guarded |
| **P1 VectorizedDF** | `PanelFrame`/`PanelSchema`; `PandasMIOps` (today's code) + `NwPanelOps` (narwhals stable.v2) for every other backend; top-level narwhals validation; one whole-frame `to_inner`; **trusted per-slice lane** (g1) + per-panel time axis (g3 C3) | All containers: g3 C3 cumulative **1.55–3.03x** (naive 1.55–2.02, poly 2.23–3.03); g1 full fit/predict/interval/update workload **1.07–1.72x**, outputs identical, 7710 tests pass delta 0. Non-pandas-specific: validation 1.23–1.56 s → 15–16 ms at 10k (**0.010–0.025x** of sktime's polars check, DN-B1); polars→pandas via adapter without the check 0.11 s vs 1.33 s convert_to at 10k (T3). So polars users stop paying today's 1.04–1.18x penalty over pandas (T4) (*expected*). Ingest memory +6.8–29 MB vs +27–46 MB (T5, x5 prepare) | medium: the trusted lane duplicates `_check_fh`/`update_data` logic (g1 caveat). Must share code with the public methods |
| **P2 BaseForecaster boundary** | `nw_long` mtype family (eager+lazy) registered via a **public `register_mtype`** (g2: private-cache mutation unregisters silently); `backend_of` registry; output-follows-input + relift; top-level X/y containment (DN-B3: 19–26 ms at 10k, flags 10000/10000 bad keys, 0 false positives); cutoffs max+min (DN-B2); proba long format behind config (g4); `forecasters_`/`fitted_state_` split; native `ops.window` splitter for `update_predict`/`evaluate` | g5 coverage target: polars cells "works & strict-equal" from 9/24 (stock) → ≥22/24, i.e. pandas parity (*expected*). Proba egress 14–85 ms vs 205 ms concat at 10k (g4). Lazy input becomes accepted (x9: today TypeError). Selective reads pushed before ingest: up to **875x / 25x less memory** (T6 B2 vs B1), 332x with adapter conversion (T6 B5b). **Lookback-only** pushdown does *not* pay in time: semi-join 0.26–0.64x of collect-all on polars/duckdb, 1.13–2.10x only when the to_pandas is charged at 10k, **0.09–0.12x on dask** (DN-B4) → `lookback_window` is applied only on polars/duckdb/ibis, never dask | medium: lazy semantics (no row order → sorted output order for lazy, x5 §6.2); duckdb relift lands on the default connection (x1) |
| **P3 panel-native estimators** | `capability:panel_native` with narwhals kernels (Naive family incl. intervals) and numpy-segment kernels (Poly/OLS, Theta-closed-form); kernel routing pandas→polars; composites delegate | Naive on polars **36–943x**, pandas MI 14–147x vs the stock loop, numerically identical (x5.4/.5; verifier ×304 / ×88 at 1k). Routing pandas input to polars: 2.0–11.8x more than narwhals-on-pandas (DN-R). Numpy kernels: g2 **498–2,735x** (pandas/pyarrow), g3 C5 105–4,560x. Native interval kernel 16 ms vs 18.1 s at 1k (~1100x, g4). Coverage ≈9/160 leaves + ~41 composites (x7) | high per kernel: silent backend bugs (x5 §5). Mitigated by the backend allow-list + ≥1000-group conformance tests. Behavioural differences found by the x5 verifier (int-time pandas MI, pd3 Period M/Q, series shorter than sp, drift NaN raising) must be fixed before shipping |
| **P4 globals + contract flips** (N+2) | `nw_long` inner for darts/neuralforecast/Nixtla wrappers; proba default "native" for non-pandas; `forecasters_` → EstimatorGrid for panel-native | G6 *unmeasured*: r3 polars-native vs convert-to-pandas 1.24–1.77x for statsforecast Naive. Fitted state 175x smaller (x7) | medium: metrics consume `forecasters_` DataFrames (a1: 4/4 configs break) |

What realistic users get: a pandas user with a per-instance forecaster gets 1.5–3x (P0+P1). A polars user
gets the same, plus the removal of a 1.04–1.18x penalty and 1.6–4x lower ingest memory (T5, x5 prepare). A lazy/parquet user gets
pushdown of selections (10²x) and kernels that never collect. Users of the ~9 kernelized
forecasters get 10²–10³x.

---

## 6. Evidence

### For
* C1 handled: under stable.v2, duckdb and dask are `LazyFrame`; `collect(backend="polars"|"pandas")` works; `collect().lazy(backend="duckdb")` relifts (DN-API). Lazy kernels keep the input type end to end, with 0 hidden UDF calls (x5 verifier K4).
* C2 handled: the schema recipe passes 198/198 fidelity checks on pd2 and pd3 (x2). x5's narwhals VectorizedDF matches stock 17/17 on pandas (verified) and round-trips the type on 6 backends.
* A single narwhals implementation validates non-pandas input 40–100x faster than sktime's polars check and catches shuffled rows, duplicates and null keys (DN-B1). Cutoffs, containment and lookback each take one portable expression and run on polars/pl-lazy/duckdb/dask (DN-B2–B4).
* Panel-native narwhals kernels: Naive family identical to stock on 7–8 backends, ×36–943 on polars (x5). Lazy plan 4.4 ms (x5 §4.3).
* Prior art: sklearn ≥1.9 hard-depends on narwhals at the boundary (r1), so it is already a transitive dependency. darts, Nixtla, scikit-lego and fairlearn use it at their edges (r2). pandera's narwhals backend uses the same "pandas index → native backend" split (r4).
* Non-pandas ingest memory: polars prepare +6.8–29 MB vs +27–46 MB today (T5).

### Against (strongest first, honestly)
1. **narwhals is not where the speed comes from.** For the 160 per-instance forecasters every measured gain is container-independent: g3 C1–C3 (1.55–3.03x) and g1 work the same on pandas and polars. The hub/positional step adds only 0.93–1.14x (g3 C4). On those paths narwhals only removes the non-pandas penalty, ≤1.18x (T4).
2. **narwhals overhead is real:** 7–11x over hand-written native code on the same backend, and narwhals-on-pandas is 9.8–11.5x slower than a plain pandas groupby (x5 verifier). Even routed to polars, the narwhals naive kernel is 3.6–4.2x slower than plain pandas `groupby.last` (DN-R). x7: the narwhals plan costs 2.2–3.0x the hand-written cost for naive.
3. **numpy beats expressions for kernels:** GroupedArray is 8.5–11.2x faster than pandas groupby and faster than polars (x7; r3 verifier refuted "polars beats hub"). g2's numpy inner gives 498–2,735x. So the strongest kernel tier in this design is numpy reached *through* narwhals, not narwhals itself.
4. **Silent backend bugs and coverage holes** in narwhals 2.27.1: pyarrow `over` mis-assignment, `when/then` in `agg` wrong on pandas, duckdb `total_seconds` dropping days (x5 §5, x7). Coverage differs by backend: dask 23–26/34 ops, no `over(order_by)`, no `offset_by("mo")` on pyarrow/dask (x1, x5). Stable namespaces lack `last(order_by)` (x1 verifier, DN-API).
5. **Dual path:** the pandas lane cannot be narwhals (T1/T3b: 1.7–8.3x slower; DN-B1: validation 70–620x slower). Two implementations of `PanelOps` must be kept equivalent (G10 cost unmeasured). This is the same compromise sklearn and pandera made.
6. **narwhals spokes lose fidelity** if used naively: 21–22/30 vs 29/29 for hand-written spokes (x6: freq, Period, RangeIndex, arrow types, npartitions). The schema has to carry everything.
7. **Laziness rarely pays in forecasting workloads:** g3 found that laziness gives nothing in fit+predict, and lazy is 1.1–2.6x slower at C5. Lookback pushdown is slower than collect-all on these sizes (DN-B4). Only selective reads (instances/columns) pay (T6).
8. Group order from narwhals on polars is nondeterministic (r2, x2), and lazy inputs have no row order, so output order differs from sktime's first-appearance order for lazy input.
9. Kernel coverage is small: ≈9/160 closed-form candidates (x7). The 33 iterative and 52 global/DL models gain nothing from expressions.

---

## 7. Open experiments (needed before committing)

1. **Dual-path equivalence cost (G10):** run one conformance suite (g5 matrix + M4 adversarial fixtures) against `PandasMIOps` and `NwPanelOps` and count divergences; measure maintenance LOC.
2. **Kernel conformance at scale:** NwPanelNaive × {polars, pl-lazy, duckdb, ibis, routed-pandas} × strategies × {1k, 10k groups} × M4 fixtures. This includes fixing the x5 verifier gaps (int time on pandas MI, pd3 Period M/Q, n<sp, drift NaN).
3. **Narwhals vs plain-polars vs numpy port of PolynomialTrend** (wrap_critic g11), to set the `panel_kernel:engine` default per kernel class.
4. **Lookback pushdown with memory** in a fresh-process memray harness (M1), on a cold parquet with 10–50 M rows. DN-B4 measured only time on warm in-memory parquet. Its memory value (x6: duckdb 4x lower RSS) is still the open argument for the tag.
5. **FutureGrid join for ragged cutoffs** on duckdb (cross-connection join failure, x5 §6.6) and dask; cost vs per-step `lit`.
6. **`nw_long` inner for darts/neuralforecast/statsforecast wrappers** (G6), with conversion charged.
7. **stable.v2 pin audit** (K6/K7): run the x2 37-op and x1 34-op probes under v2 only, and track narwhals issue #3210 (remember the backend on collect).
8. **End-to-end polars user** at P1: polars input + trusted lane + narwhals validation vs pandas input, to confirm the expected ≤1.0x penalty (currently inferred from T3/T4 + DN-B1).

---

## 8. New measurements in this report (commands + outputs)

Env: `EXP_PY` (pandas 2.3.3, polars 2.0.0, narwhals 2.27.1 via `narwhals.stable.v2`, duckdb 1.5.6, dask 2026.8,
pyarrow 26). Shared 4-core box, load avg ≈0.2–0.5 at start. Timings are min of 2–5 repeats in one process.
Warm cache, in-memory-sized parquet.

```
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/design_narwhals_boundary.py --grid dev
  -> results/design_narwhals_boundary_dev.json   (smoke grid also run -> ..._smoke.json)
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/design_narwhals_routing.py --grid dev
  -> results/design_narwhals_routing_dev.json    (smoke -> ..._smoke.json)
```

**DN-API (stable.v2):** `with_row_index(order_by=)` ok. `Expr.last(order_by=)` → TypeError. `collect(backend=polars|pandas)` ok.
duckdb `from_native` → LazyFrame, default collect → pyarrow Table, `.collect().lazy(backend="duckdb")` → DuckDBPyRelation.
dask → LazyFrame, default collect → pandas DataFrame, `over(order_by)` → NotImplementedError.

**DN-B1 top-level validation** (null keys, contiguity, strictly increasing time within key, duplicates), seconds:

| n×T | sktime check polars_panel | narwhals on polars | ratio | sktime check pd-multiindex | narwhals on pandas-long | ratio |
|---|---|---|---|---|---|---|
| 1000×100 | 0.125–0.130 | 0.0028–0.0031 | 0.021–0.025 | 0.00036–0.00038 | 0.026–0.027 | 70–73 |
| 10000×100 | 1.23–1.56 | 0.0155–0.0163 | 0.010–0.013 | 0.00036–0.00048 | 0.216–0.220 | 451–618 |

(Ranges cover the two dev runs.) Adversarial flags were correct at both sizes: shuffled → `sorted_strict=False, contiguous=False`; one
duplicate row → `bad_order=1`; one null key → `null_keys=1`. Caveat: the stock pd check was repeated on the same object,
so pandas' cached `is_monotonic` may make it cheaper than a first call.

**DN-B2 cutoffs (max+min per key)** at 10k×100: narwhals polars 16 ms, narwhals pandas-long 77 ms, pandas MI
`groupby(level).max` 67 ms. At 1k: 3.8 / 10.2 / 7.7 ms.

**DN-B3 X/y containment** (per-key min/max join): 10k×100 polars eager 25.6 ms, lazy 18.7 ms. Correct X: 0 violations;
X missing the last date: 10000/10000 keys flagged (1000/1000 at 1k).

**DN-B4 lookback pushdown** (last w=10 of 100 points per key, parquet scan), ratio = collect-all / pushdown (>1 means pushdown wins):

| backend | n | to polars: semi-join | to polars: `max().over` window | to pandas: semi-join |
|---|---|---|---|---|
| polars lazy | 1k | 0.33–0.38 | ≈0.42 | 0.57 |
| polars lazy | 10k | 0.26–0.31 | 0.23 | **1.13** |
| duckdb | 1k | 0.47–0.62 | ≈1.04 | 0.66 |
| duckdb | 10k | 0.47–0.64 | 1.36 | **2.10** |
| dask | 1k | 0.09–0.10 | 0.18 | 0.11 |
| dask | 10k | 0.11–0.12 | 0.14 | 0.09 |

Row counts were exact (n·w) on every backend. The native type was kept until collect (LazyFrame, DuckDBPyRelation, dask DataFrame).
Lazy cutoffs without collecting values: 9–15 ms (polars/duckdb), 112–118 ms (dask) at 10k.

**DN-B5 pyarrow rerouting** at 10k×100: `pl.from_arrow` 6.7 ms vs `to_pandas` 17.1 ms. narwhals validation on the rerouted
polars frame took 25.6 ms.

**DN-R kernel routing** (x5 `NwPanelNaive`, pandas MI in/out, fh=1..12). Route B = reset_index → `pl.from_pandas` → kernel
on polars → `to_pandas` → restore time dtype → `set_index`. Routes A and B are equal: index equal, max abs diff 0 (last, last_sp7) and
≤1.1e-12 (mean). Without the dtype restore, the time level drifted datetime64[ns]→[us] (found and fixed).

| n×T | strategy | A narwhals-on-pandas s | B routed to polars s | A/B | C plain pandas groupby.last s | B/C |
|---|---|---|---|---|---|---|
| 1000×100 | last | 0.157 | 0.049 | 3.17 | 0.0118 | 4.19 |
| 1000×100 | last_sp7 | 0.568 | 0.058 | 9.81 | – | – |
| 1000×100 | mean | 0.091 | 0.045 | 2.01 | – | – |
| 10000×100 | last | 1.48 | 0.313 | 4.73 | 0.0864 | 3.62 |
| 10000×100 | last_sp7 | 3.80 | 0.322 | 11.81 | – | – |
| 10000×100 | mean | 0.681 | 0.296 | 2.30 | – | – |

(The first dev run, before the dtype fix, gave A/B 2.4–11.7 for the same cells.)

Caveats: single machine, warm caches, no memory measured in DN-*. The DN-B4 parquet is small (1M rows), so it
favours collect-all. DN-R uses the x5 prototype kernel, which has known gaps (x5 verifier).
