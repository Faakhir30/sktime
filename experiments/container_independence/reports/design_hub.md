# design_hub: one canonical panel (values + indptr + keys + TimeAxis) with per-backend spokes

Architect angle: **HUB / CANONICAL.** Every input is converted **once** into a small, library-neutral, numpy-backed
`PanelFrame`, in the style of Nixtla `GroupedArray` and Darts `TimeSeries`. Validation, cutoff, horizon arithmetic,
iteration and output assembly all run on that one representation. A forecaster gets one of three things from it:
- zero-copy numpy views;
- a lazily built pandas view, for the ~150 unchanged forecasters;
- the whole panel, for batched kernels.

The result goes back to the user's container through the same spoke.

Each number below is either cited as `report-key claim` or comes from the script I ran for this report (§9,
`design_hub_guard_bench`). Anything not measured is labelled *unmeasured/expected*. Nothing under `sktime/` was
modified.

---

## 0. Thesis

The hub is **not** a speed lever for unchanged forecasters. g3 measured it (C4) at 0.93–1.14x, which is noise. It is
the **substrate** that makes the real levers safe and cheap:
- one guarded validation per call, so the trusted per-slice path is sound (g1, g5 rule 4);
- one `TimeAxis` per panel, so fh/cutoff work happens per unique cutoff, not per instance (g3 C3, a3 §0.4);
- zero-copy ingress, which cuts peak RSS 1.4–2.2x (g3 C4);
- a `values + indptr` layout that panel-native kernels consume directly, for 10²–10³x (g2, g3 C5);
- one place to push estimator lookback into lazy plans, which gives 4x lower RSS on duckdb (x6 §3.7).

pandas input stays bit-identical, and pandas becomes one spoke among several.

---

## 1. Why a hub, and what the evidence forces on it

| Evidence | Consequence |
|---|---|
| Container-only changes under unchanged per-instance calls are capped at ~1.05x (x8). Hub iteration on top of C3 adds 0.93–1.14x (g3 C4). | The hub alone must not be sold as a speed feature. It ships together with the trusted executor (g1) and the per-panel TimeAxis (g3 C3), which give 1.55–3.03x cumulatively (g3). |
| Hub ingress vs `prepare_VectorizedDF` incl. conversion: pandas MI 0.06–0.28x of the time, sktime polars 258x faster at 10k and 358x at 50k (final_bench T3). But against an *unchecked* polars→pandas conversion it is only 6–10x faster (x6 verifier R5). | The large ingress ratio comes from removing sktime's slow polars check, which any design can do. The hub's lasting advantage is memory: polars ingress +30 MB vs +451 MB at 50000x100 (final_bench T5b). |
| An unguarded hub is **silently wrong** where stock raises or stays correct: NaN key merges instances, mixed freq gives wrong timestamps, multivariate kernels read column 0 only (x6 verifier R2/R7). | Guards are part of ingress and cannot be turned off. I measured their cost (§9): guarded ingress is 1.4–3.5x unguarded and still 34–496x cheaper than stock `check_is_mtype` with the `is_equally_spaced` metadata. |
| Hand-written spokes: 29/29 strict round trips on pd2 and pd3. narwhals spoke 22/30 and 21/30, capsule spoke 26/30 (x6 §3.8). | Spokes are hand-written for pandas, polars and pyarrow. The Arrow PyCapsule spoke is the generic fallback for duckdb, ibis and other unknown producers. narwhals is optional, never the reference path. |
| Validated pandas slice views are 2.1–2.3x *slower* than today's `items()`. Trusted-freq views are 1.9–2.3x faster but use private `_simple_new` (x6 verifier R6). | Pandas views are built through one internal, version-tested constructor that trusts the `Step` validated at ingress. g3 hit two pd3 bugs in exactly this code (Day vs 24h, datetime unit), so it needs a pd2×pd3 CI matrix. |
| `freq` as a pandas object causes real bugs: CustomBusinessDay holidays dropped (30% wrong timestamps), tz fh differs between pd2 and pd3 (a3 §0.2). A `Step` value object matches sktime on 23/23 freq cases (a3 §0.3). | The TimeAxis is owned by the hub (`time` array plus `Step`), not by an index attribute. pandas offsets remain the fallback engine for exotic offsets. |
| g2: an inner mtype `np_grouped` (the hub plus a3 Step) works with BaseForecaster unchanged apart from registration, `get_cutoff` and `update_data`. 315/315 predict and 233/233 interval comparisons match on pd2 and pd3. | The hub *is* the opt-in inner mtype. Kernel forecasters declare `y_inner_mtype="np_grouped"`, and ingress conversion happens once. |
| Proba: x6 has no lossless proba layout (g4 §2). A long schema plus `ProbaSpec` round-trips stock exactly (g4 §3a). | The hub's output side gets a `ProbaFrame`: the long schema of g4 backed by hub arrays. The egress spoke turns it into the MultiIndex columns for pandas. |

---

## 2. Architecture

### 2.1 Components

```
 native input ──► Spoke.ingest(select=LookbackPlan) ──► Guards ──► PanelFrame ───────────────┐
 (pandas / polars / pl.Lazy / pyarrow / duckdb / dask / capsule)        │                   │
                                                                         ▼                   ▼
                                                 Executor (replaces VectorizedDF)    KernelForecaster
                                                 ├ ViewFactory: numpy view | pandas view   (np_grouped, batched)
                                                 ├ TrustedSlicePath (g1) → unchanged _fit/_predict
                                                 └ TimeAxis: cutoffs, fh grid once per unique cutoff
                                                                         │
                                         OutputFrame / ProbaFrame (hub arrays + keys + time)
                                                                         ▼
                                                   Spoke.egress(EgressSpec) ──► native output
```

| component | owns | replaces today |
|---|---|---|
| `PanelFrame` | `values` (n_rows×C float64, view where possible), `indptr`, `KeyTable`, `TimeAxis`, `aux` (non-float columns), `ColumnSpec`, `EgressSpec` | the `(frame, schema)` tuple in `_vectorize.py`, and the pandas MultiIndex as the internal layout |
| `Spoke` registry | ingress, egress and pushdown per backend | `convert_to` chains for Panel/Hier, `prepare_VectorizedDF`, `reconstruct` |
| `Guards` | NaN/null keys, strictly increasing time within each instance (catches unsorted, duplicate and NaT), all-instance `Step`, X/y (key,time) containment | per-slice `check_is_scitype`, `_check_X_y` re-entry, `_enforce_index_freq` |
| `TimeAxis` | `time` array, `Step` (a3), per-instance cutoffs, `fh_grid(fh)` | `get_cutoff`, per-instance `fh.to_absolute`, `pd.infer_freq` |
| `Executor` | positional iteration and the trusted per-slice path (g1), one call per batch for kernels | `BaseForecaster._vectorize` + `VectorizedDF.items` |
| `ViewFactory` | `SeriesView` (numpy), `PandasView` (Series/DataFrame with trusted freq), `PanelPandasView` (whole-panel MI for the 29 multiindex-native forecasters) | per-slice `convert_to` |
| `OutputFrame` / `ProbaFrame` | predictions as hub arrays, plus g4 `ProbaSpec` for interval/quantile/var | `pd.concat(keys=...)`, per-instance `_get_columns` |
| `EstimatorGrid` | fitted per-instance clones, keyed by `KeyTable` position (a1 §5) | the `forecasters_` pandas pivot |

### 2.2 Protocols (signatures)

```python
# sktime/datatypes/_hub/_frame.py  (new; ~450 LOC framework per g2 + 67 core per x6)
@dataclass(frozen=True)
class Step:                        # a3 protos/a3_timeindex.py
    kind: Literal["int","fixed","calday","month","bday","bmonth","halfmonth","pandas"]
    n: int = 1; nanos: int = 0; anchor: str | None = None
    weekmask: str | None = None; holidays: tuple = (); tz: str | None = None
    pandas_offset: object | None = None   # fallback only (FY5253, Easter, BusinessHour...)

@dataclass(frozen=True)
class TimeAxis:
    time: np.ndarray               # datetime64[unit] (UTC-naive + tz in Step) | int64 (int or period ordinals)
    kind: Literal["datetime","period","int","range"]
    step: Step | None              # None => irregular; time arithmetic raises, iteration still works
    unit: str; period_freq: str | None; name: object
    def cutoffs(self, indptr, reverse_order=False) -> np.ndarray          # per instance, O(U)
    def cutoff(self, indptr, reverse_order=False) -> np.generic           # global max/min
    def fh_grid(self, fh_rel: np.ndarray, cutoffs: np.ndarray) -> np.ndarray   # (U, H); memo per unique cutoff
    def steps_between(self, a, b) -> np.ndarray
    def to_pandas_index(self, t: np.ndarray, *, trusted=True) -> pd.Index  # small objects only

@dataclass(frozen=True)
class KeyTable:
    names: tuple                   # level names (may contain None); "__total" kept verbatim
    values: tuple[np.ndarray, ...] # one array per level, length U
    dtypes: tuple                  # native dtypes for egress (categorical, string[pyarrow], Enum…)

@dataclass
class PanelFrame:
    values: np.ndarray             # (n_rows, C) float64; read-only when it aliases user memory
    indptr: np.ndarray             # int64 (U+1,)
    keys: KeyTable
    time: TimeAxis
    columns: ColumnSpec            # names, original dtypes, null/NaN masks (see §7 E4)
    aux: dict[str, np.ndarray]     # non-float columns, row aligned; kernels never see them
    scitype: Literal["Series","Panel","Hierarchical"]
    egress: EgressSpec             # source lib, layout, schema, row permutation, lazy/npartitions, Series-ness
    def view(self, i) -> "SeriesView"
    def views(self) -> Iterator["SeriesView"]
    def select(self, rows=None, instances=None, cols=None) -> "PanelFrame"   # positional, zero-copy when contiguous
    def with_values(self, indptr, time, values, columns=None) -> "PanelFrame" # same keys, new rows (outputs)

# sktime/datatypes/_hub/_spokes.py
class Spoke(Protocol):
    name: str                                         # "pandas", "polars", "pyarrow", "duckdb", "dask", "capsule"
    def can_ingest(self, obj) -> bool                 # explicit type check, no duck typing (g5 rule 5)
    def ingest(self, obj, *, keys=None, time=None, plan: "LookbackPlan|None" = None) -> PanelFrame
    def egress(self, pf: PanelFrame | "OutputFrame", spec: EgressSpec) -> object
    def egress_proba(self, pr: "ProbaFrame", spec: EgressSpec, fmt: Literal["pandas_mi","long"]) -> object
def register_spoke(spoke: Spoke, *, priority: int = 0) -> None          # public, replaces lru_cache surgery (g2 §1.1)
def to_hub(obj, **kw) -> PanelFrame                                       # dispatch + Guards
def from_hub(out, spec: EgressSpec) -> object

@dataclass(frozen=True)
class LookbackPlan:                                   # pushed into lazy plans (x6 §3.7)
    tail: int | None                                  # max rows per instance the estimator reads (None = all)
    instances: object | None = None                   # optional instance filter
    columns: tuple | None = None

# sktime/forecasting/base/_hub_exec.py
class Executor:
    def fit(self, est, pf_y: PanelFrame, pf_X: PanelFrame | None, fh) -> None
    def predict(self, est, fh, pf_X=None) -> "OutputFrame"
    def predict_proba_like(self, est, method: str, fh, pf_X=None, **kw) -> "ProbaFrame"
    def update(self, est, pf_y_new, pf_X_new=None, update_params=True) -> None

class ViewFactory:
    def numpy(self, pf, i) -> SeriesView
    def pandas(self, pf, i, *, mtype: Literal["pd.Series","pd.DataFrame"]) -> pd.Series | pd.DataFrame
    def pandas_panel(self, pf, *, mtype: Literal["pd-multiindex","pd_multiindex_hier"]) -> pd.DataFrame

# kernel opt-in (g2): a forecaster that consumes the hub directly
#   tags: y_inner_mtype = ["np_grouped_series","np_grouped","np_grouped_hier"]
#         "fitted_state": "batched"   (no per-instance clones, see §3)
class _HubKernelMixin:
    def _fit(self, y: PanelFrame, X: PanelFrame | None, fh) -> Self
    def _predict(self, fh, X=None) -> "OutputFrame"
    def _predict_interval(self, fh, X, coverage) -> "ProbaFrame"         # optional; else fallback (g4: x7 lost the gain this way)
```

`OutputFrame` is a `PanelFrame` whose `indptr` is `arange(U+1)*H` and whose `time` is the `fh_grid`. `ProbaFrame` is the
g4 long schema stored as hub arrays: `values` (n_rows × V × L × {lower, upper} or {value}) plus `ProbaSpec`.

### 2.3 Where each pandas touchpoint category goes (a1: 229 touchpoints)

| a1 category | today | in the hub design |
|---|---|---|
| **Container ops** (42 touchpoints: group/iterate, slice/select, concat/reconstruct, pivot) | groupby on the MI level, `.loc`, `pd.concat(keys=)` | `indptr` slicing (0.6–5 µs/slice vs 115–274 µs, final_bench T1) and numpy concat at egress (reconstruct 0.05–0.15x, T2). No pandas. |
| **Time / freq / cutoff** (78 touchpoints, the largest) | `.freq`, `infer_freq`, per-instance `get_cutoff`, `fh.to_absolute` | `TimeAxis` with a3 `Step` and numpy ops; pandas offsets only as the `Step(kind="pandas")` fallback. Cutoffs are one `time[indptr[1:]-1]`. The fh grid is computed once per unique cutoff (g3 C3: 999/1000 memo hits). |
| **Metadata** (dispatch, names, dtypes, `has_nans`, duck typing) | `hasattr(obj,"index"/"X")`, `check_is_scitype` per slice | `Spoke.can_ingest`, `KeyTable`, `ColumnSpec` and `EgressSpec`, filled once at ingress. Removes the stale duck-typing bugs (a1 §6; g5 rule 5). |
| **Output construction** (row index, proba columns, `forecasters_` pivot) | `pd.MultiIndex` per instance, `_get_columns` 0.7–0.8 ms/instance (g4) | Built once at egress from `KeyTable`, `fh_grid` and `ProbaSpec`. pandas MI egress from long: 14–85 ms at 10k vs 205 ms for today's final concat and 7.9 s of per-instance `_get_columns` (g4 §4). |
| **Public-API value types** (44 touchpoints / 4 contracts) | pandas `forecasters_`, `cutoff`, fh, MI proba columns, `"forecasters.loc[...]"` keys | Small pandas objects are kept and materialized lazily from the hub. Large outputs follow the input container. Deprecation path in §3. |

### 2.4 Data flow per input container

| input | ingress | what the forecaster sees | egress |
|---|---|---|---|
| pandas Series/DataFrame/MI | hand-written spoke reads `MultiIndex.codes` (no `reset_index`, names kept, x6 §3.9). Values are a view of a single float block; time is one 8 B/row copy. | unchanged forecasters: `PandasView` with trusted freq (C4 equal to stock 24/24, g3). Kernels: `PanelFrame`. pd-multiindex natives: `PanelPandasView`, which is the original object when unchanged (identity, zero cost). | pandas MI built once from codes |
| polars eager (natural layout or `__index__`) | `rle_id` run starts; values and time zero-copy for 1 column. A multi-column parquet frame needs one 2D copy (x6 verifier R4: 4.4x lower RSS, not 22x). | same as above; no pandas MultiIndex is ever built (g3 C4) | polars DataFrame with the same schema |
| polars LazyFrame | `LookbackPlan` applied in the plan (`group_by.tail(w)`), then **one** `collect()` into polars | same | eager polars (outputs are O(U·H)); `output:lazy=True` re-wraps with `.lazy()` |
| pyarrow Table / dataset | arrow spoke (dictionary keys → codes); multi-chunk tables are combined | same | pyarrow Table with the schema restored |
| duckdb relation, ibis, other `__arrow_c_stream__` producers | `LookbackPlan` → SQL `QUALIFY row_number() <= w`, then the capsule stream → arrow spoke | same | Arrow-backed relation. It is **not** bound to the user's connection (x6 verifier R9), which is documented. |
| dask DataFrame | phase 3: `compute()` once (x6: 1.5x faster than stock ingress). Phase 5: partition-and-delegate by key, with the hub per partition (unmeasured). | same | dask with `npartitions` restored from `EgressSpec` |

The rule for lazy input: **materialize exactly once, after pushdown**. The hub never iterates a lazy frame.
Per-instance lazy evaluation is 3–77x slower (x4, via wrap_critic K10), and stay-lazy without pushdown gives nothing
(g3: lazy within ±15% of eager at C0–C4).

---

## 3. Public API

| contract | today | this design, release N (no break) | N+2 |
|---|---|---|---|
| `predict` return type | follows input for pandas and polars | follows input for every registered spoke, including pyarrow and duckdb. Fixes the stock polars `y`→`"0"` rename (g5) because names live in `ColumnSpec`. | — |
| `cutoff` | 1-element pandas Index (freq sometimes `None` on MI, g2 §2) | still a 1-element pandas Index, built from `TimeAxis`, with freq from `Step`. **New** `cutoffs_` returns a native frame (keys, cutoff) in the input backend, computed in O(U). | — |
| `fh` | `ForecastingHorizon` (pandas) | unchanged. Internally it is turned into an int array plus a `Step` once per call. `fh.to_absolute` returns pandas for the outer call only. | `ForecastingHorizon` gains a `Step`-based freq; the pandas offset becomes derived |
| `predict_interval` / `quantiles` | pandas MI columns even for polars input (a1 probe) | pandas input: unchanged MI. Non-pandas input: config `output:proba_format="pandas"` by default, plus a one-time `FutureWarning`; `"native"` gives the g4 long schema in the input backend. `get_proba_spec()` added. | the default becomes `"native"` **for non-pandas input only** (g4 §6) |
| `predict_var` | columns = variables; NaiveForecaster names it column 0 (g4 stock bug) | predict layout, bug fixed. `cov=True` raises for non-pandas. | — |
| `predict_proba` | skpro distribution; vectorized path broken (`_yvec`, g4) | still skpro, but built from one long parameter table (3.2 ms vs 826 ms at 1k, g4 §5). `BaseProbaForecaster` vectorization is folded into the Executor. | — |
| `forecasters_` | `pd.DataFrame` of clones | `EstimatorGrid`, with a DataFrame-compatible `.loc/.iloc/.values/.index` shim that emits no warning. For `fitted_state="batched"` kernels: `forecasters_` builds per-instance **`FittedInstanceView`** objects on access (params sliced from arrays, with `predict` delegating to the kernel restricted to one instance), plus a `FutureWarning` that it is O(U). | `forecasters_` returns `EstimatorGrid`; DataFrame access goes through `.to_pandas()` (a1: 0 behavioural failures in 204 tests; 4/4 metric configs need porting first) |
| `get_fitted_params` keys | `"forecasters.loc[...]"` strings | unchanged for vectorized forecasters. Batched kernels return arrays with a leading instance axis plus `"instance_keys"`. Add `get_fitted_params(style="long")` → native frame (keys, param, value). | `style="long"` becomes the documented form for panels |

The only places where pandas stays in the **public** API indefinitely are small O(1)/O(H) objects (`cutoff`,
`ForecastingHorizon`) and pandas-input outputs. This follows g5 rule 1: these are metadata, and the contract says so
explicitly.

---

## 4. Estimator migration (~160 forecasters)

Nobody **must** change anything. The default inner family is "pandas view per slice", which is what stock does today,
and it is output-identical (g3 C4 24/24; g1 7710 tests, 0 delta).

| group (a2 / g1 census) | n | what happens | who changes what |
|---|---|---|---|
| Vectorized via the Executor, eligible for the trusted path | 123 of 124 vectorized (g1) | `PandasView` per slice, trusted state set from the hub (g1 invariants 1–7) | nobody. ChronosForecaster falls back (it overrides `predict`). Opt-out is the class tag `capability:trusted_vectorize=False`. |
| pd-multiindex-native (global, DL, reducers in global mode) | 29 (a2) | `PanelPandasView`: identity for pandas input, otherwise one egress to a pandas MI (0.06–0.74x of the stock concat, T2) | nobody. 8 classes whose libraries accept polars natively (darts, neuralforecast; a2 verifier) may declare `y_inner_mtype="polars_long"` and receive a polars egress (gain unmeasured, G6). |
| V1 boundary-only (core already numpy) | 63 (a2) | still pandas views by default. Opt-in `np_grouped` gives numpy views plus the `TimeAxis`, so per-series conversion goes away. | optional. The ceiling for per-series third-party fits is 6–9x (g2 reducer); statsmodels adapters are expected to be similar (unmeasured). |
| V2 glue rewrite (label ops in glue) | 56 (a2) | pandas views. Porting means replacing `.loc`/`reindex` with positional ops against `SeriesView`. Fixing `_sktime.py:151` alone unblocks 7 classes (a2). | optional, mechanical |
| V3 core in pandas (Naive, reducers, …) | 13 (a2) | pandas views. Naive is the flagship kernel port (g2: 116 LOC vs 358). | maintainers, for the 5–10 baseline families (x7; strict closed-form 6–8, x7 verifier) |
| V4 third-party pandas-bound | 28 (a2) | pandas views, or `PanelPandasView` for the 29 global ones. Torch models (7 classes) could take numpy windows from the hub (unmeasured). | nobody. This cannot be fixed on the sktime side for 21 classes. |
| Composites / tuners | 61 (a2) | composites run on the hub when their child pipeline does. `evaluate` and splitters run on hub positions (g5 rule 6) instead of converting to pandas at entry (g5 S9). | framework: TTF, GridCV, Reconciler and `evaluate` call `to_hub` once and pass `PanelFrame` down (phase 3) |

New tags:
- `y_inner_mtype` may list `np_grouped*` (kernel opt-in).
- `fitted_state: "per_instance" | "batched"`.
- `capability:lookback` (int | `"sp"` | `"window_length"` | None), which feeds `LookbackPlan`.
- `capability:trusted_vectorize` (default True).
- `proba_inner_format: "pandas" | "long"`.

---

## 5. Phased roadmap (VectorizedDF → BaseForecaster boundary → estimators)

| phase | scope | expected impact (cited) | risk |
|---|---|---|---|
| **P0. Decision-independent fixes** (1 release) | vectorized polars monotonic check with a null guard; pass the known mtype (`convert(from_type)`); `_HashIndex.__eq__` so the fh memo hits; g5 stock bugs (1–2-row polars update, Panel `predict_residuals`, `y`→`"0"` rename, disjoint-X no-op); pd3 caller mutation | C1 1.05–1.34x e2e (g3); fh memo 1.05–1.53x on top of the trusted path (g3 C3); polars prepare 0.24x with a known mtype (final_bench T3) | low. A null guard is needed: the vectorized check is not equivalent with nulls (x9 verifier, 58/75). |
| **P1. Hub core + spokes behind VectorizedDF** | `PanelFrame`, `TimeAxis`/`Step`, Guards, hand-written pandas/polars/pyarrow spokes plus a capsule fallback; `register_spoke`/`register_mtype` public API; `prepare_VectorizedDF`/`items`/`reconstruct` reimplemented on the hub, with forecasters still called through the public per-slice path | time ≈1.0x e2e (container-only cap, x8; C4 0.93–1.14, g3). RSS 1.4–2.2x lower (g3 C4: 161→54 MB polars 5000x100). Guarded ingress 3.7–17.8 ms pandas / 3.6–29.5 ms polars at 1k–50k series (§9). Polars ingress 2.84 s → 11 ms at 10k (final_bench T3). Reconstruct 0.05–0.15x (T2). | medium. Fidelity regressions: 7–8/33 adversarial fixtures failed in x6 (NaN key, NaN/null, int64>2^53) → ColumnSpec masks and int passthrough (§7 E4). pd3 time-unit/offset fragility (g3 found 2 bugs). Gate: x6 29/29 + x6v 33 fixtures + g5 240-cell matrix on pd2 and pd3. |
| **P2. BaseForecaster boundary on the hub** | public methods call `to_hub` once; validation once on the hub (g5 rules 3–4, incl. X/y (key,time) anti-join); trusted Executor (g1) builds per-slice state from hub metadata; cutoff/fh via `TimeAxis` once per unique cutoff; `update` via hub append (g2 `npg_append`); proba via `ProbaFrame` + spec egress; `EstimatorGrid` behind `forecasters_` | cumulative **1.55–3.03x** fit+predict for unchanged forecasters (g3 C3); 1.59–1.91x Naive and 2.44–3.31x Poly with the hub (g3 C4). Full fit/predict/interval/update/predict workload 1.07–1.72x (g1). Proba egress: per-instance `_get_columns` (4% of stock interval time at 1k, g4) and the 205 ms final concat → 14–85 ms at 10k. | medium-high. g1 copies stock logic (`_check_fh`, `update_data`), so P2 must refactor the public methods to share it, or drift follows. `update_predict`/`predict_proba` keep falling back until ported (g1 inv. 6). |
| **P3. Composites, evaluate, nested transformers** | TTF/GridCV/Reconciler/`evaluate` pass `PanelFrame` down; native window splitter on hub positions; trusted entry for nested `BaseTransformer` (Lag, Detrender) | removes the remaining predict-side re-entry: after C4, nested `Lag` is 63% of Naive time (g3 §3.4), so the ceiling is ~1.6–2.7x on top of P2 for Naive-like forecasters (*unmeasured/expected*, derived from that share). Without P3 composites get no container value (g5 rule 8). | high. Transformer base has its own vectorization; a second trusted path doubles the invariant surface. |
| **P4. Kernel opt-in (`np_grouped`)** | port the 5–10 baseline families (Naive incl. intervals, PolynomialTrend, Trend, Croston, NaiveVariance, seasonal naive…); batched `_predict_interval` | **10²–10³x** for ported ones: g2 498–2,735x pandas input, 498–1,486x pyarrow at 1000x100; g3 C5 105–4,560x; native Naive interval ~1,100x (g4). Per-series third-party fits: 6–9x (g2 reducer). Port cost 40–120 LOC/forecaster (g2). | medium. Ports must replicate stock quirks to stay equal (sklearn `cond=tol`, drift slope; g2 §7). `forecasters_` parity needs `FittedInstanceView`. Polars input pays the stock polars check until P0 lands (g2: 99% of residual time). |
| **P5. Lazy and distributed** | `capability:lookback` → `LookbackPlan` pushdown for polars-lazy/duckdb/pyarrow-dataset; dask partition-by-key delegate | selective reads: polars lazy 875x, duckdb 360x vs eager sktime read+convert (final_bench T6). Lookback pushdown on duckdb: 4.0x lower peak, 2.2x faster (x6 verifier R8). Pushdown over selection-aware eager: only 1.9x (x9 verifier). Dask partition delegate: 1.2–1.5x like-for-like (r4 verifier); break-even unmeasured (G7). | medium. Polars `group_by.tail` gave no RSS cut on polars 2.0.0 (x6 §3.7). The share of forecasters that can declare a finite lookback is unmeasured (G14). |
| **P6. Contract flips (N+2)** | proba `"native"` default for non-pandas input; `forecasters_` → `EstimatorGrid`; `get_fitted_params(style="long")` documented | correctness and consistency, no speed. | 4/4 hierarchical metric configs break without porting (a1 §5); 31 leaf files build the proba MI (g4). |

Summary of expected totals (fit+predict, 1k–5k series, against stock):

| configuration | gain |
|---|---|
| unchanged forecasters after P2 | 1.6–3.3x (g3 C4) |
| plus nested-transformer trust (P3) | *expected* up to ~2–4x for Naive-like forecasters (unmeasured) |
| ported kernels (P4) | 10²–10³x |
| peak memory | 1.4–2.2x lower from P1 on (g3) |
| pushdown-capable lazy reads (P5) | 2–4x lower memory |

---

## 6. Evidence for

1. g3 C4: hub ingress, positional iteration and trusted pandas views to unchanged forecasters are output- and state-identical to stock in 24/24 rows on pd2, and at 100x100 on pd3. Naive 1.59–1.91x, Poly 2.44–3.31x end to end; peak RSS 1.4–2.2x lower.
2. g2: one container class serves as the inner mtype for 3 scitypes and 3 input backends. 315/315 predict and 233/233 interval comparisons; conversions 2 instead of ~804; mtype checks 36 instead of 7,484–12,084; no pandas object built for polars or pyarrow input.
3. g3 C5 / g2: kernels on the hub layout give 105–4,560x. This is the only 10²–10³x lever, and it needs exactly the `values + indptr + Step` layout.
4. x6 §3.8: hand-written spokes round-trip 29/29 fixtures on pd2 and pd3, at ~100–140 LOC per library, confined to 2 functions per library.
5. final_bench T1/T2/T3/T5b: slice view 54–236x cheaper than `items`; reconstruct 0.05–0.15x; polars ingress 258–358x faster with +30 MB vs +451 MB.
6. a3: a numpy `Step` reproduces sktime on 23/23 frequency cases (0/3,600 mismatches) and fixes two pandas-offset bugs by construction. Per-panel time work is ≤20 ms at 10k series vs 20–36 s per-instance.
7. g4: the long schema plus `ProbaSpec` is lossless against stock. Egress to the pandas MI is 0.07–0.42x of today's final concat, and a native Naive interval kernel runs at ~1,100x.
8. x6 §3.7: only a design that knows the estimator's lookback before materializing can push it down. duckdb peak 158.7 → 39.7 MB.
9. §9 (this report): three silent-corruption faults of the unguarded hub (NaN key, mixed freq, duplicate time) are all caught by guards that cost 1.4–3.5x of unguarded ingress. Even guarded, ingress is still 34–496x cheaper than stock `check_is_mtype` with `is_equally_spaced`. Stock's bare check *accepts* all three faulty frames.

## 7. Evidence against (honest)

- **E1. The hub buys no time for unchanged forecasters.** C4 adds 0.93–1.14x (g3), and container-only changes are capped at ~1.05x (x8). Every speed number for unchanged forecasters in §5 comes from g1/g3 levers that an adapter design gets as well (see design_adapter). The hub's own value is memory, the kernel substrate and pushdown.
- **E2. On pandas input, guarded hub ingress costs more than stock's bare check.** Measured here: 3.7–8.4 ms vs 0.37 ms for `check_is_mtype(y, "pd-multiindex")` without metadata, i.e. **10–25x more** (this check on a reused index object is probably helped by pandas' cached monotonic flags). It only wins against the metadata-carrying check: 173 ms–8.8 s. At 11–13 ms per instance of forecaster time (x8) this is <0.1% e2e, but it is a real regression for "pandas in, cheap check".
- **E3. Pandas views need private constructors.** The safe public `DatetimeIndex(freq=…)` view is 2.1–2.3x *slower* than today's `items` (x6 verifier R6). The fast path uses `_simple_new`. g3 hit two pandas-3 bugs (Day vs 24h, ns vs us) in this code. This is a standing maintenance cost across pandas versions.
- **E4. The float64 hub loses information.** NaN vs null distinction lost, int64 > 2^53 rounded, MultiIndex columns flattened, timedelta index crashes on pd3 (x6 verifier R2: 25/33 and 26/33 adversarial passes). Fixing it needs `ColumnSpec` null masks and non-float passthrough in `aux`, at an unmeasured cost. Kernels see only floats.
- **E5. Zero-copy is fixture-dependent.** Parquet-read multi-column polars and multi-chunk arrow are copied (polars 4-column RSS gain 4.4x, not 22x; x6 verifier R4). pandas time is always copied (8 B/row). On pd2 the egress output aliases user input, so a mutation propagates (x6 verifier R3).
- **E6. Unsorted and time-major input is expensive.** Measured here: guarded hub ingress on time-major input is 2.7–6.7x (pandas) and 5.0–7.6x (polars) slower than on instance-sorted input. x6 verifier: hub 7–16x slower on time-major, though stock degrades 23–27x on the same input. SQL `ORDER BY time` is common.
- **E7. A second internal data model.** Spokes (449 LOC) plus framework (443 LOC, g2) plus time ops (130 LOC). Each new backend needs a hand-written spoke for full fidelity; the generic narwhals and capsule styles reach only 22/30 and 26/30 (x6). An adapter design reuses pandas as the internal model and so avoids keeping two in sync.
- **E8. Public-API friction for kernels.** Batched kernels have no per-instance clones, so `forecasters_` and `get_fitted_params` keys change shape (g2 §7, g3 C5). The `FittedInstanceView` shim is unbuilt and unmeasured. g3 C5 also drops the pandas time-level `freq` on egress.
- **E9. Lazy and distributed value is narrow.** Laziness pays nothing without pushdown (g3: lazy within ±15% of eager at C0–C4, 1.1–2.6x slower at C5). The hub is single-process; dask is `compute()` (x6). Egress to duckdb returns a relation that is not on the user's connection, and dask `npartitions` is lost (x6 verifier R9).
- **E10. Calendar and tz edge cases still need pandas.** The a3 `Step` failed SME, tz W, numpy BMS/BME and int step 5 in verification (wrap_critic a3.1). numpy has no time zones (a3 §0.1). So `Step(kind="pandas")` stays, and with it the per-panel pandas offset path.
- **E11. The registry is closed today.** Extending mtypes from outside means mutating private lru_caches, and a later `cache_clear()` silently unregisters them (g2 §1.1). A public `register_mtype` is a hard prerequisite for P1.
- **E12. The real residual cost is out of the hub's reach.** After C4, 63% of Naive time is the nested `Lag` transformer and 63% of Poly time is the sklearn fit (g3 §3.4). Only P3 (trusted transformers) or P4 (kernels) touch it, and P4 covers only 5–10 families.

## 8. Open experiments

1. `design_hub_g12_full`: wire the §9 guards into the g3 C4 path, add the X/y (key,time) anti-join and `ColumnSpec` null masks, then re-run the g5 240-cell matrix and the x6 verifier's 33 fixtures. Target: 0 silent mismatches and the time cost per guard.
2. `design_hub_fitted_view`: prototype `FittedInstanceView` for G2Naive/G2Poly. Measure the cost of `forecasters_` access at 1k/10k and pass the forecasting-base tests that index `forecasters_`.
3. `design_hub_nested_trust`: a trusted `BaseTransformer` entry for Lag/Detrender inside NaiveForecaster on the C4 path. This quantifies P3, which is currently an expectation derived from the 63% share.
4. `design_hub_unsorted`: replace the full permutation with an arg-sort by (key, time) on codes only. Time it on time-major and shuffled 50k-series inputs against stock.
5. `design_hub_pd3_view`: a pd2×pd3 matrix of the trusted pandas view constructor across Step kinds (fixed, month, bday, period, tz) against `items()`. This is the fragility gate for E3.
6. `design_hub_lookback_census`: count the forecasters that can declare a finite `capability:lookback` (G14). Then measure the eager-select vs lazy-pushdown memory on a 2M-row parquet through the full public API.
7. `design_hub_update_proba`: timing of `update` and `predict_interval` through the hub Executor (g3 covered only fit+predict).

## 9. Measurement performed for this report

Script: `experiments/container_independence/scripts/design_hub_guard_bench.py`, with a follow-up stock-without-metadata
control in the scratchpad (`design_hub_s0.py`). Outputs: `results/design_hub_guard_bench.json` and
`results/design_hub_guard_bench_s0.json`.

```
cd /home/user/sktime
PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
  $EXP_PY -W ignore experiments/container_independence/scripts/design_hub_guard_bench.py
```

**Definitions:**
- U = x6 `from_pandas`/`from_polars`.
- G = U + guards: NaN/null key check, strictly increasing time within instance (masked `np.diff` over `indptr`), and all-instance g2 `infer_step` (which raises on irregular steps).
- S = stock `check_is_mtype(..., return_metadata=["is_equally_spaced"])`.
- S0 = stock `check_is_mtype` without metadata.
- GT = G on time-major input.

All timings are min of 3–5 repeats, two interleaved rounds, in one process on the shared 4-core machine. Inputs are
`make_panel_fast`; polars input is `convert_to(y, "polars_panel")`.

**Adversarial fixtures** (4×10 panel):

| fixture | unguarded hub | guarded hub | stock `check_is_mtype` |
|---|---|---|---|
| NaN key | accepted (silent) | raises "NaN in instance key" | valid=True |
| mixed freq | accepted (silent) | raises "no regular step" | valid=True |
| duplicate time | accepted (silent) | raises "not strictly increasing" | valid=True |

**Timing, ms** (ratios in parentheses):

| size | pd U | pd G (G/U) | pd S (S/G) | pd GT (GT/G) | pl U | pl G (G/U) | pl S (S/G) | pl GT (GT/G) |
|---|---|---|---|---|---|---|---|---|
| 1000x100 | 1.06 | 3.69 (3.49) | 173 (47) | 9.9 (2.69) | 2.56 | 3.56 (1.39) | 121 (34) | 17.8 (5.00) |
| 10000x50 | 3.07 | 8.41 (2.74) | 1,929 (229) | 56.2 (6.69) | 7.16 | 13.0 (1.81) | 972 (75) | 98.8 (7.60) |
| 50000x20 | 7.08 | 17.8 (2.51) | 8,808 (496) | 89.3 (5.03) | 17.9 | 29.5 (1.65) | 3,913 (133) | 176.5 (5.98) |

**Control: stock without metadata** (S0, separate run):

| size | pd S0 | pd G | pl S0 | pl G |
|---|---|---|---|---|
| 1000x100 | 0.38 | 3.67 | 145 | 5.02 |
| 10000x50 | 0.37 | 8.39 | 1,009 | 14.3 |

- On pandas, the guarded hub is 10–25x *more* expensive than stock's bare check.
- On polars, it is 29–71x cheaper.

**Caveats:**
- Only fixed-step daily data was used.
- The guard set does not include the X/y containment join or `ColumnSpec` masks.
- The pandas S0 figure may benefit from cached index monotonicity flags on the reused object.
