# g4_proba: probabilistic outputs and the second vectorization path (critic G4)

**Headline:** proba path: 2 of 4 prototypes work, and both only because they fall back to
pandas. x4 works because per-instance outputs are pandas and `reconstruct` dispatches on the
outputs. x7 works by materializing the stock per-instance loop. x5 and x6 have no proba
contract: they return unparseable or flattened column names. A backend-neutral long
schema round-trips stock Naive and Theta outputs exactly. Converting long to today's
MultiIndex costs 14–85 ms at 10k series (0.07–0.42x of today's final `pd.concat` alone).
A panel-native Naive interval kernel in polars matches stock to 1.4e-14 and runs in 16 ms
against stock's 18.1 s at 1k series. Measured stock bugs: the second vectorization path
(`BaseProbaForecaster`) is broken on this branch, and `predict_var` loses variable names.

Environment: own venv `scratchpad/venvs/g4_proba`, built because EXP_PY has neither
statsmodels nor skpro. It pins the same versions as EXP_PY (pandas 2.3.3, polars 2.0.0,
narwhals 2.27.1, numpy 2.2.6, sklearn 1.9.1, scipy 1.18.1, pyarrow 26) and adds
statsmodels 0.15.0 and skpro 2.16.0. sktime is the WT checkout. 4 cores. Timings are in-process
min-of-R. "interleaved" means round-robin in one process (`interleaved()` in
`scripts/g4_proba_bench.py`).

Files:
- `protos/g4_proba_schema.py`: long schema, `ProbaSpec`, pandas↔long, guard, canonical sort,
  interval↔quantile relabel, flat-column variant, polars-native Naive interval kernel
- `scripts/g4_proto_probe.py` → `results/g4_proto_probe_n200.json` (prototype matrix)
- `scripts/g4_proba_bench.py` → `results/g4_proba_bench_ABC.json`, `.log` (correctness + cost + distributions)

Commands (from WT, `G4_PY=$SCRATCH/venvs/g4_proba/bin/python`,
`PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos`):
```
$G4_PY -W ignore experiments/container_independence/scripts/g4_proto_probe.py --n 200
$G4_PY -W ignore experiments/container_independence/scripts/g4_proba_bench.py \
    --ncorr 1000 --ntheta 200 --sizes 1000,10000 --rounds 5 --ndist 1000 --parts ABC
```

---

## 1. Pandas touchpoints (file:line)

Note: the task names `sktime/forecasting/base/_proba/_mixin.py`. That path does not exist.
The mixin is `sktime/base/_proba/_mixin.py`, and `BaseForecaster` inherits from it
(`_base.py:55,86`).

### 1a. Path 1: `BaseForecaster` predict_interval / predict_quantiles / predict_var

| site | what | pandas coupling |
|---|---|---|
| `_base.py:722-787` `predict_quantiles`, `817-876` `predict_interval`, `912-957` `predict_var` | check fh/X, then `_predict_*` or `_vectorize("predict_*")` | **No output conversion step**, unlike `predict`. The inner pandas frame is returned as is, so polars input gives pandas output (re-confirmed: `stock_pl` in probe, as in a1.4) |
| `_base.py:2398-2402` | `PREDICT_METHODS` includes the 3 proba methods | same per-instance loop as predict (public re-entry per slice) |
| `_base.py:2448-2467` `_vectorize` | `vectorize_est(return_type="list")` → `_y_schema.reconstruct(..., col_multiindex=...)`; `y_pred.columns.droplevel(1)` (2466) | column MultiIndex manipulation for multivariate (column-vectorized) y |
| `_datatypes/_vectorize.py:538` | `pd.concat(df_list, keys=row_ix)` | row MultiIndex built from the instance keys. Column MultiIndex of each part is carried through |
| `_vectorize.py:545, 557, 559` | `pd.concat(axis=1, keys=col_keys)` for column vectorization | adds a top column level |
| `_vectorize.py:564-565` | `flatten_multiindex` | would destroy proba columns if `force_flat` triggered |
| `_vectorize.py:567-568` | `X_mi_index.set_names(...)` | index level names |
| `_base.py:2764-2805` `_get_columns` | `pd.MultiIndex.from_product([vars, coverage, ["lower","upper"]])` (2797), `[vars, alpha]` (2805) | **the** layout definition (a1 C3). Called once per instance |
| `_base.py:2748-2762` `_get_varnames` | `_y_metadata["feature_names"]` | variable labels (any hashable) |

### 1b. `_PredictProbaMixin` defaults (`sktime/base/_proba/_mixin.py`)

| line | default | pandas op |
|---|---|---|
| 78-93 | `_predict_interval` → `_predict_quantiles(alpha=0.5±c/2)`, then `pred_int.columns = _get_columns(...)` | column relabel by MultiIndex |
| 142-171 | `_predict_quantiles` from `_predict_interval` | `np.round(abs(1-2a),12)` for coverage keys (149), `pred_int.iloc[:, col_ix]` (166), `.columns =` (171) |
| 173-175 | `_predict_quantiles` from proba | `self.predict_proba(**kwargs).quantile(alpha)`: **public** re-entry, skpro frame |
| 237-242 | `_predict_var` from proba | `predict_proba().var()` |
| 247-263 | `_predict_var` from interval | `pred_int.columns.get_level_values(0).unique()`, `pred_int[i]`, `.iloc[:,1]-.iloc[:,0]`, `pd.DataFrame(dict)` |
| 312-331 | `_predict_proba` from var | `self._predict_var`, **public** `self.predict`, `convert_to(pred_mean, pd types)`, `Normal(mu=DataFrame, sigma=..., index=, columns=)` |
| 335-341 | `_add_det_to_proba` | `y_proba[col] + y_pred[var]` by column label |

Leaf implementations (static grep, files under `sktime/forecasting` excluding tests):
- 27 files define `_predict_interval`, 28 define `_predict_quantiles`, 13 define
  `_predict_var` and 18 define `_predict_proba`. 57 files set `capability:pred_int: True`.
- **31** of the files implementing interval/quantiles build a pandas `MultiIndex` themselves
  (`from_product`/`from_tuples`/`from_arrays`). The column layout is therefore a de facto
  leaf contract, not only a base-class detail.
- Example: `NaiveForecaster._predict_quantiles` (`naive/_naive.py:441-484`) calls
  **public** `self.predict` and `self.predict_var` per instance and builds
  `pd.MultiIndex.from_product`.

### 1c. Path 2: `BaseProbaForecaster` (`forecasting/base/_base_proba.py`)

| line | what | coupling |
|---|---|---|
| 74-75 | non-vectorized → `BaseForecaster.predict_proba` | — |
| 97 | `X_inner = self._check_X(X=X)` | **stale**: `_check_X` now returns `(X_inner, X_schema)` (`_base.py:590`), so this passes a tuple |
| 135 | `yvec = self._yvec` | **stale**: `_yvec` is never set anywhere in sktime after the stateless refactor (`grep _yvec` shows only this line). Measured: `AttributeError: 'NaiveProba' object has no attribute '_yvec'` (probe, `BaseProbaForecaster_panel_predict_proba`) |
| 144-156 | `yvec.vectorize_est(forecasters, method="_predict_proba", return_type="list")` | calls the **private** `_predict_proba` per instance. The X is the old row-vec convention |
| 198-201 | `yvec.get_iter_indices()` → `pd.RangeIndex` fallback | pandas keys |
| 210-241 | per-instance `pd.MultiIndex.from_tuples(instance+time)` and `pd.concat([idx.to_frame()...]).index` | O(n_instances) Python loops plus pandas index building |
| 244-283 | parameter stacking (`_coerce_distribution_param`, 290-320) → `dist_class(**stacked, index=full_index, columns=columns)` | skpro distributions are pandas-indexed (index/columns) |
| 340-414 | Empirical: per-instance `spl` index rewrite + `pd.concat(spl_dfs)` | pandas sample frame |

Reachability:
- Only `MCRecursiveProbaReductionForecaster` (`compose/_reduce_proba.py:148`) subclasses it.
  Its `y_inner_mtype` includes `pd-multiindex`/`pd_multiindex_hier` (`:246`), so it is
  never vectorized. Path 2 is therefore dead in-tree and broken for any Series-inner subclass
  (third party).
- Stock `BaseForecaster.predict_proba` on Panel raises
  `NotImplementedError('automated vectorization for predict_proba is not implemented')`
  (`_base.py:1061-1064`, probe `stock_panel_predict_proba`).

Two more stock defects found (leaf-level, pre-existing, not container issues):
- `NaiveForecaster.predict_var()` returns column `0` instead of the variable name `c0`, both
  for Series and Panel (probe `stock_pd.cols.predict_var = ["0"]`; `_naive.py:608`
  `pd.DataFrame(marginal_vars, index=fh_idx)` without columns). This violates the documented
  contract ("Column names are exactly those of y").
- `predict_var(cov=True)` returns all-NaN, because `np.fill_diagonal` returns None
  (`_naive.py:603`). Measured: `all nan True`.

---

## 2. Prototype matrix: what each one needs (`g4_proto_probe.py --n 200`, Naive(last), fh=1..3, coverage [.8,.9])

| route | output type | equal to stock | seconds (interval / quantiles / var) | verdict |
|---|---|---|---|---|
| stock, pandas in | pd | — | 2.74 / 2.64 / 0.44 | reference |
| stock, polars in | **pd** | 3/3 | 2.91 / 2.71 / 0.45 | works, wrong container (a1.4) |
| x4 adapter, `policy=native` + cutoff bridge, polars in | **pd** | 3/3 | 3.45 / 3.13 / 0.40 (not interleaved, ±noise) | **works via pandas fallback**: leaves are pandas-inner, `VectorizedDF2.reconstruct` dispatches on OUTPUTS → `PandasAdapter.concat_rows_with_keys` = `pd.concat(keys)` keeps the column MultiIndex |
| x4 PolarsAdapter with polars per-instance proba parts | pl | — | — | works only with a column encoding (`"c0\|0.8\|lower"`), and the time column becomes `index` because the per-instance index is unnamed. Needs a spec, see §3 |
| x5 narwhals | — | — | — | **no contract**. `pandas_to_long` renames tuple columns to `__col0__..` and keeps the MultiIndex only in `PandasIndexMeta` (pandas-only round trip). For polars, `reconstruct` "succeeds" with column names `"('c0', 0.8, 'lower')"` (str(tuple), float repr) and time `"('index', '', '')"`. These are not parseable reliably |
| x6 hub | — | — | — | **no contract**. `from_outputs(columns=tuples)` → `to_pandas` gives a flat Index of tuples, not a MultiIndex. `to_polars` raises `KeyError('schema')` for any pandas-ingressed hub (prototype gap, not proba specific) |
| x7 hook, pd in | pd | yes | native fit+predict 0.0056; fit+predict+**interval 3.58** vs stock fit+interval 3.61 | **works via materialization**: proba recomputes the whole stock loop, which costs ~640x the native fit+predict and leaves no gain |
| x7 hook, pl in | pd | yes | 0.0038; 3.86 vs 3.92 | same |

What each approach needs:
- **x4 (adapter):** no new column-level op if the long schema is used. `concat_rows_with_keys`
  already does it. The flat-column variant needs `concat_cols(keys=...)` (exists) plus a
  `ProbaSpec`, so that labels are never parsed. The `_base.py:2466`
  `columns.droplevel(1)` for multivariate must move behind the spec.
- **x5 (narwhals):** needs the long schema: variable and coverage/alpha as **data columns**,
  because narwhals and polars only allow str column names. This is a natural fit. Without it
  there is no lossless path.
- **x6 (hub):** values are a 2-D block, so the natural layout is the wide flat block
  (`n_rows × V·C·2`) plus a `ProbaSpec` in `meta`. This is the cheapest egress to
  MultiIndex (§4: 17–21 ms at 10k).
- **x7 (hook):** needs native `_predict_quantiles_panel`/`_predict_var_panel` that return
  the long schema; otherwise the fallback cost is the full stock cost (above). Demonstrated in
  §3c for Naive(last).

---

## 3. The long/tidy schema (`protos/g4_proba_schema.py`)

```
interval  : key_1..key_k | time | variable | coverage | lower | upper
quantiles : key_1..key_k | time | variable | alpha    | value
var       : key_1..key_k | time | variable | var          (wide = predict layout also fine)
+ ProbaSpec(kind, key_names (may be None), time_name, variables (original labels),
            levels (coverage/alpha in REQUESTED order), physical key/time column names)
```
Canonical row order:
- rows are ordered by instance (contiguous, any order), then time (strictly increasing), then
  variable and level in spec order;
- egress to pandas is then a reshape plus a strided key/time take;
- a guard checks this in O(rows), and if the check fails the frame is re-sorted
  lexicographically by keys and time.

Every backend can hold it (only scalar str/float columns). Interval↔quantiles is a pure
relabel (`alpha = round(0.5 ± c/2, 12)`).

### 3a. Correctness vs stock (exact `assert_frame_equal(check_exact=True)`)

| case | interval pd/pl | quantiles pd/pl | var pd/pl | shuffled → guard → sort |
|---|---|---|---|---|
| Naive Panel n=1000 | ✓/✓ | ✓/✓ | ✓/✓ | ✓ all 6 |
| Naive Hierarchical 3 levels | ✓/✓ | ✓/✓ | ✓/✓ | ✓ |
| Naive unnamed levels (None) + int time | ✓/✓ | ✓/✓ | ✓/✓ | ✓ |
| Naive 2 variables (column vectorization) | ✓/✓ | ✓/✓ | ✓/✓ | ✓ |
| Theta(sp=1) Panel n=200 | ✓/✓ | ✓/✓ | ✓/✓ | ✓ |

Notes on these checks:
- Coverage was passed unsorted (`[0.9, 0.8]`) and alpha unsorted (`[.05,.5,.95,.1]`). The
  order is preserved through the spec.
- First-round finding: a guard that only checks the `(variable, level)` pattern is
  **insufficient**. With V=1 and no levels, `var` passes on shuffled rows. The final guard
  also checks block constancy, instance contiguity and increasing time.
- Pitfall: `0.5 - 0.9/2 == 0.04999999999999999`. Float level keys must be rounded (12 dp,
  as the mixin does), otherwise joins and maps on alpha fail.

### 3b. Interval→quantile relabel in polars

Running `interval_long_to_quantiles_long_pl` on the native interval output, then
`canonical_sort` and conversion to pandas, equals stock `predict_quantiles(alpha=[.05,.1,.9,.95])`
(rtol 1e-12) ✓.

### 3c. Panel-native proba kernel (x7-style), Naive(last, sp=1), polars lazy

All instances are computed in one `group_by(maintain_order).agg(last, mse)`, followed by a
cross join with h and `last ± sqrt(mse·h)·Φ⁻¹(0.5±c/2)`. The output is in the long schema.
- **Equal to stock:** index and columns identical, values within 2.8e-14 abs at n=1000 with
  fh=1..3. At fh=1..12, n=200, the max abs diff is 1.4e-14 and `assert_frame_equal(rtol=1e-12)`
  passes. (The bench's `np.allclose(rtol=1e-12, atol=0)` flag reads False only because of a
  near-zero element with rel. err 9e-12.)
- Assumes rows sorted by time within instance, no NaN, and daily freq (the sortedness/NaN
  guards are the G13 items, and freq is the a3 time axis).

| n series, fh=1..12, cov [.9,.8] | native polars → long | + → pandas MI | stock `predict_interval` (single run) | stock `predict_var` |
|---|---|---|---|---|
| 1,000 | **16.0 ms** | 34.1 ms | 18,083 ms | 3,108 ms |
| 10,000 | 187.7 ms | 264.3 ms | unmeasured (linear extrapolation ≈ 180 s) | — |

Stock interval vs var: interval is 5.8x var at 1k. This is because Naive's
`_predict_quantiles` re-enters public `predict` and `predict_var` per instance. In a cProfile
of stock interval at n=200 (profiler-inflated total 5.6 s):
- `predict` re-entry is 3.36 s;
- `predict_var` is 0.90 s;
- `_get_columns` is 0.36 s;
- the final concat is ≤0.55 s.

So the proba path is dominated by per-slice public re-entry, the same as for fit/predict in
the settled facts, and **not** by the output layout.

---

## 4. Conversion cost: long ↔ today's pandas layout (`g4_proba_bench.py` part B)

Synthetic stock-layout interval frame (fh=12, coverage [.9,.8]), interleaved, min of 5
rounds (3 at 10k). Baseline is **today's final reconstruct only**: `pd.concat(per_instance,
keys=...)`. That is a lower bound of today's proba egress (it excludes per-instance work).

| op | 1k, 1 var (ms) | ratio | 10k, 1 var (ms) | ratio | 10k, 3 vars (ms) | ratio |
|---|---|---|---|---|---|---|
| today: `pd.concat(keys)` of per-instance MI frames | 22.3 | 1.00 | 205.2 | 1.00 | 213.7 | 1.00 |
| today: `_get_columns` MultiIndex × n_instances | 704 | 31.5 | 7,964 | 38.8 | 7,382 | 34.5 |
| pandas MI → long pandas | 3.9 | 0.17 | 39.1 | 0.19 | 178.4 | 0.83 |
| pandas MI → long polars | 9.3 | 0.42 | 92.7 | 0.45 | 224.5 | 1.05 |
| long pandas → MI, guarded | 4.9 | 0.22 | 29.7 | 0.14 | 51.7 | 0.24 |
| long pandas → MI, unguarded | 3.3 | 0.15 | 14.3 | 0.07 | 17.3 | 0.08 |
| **long polars → MI, guarded** | 10.0 | 0.45 | **85.4** | 0.42 | 232.3 | 1.09 |
| long polars → MI, unguarded | 6.0 | 0.27 | 27.2 | 0.13 | 72.4 | 0.34 |
| long polars, shuffled → sort → MI | 15.6 | 0.70 | 230.3 | 1.12 | 979.6 | 4.58 |
| **wide flat polars (+spec) → MI** | 4.3 | 0.19 | **17.4** | 0.08 | 21.1 | 0.10 |
| polars long → pandas long (`to_pandas`) | 3.4 | 0.15 | 18.1 | 0.09 | 48.3 | 0.23 |
| per-instance numpy blocks → one polars long table | 10.2 | 0.46 | 93.3 | 0.45 | 312.4 | 1.46 |

Rows: 10k × 12 = 120k wide rows. Long is 240k rows (1 var) or 720k rows (3 vars). Peak
process RSS was 648 MB.

Reading:
- (1) Long → MultiIndex for the pandas boundary is cheap. In canonical order it is ≤0.45x
  of today's concat for 1 variable, and the guard costs 2–3x the unguarded reshape on polars
  (object-string compare of `variable`; a categorical or int-coded variable column would
  remove most of that, *expected*).
- (2) The wide flat layout with a spec is the cheapest egress (0.08–0.19x) and suits x6/x4.
  Long is the most portable (narwhals/x5, SQL engines, lazy) and the one native kernels emit.
- (3) With 3 variables, long multiplies rows by V·C. Long polars → MI then reaches ~1.1x of
  today's concat, and unsorted input costs 4.6x. Producers must emit canonical order.
- (4) Today's per-instance `_get_columns` MultiIndex construction (0.7–0.8 ms per instance)
  is 31–39x the final concat. A spec-based layout built **once** at the boundary removes it.
  This is the largest layout-attributable cost found, about 4–7% of stock interval time
  (704 ms of 18.1 s at 1k).

All container-only effects remain small next to the per-instance re-entry (18.1 s at 1k).
This is consistent with the settled "~1.05x cap" for container-only changes.

---

## 5. Distributions (`predict_proba`) concat (`g4_proba_bench.py` part C, n=1000, skpro `Normal`)

| step | time |
|---|---|
| per-instance `predict_proba()` × 1000 (stock leaves) | 11.75 s |
| today `_concat_distributions_hierarchical` (fed the schema's `get_iter_indices`, bypassing the `_yvec` bug) | **826 ms** |
| one `Normal(mu, sigma, index=MultiIndex.from_arrays, columns)` from a long `(keys, time, variable, mu, sigma)` polars table | **3.24 ms** (255x faster) |
| equality (index, mean, var) | ✓ / ✓ / ✓ |

Concat requirements for distributions:
1. All parts must have the same distribution class (checked, `_base_proba.py:186-191`).
2. Parameters must be stackable to `(n_rows, n_cols)` arrays (`_coerce_distribution_param`),
   or `Empirical` `spl` must be concatenable with an instance-prefixed index.
3. The index is a pandas MultiIndex (instance levels + time), with columns = variables.
   skpro's `BaseDistribution` is pandas-indexed, so a non-pandas *distribution object* is out
   of sktime's scope (skpro would have to change).
4. The long **parameter table** `(keys, time, variable, <param_1..p>)` is the natural
   backend-neutral form for parametric families. Empirical needs `(keys, time, sample_id,
   variable, value)`. Both convert to the skpro object in one constructor call.
5. Ordering, guard and spec rules are the same as for interval/quantiles.

---

## 6. Recommended public-API contract for non-pandas proba outputs

Principles:
- the output container follows the input container, as `predict` already does;
- non-pandas proba outputs use the **long schema** (§3);
- pandas users see **no change**.

| method | pandas input (unchanged) | non-pandas input, target contract |
|---|---|---|
| `predict_interval` | row MI (keys, time); columns MI (var, coverage, lower/upper) | long `keys..., time, variable, coverage, lower, upper` in the input's backend (eager; lazy input → eager in the collect backend, since outputs are O(n·h·V·C)) |
| `predict_quantiles` | columns MI (var, alpha) | long `keys..., time, variable, alpha, value` |
| `predict_var(cov=False)` | columns = variables | same layout as `predict` (keys/time + one column per variable). It is already flat, so no long form is needed |
| `predict_var(cov=True)` | columns MI (var, fh) | `NotImplementedError` for non-pandas in phase 1. Later long `keys, time, time_2, variable, cov` |
| `predict_proba` | skpro distribution (pandas-indexed) | **still a skpro distribution** (skpro is pandas-indexed). Add `return_params="long"` / a `dist.to_long()` utility for the parameter table |

Level semantics:
- coverage/alpha keep the user's order;
- float keys are rounded to 12 dp;
- variable labels are str in the frame, and the original labels are available through
  `forecaster.get_proba_spec()` (or attached as frame metadata where the backend supports it,
  e.g. polars does not: use the spec).

Deprecation path:
1. **Now (no behaviour change):**
   - Ship `sktime.datatypes.proba_to_long(df, kind)` / `long_to_proba(long, spec)` (the
     prototype functions) and document that proba outputs are pandas regardless of input.
   - Fix `predict_var` column names (§1c).
   - Repair or remove `BaseProbaForecaster` (`_yvec`, `_check_X` tuple). Better: fold the
     vectorized `predict_proba` into `BaseForecaster._vectorize` and build it from a long
     parameter table (§5, 255x on the concat).
2. **Phase 1 (release N):**
   - Add config `"output:proba_format": "pandas" | "native"`, default `"pandas"`.
   - For non-pandas input with the default, emit a `FutureWarning` once per estimator: "proba
     outputs will follow the input container (long format) from N+2; set
     output:proba_format to silence".
   - Internally, `BaseForecaster` accepts either layout from leaves (`_predict_*` may return
     long + spec via a new tag `proba_inner_format: "pandas" | "long"`). It converts at the
     boundary once, using spec-based egress (§4: ≤0.45x of today's concat for 1 variable;
     no per-instance MultiIndex).
3. **Phase 2 (N+2):**
   - The default becomes `"native"` **for non-pandas input only**. Pandas input keeps the
     MultiIndex layout indefinitely (removing it would break 31 leaf files that build it
     plus all metrics, a1's "4/4 metric configs break").
   - Metrics and evaluate gain long-format readers before the flip (G5/G16 dependency).
4. **Prototype implications:**
   - x4 adds no column op;
   - x5 needs the long schema;
   - x6 uses the wide flat block + spec;
   - x7 hooks must expose native proba kernels, otherwise proba erases the gain (§2).

## Caveats
- Timings are single-machine and in-process. Stock 1k interval and var are single runs (18 s
  per run). The x4-vs-stock proba timings in §2 are not interleaved, and their ±20% is noise
  level. The 10k stock figure is an extrapolation (unmeasured).
- The cost table in §4 uses synthetic stock-layout frames, not timed stock outputs.
  Correctness (§3a) uses real stock outputs.
- The native kernel covers only Naive(last, sp=1), daily freq, sorted, no NaN. Other
  strategies and families (Theta, ETS, ARIMA intervals) need their own panel kernels. Most
  leaves have no closed form across instances.
- The census counts in §1b are static grep file counts, not class counts.
- Memory was not measured beyond process RSS.
- skpro was installed only in the own venv. EXP_PY has no skpro or statsmodels.
