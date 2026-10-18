# r1 — Prior art: scikit-learn and skrub on container independence

Workstream key: `r1_sklearn_skrub`. Date: 2026-10-18.
Scope: (1) sklearn `set_output` adapter dispatcher, (2) sklearn Array API incremental port,
(3) sklearn's dataframe-agnosticism / narwhals decision, (4) skrub's own dispatch layer,
(5) status of `__dataframe__` vs Arrow PyCapsule vs DataFrame API Standard.
Every number below comes from a script in `scripts/r1_*.py` that I ran; source
facts cite file:line in the installed packages; history cites GitHub/docs URLs.
Anything not measured is labelled **unmeasured/expected**.

## TL;DR

**Headline:** neither sklearn nor skrub made their *compute* container-agnostic. Both keep a
thin container layer at the boundary and compute in numpy. Both also treat the row index as
non-semantic. That is exactly the assumption sktime cannot make, because the index holds the
time points and instance keys.

1. **sklearn's settled model is "narrow waist at the boundary."**
   - **In:** narwhals handles validation and selection, through 3 files and 20 call sites (5 distinct `nw.*` entry points).
   - **Compute:** numpy, or Array-API `xp` on an opt-in basis.
   - **Out:** a 4-method adapter registry (`set_output`).
   - **Measured:** with pandas *not installed*, polars → StandardScaler / ColumnTransformer / Ridge / train_test_split all work (4/4).
2. **sklearn's output dispatcher is tiny and open.** The protocol is 1 attribute plus 4 methods (`create_container`, `is_supported_container`, `rename_columns`, `hstack`); the pandas adapter is 25 LOC and the polars adapter 23 LOC. My 24-LOC third-party pyarrow adapter, registered through the private `ADAPTERS_MANAGER`, worked in 10/10 cases (5 estimators/composites × {pandas, polars} input). It stays this small only because sklearn has no index semantics.
3. **Array API is the model for incremental, per-estimator porting.** The pieces:
   - a global opt-in switch (`array_api_dispatch=False` by default);
   - a per-estimator **dynamic** tag (`array_api_support`), which can depend on hyperparameters and which composites derive from their components;
   - a test matrix keyed on that tag, plus a "dispatch on, numpy input, results unchanged" check for *non*-supporting estimators.

   The cost: 19 of 210 estimators after ~3.5 years (1.2, Dec 2022 → 1.9.1), with 235 `get_namespace` call sites across 38 files.
4. **sklearn adopted narwhals as a hard dependency in 1.9** (`Requires-Dist: narwhals>=2.0.1`), after a 14-month RFC.
   - **Trigger:** polars deprecating, then removing, `__dataframe__`.
   - **Kept anyway:** a pandas-special path.
   - **Still unsupported:** LazyFrames (measured: a cryptic `ValueError`).
   - **Collateral:** sktime (`scikit-learn<1.10`) gets narwhals transitively whenever sklearn 1.9.x is installed, but that is not guaranteed by sktime's own bounds.
5. **skrub built its own `functools.singledispatch` layer and declined narwhals** (#1818, closed "not planned" 2026-07-29).
   - **Size:** 72 generic ops (30 pd.DataFrame / 26 pl.DataFrame / 13 pl.LazyFrame specialisations), plus 17 more generic functions with 35 specialisations scattered over 14 estimator modules. The shared namespace never covers everything.
   - **Bug class:** silent per-backend divergence. I reproduced #2335: identical data gives pandas session ids `[0, 1]` but polars `[0, 0]`.
   - **pandas:** remains a hard dependency.
6. **Protocols in 2026:** `__dataframe__` is dead, the DataFrame API Standard is dormant, and the Arrow PyCapsule `__arrow_c_stream__` is the live ingress/egress path.
   - **`__dataframe__`:** removed in polars 2.0 (`AttributeRemovedError`); deprecated in pandas 3.0 (`Pandas4Warning`), with removal planned for 4.0; dropped by sklearn 1.9.
   - **DataFrame API Standard:** repos untouched since 2024-04; pandas 3 dropped `__dataframe_consortium_standard__`.
   - **PyCapsule `__arrow_c_stream__`:** pandas, polars eager, pyarrow, duckdb and narwhals all have it. Polars LazyFrame and dask DataFrame do **not**.
   - **Index fidelity:** pandas exports a MultiIndex losslessly through the capsule (as trailing columns plus metadata).
   - **Gotcha:** `pl.from_pandas()` silently drops the index by default.

## Environment

- `EXP_PY` (pandas 2.3.3, numpy 2.2.6, polars 2.0.0, narwhals 2.27.1, pyarrow 26.0.0, dask 2026.8.0, duckdb 1.5.6, scikit-learn 1.9.1)
- `EXP_PY3` (same, but pandas 3.0.6)
- `R1_PY` = my own venv `scratchpad/venvs/r1_sklearn_skrub` (skrub 0.11.0, scikit-learn 1.9.1, pandas 3.0.6, polars 2.0.0, narwhals 2.27.1, numpy 2.5.4)
- `R1_NOPANDAS` = my own venv `scratchpad/venvs/r1_nopandas`: scikit-learn 1.9.1, polars 2.0.0, narwhals 2.27.1, scikit-base 1.2.0, scipy, numpy. **pandas and pyarrow not installed** (verified with `importlib.util.find_spec`).
- The machine was shared by ~14 concurrent agents, so absolute timings are noisy. I report within-process ratios and use min-of-repeats, interleaved.

All commands run from the worktree root with `PYTHONPATH=.:experiments/container_independence`.

---

## 1. sklearn `set_output`: an output-container dispatcher

**Category:** adapter/dispatcher pattern, for OUTPUT containers only.

**How it works.** Source: `sklearn/utils/_set_output.py`.
- **Protocol:** `ContainerAdapterProtocol` (l.39–114) is a `runtime_checkable` Protocol with `container_lib: str` plus four methods.
- **Adapters:** `PandasAdapter` (l.117) and `PolarsAdapter` (l.159), registered in the module-global `ADAPTERS_MANAGER` (l.209–211).
- **Selection:**
  - by *config string*: `set_output(transform="polars")`, then `ADAPTERS_MANAGER.adapters[...]` (l.232–238, l.314);
  - or by *input object*: `_get_adapter_from_container` keys on `type(X).__module__.split(".")[0]` (l.214–229).
- **Wiring:** `_SetOutputMixin.__init_subclass__` (l.368) wraps `transform`/`fit_transform` automatically. Estimators therefore return numpy, and the mixin wraps the result.
- **Consumers:** `pipeline.py:2108` (FeatureUnion hstack), `compose/_column_transformer.py:1121`, `preprocessing/_function_transformer.py:278`, `inspection/_plot/decision_boundary.py:659`.

**Measured** (`scripts/r1_sklearn_census.py`, `scripts/r1_set_output_ext.py`):

```
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/r1_sklearn_census.py --grid dev
  set_output_protocol: methods [create_container, is_supported_container, rename_columns, hstack]; attributes [container_lib]
  registered_adapters [pandas, polars]; loc_pandas_adapter 25; loc_polars_adapter 23
  tag_census: n_estimators 210, instantiated 209, n_set_output_available 78
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/r1_set_output_ext.py --grid dev
  adapter_loc 24; registered_after [pandas, polars, pyarrow]
  StandardScaler/ColumnTransformer/FeatureUnion/Pipeline/FunctionTransformer x {pandas, polars} input -> "ok: pyarrow.Table" (10/10)
```

The third-party adapter is `protos/r1_sklearn_pyarrow_adapter.py`.

**Boundary tax of the "numpy inside, wrap at edge" model.** StandardScaler.fit_transform, 4 columns, ratio vs C-ordered numpy in with default out, same process:

| rows | numpy→default | numpy→polars | numpy_F→default | pandas→default | pandas→polars | polars→default | polars→polars |
|---|---|---|---|---|---|---|---|
| 10 000 | 1.00 (3.86 ms) | 1.08 | 0.52 | 1.33 | 1.57 | 1.33 | 1.00 |
| 100 000 | 1.00 (19.9 ms) | 1.30 | 0.44 | 1.15 | 1.49 | 0.54 | 0.53 |
| 500 000 | 1.00 (80.7 ms) | 1.52 | 0.40 | 1.18 | 1.46 | 0.49 | 0.44 |

Two effects show up:

- **Egress to polars** costs +30–52% at 100k–500k rows. `PolarsAdapter.create_container` builds `pl.DataFrame(X, orient="row")` from a C-ordered array.
- **Polars *input* is 2× faster.** The cause is memory layout, not the container. `to_numpy_layout {'numpy C': 'C', 'pandas.to_numpy()': 'C', 'polars.to_numpy()': 'F'}`, and Fortran-ordered numpy reproduces the speedup (x0.40–0.52).

So for column-wise estimators, the biggest "container" effect is the array layout the container hands over.

**What went well**
- A tiny protocol.
- Adding polars output in 1.4 meant one 23-LOC class. The `set_output` docstring says `versionadded:: 1.4 "polars" option was added`.
- Composites (ColumnTransformer, FeatureUnion) only need `hstack` and `rename_columns`.
- The registry is open in practice (my pyarrow test), though private.

**What went wrong / limits**
- Output only. Input handling lives elsewhere (validation, below).
- No index semantics. `PolarsAdapter.create_container` ignores `X_original` entirely; only pandas propagates the index.
- **Breakage from a fast-moving backend:** polars ≥1.42.1 deprecated `concat(how="horizontal")`, and sklearn 1.9.1 had to ship #34447, which uses `how="horizontal_extend"` (visible at `_set_output.py:188–194`).
- `ADAPTERS_MANAGER` is private API, so there is no supported extension story for users.

**Relevance to sktime**
- sktime's `VectorizedDF` reconstruct / `pd.concat(keys=...)` is the analogue of `hstack`. However, sktime's protocol would need index-bearing ops: instance-key enumeration, per-instance split, concat with keys, time-index carry-through, cutoff/fh arithmetic.
- **Unmeasured/expected:** an sktime adapter is roughly 8–12 methods, not 4, because sktime's rows have identity.
- The *mechanism* transfers directly: a registry keyed on module root or config string, a Protocol class, and wrapping in a base-class mixin (sktime's `BaseForecaster` already wraps `fit`/`predict` the same way).

## 2. sklearn Array API: incremental, per-estimator, tag-gated porting

**Category:** standard-based namespace dispatch (`xp = get_namespace(X)`), behind an opt-in flag.

**How it works** (`sklearn/utils/_array_api.py`)
- `get_namespace` (l.375–467) returns `np_compat` **unless** `set_config(array_api_dispatch=True)`. It also requires `SCIPY_ARRAY_API=1` (`_check_array_api_dispatch`, l.164–188).
- sklearn vendors `array_api_compat` 1.15.0 and `array_api_extra` under `sklearn/externals/`.
- `move_to` (l.519) prefers DLPack, then falls back through numpy.
- `move_estimator_to` (l.1084) and `check_same_namespace` (l.1110) enforce "fit namespace == predict namespace."
- **Rule from the docs:** "everything follows X"; outputs come back in the input's namespace and device.

**Per-estimator opt-in**
- **The tag:** `Tags.array_api_support: bool = False` (`utils/_tags.py:215,242`). It is set in `__sklearn_tags__` and is **dynamic**:
  - `LogisticRegression`: `tags.array_api_support = self.solver == "lbfgs"` (`linear_model/_logistic.py:1631`);
  - `PCA`: conditional on solver (`decomposition/_pca.py:844`);
  - `FeatureUnion`: `all(get_tags(t).array_api_support ...)` (`pipeline.py:2204`);
  - `GridSearchCV` inherits from its sub-estimator (`model_selection/_search.py:500`).
- **The test strategy:** `estimator_checks.py:209,346–380`.
  - For tag=True: `check_array_api_input` runs over every (namespace, device, dtype) combination: numpy, array_api_strict, cupy, torch {cpu, cuda, xpu, mps} × {float64, float32}, dpnp. Mixed-namespace checks run too, with a per-estimator XFAIL list.
  - For tag=False: one "safety-net" check, `check_array_api_input(array_namespace="numpy", expect_only_array_outputs=False)`. It asserts that turning dispatch on with numpy inputs changes nothing.

**Measured** (`r1_sklearn_census.py`, runtime tag census with default hyperparameters):

```
n_estimators_all_estimators 210, n_instantiated 209, n_array_api_support_true 19
array_api_support_true: Binarizer, FeatureUnion, GaussianNB, GridSearchCV, KernelCenterer, LabelEncoder,
  LinearDiscriminantAnalysis, LogisticRegression, MinMaxScaler, Normalizer, Nystroem, PCA, PoissonRegressor,
  PolynomialFeatures, RandomizedSearchCV, Ridge, RidgeCV, RidgeClassifierCV, StandardScaler
array_api_footprint: n_files_calling_get_namespace 38, n_get_namespace_call_sites 235 (excl. _array_api.py)
```

A static grep finds only 12 `array_api_support = True` literals, so a runtime census is required. Tags are computed rather than declared: `__sklearn_tags__`, conditional expressions, and composites.

**History:** Array API landed in 1.2 (Dec 2022), with only `LinearDiscriminantAnalysis`. That is about 3.5 years to reach 19/210 estimators with default hyperparameters. Per the user guide, support is still "considered experimental" and "backwards compatibility is not guaranteed." The 1.9.0/1.9.1 changelog has about 20 array-API entries: 11 features/enhancements (e.g. LogisticRegression and PoissonRegressor with lbfgs, FeatureUnion, Nystroem, RidgeCV, several metrics) and 9 fixes. The porting is still active.

**What went well**
- Nobody waits: estimators port one at a time.
- The default path (numpy) is untouched.
- The tag drives the test matrix automatically, and the safety-net check catches regressions in non-ported estimators.
- Composites compute capability from their parts.

**What went wrong / limits**
- **Slow:** 9% of estimators after 3.5 years.
- **Pervasive:** `xp` plumbing touches 38 files and 235 call sites.
- **Hyperparameter-conditional support:** only some solvers. Users only find out through tags or the docs.
- **Environment coupling:** requires `SCIPY_ARRAY_API=1`.
- **Mixed-namespace pain:** for example #34779, pandas labels combined with torch predictions.

**Relevance to sktime: the closest analogue to what sktime needs**
- sktime's `y_inner_mtype` / `X_inner_mtype` tags are *already* a per-estimator opt-in for containers. Today they can list several mtypes, and the framework converts to the first compatible one.
- What sktime lacks, compared with Array API:
  - (a) a global "preserve input container" switch, akin to `array_api_dispatch`;
  - (b) a test matrix that takes the container family from the tag;
  - (c) a safety-net test that "with the switch on, a pandas-only forecaster still gives identical results";
  - (d) an "output follows input" plus `check_same_namespace` contract between fit and predict.
- **Unmeasured/expected:** a polars-native inner path would follow the same curve: a handful of high-value forecasters first (naive, reduction, statsforecast-backed), with the rest converting at the boundary indefinitely.

## 3. sklearn and dataframe-agnosticism: interchange protocol, then narwhals

**Category:** third-party compatibility layer (narwhals), used **only at the boundary**.

**Timeline**
- **[#25896](https://github.com/scikit-learn/scikit-learn/issues/25896) (opened 2023-03-17):** polars/pyarrow requested. thomasjpfan: "I prefer to lean as much as we can on the DataFrame exchange protocol." The interchange protocol plus `to_numpy` was adopted for polars input. The issue closed 2025-06-16, pointing to the narwhals RFC.
- **1.4:** polars output (`PolarsAdapter`).
- **[#31049](https://github.com/scikit-learn/scikit-learn/issues/31049), RFC "adopt narwhals", 2025-03-21 → 2026-05-07:**
  - An emoji poll on 2025-04-01 favoured a dependency (13 votes) over vendoring (1).
  - ogrisel preferred optional, and keeping custom pandas code "to avoid introducing a new dependency to the pandas users."
  - adrinjalali argued for a hard dependency, citing technical debt from frequent polars breakage.
  - GaelVaroquaux and StefanieSenger asked what the gain was, given that sklearn converts to numpy early.
  - Lazy frames were raised. A narwhals maintainer noted that LazyFrame row order is undefined, and thomasjpfan objected to adding a third code path.
  - The final trigger: polars deprecated `__dataframe__`. glemaitre reported, 2026-04-20, that skrub nightlies broke.
- **[#33789](https://github.com/scikit-learn/scikit-learn/pull/33789):** a stop-gap that made the polars `validate_data` path independent of `__dataframe__` ("deprecation ... in polars version 1.40", per the 1.9 changelog).
- **[#31127](https://github.com/scikit-learn/scikit-learn/pull/31127), merged 2026-05-07:** narwhals became a hard dependency and the PR "replaces any use and mention of `__dataframe__`." The 1.9 changelog says narwhals "simplifies the support of dataframe input X and dataframe output as specified in the set_output API."

**What sklearn actually uses** (measured by `r1_sklearn_census.py`, AST scan):

```
files_importing_narwhals: compose/_column_transformer.py, utils/_indexing.py, utils/validation.py (3 files)
nw_attribute_chains: dependencies.is_into_dataframe 9, from_native 6, dependencies.is_into_series 3, Categorical 1, Enum 1 (20 call sites)
```

The narwhals frame methods called, read from the source, are about 10: `.shape`, `.columns`, `.select`, `.filter`, `[...]`, `.to_native`, `.iter_columns`, `.item`, `.implementation.is_pandas()`, `.schema.dtypes()`. These cover sample counts, feature names, column/row selection, categorical-dtype detection, and the dtype probe in `check_array`. There are **zero** compute ops.

- **pandas special path kept:** `_safe_indexing` still checks `hasattr(X, "iloc")` → `_pandas_indexing` *before* the narwhals path (`utils/_indexing.py:290–303`). The comment cites `test_safe_indexing_pandas_no_settingwithcopy_warning`, which fails under narwhals because narwhals objects are immutable.
- **Workaround:** narwhals `.columns` can return ints for pandas (`validation.py:2361–2364`, narwhals#3571).

**Measured: pandas genuinely optional for polars users** (`scripts/r1_pandas_free.py` in `R1_NOPANDAS`):

```
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/r1_pandas_free.py --py $R1_NOPANDAS/bin/python --tag nopandas
sklearn.StandardScaler set_output=polars   ok  polars.DataFrame   pandas_loaded=false
sklearn.ColumnTransformer set_output=polars ok  ['a__c0','a__c1','b__c2']
sklearn.Ridge fit/predict polars           ok  ndarray(200,)
sklearn.train_test_split polars            ok  DataFrame(150,3)
sklearn.StandardScaler on polars LazyFrame FAIL ValueError: setting an array element with a sequence.
narwhals ops on polars                     ok
sktime import sktime.datatypes             FAIL ModuleNotFoundError: No module named 'pandas'
sktime check_is_mtype pl.DataFrame         FAIL ModuleNotFoundError: No module named 'pandas'
sktime NaiveForecaster import              FAIL ModuleNotFoundError: No module named 'pandas'
```

With pandas installed (`--tag exp`), sklearn still imports it: `sklearn/utils/fixes.py:18–21` does `try: import pandas`. I traced the chain with `-X importtime`: sklearn → base → … → `_array_api` → `fixes`. pandas is therefore loaded at import even on a polars-only path.

*Methodology note:* my first attempt blocked pandas with a meta-path finder that raised ImportError. That is **invalid**, because polars probes optional dependencies with `importlib.util.find_spec`, which then raises. It was replaced by the genuinely pandas-free venv, and the script documents this.

**Per-call overhead.** This matters for any per-instance loop. Measured by `scripts/r1_dispatch_overhead.py`, one series of 100 rows, interleaved min-of-5 × 2000 calls, run in `EXP_PY`:

| op (polars) | native | dict-dispatch (sklearn-style) | singledispatch (skrub-style) | narwhals `from_native`+op | sklearn validator |
|---|---|---|---|---|---|
| n_rows | 0.31 µs | 0.61 µs (x1.98) | 0.80 µs (x2.58) | 23.5 µs (x75.9) | `_num_samples` 7.4 µs |
| col mean | 1.91 µs | 2.30 µs (x1.20) | 2.87 µs (x1.50) | 67.5 µs (x35.3) | – |
| to_numpy | 22.0 µs | 23.4 µs (x1.06) | – | 79.8 µs (x3.62) | `check_array` 222 µs (x10.1) |

For pandas: n_rows native 0.55 µs, narwhals 46.7 µs (x84.8); to_numpy native 160 µs, narwhals 867 µs (x5.4), `check_array` 1636 µs (x10.2).

The narwhals cost splits as follows:

- `from_native` (stable.v2) costs 25.1 µs for polars and 39.7 µs for pandas. The main namespace costs 21.7 / 32.8 µs.
- A method on a **pre-wrapped** frame is cheap: `.shape[0]` takes 0.60 / 0.91 µs, and `.to_native()` 0.21 / 0.13 µs.

For reference, sktime's own `check_is_mtype` in the same process takes 29.9 µs (polars) and 72.4 µs (pandas). So a narwhals wrap is the same order of magnitude as the per-call check sktime already pays, and an order below sklearn's `check_array`.

**What went well**
- Polars and pyarrow input without pandas.
- The deprecated `__dataframe__` is gone.
- narwhals has no transitive dependencies.
- The narwhals footprint is small and boundary-only, so the risk is contained.

**What went wrong / limits**
- 14 months of debate.
- A pandas path was kept anyway.
- No lazy support: a measured cryptic error.
- Two polars API breaks in one release cycle (#33789, #34447).
- narwhals `.columns` typing quirk.
- pandas is still imported opportunistically.

**Relevance to sktime**
- **(a) Dependency cost.** sktime pins `scikit-learn>=0.24,<1.10.0` (`pyproject.toml:61`), so every sktime user on sklearn 1.9.x already has narwhals. The marginal dependency cost of narwhals in sktime is therefore ~0 for current users, but not guaranteed by sktime's own bounds.
- **(b) Where narwhals fits.** sklearn's narwhals usage maps onto sktime's *boundary* (input checks, column/instance selection, dtype probes), not onto forecaster internals.
- **(c) Wrap once per batch, never per instance.** At ~22–47 µs per `from_native`, wrapping inside a 10,000-instance per-series loop adds about 0.22–0.47 s per op. Hoisting the wrap brings that to ~6–9 ms (0.6–0.9 µs per op).
- **(d) Rows are positional in sklearn.** sklearn never needed narwhals to carry an index. The dev claim C2 (no MultiIndex in narwhals) is therefore irrelevant to sklearn but central to sktime.
  - narwhals 2.27.1 does ship pandas-index escape hatches: `maybe_get_index`, `maybe_set_index`, `maybe_reset_index`, `maybe_align_index`, `maybe_convert_dtypes`.
  - These are not an index model.

## 4. skrub: a home-grown singledispatch layer for pandas and polars

**Category:** dispatcher pattern, used as the *entire* dataframe API of the library, with per-function specialisation.

**How it works** (`skrub/_dispatch.py`, 274 lines; PR [#888](https://github.com/skrub-data/skrub/pull/888), merged 2024-03-01)
- `@dispatch` wraps `functools.singledispatch`. `fn.specialize("pandas"|"polars", argument_type="DataFrame"|"Column"|"LazyFrame"|"EagerFrame")` registers implementations by *library name*.
- If polars is not installed, the decorator is a no-op, so polars stays optional.
- Dispatch is on the **first argument's type**.
- **The house rule** (module docstring): outside a specialisation, skrub code "must not directly access methods of a container", not even `df.shape`. Code calls `sbd.shape(df)` instead.
- PR #888's rationale: callers don't need to know whether a function is dispatched; specialisations can live next to the feature that needs them; a function and its specialisations sit together.
- The namespace is `skrub/_dataframe/_common.py` (1532 lines). It **imports pandas unconditionally** (l.6), and skrub's `Requires-Dist` includes `pandas>=1.5.3`.

**Measured** (`scripts/r1_skrub_census.py` in `R1_PY`):

```
PYTHONPATH=.:experiments/container_independence $R1_PY -W ignore experiments/container_independence/scripts/r1_skrub_census.py
n_generic_functions 72 (public names 79, 71 of them generic)
specializations_per_type {pd.DataFrame: 30, pd.Series: 60, pl.DataFrame: 26, pl.LazyFrame: 13, pl.Series: 56}
lazyframe_supported_functions: collect, dataframe_module_name, filter, head, is_lazyframe, is_pandas_extension_dtype,
   make_column_like, make_dataframe_like, null_value_for, sample, select_rows, slice, where_row
library modules outside _dataframe (excl. tests/_docs/conftest): 75; using the sbd namespace: 40
modules defining their OWN @dispatch: 14 (17 generic functions, 35 specializations)
modules with direct pd.*/pl.* attribute use (AST, docstrings excluded): 16 (32 pd.*, 24 pl.*)
```

The unsupported-type error literally says "LazyFrames are not yet supported; please call collect()" (`_dispatch.py:265–270`).

**Index stance.** `_concat_pandas(axis=0)` uses `ignore_index=True`, and `axis=1` resets then restores the first frame's index (`_common.py:396–406`). `reset_index` / `copy_index` / `index` exist **only for pandas**. skrub treats the index as an artefact to neutralise, not as data.

**Backend divergence** (`scripts/r1_backend_divergence.py` in `R1_PY`). This reproduces skrub issue [#2335](https://github.com/skrub-data/skrub/issues/2335), filed 2026-10-15:

```
skrub_2335_session_ids: gap 1800.5s vs session_gap=1800 -> pandas [0, 1], polars [0, 0]
                        gap 0.8s   vs session_gap=0.5  -> pandas [0, 1], polars [0, 0]
total_seconds primitive: pandas 1800.5 | polars 1800.0 | polars(fractional=True) 1800.5
                         narwhals[pandas] 1800.0 | narwhals[polars] 1800.0
group order: pandas default [a,b,c]; pandas sort=False [b,a,c]; polars default [c,a,b]; polars maintain_order=True [b,a,c]
std default ddof: pandas = polars = narwhals = 3.0957 (ddof=1) vs np.nanstd 2.6810 (ddof=0)
```

- **Hand-written per-backend code drifts silently.** The common namespace already had the right primitive (`sbd.total_seconds` goes through microseconds), but an estimator-local specialisation re-implemented it wrongly.
- **narwhals removes the divergence by imposing polars semantics on pandas.** A pandas user's result changes, from 1800.5 to 1800.0.
- For sktime's VectorizedDF, which relies on `groupby(level, sort=False)` first-appearance order, the polars default `group_by` order differs (`[c,a,b]`). It needs `maintain_order=True`.

**Decision on narwhals:** issue [#1818](https://github.com/skrub-data/skrub/issues/1818), "Replacing the dataframe API with narwhals", opened 2025-12-19, closed **not_planned** 2026-07-29.

- **Reasons given:**
  - "it would require a huge rewrite" (rcap107);
  - functions defined outside `_common.py` "are more likely to be missing in Narwhals";
  - "pandas is being used for various heuristics throughout the library (guessing datetime formats and null values)";
  - jeromedockes (2026-02-04) on what makes duckdb/ibis support meaningful: "the difficulty ... is not addressing a slightly different API but the fact that _all_ the code in skrub and scikit-learn only works on in-memory arrays";
  - final close: "no plans to make this change."
- A narwhals maintainer (FBruzzesi) mapped the flagged gaps and found nearly all of them covered (exception: `pearson_corr`; `replace` only partly). **API coverage was not the blocker.**

**What went well**
- Explicit, debuggable, and fast: dispatch costs ~1.2 µs vs 0.3 µs native for `shape` (`r1_dispatch_overhead.py --tag skrubvenv`; `sbd.mean` on polars took 3.96 µs vs native 2.30 µs).
- Specialisations live next to features.
- Optional polars.
- Full control of semantics per backend.

**What went wrong**
- O(ops × backends) maintenance: 72 + 17 generic functions × 2 backends.
- Silent semantic divergence (#2335).
- LazyFrame support is thin: 13 of 72 ops, mostly metadata and row selection.
- pandas stays a hard dependency.
- Adding a third backend means touching ~89 generic functions.

**Relevance to sktime.** skrub is the concrete cost model for idea (1), the "adapter per backend" approach:

- **(a) The namespace grows.** skrub needed 72 shared ops plus local specialisations in 14/75 modules. Within VectorizedDF the surface is much smaller (**unmeasured/expected:** ~10 ops). For 100s of forecasters, an skrub-style rewrite would multiply this.
- **(b) Divergence needs cross-backend parity tests on every op.** skrub's bug escaped because parity tests were per-estimator, not per-primitive.
- **(c) skrub's own maintainers concluded that lazy backends are blocked by in-memory compute, not by API.** That is true for sktime too: statsmodels, numpy and numba forecasters need arrays.

## 5. Cross-library protocols in 2026: interchange, Arrow PyCapsule, DataFrame Standard

**Measured** (`scripts/r1_interop_protocols.py`, both envs):

```
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/r1_interop_protocols.py --grid dev
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/r1_interop_protocols.py --grid dev
```

Protocol surface ("yes" = attribute present):

| object | `__dataframe__` | `__arrow_c_stream__` | `__dataframe_consortium_standard__` |
|---|---|---|---|
| pandas 2.3.3 DataFrame | yes | yes | **yes** |
| pandas 3.0.6 DataFrame | yes (deprecated) | yes | **no** |
| pandas Series | no | 2.3: no / **3.0: yes** | no |
| polars 2.0 DataFrame / Series | **no (removed)** | yes | no |
| polars 2.0 LazyFrame | no | **no** | no |
| pyarrow Table | yes | yes | no |
| duckdb relation | no | yes | no |
| dask DataFrame | yes | **no** | no |
| narwhals DataFrame (polars / pandas) | no | yes | no |

**Deprecation status:**

- **polars 2.0.0:** `pl_df.__dataframe__()` raises `AttributeRemovedError: __dataframe__ was removed in version 2.0; the dataframe interchange protocol is not supported anymore. Consider using to_arrow or to_pandas instead.`
- **pandas 3.0.6:** `pd_df.__dataframe__()` and `pyarrow.interchange.from_dataframe(pd_df)` both emit `Pandas4Warning: The Dataframe Interchange Protocol is deprecated`. The warning recommends the Arrow PyCapsule Interface or narwhals.
- **pandas 3.0.6 docstring:** "From pandas 3.0 onwards, from_dataframe uses the PyCapsule Interface, only falling back to the interchange protocol if that fails. From pandas 4.0 onwards, that fallback will no longer be available."
- **pandas 2.3.3:** no warning. Its docstring says capsule-first "from pandas 2.3 onwards."
- **pandas 3.0.6** adds `pd.DataFrame.from_arrow` (capsule ingest). Absent in 2.3.3.
- **DataFrame API Standard:** `gh api` shows `data-apis/dataframe-api` last pushed 2024-03-29 and `data-apis/dataframe-api-compat` last pushed 2024-04-26. Neither is archived, but both are dormant, while `array-api-compat` was pushed 2026-10-13. pandas 3 dropped the `__dataframe_consortium_standard__` hook (table above).

**Index fidelity** (pandas MultiIndex `(h0, time)`, columns `c0, c1`; identical in both envs):

- **Lossless through the capsule:**
  - `pa.table(pd_mi)` / `RecordBatchReader.from_stream(pd_mi)` produce columns `[c0, c1, h0, time]` plus pandas metadata (index levels go **last**);
  - `pa.table(pd_mi).to_pandas()` returns the MultiIndex `[h0, time]` (lossless);
  - `pd.DataFrame.from_arrow(pa.table(pd_mi))` (pandas 3) also returns the MultiIndex (lossless).
- **`pl.from_dataframe(pd_mi)`:** returns `[c0, c1, h0, time]`, with index levels as plain columns. On pandas 2.3 and 3.0 this is capsule-first.
- **Gotcha:** `pl.from_pandas(pd_mi)` (default `include_index=False`) and `pl.DataFrame(pd_mi)` return `[c0, c1]` only. The **index is silently dropped**.
- `pl.from_pandas(pd_mi, include_index=True)` returns `[h0, time, c0, c1]`.

**Cost** (polars↔pandas, 1k–500k rows): every route was within x0.56–x3.4 of `pl.from_pandas` / `pl.to_pandas`, with run-to-run noise up to 2× under load. For example, the 500k-row pd→pl route measured 28 ms in one run and 59 ms in the next. No route consistently wins, so the choice of route should rest on deprecation status and index fidelity, not on speed. One consistent signal: materialising the MultiIndex (`include_index=True`) was x1.3–x3.4 slower than converting a flat frame under pandas 3.

**Relevance to sktime**

- **(a)** sktime does not call `__dataframe__` anywhere. The only mention is the `DtypeKind` naming in `sktime/datatypes/_dtypekind.py:3`. Nothing breaks.
- **(b)** sktime's polars extra is pinned `polars[pandas]>=0.20,<2.0` (`pyproject.toml:89`). The polars 2.0 removals and the `concat(how=)` deprecation that hit sklearn are **still ahead** for sktime.
- **(c)** `__arrow_c_stream__` is the one ingress hook shared by pandas, polars-eager, pyarrow, duckdb and narwhals. A single "anything with `__arrow_c_stream__`" ingress path could replace N per-library `convert_to` edges. It must be paired with a schema (which columns are instance keys / time / values). The stateless VectorizedDF schema can carry exactly this.
- **(d)** Lazy (polars LazyFrame) and dask have **no** capsule, so ingress still needs explicit `collect()` / `compute()`. That has to be a conscious API decision, as in skrub ("please call collect()").
- **(e)** sktime's own polars adapter uses `pl.from_pandas` (`sktime/datatypes/_adapter/polars.py:85–105`). It must keep passing the index explicitly; the default silently drops it, as measured.

---

## Cross-cutting lessons for sktime

1. **Both prior-art libraries separate concerns three ways.**
   - Boundary: containers handled by a small layer.
   - Compute: numpy or Array API.
   - Output: an adapter keyed by config or input type.

   Neither tried to make *estimator internals* dataframe-generic. For sktime this suggests that VectorizedDF plus the `BaseForecaster` boundary is where container abstraction pays off. Forecaster internals keep consuming an inner representation chosen by tag. **Unmeasured/expected:** at least one numpy-native grouped-array inner type would let most forecasters skip pandas without being rewritten.
2. **The Array API rollout is the template for sktime's 100s of forecasters.**
   - A global opt-in "preserve input container" config.
   - A dynamic per-estimator capability tag (sktime's `*_inner_mtype` already lists containers), computed for composites.
   - An automatic test matrix keyed on the tag.
   - A safety-net parity test for everything else.

   Expect slow coverage growth: sklearn reached 9% of estimators in 3.5 years.
3. **Dispatch mechanism cost is negligible; wrapping cost is not.** dict- and singledispatch add ~0.3–0.6 µs per call. narwhals `from_native` adds ~22–47 µs per call.
   - Either choice is fine at the VectorizedDF level, provided frames are wrapped once per call, not once per instance.
   - Per-instance work should run on numpy slices or on pre-wrapped frames.
4. **Semantics need an owner.**
   - Per-backend adapters drift (skrub #2335, reproduced).
   - narwhals harmonises by adopting polars semantics, which changes pandas results (`total_seconds` 1800.5 → 1800.0).
   - Group ordering differs (pandas sorted vs polars unordered).

   Whatever sktime picks, it needs per-primitive cross-backend parity tests.
5. **Index semantics are sktime's unique requirement.** sklearn (positional rows) and skrub (`ignore_index=True`) both sidestep them, so neither provides a precedent for a container-agnostic MultiIndex. The closest workable carriers are:
   - Arrow's pandas metadata (lossless round trip, measured);
   - sktime's own `__index__` column convention;
   - narwhals' `maybe_*_index` helpers (pandas-only escape hatches).
6. **Interop:** standardise ingress/egress on the Arrow PyCapsule. Drop any idea of relying on `__dataframe__` or the DataFrame API Standard. Handle lazy and dask explicitly.

## Caveats

- Timings come from a heavily shared 16-core machine. Absolute µs/ms values vary up to ~2× between runs; within-process ratios are more stable but still noisy. The interop conversion costs in particular should be re-run in the serial phase (`--grid large`).
- Tag census uses default hyperparameters (`_construct_instances`). Conditional tags (PCA, LogisticRegression, composites) can flip with other settings. FrozenEstimator could not be instantiated (SkipTest).
- The skrub measurements run in a separate venv: pandas 3.0.6, numpy 2.5.4 (vs numpy 2.2.6 in `EXP_PY`). The skrubvenv dispatch-overhead run predates the narwhals decomposition cases.
- The static scans (narwhals call sites, `get_namespace` sites, skrub leakage) are AST/regex approximations. Method calls on narwhals objects were counted by reading the source, not by script.
- polars 2.0.0 in the test envs is outside sktime's declared `polars<2.0` bound. Behaviour on older polars (e.g. `__dataframe__` present) differs.
- History and discussion content (GitHub threads, changelog wording) was read through `gh` and web fetches. Positions are paraphrased except where quoted.

## Sources

- **Installed source:**
  - `sklearn/utils/_set_output.py`, `_array_api.py`, `_dataframe.py`, `_indexing.py`, `validation.py`, `_tags.py`, `estimator_checks.py`, `fixes.py`, `_min_dependencies.py`;
  - `compose/_column_transformer.py`, `pipeline.py`, `linear_model/_ridge.py`, `linear_model/_logistic.py`, `decomposition/_pca.py`, `model_selection/_search.py`;
  - `scikit_learn-1.9.1.dist-info/METADATA`;
  - `skrub/_dispatch.py`, `skrub/_dataframe/_common.py`, `skrub-0.11.0.dist-info/METADATA`.
- **sklearn:**
  - [#25896](https://github.com/scikit-learn/scikit-learn/issues/25896), [#31049](https://github.com/scikit-learn/scikit-learn/issues/31049), [#31127](https://github.com/scikit-learn/scikit-learn/pull/31127), [#33789](https://github.com/scikit-learn/scikit-learn/pull/33789), [#34447](https://github.com/scikit-learn/scikit-learn/pull/34447);
  - [1.9 changelog](https://scikit-learn.org/stable/whats_new/v1.9.html), [Array API guide](https://scikit-learn.org/stable/modules/array_api.html), [1.2 release highlights](https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_1_2_0.html).
- **skrub:** [#888](https://github.com/skrub-data/skrub/pull/888), [#1818](https://github.com/skrub-data/skrub/issues/1818), [#2335](https://github.com/skrub-data/skrub/issues/2335), [#1593](https://github.com/skrub-data/skrub/issues/1593).
- **data-apis:** `gh api repos/data-apis/{dataframe-api,dataframe-api-compat,array-api-compat}` (pushed_at dates as above).

## Files

- **Scripts:** `scripts/r1_sklearn_census.py`, `scripts/r1_skrub_census.py`, `scripts/r1_interop_protocols.py`, `scripts/r1_backend_divergence.py`, `scripts/r1_pandas_free.py`, `scripts/r1_dispatch_overhead.py`, `scripts/r1_set_output_ext.py`
- **Prototype:** `protos/r1_sklearn_pyarrow_adapter.py`
- **Results:**
  - `results/r1_sklearn_census.json`, `results/r1_skrub_census.json`;
  - `results/r1_interop_protocols_{pd2,pd3}_dev.{json,txt}`;
  - `results/r1_backend_divergence.json`;
  - `results/r1_pandas_free_{nopandas,exp,skrubvenv}.json`;
  - `results/r1_dispatch_overhead_{exp,skrubvenv}_dev.{json,txt}`;
  - `results/r1_set_output_ext_dev.{json,txt}`.

## Cloud re-run (2026-10-18, 4-core machine, ~2 concurrent agents)

Re-ran `r1_sklearn_census.py`, `r1_set_output_ext.py`, `r1_dispatch_overhead.py` (EXP_PY, `--grid dev`).

- **Census: identical.** 210 estimators / 209 instantiated / 19 `array_api_support` / 78 set_output; 12 static literals; 38 files / 235 `get_namespace` sites; narwhals in 3 files / 20 call sites.
- **pyarrow adapter: identical.** 24 LOC; 10/10 ok; `polars.to_numpy()` is F-ordered.
- **StandardScaler boundary tax** (ratio vs numpy C → default):
  - numpy → polars egress: x1.29 (10k), x1.24 (100k), x1.18 (500k). That is +18–29%, vs +8–52% before.
  - polars input: x0.61–0.77 (100k), x0.44–0.65 (500k).
  - F-ordered numpy: x0.42–0.65.
  - Conclusion unchanged: layout (F order) explains the polars-input speedup.
  - Absolute: numpy → default took 12.6 ms (100k) and 64.7 ms (500k).
- **Dispatch overhead.** Absolute numbers are ~2× faster than the earlier run; the ratios hold.
  - narwhals `from_native`: 10.7–11.1 µs (polars), 17.0–17.8 µs (pandas), vs 22–40 µs earlier.
  - narwhals n_rows: x100–102 vs native.
  - dict-dispatch: x2.2–2.7 (+0.2 µs); singledispatch: x2.6–3.4 (+0.3 µs).
  - pre-wrapped `.shape[0]`: 0.25 / 0.38 µs.
  - sktime `check_is_mtype`: 12.7 µs (polars), 36.4 µs (pandas).
  - `check_array` to_numpy: x3.9 (polars), x10.0 (pandas).
  - **Revised estimate:** wrapping per instance over 10k instances now costs ~0.11–0.17 s per op (vs 0.22–0.47 s earlier). Conclusion unchanged: wrap once per batch.
