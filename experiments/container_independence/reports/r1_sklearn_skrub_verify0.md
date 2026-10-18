# r1_sklearn_skrub_verify0: adversarial verification of r1_sklearn_skrub

Date: 2026-10-18. Cloud machine (4 cores). Lens: open every cited source or URL, check each claim and whether it is still current, and re-run or independently re-implement the experiments.

## Environments

- `EXP_PY` (pandas 2.3.3) and `EXP_PY3` (pandas 3.0.6). Both have polars 2.0.0, narwhals 2.27.1 and sklearn 1.9.1.
- `venvs/r1v0`, my own venv: skrub 0.11.0, sklearn 1.9.1, pandas 3.0.6, polars 2.0.0, narwhals 2.27.1, numpy 2.5.4.
- `venvs/r1v0_nopd`, my own venv: sklearn 1.9.1, polars 2.0.0, narwhals 2.27.1, with pandas and pyarrow **not installed** (checked with `find_spec`).

The original agent's venvs (R1_PY, R1_NOPANDAS) no longer existed, so I rebuilt equivalents and re-ran its skrub, divergence and pandas-free checks in them.

Every command below was run from the WT root with `PYTHONPATH=.:experiments/container_independence`.

## Verdicts

### .1 set_output protocol, 4 methods; pyarrow adapter in 24 LOC, 10/10 OK: **confirmed**

- **Re-run of the original script.** `$EXP_PY -W ignore .../r1_set_output_ext.py --grid dev` printed:
  - `adapter_loc 24`;
  - registered adapters `[pandas, polars, pyarrow]`;
  - all 10 cases `ok: pyarrow.Table`.
- **Independent check:** `scripts/r1_sklearn_skrub_verify0_adapter.py`.
  - The protocol's `def`s, parsed from the source text, are `create_container, is_supported_container, rename_columns, hstack`.
  - The original only checked the type and column names, which is a weak test. I added a value check: the pyarrow output against a plain numpy `fit_transform`, with `np.allclose`. It gave **8/8 equal** (SS, CT, FU, Pipeline × {pandas, polars} input).
- **Caveat:** `ADAPTERS_MANAGER` is private, as the original report already says.

### .2 Array API: 19/210 estimators; 38 files, 235 call sites; 12 static literals: **partially_confirmed**

- **Re-run of `r1_sklearn_census.py`:** identical (210 / 209 / 19, 38 / 235, 12).
- **Independent grep** of the site-packages:
  - `get_namespace(_and_device)?\(` outside `_array_api.py` and tests: 235 sites in 38 files.
  - The bare `get_namespace(` alone gives 142 sites in 30 files, so the "235 get_namespace call sites" figure includes `get_namespace_and_device`.
  - 12 `array_api_support = True` literals (21 assignments in total, the rest being conditional).
- **Independent census** with plain default constructors instead of sklearn's `_construct_instances` test helper (`scripts/r1_sklearn_skrub_verify0_tags.py`):
  - `all_estimators()` returns **208**, not 210. The extra 2, `HalvingGridSearchCV` and `HalvingRandomSearchCV`, appear only because importing the test helper enables experimental halving.
  - Of 183 estimators that can be built with default arguments, **16 leaf estimators** have the tag set.
  - The other 3 of the "19" are FeatureUnion, GridSearchCV and RandomizedSearchCV. Their tag comes from sub-estimators the test helper chose, so they are not "default hyperparameters".
- **Conclusion:** the direction holds (about 8–9%, slow rollout). The precise statement is "16 leaf estimators, plus composites that inherit the tag".

### .3 narwhals is a hard dependency since 1.9 (#31127, merged 2026-05-07, after RFC #31049); 3 files, 20 call sites; a pandas path is kept: **confirmed**

- **Source checks:**
  - METADATA has `Requires-Dist: narwhals>=2.0.1`.
  - `grep` finds narwhals imported in exactly 3 code modules: `compose/_column_transformer.py`, `utils/_indexing.py`, `utils/validation.py`. The other hits are docstrings, `_min_dependencies` and `_show_versions`.
  - I counted the code call sites by hand: 3 + 7 + 10 = 20.
  - `utils/_indexing.py:294–301` has `hasattr(X, "iloc")` → `_pandas_indexing` before the narwhals branch.
  - **Nuance:** the comment there only says that `test_safe_indexing_pandas_no_settingwithcopy_warning` does not pass under narwhals. "Because narwhals objects are immutable" is the agent's own inference.
- **WebFetch of #31127:** title "ENH add narwhals as dependency", merged May 7 2026 by adrinjalali, "replaces any use and mention of `__dataframe__`".
- **WebFetch of #31049:** opened 2025-03-21 and closed as completed through #31127.
  - Poll: dependency got about 13 votes, vendoring 1.
  - The positions of ogrisel, adrinjalali and GaelVaroquaux, the lazy-frame discussion (Marco Gorelli on row order) and glemaitre's 2026-04-20 report about `__dataframe__` all match.
- **Small discrepancy:** the PR text mentions a minimum of 2.20, while the installed METADATA says `>=2.0.1`. This does not matter for the conclusion.

### .5 skrub: 72 generic ops plus 17 local generic functions in 14 modules; pandas is a hard dependency; narwhals declined in #1818: **confirmed**

- **Re-run of `r1_skrub_census.py`** in `venvs/r1v0`: identical numbers (72; specialisations pd.DataFrame 30, pl.DataFrame 26, pl.LazyFrame 13; 14 modules / 17 generic functions).
- **Independent grep:**
  - `^@dispatch` appears 72 times in `_dataframe/_common.py`.
  - 14 other non-test modules contain 17 `@dispatch` functions in total.
  - `_common.py:6` has `import pandas as pd`.
  - METADATA has `Requires-Dist: pandas>=1.5.3`, and polars appears only as an extra.
- **WebFetch of #1818:**
  - Closed as not planned on 2026-07-29.
  - Reasons: a "huge rewrite"; no strong enough reason to move; jeromedockes's point that in-memory compute, not the API, is what blocks duckdb and ibis.
  - FBruzzesi (narwhals maintainer) said narwhals covers almost everything except `pearson_corr`, and that `replace` needs workarounds.
- **Missed nuance:** skrub requires sklearn, which pulls in narwhals as of 1.9. skrub users therefore *have* narwhals installed anyway, so the refusal concerned the rewrite and the semantics, not the dependency.

### .6 Silent backend divergence: #2335 gives pandas [0,1] vs polars [0,0]; narwhals imposes polars semantics; polars group order is [c,a,b]: **partially_confirmed**

- **Re-run of `r1_backend_divergence.py`** in `venvs/r1v0`. It uses the *real* `skrub.SessionEncoder`.
  - Session ids: pandas [0,1] vs polars [0,0], in both cases.
  - `total_seconds`: pandas 1800.5, polars 1800.0, narwhals[pandas] 1800.0. All reproduced.
- **WebFetch of #2335:** opened 2026-10-15 and still open. The cause is `_get_session_column_polars` using `.dt.total_seconds()`. Matches the claim.
- **Refuted detail: polars default `group_by` order.**
  - My re-run of the script printed `[a,b,c]`, not `[c,a,b]`.
  - Six fresh processes running `group_by('k').agg(...)` on `bacbca` gave `[b,a,c] [a,c,b] [b,a,c] [b,c,a] [b,a,c] [a,c,b]`.
  - The order is **non-deterministic** (hash-based). The correct statement is "polars order is unspecified unless `maintain_order=True`", not a fixed order that differs from pandas.
  - This matters even more for sktime: a test can pass by chance.

### .7 Dispatch overhead +0.2–0.3 µs; narwhals `from_native` 11–18 µs; a pre-wrapped op costs 0.25–0.38 µs: **partially_confirmed**

- **Re-run:** `$EXP_PY -W ignore .../r1_dispatch_overhead.py --grid dev --tag verify0`.
  - dict-dispatch +0.27 µs (polars) / +0.27 µs (pandas); singledispatch +0.36 / +0.37 µs.
  - `from_native`: 12.9 µs (polars) / 20.1 µs (pandas).
  - Pre-wrapped `.shape[0]`: 0.38 / 0.59 µs.
  - `check_is_mtype`: 15.0 / 48.1 µs.
- **Independent timeit script** (`scripts/r1_sklearn_skrub_verify0_nwops.py`, different timer, 100-row series):

| frame | native mean | from_native | pre-wrapped `w['c0'].mean()` | pre-wrapped `w.select(nw.col('c0').mean())` | pre-wrapped `get_column().mean()` |
|---|---|---|---|---|---|
| polars | 1.03 µs | 11.7 µs | 14.4 µs (×13.9, +13.3 µs) | 68.0 µs (×66) | 5.3 µs |
| pandas (DatetimeIndex) | 14.0 µs | 20.1 µs | 39.5 µs (×2.8, +25.5 µs) | 258 µs (×18) | 21.8 µs |

- **Methodology flaw:** the "wrap once per batch → about 2.5–4 ms per 10k ops" impact statement rests on a metadata op (`.shape[0]`) only.
- For real compute ops, a **pre-wrapped** narwhals frame still costs **+13 µs (polars) to +25 µs (pandas) per op**. The idiomatic expression API costs +67 µs (polars) to +244 µs (pandas). That is the same order as, or larger than, the wrap itself.
- **Correction:** hoisting `from_native` out of the per-instance loop does *not* make narwhals per-instance compute cheap. Per-instance work must run on numpy slices or on a single grouped expression over the whole panel. The per-call `from_native` numbers themselves are confirmed.

### .8 Polars input runs StandardScaler at 0.44–0.65× because of F-order; egress to polars costs +18–29%: **partially_confirmed**

- **Re-run of `r1_set_output_ext.py --grid dev`**, ratios against C-ordered numpy:
  - polars→default: ×0.54 (100k) and ×0.52 (500k);
  - numpy_F: ×0.38 and ×0.41;
  - numpy→polars: ×1.15 and ×1.27;
  - at 10k, numpy→polars was ×0.90. The egress overhead is noise-level at 10k.
- **Independent check** (`scripts/r1_sklearn_skrub_verify0_layout.py`):
  - Forcing `np.ascontiguousarray(polars.to_numpy())` removes the speedup: ×1.23 / ×1.22 vs polars ×0.50 / ×0.51. **The layout explanation is confirmed.**
- **Missed nuance:** the egress cost depends on layout.
  - `PolarsAdapter.create_container` costs 1.23 ms / 6.26 ms for a C-ordered result, but only 0.17 ms / 0.20 ms for an F-ordered one, which is near zero-copy.
  - With polars input (F-ordered result), polars output is therefore essentially free: the re-run gave polars→polars ×0.51.
  - The "+18–29%" applies only to C-ordered compute output.
- **Also unreported:** at 1k rows, pandas input is ×2.0 and polars input ×1.6 of numpy, because per-call validation dominates. This is the regime that matters for sktime's per-instance loops.

### .9 `__dataframe__` is dead; PyCapsule is the live path; LazyFrame and dask lack it; pandas MultiIndex round-trips losslessly; `pl.from_pandas` drops the index: **confirmed**

- **Re-run of `r1_interop_protocols.py`** in both environments: the protocol-surface table reproduced. That includes `__dataframe_consortium_standard__` on pandas 2.3.3 but not on 3.0.6, and `pd.Series.__arrow_c_stream__` only on pandas 3.
- **Independent script** (`scripts/r1_sklearn_skrub_verify0_interop.py`) actually *calls* the protocols instead of using `hasattr`:
  - **polars 2.0** raises `AttributeRemovedError: __dataframe__ was removed in version 2.0`.
  - **pandas 3.0.6** emits `Pandas4Warning: The Dataframe Interchange Protocol is deprecated`. pandas 2.3.3 gives no warning.
  - **Capsule round trip:** `RecordBatchReader.from_stream(pd_mi).read_all().to_pandas()` equals the original, including MultiIndex names `[h0, time]`. The columns are `[c0, h0, time]`.
  - **Polars conversion:** `pl.from_pandas` defaults to `[c0]`, and gives `[h0, time, c0]` with `include_index=True`.
  - **No capsule:** `pa.table(LazyFrame)` and `pa.table(dask)` raise TypeError, and a narwhals LazyFrame has no capsule.
  - **duckdb:** a relation exports through the capsule.
  - **pandas 3:** has `pd.DataFrame.from_arrow`; pandas 2.3.3 does not.
- **Not re-checked:** the dates of the data-apis repositories. `gh api` is blocked in this session (HTTP 403), so the "dormant since 2024-04" figure is **unverifiable** here.
- **Nuance:** the capsule is lossless only for consumers that read pandas metadata (pyarrow→pandas). For polars and other consumers the index arrives as plain trailing columns, so the "losslessness" really lives in the schema metadata.

### Not in the 8, also checked: .4 pandas-free sklearn: **confirmed**

- **Rebuilt** a pandas- and pyarrow-free venv (`venvs/r1v0_nopd`) and ran `scratchpad/r1v0_nopd_check.py` with `-I`.
- **Results:**
  - StandardScaler with polars output: ok.
  - ColumnTransformer: ok, columns `['a__c0','a__c1','b__c2']`.
  - Ridge: ok, `(200,)`.
  - train_test_split: ok, `(150,3)`.
  - StandardScaler on a LazyFrame: `ValueError: setting an array element with a sequence.`
  - pandas was never loaded.

## Missed or wrong in the original report

1. **Pre-wrapped narwhals compute ops are expensive.**
   - +13 µs (polars) / +25 µs (pandas) per `mean`; ×66 / ×18 through the expression API.
   - The "wrap once per batch" recommendation is necessary but not sufficient. Only metadata ops become cheap.
2. **Polars `group_by` order is non-deterministic, not `[c,a,b]`.** Any adapter must force `maintain_order=True` (or sort explicitly) to reproduce `groupby(level, sort=False)`.
3. **The Array API "19" mixes 16 leaf estimators with 3 composites** whose tag depends on the test helper's sub-estimators. The 210 is inflated by 2 because the test helper enables experimental halving search (the plain count is 208).
4. **Egress to polars is free when the result is F-ordered**, which it is when the input was polars. The "+18–29%" applies only to C-ordered results.
5. **Small-size regime is not surfaced.** At 1k rows, container validation costs ×1.6–2.0 for a numpy-compute estimator. This is the per-instance regime of sktime's vectorized loops.
6. **The body of the report is inconsistent with its own cloud re-run.** The main sections still say 22–47 µs and +30–52%, while the summary uses 11–18 µs and +18–29%.
7. **skrub declining narwhals is not a dependency decision.** skrub users on sklearn ≥1.9 already get narwhals transitively.
8. **C1 is untested.** The report does not test narwhals `collect()` routing (dask→pandas, duckdb→arrow), even though it is directly relevant to the contested claim C1. This is out of r1's scope, but the cross-cutting lessons lean on narwhals without checking it.

## Files

- **Scripts:**
  - `scripts/r1_sklearn_skrub_verify0_tags.py`
  - `scripts/r1_sklearn_skrub_verify0_layout.py`
  - `scripts/r1_sklearn_skrub_verify0_nwops.py`
  - `scripts/r1_sklearn_skrub_verify0_interop.py`
  - `scripts/r1_sklearn_skrub_verify0_adapter.py`
- **Scratch:** `scratchpad/r1v0_nopd_check.py`, outside the repo.
- **Note:** `r1_dispatch_overhead.py --tag verify0` wrote `results/r1_dispatch_overhead_verify0_dev.*`. `r1_set_output_ext.py` re-run overwrote `results/r1_set_output_ext_dev.json` with the numbers quoted above.
