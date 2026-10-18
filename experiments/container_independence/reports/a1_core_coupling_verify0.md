# a1_core_coupling: adversarial verification (verify0)

Lens: independent recount and file:line spot-checks. I verified 8 of the 13 claims and re-ran 3 more (a1.9, a1.13, a1.6-adjacent).
Environment: `EXP_PY` (pandas 2.3.3, polars 2.0.0), plus `EXP_PY3` (pandas 3.0.6) for the behavioural probes. 4-core cloud machine shared with other agents, so I compare ratios measured inside one process rather than absolute times.

All commands run from the WT root `/home/user/sktime` with
`PYTHONPATH=.:experiments/container_independence[:experiments/container_independence/protos] $EXP_PY -W ignore <script>`.

New files:
- `scripts/a1_core_coupling_verify0_probes.py` → `results/a1_core_coupling_verify0_probes.json` (pandas 2), `..._probes_pd3.json` (pandas 3)
- `scripts/a1_core_coupling_verify0_census.py` → `results/a1_core_coupling_verify0_census.json`
- `scripts/a1_core_coupling_verify0_overhead.py --grid {dev,large}` → `results/a1_core_coupling_verify0_overhead_dev.json`
- `scripts/a1_core_coupling_verify0_grid_masked.py` → `results/a1_core_coupling_verify0_grid_masked.json`

I also re-ran the original `a1_static_touchpoints.py`, `a1_grid_bench.py --grid dev`, `a1_grid_metrics_probe.py` and `a1_cutoff_bench.py --grid dev`.

## Verdicts

| claim | verdict | summary |
|---|---|---|
| a1.1 census 229 / 170 lines / 46 APIs | partially_confirmed | Reproduces exactly within its own scope. But 32 of the 229 are not pandas API calls, and the scope leaves out several boundary files. |
| a1.4 API contracts, mixed output types for polars | confirmed | Independent probe gives the same result on pandas 2 and 3. C3 is undercounted. |
| a1.5 overhead shares | partially_confirmed | Container share reproduces (15%). Re-entry is 52–63% by subtraction, not 78–84%. |
| a1.7 EstimatorGrid speedups | confirmed | Re-run: build 3–77×, consume 24–355×. The ranges shift but the conclusion holds. |
| a1.8 pytest 0 new behavioural failures | confirmed | Counts reproduce from the saved logs. The assertions that the type check masked also pass (26/26). |
| a1.10 polars X/y check silently passes | confirmed | Independent probe, different forecaster; pandas 2 and 3. |
| a1.11 column "X" breaks fit | confirmed, and the scope is wider than claimed | Also breaks non-vectorized, panel-native forecasters. A second duck-typing site exists. |
| a1.12 update_predict / predict_residuals with polars | confirmed | Same error lines; pandas 2 and 3. |

### a1.1: census recount (partially_confirmed)

**Re-run.** `a1_static_touchpoints.py` prints the same numbers: `touchpoints: 229 false positives: 6 unclassified: 0`, `distinct canonical pandas API: 46`, and identical by-file and by-category counts.

**Line accuracy.** `a1_core_coupling_verify0_census.py` checks that each row's token appears on the cited source line.
- Result: `rows parsed: 229  token not found on cited line: 4`.
- All 4 are `_utilities.py:316` (`.groupby/.nth/.agg/.iloc`). That is a multi-line method chain whose AST start line is 316, so the line is acceptable.
- Manual spot-checks also match: `_utilities.py:87-90`, `221-222`, `235`, `372-375`; `_base.py:1715`, `1728`, `1735`, `2073`.

**Problems with the headline number.**
1. **32 of the 229 "pandas touchpoints" are not pandas API calls.** The script's own output lists them under `non-pandas-API touchpoints`: mtype strings 22, `hasattr(..,'X')` 2, `convert_to` 3, `check_equal_time_index`, `len`, `'forecasters.loc[...]'`, `_shift` and `cv.split(y)`. "229 pandas touchpoints" is therefore really "197 pandas-API sites + 32 pandas-coupled sites".
2. **Scope misses boundary code that does the same job.** I counted pandas regex hits in code lines only (docstrings and comments excluded):
   - `sktime/forecasting/base/_base_proba.py`: **30 hits**. `BaseProbaForecaster._vectorize_predict_proba` and `_concat_distributions_hierarchical` form a *second vectorization/reconstruct path* for `predict_proba`, with its own concat and MultiIndex handling. a1 does not mention it.
   - `sktime/base/_proba/_mixin.py`: **12 hits**. These are the default `_predict_interval`, `_predict_quantiles`, `_predict_var` and `_predict_proba`, which build the C3 output columns when a forecaster implements only one of them. a1's C3 count (2 touchpoints) only covers `_base.py:2797/2805`.
   - `sktime/forecasting/base/_fh.py`: **102 hits**. This is the C2 contract (fh is pandas-index based); `_check_fh` delegates to it. a1 classes C2 as 11 touchpoints.
   - `_utilities.py` `get_window` / `get_slice`: 14 hits. These are out of scope, but they are datatypes utilities that some forecasters use for windowing.

   These are syntactic regex counts, not classified touchpoints. They show that the 229/46 figure depends heavily on scope. The claim to have mapped *every* touchpoint in the vectorization layer is not met for the proba path. C2 and C3 are undercounted by roughly an order of magnitude if `_fh.py` and the proba mixin are counted.

### a1.4: contracts and mixed output types (confirmed)

I wrote a separate probe using `NaiveForecaster(strategy="mean")`, a 4×20 Panel and `convert_to(..., "polars_panel")`. Output:
- `predict`: `polars.DataFrame`
- `predict_interval`, `predict_quantiles`, `predict_var`: `pandas.DataFrame`
- `cutoff`: `DatetimeIndex`
- `forecasters_`: `pandas.DataFrame`
- fitted-parameter keys: `"forecasters.loc['s0','forecasters']"`

The result is identical on pandas 3.0.6. As noted under a1.1, C3 understates the output-construction code: the `_mixin.py` defaults also build pandas MultiIndex columns.

### a1.5: overhead shares (partially_confirmed)

**Re-run of a1's instrumented script:** not repeated (it takes several minutes). I relied on the JSON shares and the method instead.

**Independent method.** `a1_core_coupling_verify0_overhead.py` uses subtraction with no wrappers. All variants are interleaved and min-of-repeats is taken, for `NaiveForecaster(strategy="last").fit` on a Panel.

```
== n_series=100 n_time=100           == n_series=1000 n_time=100
 total          0.3530s               total          3.7033s
 outer_check    0.0010s               outer_check    0.0011s
 prepare        0.0024s               prepare        0.0090s
 slices         0.0505s               slices         0.5591s
 trusted        0.0437s               trusted        0.4865s
 pub_per_slice  0.2661s               pub_per_slice  2.4181s
 container (prepare+slices)  15.0%    15.3%
 re-entry (pub - trusted)    63.0%    52.2%
 trusted inner (clone+_fit)  12.4%    13.1%
 residual (pivot/glue)        9.3%    19.3%
 pub-per-slice / trusted = 6.1x       5.0x
```

**Findings.**
- The container share (~15%) matches a1's 15.8–16.6%.
- Per-slice re-entry measured this way is **52–63%**, against a1's 77.6–83.7% for naive.fit. a1's "re-entry" group also contains the `public-glue` bucket. That bucket includes the *outer* `fit`, `_vectorize` and `_set_cutoff_from_y` glue and part of the pivot bookkeeping, so it is not purely per-slice.
- My "inner" share (clone + FH construction + `_fit`, 12–13%) is far above a1's 0.3%. a1's inner bucket counts only `_fit`, and the clone and FH costs land in "re-entry".
- The outer whole-panel `check_is_scitype` is only about 1 ms, so it does not explain the gap.

**Direction holds.** Re-entry is still the largest bucket, and the container layer is about 15%.

**New quantification of the "unmeasured/expected" trusted path.** Calling clone + `_fit` directly on pre-made slices is **5.0–6.1×** faster than per-slice public `fit`. That would cut total vectorized fit by about 52–63%: roughly 2.1–2.7× end-to-end, before any backend change.

### a1.7: EstimatorGrid build/consume (confirmed)

Re-run of `a1_grid_bench.py --grid dev`:
- build ratios 3–77× (claimed 4–82×)
- consume ratios 24–355× (claimed 26–314×)
- at 10k series, hierarchical: build 26.3 ms vs 2.8 ms; consume 696 ms vs 1.96 ms

The script asserts that cells, keys and order are equal.

Caveat: the "consume" side (`iterrows` + a `.loc` loop) reached 0.70 s at 10k hierarchical series in this run, against 0.48 s claimed. For comparison, vectorized fit is about 3.7 ms per instance, so about 37 s at 10k series (extrapolated, not measured at 10k). That puts consume at roughly 2%, not "≲1%". Still minor, so the conclusion holds.

### a1.8: pytest under the grid (confirmed, gap closed)

Recount from the saved logs:
- `701 failed / 3821 passed` (off) and `2813 / 1709` (strict and compat).
- `comm` of the FAILED ids: 2112 added, all `test_vectorize.py::test_vectorize_est`; 0 removed.
- 2112 occurrences of `test_vectorize.py:535`.

**Methodology flaw.** Line 535 (`assert isinstance(result, pd.DataFrame)`) is the *first* assertion. The shape and "all cells are fitted NaiveForecaster" assertions after it never ran, so "type contract only, not behaviour" was not shown by the test run.

**Gap closed.** `a1_core_coupling_verify0_grid_masked.py` replays the test's fixture loop: Panel pd-multiindex/numpy3D/df-list and Hierarchical, iterate_as Series/Panel, iterate_cols on and off. It checks the masked assertions on the strict grid plus key equality with the pivot DataFrame. Result: **26/26 cases ok**. The claim stands.

The pytests were not re-run (about 25 min each).

### a1.10: polars X/y containment (confirmed)

Independent probe:
- Setup: `make_reduction(LinearRegression(), strategy="recursive")` (tags `X-y-must-have-same-index=True`, `capability:exogenous=True`), y dated 2001 and X dated 2010 (disjoint).
- pandas: `ValueError ... not contained` at `utils/validation/series.py:283`.
- polars `pl.DataFrame`: fit completes with no error.
- Same result on pandas 3.

Cause confirmed at `_utilities.py:87-90`: `hasattr(obj,"index")`, otherwise `RangeIndex(0, n)`. Both polars inputs get identical RangeIndexes.

### a1.11: column named "X" (confirmed; broader than claimed)

- A Panel with column `"x"` fits; with `"X"` it gives `TypeError: No valid mtype ...` for both `NaiveForecaster` and `PolynomialTrendForecaster`.
- A Series-scitype frame with column `"X"` is fine, because `obj.X` is still a Series with the same index.
- **Missed scope:** a *non-vectorized* forecaster with `y_inner_mtype="pd-multiindex"` fails the same way. The bug is in `get_cutoff` (`_set_cutoff_from_y`), not in vectorization.
- **Missed site:** `get_time_index` has the same duck typing at `_utilities.py:59` (`elif hasattr(X, "X")`).
- I confirmed that `VectorizedDF` no longer has an `X` attribute (grep of `_vectorize.py`: no `self.X` or `def X`).
- Reproduces on pandas 3.

### a1.12: polars update_predict / predict_residuals (confirmed)

Using PolynomialTrendForecaster (a different forecaster from a1's probe):
- `update_predict(polars)` gives `ValueError: y_preds must be a list of pd.Series or pd.DataFrame` at `_base.py:2838`.
- `predict_residuals(polars)` gives `TypeError: y must be a supported Series mtype` at `_base.py:1728`.
- The pandas equivalents work.
- Same on pandas 3.

### Re-runs outside the 8 verdicts

**a1.9 metrics probe:** reproduced exactly. off 4 ok; strict and compat 4 FAIL each, with the same error messages.

**a1.13 cutoff bench:**

```
    10x100  sktime=0.0036s  pd_vec=0.696ms  pl=0.268ms   sk/pdvec=5x  sk/pl=14x
  1000x100  sktime=0.0183s  pd_vec=1.263ms  pl=0.590ms   sk/pdvec=15x sk/pl=31x
 10000x50   sktime=0.0660s  pd_vec=3.848ms  pl=0.729ms   sk/pdvec=17x sk/pl=90x
```

Ratios are 5–17× and 14–90×, against the claimed 6–17× and 13–81×.

The polars timing is asymmetric: data is pre-converted, and freq inference and Index construction are skipped. The 5–17× pandas-vectorized gain needs no backend change. Polars adds only about 2–5× more on top, and only for frames already in polars. In absolute terms it is ≤66 ms at 500k rows, so it is irrelevant next to the per-slice `get_cutoff` calls (a1 itself says so).

## Things missed or wrong

1. **The proba vectorization path is unmapped.** `BaseProbaForecaster._vectorize_predict_proba` and `_concat_distributions_hierarchical` in `forecasting/base/_base_proba.py` (30 pandas hits) are a second iterate/reconstruct implementation. Any backend protocol must cover them too: P8 would need a "concat distributions" variant.
2. **C2 and C3 are undercounted.** C2 shows 11 touchpoints but `_fh.py` has about 102 pandas hits. C3 shows 2 but the default `_predict_interval/_quantiles/_var/_proba` conversions in `base/_proba/_mixin.py` add about 12.
3. **"229 pandas touchpoints" includes 32 non-API sites** (mtype strings, helper names), which inflates the headline.
4. **The re-entry share is overstated.** a1's 78–84% includes outer glue. Subtraction gives 52–63%. The trusted inner path is now quantified: per-slice speedup 5–6×, about 2.1–2.7× end-to-end on vectorized NaiveForecaster.fit.
5. **The column-"X" bug is not vectorization-specific**: it affects any Panel/Hierarchical `y` through `get_cutoff`. There is also a second stale `hasattr(X,"X")` site in `get_time_index` (`_utilities.py:59`).
6. **test_vectorize_est stops at its first assertion.** a1's "only the type contract fails" conclusion was not shown by the pytest run. The direct replay here closes that gap (26/26).

## Caveats

- Regex hit counts for out-of-scope files are syntactic upper bounds (they include `.index`/`.columns` on non-pandas objects). They show order of magnitude, not a classified census.
- Subtractive timings ran under concurrent load (2 repeats at n=1000). The "slices" variant includes `pd.infer_freq` per slice, as `_enforce_index_freq` does.
- I did not re-run a1.6 (dynamic trace) or the full pytest suites.
