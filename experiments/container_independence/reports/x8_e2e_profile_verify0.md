# x8_e2e_profile_verify0: adversarial check of x8_e2e_profile (methodology lens)

**Lens.** I re-profiled with methods different from x8's monkeypatched exclusive-time wrappers:
- (a) a statistical **sampling profiler**: a background thread snapshots the main thread's stack every
  ~1 ms via `sys._current_frames()`, and stack-precedence rules map each sample to a bucket;
- (b) **subtraction/ablation** timings, interleaved in one process with min-of-reps;
- (c) **call counts via `sys.setprofile`** instead of cProfile.

I also re-ran the original scripts at reduced size.

All commands were run from the WT root with
`PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore …`, where `EXP_PY` is pandas 2.3.3,
polars 2.0.0 and sklearn 1.9.1, on a shared 4-core box.

## Scripts and outputs

| what | command (script under experiments/container_independence/scripts/) | output (results/) |
|---|---|---|
| re-run of the original, n ≤ 1000 | `x8_e2e_profile.py --grid dev --max-n 1000 --containers pd,pl,dask --cprofile-max-n 0 --out …/x8_e2e_profile_verify0_rerun.json` | `x8_e2e_profile_verify0_rerun.json` |
| sampling profiler + subtraction + ablation + strict equality + dask | `x8_e2e_profile_verify0_sampling.py --grid dev --reps 3` (n = 200, 1000) | `x8_e2e_profile_verify0_sampling_dev.json` |
| fairness of the pandas T3 tier; warnings recount | `x8_e2e_profile_verify0_tiers.py --grid dev` (n = 1000, 5000, 20000) | `x8_e2e_profile_verify0_tiers_dev.json` |
| call counts via sys.setprofile | `x8_e2e_profile_verify0_counts.py --n 100` | `x8_e2e_profile_verify0_counts.json` |
| re-run of the side checks | `x8_side_checks.py --grid dev --out …/x8_e2e_profile_verify0_side_checks_rerun.json` | same |
| per-slice check/convert cost, pd vs pl | `x8_e2e_profile_verify0_slicecheck.py` | `x8_e2e_profile_verify0_slicecheck.json` |

## Raw numbers

**Re-run of the original wrappers (n=1000, T=100).** Columns are percent of instrumented time.

| case | wall pd / pl / dask (s) | pl/pd | container_once pl | per-instance framework pd / pl | pi_boilerplate pd / pl | pd/T1 | pd/T3pd | pl/T3pl | T3pd/T3pl |
|---|---|---|---|---|---|---|---|---|---|
| Panel naive | 11.07 / 11.69 / 13.50 | 1.06 | 3.4% | 31.8 / 35.7 | 26.0 / 29.3 | 1.29 | 576 | 1336 | 2.2 |
| Panel poly | 8.67 / 8.91 / 10.55 | 1.03 | 5.1% | 57.8 / 55.6 | 48.6 / 46.4 | 2.23 | 201 | 489 | 2.36 |
| Hier naive | 12.22 / 12.33 / 13.82 | 1.01 | 4.5% | 38.5 / 33.0 | 26.2 / 26.5 | 1.25 | 404 | 1562 | 3.83 |
| Hier poly | 10.02 / 9.40 / 11.42 | 0.94 | 5.9% | 59.6 / 55.0 | 42.8 / 45.9 | 2.45 | 143 | 598 | 4.46 |

- Every validity check is True.
- instr/wall is 0.92–1.12.

**Sampling profiler (Panel).** Rollups are percent of samples.

| case | samples | container_once pd / pl | of which pl monotonic loop | per-instance framework pd / pl | inner total pd / pl |
|---|---|---|---|---|---|
| naive n=200 | 1065 | 0.7 / 5.7 | 4.3 | 35.9 / 34.6 | 62.7 / 59.1 |
| naive n=1000 | 5700 | 0.4 / 5.5 | 4.9 | 31.2 / 34.5 | 67.9 / 59.5 |
| poly n=200 | 747 | 0.8 / 5.7 | 4.5 | 50.3 / 47.3 | 48.1 / 46.1 |
| poly n=1000 | 4194 | 0.6 / 5.5 | 4.8 | 52.5 / 49.6 | 46.4 / 44.4 |

- For naive at n=1000 (pd), the nested Lag work splits into nested boilerplate 20.5 % and nested core
  17.9 %, so 38 % in total. That matches x8's ~38 % (12.48 s + 11.59 s of 63.9 s).
- **Sampler caveat.** The sampler thread can only take a sample once the main thread releases the GIL.
  Code that releases the GIL often (polars calls inside the monotonic loop) is therefore over-sampled.
  The 4.3–4.9 % monotonic share is an upper estimate. The subtraction numbers below are lower.

**Subtraction (uninstrumented, interleaved, min of 3).**
`pl_vecmono` is the polars run with `is_monotonically_increasing` swapped in-process for a vectorized
group_by version.

| case | pd s | pl s | pl_vecmono s | pl/pd | saving from vectorized monotonic, % of pl |
|---|---|---|---|---|---|
| naive 200 | 2.270 | 2.396 | 2.233 | 1.055 | 6.8 |
| naive 1000 | 11.32 | 11.44 | 11.33 | 1.011 | 1.0 |
| poly 200 | 1.654 | 1.693 | 1.597 | 1.023 | 5.7 |
| poly 1000 | 8.34 | 8.24 | 8.00 | 0.988 | 2.9 |

At n=1000, which conversion path is used matters more than the conversion itself:

| operation (n=1000) | seconds |
|---|---|
| `convert_to(pl → pd-multiindex)` | 0.1437 |
| `convert(pl, "polars_panel", "pd-multiindex")`, from_type given | 0.0139 (10x less) |
| monotonic check, Python loop | 0.120 |
| monotonic check, vectorized (agrees on this sorted input) | 0.011 |

**Ablation of per-instance boilerplate (pandas input, interleaved, min of 3).**
"Full single-series loop" means a user-written loop of `est.clone().fit(s).predict()` per series.

| case | vectorized sktime | full single-series loop | T1 lite loop | vectorized / loop | vectorized / lite | slicing alone % |
|---|---|---|---|---|---|---|
| naive 200 | 2.225 | 2.211 | 1.676 | 1.01 | 1.33 | 2.7 |
| naive 1000 | 11.50 | 11.92 | 9.16 | 0.96 | 1.26 | 2.6 |
| poly 200 | 1.539 | 1.633 | 0.765 | 0.94 | 2.01 | 4.4 |
| poly 1000 | 8.18 | 8.76 | 4.08 | 0.93 | 2.01 | 3.8 |

All outputs are equal to the vectorized result under a strict check: instance keys, index names and
values.

**T3 fairness (claim 9).** Times are seconds. The ratio columns divide each pandas time by polars eager.

| n | est | x8 pandas T3 | leaner pandas groupby | polars eager | polars lazy | x8pd/pl | lean/pl |
|---|---|---|---|---|---|---|---|
| 1000 | naive | 0.0237 | 0.0220 | 0.0072 | 0.0085 | 3.27 | 3.04 |
| 1000 | poly | 0.0464 | 0.0374 | 0.0166 | 0.0150 | 2.79 | 2.25 |
| 5000 | naive | 0.0989 | 0.0813 | 0.0312 | 0.0334 | 3.17 | 2.61 |
| 5000 | poly | 0.2115 | 0.1921 | 0.0705 | 0.0662 | 3.00 | 2.73 |
| 20000 | naive | 0.567 | 0.270 | 0.119 | 0.123 | 4.76 | 2.27 |
| 20000 | poly | 1.107 | 0.579 | 0.274 | 0.250 | 4.04 | 2.11 |

The lean pandas version matches x8's pandas tier exactly: same keys, max diff ≤ 3.4e-10.

**Side checks re-run.**

| n | check pd (s) | check pl (s) | pl/pd | pl monotonic loop (s) | convert_to pl→pd (s) | slicing per group (µs) |
|---|---|---|---|---|---|---|
| 1000 | 0.00142 | 0.1223 | — | 0.119 | 0.144 | 121 |
| 20000 | 0.00401 | 2.623 | **654x** | 2.786 | 2.951 | 120 |

x8 reported 85–96 µs per group for slicing; this re-run gives 120–136 µs.

**Call counts with sys.setprofile (n=100), per instance.** The counts are identical for pd and pl input.

| forecaster | convert_to | mtype | check_is_mtype | check_is_scitype | get_cutoff | clone |
|---|---|---|---|---|---|---|
| naive | 8.04 | 6.05 | 60.46 | 1.01 | 2.02 | 1.0 |
| poly | 5.04 | 4.05 | 38.46 | 1.01 | 2.02 | 1.0 |

x8_callers reported 8.0 / 6.0 / 60.2 (naive) and 5 / 4 / 38.2 (poly), so the counts are reproduced.

**Warnings.**
- Stderr contains 49 `FutureWarning` lines for n=50 and 199 for n=200.
- The string `remember_data` appears 147 and 597 times.
- `catch_warnings(record=True)` with n=20 records 19 warnings.
- The filter left on top afterwards is `('default', None, FutureWarning)`.
- Source: `sktime/utils/warnings.py:64-65`, where `__exit__` calls `filterwarnings("default", category=…)`.

**Per-slice cost if slices stayed polars (single series, T=100).**

| operation | µs |
|---|---|
| `check_is_scitype`, pandas | 206 |
| `check_is_scitype`, polars | 330 |
| `convert_to`, pandas no-op | 225 |
| `convert_to`, polars → pandas | 1281 |

**dask, unpatched.**
- `check_is_scitype(dask_panel)` returns False.
- `NaiveForecaster().fit(dask)` raises `TypeError: Unsupported input data type`.
- Source: tag `python_type="dask.dataframe"` at `_panel/_check.py:1024`. `_precheck` compares the
  class name `"dataframe"` against `DataFrame`.

## Verdicts

- **x8.1 (pl ≈ pd, 0.92–1.11): confirmed.**
  - Re-run at n=1000: pl/pd 0.94–1.06.
  - Independent interleaved run: 1.01 / 0.99.
  - Cost per instance is 8.2–12.2 ms.
- **x8.2 (container cost 2.7–4.8 % for pl; conversion proper 0.1–0.3 %): partially confirmed.**
  - The wrapper re-run gives 3.4–5.9 % at n=1000, slightly above the stated range.
  - Sampling gives 5.5 %, an upper estimate (GIL bias).
  - Subtraction gives pl−pd of −1 to +1 % at n=1000. A vectorized monotonic check saves 1.0–2.9 %.
  - Conversion proper is 0.1–0.3 %, confirmed.
  - Order of magnitude (≤ ~6 %) holds. The upper bound should read ~6 %, not 4.8 %.
- **x8.3 (pl check ~400x pd, mostly the monotonic loop, ~75 % of convert_to): partially confirmed.**
  - The ratio was 654x in this run, not 400x: it is load-dependent, and the pandas check is only 4 ms.
  - The monotonic loop is essentially all of the polars check.
  - mtype inference is about 90 % of `convert_to(pl→pd)`: `convert` with an explicit from_type is 10x
    cheaper. The loop alone is 83 %.
  - Wording error: `convert_to` has **no `from_type` parameter**. The fix is to call
    `convert(obj, from_type, to_type)` or reuse an already-inferred mtype. x8's "pass from_type in
    convert_to" is not an available API.
- **x8.4 (per-instance framework 33–44 % naive, 53–61 % poly; boilerplate 26–28 / 42–48 %):
  partially confirmed.**
  - Wrapper re-run: naive 31.8–38.5 %, poly 55.0–61.6 %; boilerplate 24.4–29.3 % and 40.4–48.6 %.
  - The independent sampler agrees for naive (31–36 %) but gives poly **47–53 %**, 5–8 points lower.
    Attribution near the boundary of sklearn/inner code depends on the method.
  - The qualitative split is robust.
- **x8.7 (T1 lite loop 1.09–1.67x naive, 1.66–2.42x poly): confirmed.**
  - Re-run: 1.25–1.29 naive, 2.23–2.45 poly at n=1000.
  - Independent: 1.26–1.33 naive, 2.01 poly.
  - Hier poly 2.45 is marginally above the stated maximum.
- **x8.8 (vectorized tiers 130–750x pd; polars 650–1800x; outputs match): partially confirmed.**
  - pd/T3pd re-run: 143–576. Matches.
  - pl/T3pl for poly at n=1000 was **489–598**, below the stated 650 floor. naive was 1336–1562.
  - Output equality was confirmed with a stricter comparison (instance keys and names) for naive.
    x8's own `compare` checks only the values and the time level, not instance keys.
- **x8.9 (polars T3 2.3–5.4x faster than pandas groupby): partially confirmed.**
  - Against x8's pandas code the gap is 2.2–4.8x.
  - Against a leaner but equivalent pandas groupby it shrinks to **2.1–3.0x**.
  - At n=20000, x8's pandas tier is 2.1x slower than the lean one. The upper end of the range is partly
    a pandas-implementation artefact.
  - lazy ≈ eager is confirmed (±15 %).
- **x8.11 (~3 FutureWarnings per instance; 597 for n=200; 141 kB): partially confirmed.**
  - The mechanism and the 141 kB are confirmed.
  - The count is wrong: **about 1 warning per instance** (49 for n=50, 199 for n=200, 19 for n=20).
  - The 147/597 figures count occurrences of the string `remember_data`, which appears 3 times per message.

Also spot-confirmed, not counted among the 8:
- x8.5: call counts reproduced with sys.setprofile.
- x8.6: `_naive.py:358-382` builds a `Lag` and calls `fit_transform` per series. The sampler gives the
  nested Lag share as ~38 %.
- x8.10: unpatched dask is rejected.

## Things x8 missed or got wrong

1. **The vectorization layer adds nothing over a user's own loop.** Vectorized sktime takes 0.93–1.01x the
   time of a hand-written loop of full single-series `clone().fit().predict()`. All the cost is running
   one BaseForecaster (plus its nested transformer) per series. Optimizing VectorizedDF iteration,
   reconstruct or schema handling cannot help: slicing alone is 2.6–4.4 %.
2. **The "≤1.05x" bound holds only for the convert-once design.** Suppose an adapter or narwhals path kept
   per-instance BaseForecaster but left the slices in polars while forecasters stay pandas-inner. Each
   slice then pays 1.28 ms per `convert_to` pl→pd (vs 0.23 ms for a pandas no-op). The polars check is
   1.6x the pandas one. At 5–8 convert calls per instance, that is an expected **regression** of roughly
   +50–80 % (unmeasured end-to-end, expected from the per-slice micro-bench).
3. **`convert_to` has no `from_type`.** The cheap fix is `convert(obj, from_type, to_type)`, which is
   10x cheaper here.
4. **Profiler methodology.** The wrapper and sampling methods agree to within ~5 points for naive.
   They diverge for poly in how per-instance checks are separated from sklearn work, so the
   boilerplate share is method-dependent at about ±5 points.
5. **The T3 pandas baseline is not tuned.** The "polars beats pandas by up to 5.4x" figure overstates the
   gap at large n. A fair figure is ~2–3x.
6. The vectorized monotonic replacement was only checked on sorted input; unsorted and duplicate cases
   were not tested.
7. Only n ≤ 1000 was re-run end-to-end (n ≤ 20000 for the tiers and side checks). x8's n=5000 numbers
   are single-rep and were not reproduced here.

## Bottom line

x8's architectural conclusion survives every method I tried:
- Container decoupling alone is worth a few percent at most.
- Per-instance BaseForecaster plus nested-transformer overhead dominates.
- Only panel-native, vectorized implementations deliver order-of-magnitude gains.

Several numbers need correcting:
- polars container share is up to ~6 %, not 4.8 %;
- the pl/pd check ratio is load-dependent (400–650x);
- poly pl/T3 is ~500x at n=1000;
- the polars-vs-pandas T3 gap is ~2–3x;
- there is ~1 warning per instance, not 3.

Two points were missing:
- per-instance native-container dispatch would likely be slower than today;
- `convert_to` has no `from_type` argument.
