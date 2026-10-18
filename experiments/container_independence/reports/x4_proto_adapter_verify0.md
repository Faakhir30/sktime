# x4_proto_adapter — adversarial verification (verify0)

Lens: correctness. I re-ran the original equivalence tests, added adversarial cases, re-ran the forecaster and VectorizedDF benchmarks, and wrote three independent checks that use different code paths. All runs were on the shared 4-core cloud box, so timings are min-of-repeats and should be read as ratios.

Env: EXP_PY (pandas 2.3.3, polars 2.0.0) unless marked pd3 (EXP_PY3, pandas 3.0.6).
Prefix used below: `PP="PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos"`, run from the worktree root.

## Verdict summary

| claim | verdict | one line |
|---|---|---|
| x4.1 pandas adapter == current, 78/78 | **confirmed** | 78/78 re-run on pd2 and pd3. Adversarial pandas cases 28/28 identical. |
| x4.2 polars native, 0 conversions, 26/26 | **partially_confirmed** | 26/26 and 0 conversions reproduce. Adversarial polars 26/28: unnamed index levels get the wrong time-level name in slices. Empty frame fails, but current sktime fails the same way. |
| x4.3 pandas overhead noise; polars iterate 1.4–32x, total 2.5–3.3x | **confirmed** | Re-run: P new/cur 1.00–1.05; L iterate 1.4–43.5x; L total 2.2–3.7x. |
| x4.4 per-slice convert 1.5–4.3x, "convert-once is correct" | **partially_confirmed** (inference refuted) | Ratio reproduces (1.4–3.0x). An independent hand-written per-slice converter is 0.08–0.12x of convert-once, so the penalty comes from sktime `convert_to` overhead, not from slicing. |
| x4.5 A2 fit 0.98–1.31x of A1, never faster | **confirmed** | A2/A1 fit 1.07 / 1.17 / 1.16 at n=10/100/1000. Conversions 1000+1000 at n=1000. |
| x4.6 polars-inner toy: 0 conversions, fit 0.36–0.43x | **partially_confirmed** | Counts and B2/B1 ratios reproduce (0.51/0.36/0.37). B1 is a strawman baseline, though: against today's pandas path (A0), B2 fit is 1.04x / 0.96x / 1.17x, i.e. no speed-up. |
| x4.9 polars check 683 ms vs pandas 0.6 ms; prepare 1443 → 16 ms | **partially_confirmed** | The pandas 0.6 ms is a warm-cache number (cold: 3.5 ms). Polars is still 260x slower (0.93 s). The 75–90x prepare gain comes from skipping a redundant check. |
| x4.11 152/160 pandas-only, 8 polars-listing compositors | **confirmed** | Independent re-count: 152 pandas-only, 8 "accept all" compositors. Missed: 29 of the 152 take pd-multiindex natively, so they never go through VectorizedDF. |

Not selected but checked in passing:
- x4.7 confirmed: B2u fit 2002+0 conversions / 7.27 s vs B2 0+0 / 3.58 s at n=1000.
- x4.8 confirmed: same `_base.py:2466 droplevel` AttributeError for both multivariate native runs. It is the only data-API pandas call inside `_vectorize`.
- x4.10 confirmed: per-slice lazy filter+collect is 4.4–46x (in-memory) and 5.8–74x (parquet) slower than collect-once.

## 1. Equivalence tests re-run (x4.1, x4.2)

```
$PP $EXP_PY  -W ignore experiments/container_independence/scripts/x4_test_adapter.py
  PASS T1 pandas equivalence: 78/78 configs identical
  PASS T2 polars native: 26/26 configs equal to pandas ref, max pandas<->polars conversions=0
  ... 10 passed, 0 failed
$PP $EXP_PY3 -W ignore experiments/container_independence/scripts/x4_test_adapter.py   (pandas 3.0.6)
  PASS T1 78/78, PASS T2 26/26, 10 passed, 0 failed
```

The pd3 json is saved as `results/x4_proto_adapter_verify0_test_adapter_pd3.json`.

Weaknesses of the original suite:
- Every input comes from `make_panel`: equal lengths, sorted rows, contiguous instances, string keys, named levels.
- T2 excludes int and period time indices.
- T2 compares polars slices only by `allclose` of values plus time values and names.
- T3's "reconstruct stays lazy" check feeds `s.lazy()` of already-eager slices, so it is trivially lazy.

## 2. Adversarial cases (new: `scripts/x4_proto_adapter_verify0_adversarial.py`)

This is an independent harness. It builds pandas frames, then builds polars frames in the same row order (it does **not** use `convert_to`, which sorts). It compares slices and reconstruct results with `pd.testing.assert_frame_equal`, and compares `adapter_get_cutoff` (polars) with sktime `get_cutoff` (pandas).

The cases are:
- unequal lengths, and ragged start/end times
- int instance keys
- int time index with unequal lengths
- interleaved (non-contiguous) instances
- NaNs
- a single instance
- unnamed index levels
- Hierarchical with unequal lengths, int keys, and interleaving, iterated as Series and as Panel
- every case with iterate_cols False and True
- an empty frame

```
$PP $EXP_PY  -W ignore experiments/container_independence/scripts/x4_proto_adapter_verify0_adversarial.py
SUMMARY {'P': [28, 28], 'L': [26, 28], 'C': [11, 11], 'Cfirst': [0, 11]}
$PP $EXP_PY3 -W ignore ...  -> same failures (unnamed_levels L x2, empty x2, Cfirst x11)
```

Results are in `results/x4_proto_adapter_verify0_adversarial_{pd2,pd3}.json`. Findings:

1. **P (pandas adapter vs current): 28/28 identical**, including interleaved instances, Hierarchical→Panel, int keys and NaNs. x4.1 holds beyond the original grid.
2. **L (polars native vs pandas ref): 26/28.**
   - Unequal lengths, int keys, int time, interleaved instances, Hierarchical→Panel and iterate_cols all match strictly.
   - **Bug: unnamed index levels.** A frame with `__index__0, __index__1` yields slices with column `__index__1`. Once the key column is dropped, `_to_logical` reads that as level name `"1"` instead of `None` (time-level name `['1']` vs `[None]`).
   - Cause: the positional placeholder convention `__index__<position>` does not survive slicing. Reconstruct still round-trips, because the names are re-applied at that point.
   - This is the same level-name-fidelity problem as contested claim C2, now on the polars side. The adapter does not fully solve it.
3. **Empty frame:** both the pandas and polars paths raise `ValueError: No objects to concatenate`. Current `VectorizedDF` raises the same error, so this is parity, not a regression.
4. **Shuffled rows (time not monotonic):** sktime rejects these for both pandas and polars, so they are not valid inputs and I excluded them from the counts.
5. **Cutoff bridge (`adapter_get_cutoff`):**
   - Values and dtypes match `get_cutoff` in 11/11 cases.
   - But the bridge sets `freq='D'` where current pandas returns `freq=None` for multiindex frames. That is a behaviour difference: harmless in the tested fit/predict path, but unverified elsewhere.
   - **Bug:** `reverse_order=True` is silently swallowed by `**kwargs`. It returns the **max** instead of the min time in 11/11 cases. `BaseForecaster.update_predict` (`_base.py:2721`) calls `get_cutoff(y, return_index=True, reverse_order=True)`, so the proposed backend-aware cutoff would set a wrong initial cutoff there.
   - It also ignores the `cutoff` fallback for empty frames.
   - Conclusion: the "small framework fix" in x4.6/x4.7 is incomplete as prototyped.

## 3. Forecaster-layer re-run (x4.5, x4.6, x4.7, x4.8)

```
$PP $EXP_PY -W ignore experiments/container_independence/scripts/x4_forecaster_ext.py --grid dev --max-series 1000 --repeat 3 --profile-n 100
```

Saved as `results/x4_proto_adapter_verify0_fext_rerun.json`. The original `x4_forecaster_ext_dev.json` was restored from backup.

Fit times are in seconds. Conversion counts are given as fit then predict, each as `to_pandas+from_pandas`.

| n | A0 pd/pd | A1 pl, current | A2 pl, native per-slice | B1 toy, current | B2u | B2 toy, native+cutoff | B4 pd in, toy, bridge |
|---|---|---|---|---|---|---|---|
| 10 | 0.049 | 0.055 (1+0 / 0+1) | 0.059 (10+0 / 0+10) | 0.099 (21+10 / 10+1) | 0.085 | 0.051 (0+0 / 0+0) | 0.048 |
| 100 | 0.314 | 0.367 | 0.429 | 0.848 | 0.669 | 0.302 | 0.272 |
| 1000 | 3.056 | 3.805 | 4.422 (1000+0 / 0+1000) | 9.564 (2001+1000 / 1000+1) | 7.269 (2002+0) | 3.578 | 3.039 |

All predictions equal A0.

Ratios:
- A2/A1 = 1.07 / 1.17 / 1.16 (x4.5 confirmed).
- B2/B1 = 0.51 / 0.36 / 0.37 (x4.6 reproduces).
- **B2/A0 = 1.04 / 0.96 / 1.17.**

### Methodology flaw in x4.6's headline

The "fit 0.36–0.43x of today's time" compares against B1. B1 is a *polars-inner* forecaster run through today's framework, and no such leaf forecaster exists (x4.11). The decision-relevant baseline is today's pandas path, A0 or A1:
- Against A0, the zero-conversion polars path is **not faster** (0.96–1.17x).
- Against A1 (polars input, convert once) it is 0.82–0.94x.

Per-slice framework overhead dominates: about 3 ms per series for clone, `check_is_scitype`, `convert_to`, `_check_fh` and cutoff handling (see the profile in the json). So container choice moves end-to-end fit by at most about 20% for per-instance forecasters at these sizes.

Predict comparisons (B vs A) mix two different algorithms. NaiveForecaster's `_predict` is 1.7 s of 2.6 s at n=100; the toy predict is trivial. They are not container effects.

Also noted: the FutureWarning on `remember_data` is printed once per clone despite `-W ignore`; the log was 7.8 MB. It adds a uniform per-slice cost to all scenarios, which slightly dilutes the ratios.

### Multivariate (x4.8)

`mv_native_naive` and `mv_native_toy` both fail with `AttributeError: 'list' object has no attribute 'droplevel'` at `_base.py:2466`. Confirmed.

Separately, the polars `concat_cols(keys=...)` writes columns as `f"{key}__{col}"`. A replacement for `droplevel(1)` must therefore strip a string prefix, not drop a level. The column-name convention is a second, silent incompatibility.

## 4. Conversion placement — independent check (x4.4)

The new script `scripts/x4_proto_adapter_verify0_slice_convert.py` uses a different code path: no `x4_adapter`, plain polars `partition_by`. It produces N `pd.Series` slices in four ways and asserts every result equals the sktime convert-once output.

```
PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/x4_proto_adapter_verify0_slice_convert.py --grid dev
```

Times are in milliseconds; the last three columns are ratios to `once_sktime`.

| n x T | once_sktime | slice_sktime | once_raw | slice_numpy | slice_sktime ratio | once_raw ratio | slice_numpy ratio |
|---|---|---|---|---|---|---|---|
| 10x100 | 11.4 | 15.2 | 4.9 | 1.2 | 1.34 | 0.43 | **0.10** |
| 100x100 | 66.2 | 126.8 | 20.9 | 5.6 | 1.92 | 0.32 | **0.085** |
| 1000x100 | 663 | 1433 | 182 | 52.6 | 2.16 | 0.27 | **0.079** |
| 10000x50 | 6090 | 14986 | 1885 | 752 | 2.46 | 0.31 | **0.12** |

What the four ways are:
- `slice_numpy`: `partition_by`, then `pd.Series(col.to_numpy(), index=pd.DatetimeIndex(time.to_numpy()))`.
- `once_raw`: `to_pandas`, then `set_index` and `groupby`.

What this shows:
- The x4 ratio (per-slice `convert_to` 1.4–3.0x worse) reproduces. Re-run of `x4_bench_vectorize.py`: `C_per_slice/C_once` = 1.40 / 2.24 / 2.86 / 3.04.
- But per-slice conversion is **8–13x cheaper** than today's convert-once when done through numpy instead of `convert_to` / `pl.DataFrame.to_pandas`. It is also 2.5–4x cheaper than a raw whole-frame `to_pandas` + groupby.
- So "convert-once is correct for pandas-inner forecasters" is an artifact of sktime's converter overhead. Per-call mtype inference and `to_pandas` fixed cost (pyarrow round trip) are what make per-slice expensive. Slicing itself is not the cause.
- A cheap polars→pd.Series slice edge, or the numpy family x4.14 hints at, would flip x4.4's conclusion at the conversion level. End-to-end it is still bounded by the ~20% framework-overhead ceiling from section 3.

## 5. VectorizedDF benchmark re-run (x4.3, x4.9, x4.10)

```
X4_TMP=<scratch> $PP $EXP_PY -W ignore experiments/container_independence/scripts/x4_bench_vectorize.py --grid dev --tag _verify0rerun
```

Results are in `results/x4_proto_adapter_verify0_bench_vectorize_rerun.json`. Times are in milliseconds, at 10x100, 100x100, 1000x100 and 10000x50 respectively.

| metric | 10x100 | 100x100 | 1000x100 | 10000x50 |
|---|---|---|---|---|
| P_total new/cur | 1.00 | 1.01 | 1.02 | 1.05 |
| L_iter cur/new | 1.4 | 6.8 | 18.3 | 43.5 |
| L_total cur/new | 2.17 | 2.55 | 2.88 | 3.70 |
| L_mtype_check_pl (ms) | 3.6 | 14.7 | 119.7 | 917.5 |
| L_prepare_cur (ms) | 8.7 | 32.2 | 291.1 | 1962 |
| L_prepare_new, check included (ms) | 4.7 | 16.6 | 124.6 | 899 |
| L_prepare_new_known_mtype (ms) | 2.8 | 2.6 | 4.8 | 11.9 |

Lazy (Z):
- Per-slice filter+collect divided by collect_once:
  - in-memory: 4.4 / 14.4 / 46
  - parquet: 5.8 / 30 / 74
- collect_all divided by collect_once: 2.3–27.

### x4.9: pandas caching artifact

The benchmark calls `mtype(y_pd)` repeatedly on the *same* object, so the pandas Index caches `is_monotonic`/`is_unique`. The new script `scripts/x4_proto_adapter_verify0_mtype_check.py` uses a fresh index per call:

| size | pandas warm | pandas cold | polars | polars / pandas cold |
|---|---|---|---|---|
| 1000x100 | 0.62 ms | 1.6 ms | 114 ms | 71x |
| 10000x50 | 0.68 ms | 3.5 ms | 925 ms | 263x |

The gap is real but about 5x smaller than reported. The 1443 → 16 ms ("~90x") figure comes from *skipping* a check that `BaseForecaster._check_X_y` has already done. That fix is legitimate, but it is about removing a redundant validation, not about the adapter.

## 6. Census (x4.11)

The new script `scripts/x4_proto_adapter_verify0_census.py` reads class tags via `get_class_tag`, without instantiating anything:
- `n_forecasters 160`
- pandas-only: 152
- "all mtypes" pass-through compositors (ConformalIntervals, DontUpdate, FallbackForecaster, ForecastByLevel, MultiplexForecaster, Permute, UpdateEvery, UpdateRefitsEvery): 8

This confirms the count, and no forecaster is numpy-only either.

Missed by x4: **29 of the 152 pandas-only forecasters list `pd-multiindex` / `pd_multiindex_hier` natively.** These include the reduction forecasters, pipelines, and global/DL models (NeuralForecast*, PytorchForecasting*, MOIRAI, LagLlama, ...). They never enter `VectorizedDF`, so a VectorizedDF-level adapter gives them nothing. They need whole-frame conversion (or a native rewrite) no matter how vectorization dispatches.

## 7. What x4 missed or got wrong (summary)

1. The B2-vs-B1 "0.36–0.43x" headline uses a strawman baseline. Against today's pandas path the polars-native path is 0.96–1.17x, i.e. no win. The per-instance framework overhead of about 3 ms per series is the real bottleneck, and the adapter does not touch it.
2. The per-slice conversion penalty is caused by sktime's `convert_to` / `to_pandas` fixed costs. A numpy-based slice converter is 8–13x cheaper than convert-once. "Dispatching on input type is the wrong key" is therefore contingent on converter overhead, not a structural fact.
3. `adapter_get_cutoff` ignores `reverse_order` and returns the wrong cutoff for `update_predict`. It also sets freq to 'D' where today's code gives None. The cutoff bridge is not drop-in.
4. The `__index__<position>` naming for unnamed levels breaks in slices: the time level becomes `"1"`. Level-name fidelity (C2) is not fully solved in the polars adapter.
5. The adapter scope excludes 29 natively panel-capable pandas forecasters. The VectorizedDF layer is not on their path at all.
6. The pandas mtype-check number is warm-cache: 0.6 ms warm vs 3.5 ms cold.
7. The polars column-vectorized output naming (`key__col`) differs from pandas' MultiIndex columns. The `droplevel` site needs a semantic replacement, not just a missing method.
8. The original suite has no int or period polars coverage. Int time works in my adversarial run; period was not checked (sktime polars converts periods to timestamps).

## Files (all new, verify0-prefixed)

- `scripts/x4_proto_adapter_verify0_adversarial.py`, with results in `results/x4_proto_adapter_verify0_adversarial_pd2.json` and `_pd3.json`
- `scripts/x4_proto_adapter_verify0_slice_convert.py` (`--grid dev|large`), with results in `results/x4_proto_adapter_verify0_slice_convert_dev.json`
- `scripts/x4_proto_adapter_verify0_mtype_check.py` (`--grid dev|large`), with results in `results/x4_proto_adapter_verify0_mtype_check_dev.json`
- `scripts/x4_proto_adapter_verify0_census.py`, with results in `results/x4_proto_adapter_verify0_census.json`
- Re-runs of the original scripts: `results/x4_proto_adapter_verify0_fext_rerun.json`, `results/x4_proto_adapter_verify0_bench_vectorize_rerun.json`, `results/x4_proto_adapter_verify0_test_adapter_pd3.json`
