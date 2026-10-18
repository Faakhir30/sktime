# x2_nw_index_fidelity — adversarial verification (verify0)

Lens: methodology. I re-measured the reset_index copy and its cost with methods the original did not use, re-ran the original scripts, and added adversarial inputs.
Environments: EXP_PY (pandas 2.3.3), EXP_PY3 (pandas 3.0.6), narwhals 2.27.1, polars 2.0.0. Machine: cloud, 4 cores, load avg ~1.

New scripts (all under `experiments/container_independence/scripts/`):

- `x2_nw_index_fidelity_verify0_reset_mem.py`: one fresh subprocess per config with `MALLOC_MMAP_THRESHOLD_=131072`. It resets the kernel peak-RSS counter (`/proc/self/clear_refs` = 5) and then reads VmHWM/VmRSS deltas, plus memray peak (pd2 env only; memray is not installed in the pd3 env). Sharing is checked by **buffer-address overlap of every block** of the reset frame against every block of the original. Timing uses paired alternating rounds and the **median of per-round ratios**, where the original took the min of repeats.
- `x2_nw_index_fidelity_verify0_adversarial.py`: runs the recipe on mixed data dtypes and on unequal-length/gappy series with NaN. It also probes when the pd3 mutation happens.
- `x2_nw_index_fidelity_verify0_iter.py`: symmetric iteration accounting (with and without droplevel, with and without charging reset_index), correctness of `nw_offsets_iloc` on non-contiguous rows, and the group order of narwhals on polars.

Commands (from WT root, `export PYTHONPATH=.:experiments/container_independence`):

```
$EXP_PY  -W ignore experiments/container_independence/scripts/x2_nw_index_fidelity_verify0_reset_mem.py --grid dev
$EXP_PY3 -W ignore experiments/container_independence/scripts/x2_nw_index_fidelity_verify0_reset_mem.py --grid dev
$EXP_PY{,3} -W ignore experiments/container_independence/scripts/x2_nw_index_fidelity_verify0_adversarial.py
$EXP_PY{,3} -W ignore experiments/container_independence/scripts/x2_nw_index_fidelity_verify0_iter.py --grid dev
# originals re-run:
$EXP_PY{,3} -W ignore experiments/container_independence/scripts/{x2_freq_mutation_probe,x2_recipe_fidelity,x2_nw_ops_index,x2_nw_roundtrip_fidelity}.py
$EXP_PY -W ignore experiments/container_independence/scripts/x2_freq_attach_cost.py --grid dev
```

Results: `results/x2_nw_index_fidelity_verify0_{reset_mem,adversarial,iter}_pd{2,3}*.json`. Re-running the originals also rewrote `results/x2_recipe_fidelity_pd{2,3}.json`, `x2_nw_ops_index_pd{2,3}.json`, `x2_nw_roundtrip_fidelity_pd{2,3}.json` and `x2_freq_attach_cost_pd2_dev.json`.

## 1. reset_index copy: independent re-measurement (claim .3)

All rows are `make_panel_fast`. "blocks shared" means the float block's buffer overlaps an original block. "new MB" is the bytes of reset-frame blocks that overlap no original buffer.

| config | rows x cols | float block shared | data cols shared (shares_memory vs any orig block) | new block MB | RSS retained MB | RSS peak (HWM) MB | memray peak MB |
|---|---|---|---|---|---|---|---|
| pd2 | 100k x 1 | F | 0/1 | 2.29 | 2.30 | 3.02 | 6.29 |
| pd2 | 100k x 10 | F | 0/10 | 9.16 | 9.16 | 9.88 | 13.16 |
| pd2 | 500k x 1 | F | 0/1 | 11.44 | 11.45 | 15.22 | 31.47 |
| pd2 | 500k x 10 | F | 0/10 | 45.78 | 45.79 | 49.51 | **65.80** |
| pd2+CoW | 500k x 1 | T | 1/1 | 7.63 | 7.64 | 11.18 | 27.66 |
| pd2+CoW | 500k x 10 | T | 10/10 | 7.63 | 7.64 | 11.36 | 27.66 |
| pd3 | 100k x 1/10 | T | all | 2.30 | 0.77 | 2.21 / 2.15 | n/a |
| pd3 | 500k x 1 | T | 1/1 | 11.5 | 3.82 | 11.82 | n/a |
| pd3 | 500k x 10 | T | 10/10 | 11.5 | 3.82 | 11.84 | n/a |

- memray gives exactly the original's tracemalloc numbers: 65.8 MB (pd2, 500k x 10), 31.47 MB (500k x 1), and 27.66 MB on pd2+CoW. RSS shows the same thing more tightly. On pd2 with CoW off, the retained memory is the data (38.15 MB) plus the two index columns (7.6 MB). With CoW on (pd2+CoW or pd3), it is the index columns only and does not depend on n_cols. **Confirmed.**
- On pd3, RSS retained (3.8 MB) is lower than the new block bytes (11.5 MB) because the arrow pool reuses memory freed by my warm-up call for the `str` column. The original's tracemalloc+arrow-pool sum (11.9 + 7.7 MB) is an upper-bound style figure. Either way it is about 2 columns' worth of memory.

## 2. reset_index cost in proportion (claim .4)

Median of paired per-round ratios:

| config | rows x cols | reset ms | items ms | reset/items | (reset+set)/items | gb_level ms (no sktime overhead) | **reset/gb_level** |
|---|---|---|---|---|---|---|---|
| pd2 | 100k x 1 | 6.2 | 126 | 4.9% | 15.8% | 33.8 | 18.2% |
| pd2 | 100k x 10 | 10.0 | 174 | 5.9% | 16.4% | 40.3 | 25.6% |
| pd2 | 500k x 1 | 18.9 | 1617 | 1.2% | 5.4% | 555 | 3.4% |
| pd2 | 500k x 10 | 62.6 | 1783 | 3.5% | 6.5% | 537 | 11.0% |
| pd2+CoW | 500k x 10 | 28.6 | 1500 | 1.8% | 5.5% | 526 | 5.4% |
| pd3 | 100k x 1 | 6.1 | 165 | 3.6% | 8.1% | 37.5 | 15.9% |
| pd3 | 100k x 10 | 6.3 | 173 | 3.9% | 8.6% | 41.6 | 16.1% |
| pd3 | 500k x 1 | 20.7 | 1715 | 1.2% | 2.8% | 554 | 3.8% |
| pd3 | 500k x 10 | 22.9 | 1815 | 1.2% | 2.8% | 545 | 4.0% |

- reset/items is 1.2–5.9% and reset+set is 2.8–16.4%. Both fall inside the claimed 0.7–8.2% and 1.9–18%. **Confirmed.**
- Caveat the original underplayed: `items()` is about 3x slower than a bare `groupby(level)` pass because of sktime overhead (`_enforce_index_freq` and so on). Measured against the bare pandas iteration, reset_index is **3.4–25.6%**. Against `items()` it is 1.2–5.9%. Calling it "small" is still fair at 500k rows, but at 100k rows it is not negligible against an optimised baseline.

## 3. Iteration claims (.8, .9, .10)

Paired median ratios, n_cols=1:

| pandas | size | nw_pd/gb_level+droplevel | nw_pd/gb_level **plain** | nw_pd incl. reset / gb_level plain |
|---|---|---|---|---|
| 2.3.3 | 100x100 | 0.37 | 0.82 | 1.05 |
| 2.3.3 | 1000x100 | 0.33 | 0.81 | 0.89 |
| 2.3.3 | 10000x50 | 0.41 | 0.68 | 0.68 |
| 3.0.6 | 100x100 | 0.54 | **1.82** | 2.31 |
| 3.0.6 | 1000x100 | 0.51 | **2.26** | 2.58 |
| 3.0.6 | 10000x50 | 0.61 | **1.69** | 1.77 |

- Claim .8's 0.33–0.75x reproduces (0.33–0.61), but **most of the advantage is the per-group `droplevel` charged to the pandas side only**. narwhals groups also lack a time index, so they are not usable slices, and the flat frame was pre-built. With the droplevel removed, narwhals iteration is 0.68–0.82x on pd2 and **1.69–2.26x slower on pd3**. Including the reset gives 0.68–1.05x on pd2 and 1.77–2.58x on pd3. Verdict: **partially confirmed**. The headline "cheaper" holds only under asymmetric accounting, and it is reversed on pandas 3.
- Claim .9 `nw_offsets_iloc` (≈1.0–1.34x): the method takes the min/max row position per group and slices `iloc[a:b+1]`. That is **only correct if each instance's rows are contiguous**. On a valid pd-multiindex frame in time-major row order, all 5/5 groups returned wrong slices, mixing instances, on both pandas versions. The bench uses contiguous `make_permuted` data, so the error never showed up. A correct variant needs a stable sort first or a take per group; that cost is unmeasured. The 2–3.5x and 7–12x numbers were not re-run. Verdict: **partially confirmed**. The cheapest option the claim recommends is unsafe as written.
- Claim .10: narwhals-on-polars group order was neither first-appearance nor stable in 5 runs (0/5 first-appearance, stable=False). This holds on both pandas versions. narwhals source `_polars/group_by.py` calls `native.group_by(keys)` without `maintain_order`, and `__iter__` iterates it directly. narwhals on pandas gives first-appearance order. **Confirmed** (the partition_by ratio was not re-run).

## 4. Fidelity, recipe, mutation (.5, .6, .7)

- **.5** I re-ran `x2_nw_roundtrip_fidelity.py`. Path B (reset → nw group_by → set_index) passes 9/9 cases on names, level order, level dtypes, row order and values on pd2 and pd3. D, D2, D3 and D4 return a RangeIndex. narwhals source matches the claim: `_utils.py:1057` has `maybe_reset_index` → `reset_index(drop=True)`, and `_pandas_like/group_by.py:278` resets with drop when names overlap (issue #1907). **Confirmed.**
- **.7** I re-ran `x2_recipe_fidelity.py`: 0 False on pd2 and on pd3. **But the test data is float-only, equal-length and regular.** On adversarial inputs:
  - **Mixed dtypes** (float64, int64, bool): `iter_slices` returns float64/float64/float64, where `VectorizedDF.items` keeps float64/int64/bool. The cause is `np.column_stack`. Both backends and both pandas versions are affected. `long_to_mi` is fine (equals True).
  - **Series with their own freq**: in an unequal-length series sampled every 2 days, `items()` gives freq `2D` and the recipe gives `None`, because the recipe uses one global freq from the schema. Values and index values match.

  Verdict: **partially confirmed**. 198/198 holds only for the tested input class, and the "exact" claim for slices fails on mixed-dtype frames and on per-series freq.
- **.6** The mutation reproduces on pd3: `x2_freq_mutation_probe.py` prints `caller frame mutated by VectorizedDF.items(): True` on 3.0.6 and False on 2.3.3. My probes narrow the condition. The mutation happens **only when a slice's time values equal the whole time level** (all series share the same timestamps): equal-length case on pd3, level freq None → Day. With unequal or gappy series the level freq stayed None, and no wrong freq was written. Re-running `x2_freq_attach_cost.py` on pd2 gives an items() slowdown without level freq of **2.27–3.18x** (claimed 3.0–5.4x) and infer/grid of **3.8–5.3x** (claimed 6.6–7.9x). Verdict: **partially confirmed**. The direction holds, the magnitudes are about 40% lower in this run, and the mutation is conditional.

## 5. Spot checks of other claims

- Claim .1, "27 of 37": re-counted on pd3, there are 37 rows. 22 frame ops are `MultiIndex aligned=True`, and 5 more are Series ops whose alignment is **not checked** (column shows `-`). Two of the 22 are the pandas-only "LEAK" rows. The count is right, but "aligned" is unverified for 5 of the 27.

## Things the original missed or got wrong

1. The iteration speed-up (.8) comes from charging `droplevel` only to the pandas side. Without it, narwhals is slower on pandas 3 (1.7–2.3x).
2. `nw_offsets_iloc` assumes contiguous instance rows. It silently returns wrong slices on valid, non-contiguous pd-multiindex frames.
3. The recipe `iter_slices` upcasts mixed-dtype data (int, bool → float64) and drops per-series freq that differs from the global freq. The fidelity suite is float-only and regular.
4. The reset_index cost is reported only against `items()`, which includes sktime overhead. Against bare `groupby(level)` it is 3–26%.
5. The pd3 mutation needs every series to share the full time level. It is still a real side effect in the most common equal-length panel case.
6. freq-inference cost ratios did not reproduce at the stated magnitude (2.3–3.2x, not 3.0–5.4x).
7. Report inconsistency: the bottom line says "36/36 combinations", while §6 says 18 per pandas version. These agree only if both versions are summed; the claims use 198 = 18 x 11 per version.
