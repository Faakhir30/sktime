# r2_narwhals_verify0 — adversarial verification of r2_narwhals

Verifier lens: open every cited narwhals 2.27.1 source line / URL and re-check the claims, using code paths independent of `scripts/r2_narwhals_probe.py`.

## Commands (all run on the cloud machine, 2026-10-18)

```
# pandas 2.3.3 env
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/r2_narwhals_verify0_checks.py --grid dev
# pandas 3.0.6 env (no ibis)
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/r2_narwhals_verify0_checks.py --grid dev
# pyarrow scaling at larger G (pd2 env)
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/r2_narwhals_verify0_pyarrow_scaling.py --grid dev
```
Raw outputs:
- `results/r2_narwhals_verify0_pd2_dev.json`
- `results/r2_narwhals_verify0_pd3_dev.json`
- `results/r2_narwhals_verify0_pyarrow_scaling_dev.json`

Peak RSS was below 370 MB. Versions: narwhals 2.27.1, polars 2.0.0, pyarrow 26, dask 2026.8.0, duckdb 1.5.6.

Source checks used the narwhals 2.27.1 site-packages in EXP_PY. `gh api` returned 403 for these repos in this session, so GitHub issues were read with WebFetch.

## Verdicts on the 8 claims I selected

### r2_narwhals.1 — collect defaults and allowed backends (C1): **confirmed**

**Independent check.** I ran an `isinstance` check on the native result for every backend × target pair (V1).
- Default targets: polars lazy → `pl.DataFrame`; dask → `pd.DataFrame`; duckdb → `pa.Table`; ibis → `pa.Table`.
- `collect('dask' | 'duckdb' | 'ibis')` raises `ValueError: Unsupported backend value`.
- `to_native()` after `filter` keeps the native lazy type for all four backends.

**Source.**
- `dataframe.py:2496` defines `collect(backend: IntoBackend[Polars|Pandas|Arrow])`.
- `_utils.py` `can_lazyframe_collect` = {PANDAS, POLARS, PYARROW}.

**Related findings (claim .2).**
- Dask → polars goes through `pl.from_pandas` exactly once (spy count = 1; `_dask/dataframe.py:~143`).
- `duckdb.collect().lazy()` keeps a `pyarrow.Table`. `lazy('duckdb')` restores a `DuckDBPyRelation`.
- Issue #3210 is open, as stated. Maintainers say the collect/lazy mapping "shouldn't be symmetric, it's not a bijective mapping". This makes it unlikely the backend will be remembered automatically.
- Kwargs pass through: `collect(engine='streaming')` works on polars lazy.

### r2_narwhals.3 — pandas MultiIndex handling: **partially confirmed (incomplete)**

**What reproduces.** Iterating `group_by` by an index-level name keeps the sub-frame's MultiIndex and its names (V2, both pandas versions, all three time kinds). `maybe_reset_index` uses `reset_index(drop=True)` (`_utils.py:1057`).

**What the claim misses: places where narwhals silently loses the index** (V2, identical on pd2 and pd3, all time kinds):
- `to_polars()`, `lazy('polars').collect()` and `lazy('duckdb').collect('pandas')` all return only `['c0']`. The instance and time index is dropped without a warning.
- `to_arrow()` instead returns `['c0','h0','time']`, so the index is kept as columns. Behaviour is inconsistent across targets.
- `write_csv` writes header `c0` only.
- `to_polars().to_pandas()` comes back with a `RangeIndex`.
- `lazy('dask')` on a MultiIndex frame raises `NotImplementedError: isna is not defined for MultiIndex`.
- **Name collision:** when a column has the same name as an index level, `group_by('h0')` groups by the column. The sub-frame index is then a flat `Index` with name `None`, and the level names are gone. This is exactly the C2 symptom ("level names were gone"), reproduced by a different path than the one r2 tested.

So any non-pandas hop through narwhals must `reset_index` first. Doing so is safe (see .4), but forgetting it loses data silently.

### r2_narwhals.4 — reset_index → set_index round trip: **confirmed**

**Different equality check.** I used `pd.testing.assert_frame_equal(check_index_type=True, check_exact=True)` instead of `.equals()`. It passes for datetime, period and int indexes on pd2 and pd3. It also passes on a polars-backend round trip (`lazy('polars').collect('pandas').set_index(...)`).

Level names survive as column names. The C2 statement "the index that came back was a RangeIndex" is true only if you do not `set_index`, or if you use `maybe_reset_index`, the to_polars path, or the collision case above.

### r2_narwhals.5 — reset_index copy and cost (C2): **partially confirmed; the cost framing is misleading**

**Copy semantics confirmed by 3 independent methods (V3).**

| Check | pd 2.3.3 | pd 3.0.6 |
|---|---|---|
| Block-level `np.shares_memory` between original and reset frame | no shared block | blocks shared |
| In-place mutation of the reset frame propagates back | no (it is a copy) | n/a |
| tracemalloc retained bytes ÷ value-column bytes, at 100k and 200k rows | 3.0× (value column + index columns; `h0` is object) | 1.0× (only the new key columns; the value column is shared) |

**The cost comparison is the problem.** r2 compares a one-time `reset_index` with a *bare* `groupby(level=0)` loop. That loop does not represent what sktime does today, and it is not what the reset path has to do (V6, end-to-end delivery, ABAB interleaved, median of per-round ratios).

The bare loop is only a fraction of the real `VectorizedDF.items()` cost:

| | pd2 | pd3 |
|---|---|---|
| Bare loop ÷ real `items()` | 0.07–0.18 | 0.17–0.48 |

So r2's "relative to iteration" ratios use a denominator that is 2–14× too small.

The real C2 cost is **per group**. A pandas inner estimator needs a time index, so every sub-frame must be `set_index('time')`. Relative to the current `items()`:

| Path | pd2 | pd3 |
|---|---|---|
| `nw[pd flat] iter + set_index(time)` | 1.13–1.45× | 3.5–7.0× |
| same, + freq enforcement | 1.65–2.5× | 4.7–9.8× |

**Conclusion.** C2's "it is a copy" is real on pd2 and nearly free on pd3. The meaningful cost is rebuilding the per-series index, and that cost is larger on pandas 3.

The cheaper option, keeping the MultiIndex and using the undocumented level-name passthrough, costs:

| Path | pd2 | pd3 |
|---|---|---|
| `nw[pd MI] iter + droplevel` | 0.36–0.58× | 0.91–1.35× |
| same, + freq enforcement | 0.91–1.10× | 0.91–1.54× |

All ratios are relative to the current `items()`.

### r2_narwhals.7 — polars group order is nondeterministic: **confirmed; attribution corrected**

Results (V4, 30 runs per setting):
- **Direct polars, no narwhals:** 25–30 distinct orders in 30 runs, even with only 5 groups. First-appearance order was hit 0/30 times.
- **Through narwhals:** 26–30 distinct orders, the same behaviour. It is polars' hash group_by behaviour, which narwhals passes through without a `maintain_order` knob.

Fix options at 1000×100:

| Option | Time ÷ plain narwhals iteration |
|---|---|
| Reorder by first appearance through narwhals | 1.09× (pd2 env), 1.29× (pd3 env) |
| Native `partition_by(as_dict, maintain_order=True)` | 0.23–0.38× (2.6–4.3× faster) |

Status: #3309 is still open, but a contributor started work on 2026-10-07 and **PR #4042 (open)** adds eager `maintain_order`. Gorelli supports it for eager `DataFrame` only, not `LazyFrame`. r2 does not mention the PR.

### r2_narwhals.8 — grouped iteration ratios, pyarrow O(groups×rows): **partially confirmed**

**Source.** `_arrow/group_by.py:188-213` runs one `table.filter(pc.equal(...))` over the full table for each unique key, so the cost is O(G·N) asymptotically. Confirmed.

**Magnitude at 1000×100 vs bare `groupby(level=0)`** (`verify0_pyarrow_scaling.py`):

| Size (G × n_time) | nw[pyarrow] | bare pandas | Ratio |
|---|---|---|---|
| 1000 × 100 | 409 ms | 29.6 ms | 13.8× (r2: 19.9–27.5×) |
| 2000 × 100 | — | — | 19.1× |
| G = 250–2000 × 10 | — | — | 4.3–7.0× |

Log–log slope over G at fixed rows per group:
- n_time = 100: 1.61
- n_time = 10: 1.13
- pandas: 0.90–0.97

So the trend is superlinear and turning quadratic, but at DEV sizes it is not yet the stated magnitude.

**A linear alternative exists.** pyarrow `dictionary_encode` + `sort_indices` + `slice` runs at 0.17–0.26× of the pandas bare loop (V5, slope 0.48).

**The other ratios are against the bare baseline** (see .5). Against the real current `items()`, narwhals is not slower:

| Path | pd2 | pd3 |
|---|---|---|
| `nw[pl] iter + to_native` | 0.18–0.53× | 0.33–0.59× |
| Native `partition_by(maintain_order)` | 0.06–0.16× | 0.16–0.21× |

### r2_narwhals.9 — `.over(partition_by, order_by)` window matrix: **confirmed**

**Different oracle.** I compared against a pure-numpy per-group reference rather than pandas, on shuffled rows with nulls, over 7 ops (mean, cum_sum, shift, diff, rolling_mean, rolling_std, ffill).

| Backend | Ops passing (7 total) |
|---|---|
| pandas | 7 |
| polars | 7 |
| polars lazy | 7 |
| duckdb | 7 |
| ibis | 6 (ffill raises NotImplementedError) |
| pyarrow | 1 (all ordered ops raise NotImplementedError) |
| dask | 1 (all ordered ops raise NotImplementedError) |

Results are the same on pd2 and pd3 (pd3 has no ibis).

**ewm (claim .14), checked against a hand-written adjust=True, ignore_nulls=False recurrence.** **polars matches the hand-written version**, and **pandas deviates**: it carries the previous value forward at null positions. So the "ewm mismatch" is a pandas-side difference, not a polars bug.

### r2_narwhals.10 — per-call overhead: **partially confirmed; the baseline is asymmetric**

r2 pairs `nw.with_columns(shift).to_native()` with native `df.shift(1)`. The narwhals call builds a new frame with all columns; `df.shift` does not.

With r2's pairing I reproduce:
- pd2: 7.8–8.2×
- pd3: 11.4–12.6×

With a like-for-like native, `df.assign(c0=df.c0.shift(1))` (V7):

| Measure | pd2 | pd3 |
|---|---|---|
| Ratio, like-for-like | 2.4–2.8× | 4.1–4.4× |
| Absolute added time per call | 151–163 µs | 338 µs |

The added time is the same at 100 and 10,000 rows, so it is a fixed overhead.

Other measurements:
- `nw[pl] with_columns`: 1.0–1.8× native polars.
- Native polars 2.0 `with_columns(shift)` alone costs about 115–204 µs per call (pd2 env). So per-series polars is **not** cheaper than pandas for small series.
- `nw[pd]` `Series.mean`: 2.5–4.2× native, about 36–47 µs added.

The polars `Series.mean` 56–80× figure was not re-checked.

## Things r2 missed or got wrong

1. **Wrong baseline.** Every "relative to current" iteration ratio in r2 uses a bare `groupby(level=0)` loop. That loop is only 7–48% of the real `VectorizedDF.items()` cost, which includes droplevel and freq enforcement. With the right baseline, narwhals-at-VectorizedDF is cost-neutral or a win for polars. It is ≈0.9–1.5× for pandas when the MultiIndex passthrough is used.
2. **The real C2 cost is per-group `set_index`, not the one-time `reset_index`.** The flat path is 1.1–2.5× (pd2) and 3.5–9.8× (pd3) the current `items()`. This argues for keeping the MultiIndex on the pandas path, or for handing estimators numpy and time arrays rather than pandas sub-frames.
3. **Silent index loss on cross-backend conversion** of a pandas MultiIndex frame:
   - `to_polars`, `lazy('polars')` and `lazy('duckdb')` drop the index.
   - `to_arrow` keeps it as columns.
   - `lazy('dask')` raises.
   - A column/level name collision loses the level names in group iteration.
4. **Polars order nondeterminism is polars-native**, and a narwhals fix (PR #4042, eager `maintain_order`) is in progress. A workaround costs 1.1–1.3×. Native `partition_by(maintain_order=True)` is 2.6–4.3× faster than narwhals iteration.
5. **The pyarrow quadratic magnitude is overstated at DEV sizes** (13.8× rather than 20–28× at 1000×100). A linear pyarrow-native alternative exists at 0.2× of the pandas loop.
6. **ewm semantics.** pandas is the outlier versus the standard recurrence, not polars.

## Caveats

- The machine was shared, so timings are noisy. All comparisons are in-process, interleaved, and min- or median-of-rounds.
- pd3 ratios for the `items()` baseline differ from pd2 partly because current `items()` is about 2.7× faster under pd3 at 1000×100 (143 ms vs 385 ms).
- Not checked:
  - Adopter census (.12) and the sklearn hard dependency (.11). Only METADATA was checked for .11: sklearn 1.9.1 has `Requires-Dist: narwhals>=2.0.1` unconditionally, and narwhals has 0 non-extra requirements.
  - Period → Unknown dtype (.6).
