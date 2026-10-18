# x3_iter_bench_verify0 — adversarial verification of x3_iter_bench

Lens: are the comparisons fair? Specifically, do the strategies produce equivalent slices and outputs, and are conversion and setup costs counted the same way for each? Everything below was re-run.

The original workstream never wrote `reports/x3_iter_bench.md` or `results/x3_iter_bench.json`. Its background dev-grid run (`scratchpad/x3_dev_pd2.log`) died at the first point, so its claims rest on a single 10x100 smoke run. I re-ran its script at 10x100 and 1000x100 on pandas 2.3.3, and at 1000x100 on pandas 3.0.6. I also wrote two independent check scripts that use different code paths.

## Commands (run from the WT root, `S=/tmp/claude-0/-home-user-sktime/c9329e73-89f5-514d-ae5a-9180bf0e4a71/scratchpad`)

1. Original script:
   `PYTHONPATH=.:experiments/container_independence $S/expenv/bin/python -W ignore experiments/container_independence/scripts/x3_iter_bench.py --only {10x100,1000x100} --tag verify0_<p> --rounds 3 --skip-mem`
   - Output: `results/x3_iter_bench__verify0_10x100.json`, `__verify0_1000x100.json`; log `$S/x3v0_orig.log`.
   - The same command with `$S/expenv_pd3/bin/python` and `--only 1000x100 --tag verify0_pd3_1000x100` writes `results/x3_iter_bench__verify0_pd3_1000x100.json`.
2. Independent check (full slice equivalence over all instances, freq-less control built differently, accounting that includes setup and conversion symmetrically, end-to-end NaiveForecaster):
   `... scripts/x3_iter_bench_verify0_check.py --grid dev` (pd2), and `--only 1000x100` (pd3).
   - Output: `results/x3_iter_bench_verify0_check_dev_pd{2,3}.json`.
3. Narwhals reconstruct, literal-key strawman vs vectorized version, plus narwhals[polars] group order:
   `... scripts/x3_iter_bench_verify0_nwrec.py --grid dev` (pd2 and pd3). Log: `$S/x3v0_nwrec.log`.
4. Ad-hoc probes of the freq control, all in the scratchpad, run with both pythons: `$S/x3v0_fresh.py`, `$S/x3v0_stag.py`, `$S/x3v0_if.py`. Logs: `$S/x3v0_fresh.log`, `$S/x3v0_stag.log`.

Timings are min-of-3, taken with the strategies interleaved, with machine load around 1. Absolute numbers are noisy; read the ratios.

## Re-run numbers: (B) iterate, ratio vs S1_sktime (per-slice us in parentheses)

| strategy | 10x100 pd2 | 1000x100 pd2 | 1000x100 pd3 | original smoke (10x100) |
|---|---|---|---|---|
| S1_sktime | 1 (272) | 1 (113) | 1 (143) | 1 (217) |
| S1_sktime_nofreq | 2.08 | 3.44 | **0.97** | 2.28 |
| S2_iloc | 0.61 | **0.99** | 0.92 | 0.65 |
| S2_build (no freq) | 0.17 | 0.17 | 0.15 | 0.175 |
| S3 partition_by | 0.52 | **0.061** | 0.052 | 0.52 |
| S4 polars group_by | **1.78** | **2.16** | 1.24 | 0.94 |
| S5 polars slice | 0.041 (11.0) | 0.012 (1.3) | 0.010 | 0.05 |
| S5p (to_pandas+set_index) | 3.09 | 5.60 | 6.92 | 4.1 |
| S5p_fast (no freq!) | 0.85 | 1.26 | 1.51 | 1.25 |
| S6 numpy | 0.018 (5.0) | 0.006 (0.6) | 0.005 | 0.016 |
| S7a narwhals[pd] | 1.56 | 2.24 | 4.72 | 1.69 |
| S7a_pd | 2.62 | 4.43 | 6.71 | 3.4 |
| S7b narwhals[pl] | 1.58 | 2.76 | 1.35 | 1.27 |
| S8 arrow slice | 0.034 (9.4) | 0.010 (1.1) | 0.008 | 0.037 |

(C) reconstruct, ratio vs S1:

| | 10x100 pd2 | 1000x100 pd2 | 1000x100 pd3 |
|---|---|---|---|
| S2_codes | 0.62 | 0.38 | 0.48 |
| S6 | 0.11 | 0.064 | 0.096 |
| S6_to_pd | 0.61 | 0.107 | 0.174 |
| polars vec (gather) | 0.74 | 0.147 | 0.175 |
| polars literal | 0.91 | 1.62 | 2.36 |
| S7a_lit (narwhals) | 6.1 | 30.7 | 92.5 |
| S8 | 0.32 | 0.085 | 0.117 |

Every equality check in the original script passed in all three runs.

## Independent checks

- **Full slice equivalence**, across all instances at every dev point including 10000x50, compared with VectorizedDF.items:
  - S2_iloc: frame equal and freq equal on 10000 of 10000 instances.
  - S2_build: frame equal, but the freq matches on 0 instances.
  - S2_build with validated `DatetimeIndex(freq='D')`: frame and freq equal on all instances.
  - S5, S6, S8 and S7a: values and time arrays equal on every instance.
  - So the original check, which compared only the first slice plus row counts, was weak, but its conclusion holds.
- **Freq-less control built a different way** (flat frame, raw datetime64 time, set_index). Cost of `items` vs the freq-carrying frame:
  - pd2: 2.38x at 10, 3.31x at 100, 3.51x at 1000, 3.53x at 10000 series.
  - pd3 at 1000x100: **1.03x**.
  - Cause on pd3: every series shares one time grid, and the slices share the level's index object. `_enforce_index_freq` assigns `item.index.freq = ...` in place, so the first slice sets the freq for all the others. Afterwards even a fresh groupby slice of the user's input frame carries `<Day>` (`$S/x3v0_fresh.log`). This is an in-place side effect on user data. It also contaminates the original script's pd3 control: its check `S1_nofreq_slice_freq_before_enforce` reports `<Day>` on pd3.
- **The S1 baseline is the best case.** Panels with staggered start dates (1000x100, 1099 unique timestamps, `$S/x3v0_stag.py`) cost about 3.5x more even when `freq="D"` is passed:
  - pd2: 406 ms, vs 116 ms on the aligned grid.
  - pd3: 458 ms with freq given, 414 ms without.
  - With staggered starts the slice index is rebuilt per slice, so infer_freq runs on every slice.
- **Symmetric accounting**: S5 with the rle setup included, plus the pd->polars conversion when the input is pandas, vs S1 with the VectorizedDF schema included:
  - Polars input: 0.37x at 10, 0.098x at 100, 0.023x at 1000, 0.015x at 10000 series.
  - Pandas input, conversion included: 1.03x / 0.30x / 0.106x / 0.056x.
  - pd3 at 1000: 0.018x (polars input) and 0.056x (pandas input).
- **End-to-end calibration**, real vectorized NaiveForecaster fit+predict on the panel:
  - One S1 iteration pass is **1.1–1.6% of panel fit+predict** at 10, 100 and 1000 series on pd2, and 1.4% on pd3.
  - Panel time / (n × single-slice time) is 1.40 at 10 series, 1.07 at 100, 1.01 at 1000, and 0.96 on pd3.
  - Per-instance cost is about 11 ms, dominated by the forecaster's own framework overhead.
- **The narwhals reconstruct (S7a_lit) is a strawman.** I replaced it with a vectorized version: one `nw.concat`, one repeated key Series, then `set_index`. That gives **2.8x S1 at 1000 series on pd2 and 4.5x on pd3**, against 28.7x and 84.8x for the literal-per-part version. Values and index names equal S1 in both versions. A narwhals-based reconstruct is still slower than pd.concat(keys=), but the original overstates the gap by roughly 10–19x at 1000 series.
- **narwhals[polars] group order.** The narwhals 2.27.1 source (`_polars/group_by.py:36`) calls `native.group_by(keys)` with no `maintain_order` parameter, and nothing exposes one. Order differed from first appearance at every point. A Python-side sort by first appearance fixes it. The pandas-like backend uses `sort=False` (`_pandas_like/utils.py:676`), so its order matched.

## Verdicts

The verdicts are listed in the structured output; the main points are below.

- x3.1 positional slicing: **partially confirmed**. It is 25–54x faster at 10x100 and 85–180x at 1000x100. But S1 costs about 113–143 us per slice at scale, not 217, because the 10x100 figure is inflated by fixed overhead. Also, S5, S6 and S8 slices are not what any current pandas-inner forecaster receives.
- x3.2 converting back to pandas per slice: **confirmed, and stronger at scale**. The extra caveat is that S5p_fast slices have no freq.
- x3.3 freq inference: **partially confirmed**. It holds on pandas 2 only, and the baseline itself is best-case.
- x3.4 codes-based reconstruct: **confirmed**, and the advantage grows with scale.
- x3.5 C2 evidence: **partially confirmed**. The slice index is `Index` on pd2 and `RangeIndex` on pd3. Names are not lost after `set_index` on the known key columns. The order claim is confirmed.
- x3.6 polars setup cost: **partially confirmed**. It is 2.3x at 10x100 but only 1.6x (pd2) and 1.2x (pd3) at 1000x100. Within `convert_to`, the checks take 88% of the polars->pd time at 1000x100 (136 ms vs 16 ms for the raw converter).
- Calibration (~2% share): **confirmed** at 1.1–1.6%.
- Unlisted summary claim S4 = 0.94x: **refuted** at scale; it is 1.78–2.16x on pd2.

## What the original missed

1. The report and merged json were never written, and nothing was measured above 10x100.
2. Ratios depend heavily on scale. S3 drops from 0.52 to 0.06, S2_iloc rises from 0.65 to 0.99, S4 rises from 0.94 to 2.16. Small-scale numbers would mislead the architecture choice.
3. The S1 baseline uses the most favorable layout (an aligned, shared time grid). Panels with staggered starts make S1 about 3.5x slower.
4. sktime's `_enforce_index_freq` mutates the input in place on pandas 3, and this invalidates the pd3 freq control.
5. The narwhals reconstruct is a strawman, overstating the gap by about 10–19x.
6. The most decision-relevant point was under-emphasized: for a real forecaster, iteration is about 1–2% of wall time. Container choice for iteration only matters for very cheap per-series fits or for native, vectorized (non-loop) forecasters.
7. Memory was not measured. The original's peak-RSS numbers were all below 1 MB at 10x100, and I did not re-run the memory mode.
