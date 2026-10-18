# x5_proto_narwhals_verify0: adversarial verification of the narwhals prototype (x5)

Lens: correctness and fidelity. I re-ran the original tests, added adversarial inputs, and checked
the claims a second way with different code.
Env: narwhals 2.27.1, polars 2.0.0, pyarrow 26, pandas 2.3.3 (`EXP_PY`) / 3.0.6 (`EXP_PY3`).
The machine is shared (4 cores, other agents running). Ratios are measured within one process.

## Commands (from WT root, `P=PYTHONPATH=.:experiments/container_independence`)
```
$P $EXP_PY  -W ignore experiments/container_independence/scripts/x5_test_narwhals.py                 # rerun of original -> results/x5_test_pd2.json
$P $EXP_PY3 -W ignore experiments/container_independence/scripts/x5_test_narwhals.py                 # -> results/x5_test_pd3.json
$P $EXP_PY  -W ignore experiments/container_independence/scripts/x5_bench_vectorize.py --grid dev    # rerun of original -> results/x5_bench_vectorize_dev.json
$P $EXP_PY  -W ignore experiments/container_independence/scripts/x5_proto_narwhals_verify0_adversarial.py   # -> results/x5_proto_narwhals_verify0_adversarial_pd2_dev.json
$P $EXP_PY3 -W ignore experiments/container_independence/scripts/x5_proto_narwhals_verify0_adversarial.py   # -> ..._pd3_dev.json
$P $EXP_PY  -W ignore experiments/container_independence/scripts/x5_proto_narwhals_verify0_claims.py --grid dev            # -> results/x5_proto_narwhals_verify0_claims_pd2_dev.json
$P $EXP_PY3 -W ignore experiments/container_independence/scripts/x5_proto_narwhals_verify0_claims.py --grid dev --skip k3,k5   # -> ..._pd3_dev.json
```
I copied the original result JSONs to the scratchpad (`x5v0_*_orig.json`) before the reruns overwrote them.

## Verdicts

| claim | verdict | evidence |
|---|---|---|
| x5.1 VectorizedDF pandas parity 17/17 | **confirmed** | The rerun gives 17/17 on pd2 and pd3, with `fast_slices_equal=False` only for `panel_none_names`. In my extra V1 cases (time-major interleaved rows, unequal lengths, a 3-level hierarchy interleaved, iterate_as Panel), items, names and reconstruct are equal to sktime in 4/4 cases on both pandas versions. |
| x5.3 VectorizedDF bench | **partially confirmed** | Rerun round-trip ratios vs cur_pd: nw_pd ×6.30 at 1k and ×5.86 at 10k (claimed ×6.6–8.6; ×5.96 at 100). nw_pd_fast ×2.00 / ×1.88 (claimed ×1.8–2.2). nw_pl ×0.32 / ×0.26 (claimed ×0.27–0.37). cur_pl/nw_pl at 10k = 3893/455 ms = **×8.6** (claimed ≈8×). The direction holds; the generic-path slowdown is somewhat smaller than stated. At 10 series nw_pl is ×1.55, *slower* than current. The polars gain only shows up from about 100 series. |
| x5.4 numeric identity with NaiveForecaster | **partially confirmed** | It holds for the tested matrix: datetime D/h/W/ME/MS/QE/YE, 3-level hierarchy, 3 columns with 15% scattered NaN, unequal lengths and ends at **1000 series on all 7–8 backends** (pyarrow bug zone), and rows shuffled within and across series. Adversarial totals: pd2 215/243 backend×case pass, pd3 186/220. The failures are real gaps, listed under "Missed". |
| x5.5 speed-ups vs sktime loop | **partially confirmed** | At 1000×100, `last`: sktime 12.8 s (single run), nw_pl 42 ms (**×304**; claimed ×359), nw_pdmi 146 ms (**×88**; claimed ×84). I did not re-measure 10k (sktime about 100 s). Methodology flaw: the speed-up comes from vectorization, not from narwhals. A plain vectorized pandas `groupby(level=0).last()` with no narwhals takes 14.9 ms, which is **×859** vs sktime and **9.8× faster than the narwhals-on-pandas proto** (11.5× at 10k). Plain polars is 7–11.5× faster than narwhals-on-polars. The plain versions assume sorted time and are simpler, so this is an upper bound on the overhead. |
| x5.6 C1 | **confirmed** | Independent code (K1): default collect gives polars-lazy→polars.DataFrame, dask→pandas, duckdb→pyarrow, ibis→pyarrow. `collect("polars"/"pandas"/"pyarrow")` works on all four. `collect(<own lazy backend>)` raises `ValueError: Unsupported backend` for dask, duckdb and ibis. Eager output can therefore never stay dask/duckdb; only `.lazy(backend=)` re-wraps it. |
| x5.7 C2 | **partially confirmed** | Qualitative part confirmed: `maybe_reset_index` returns a RangeIndex and drops the levels, while native `reset_index` keeps the names. Time ratio confirmed: reset/prepare is 0.45 / 0.33 on pd2 and 0.24 / 0.14 on pd3 (claimed 0.18–0.49). **The byte numbers are a measurement artifact.** `memory_usage(deep=True)` counts each shared Python string once per row. tracemalloc shows reset_index **retains ×2.0 / ×1.9** of frame bytes on pd2 (claimed ×6.1 / ×5.8) and **×0.72 / ×0.71** on pd3 (claimed ×2.2 / ×2.1). The transient peak is ×5.5 / ×5.2 (pd2) and ×2.2 (pd3). |
| x5.8 pyarrow over() misalignment | **confirmed** | Independent repro (K3) with a different reference (np.bincount means) and `mean` instead of `min`: at 1000 groups 6/6 trials wrong and **100%** of rows wrong; at 5000 groups 6/6 and 80%; at 100 and 500 groups 0/6. Root cause confirmed in two ways. The raw `pa.Table.join` reorders the left rows (5/6 trials at 1000 groups, 6/6 at 5000). narwhals `_arrow/expr.py::over` returns `partition_tbl.join(tmp).get_column(alias)`, which assumes the left order is kept. The threshold is nondeterministic: 0/6 at 500 here, 6/10 in the original run. |
| x5.10 lazy stays lazy | **confirmed** (stronger) | K4 used a polars `scan_parquet` source with a counting `map_batches` UDF on the value column. After fit and predict the UDF had been called 0 times; it ran only on the user's collect. This holds even with `freq=None`: the frequency-inference queries touch only the key and time columns, so projection pushdown skips the UDF. The output types are `polars.LazyFrame` and `DuckDBPyRelation`. I did not re-measure the plan-build times (4.4 / 77 / 71 ms). |

## Missed or wrong (adversarial findings; results/x5_proto_narwhals_verify0_adversarial_{pd2,pd3}_dev.json)
1. **pandas-MI with integer time cannot forecast.** This fails on pd2 and pd3 for every strategy with
   `ValueError: cannot generate future time points without a frequency`. `infer_freq_nw` returns None
   for a RangeIndex level. The long-format backends work because they infer an int step. The VectorizedDF `panel_int` test hid this.
   Integer-indexed panels are common in sktime.
2. **pandas 3 + PeriodIndex (M, Q) fails on the pandas-MI path.** On every strategy, `to_offset("M")` gives
   `'M' is no longer supported for offsets, use 'ME'`, and `Q-DEC` fails the same way. The prototype passes the Period freqstr to
   `to_offset`. "Correctness checked on pd3" covered only datetime freq D. On pd2, Period M and Q pass.
3. **Series shorter than sp (`last`/`mean`, sp=7, 4 observations): the prototype drops rows instead of
   returning NaN.** It outputs 66 rows vs sktime's 72. There is no state row for an unseen season, so `filter(S == season)`
   yields nothing. The forecast frame then has the wrong shape. The values on the rows both outputs share are equal.
4. **Business-day freq `B` is unsupported on every backend** (`NotImplementedError`). MS, QE and YE are also unsupported on
   pyarrow and dask. Per freq: h, W-SUN, ME, MS, QE and YE are OK on pandas-MI, polars, polars-lazy and duckdb.
5. **drift with NaN at the window end:** sktime raises `ValueError`. The prototype silently returns NaN (20% of forecast values in
   the test). This is a behavioural difference.
6. **Speed attribution (decision-relevant):** in this prototype narwhals costs about 7–11× over hand-written native
   code on the same backend. The ×100–900 gain belongs to "panel-native vectorized algorithm", and a
   plain pandas implementation gets it too (×859 at 1k series). "Re-implement in narwhals" and "vectorize the
   naive family" are separate decisions, and the report combines them.
7. The C2 memory numbers (×6 / ×2) are overstated, as explained under x5.7.
8. Minor: at small n (10 series) the narwhals VectorizedDF on polars is slower than current pandas (×1.55).

## Caveats
* Timings ran under concurrent load. sktime was run once per size (12.8 s at 1k). The 10k sktime figure was not re-measured.
* The plain pandas/polars baselines implement `last` only, for sorted time and no NaN→null normalisation.
* The tracemalloc numbers cover numpy/Python allocations. pd3 string columns are pyarrow-backed, and the
  pd3 retained figure might under-count arrow buffers allocated outside tracemalloc (unmeasured).
