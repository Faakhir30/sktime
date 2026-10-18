# x6_proto_hub_verify0 — adversarial verification of x6_proto_hub (lens: fidelity & zero-copy, re-measured costs)

Verifier for `reports/x6_proto_hub.md` / `protos/x6_hub.py`. All numbers below come from scripts run in
this session (cloud, 4 cores, shared). Envs: `$EXP_PY` (pandas 2.3.3, polars 2.0.0, narwhals 2.27.1,
pyarrow 26, duckdb 1.5.6, dask 2026.8) and `$EXP_PY3` (pandas 3.0.6).
Prefix for all commands: `cd /home/user/sktime && PYTHONPATH=.:experiments/container_independence`.

## Scripts written (independent code paths)

| script | what it does differently from the original |
|---|---|
| `scripts/x6_proto_hub_verify0_zerocopy.py` | Checks zero-copy by comparing **buffer address ranges** (pandas block-manager arrays, Arrow buffer addresses) with hub array pointers. Also tests pandas 3 CoW, multi-block pandas, 4-column polars and arrow, and whether egress output aliases the user's input. |
| `scripts/x6_proto_hub_verify0_fidelity.py` | 33 adversarial fixtures that the original 30 do not cover: NaN in a key level, int keys, nullable dtypes, int64 > 2^53, bool, time-major order, unused levels, B/timedelta/float index, MultiIndex columns, empty input, mixed per-instance freq, NaN+null in one column, tz, Enum, large_string, and others. Equality is checked by explicit per-level, dtype, freq and NaN/null-mask comparisons. |
| `scripts/x6_proto_hub_verify0_bench.py --grid dev` | Splits current ingress into mtype validation and the rest, and adds an **unchecked pandas-conversion** baseline. Also times hub vs current on **time-major** input, and compares slicing like-for-like (DataFrame vs DataFrame). |
| `scripts/x6_proto_hub_verify0_mem.py --grid dev` | Peak RSS with inputs **read from parquet**, as a real reader lays them out, in a fresh process per case, 3 reps, median. It also reports an analytic count of the bytes the hub allocates. |
| `scripts/x6_proto_hub_verify0_forecast.py --grid dev` | Compares the HubNaive and batched kernels with sktime NaiveForecaster on NaN-at-end, unequal-length, 2-column, ME and Period inputs, with `check_freq=True` (the original used `check_freq=False`). Re-times current vs hub. |
| `scripts/x6_proto_hub_verify0_c1.py` | Checks narwhals `collect()` defaults and `backend=` options per source, and the identity of the duckdb relation that egress returns. |

I also re-ran the original `x6_fidelity.py` (with both envs) and `x6_pushdown.py --grid dev`. For the latter I set
`X6_SCRATCH` to the session scratchpad, because its default path points to the old local machine.

## Results

### R1. Original fidelity matrix: reproduced exactly
```
$EXP_PY  -W ignore experiments/container_independence/scripts/x6_fidelity.py
  {"hand": {"pass": 29, "n/a": 1}, "nw": {"pass": 22}, "cap": {"pass": 26}}   LOC 449 / 120 / 145 / core 67
$EXP_PY3 -W ignore experiments/container_independence/scripts/x6_fidelity.py
  {"hand": {"pass": 29, "n/a": 1}, "nw": {"pass": 21}, "cap": {"pass": 26}}
```
The failure reasons match the report.

### R2. Adversarial fixtures (hand-written spokes): 25/33 pass (pd 2), 26/33 pass (pd 3)
`$EXP_PY -W ignore …/x6_proto_hub_verify0_fidelity.py` (and `$EXP_PY3`)

| fixture | result | severity |
|---|---|---|
| `pd_nan_in_key_level` | **silent data corruption**: input keys `['s0', nan, 's2', 's3']` become hub keys `['s0','s3','s2','s3']`, and the output has 3 instances. Cause: `np.asarray(levels)[code]` with code −1 picks the last level value. | high |
| `pl_nan_and_null_same_col`, `pa_nan_and_null_same_col` | NaN/null distinction lost: every NaN comes back as null (`fill_nan(None)` / `mask=isnan`). | medium |
| `pd_int64_gt_2p53`, `pl_int64_big` | int columns are routed through float64: 9007199254740993 → 9007199254740992 | medium |
| `pd_series_timedelta` | freq lost (pd 2); TypeError crash (pd 3) | low |
| `pd_multiindex_columns` | column MultiIndex flattened to an Index of tuples | low |
| `pd_empty` | index differs (pd 2 only) | low |
| passes | int keys, nullable Float64/Int64 with NA, float32, bool, time-major, unused levels, B freq, float index, `datetime64[s]`, single-row instances, duplicate timestamps, `string[pyarrow]` keys, polars tz/null-key/Float32/two keys/int time/Enum, arrow large_string/`timestamp[ms,UTC]`/time-major | |

**Mixed per-instance frequency panel** (instance a daily, instance b hourly):
- The hub stores one global `tmeta.freq="D"`, taken from instance 0.
- The "trusted" pandas view of instance b carries `freq=<Day>` on hourly data. `pd.DatetimeIndex(values, freq=D)`
  rejects it: `Inferred frequency h … does not conform to passed frequency D`.
- `HubNaive` predicts b at `2000-01-02 09:00, 01-03 09:00, …`. The correct times are `01-01 10:00, 11:00, 12:00`.
  The output is **silently wrong**.
- Current sktime raises `ValueError: Frequencies from two sources do not coincide` for the panel fit. So
  today's behaviour is loud and the hub's is silent.

### R3. Zero-copy and writability (address-range method)
`$EXP_PY` / `$EXP_PY3 -W ignore …/x6_proto_hub_verify0_zerocopy.py` (500x40)

| source | pd 2.3.3 | pd 3.0.6 |
|---|---|---|
| pandas MI, single float block, 1 or 4 cols | values in block ✔, **writeable**, egress output aliases input; mutating the round-trip output **changes the user's input** | values in block ✔ but **read-only** (CoW); the write raises |
| pandas MI, 2 blocks (frame built column-wise) | **copy** (not zero-copy) | copy |
| polars long (from pandas), 1 and 4 cols | values and time in source ✔, read-only, F layout | same |
| arrow long (from pandas), 1 col | values and time in source ✔, read-only | same |
| arrow long, 4 cols | values copied (stacking), time ✔ | same |

With inputs **read from parquet** (R4), values were *not* zero-copy for 4-column polars or for 1-column arrow
(multi-chunk → `combine_chunks`).

### R4. Peak ingress memory, inputs read from parquet (10000x100)
`$EXP_PY -W ignore …/x6_proto_hub_verify0_mem.py --grid dev`

| src | cols | current MB | hub MB | ratio | hub bytes allocated (analytic) | values zero-copy |
|---|---|---|---|---|---|---|
| pandas | 1 | 10.2 | 0.02 | (n/a) | 7.7 MB | yes |
| polars | 1 | 116.3 | 6.5 | 18× | 0.08 MB | yes |
| arrow | 1 | 94.9 | 14.9 | 6.4× | 15.3 MB | **no** |
| pandas | 4 | 9.8 | 0.02 | (n/a) | 7.7 MB | yes |
| polars | 4 | 163.2 | 37.2 | **4.4×** (report: 22×) | 30.6 MB | **no** |
| arrow | 4 | 139.8 | 45.4 | 3.1× (report: 4.3×) | 38.2 MB | no |

- **Methodology flaw.** For pandas, the VmHWM delta is 0.02 MB although the hub allocates at least 7.7 MB (the time copy).
  The `VmHWM − RSS_before` method misses allocations that reuse pages already resident in the
  allocator from earlier, freed work (read_parquet, set_index). It therefore under-counts both sides, depending on
  what ran before. The report's pandas figure (53.3 → 15.4 MB) and mine (10.2 → 0.02) differ by about 5×, so
  the pandas ratio is not robust.
- **The polars 4-column 22× is a fixture artifact.** The original built polars frames from one pandas
  2D block, so the column buffers happened to be adjacent and `to_numpy(order="fortran")` was zero-copy.
  A frame read from parquet needs a 2D copy (about 31 MB). The realistic ratio is about 4.4×.

### R5. Ingress decomposition and unchecked baseline
`$EXP_PY -W ignore …/x6_proto_hub_verify0_bench.py --grid dev`

| 10000x50 (2000x50 in brackets) | hub ms | current ms | current/hub | share of current spent in mtype check | unchecked pandas conversion / hub |
|---|---|---|---|---|---|
| pandas_mi | 2.86 (1.06) | 45.0 (9.6) | **15.7×** (9.0×) | 1.4% (6.5%) | — |
| polars `__index__` | 6.88 (2.92) | 1931 (391) | **281×** (134×) | 46% (62%) | **9.4×** (6.1×) |
| polars_long | 6.28 (2.88) | 109 (28.5) | 17.4× (9.9×) | — | 10.6× (5.5×) |

- The headline 309× / 281× for polars_panel measures sktime's slow polars converter and check. It does not
  measure the hub architecture. A plain, unchecked `pl.to_pandas().set_index()` is only 6–9× slower than the hub.
- The mtype check is 46–62% of the current polars path, not the "93%" in the report's profile (measured at 2000x50
  under cProfile, which inflates Python-call-heavy code).
- The hub performs **no validation**: no monotonic-time check and no duplicate check. The comparison
  is therefore validation-plus-conversion vs conversion-only, and the report does not state this explicitly.

**Time-major input** (`sort_index(level="time")`, typical of SQL `ORDER BY time`), also in `x6_proto_hub_verify0_bench.py`:
- Hub ingress becomes 7.3–16× slower than on sorted input (pandas 1.32 → 9.7 ms at 2000x50; 2.87 → 46.7 ms at 10000x50).
- The **current path degrades even more**, by 23–27× (11.3 → 261 ms; 49 → 1346 ms).
- So on the same time-major input the hub is still 27–29× faster (pandas) and 14–15× faster (polars).
- The report listed this comparison as "unmeasured". The hub advantage holds.
- Polars values are not zero-copy after the permutation, and the round trip restores the original order (equal = True).

### R6. Slicing, like-for-like
Same script. Current `items()` yields a **DataFrame with freq `<Day>`**.

| per slice | 2000x50 µs | 10000x50 µs |
|---|---|---|
| current items (DataFrame, freq) | 101.5 | 116.3 |
| hub view | 1.51 (67×) | 1.47 (79×) |
| hub → pandas **Series**, trusted freq (the report's "26 µs") | 24.2 | 24.7 |
| hub → pandas **DataFrame**, trusted freq | 53.1 (**1.9×** cheaper) | 51.2 (**2.3×**) |
| hub → pandas DataFrame, validated freq (public ctor) | 236.7 (**2.3× slower** than current) | 239.2 (**2.1× slower**) |

- The report's 4.1× compares a Series against the current path's DataFrame.
- The only safe public-API variant (validated freq) is *slower* than today.
- The trusted variant needs private `_simple_new`, and it is wrong for mixed-frequency panels (R2).

### R7. Forecaster kernels
`$EXP_PY -W ignore …/x6_proto_hub_verify0_forecast.py --grid dev`

- **Equality with `check_freq=True`:**
  - Passes for last/mean/drift on NaN-at-end, unequal length, `ME` and Period M, in both loop and batched form.
  - **All 6 two-column cases fail.** The kernels read only `values[:, 0]`.
  - Drift with NaN at the window end: sktime raises `ValueError`, while the hub returns 10 NaN rows without warning.
- **Speed:**

| size | current | hub loop | hub batched |
|---|---|---|---|
| 200x100 | 2.23 s | 6.0 ms (374×) | 1.9 ms (1175×) |
| 1000x100 | 10.9 s | 23.1 ms (472×) | 2.7 ms (4021×) |

  These are consistent with the report.
- I did **not** re-run the "unchanged pandas forecaster on hub views 1.01–1.28×" variant.

### R8. Pushdown: re-run of the original script
`X6_SCRATCH=<scratchpad>/x6v0 $EXP_PY -W ignore …/scripts/x6_pushdown.py --grid dev` (5000x200 parquet, w=10)

| case | t_min s | peak MB |
|---|---|---|
| pd_current_ingress | 0.224 (report: 0.684) | 163.2 |
| pl_full | 0.069 | 95.8 |
| pl_tail | 0.061 | 96.6 |
| pl_tail_streaming | 0.064 | 96.6 |
| duck_full | 0.187 | 158.8 |
| duck_qualify | 0.084 | **39.4** |

- The memory results reproduce: duckdb 4.0× lower peak, 2.2× faster; polars shows no peak reduction.
- The claim that the hub is **"7–11× faster than today's pandas ingress" is not reproduced**: the hub paths are 1.2–3.7× faster now,
  because the current path ran 3× faster in this run.
- duck_full has the same peak as the pandas current path (159 vs 163 MB).
- The current case only does ingress, while the hub cases also predict, so the comparison favours the current path (fine).

### R9. C1 / C2
`$EXP_PY -W ignore …/x6_proto_hub_verify0_c1.py` (narwhals 2.27.1)

- **`collect()` defaults:**
  - polars lazy → polars
  - dask → **pandas**
  - duckdb → **pyarrow**
- **`collect(backend=…)`:** accepts only polars, pandas or pyarrow. `dask` and `duckdb` raise
  `ValueError: Unsupported backend`. So the collected result can be forced to one of the three eager backends, never
  kept lazy. **C1 is true.**
- **Hub egress for duckdb:** returns a `DuckDBPyRelation` that is an `ARROW_SCAN` over the materialized data on duckdb's default
  connection. Querying it from the user's original connection fails (`not suitable for replacement scan`).
  "Restores duckdb" therefore means restoring the type only, not the connection or the query plan.
- **Dask:** npartitions comes back as 1 instead of 2.
- **C2 holds for the hand-written spoke.** It reads `MultiIndex.codes`, and names survive. But the same codes path has the NaN-key
  corruption bug (R2). The report's "NaN" MultiIndex fixture was NaN in *values*, not in keys.

## Verdicts (summary)

| claim | verdict | why |
|---|---|---|
| x6.1 ingress 4–15×, 309× polars_panel | partially_confirmed | Ratios reproduce (15.7×, 281×, 17×). But 309× is sktime converter and check overhead: an unchecked pandas conversion is only 6–9× slower than the hub. The hub does no validation. The check is 46–62% of the current path, not 93%. |
| x6.2 peak memory 3.5–22× lower | partially_confirmed | With parquet inputs: polars 1 col 18× ✔; polars 4 col **4.4×, not 22×** (fixture artifact); arrow 6.4×/3.1×. The pandas figure cannot be measured with the VmHWM method (it shows 0.02 MB while the hub allocates ≥ 7.7 MB). |
| x6.3 zero-copy contract | partially_confirmed | True for the original fixtures. On pandas 3 the pandas views are read-only. Multi-block pandas, parquet-read multi-column polars and multi-chunk arrow are copied. On pandas 2 the egress output aliases the user's input, so mutations propagate. |
| x6.4 fidelity 29/29, 22/30, 21/30, 26/30, LOC | partially_confirmed | Exactly reproduced on the original fixtures. The hand-written spoke fails 7–8 of 33 adversarial fixtures, including silent key corruption (NaN key), NaN→null and int64 precision loss. |
| x6.5 slicing 72×, pandas view 26 µs (4.1×) | partially_confirmed | View 67–79× ✔. Like-for-like DataFrame with trusted freq is 51–53 µs (1.9–2.3×). With validated freq the hub is 2.1–2.3× *slower* than current. Trusted freq is wrong for mixed-freq panels. |
| x6.6 numpy kernel 365–564× / 2339–4676× | partially_confirmed | Speed reproduces (374–472×, 1175–4021×) and equality holds on harder univariate fixtures. Multivariate input is silently wrong (only column 0), and the drift NaN case is silent where sktime raises. The pandas-view 1.01–1.28× part was not re-run. |
| x6.8 pushdown 4× memory duckdb, "7–11× faster than current" | partially_confirmed | Memory and duck time reproduce. The 7–11× is not reproduced (now 1.2–3.7×). |
| x6.10 C1 true but irrelevant, C2 does not arise | partially_confirmed | C1 is confirmed, and `backend=` can only force an eager polars, pandas or pyarrow result. The duckdb egress relation is not bound to the user's connection, and npartitions is lost. C2 is confirmed for names, but the codes path corrupts NaN keys. |

## Missed or wrong in the original
1. **Silent wrong results** where sktime today raises or keeps the data correct:
   - NaN instance key: instances get merged.
   - Mixed per-instance freq: wrong freq and wrong forecast timestamps.
   - Multivariate input to the Naive kernels: only column 0 is used.
   - Drift with NaN: NaN output instead of an error.
2. **The hub skips all validation.** Ingress ratios compare check-plus-convert with convert-only. A fair architecture comparison also
   needs an "unchecked pandas" baseline (6–10× on polars here, not 280×).
3. **The 22× polars memory number and "polars multi-column values zero-copy" depend on the fixture.**
   The original built polars frames from pandas, which gives adjacent buffers. Parquet-read frames are copied.
4. **The VmHWM-delta memory method under-counts small allocations** that reuse resident freed pages. The pandas hub
   shows 0.02 MB while allocating 7.7 MB, so the absolute numbers and the pandas ratio are unreliable. Use analytic byte counts
   or a memray/LD_PRELOAD allocator tracer as a cross-check.
5. **On pandas 3 (CoW) pandas-sourced hub values are read-only**, the same as polars and arrow. On pandas 2 they are writeable and
   the egress output aliases the input.
6. **The slicing comparison is apples to oranges** (Series vs DataFrame). The safe public-API pandas view is slower than today's `items`.
7. **NaN vs null and int64 > 2^53 are lost in all spokes**, because every numeric column goes through float64. The report mentions
   "nullable → float NaN + schema restore" but does not say that the null and NaN distinction and int precision are lost.
8. **Time-major (interleaved) input was listed as unmeasured for the current path.** Measured here: the hub is 7–16× slower than on sorted
   input, but the current path is 23–27× slower on the same input, so the hub is still 14–29× faster than today.
9. **The pushdown default scratch path was hard-coded** to the old local machine. The time ratio against current is unstable
   (0.684 s then, 0.224 s now).

## Caveats
- The machine is shared, but all ratios were measured within one process with min-of-repeats.
- Only dev sizes were run.
- The forecaster "pandas view per slice" sub-claim was not re-run.
