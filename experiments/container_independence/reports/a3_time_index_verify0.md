# a3_time_index_verify0 — adversarial verification of workstream a3_time_index

Lens: re-run the experiments and check each pandas-vs-polars semantic difference independently.
Envs: `$EXP_PY` (pandas 2.3.3) and `$EXP_PY3` (pandas 3.0.6), polars 2.0.0, narwhals 2.27.1,
dask 2026.8.0. Cloud machine (4 cores), 2026-10-18. All commands run from the WT root with
`PYTHONPATH=.:experiments/container_independence <py> -W ignore <script>`.

New scripts (independent code paths):

* `scripts/a3_time_index_verify0_semantics.py`: stress test of the `Step`/`TimeOps` prototype.
  It uses 200 **random** on-grid cutoffs from 1950–2100 (the original used 150 consecutive
  cutoffs from 2000) and H=24. Truth is `pd.date_range(cutoff, periods=H+1, freq=<offset object>)`
  and sktime `fh.to_absolute_index`. Instants are compared as UTC int64 ns. New conditions:
  non-midnight time of day, the NY/Sydney/Kolkata time zones, tz-aware MS/ME/B/W/SME,
  multiples (2B, 3MS, 2W-SUN, 2QS), QS-FEB, and integer indexes.
  Output: `results/a3_time_index_verify0_semantics_pd{2,3}.json`.
* `scripts/a3_time_index_verify0_cost.py --grid dev`: compares the original P1 (pure polars)
  and P2 (hybrid) with a new **P1d, a deduplicating pure-polars variant**. The runs are
  interleaved and the min of 7 is taken. Equality is checked on the sorted frames, and the
  row count is checked too. The script also instruments a **real** vectorised NaiveForecaster
  fit+predict: it counts calls, times `_to_absolute`/`_to_relative`, and reads lru_cache hits.
  Output: `results/a3_time_index_verify0_cost_dev.json`.
* `scripts/a3_time_index_verify0_probes.py`: end-to-end probes. It runs a CBD holiday case
  through NaiveForecaster, a Period round trip via `convert_to`, Date/naming on polars input,
  dask convert_to plus the adapter, and narwhals `offset_by("1mo")`/`truncate` on 6 backends.
  Output: `results/a3_time_index_verify0_probes_pd{2,3}.json`.

Original scripts re-run: `a3_offset_equiv.py` (pd2 + pd3), `a3_timeindex_check.py` (pd3; the
original re-ran pd2 only) and `a3_time_engine_cost.py --grid dev --profile-max-series 0` (pd2).

---

## Verdicts

| id | claim (short) | verdict |
|---|---|---|
| a3.1 | polars/numpy reproduce fh.to_absolute exactly, 23 cases, 0/3,600 | **partially confirmed**: true on the original test set, false outside it |
| a3.2 | CustomBusinessDay holidays dropped, 1,095/3,600 wrong | **confirmed** (re-run and end to end) |
| a3.3 | naive polars intervals: ME 992, QE 900, B 3,300 wrong | **confirmed** (re-run, both pandas versions) |
| a3.4 | tz daily fh differs pd2 vs pd3, 300/3,600 | **confirmed**, and it generalises to other zones |
| a3.6 | hybrid costs 0.6x–2.5x pure polars | **partially confirmed**: the "<1x" part is an artifact of P1 not deduplicating |
| a3.7 | per-instance pattern 1.0–1.4 ms/inst, 456x–1,099x | **partially confirmed**: ms confirmed; the ratio is noisy, and its end-to-end meaning is much smaller |
| a3.9 | polars mtypes lose time semantics (Period→END, Date→Datetime, y→0) | **confirmed** |
| a3.10 / a3.11 | dask mtype unusable + lexicographic int order; narwhals no "mo" on pyarrow/dask | **confirmed** |
| a3.13 | 44/156 forecasting files use pandas-only time semantics | **confirmed** (independent grep: 44/156) |

### a3.1: exactness of the Step prototype (partially confirmed)

Re-running `a3_timeindex_check.py` on pd3 reproduces it: 0/3,600 for every case,
CBD 1,095, and infer 22/22. `a3_offset_equiv.py` also reproduces on pd2 and pd3.

The independent stress test (`a3_time_index_verify0_semantics.py`, out of 4,800 pairs per case)
matches on every baseline frequency with random cutoffs between 1950 and 2100. That covers D, B,
W-SUN, MS, ME, QE, YE, SME, BMS, BME, 2B, 3MS, 2W-SUN, QS-FEB and 2QS, plus MS/ME/B at 09:30
(polars and numpy both 0). It **breaks outside the tested set**:

| case | polars vs sktime (pd2 / pd3) | numpy | cause |
|---|---|---|---|
| SME at 09:30 | 4,800 / 4,800 | 4,800 | halfmonth rule rebuilds dates with `pl.datetime(y,m,1)`, so the **time of day is dropped** |
| SME tz=America/New_York | 4,800 / 4,800 | NA | same rule drops the **tz** (returns naive midnight) |
| BMS / BME at 09:30 | 0 / 0 | **4,800 / 4,800** | numpy bmonth path drops the time of day |
| W-SUN tz=America/New_York | **882 / 2,140** | NA | W-xxx maps to `fixed 7d`. pandas `Week` on tz data is wall-clock (calendar) |
| MS NY / ME Sydney / B NY | 1,107 / 1,836 / 122 (pd2); 11 / 0 / 0 (pd3) | NA | pd2 differences are all after 2037 (pandas 2 applies no DST past 2037; polars does). pd3: 11 differences on 2100-04-01, where polars appears to stop DST after 2099 |
| int index step 5 (0,5,10,…) | 4,800 / 4,800 | – | sktime treats an int index as unit step: `NaiveForecaster` on index 0..45 step 5 predicts **46,47,48**. The prototype `infer_step` returns `Step(int, n=5)` → 50,55,60 |

So the claim "exact on all 23 cases" holds only for the 23 tested cases. These are all
midnight-aligned, tz-free except D/h Berlin, and int-free. The fixes are small: keep the
time-of-day offset in halfmonth/bmonth, reapply the tz, and map tz-aware W-xxx to
`calday(7n)`. But they show that the Step mapping needs a test matrix of freq × time-of-day × tz.
The integer-step choice is a real semantic decision the report calls "trivial" (§6.6). It is not
trivial: sktime's convention is unit steps, and the prototype silently changes it.

### a3.2: CustomBusinessDay holidays (confirmed)

Re-run: sktime vs holiday-aware truth gives 1,095/3,600 on pd3 (and the original pd2 re-run).
The independent end-to-end probe, NaiveForecaster on a CBD-indexed series with a holiday on
2001-01-15, predicts `2001-01-11, 12, **15**, 16, 17` on both pandas versions. Truth is
`11, 12, 16, 17, 18`. The code path explains it: `_to_absolute` uses `to_offset(fh.freq)`, and
`fh.freq` returns `self._freq.freqstr == "C"`, which loses the holidays.

### a3.3: naive polars interval rules (confirmed)

`a3_offset_equiv.py` re-run on pd2 and pd3 gives NAIVE `offset_by(kmo)` for ME = 992,
`offset_by(kq)` for QE = 900, and `offset_by(kd)` for B = 3,300, identical in both versions.
The narwhals probe independently shows the same pitfall: `offset_by("1mo")` on 2000-02-29 gives
2000-03-29 on all backends that support it.

### a3.4: tz daily, pandas 2 vs pandas 3 (confirmed and generalised)

`a3_offset_equiv.py`, D tz=Berlin: sktime vs `date_range` is 300/3,600 on pd2 and 0 on pd3.
The independent run with other zones gives pd2 sktime vs date_range = 248/4,800 (New York),
220 (Sydney) and 0 (Kolkata, no DST); pd3 gives 0 everywhere. So the difference comes from the
pandas major version and appears wherever DST exists.

### a3.6: hybrid vs pure polars (partially confirmed)

Re-run of `a3_time_engine_cost.py --grid dev` (pd2): P2/P1 ranges 0.4–1.9 (MS 10k: P1 28.7 ms,
P2 12.8 ms; D 10k: P1 14.0, P2 14.8). eq=True in all 16 configurations. This is consistent with
the claim.

The verifier's interleaved run with the deduplicating polars variant (P1d):

| freq | N | P1 | P1d | P2 (ms) | P2/P1 | **P2/P1d** |
|---|---|---|---|---|---|---|
| MS aligned | 10,000 | 23.17 | 12.02 | 12.67 | 0.55 | **1.05** |
| MS ragged | 10,000 | 21.87 | 10.17 | 11.26 | 0.51 | **1.11** |
| D aligned | 10,000 | 10.89 | 10.71 | 11.37 | 1.04 | 1.06 |
| D ragged | 10,000 | 9.01 | 8.88 | 10.24 | 1.14 | 1.15 |
| all 16 configs | | | | | 0.51–3.07 | **1.05–1.98** |

The "hybrid is faster than pure polars at 10k" reading (§0.4, §7) is an **artifact**: P1
evaluates a per-row formatted interval string on N×H rows. A pure-polars variant that
deduplicates first is never slower than the hybrid. Against the fair baseline the hybrid costs
1.05x–2x, an absolute overhead of ≤ ~1.5 ms at 10k series. The architectural conclusion
("pandas as a time engine on unique cutoffs is cheap") still holds, and is if anything
strengthened: the cost is bounded and small. The report should not present the hybrid as
faster.

### a3.7: per-instance cost (partially confirmed)

The re-run gives P4 = 11.05/10.66 s (MS 10k) and 14.58/15.91 s (D 10k), i.e. 1.0–1.6 ms per
instance. P4/P1 = 386x–1,507x (claimed 456x–1,099x). The ratio range moves with the noisy P1
denominator, and against P1d it would differ again, so treat the "x" as an order of magnitude
(~10^3).

Independent instrumentation of a **real** vectorised `NaiveForecaster` fit+predict (pd2):

* 1 `_to_absolute` call per instance (100→100, 1000→1000) and 0 `_to_relative`.
* `_to_absolute` takes 1.1–1.6 ms per instance inclusive, matching P4.
* **lru_cache hits = 0** even in the aligned case, where all 1,000 instances share one cutoff.
  `_HashIndex` defines `__hash__` but no `__eq__`, so every lookup misses. The cache adds the
  `hash_pandas_object` cost and never pays it back. The original names the hash cost but does
  not note that the cache can never hit.
* fit+predict costs 11.7–13.9 ms per instance in total. `_to_absolute` is only **8–12%** of it.

So the 10^3x is a speed-up of one component. End to end for NaiveForecaster, removing
`to_absolute` alone gives ≤ ~1.1x. Even the original's broader 17–27% bookkeeping share (claim
a3.8, not re-run here, my partial measure 8–12%) caps the gain at ~1.2–1.4x. The headline
"456x–1,099x" must not be read as a forecaster speed-up.

### a3.9: polars mtypes lose time semantics (confirmed)

The `convert_to` probe (pd2/pd3) shows:

* `period[M]`: 2000-01 → polars `2000-01-31 00:00` (Datetime ns on pd2, us on pd3).
* `period[Q]`: → 2000-03-31.
* `W-SUN` (1999-12-27/2000-01-02): → 2000-01-02.
* Back conversion gives a `DatetimeIndex`. The Series column is `__index__0`. The panel keeps
  `__index__time`.

NaiveForecaster on a `pl.Date` series returns the schema
`{'__index__0': Datetime(ns) [pd2] / Datetime(ms) [pd3], '0': Float64}`, with names and dtype
changed exactly as claimed.

### a3.10 / a3.11: dask and narwhals (confirmed)

Dask: `convert_to(df, "dask_panel")` works, but converting back raises `TypeError: No valid
mtype could be identified ... dask_expr._collection.DataFrame`. The adapter round trip of int
time gives `[0, 1, 10, 11, 2, …, 9]`, and `check_is_mtype` = False.

Narwhals 2.27.1: `offset_by("1mo")` raises NotImplementedError on pyarrow and dask, and
`truncate("1mo")` raises on dask. It works on pandas, polars, polars-lazy and duckdb, and
`offset_by("1h")` works on all 6 backends. There is no `dt.month_end` and no
`add_business_days` attribute.

### a3.13: census (confirmed)

An independent grep over 156 non-test `sktime/forecasting` files for
freq/freqstr/infer_freq/to_offset/DateOffset/pd.offsets/Period*/period_range/to_period/to_timestamp
matches 44 files, the same number. A wider pattern that also includes fh./cutoff/
ForecastingHorizon/DatetimeIndex/date_range matches 118 (the original reports 111). This is
regex-level only.

---

## Missed or wrong in the original

1. The prototype's exactness was only shown for midnight, tz-free (except D/h Berlin) cases.
   SME loses time-of-day and tz, numpy BMS/BME lose time-of-day, and tz-aware W-xxx is wrong
   (882–2,140/4,800). See a3.1.
2. Integer index semantics: sktime uses unit steps on any int index (index step 5 → predicts
   +1, +2, +3). The prototype's `infer_step` would change this to +5k. This decision has to be
   made explicitly, and the report calls it "trivial".
3. The "hybrid faster than polars" statement is an artifact of a non-deduplicating baseline. A
   fair P2/P1d is 1.05–1.98x.
4. The `_to_absolute` lru_cache is structurally ineffective (0 hits; `_HashIndex` has no
   `__eq__`). This is a cheap fix: about 1–1.6 ms per instance is saved whenever cutoffs
   repeat, which is the common aligned-panel case.
5. The 10^3x per-instance ratio concerns one component that is ~8–12% of NaiveForecaster
   fit+predict wall time. End-to-end impact is ≤1.1–1.4x. The report says this in §7.1, but the
   summary and impact statements lead with 456x–1,099x.
6. Far-future tz: pandas 2 applies no DST after 2037, and polars appears to drop DST after
   2099. This is a backend-dependent semantic difference that cannot be removed by a Step
   abstraction. It is low practical relevance but should be documented if polars becomes the
   time engine.
7. Not checked here (unverifiable in this pass): a3.5's infer-step alias claim beyond the
   re-run (22/22 reproduced), a3.8 (17–27% share, not re-run), and a3.12 (prior-art venv not
   re-run).

## Commands and raw outputs

```
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/a3_offset_equiv.py      # NAIVE ME 992, QE 900, B 3300; D tz 300
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/a3_offset_equiv.py      # same NAIVE; D tz sktime-vs-date_range 0
PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore experiments/container_independence/scripts/a3_timeindex_check.py   # 0/3600 all; C 1095; infer 22/22
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/a3_time_engine_cost.py --grid dev --profile-max-series 0
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/a3_time_index_verify0_semantics.py   # and $EXP_PY3
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/a3_time_index_verify0_cost.py --grid dev
PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore experiments/container_independence/scripts/a3_time_index_verify0_probes.py      # and $EXP_PY3
```

a3_time_engine_cost re-run (pd2; ms): MS 10k aligned P1 28.65 / P2 12.77 / P3 13.98 / P4 11,051.
MS 10k ragged 21.04 / 12.12 / 40.18 / 10,661. D 10k aligned 13.99 / 14.80 / 18.29 / 14,581.
D 10k ragged 10.56 / 12.14 / 49.86 / 15,915. N=1000 P4 998–1,540. eq=True in all configurations.

Real NaiveForecaster (pd2):

| N | freq | fit (s) | predict (s) | `_to_absolute` calls | `_to_absolute` time (s) | cache hits | share | ms per instance |
|---|---|---|---|---|---|---|---|---|
| 100 | MS | 0.42 | 0.93 | 100 | 0.109 | 0 | 8.1% | 13.5 |
| 100 | D | | | | 0.144 | | 12.3% | 11.7 |
| 1000 | MS | 4.47 | 9.47 | 1000 | 1.131 | 0 | 8.1% | 13.9 |
| 1000 | D | | | | 1.568 | | 12.4% | 12.6 |

Caveat: sktime emits a `remember_data` FutureWarning per instance in this run (2,204 lines to
stderr). This inflates the absolute per-instance cost but applies equally to all rows.
