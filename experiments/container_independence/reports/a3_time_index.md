# a3 — Time index / forecasting horizon: pandas coupling and non-pandas equivalents

Workstream key: `a3_time_index`. All numbers below come from scripts in
`experiments/container_independence/scripts/a3_*.py`, run from the worktree root with

    PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore <script>   # pandas 2.3.3
    PYTHONPATH=.:experiments/container_independence $EXP_PY3 -W ignore <script>   # pandas 3.0.6

(polars 2.0.0, numpy 2.2.6, pyarrow 26, duckdb 1.5.6, narwhals 2.27.1, dask 2026.8).
Prior-art checks ran in a private venv (`scratchpad/venvs/a3`, utilsforecast 0.2.17,
functime 1.0.0, u8darts 0.41.0, pandas 3.0.6, polars 2.0.0) with `python -I`.
Raw outputs are in `results/a3_*.{json,txt}`. Anything not measured is labelled
"unmeasured/expected".

---------------------------------------------------------------------------------

## 0. Verdict (TL;DR)

1. **Time arithmetic is not the obstacle to decoupling.** Every pandas frequency that
   sktime users realistically use (int, fixed durations h/min/s, D, W-xxx, B, C with a
   holiday list, MS/ME, QS/QE, YS/YE, 2ME etc., BMS/BME, SME, tz-aware D/h, and Period
   M/Q/W/D) is reproducible in polars and in numpy with **0 mismatches out of 3,600
   (cutoff, step) pairs per frequency** against sktime's own `ForecastingHorizon.to_absolute`
   (exp. c, prototype check). numpy cannot do time zones. pyarrow compute cannot add calendar
   months (it can only add durations). duckdb does months/years but has no business days.
   narwhals can only express calendar-month offsets on pandas/polars/duckdb (not on pyarrow
   or dask), and has no business-day API.
2. **The real coupling is that `freq` is a pandas object.** A frequency is stored as a pandas
   offset or freqstr and re-parsed, and it is inferred with `pd.infer_freq`. This causes
   measured bugs that have nothing to do with polars:
   (i) `CustomBusinessDay` holidays are silently dropped (fh stores freq as `"C"`):
   **1,095/3,600 (30%) forecast timestamps wrong**;
   (ii) tz-aware daily data: sktime's absolute fh **differs between pandas 2 and pandas 3**
   (+24h absolute vs calendar day): **300/3,600 (8.3%)** pairs move;
   (iii) alias drift after a polars round trip (`QS-JAN` becomes `QS-OCT`) makes an absolute fh
   built with `pd.date_range(freq="QS")` raise `ValueError: Frequencies from two sources do not coincide`.
3. **Proposed abstraction:** a small plain-python `Step` value object (kind ∈ {int, fixed,
   calday, month(anchor), bday(weekmask, holidays), bmonth, halfmonth}) plus a `TimeOps`
   backend with 5 operations (`cutoff`, `add_steps`, `steps_between`, `range`, `infer_step`).
   The prototype is in `protos/a3_timeindex.py` and has pandas, polars and numpy backends.
   It reproduces sktime exactly on all 23 frequency cases. It agrees with `pd.infer_freq` on
   22/22 inference cases. It fixes bugs (i) and (iii) by construction, and it makes (ii) an
   explicit choice (`calday` vs `fixed`).
4. **"Keep pandas only as the time engine on the small index" is cheap if done once per
   panel, and very expensive if done once per instance.** Building the prediction index
   (per-series cutoff, then 12-step absolute fh, then output keys) for 10,000 series ×
   50 points takes 17–42 ms in the hybrid (polars data, pandas offsets on the U unique
   cutoffs) and 23–39 ms in pure polars. That is within **0.4×–2.4×** of pure polars at
   every grid point. sktime's current pattern, one `fh.to_absolute_index` per instance in
   the vectorisation loop, costs **1.7–3.8 ms per instance**, which is **20–36 s for 10k
   series (525×–1,535× pure polars)**. In NaiveForecaster, per-instance fh/cutoff
   bookkeeping (2× get_cutoff, 2× check_fh, 1× fh construction, 1× to_absolute per instance)
   takes **~17–27% of fit+predict wall time (~9 ms/instance)**, both for pandas and for
   polars input. So the decision that matters is where the time-axis computation happens:
   once per panel, or once per instance. Which library computes it matters much less.
5. **sktime's current polars/dask path loses time semantics** (exp. a/b). PeriodIndex becomes
   period-END timestamps (month/quarter/week), with a class change and a cutoff type change.
   The forecaster output time dtype changes (pl.Date to Datetime(ns); us to ns on pandas 2).
   The polars Series output renames columns (`y` to `0`, `__index__time` to `__index__0`).
   Irregular data fails with a cryptic `TypeError` (same as pandas). LazyFrames and the
   natural polars layout (time column without the `__index__` prefix) are rejected. The dask
   mtypes are unusable with current dask (`python_type` tag bug), and the dask adapter
   scrambles integer time order.

Feasibility verdict: **decoupling ForecastingHorizon/cutoff from pandas is feasible and
low-risk** if (a) frequency becomes a library-neutral `Step` (not a pandas offset), (b) the
framework computes cutoffs and absolute horizons once per panel (vectorised), not per
instance, and (c) pandas stays available as the reference/fallback engine for exotic
offsets (FY5253, Easter, BusinessHour, calendar objects — unmeasured) via the hybrid path,
whose measured cost is ≤ ~20 ms at 10k series. The public `ForecastingHorizon` / `cutoff`
API can keep returning pandas objects (materialised lazily) for backward compatibility.

---------------------------------------------------------------------------------

## 1. Where the time axis is coupled to pandas (map)

### 1.1 Code points

| Component | pandas-specific mechanics |
|---|---|
| `forecasting/base/_fh.py` `ForecastingHorizon` | wraps a `pd.Index` (`_values`); `_check_values` coerces everything to `pd.Index`; delegates `__add__`, `__getitem__`, `max`, … to the pandas index; `freq` stored as pandas offset; `_freq.freqstr` re-parsed with `to_offset(fh.freq)` in `_to_absolute`; Period coercion (`_coerce_to_period`, incl. MonthBegin→"M" hack for pandas 3); `_HashIndex` uses `pd.util.hash_pandas_object` for the lru_cache key; `get_expected_pred_idx` builds MultiIndex tuples in python |
| `forecasting/base/_base.py` | `_cutoff` is a length-1 `pd.Index` carrying `.freq`; `_set_cutoff_from_y` → `get_cutoff(..., return_index=True)`; `_check_fh` → `check_fh(fh, freq=self._cutoff)` (freq consistency check by offset equality); `cutoff` property returns `pd.Index` |
| `datatypes/_utilities.py` `get_cutoff`, `get_time_index` | MultiIndex `to_frame().groupby(level=…).nth(-1)`; `sub_idx` calls `pd.infer_freq` when the index has no freq |
| `utils/datetime.py` | `_coerce_duration_to_int` (offset `.n` / Timedelta division), `_to_offset_compat` (pandas-3 period aliases), `infer_freq` (singledispatch on pandas/VectorizedDF), `_shift`, `_get_duration` |
| `utils/validation/series.py` | `VALID_INDEX_TYPES = (RangeIndex, PeriodIndex, DatetimeIndex, TimedeltaIndex)`; relative = Range/Timedelta/int, absolute = Range/Datetime/Period/int |
| `datatypes/_adapter/polars.py` | Period→Datetime via `to_timestamp(freq=index.freq)` (yields period END for M/Q/W); polars→pandas re-infers freq with `pd.infer_freq` (also on MultiIndex `levels[-1]`) |
| `datatypes/_adapter/dask_to_pd.py` | MultiIndex→string index `str(tuple)`, `from_pandas(sort=True)` sorts by that string |

### 1.2 Static census (non-test code), `scripts/a3_coupling_census.py`

Command: `PYTHONPATH=.:experiments/container_independence $EXP_PY experiments/container_independence/scripts/a3_coupling_census.py`
Output: `results/a3_coupling_census.json`

| pattern | forecasting lines / files | rest of sktime lines / files |
|---|---|---|
| `fh.to_absolute(` | 84 / 51 | 0 / 0 |
| `fh.to_absolute_index(` | 77 / 45 | 6 / 2 |
| `fh.to_relative(` | 123 / 63 | 2 / 1 |
| `fh.to_absolute_int(` | 14 / 8 | 0 / 0 |
| `fh.to_indexer(` | 22 / 12 | 0 / 0 |
| `get_expected_pred_idx(` | 32 / 18 | 1 / 1 |
| `self.cutoff` / `self._cutoff` | 487 / 100 | 27 / 5 |
| `.freq` / `.freqstr` | 95 / 31 | 96 / 23 |
| `infer_freq(` | 21 / 9 | 16 / 11 |
| `to_offset` / `DateOffset` / `pd.offsets.` | 23 / 8 | 48 / 18 |
| `PeriodIndex` / `pd.Period` / `period_range` | 112 / 26 | 103 / 31 |
| `.to_period(` / `.to_timestamp(` | 47 / 18 | 37 / 20 |
| `DatetimeIndex` / `date_range` | 128 / 33 | 156 / 55 |
| `get_level_values(-1)` / `levels[-1]` | 57 / 22 | 30 / 17 |

Summary: 156 non-test python files in `sktime/forecasting`. 111 of them touch time-axis
semantics in some form. **44 use pandas-only time semantics directly** (freq / Period /
offsets / to_period / to_timestamp). Examples are `naive/_naive.py` (`_y.index.freq`, l.367),
`mira.py` (`isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex))`, l.284), the adapters
(darts, prophet, neuralforecast, pytorch-forecasting) and the trend forecasters. The
remaining ~67 files reach time only through the `ForecastingHorizon` / `cutoff` API. That
API is therefore the natural seam: if `fh.to_relative / to_indexer / to_absolute` return
container-neutral values, most forecasters do not need to change. (Regex-based counts are
indicative, not exact.)

---------------------------------------------------------------------------------

## 2. pandas time semantics vs non-pandas equivalents

Mismatch counts are out of 3,600 (150 consecutive cutoffs × 24 steps) against sktime
`ForecastingHorizon(1..24).to_absolute(cutoff)` (script `a3_offset_equiv.py`, pandas 2.3
and 3.0, identical except the tz-D row). "NA" = no API in that library.

| pandas concept (used for) | polars | numpy | pyarrow compute | duckdb | narwhals (`a3_narwhals_time_probe.py`) |
|---|---|---|---|---|---|
| `RangeIndex` / int index (int time) | Int64 col, `+k*n` — exact | int arrays — exact | ints | ints | yes |
| `DatetimeIndex` naive | `Datetime(us/ns)`, `Date` | `datetime64` | `timestamp` | `TIMESTAMP` | yes |
| tz-aware `DatetimeIndex` | `Datetime(tz)`; calendar day = `offset_by("kd")` | **no tz** (UTC only) | storage only | `TIMESTAMPTZ` (ICU; untested) | `replace/convert_time_zone` |
| `PeriodIndex` / `Period` | **no Period dtype**; ordinal Int or period-start Datetime + Step: 0 mismatches (M,Q,W-SUN,D) | `datetime64[M]`/`[Y]`/`[D]` units are period-like: 0 | NA (calendar) | months via INTERVAL: 0 | no |
| `.freq` attribute | **none**; must be carried outside the data (Step) or inferred | none | none | none | none |
| `pd.infer_freq` | none built in; ~40-line polars `infer_step` (prototype) agrees 22/22 | — | — | — | — |
| Tick: h, 2h, 15min, D (naive) | `+pl.duration` / `offset_by`: 0 | `+timedelta64`: 0 | `pc.add(duration)`: 0 | `+to_hours/…`: 0 | `offset_by`: 0 (all 6 backends for h) |
| `Day` on tz-aware, across DST | `offset_by("kd")` = calendar day: 0 vs pandas 3, **300** vs pandas 2; `+duration(24h)`: 0 vs pandas 2, 300 vs pandas 3 | UTC+24h = pandas-2 semantics only | — | — | — |
| `W-SUN`, `W-WED` | `offset_by(7k d)`: 0 | `+7k D`: 0 | `pc.add`: 0 | `+to_days(7k)`: 0 | — |
| `MS`, `QS`, `YS` | `offset_by("k mo"/"3k mo"/"k y")`: 0 | `M8[M]+k`, `M8[Y]+k`: 0 | **NA** (`add` has no timestamp+month_day_nano kernel) | `+to_months/to_years`: 0 | pandas/polars/duckdb: 0; **pyarrow, dask: NotImplementedError** |
| `ME`, `QE`, `YE`, `2ME` | `offset_by(..).dt.month_end()`: 0. **Naive `offset_by("kmo")`: 992 wrong (ME), 900 wrong (`offset_by("kq")` for QE)** | `(M8[M]+k+1)→D −1`: 0 | NA | `last_day(c+to_months(k))`: 0 | naive offset_by: 44/144 wrong; `truncate("1mo")+(k+1)mo−1d`: 0 (pd/pl/duckdb) |
| `B` | `dt.add_business_days(k)`: 0 (`offset_by` has no "bd" unit; naive `offset_by(kd)`: 3,300 wrong) | `np.busday_offset`: 0 | NA | NA (no builtin) | **NA** (no business-day API) |
| `C` (CustomBusinessDay + holidays) | `add_business_days(holidays=…)`: 0 vs holiday-aware truth | `busday_offset(holidays=…)`: 0 | NA | NA | NA |
| `BMS`, `BME` | month_start/end + `add_business_days(0, roll=…)`: 0 | `busday_offset(…, roll)`: 0 | NA | NA (untested) | NA |
| `SME` (semi-month end) | half-month ordinal expression: 0 | half-month ordinal: 0 | NA | NA (untested) | NA |
| FY5253, Easter, BusinessHour, `CustomBusinessMonth*` with calendar objects | none (unmeasured/expected) | none | none | none | none |
| relative ↔ absolute (`to_relative`, steps-between) | `(ym_b−ym_a)//n`, `total_nanoseconds//step`, `business_day_count`: round trip 0 mismatches | `M8[M]` diff, `busday_count`: 0 | `months_between`, `days_between`, … exist (untested) | `datediff` (untested) | — |
| `TimedeltaIndex` relative fh | `Duration` dtype | `timedelta64` | `duration` | `INTERVAL` | Duration |

Notes:
* The polars month-end pitfall is real. `offset_by("1mo")` from Feb 29 gives Mar 29. Month
  ends need `.dt.month_end()`, so the *anchor* (start/end) must be known. The anchor is the
  information a pandas offset carries and a polars interval string does not.
* `pd.infer_freq` cannot infer `SME` either (returns None). Both it and the polars
  prototype return None on irregular data and on data with a single gap.

---------------------------------------------------------------------------------

## 3. Experiment (a): round trip through sktime's polars / dask mtypes

Script: `scripts/a3_roundtrip.py`. Results: `results/a3_roundtrip_pd{2,3}.{json,txt}`. Each
case is Series (`pd.DataFrame`↔`pl.DataFrame`, `dask_series`), Panel
(`pd-multiindex`↔`polars_panel`, `dask_panel`) or Hierarchical
(`pd_multiindex_hier`↔`polars_hierarchical`, `dask_hierarchical`), times 18 index kinds,
24 time points, converted with `convert_to` there and back.

Findings (identical across Series/Panel/Hierarchical unless noted):

| index | intermediate polars dtype | what is lost after the round trip |
|---|---|---|
| `period[M]`, `period[Q-DEC]`, `period[W-SUN]` | `Datetime(ns)` (pd2) / `Datetime(us)` (pd3) | class PeriodIndex→DatetimeIndex; dtype period→datetime64; cutoff `Period('2001-12','M')` → `Timestamp('2001-12-31 00:00')`; **values = period END date at midnight** (not period start, not the end instant); freq becomes `ME`/`QE-DEC`/`W-SUN` |
| `period[D]` | Datetime | class/dtype/cutoff type change, values equal |
| `dt` D, h, B, W-SUN, MS, ME, QE, YE, tz=Europe/Berlin h, irregular | Datetime (tz preserved) | nothing (Series/Panel). Hierarchical: freq *gained* (None → inferred) because `convert_polars_to_pandas.set_freq` mutates `levels[-1].freq` |
| `dt` QS | Datetime | freq name `QS-JAN` → `QS-OCT` (pd.infer_freq naming) |
| `RangeIndex` | Int64 | pd2: RangeIndex → `Index[int64]` (Series/Panel); pd3: preserved; pd3 turns a step-10 int index into RangeIndex |
| unit | | pandas 2: `datetime64[ns]` throughout; pandas 3: `us` |

Dask (`_adapter/dask_to_pd.py`):
* `convert_to(<dask frame>, "pd…")` fails for **every** case with current dask (dask-expr
  collections). The cause is the mtype tag `"python_type": "dask.dataframe"`, which the base
  checker splits into module `dask` + class name `dataframe`. The class is `DataFrame`, so
  `_isinstance_by_name` fails. The script therefore falls back to the adapter function
  directly.
* With the adapter, dask (pandas-backed) **keeps `period[...]` dtypes**. It drops `freq` for
  Panel datetime cases.
* **Integer time index, Panel/Hierarchical: the dask adapter scrambles time order.** The
  index is turned into `str(tuple)` and `from_pandas(sort=True)` sorts lexicographically, so
  times come back as 0, 1, 10, 11, … The resulting pd-multiindex fails `check_is_mtype`
  (non-monotonic).

---------------------------------------------------------------------------------

## 4. Experiment (b): NaiveForecaster on polars inputs

Script: `scripts/a3_naive_polars.py`. Results: `results/a3_naive_polars_pd{2,3}.{json,txt}`.
`NaiveForecaster(strategy="last").fit(y, fh=[1,2,3]).predict()`, 24 monthly points.

| input | output | predicted times correct? | dtype / naming changes |
|---|---|---|---|
| Series `__index__time` Datetime(us), month starts | `pl.DataFrame` | yes: 2002-01-01, 02-01, 03-01 | time **us→ns (pd2)**, kept us (pd3); columns `{__index__time, y}` → **`{__index__0, 0}`** |
| Series month ends | pl.DataFrame | yes: 2002-01-31, 02-28, 03-31 | same renaming |
| Series `pl.Date` | pl.DataFrame | yes | **Date → Datetime(ns)** (pd2) / **Datetime(ms)** (pd3) |
| from PeriodIndex(M) via convert_to | pl.DataFrame | yes, as month-end timestamps | period-ness gone |
| tz Europe/Berlin hourly over DST | pl.DataFrame | yes (tz kept) | renaming |
| business days | pl.DataFrame | yes (B inferred) | renaming |
| Int64 time / no index column | pl.DataFrame | yes (34,35,36 / 24,25,26) | renaming |
| Panel / Hierarchical (polars_panel / polars_hierarchical), MS, ME, Date | pl.DataFrame | yes | column names kept; time us→ns (pd2), Date→Datetime |
| irregular (one month missing), Series or Panel | **TypeError: unsupported operand type(s) for \*: 'int' and 'NoneType'** | — | same error for the pandas control (sktime cannot use int fh without freq) |
| natural polars layout (`time` column, no `__index__` prefix) | **TypeError: Forecasters do not support categorical features** (misleading) | — | |
| `pl.LazyFrame` series | **TypeError: Unsupported input data type** (no lazy Series/Panel mtype) | — | |
| pandas control (DataFrame, index name `time`, column `y`) | pd.DataFrame | yes | column `y` kept, **index name lost** (also on pandas) |

So the polars path gives correct timestamps for regular data, because pandas does the time
arithmetic after conversion. It does not preserve the time dtype or the output column names
in the Series case.

---------------------------------------------------------------------------------

## 5. Experiment (c): can `fh.to_absolute` be expressed with polars `offset_by` (and others)?

Script: `scripts/a3_offset_equiv.py`. Results: `results/a3_offset_equiv_pd{2,3}.{json,txt}`.
The table in §2 lists all the numbers. Key points:

* All 23 frequency cases have an exact polars rule (0/3,600). Several need more than
  `offset_by`: ME/QE/YE need `month_end()`, B needs `add_business_days`, BMS/BME need a roll,
  SME needs a half-month ordinal.
* The naive "polars interval string" rules are wrong for anchored-end frequencies: ME 992/3,600,
  QE (`"kq"`) 900/3,600, B as days 3,300/3,600.
* **sktime vs pandas' own `date_range` disagree on tz-aware daily data in pandas 2** (300/3,600).
  sktime computes `cutoff + k*Day` (absolute 24h in pandas 2), while `date_range` steps by
  calendar day. In pandas 3 both are calendar days and the polars calendar-day rule matches. In
  pandas 2 the absolute-24h rule matches. **sktime's forecast timestamps for tz-aware daily
  data therefore change with the pandas major version.**

---------------------------------------------------------------------------------

## 6. Proposed minimal TimeIndex abstraction (prototype `protos/a3_timeindex.py`)

```python
@dataclass(frozen=True)
class Step:            # replaces pandas offset / freqstr as the SEMANTIC step
    kind: str          # int | fixed | calday | month | bday | bmonth | halfmonth
    n: int = 1         # multiplier (int step, months, business days, days)
    nanos: int = 0     # fixed durations (h, min, D-naive, W)
    anchor: str|None   # "start" | "end" for month/bmonth (MS vs ME, QS vs QE, ...)
    weekmask: str = "1111100"; holidays: tuple = ()
    tz: str|None = None
    @classmethod from_pandas(freq, tz) -> Step|None     # FY5253/Easter/BH -> None
    def to_polars_interval() -> str|None

class TimeOps:          # one implementation per backend: Pandas / Polars / Numpy
    cutoff(times)                       # last value (group max for panels)
    add_steps(cutoffs, k, step)         # fh.to_absolute   (vectorised U x H)
    steps_between(a, b, step)           # fh.to_relative / to_absolute_int / in-sample checks
    range(start, n, step)               # index generation
    infer_step(times) -> Step|None      # replaces pd.infer_freq
```

Mapping of pandas offsets → Step: Day→`fixed 24h` (pandas 2) or `calday` (pandas 3 when tz
is set), Tick→`fixed`, Week→`fixed 7d` (+ week_anchor), BusinessDay→`bday`,
CustomBusinessDay→`bday(holidays, weekmask)`, MonthBegin/End, QuarterBegin/End,
YearBegin/End→`month(n·1/3/12, anchor)`, BMS/BME→`bmonth`, SME→`halfmonth`. Period[F]
values are stored as period-start timestamps (or integer ordinals) plus the Step.

Correctness (`scripts/a3_timeindex_check.py`, `results/a3_timeindex_check_pd{2,3}.{json,txt}`):

* `add_steps`: pandas, polars and numpy backends give **0/3,600 mismatches for all 23 cases**
  vs sktime. The only exception is numpy with tz ("NA: numpy datetime64 has no time zones").
* `steps_between(cutoff, add_steps(cutoff, k)) == k`: 0 mismatches (polars and numpy, all
  applicable cases).
* **Custom business day with 4 holidays:** sktime vs holiday-aware truth (`pd.date_range(…,
  freq=CustomBusinessDay(holidays=…))`) **1,095/3,600 mismatches** (sktime fh freq string is
  `'C'`, holidays lost). The polars and numpy prototypes have 0 mismatches vs the
  holiday-aware truth.
* `infer_step` (polars) vs `pd.infer_freq`: same continuation in **22/22** cases (D, h,
  15min, 3D, B, W-SUN, W-WED, MS, ME, 2ME, QS, QE, YS, YE, BMS, BME, SME→None both, tz D
  (calday), tz h, irregular→None, one-gap D→None, one-gap MS→None). `QS-JAN` and `QS-OCT`
  map to the same `Step(month, n=3, start)`, which removes the alias-drift class of errors.

Semantic mismatches the abstraction must make explicit (it cannot hide them):

1. **Calendar months:** anchor start vs end must be stored. The polars interval `"1mo"`
   alone is ambiguous (pitfall: 992/3,600 wrong for ME). utilsforecast infers the anchor
   from the data. Our Step stores it.
2. **Business days / holidays:** need weekmask + holiday list. polars and numpy support
   these. pyarrow, duckdb and narwhals do not.
3. **Calendar day vs 24h on tz-aware data:** pandas 2 and pandas 3 disagree. The Step must
   choose (`calday` vs `fixed`) and sktime must define the semantics. Recommendation:
   calendar day, matching pandas 3, polars and `date_range`.
4. **Periods:** no Period dtype outside pandas. A Period is a (Step, ordinal) or
   (Step, start timestamp) pair. Output type fidelity (returning a PeriodIndex to pandas
   users) needs a "time kind" flag next to the Step. sktime's polars mtypes currently use
   period END timestamps (§3), and that convention has to be chosen deliberately.
5. **Irregular timestamps:** `infer_step` returns None. Integer fh is then undefined (today
   it raises a cryptic TypeError on both pandas and polars). Only Duration-valued fh can
   work in that case.
6. **Integer / range indexes:** trivial in all backends. RangeIndex vs Index[int64] is
   cosmetic (pandas 2 converts one into the other in the polars round trip).
7. **Off-grid cutoffs:** pandas offsets also *roll* an off-grid timestamp onto the grid
   (e.g. 2000-02-15 + MonthEnd → 2000-02-29). Step arithmetic assumes the cutoff is on-grid,
   which is true for observed cutoffs. Off-grid user-supplied absolute fh need a validation
   step. (Unmeasured/expected.)

---------------------------------------------------------------------------------

## 7. Cost of "pandas only as the time engine on the small index, data in polars"

Script: `scripts/a3_time_engine_cost.py --grid {dev,large}`. Results:
`results/a3_time_engine_cost_dev_pd2.{json,txt}` (pd3 run: `…_dev_pd3.*`). The task is
the prediction-index frame `{(h0, cutoff(h0)+k): k=1..12}` from a long polars panel.
Times are min-of-5 (min-of-2 for long runs), measured in the same process under concurrent
machine load.

* P1 pure polars: `group_by().max()`, cross join k, then the TimeIndex polars expression.
* P2 hybrid: polars `group_by().max()`, unique cutoffs (U) to pandas, vectorised pandas
  offsets (U×H), back to polars and join.
* P3 hybrid with exact sktime semantics: one `ForecastingHorizon.to_absolute_index` per
  unique cutoff.
* P4 current sktime pattern: one `fh.to_absolute_index` per instance (vectorised predict
  does this once per instance) plus `get_cutoff` on the pandas MultiIndex frame.

P1 = P2 = P3 output equality: True for all 16 configurations.

| freq | ends | N | T | U | P1 polars ms | P2 hybrid ms | P3 sktime-fh/unique ms | P4 per-instance ms | P4 get_cutoff ms | P2/P1 | P4/P1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MS | aligned | 10 | 100 | 1 | 4.12 | 5.80 | 6.99 | 20.3 | 3.8 | 1.4 | 4.9 |
| MS | ragged | 10 | 100 | 10 | 1.90 | 4.27 | 21.1 | 16.8 | 3.3 | 2.2 | 8.8 |
| MS | aligned | 100 | 100 | 1 | 2.40 | 3.83 | 4.95 | 189 | 4.9 | 1.6 | 79 |
| MS | ragged | 100 | 100 | 50 | 2.21 | 4.46 | 95.6 | 178 | 4.2 | 2.0 | 80 |
| MS | aligned | 1000 | 100 | 1 | 8.17 | 7.01 | 8.85 | 1,785 | 14.2 | 0.9 | 218 |
| MS | ragged | 1000 | 100 | 50 | 4.91 | 4.73 | 78.4 | 1,953 | 13.7 | 1.0 | 398 |
| MS | aligned | 10000 | 50 | 1 | 38.8 | 17.3 | 16.5 | 20,385 | 83.7 | 0.4 | 525 |
| MS | ragged | 10000 | 50 | 25 | 39.2 | 25.0 | 94.6 | 26,807 | 79.5 | 0.6 | 685 |
| D | aligned | 10 | 100 | 1 | 4.41 | 6.77 | 8.19 | 33.0 | 6.1 | 1.5 | 7.5 |
| D | ragged | 10 | 100 | 10 | 2.17 | 5.11 | 31.3 | 38.3 | 4.7 | 2.4 | 17.7 |
| D | aligned | 100 | 100 | 1 | 3.42 | 6.43 | 8.44 | 268 | 5.0 | 1.9 | 78 |
| D | ragged | 100 | 100 | 50 | 2.70 | 5.93 | 171 | 361 | 7.3 | 2.2 | 134 |
| D | aligned | 1000 | 100 | 1 | 8.47 | 11.3 | 13.0 | 3,814 | 21.9 | 1.3 | 450 |
| D | ragged | 1000 | 100 | 50 | 4.88 | 7.47 | 137 | 3,140 | 14.6 | 1.5 | 644 |
| D | aligned | 10000 | 50 | 1 | 23.0 | 25.0 | 30.9 | 34,269 | 147 | 1.1 | 1,488 |
| D | ragged | 10000 | 50 | 25 | 23.5 | 42.4 | 147 | 36,095 | 85.8 | 1.8 | 1,535 |

Reading:

* **The hybrid costs 0.4×–2.4× pure polars, with an absolute overhead of ≤ ~19 ms at 10k
  series.** At 10k series with MS the hybrid is even faster than pure polars. Pandas works
  on U ≤ 25 unique cutoffs, while P1 evaluates `offset_by` with a per-row string interval on
  N×H = 120k rows. (A polars variant that also deduplicates cutoffs first would remove this
  difference; unmeasured.)
* With exact sktime `ForecastingHorizon` semantics, the hybrid (P3) costs ~1.6–3.8 ms **per
  unique cutoff** (aligned rows include fixed group_by/join costs). That is fine when ends
  are aligned (U=1) and grows linearly with U.
* **The current per-instance pattern (P4) costs 1.7–3.8 ms per instance**, i.e. 20–36 s for
  10k series, 525×–1,535× slower than P1. The cost comes from pandas scalar/1-element
  overheads, not from date arithmetic (`scripts/a3_fh_call_profile.py`,
  `results/a3_fh_call_profile_pd2.txt`). Per call: `pd.DatetimeIndex(absolute, freq=…)`
  validation, which runs `pd.infer_freq` (~0.5–0.8 of ~2.5–3.6 ms profiled); the
  `_HashIndex` lru_cache key via `hash_pandas_object` (~0.5 ms); 12 scalar offset additions
  with a `to_offset(freqstr)` re-parse each (~0.2–0.8 ms); and constructing the new
  `ForecastingHorizon` (`_new`/`_check_values`, ~0.5 ms).

NaiveForecaster-level share of wall time spent in time-axis code: see §7.1.

### 7.1 Share of NaiveForecaster runtime spent in fh / cutoff code

Script: `scripts/a3_naive_time_share.py --grid dev`. Results:
`results/a3_naive_time_share_dev_pd2.{json,txt}`. No profiler is used. The outermost
time-axis entry points (`ForecastingHorizon.__init__/to_absolute/to_relative/
to_absolute_index/to_absolute_int/to_indexer/is_all_*_sample/get_expected_pred_idx`,
`get_cutoff` and `check_fh` as referenced from `forecasting/base/_base.py`) are wrapped
in-process with a re-entrancy guard (no file modified), and their time is summed.
`NaiveForecaster(strategy="last").fit(y, fh=1..12).predict()` runs on a pd-multiindex panel
(vectorised over N instances) and on the same data as `polars_panel`.

| N | T | freq | wall pd-multiindex (s) | time-axis (s) | share | wall polars_panel (s) | share (polars) | time-axis per instance |
|---|---|---|---|---|---|---|---|---|
| 10 | 100 | MS | 0.67 | 0.22 | 21.8% | 0.47 | 21.0% | 22.0 ms |
| 10 | 100 | D | 0.43 | 0.16 | 25.4% | 0.40 | 27.0% | 16.4 ms |
| 100 | 100 | MS | 3.83 | 0.85 | 16.7% | 3.41 | 19.2% | 8.5 ms |
| 100 | 100 | D | 3.30 | 0.89 | 24.9% | 3.16 | 24.7% | 8.9 ms |
| 1000 | 100 | MS | 39.3 | 9.29 | 19.5% | 34.7 | 21.1% | 9.3 ms |
| 1000 | 100 | D | 40.9 | 9.76 | 24.2% | 40.3 | 23.6% | 9.8 ms |

(The share is time-axis divided by the wall time of the *wrapped* run. Unwrapped wall times
are shown.) The call counts per instance are constant: 2× `get_cutoff`, 2× `check_fh`,
1× `ForecastingHorizon` construction, 1× `to_absolute`. For N=1000: 2,002 / 2,001 / 1,001 /
1,000.

So **~17–27% of NaiveForecaster's fit+predict wall time is per-instance time-axis
bookkeeping (~9 ms/instance)**. The same work done once per panel costs 5–8 ms *in total*
for 1,000 series (P1/P2 table above). The polars input does not change the picture (the
shares are within ±3 points), because conversion happens once and the per-instance loop
dominates. This is an upper bound on the speed-up from changing the time engine alone: up to
~1.2–1.4× for a cheap forecaster. That gain needs no polars, only once-per-panel time-axis
computation.

---------------------------------------------------------------------------------

## 8. Prior art (how others handle the time axis without / beside pandas)

| library | approach | measured / read |
|---|---|---|
| **utilsforecast** (Nixtla; used by statsforecast, mlforecast, neuralforecast) | *Dual dispatch by container*: `offset_times(times, freq, n)` / `time_ranges(starts, freq, periods)` branch on pandas vs polars. `freq` is a pandas offset/str for pandas, a polars interval string (`"1mo"`) for polars, or an int. The month-end anchor is **inferred from the data** (`_ensure_month_ends`: if *all* times are month ends, apply `.dt.month_end()`). No business days on the polars path. | Read `utilsforecast/processing.py` (0.2.17). `scripts/a3_prior_art_monthend.py`: 0/144 mismatches for an all-month-end batch. For a mixed batch (one month-end series, one mid-month series) the heuristic switches off for the whole batch: 2000-02-29 + 1mo → **2000-03-29**. |
| **functime** (polars-native) | polars only. `freq` is a polars interval (`"1mo"`, `"1i"` for ints). `make_future_ranges` = `pl.datetime_ranges(cutoff, cutoff.offset_by(fh*n), interval=freq, closed="right")`. No anchor handling. | `a3_prior_art_monthend.py`: **39/144 mismatches** on month-end cutoffs (2000-02-29 → 2000-03-29, 04-29, 05-29). |
| **darts** 0.41 | Accepts any narwhals-compatible frame at the boundary (`nw.from_native`), then converts the time column to a **pandas DatetimeIndex/RangeIndex** and values to numpy. Time arithmetic is pandas (`self._time_index + n * self.freq`). This is the "pandas as time engine, numpy/other for data" hybrid. | Read `darts/timeseries.py`, `darts/utils/utils.py::dataframe_col_to_time_index`. |
| **narwhals** 2.27 | `Expr.dt.offset_by` (str only, not per-row), `truncate`, tz ops. No `month_end`, no business days. | `a3_narwhals_time_probe.py`: months OK on pandas/polars/polars-lazy/duckdb. **pyarrow and dask raise NotImplementedError for "mo"** (dask also for `truncate("1mo")`). Hourly offsets OK on all 6 backends. |

Take-aways: nobody outside pandas has a "freq" object. polars-native libraries carry
frequency as a polars interval string, and their bugs are exactly the anchor ambiguity
(functime always, utilsforecast in mixed batches). The `Step` design (explicit anchor,
holidays, calday vs fixed) is the minimal fix. darts shows that keeping pandas only for the
time index is a shipping, production pattern.

---------------------------------------------------------------------------------

## 9. Recommendations

1. Introduce `Step` + `TimeOps` (pandas / polars / numpy backends) as the internal time-axis
   engine (cf. `protos/a3_timeindex.py`). Store `ForecastingHorizon` internally as
   (relative int array or absolute time array, Step, time-kind {int, datetime, date, period},
   tz). Keep `to_pandas()`, `cutoff` (as `pd.Index`) and `freq` (freqstr) as lazily
   materialised compatibility views for the ~44 forecasting files that use pandas time
   semantics directly.
2. Compute the time axis **once per panel**: cutoffs via group max in the native container,
   absolute fh for the U unique cutoffs, output keys by join. Do not compute it per instance
   inside the vectorisation loop. This is the 500×–1,500× lever measured in §7 and it is
   independent of polars.
3. Keep pandas as the **fallback engine** for offsets with no Step mapping (FY5253, Easter,
   BusinessHour, calendar objects), through the hybrid path on unique cutoffs (≤ ~20 ms at
   10k series measured; P2).
4. Fix independently of the redesign (each measured here): CustomBusinessDay holidays lost
   (store the offset, not the freqstr); alias-drift `QS-JAN`/`QS-OCT` equality (compare
   semantically); tz-aware daily semantics pinned to calendar day; polars Series output
   column renaming; Date/us dtype preservation; dask `python_type` tag; dask string-sort of
   integer time; a clear error for irregular data with int fh.
5. Decide the Period convention for non-pandas containers. Store period start timestamps
   plus a period flag (recommended, matching numpy `M8[M]`) or ordinals. Today the polars
   mtypes store period END dates (§3).

---------------------------------------------------------------------------------

## 10. Caveats

* Timings were taken on a shared 16-core machine with ~14 concurrent agents. Absolute ms are
  noisy (the same `fh.to_absolute_index` call measured 1.7–3.8 ms in one script and
  3.1–7.5 ms in another). Ratios measured in the same process are more reliable. The large
  grid has not been run; scripts accept `--grid large`.
* Correctness checks use 150 consecutive cutoffs per frequency starting in 2000. Leap years
  and month lengths are covered, but far-future/past dates and non-Berlin time zones are not.
  tz handling was tested with Europe/Berlin only.
* The census is regex-based (it may over- or under-count, e.g. `.freq` attribute on
  non-index objects).
* `infer_step` is a prototype. It does not handle multiples of business days, anchored
  weekly multiples ("2W"), or tolerance for occasional gaps, and it was only tested on clean
  synthetic data.
* duckdb tz, BMS/BME/SME rules and pyarrow `*_between` functions were not tested
  (unmeasured).
* The "P4" per-instance cost isolates one `to_absolute_index` per instance. The real
  vectorised path calls several fh/cutoff functions per instance (§7.1: ~9 ms/instance
  in total). The §7.1 shares come from single runs (no repeats) on a loaded machine.
  The wrapped runs' wall times differ from the unwrapped ones by −2% to +50% (load noise
  plus wrapper overhead), so the shares are ±several points.

---------------------------------------------------------------------------------

## 11. Commands and raw outputs

| script | command (from WT root) | outputs |
|---|---|---|
| census | `PYTHONPATH=.:experiments/container_independence $EXP_PY experiments/container_independence/scripts/a3_coupling_census.py` | `results/a3_coupling_census.json` |
| (a) round trip | `PYTHONPATH=… $EXP_PY -W ignore …/a3_roundtrip.py` (and `$EXP_PY3`) | `results/a3_roundtrip_pd{2,3}.{json,txt}` |
| (b) naive on polars | `… …/a3_naive_polars.py` (both envs) | `results/a3_naive_polars_pd{2,3}.{json,txt}` |
| (c) offset equivalence | `… …/a3_offset_equiv.py` (both envs) | `results/a3_offset_equiv_pd{2,3}.{json,txt}` |
| prototype check | `… …/a3_timeindex_check.py` (both envs) | `results/a3_timeindex_check_pd{2,3}.{json,txt}` |
| freq alias probe | `… …/a3_freq_alias_probe.py` (both envs) | `results/a3_freq_alias_probe.txt`, `…_pd{2,3}.json` |
| narwhals time ops | `… $EXP_PY …/a3_narwhals_time_probe.py` | `results/a3_narwhals_time_probe.{json,txt}` |
| hybrid cost | `… $EXP_PY …/a3_time_engine_cost.py --grid dev`; `… $EXP_PY3 … --freqs MS --profile-max-series 0` | `results/a3_time_engine_cost_dev_pd{2,3}.{json,txt}` |
| fh call profile | `… $EXP_PY …/a3_fh_call_profile.py` | `results/a3_fh_call_profile_pd2.txt` |
| time share | `… $EXP_PY …/a3_naive_time_share.py --grid dev` | `results/a3_naive_time_share_dev_pd2.{json,txt}` |
| prior art | `$A3_PY -I experiments/container_independence/scripts/a3_prior_art_monthend.py` (private venv) | `results/a3_prior_art_monthend.json` |

---------------------------------------------------------------------------------

## 12. Cloud re-run (4-core machine, ~2 concurrent agents, 2026-10-18)

Re-ran two headline scripts (pandas 2.3.3 env):

* `PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/a3_timeindex_check.py`
  (10.6 s): **unchanged** — pandas/polars/numpy `add_steps` 0/3,600 mismatches on all 23
  cases (numpy NA for tz), steps_between round trip 0, CustomBusinessDay: sktime vs
  holiday-aware truth 1,095/3,600 (fh freq string `'C'`), polars/numpy vs truth 0;
  `infer_step` agrees with `pd.infer_freq` 22/22.
* `PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/a3_time_engine_cost.py --grid dev --profile-max-series 0`
  (4 min). P1=P2=P3 equality True in all 16 configs. Absolute times are lower than in the
  loaded local run, so the per-instance figures change materially:

| freq | ends | N | T | U | P1 ms | P2 ms | P3 ms | P4 ms | P2/P1 | P4/P1 |
|---|---|---|---|---|---|---|---|---|---|---|
| MS | aligned | 1000 | 100 | 1 | 6.61 | 5.47 | 7.38 | 1,079 | 0.8 | 163 |
| MS | ragged | 1000 | 100 | 50 | 4.83 | 5.89 | 50.96 | 1,038 | 1.2 | 215 |
| MS | aligned | 10000 | 50 | 1 | 26.51 | 17.44 | 20.90 | 12,077 | 0.7 | 456 |
| MS | ragged | 10000 | 50 | 25 | 23.20 | 15.01 | 41.25 | 11,288 | 0.6 | 487 |
| D | aligned | 1000 | 100 | 1 | 4.31 | 5.67 | 7.34 | 1,390 | 1.3 | 322 |
| D | ragged | 1000 | 100 | 50 | 4.75 | 6.68 | 74.60 | 1,398 | 1.4 | 294 |
| D | aligned | 10000 | 50 | 1 | 15.21 | 15.52 | 16.98 | 12,971 | 1.0 | 853 |
| D | ragged | 10000 | 50 | 25 | 12.14 | 11.95 | 47.46 | 13,340 | 1.0 | 1,099 |

(Smaller grid points: P2/P1 1.5–2.5, P4/P1 4.9–69.)

Updated headline numbers (supersede §0.4 / §7 where they differ):
* Hybrid P2 costs **0.6×–2.5×** pure polars over the whole grid; at 10k series P2 is
  12–17 ms vs P1 12–27 ms.
* Current per-instance pattern P4 costs **~1.0–1.4 ms per instance** (was 1.7–3.8 ms),
  i.e. **11–13 s for 10k series, 456×–1,099× pure polars** (was 20–36 s, 525×–1,535×).
* P3 (exact sktime fh per unique cutoff) ≈ 0.8–1.4 ms per unique cutoff.
The qualitative conclusion is unchanged: where the time axis is computed (once per panel
vs once per instance) dominates; which library computes it does not.
`a3_naive_time_share.py` (§7.1, ~17–27% / ~9 ms per instance) was not re-run; those
numbers are from the earlier loaded local run, where per-call costs were ~2× higher, so
the per-instance ms is likely lower here (unmeasured).
