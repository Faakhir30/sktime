# x9_cheap_paths — adversarial verification (verify0)

Lens: correctness. Re-run the original scripts, check the zero-copy claims independently with pyarrow buffer addresses and `np.shares_memory`, and test the sktime-compatibility claims on more forecasters.

Environments: EXP_PY (pandas 2.3.3) and EXP_PY3 (pandas 3.0.6), with polars 2.0.0 and pyarrow 26. statsmodels comes from the original agent's `--target` dir (`$SCRATCH/venvs/x9_smtarget`). Every command was run from the WT root with `PYTHONPATH=.:experiments/container_independence[:$SM]`.

New scripts (all in `experiments/container_independence/scripts/`):

| script | purpose |
|---|---|
| `x9_cheap_paths_verify0_mono.py` | checks the vectorized monotonic check against the original on 300 random/adversarial frames, and measures the end-to-end `convert_to` effect with the check monkeypatched |
| `x9_cheap_paths_verify0_zerocopy.py` | compares pyarrow buffer addresses of polars `.to_arrow()` and the pandas `ArrowExtensionArray._pa_array`; tests chunked input, the hybrid layout and the reverse direction |
| `x9_cheap_paths_verify0_lazy.py` | splits the Part B gain into "filter before converting" (eager) and "push the filter into the scan" (lazy); re-baselines after the monotonic fix; every variant returns a pandas pd-multiindex frame |
| `x9_cheap_paths_verify0_census.py` | runs every installable sktime forecaster (71 of 160) on an Arrow-valued y with a numpy DatetimeIndex, against a numpy baseline |
| `x9_cheap_paths_verify0_index.py` | checks Arrow index acceptance level by level, and the cost and zero-copy behaviour of coercing an Arrow timestamp index to a DatetimeIndex |

Ad-hoc scratch checks (`$SCRATCH/x9v_chunk*.py`) measured multi-chunk and parquet-read zero-copy. Their output is quoted below.

Original scripts re-run: `x9_profile_convert.py` (pd2), `x9_arrow_compat.py` (pd2 and pd3), `x9_lazy_mtype_probe.py` (pd2), `x9_lazy_pushdown.py --grid dev` (pd2) and `x9_arrow_convert.py --grid dev` (pd2). These runs overwrote `results/x9_arrow_convert_dev_pd2.json` and `results/x9_lazy_pushdown_dev_pd2.json`, and `x9_lazy_mtype_probe_pd2.json`, with the same kind of values.

---

## x9.1 zero-copy: PARTIALLY CONFIRMED (holds only for single-chunk frames)

`x9_cheap_paths_verify0_zerocopy.py` on pd2 and pd3 (identical output) uses a frame from `make_pl_panel(1000,100)`, which is rechunked:

```
arrow c0 same_addr=True | numpy c0 same_addr=False | arrow __index__time same_addr=True
arrow __index__h0 (string) same_addr=False
np.shares_memory(c0 polars, arrow pandas): True | vs numpy route: False
hybrid c0 after index assign same_addr=True ; hybrid loc slice dtype double[pyarrow]
back c0: same_addr_as_src=True ; back __index__time: same_addr_as_src=True
```

The claim holds for a single-chunk frame.

**What the original missed.** Every frame the original measured had been explicitly `.rechunk()`ed. Real inputs usually are not:

- `pl.read_parquet` returned 20 chunks per column (one per row group) on a 2M-row file.
- So did `scan_parquet().collect()` and `scan().filter().collect()`, also 20 chunks.

For such frames, `to_pandas(use_pyarrow_extension_array=True)` **rechunks, which is a copy**:

```
x9v_chunk.py  (pl.concat of 2 frames, rechunk=False): src chunks 2 -> pd chunks 1, zero-copy per chunk: False, any shared buffer: False
x9v_chunk3.py (pl.read_parquet, 2M rows x [str, ts, 4 float]):
   src chunks 20 pd chunks 1 shared bufs 0
   arrow multichunk peak RSS delta 140.0 MB  vs numpy 242.5 MB   (polars frame 91.6 MB)
```

On parquet-loaded data the Arrow route therefore uses about 1.7× less memory than the numpy route, not about 0 MB. The original's ~0–4 MB numbers come from rechunked, generator-built frames.

The original's "same address" for the multi-chunk case in my first script is misleading: `Series.to_arrow()` itself rechunks, so I measured with `get_chunks()` instead.

A further caveat on the near-zero RSS deltas: when the source frame is built in the same process, freed allocator memory is reused, so RSS deltas are a **lower bound**. In my check, a 30.5 MB string-key copy showed a 0.0–0.1 MB delta.

## x9.2 conversion timings and RSS: PARTIALLY CONFIRMED (numbers reproduce; the headline comparison is apples to oranges)

Re-run of `x9_arrow_convert.py --grid dev` (pd2) at 500k rows:

| variant | time | ratio to `sktime_adapter` | peak RSS delta |
|---|---|---|---|
| `sktime_adapter` | 67.8 ms | 1.0 | 43.6 MB |
| `np_flat` | — | 0.192 | — |
| `arrow_flat` | 3.80 ms | **0.056** | 0.1 MB |
| `hybrid_mi` | — | 0.706 | — |
| `arrow_mi` | — | 0.496 | — |

At 100k×10 columns, `hybrid_mi` is 0.772× the adapter with 5.1 MB peak RSS, against 21.4 MB for the adapter. The original reported 0.69 and 6.6 MB. These numbers reproduce the original's values (0.05, 0.73, 6.6/22 MB).

However, the headline "0.05–0.13× of sktime's adapter" compares a frame **without any index** (`arrow_flat`) with the adapter, which builds a MultiIndex and runs `infer_freq`. sktime cannot use `arrow_flat`.

The like-for-like comparisons are:
- `arrow_flat` vs `np_flat`: 0.29×.
- `hybrid_mi` (the only Arrow layout sktime accepts) vs the adapter: 0.71–0.81×.

So the conversion-time gain for a sktime-usable frame is about **1.2–1.4×, not 8–20×**. The zero-copy RSS advantage also disappears for multi-chunk inputs (see x9.1).

## x9.3 monotonic check: PARTIALLY CONFIRMED (the check speedup holds; the end-to-end effect is ×10, not ×25–28, and the replacement is not equivalent with null timestamps)

Re-run of `x9_profile_convert.py` (pd2):

```
is_monotonically_increasing cumtime 1.528 s of 1.739 s under profiler
[sorted] rows=500000 current=True 857.4 ms | vectorized=True 38.12 ms | speedup x22
[unsorted_at_row7] current=False 49.8 ms | vectorized=False 38.28 ms | speedup x1
convert_to total 979.5 ms; current monotonic check 914.6 ms = 93% of convert_to
```

These match the original: ×22 here against ×25–28 reported, and 93% against 96%.

**End to end** (`x9_cheap_paths_verify0_mono.py`, check monkeypatched inside `convert_to`, 10k×50):

```
convert_to: orig 975.6 ms | vectorized check 101.4 ms | no check 65.7 ms | end-to-end speedup x9.6
```

The report's conclusion table says "`convert_to` from polars about 25–28× cheaper". That is wrong: the **check** is about 25× cheaper, and `convert_to` is about **10×** cheaper.

**Equivalence.** I compared the two checks on 300 random frames covering four modes: sorted, random, interleaved groups, and null timestamps. There were 0 mismatches on null-free inputs and **58/75 mismatches when the time column contains nulls**. In those cases the original returns False and the vectorized check returns True, because `fill_null(True)` also masks nulls in `t` itself, not only the shift's first row. The fix needs `t.is_not_null().all()`, or `fill_null` applied only to the shifted value, before it can be called "the same results".

## x9.4 Arrow index rejected: PARTIALLY CONFIRMED (the time level only; Arrow instance levels are accepted)

Re-running `x9_arrow_compat.py` gives identical counts on pd2 and pd3 (`arrow_ts_idx` 1/13, `P_arrow_levels` 1/14). The cited lines (`_series/_check.py:51`, `_panel/_check.py:59`) contain `VALID_INDEX_TYPES = (RangeIndex, PeriodIndex, DatetimeIndex)`.

`x9_cheap_paths_verify0_index.py` tests each level separately (pd2 and pd3 identical):

```
panel keys=numpy time=numpy : check=True  naive=OK
panel keys=numpy time=arrow : check=False naive=TypeError
panel keys=arrow time=numpy : check=True  naive=OK   (levels string[pyarrow] / large_string[pyarrow] ArrowDtype + datetime64)
panel keys=arrow time=arrow : check=False naive=TypeError
series time=arrow : check False
```

Only the **time** level causes the rejection. The claim that Arrow MultiIndex levels are rejected is too broad: Arrow string instance levels pass the check and NaiveForecaster.

This also answers the original's open question about coercing the time level, by measurement. On 2M timestamps, coercing an `Index[timestamp[ns][pyarrow]]` with `pd.DatetimeIndex(idx)` takes:
- 0.25 ms and is **zero-copy** on pandas 3.0.6 (buffer shared).
- 2.9 ms as a copy on pandas 2.3.3.

The boundary fix is therefore cheap.

## x9.5 hybrid layout accepted (13/14 and 14/15): PARTIALLY CONFIRMED (reproduces; it does not generalise as cleanly)

Counts reproduce exactly on pd2 and pd3, and Theta fails at `theta.py:230`.

**Census of all installable forecasters** (`x9_cheap_paths_verify0_census.py`): 71 of 160 forecasters have their dependencies installed. Each was run with `create_test_instance`, fit plus predict, fh=[1,2,3], on y as `double[pyarrow]` with a numpy DatetimeIndex. Results were identical on pd2 and pd3:

```
OK 62, ERR 7, OK_MISMATCH 2 ; outputs still Arrow among OK: 16/64
ERR: ThetaForecaster (theta.py:230 reshape), NaiveVariance (_naive.py:903 ArrowInvalid / pd3 TypeError),
     ConformalIntervals (conformal.py:495), VAR (var.py:174), VARMAX (varmax.py:331), VECM (vecm.py:168),
     DynamicFactor (dynamic_factor.py:365) -- statsmodels "Pandas data cast to numpy dtype of object"
```

The two mismatches (BaggingForecaster, VARReduce) are nondeterministic in the **numpy-vs-numpy** comparison as well, so they are not Arrow bugs.

The real pass rate is therefore 64/71 = **90%**, with 7 genuine failures, not "only Theta". Two of the failures are the probabilistic wrappers (NaiveVariance, ConformalIntervals), and four come from the statsmodels VAR family. The 160−71 forecasters with dependencies not installed are untested.

On x9.6 (dtype survival): only **16/64 (25%)** census outputs keep Arrow, fewer than the "about half" found on the hand-picked list.

## x9.8 eager vs lazy pushdown: CONFIRMED (numbers), but the headline ratio is mostly the monotonic-loop bug

Re-run of `x9_lazy_pushdown.py --grid dev` (pd2). All 11 variants agree on rows, sum and keys.

| variant | contiguous ids | random ids |
|---|---|---|
| B1 | 2225 ms, 427 MB | 2513 ms, 387 MB |
| B2 | 7.5 ms (×297), 30 MB | 22.4 ms (×112), 36 MB |
| B3 | 13.1 ms, 10.8 MB | 34.2 ms, 20.7 MB |

This is within the reported ranges, slightly below 119× on random ids.

The lazy query is really executed: it calls `.collect()` and `fetch_arrow_table`. The plan shows the projection and selection pushed into the scan.

Two asymmetries:
- B2 and B3 return polars or Arrow, while B1 returns a pandas MultiIndex.
- About 85–90% of B1 is the monotonic Python loop (see x9.3), not eagerness.

The ×119–442 figure is therefore not a fair estimate of what laziness buys. See x9.9.

## x9.9 lazy-aware boundary 8–29× faster and 7–9× less memory: PARTIALLY CONFIRMED (reproduces; most of the gain does not need laziness)

The re-run of the original gives:
- B1c: 413/536 ms, 353/428 MB.
- B5b: 13.9/31.2 ms, 42/49 MB.

This is ×17–30 in time.

`x9_cheap_paths_verify0_lazy.py` (pd2, random ids, every output a pandas pd-multiindex, all agree) breaks the gain down:

```
V_today      2709 ms  416 MB   (read all -> convert_to -> select)
V_today_fix   503 ms  494 MB   (same, vectorized monotonic check)       x5.4 from the cheap fix alone
V_adapter     429 ms  494 MB   (= B1c)
E_filter       51 ms  148 MB   (EAGER read all -> polars filter -> adapter)
E_proj         38 ms  110 MB   (eager read_parquet(columns=3) -> filter -> adapter)
L_push         27 ms   54 MB   (= B5b, lazy scan with pushdown)
L_push_fix     31 ms   55 MB   (lazy + convert_to with vectorized check)
pushdown-only gain (E_filter / L_push): time x1.9, rss x2.8
after cheap fix (V_today_fix / L_push_fix): time x16.2, rss x9.0
```

Most of the ×8–29 comes from **converting only the needed slice**. An eager boundary can do that too by filtering in polars before `convert_to`: E_filter is 8.4× faster than B1c here. The gain from true pushdown or laziness is only **×1.4–1.9 in time and ×2–2.8 in peak RSS** on a warm local file.

The original's B1b row (eager then filter, 136–188 ms including the slow check) already contained this information, but the report attributes the whole gain to laziness.

The architectural implication: a **selection-aware boundary**, which knows which instances, time window and columns are needed and applies them before converting, captures most of the value with no lazy mtype. Laziness adds about 2×, plus the unmeasured I/O savings on cold or remote storage.

## x9.10 no lazy Series/Panel mtype: CONFIRMED

Re-running `x9_lazy_mtype_probe.py` (pd2) reproduces it:
- `check_is_scitype` on the LazyFrame is invalid for the forecasting scitypes.
- `convert_to` raises TypeError.
- `NaiveForecaster.fit(lazy)` raises TypeError.
- Eager polars predict returns `polars.DataFrame`.

In `MTYPE_REGISTER`, `pl.LazyFrame registered: False`; the only lazy-like mtypes are `dask_series`, `dask_panel`, `dask_hierarchical` and `polars_lazy_table` (Table).

The dead converter keys are in `sktime/datatypes/_series/_convert.py:319` and `:326`. The report says "`_convert.py`", which is loosely right.

One point the report understates: dask mtypes *are* lazy Series/Panel/Hierarchical mtypes, so sktime does accept a lazy container for forecasting. It collects it to pandas up front.

---

## Other issues found

1. The report's conclusion "`convert_to` 25–28× cheaper" is wrong. It is about 10× end to end (x9.3).
2. The vectorized monotonic check gives a different answer when the time column contains nulls (58/75 null cases mismatch), so it needs a null guard before it goes into a PR.
3. Zero-copy depends on chunking. Parquet-loaded and lazily collected polars frames are multi-chunk (20 chunks here), so the Arrow conversion copies: 140 MB against 243 MB for numpy, not about 0.
4. The laziness gain is mostly "filter before convert", which works eagerly. True pushdown is ×1.9 time and ×2.8 RSS (warm cache).
5. Only the Arrow **time** level is rejected; Arrow string instance levels are accepted. Coercing the time level is cheap (zero-copy on pandas 3).
6. A broader census gives a 90% pass rate (7/71 fail, including the probabilistic wrappers NaiveVariance and ConformalIntervals and the statsmodels VAR family), and Arrow survives in only 25% of outputs.
7. In-process RSS deltas near 0 MB are lower bounds, because freed allocator memory is reused.
