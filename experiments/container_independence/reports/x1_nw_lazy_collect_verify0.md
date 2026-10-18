# x1_nw_lazy_collect: adversarial verification (verify0)

Lens: methodology. I re-ran x1's scripts on EXP_PY (pd 2.3.3) and EXP_PY3 (pd 3.0.6), and tried
to break individual matrix cells and ratios using different code paths.

Versions: narwhals 2.27.1, polars 2.0.0, pyarrow 26.0.0, dask 2026.8.0, duckdb 1.5.6, ibis 12.0.0
(pd2 only), python 3.12.3.

Machine: 4-core cloud box with other agents running. Timings are min-of-repeats. Ratios are
taken within one process, or within one driver run.

## Commands (run from the WT root; `ED=experiments/container_independence`, `SP=<scratchpad>`)

```
# re-runs of the original scripts (my tags, then renamed to the verify0 prefix)
X1_TMP=$SP/x1v PYTHONPATH=.:$ED $EXP_PY  -W ignore $ED/scripts/x1_matrix.py --tag v0pd2
X1_TMP=$SP/x1v PYTHONPATH=.:$ED $EXP_PY3 -W ignore $ED/scripts/x1_matrix.py --tag v0pd3
PYTHONPATH=.:$ED $EXP_PY  -W ignore $ED/scripts/x1_fallback.py --grid dev --tag v0pd2
X1_TMP=$SP/x1v PYTHONPATH=.:$ED $EXP_PY -W ignore $ED/scripts/x1_stay_lazy.py --grid dev --tag v0pd2
PYTHONPATH=.:$ED $EXP_PY  -W ignore $ED/scripts/x1_fidelity.py --tag v0pd2   (and EXP_PY3 --tag v0pd3)
# independent checks (new scripts)
X1_TMP=$SP/x1v PYTHONPATH=.:$ED:$ED/scripts $EXP_PY  -W ignore $ED/scripts/x1_nw_lazy_collect_verify0_counter.py   (+ EXP_PY3)
X1_TMP=$SP/x1v PYTHONPATH=.:$ED:$ED/scripts $EXP_PY  -W ignore $ED/scripts/x1_nw_lazy_collect_verify0_stable.py    (+ EXP_PY3)
X1_TMP=$SP/x1v PYTHONPATH=.:$ED:$ED/scripts $EXP_PY  -W ignore $ED/scripts/x1_nw_lazy_collect_verify0_bench.py --grid dev
X1_TMP=$SP/x1v PYTHONPATH=.:$ED $EXP_PY -W ignore $ED/scripts/x1_nw_lazy_collect_verify0_iter.py --grid dev
```

Outputs:
* `results/x1_nw_lazy_collect_verify0_{matrix,counter,stable,fidelity}_{pd2,pd3}.json`
* `results/x1_nw_lazy_collect_verify0_fallback_pd2.json`
* `results/x1_nw_lazy_collect_verify0_stay_lazy_pd2.json`
* `results/x1_nw_lazy_collect_verify0_bench_dev_pd2.json`
* `results/x1_nw_lazy_collect_verify0_iter_dev_pd2.json`

The bench and iter scripts accept `--grid {dev,large}`.

## 1. What reproduced exactly

* **Matrix A/B and C, pd2 and pd3.** I diffed every status, type and check field against
  `results/x1_matrix_pd2.json`: **0 differences**, and pd3 vs pd2 also had 0. Coverage
  re-counted from the JSON: polars_lazy 34, duckdb 32, ibis 31, duckdb_owncon 28, dask 24.
* **Source check** (`narwhals/_utils.py` `can_lazyframe_collect`): {PANDAS, POLARS, PYARROW}.
  * DuckDB `collect(None)` → `to_arrow_table()` (`_duckdb/dataframe.py:133-149`).
  * ibis `collect(None)` → `to_pyarrow()` (`_ibis/dataframe.py:117-128`).
  * CUDF and MODIN are "eager-allowed", but they are not collect targets.
* **No hidden dask computes, checked a different way.** Instead of x1's dask `Callback`, I wrapped
  the input in `map_partitions(side_effect_counter)`. Partition calls while building and
  `to_native()` were 0 for all 41 ops (counter K3).
* **Fidelity (F1) on both envs.** Lossy counts reproduced:
  * pd2: polars 6, DuckDB 6, ibis 7, dask 0
  * pd3: polars 5, DuckDB 4
  * DuckDB `collect("polars")` → ComputeError (INTERVAL)
  * Nuance: the "lossy" count includes string→object, which is pandas-2's normal string dtype.
    Changes that actually lose information are fewer: about 3–4 per backend (int+null→float,
    null/NaN merged, decimal→float/object, date→datetime).
  * DuckDB's session-TZ conversion is host-dependent: Etc/UTC here, Asia/Karachi locally.

## 2. Counterexamples found

### 2a. Results depend on which narwhals API you use (x1 only used top-level `narwhals`)
Script `x1_nw_lazy_collect_verify0_stable.py`, plus counter K1. Results are identical on pd2 and
pd3.

* **`narwhals.stable.v1`** (the API most downstream libraries pin to):
  * `from_native(DuckDBPyRelation)` and `from_native(ibis.Table)` return **`DataFrame`, not
    `LazyFrame`**, and `.collect()` raises `AttributeError: 'DataFrame' object has no attribute
    'collect'`. So C1's "DuckDB → Arrow via collect" doesn't apply as written under v1.
  * Order-dependent ops without `order_by` (`last()`, `shift().over()`) and `map_batches`
    **work** on polars-lazy, pandas and modin LazyFrames. Under v1 only 4 of 40 ops fail for
    polars: `iter_groups`, `to_numpy`, `pivot` and `last(order_by=)`.
* **`stable.v1` and `stable.v2` both:** `Expr.last(order_by=...)` raises
  `TypeError: unexpected keyword argument 'order_by'`. It exists only in the unstable main
  namespace.
  * That is exactly the R2b route behind x1's headline 1.2–3x stay-lazy wins (claim .8). With a
    stable API, only the semi-join formulation R2 is available.
* **Per-backend counts under the stable APIs** (40 ops, sink_parquet excluded):

  | API | polars_lazy | duckdb | ibis | duckdb_owncon | dask |
  |---|---|---|---|---|---|
  | stable.v2 | 32 | 30 | 29 | 26 | 23 |
  | stable.v1 | 36 | 31 | 30 | 27 | 23 |

### 2b. dask matrix cells depend on how the input was built (counter K2)

| op | `from_pandas(np=4)` (x1) | `from_pandas(np=1)` | `dd.read_parquet` (1 file) |
|---|---|---|---|
| tail | ERR (multiple partitions) | **ok** | **ok** |
| filter `time == max().over(h0)` | COLLECT-ERR TypeError | AssertionError (divisions) | **ok** |
| over(order_by), rank, last(order_by) | ERR | ERR | ERR |

So dask's 24/34 is a lower bound that depends on the input. With a single-file `read_parquet`,
which is the realistic lazy source, 26/34 work. The structural gaps are `over(order_by)`, rank,
`last`, `cum_count(reverse)` and `map_batches`.

### 2c. Alternative formulations exist for most "ERR" cells (counter K4)
* `sort(desc).head(k)` works lazily on DuckDB, ibis and dask, with the same native type. This
  replaces `tail`.
* "last-k per key" via a join on `max(time) - k*freq` works on all four lazy backends, dask
  included. This replaces `rank`/`cum_count`.
* ibis `fill_null(strategy=)` stays unsupported.

### 2d. ibis on the polars backend still collects to pyarrow (counter K5)
`nw.from_native(ibis.polars.connect().create_table(...)).collect()` returns `pyarrow.Table`.
ibis' own `to_polars()` returns polars. So "the default is each backend's own materialisation" is
true for ibis only in the sense that ibis exposes `to_pyarrow` for every engine. It is not true in
the sense of "the engine's natural format".

## 3. Methodology flaws in the timing and memory claims

### 3a. Peak-memory figures are inflated by in-process allocator reuse (claim .8)
x1 measures peak RSS with a 5 ms sampling thread. All routes run in one process, in the order R1
→ R2 → R2b → …. Once R1 has grown the allocator and DuckDB buffer pools, the later routes reuse
that memory, so their deltas come out small.

I re-measured with each (backend, route) in a **fresh subprocess**, using the kernel high-water
mark (`ru_maxrss` minus RSS before the route). 500k rows, pd2:

| backend | R1 peak | R2 peak | R2b peak | R1/R2b | x1 claimed R1 -> R2b |
|---|---|---|---|---|---|
| polars | 93.1 MB | 58.0 | 72.3 | **1.3x** | 6.2x (77.4 -> 12.5) |
| duckdb | 112.0 | 43.1 | 43.4 | **2.6x** | >35x (43.3 -> 0.3) |
| ibis | 108.9 | 23.5 | 15.5 | **7.0x** | >35x (47.1 -> 1.2) |

Staying lazy does lower peak memory, but by **1.3–7x**, not "6–35x+". The 0.3 MB and 1.2 MB
"peaks" are measurement artefacts. The same sampler gave dask `collect('pandas')` a 0.0 MB peak
in my fallback re-run, so its memory columns are not reliable in general.

### 3b. The collect('pandas') "30x" comes from a near-zero denominator (claim .7)
In x1_fallback, polars' `native_collect` is `.collect()` on an in-memory LazyFrame: 0.56 ms, which
is close to a no-op. The re-run of that script gave polars x32.5, DuckDB x5.11 and ibis x5.58.

With a `scan_parquet` source the collect does real work. My fresh-process bench, 500k rows:

| backend | collect('pandas') / native collect |
|---|---|
| polars | **5.35x** (24.7 vs 4.6 ms) |
| duckdb | **4.04x** |
| ibis | **3.19x** |
| dask | 0.94x |

So the realistic range is about **3–5x**, not 4–30x.

### 3c. Relift ratios below 1 were noise (claim .6)
Re-running `x1_fallback.py` at 500k rows, `rt_manual` vs `native_collect` was:

| backend | polars | duckdb | ibis | dask |
|---|---|---|---|---|
| `rt_manual` / native collect | 1.31 | 1.17 | 1.27 | 2.25 |

That is a range of **1.17–2.25x**, against the claimed 0.82–1.92x. The relift adds nothing to a
lazy relation: `from_arrow`/`memtable` are views. The sub-1 values in x1 are therefore noise.
dask pays a real `from_pandas` re-partition, about 2x.

The fidelity facts reproduced exactly:
* `rt_manual` keeps npartitions 8→8, while `rt_nw` drops it 8→1.
* The `rt_nw` DuckDB relation can't be joined with the user's connection.
* ibis gets an unbound memtable.

### 3d. The per-group loop is compared with the wrong baseline (claim .11)
x1 compares the lazy filter+collect loop with *one grouped aggregation*. For arbitrary per-series
estimators, the real alternative is "collect once, iterate groups eagerly, take numpy per group".
`x1_nw_lazy_collect_verify0_iter.py`, 1000x100, all 1000 keys, no extrapolation:

| backend | lazy loop | collect('polars') + iterate | grouped query | loop / eager iterate | loop / grouped |
|---|---|---|---|---|---|
| polars | 2.53 s | 0.178 s | 5.4 ms | **14.2x** | 470x |
| duckdb | 3.29 s | 0.191 s (0.536 s with the pyarrow default) | 5.5 ms | **17.2x** | 603x |

The lazy loop is a dead end, which agrees with x1. But the decision-relevant penalty is **14–17x**
compared with eager iteration. Eager iteration is itself about 33–35x slower than a set-based
query.

The original script re-run gave 144x (duckdb), 223x (polars), 227x (ibis) and 536x (dask) vs R1,
against the claimed 170–184x and 543x.

### 3e. Other checks
* **R1 (collect-all) is not handicapped by the pyarrow-eager default.** My extra route R1p uses
  `collect('polars')` and then eager polars. It was 0.90–1.04x of R1 at 500k rows. So the
  stay-lazy speedup does not come from the pyarrow-eager narwhals backend being slow, and x1's
  speedups stand.
* **Stay-lazy speedups reproduce.** At 500k rows:

  | run | R2 | R2b |
  |---|---|---|
  | re-run of `x1_stay_lazy` | 1.31–1.57x | 1.89–2.73x |
  | fresh-process bench | 1.49–2.60x | 2.05–3.90x |

  At ≤10k rows R2 was 0.45–0.82x, against the claimed 0.54–0.74x. dask R2 was 0.09–0.32x
  (claimed 0.10–0.38x), and 0.18–0.41x in the fresh-process bench.
* **R0 (today's pandas route) vs R1 at 500k rows:**

  | run | polars | duckdb | ibis | slower by |
  |---|---|---|---|---|
  | re-run | x0.38 | x0.26 | x0.36 | 2.6–3.8x |
  | fresh-process | x0.35 | x0.24 | x0.49 | 2.0–4.2x |

  The claim was 2.1–2.7x. The direction is robust; the magnitude is wider than claimed.
* **All result checks matched the pandas reference** (`match` / `match(approx)`), so every lazy
  query was actually executed and was correct.
* **The x1 report contradicts itself.** The TL;DR and the "Workarounds" table still show the
  local-run numbers (R2 1.4–1.6x, 8–30x memory, "7–58x" pandas), while the appended cloud
  section supersedes them.

## 4. Verdicts

| id | verdict | note |
|---|---|---|
| .1 | confirmed | Defaults reproduced on pd2 and pd3, and match the source. Missed: under stable.v1, DuckDB and ibis inputs are `DataFrame`s with no `collect`; ibis-on-polars also collects to pyarrow. |
| .4 | partially_confirmed | Counts reproduce exactly (top-level nw), and 0 hidden computes was confirmed by an independent method. But dask's count depends on the input (read_parquet or np=1: 26/34). Under stable.v2 every backend loses `last(order_by)`. |
| .5 | partially_confirmed | True for main and stable.v2. Under stable.v1, `last()`, `shift().over()` and `map_batches` work on polars-lazy/pandas/modin, leaving only 3 by-design failures there. pyarrow pivot NotImplementedError confirmed. |
| .6 | partially_confirmed | Type, npartitions, connection and memtable facts confirmed. Cost on re-run is 1.17–2.25x, not 0.82–1.92x. |
| .7 | partially_confirmed | Re-run gives 5.1–32.5x, but 30x is an artefact of a no-op denominator. From a parquet scan it is 3.2–5.4x. Memory deltas come from an unreliable sampler. |
| .8 | partially_confirmed | Speedups confirmed, even larger in places. Memory reduction is 1.3–7x (fresh-process kernel high-water), not 6–35x+. R2b needs the unstable `last(order_by=)`. |
| .10 | partially_confirmed | Direction confirmed. Magnitude is 2.0–4.2x slower, not 2.1–2.7x. |
| .11 | partially_confirmed | The loop is slow (144–227x / 536x vs R1 on re-run), but against the realistic eager-iteration alternative it is 14–17x. |

Also confirmed, though not in the top 8: .2 (only 3 collect targets) and .3 (fidelity counts, with
the string→object caveat).

## 5. Implications
* **The C1 verdict ("true but misleading") holds.** The API you pin to matters, though:
  * if sktime adopts narwhals it must pin `stable.v2` (or main);
  * under v1, DuckDB and ibis aren't LazyFrames at all;
  * neither stable API offers `last(order_by=)`.
* **The value of staying lazy is real for speed: about 1.5–3.9x at 500k rows** on
  polars/DuckDB/ibis. The memory benefit is more modest than x1 reports (1.3–7x).
* For arbitrary per-series estimators, collect once and iterate eagerly. The loss against a lazy
  per-key loop is about 15x, not hundreds.
