# g5_api_surface: API surface beyond y-only fit+predict (critic G5)

**Headline.** API surface coverage over 24 cells (12 operations × Series/Panel/Hierarchical).
Two scores per cell: "runs without error, or raises correctly", then "strict-equal to stock-pandas and output backend = input backend".

| approach | pandas | polars |
|---|---|---|
| stock sktime | 22/24, 22/24 | **16/24, 9/24** |
| x4 adapter (native + cutoff bridge), and the x4nc / x4fix variants | 22/24, 22/24 | **16/24, 5/24** |
| x6 hub, boundary-wired (x6b) | 20/24, 16/24 | 20/24, 7/24 |

Main findings:
- **No prototype makes the non-fit/predict surface work on polars.** x4 is *worse* than stock on polars: strict cells drop from 9 to 5. It loses two safety and fidelity properties that stock gets only by accident, from per-slice pandas re-entry:
  - the Panel X/y containment check (it becomes silent);
  - the `y` column name on Panel output.
- x6b fixes the polars Series `update_predict` and `predict_residuals` failures, because stock runs on pandas inside. But its pandas round trip drops the MultiIndex time-level `freq`, so Panel `update_predict` breaks for **pandas** input too.
- The x4 `reverse_order` bug is reproduced (x4 returns 2000-01-30 where 2000-01-01 is expected). In practice it is masked: polars `update_predict` already fails earlier, at `_base.py:2726` / `:2838`.
- New stock bugs found here:
  - polars→pandas conversion raises for any time axis with fewer than 3 points. This means **`update()` with 1–2 new polars rows fails**, and short-horizon polars predictions cannot be converted back.
  - Panel `predict_residuals` fails **even for pandas**.
  - polars `pl.DataFrame` Series output renames the column `y` → `"0"`.
  - `evaluate()` returns pandas `y_pred` for polars input.

## Setup

- Env: own venv `scratchpad/venvs/g5_api_surface` (G5_PY). It mirrors EXP_PY pins (pandas 2.3.3, polars 2.0.0, narwhals 2.27.1, numpy 2.2.6, sklearn 1.9.1) plus statsmodels 0.15.0, which `Deseasonalizer` needs and EXP_PY lacks. sktime = WT checkout, unmodified. pandas 3 was **not** run (EXP_PY3 has no statsmodels). Treat pd3 as unmeasured.
- Data: daily `DatetimeIndex` (freq D), 70 points.
  - Series: y=`["y"]`, X=`["x1","x2"]`.
  - Panel: 4 instances.
  - Hierarchical: 2×2 instances, `Aggregator(flatten_single_levels=False)` adds `__total`.
  - Train = first 45 points; update = 6 points; `update_predict` test = points 45–59.
  - Polars inputs come from sktime `convert_to(..., "pl.DataFrame" | "polars_panel" | "polars_hierarchical")` (`__index__*` columns).
- Estimators:
  - `NaiveForecaster("last")`;
  - `make_reduction(LinearRegression(), window_length=5, strategy="recursive", pooling="local")`, the exogenous model (tags `capability:exogenous`, `X-y-must-have-same-index`, inner `pd.Series`, so it is vectorized on Panel);
  - `TransformedTargetForecaster([Deseasonalizer(sp=7), Naive])`;
  - `ForecastingGridSearchCV(Naive, ExpandingWindowSplitter(30,5,fh=[1,2,3]), strategy∈{last,mean,drift})`;
  - `ReconcilerForecaster(Naive, method="ols")`;
  - `evaluate(..., ExpandingWindowSplitter(initial_window=40, step_length=5, fh=[1,2,3]), return_data=True)`.
- Approaches:
  - **stock**: the shipped code.
  - **x4**: `x4_instrument.patched_base("native", cutoff=True)`. `VectorizedDF2` iterates in the input backend, plus `adapter_get_cutoff`.
  - **x4nc**: x4 without the cutoff bridge.
  - **x4fix**: x4 with a cutoff bridge that delegates to stock `get_cutoff` when `reverse_order=True` (the minimal fix for the x4 verifier's bug).
  - **x6b**: the x6 hub is *not wired into BaseForecaster* (x6 report: "not measured: sktime-wide integration"), so it is tested as the only wiring that exists today:
    - every data argument goes native → `from_polars`/`from_pandas` → `to_pandas` → stock sktime;
    - every frame output goes `from_pandas` → `to_polars` with the input's key/time metadata (pandas input = hub round trip).
    - Hub-native compute (`HubNaive`) covers fit+predict only, so it has no cells here.
- Equality:
  - outputs are normalised to pandas with `convert_to` and compared to stock+pandas with `assert_frame_equal(check_freq=False)` → `=`;
  - `~` = numeric values equal (`np.allclose`) but labels, names or freq differ;
  - `cutoff` is compared as an Index including `freq`;
  - `pdOut` = the frame output is pandas although the input was polars.
- Timing was not a goal. Per-cell wall time is in the JSON; the whole matrix took 61 s.

Commands (from WT; `G5_PY=$SCRATCH/venvs/g5_api_surface/bin/python`, `PP=.:experiments/container_independence:experiments/container_independence/protos`):
```
PYTHONPATH=$PP $G5_PY -W ignore experiments/container_independence/scripts/g5_api_surface.py   # -> results/g5_api_surface.json (+ _run.log)
python3 experiments/container_independence/scripts/g5_tabulate.py                              # -> results/g5_api_surface_table.md
PYTHONPATH=$PP $G5_PY -W ignore experiments/container_independence/scripts/g5_probes.py        # -> results/g5_probes.json
PYTHONPATH=$PP $G5_PY -W ignore experiments/container_independence/scripts/g5_short_output_probe.py
PYTHONPATH=$PP $G5_PY -W ignore experiments/container_independence/scripts/g5_colx_check.py
```

## 1. Matrix

Legend:
- `PASS=`: strict-equal to stock+pandas.
- `PASS~`: values equal, labels/freq differ (see §2).
- `pdOut`: pandas output for polars input.
- `raises`: the expected ValueError was raised.
- `SILENT`: an invalid input was accepted.
- `noref`: no stock reference exists because stock fails.
- `pa` / `po`: pandas / polars input.

| op | scitype | stock pa | stock po | x4 pa | x4 po | x4nc pa | x4nc po | x4fix pa | x4fix po | x6b pa | x6b po |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline fit(y)+predict [Naive] | Series | PASS= | PASS~ | PASS= | PASS~ | PASS= | PASS~ | PASS= | PASS~ | PASS= | PASS~ |
| baseline fit(y)+predict [Naive] | Panel | PASS= | PASS= | PASS= | PASS~ | PASS= | PASS~ | PASS= | PASS~ | PASS~ | PASS~ |
| fit(y,X)+predict(X) | Series | PASS= | PASS~ | PASS= | PASS~ | PASS= | PASS~ | PASS= | PASS~ | PASS= | PASS~ |
| fit(y,X)+predict(X) | Panel | PASS= | PASS= | PASS= | PASS~ | PASS= | PASS~ | PASS= | PASS~ | PASS~ | PASS~ |
| fit(y,X_disjoint) must raise | Series | raises | SILENT | raises | SILENT | raises | SILENT | raises | SILENT | raises | raises |
| fit(y,X_disjoint) must raise | Panel | raises | raises | raises | SILENT | raises | SILENT | raises | SILENT | raises | raises |
| update(y_new,X_new)+predict | Series | PASS= | PASS~ | PASS= | PASS~ | PASS= | PASS~ | PASS= | PASS~ | PASS= | PASS~ |
| update(y_new,X_new)+predict | Panel | PASS= | PASS= | PASS= | PASS~ | PASS= | PASS~ | PASS= | PASS~ | PASS~ | PASS~ |
| update_predict(y_test) [Naive] | Series | PASS= | FAIL | PASS= | FAIL | PASS= | FAIL | PASS= | FAIL | PASS= | PASS~ |
| update_predict(y_test) [Naive] | Panel | PASS= | FAIL | PASS= | FAIL | PASS= | FAIL | PASS= | FAIL | FAIL | FAIL |
| update_predict(y_test,X) [reduction] | Series | PASS= | FAIL | PASS= | FAIL | PASS= | FAIL | PASS= | FAIL | PASS= | PASS~ |
| update_predict(y_test,X) [reduction] | Panel | PASS= | FAIL | PASS= | FAIL | PASS= | FAIL | PASS= | FAIL | FAIL | FAIL |
| predict_residuals(y_oos) | Series | PASS= | FAIL | PASS= | FAIL | PASS= | FAIL | PASS= | FAIL | PASS= | PASS= |
| predict_residuals(y_oos) | Panel | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| evaluate(Naive, ExpandingWindow) | Series | PASS= | PASS=,pdOut | PASS= | PASS=,pdOut | PASS= | PASS=,pdOut | PASS= | PASS=,pdOut | PASS= | PASS~ |
| evaluate(Naive, ExpandingWindow) | Panel | PASS= | PASS=,pdOut | PASS= | PASS=,pdOut | PASS= | PASS=,pdOut | PASS= | PASS=,pdOut | PASS= | PASS= |
| evaluate(reduction, X) | Series | PASS= | PASS=,pdOut | PASS= | PASS=,pdOut | PASS= | PASS=,pdOut | PASS= | PASS=,pdOut | PASS= | PASS~ |
| evaluate(reduction, X) | Panel | PASS= | PASS=,pdOut | PASS= | PASS=,pdOut | PASS= | PASS=,pdOut | PASS= | PASS=,pdOut | PASS= | PASS= |
| TTF([Deseasonalizer, Naive]) | Series | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS~ |
| TTF([Deseasonalizer, Naive]) | Panel | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS~ | PASS~ |
| ForecastingGridSearchCV(Naive) | Series | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS~ |
| ForecastingGridSearchCV(Naive) | Panel | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= |
| ReconcilerForecaster(ols) | Hierarchical | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= | PASS= |
| Panel with column named 'X' | Panel | FAIL | FAIL | FAIL | PASS(noref) | FAIL | PASS(noref) | FAIL | PASS(noref) | FAIL | FAIL |

Output types (from the JSON):
- `predict`, `update_predict` and `predict_residuals` return the input backend (`polars.DataFrame` for polars) in every passing cell, except `evaluate`, whose `y_pred` is pandas for stock and x4.
- `cutoff` is always a pandas `DatetimeIndex` (length 1). `best_params_` is a dict.

How the x4 variants behave:
- x4, x4nc and x4fix are identical in every cell, because no cell reaches the cutoff `reverse_order` path with a polars `y`. See §2.4.
- Composites (TTF, GridCV, Reconciler) and `evaluate` give the same results as stock under x4. Expected (inferred, not traced): `evaluate` converts polars to pandas on entry (`model_evaluation/_functions.py:792,804`). The composites' outer inner-mtypes include pandas multiindex mtypes, so the x4 polars-native vectorization is never exercised inside them. This is the a2-verifier "verdicts are not transitive" point. A container-native architecture has to cover composites and tuning explicitly, otherwise they silently fall back to pandas.

## 2. Failures and deviations: root causes (file:line)

### 2.1 Stock bugs (also present under every prototype that runs stock internals)

| # | symptom | where | root cause |
|---|---|---|---|
| S1 | `update_predict` with polars y, Series or Panel: `ValueError: y_preds must be a list of pd.Series or pd.DataFrame` | `forecasting/base/_base.py:2738-2746` → `:2838` | Each `y_pred` is converted back to the user mtype (polars) **inside the loop**, and then fed to the pandas-only `_format_moving_cutoff_predictions`. Confirms a1.12. |
| S2 | `predict_residuals` with polars y: `TypeError: y must be a supported Series mtype` | `_base.py:1715` (`isinstance(y,(pd.DataFrame,pd.Series))`) → `:1728` | No backend dispatch; the fh is derived from `y.index`. Confirms a1.12. |
| S3 | **new:** `predict_residuals` on a **pandas Panel**: `TypeError: Invalid fh` | `_base.py:1716` → `forecasting/base/_fh.py:311/139` | `ForecastingHorizon(y.index)` gets the full `(h0,time)` MultiIndex. fh accepts only a flat time index, so Panel residuals never worked, whatever the backend. |
| S4 | Disjoint-dates X accepted for **polars Series** | `_base.py:2072-2073` → `datatypes/_utilities.py:87-90` (`get_index_for_series`: `hasattr(obj,"index")`) | `check_equal_time_index` runs on the raw polars inputs and duck-types them to a `RangeIndex`. Confirms a1.10. |
| S5 | Panel disjoint X *is* caught in stock (pandas and polars), but only **by accident** | raised at `datatypes/_vectorize.py:843` → `_base.py:515` → `_base.py:2073` → `utils/validation/series.py:283` (probe P1) | The top-level check is gated on `X_scitype == "Series"` (`_base.py:2072`; TODO comment). For Panel the check runs only inside per-slice **re-entry** of `fit` on pandas slices. |
| S6 | Panel with a column literally named `"X"`: `TypeError: No valid mtype` (pandas and polars) | `datatypes/_utilities.py:221-222` (`if hasattr(obj,"X"): obj = obj.X`) → `:235` | A stale duck-type for the removed `VectorizedDF.X`. A second such site is at `_utilities.py:59` (`get_time_index`). Confirms a1.11. |
| S7 | **new:** polars **Series** output renames the value column `y` → `"0"` (Naive, reduction, update; any `pd.Series`-inner forecaster) | `datatypes/_series/_convert.py:310-314` (`convert_uvs_to_polars_as_series` ignores `store`) | The pandas path restores the name from `store["columns"]` (`_series/_convert.py:81-85`). The polars egress converter drops the store. The forecaster's `_converter_store_y` does hold `Index(['y'])` (probe P4). Values are equal. |
| S8 | **new:** polars→pandas conversion raises `ValueError: Need at least 3 dates to infer frequency` for any datetime axis with < 3 points. **`update(y_new)` with 1 or 2 new polars rows fails** (Series and Panel); polars predictions with `len(fh) < 3` cannot be `convert_to`'d back to pandas. | `datatypes/_adapter/polars.py:150-157` (`set_freq` calls `pd.infer_freq` unguarded), reached from `_series/_convert.py:289` and `_panel/_convert.py:1150` | Unguarded `pd.infer_freq`. Measured: update with n_new ∈ {1,2} FAILS, n_new=3 ok, for both `pl.DataFrame` and `polars_panel` (`g5_short_output_probe.py`). This matters for streaming-style `update`. |
| S9 | **new:** `evaluate()` with polars y/X returns pandas `y_pred`/`y_test` frames | `forecasting/model_evaluation/_functions.py:792,804` | `evaluate` converts the inputs to `PANDAS_MTYPES` up front. Values are equal. Every tuner (`ForecastingGridSearchCV` etc.) therefore runs on pandas internally, whatever the container. |
| S10 | `update_predict` with `freq=None` datetime: `TypeError`/`NullFrequencyError` | `_base.py:2721-2722` → `utils/datetime.py:235` (`_shift`) | The cutoff shift needs pandas `freq` (a1 report line 449). Reproduced here via x6b (§2.3). |

### 2.2 x4 adapter (`protos/x4_adapter.py`, `protos/x4_instrument.py`)

| # | symptom | where | root cause |
|---|---|---|---|
| A1 | **Regression:** Panel disjoint X is silently accepted under x4 (stock raises) | `x4_adapter.VectorizedDF2` yields polars slices → `_base.py:2073` → `_utilities.py:87-90` | S5's accidental safety net depends on per-slice pandas re-entry. With native polars slices the per-slice `check_equal_time_index` duck-types to `RangeIndex`, the same as S4. Probe P1: `x4_polars` gives "NO ERROR"; `stock_polars` and `x4_pandas` raise. |
| A2 | **Regression:** Panel polars output column `y` → `"0"` (baseline, fit_X, update) | per-slice `pl.DataFrame`→`pd.Series`→`pl.DataFrame` round trip | The same converter as S7, now hit **per slice** (each leaf's egress converts to its polars input mtype). Stock converts the panel once to pandas and is unaffected. |
| A3 | `update_predict` on polars Panel: `AttributeError: 'DataFrame' object has no attribute 'iloc'` (stock fails later, at S1) | `_base.py:2726` (`y.iloc[new_window]`) | Under x4, `_check_X_y` keeps `y_inner` as polars, and `_predict_moving_cutoff` is pandas-positional. |
| A4 | **cutoff bridge ignores `reverse_order`** (x4 verifier) | `protos/x4_adapter.py:934` (`**kwargs` swallows `reverse_order`) → `:951` (`ad.time_max`) | Probe P2 (Series and Panel polars): stock gives `2000-01-01` (freq D). The x4 bridge gives **`2000-01-30`** (`x4_correct: false`). It is latent in the matrix only because A3/S1 fail first. Once those are fixed, `update_predict` would start from a wrong initial cutoff (`_base.py:2721`). x4fix (delegate on `reverse_order=True`) removes it. A proper fix needs an adapter `time_min` op. |
| A5 | Column named `"X"` **passes** under x4 for polars (stock fails) | — | The x4 path never calls `get_cutoff` on a pandas frame that has an `X` column. Values match stock on a column `"x"` (`g5_colx_check.py`: `np.allclose=True`), but the column name becomes `"0"` (A2). This is a side effect, not a fix of S6. |

### 2.3 x6 hub at the boundary (`protos/x6_hub.py`)

| # | symptom | where | root cause |
|---|---|---|---|
| H1 | Panel `cutoff.freq` is `None` instead of `Day` (pandas **and** polars input). `update_predict` on Panel then fails with `NullFrequencyError` (S10) | `protos/x6_hub.py:496` (`time_to_pandas_index(tu, tm, freq_mode="none")` for the MultiIndex time level) | The hub round trip drops the pandas MultiIndex time-level `freq`. Stock `get_cutoff` reads `freq` from that level (probe P3: `time_level_freq_orig=<Day>`, `back=None`, `frame_equal=True`). The `tmeta` freq is known but is not re-applied. |
| H2 | polars-sourced hub → pandas keeps raw `__index__h0`/`__index__time` level names | `protos/x6_hub.py:498-499` (`names=[*hub.key_names, hub.time_name]`) | `from_polars` stores the sktime `__index__` column names verbatim, and `to_pandas` does not strip the prefix (probe P3). It was harmless here because outputs are re-egressed to polars. |
| H3 | Series outputs re-egressed to polars carry time name `time`, while stock+pandas predictions have an unnamed time index (`~` cells) | x6b egress (`g5_api_surface.py:hub_out`) re-uses the input's `time_col` | Not a hub bug: stock's own predict output drops the index name. It shows that "output index names" is an unspecified contract today. |
| H4 | x6b *fixes* polars Series `update_predict` / `predict_residuals` / disjoint-X | — | Only because stock runs on pandas inside. It is the same as converting once at the boundary, with no container value realized. |

## 3. Known-issue checklist (from a1 / x4 verifiers)

- a1.10, polars X/y containment passes on disjoint dates: **reproduced** for Series (S4). For Panel, stock catches it only via re-entry (S5), and **x4 loses it** (A1).
- a1.11, column named "X": **reproduced**, pandas and polars (S6). x4 sidesteps it for polars (A5).
- a1.12, `update_predict` / `predict_residuals` polars failures: **reproduced** at `_base.py:2838` / `:1728`. x4 moves the `update_predict` failure earlier, to `:2726` (A3).
- x4 cutoff `reverse_order`: **reproduced** by direct call (A4). It is masked in the end-to-end runs by A3/S1.

## 4. Boundary invariants any new architecture must keep

Each invariant cites the cell or probe that shows it is violated or fragile today.

1. **Return type follows input type** for every frame-valued public output: `predict*`, `update_predict`, `predict_residuals`, and also `evaluate(return_data=True)` `y_pred`/`y_test` (S9). Today `cutoff`/`fh` stay pandas `Index` objects. That is O(1) metadata and acceptable, but it must be stated in the contract.
2. **Labels survive the round trip**: value column names (S7, A2), index-level names without backend-specific prefixes (H2), and hierarchy level names incl. `__total` (Reconciler passes today). Equality tests must compare labels, not only values. `~` cells pass value checks and are still user-visible breakage.
3. **Time-axis semantics are carried as metadata, not inferred from a pandas index attribute.**
   - freq must survive any re-layout (H1).
   - It must not be required for ≥ 3 points (S8: `update` with 1–2 rows).
   - `freq=None` must be handled (S10).
   - The cutoff needs both `max` and `min` (`reverse_order`) per backend (A4).
4. **Validation must run at the top level on the canonical representation, once, for every scitype.** It must not depend on per-slice re-entry (S5/A1) or on `hasattr(..., "index"/"X")` duck typing (S4, S6). In particular X/y time containment for Panel/Hierarchical must be checked by a backend-aware join or anti-join on (keys, time).
5. **No duck-typed attribute lookups on user frames** (`hasattr(obj,"X")`, `hasattr(obj,"index")`): user column names can collide (S6). Dispatch must go through an explicit backend registry (a1 P1 `backend_of`).
6. **Moving-cutoff APIs (`update_predict`, `evaluate`, tuners) must work on the internal representation.**
   - Slice windows by (key, time) position without `.iloc` (A3).
   - Collect per-window predictions internally, and convert to the user backend **once, after** formatting (S1).
   - Today `evaluate` gets backend-agnosticism by converting to pandas at entry (S9). A new architecture must either keep that explicitly (and then convert the outputs back) or provide a native splitter.
7. **Residuals are an aligned (key,time) join + subtraction**, with fh derived from the time level only (S2, S3). This also fixes Panel residuals for pandas.
8. **Composites and tuners must be covered by the same dispatch.** Under x4 every composite cell equals stock only because the composite's inner mtype converts to pandas before the patched vectorizer runs. A design that is fast for leaves but silently pandas-converts in TTF/GridCV/Reconciler realizes no container value where most real pipelines live.
9. **Errors must stay errors.** A new path may not turn a raised `ValueError` into silent acceptance (A1), and it may not turn a valid call into a crash (H1 breaking pandas `update_predict` via the hub round trip). The conformance suite should include `fit(y, X_disjoint)` must-raise cells per scitype and backend.
10. **Equality reference = stock pandas, including `cutoff.freq`.** Check `cutoff` type and freq explicitly (H1 was invisible in `frame.equals`, probe P3 `frame_equal=True`).

## 5. Caveats

- Small data: 4 instances, 70 points. This is a correctness matrix, not a performance one. No timing claims.
- x6 was evaluated only as a boundary wiring, because no BaseForecaster integration exists. Its cells measure spoke fidelity, not hub-native compute.
- One exogenous forecaster (local reduction) and one pandas-inner leaf (Naive) were used. Forecasters with `pd-multiindex` inner (the 29 global ones) bypass vectorization and were not covered (G6).
- pandas 3 (EXP_PY3) was not run.
- The x4 composite pass-through explanation (§1) is inferred from strict equality plus the inner-mtype tags. The vectorizer's inputs inside composites were not traced.
