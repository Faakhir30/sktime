"""x6: hub ingress/egress, slicing, reconstruct and forecaster benchmarks.

Compares the hub prototype (protos/x6_hub.py) against the CURRENT sktime path
(prepare_VectorizedDF / VectorizedDF.items / reconstruct / NaiveForecaster) in the
same process, interleaved, min-of-repeats (common.bench).

Run:
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/x6_bench.py --grid dev [--parts ...]
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ED, "protos"))

import x6_hub as H  # noqa: E402
from common import (  # noqa: E402
    DEV_GRID,
    LARGE_GRID,
    bench,
    env_info,
    make_panel_fast,
    to_polars_long,
)

warnings.simplefilter("ignore")
warnings.showwarning = lambda *a, **k: None  # sktime's per-clone FutureWarning

FH = np.arange(1, 11)


def abench(fn, budget=3.0, max_repeat=7):
    """bench() with repeats chosen so one cell stays within ~budget seconds."""
    t0 = time.perf_counter()
    fn()
    one = time.perf_counter() - t0
    if one * 2 > budget:
        return one, 1
    rep = int(max(1, min(max_repeat, budget / max(one, 1e-6) - 1)))
    return min(one, bench(fn, repeat=rep, warmup=0)), rep + 1


def sources(n_s, n_t, n_cols=1):
    import dask.dataframe as dd
    import pyarrow as pa

    from sktime.datatypes import convert_to

    X = make_panel_fast(n_s, n_t, n_cols=n_cols)
    pl_sk = convert_to(X, "polars_panel")
    pl_long = to_polars_long(X)
    pa_long = pa.Table.from_pandas(X.reset_index(), preserve_index=False)
    pa_long = pa_long.replace_schema_metadata(None)
    dask_sk = convert_to(X, "dask_panel")
    kw = {"key_cols": ["h0"], "time_col": "time"}
    return {
        "pandas_mi": (X, {}),
        "polars_sktime": (pl_sk, {}),
        "polars_long": (pl_long, kw),
        "polars_lazy": (pl_long.lazy(), kw),
        "arrow_long": (pa_long, kw),
        "dask_sktime": (dask_sk, {}),
    }


def current_ingress(name, src):
    """What sktime does today to get a vectorizable frame + schema."""
    from sktime.datatypes._vectorize import prepare_VectorizedDF

    if name in ("pandas_mi", "polars_sktime"):
        return lambda: prepare_VectorizedDF(src, iterate_as="Series", is_scitype="Panel")
    if name == "dask_sktime":
        # sktime's mtype check rejects dask-expr DataFrames (dask 2026.8), so call
        # the dask_panel -> pandas converter function directly (what convert_to does)
        from sktime.datatypes._adapter.dask_to_pd import convert_dask_to_pandas

        return lambda: prepare_VectorizedDF(convert_dask_to_pandas(src),
                                            iterate_as="Series", is_scitype="Panel")
    if name == "polars_long":  # user must rename/convert to the sktime convention
        return lambda: prepare_VectorizedDF(
            src.to_pandas().set_index(["h0", "time"]), iterate_as="Series",
            is_scitype="Panel")
    if name == "polars_lazy":
        return lambda: prepare_VectorizedDF(
            src.collect().to_pandas().set_index(["h0", "time"]), iterate_as="Series",
            is_scitype="Panel")
    if name == "arrow_long":
        return lambda: prepare_VectorizedDF(
            src.to_pandas().set_index(["h0", "time"]), iterate_as="Series",
            is_scitype="Panel")
    raise KeyError(name)


def src_views(src):
    import polars as pl
    import pyarrow as pa

    if isinstance(src, pd.DataFrame):
        return [src.to_numpy(copy=False), np.asarray(src.index.levels[-1])]
    if isinstance(src, pl.DataFrame):
        out = []
        for c in src.columns:
            try:
                out.append(src[c].to_numpy())
            except Exception:  # noqa: BLE001
                pass
        return out
    if isinstance(src, pa.Table):
        out = []
        for c in src.column_names:
            for ch in src.column(c).chunks:
                try:
                    out.append(ch.to_numpy(zero_copy_only=True))
                except Exception:  # noqa: BLE001
                    pass
        return out
    return []


def copied_bytes(hub, views):
    """Bytes of hub arrays that do NOT share memory with the source buffers."""
    def owned(a):
        return 0 if any(np.shares_memory(a, v) for v in views) else a.nbytes
    total = owned(hub.values) + owned(hub.time) + hub.indptr.nbytes
    total += sum(np.asarray(k).nbytes for k in hub.key_values)
    return total


# ============================================================================ parts
def part_ingress_egress(grid, n_cols):
    from sktime.datatypes import convert_to

    rows = []
    for n_s, n_t in grid:
        srcs = sources(n_s, n_t, n_cols)
        X = srcs["pandas_mi"][0]
        for name, (src, kw) in srcs.items():
            rec = {"part": "ingress", "source": name, "n_series": n_s, "n_time": n_t,
                   "n_cols": n_cols}
            hub = H.from_native(src, **kw)
            views = src_views(src)
            rec["values_zero_copy"] = any(np.shares_memory(hub.values, v)
                                          for v in views)
            rec["time_zero_copy"] = any(np.shares_memory(hub.time, v) for v in views)
            rec["hub_copied_MB"] = copied_bytes(hub, views) / 2**20
            rec["hub_total_MB"] = hub.nbytes() / 2**20
            t_hub, r1 = abench(lambda: H.from_native(src, **kw))
            t_cur, r2 = abench(current_ingress(name, src))
            rec.update(t_hub_s=t_hub, t_current_s=t_cur, speedup=t_cur / t_hub,
                       repeats=[r1, r2])
            # current: bytes of the pandas MI frame produced (if converted)
            if name != "pandas_mi":
                rec["current_copied_MB"] = X.memory_usage(index=True).sum() / 2**20
            else:
                rec["current_copied_MB"] = 0.0
            rows.append(rec)
            print(f"ingress {name:14s} {n_s:6d}x{n_t:<4d} hub {t_hub*1e3:9.2f}ms "
                  f"cur {t_cur*1e3:9.2f}ms x{t_cur/t_hub:7.1f} "
                  f"zc(v,t)=({rec['values_zero_copy']},{rec['time_zero_copy']}) "
                  f"copied hub {rec['hub_copied_MB']:.2f}MB cur "
                  f"{rec['current_copied_MB']:.2f}MB", flush=True)

            # ---- egress: reconstruct per-instance predictions (h=10) to native
            from sktime.datatypes._vectorize import prepare_VectorizedDF

            vec, X_mi = prepare_VectorizedDF(X, iterate_as="Series", is_scitype="Panel")
            h = len(FH)
            rng = np.random.default_rng(0)
            fut = pd.date_range(X.index.levels[1][-1], periods=h + 1, freq="D")[1:]
            fut.name = "time"
            pd_outs = [pd.DataFrame(rng.normal(size=(h, n_cols)), index=fut,
                                    columns=X.columns) for _ in range(n_s)]
            np_outs = [(fut.values, o.to_numpy()) for o in pd_outs]
            orig_mtype = {"pandas_mi": None, "polars_sktime": "polars_panel",
                          "dask_sktime": "dask_panel"}.get(name, "skip")

            def cur_egress():
                r = vec.reconstruct(pd_outs, overwrite_index=True)
                if orig_mtype == "dask_panel":
                    from sktime.datatypes._adapter.dask_to_pd import (
                        convert_pandas_to_dask,
                    )

                    r = convert_pandas_to_dask(r, npartitions=src.npartitions)
                elif orig_mtype not in (None, "skip"):
                    r = convert_to(r, orig_mtype)
                return r

            def hub_egress():
                return H.to_native(hub.from_outputs(np_outs))

            def hub_reconstruct_only():
                return hub.from_outputs(np_outs)

            # correctness of egress vs current (pandas) path
            if name == "pandas_mi":
                pd.testing.assert_frame_equal(hub_egress(), cur_egress(),
                                              check_freq=False)
            e = {"part": "egress", "source": name, "n_series": n_s, "n_time": n_t,
                 "n_cols": n_cols, "native_type": type(hub_egress()).__name__}
            e["t_hub_s"], _ = abench(hub_egress)
            e["t_hub_reconstruct_only_s"], _ = abench(hub_reconstruct_only)
            if orig_mtype != "skip":
                e["t_current_s"], _ = abench(cur_egress)
                e["speedup"] = e["t_current_s"] / e["t_hub_s"]
            rows.append(e)
            print(f"egress  {name:14s} {n_s:6d}x{n_t:<4d} hub {e['t_hub_s']*1e3:9.2f}ms "
                  f"(concat only {e['t_hub_reconstruct_only_s']*1e3:.2f}ms) cur "
                  f"{e.get('t_current_s', float('nan'))*1e3:9.2f}ms "
                  f"x{e.get('speedup', float('nan')):6.1f} -> {e['native_type']}",
                  flush=True)
    return rows


def part_slicing(grid):
    from sktime.datatypes._vectorize import prepare_VectorizedDF

    rows = []
    for n_s, n_t in grid:
        X = make_panel_fast(n_s, n_t)
        vec, X_mi = prepare_VectorizedDF(X, iterate_as="Series", is_scitype="Panel")
        hub = H.from_pandas(X)
        sv = hub.slice(0)
        s_pd = sv.to_pandas(as_series=True)
        rec = {"part": "slicing", "n_series": n_s, "n_time": n_t,
               "pandas_view_shares_memory": bool(np.shares_memory(s_pd.to_numpy(),
                                                                  hub.values)),
               "freq_trusted": str(s_pd.index.freq)}

        def cur():
            for _ in vec.items(X_mi):
                pass

        def hub_views():
            for _ in hub.iter_slices():
                pass

        def hub_pd(mode):
            def f():
                for s in hub.iter_slices():
                    s.to_pandas(as_series=True, freq_mode=mode)
            return f

        rec["t_current_items_s"], _ = abench(cur)
        rec["t_hub_views_s"], _ = abench(hub_views)
        for mode in ["public", "none", "trusted", "ctor", "infer"]:
            rec[f"t_hub_pandas_{mode}_s"], _ = abench(hub_pd(mode))
        per = {k.replace("_s", "_us_per_slice"): v / n_s * 1e6
               for k, v in rec.items() if k.startswith("t_")}
        rec.update(per)
        rows.append(rec)
        print(f"slicing {n_s:6d}x{n_t:<4d} us/slice: current {per['t_current_items_us_per_slice']:.1f} | "
              f"hub views {per['t_hub_views_us_per_slice']:.2f} | pandas public "
              f"{per['t_hub_pandas_public_us_per_slice']:.1f} none "
              f"{per['t_hub_pandas_none_us_per_slice']:.1f} trusted "
              f"{per['t_hub_pandas_trusted_us_per_slice']:.1f} ctor "
              f"{per['t_hub_pandas_ctor_us_per_slice']:.1f} infer "
              f"{per['t_hub_pandas_infer_us_per_slice']:.1f} | view shares mem "
              f"{rec['pandas_view_shares_memory']}", flush=True)
    return rows


def _pandas_semantic_runner(strategy):
    """Per slice: lazily materialize pd.Series from hub views, run sktime estimator."""
    from sktime.forecasting.naive import NaiveForecaster

    template = NaiveForecaster(strategy=strategy)

    def run(hub):
        outs = []
        for sv in hub.iter_slices():
            y = sv.to_pandas(as_series=True, freq_mode="trusted")
            yp = template.clone().fit(y, fh=FH).predict()
            outs.append((yp.index.values, yp.to_numpy()))
        return hub.from_outputs(outs)

    return run


def _pandas_inner_runner(strategy):
    """Per slice: pandas view + NaiveForecaster INNER _fit/_predict (no framework)."""
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.naive import NaiveForecaster

    template = NaiveForecaster(strategy=strategy)

    def run(hub):
        outs = []
        fh_rel = ForecastingHorizon(FH, is_relative=True)
        for sv in hub.iter_slices():
            y = sv.to_pandas(as_series=True, freq_mode="trusted")
            f = template.clone()
            f._y = y
            f._set_cutoff_from_y(y)
            f._fh = fh_rel
            f._fit(y, None, fh_rel)
            yp = f._predict(fh_rel, None)
            outs.append((yp.index.values, yp.to_numpy()))
        return hub.from_outputs(outs)

    return run


def part_forecaster(grid, current_max, strategies):
    from sktime.datatypes import convert_to
    from sktime.forecasting.naive import NaiveForecaster

    rows = []
    for n_s, n_t in grid:
        X = make_panel_fast(n_s, n_t)
        pl_sk = convert_to(X, "polars_panel")
        for strategy in strategies:
            for src_name, src in [("pandas_mi", X), ("polars_sktime", pl_sk)]:
                rec = {"part": "forecaster", "strategy": strategy, "source": src_name,
                       "n_series": n_s, "n_time": n_t}

                def cur():
                    return NaiveForecaster(strategy=strategy).fit(src, fh=FH).predict()

                def hub_loop():
                    hub = H.from_native(src)
                    out = H.hub_vectorize(hub, lambda: H.HubNaive(strategy), FH)
                    return H.to_native(out)

                def hub_batched():
                    hub = H.from_native(src)
                    return H.to_native(H.hub_naive_batched(hub, FH, strategy))

                sem = _pandas_semantic_runner(strategy)
                inner = _pandas_inner_runner(strategy)

                def hub_pd_sem():
                    return H.to_native(sem(H.from_native(src)))

                def hub_pd_inner():
                    return H.to_native(inner(H.from_native(src)))

                # correctness vs current at small size
                if n_s <= 100:
                    ref = cur()
                    for fn in (hub_loop, hub_batched, hub_pd_sem, hub_pd_inner):
                        got = fn()
                        if src_name == "pandas_mi":
                            pd.testing.assert_frame_equal(got, ref, check_freq=False)
                        else:
                            assert got.equals(ref), (fn.__name__, got, ref)
                    rec["equal_to_current"] = True
                rec["t_hub_batched_s"], _ = abench(hub_batched)
                rec["t_hub_loop_s"], _ = abench(hub_loop)
                if n_s <= current_max:
                    rec["t_current_s"], _ = abench(cur, budget=1.0)
                    rec["t_hub_pandas_semantic_s"], _ = abench(hub_pd_sem, budget=1.0)
                    rec["t_hub_pandas_inner_s"], _ = abench(hub_pd_inner, budget=1.0)
                    c = rec["t_current_s"]
                    rec["speedup_loop"] = c / rec["t_hub_loop_s"]
                    rec["speedup_batched"] = c / rec["t_hub_batched_s"]
                    rec["speedup_pandas_semantic"] = c / rec["t_hub_pandas_semantic_s"]
                    rec["speedup_pandas_inner"] = c / rec["t_hub_pandas_inner_s"]
                rows.append(rec)
                print(f"forecast {strategy:5s} {src_name:13s} {n_s:6d}x{n_t:<4d} "
                      f"cur {rec.get('t_current_s', float('nan')):8.3f}s | hub+sktime/slice "
                      f"{rec.get('t_hub_pandas_semantic_s', float('nan')):8.3f}s | "
                      f"hub+pandas inner {rec.get('t_hub_pandas_inner_s', float('nan')):8.3f}s | "
                      f"hub loop {rec['t_hub_loop_s']:.4f}s | batched "
                      f"{rec['t_hub_batched_s']:.4f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--parts", nargs="+",
                    default=["ingress", "slicing", "forecaster"])
    ap.add_argument("--current-max", type=int, default=10_000)
    ap.add_argument("--n-cols", type=int, default=1)
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    rows = []
    t0 = time.time()
    if "ingress" in args.parts:
        rows += part_ingress_egress(grid, args.n_cols)
    if "slicing" in args.parts:
        rows += part_slicing(grid)
    if "forecaster" in args.parts:
        rows += part_forecaster(grid, args.current_max, ["last", "drift"])
    tag = f"pd{pd.__version__.split('.')[0]}_{args.grid}_c{args.n_cols}_" + \
        "-".join(args.parts)
    out = os.path.join(ED, "results", f"x6_bench_{tag}.json")
    with open(out, "w") as f:
        json.dump({"env": env_info(), "args": vars(args), "wall_s": time.time() - t0,
                   "rows": rows}, f, indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    main()
