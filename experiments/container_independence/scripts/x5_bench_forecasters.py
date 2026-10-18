"""x5: panel-native narwhals naive forecasters vs sktime NaiveForecaster (vectorized).

fit + predict(fh=1..12) on Panel data, n_series in {100, 1000, 10000} (dev).

Paths
  sk_pd        sktime NaiveForecaster on pd-multiindex (per-series loop in
               BaseForecaster._vectorize)
  sk_pl        same, input "polars_panel" (status quo for polars users:
               convert_to pandas inside, output converted back) - "last" only
  nw_pdmi      narwhals forecaster, pandas MI in / pandas MI out
  nw_pd        narwhals forecaster, pandas long frame in / out
  nw_pl        polars eager in / out
  nw_pll       polars lazy in, LazyFrame out, timing includes .collect()
  nw_pa        pyarrow Table in / out
  nw_duckdb    duckdb relation in, relation out, timing includes .arrow()
  nw_dask      dask in, dask out, timing includes .compute()

Run:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x5_bench_forecasters.py --grid dev
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import narwhals as nw
import numpy as np
import pandas as pd

from common import bench, env_info, make_panel_fast
from protos.x5_backends import long_from_mi, make_native, to_pandas_any
from protos.x5_nw_forecasters import NwPanelNaive

warnings.filterwarnings("ignore")
ED = Path(__file__).resolve().parents[1]

GRIDS = {
    "smoke": [(20, 30)],
    "dev": [(100, 100), (1_000, 100), (10_000, 50)],
    "large": [(1_000, 500), (10_000, 100), (50_000, 50)],
}
STRATS = {
    "last": dict(strategy="last"),
    "last_sp7": dict(strategy="last", sp=7),
    "mean": dict(strategy="mean"),
    "drift": dict(strategy="drift"),
}
FH = list(range(1, 13))


def sk_run(y, kw):
    from sktime.forecasting.naive import NaiveForecaster

    f = NaiveForecaster(**kw)
    return f.fit(y).predict(FH)


def nw_run(native, kw, keys=("h0",), time_col="time", collect=None):
    m = NwPanelNaive(**kw)
    if keys is None:
        m.fit(native)
    else:
        m.fit(native, keys=list(keys), time=time_col)
    out = m.predict(FH)
    if collect == "polars":
        out = out.collect()
    elif collect == "duckdb":
        # duckdb 1.5: .arrow() returns a RecordBatchReader, not a Table
        out = nw.from_native(out).collect("pyarrow").to_native()
    elif collect == "dask":
        out = out.compute()
    return out


def max_diff(sk, got):
    s = sk.reset_index()
    s.columns = ["h0", "time", "c0"]
    g = to_pandas_any(got) if not isinstance(got, pd.DataFrame) else got
    if isinstance(g.index, pd.MultiIndex):
        g = g.reset_index()
    g = g[["h0", "time", "c0"]].sort_values(["h0", "time"]).reset_index(drop=True)
    s = s.sort_values(["h0", "time"]).reset_index(drop=True)
    if len(g) != len(s):
        return float("inf")
    t_ok = (
        pd.to_datetime(g["time"]).astype("datetime64[ns]").values
        == pd.to_datetime(s["time"]).astype("datetime64[ns]").values
    ).all()
    if not t_ok:
        return float("inf")
    return float(np.max(np.abs(g["c0"].to_numpy(float) - s["c0"].to_numpy(float))))


def timed_once(fn):
    t0 = time.perf_counter()
    try:
        r = fn()
        return time.perf_counter() - t0, r, None
    except Exception as e:  # noqa: BLE001
        return time.perf_counter() - t0, None, f"{type(e).__name__}: {e}"[:150]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["smoke", "dev", "large"], default="dev")
    ap.add_argument("--skip-sktime-above", type=int, default=10**9)
    # sktime takes ~25-30 ms per series (fit+predict); at >=10k series only
    # run it for these strategies (others: unmeasured)
    ap.add_argument("--sk-big-strategies", default="last")
    args = ap.parse_args()

    from sktime.datatypes import convert_to

    out = {"env": env_info(), "grid": args.grid, "fh": FH, "rows": []}
    for n_series, n_time in GRIDS[args.grid]:
        y = make_panel_fast(n_series, n_time)
        lng = long_from_mi(y)
        natives = {
            b: make_native(lng, b)
            for b in ["pandas", "polars", "polars_lazy", "pyarrow", "duckdb", "dask"]
        }
        y_plsk = convert_to(y, to_type="polars_panel")
        big = n_series >= 10_000
        rep_nw = 3 if big else 5
        for sname, kw in STRATS.items():
            row = {"n_series": n_series, "n_time": n_time, "strategy": sname}
            t, err, diff = {}, {}, {}
            # sktime (slow): run once for big, min of 2 otherwise
            run_sk = n_series <= args.skip_sktime_above and (
                not big or sname in args.sk_big_strategies.split(",")
            )
            if run_sk:
                if big:
                    t["sk_pd"], sk_out, e = timed_once(lambda: sk_run(y, kw))
                else:
                    t["sk_pd"] = bench(lambda: sk_run(y, kw), repeat=2, warmup=0)
                    sk_out = sk_run(y, kw)
                if sname == "last" and not big:
                    t["sk_pl"], _, e = timed_once(lambda: sk_run(y_plsk, kw))
                    if e:
                        err["sk_pl"] = e
            else:
                sk_out = None

            paths = {
                "nw_pdmi": lambda: nw_run(y, kw, keys=None),
                "nw_pd": lambda: nw_run(natives["pandas"], kw),
                "nw_pl": lambda: nw_run(natives["polars"], kw),
                "nw_pll": lambda: nw_run(natives["polars_lazy"], kw, collect="polars"),
                "nw_pa": lambda: nw_run(natives["pyarrow"], kw),
                "nw_duckdb": lambda: nw_run(natives["duckdb"], kw, collect="duckdb"),
                "nw_dask": lambda: nw_run(natives["dask"], kw, collect="dask"),
            }
            for p, fn in paths.items():
                _, res, e = timed_once(fn)  # warm-up + correctness
                if e:
                    err[p] = e
                    continue
                if sk_out is not None:
                    diff[p] = max_diff(sk_out, res)
                try:
                    t[p] = bench(fn, repeat=rep_nw, warmup=0)
                except Exception as e2:  # noqa: BLE001 (dask is flaky)
                    err[p] = f"during bench: {type(e2).__name__}: {e2}"[:150]
            row["time_s"] = t
            row["max_abs_diff_vs_sktime"] = diff
            row["errors"] = err
            if "sk_pd" in t:
                row["speedup_vs_sk_pd"] = {
                    p: t["sk_pd"] / v for p, v in t.items() if p != "sk_pd"
                }
            out["rows"].append(row)
            print(
                f"n={n_series:>6} T={n_time} {sname:9s} "
                + " ".join(f"{p}={v*1e3:.1f}ms" for p, v in t.items())
            )
            if "sk_pd" in t:
                print(
                    "     speedup vs sk_pd: "
                    + " ".join(
                        f"{p}=x{v:.0f}" for p, v in row["speedup_vs_sk_pd"].items()
                    )
                )
            print("     max|diff| vs sktime:", {k: v for k, v in diff.items()})
            if err:
                print("     errors:", err)
            p = ED / "results" / f"x5_bench_forecasters_{args.grid}.json"
            p.write_text(json.dumps(out, indent=1, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
