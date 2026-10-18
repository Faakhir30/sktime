"""x7: can ONE narwhals expression plan give engine-native panel forecasts?

naive-last and OLS-trend written once in narwhals, run on pandas (flat, reset_index),
polars eager, polars lazy, duckdb relation. Compared with the hand-written native
plans of x7_native.py (pandas groupby (a), polars eager (b)) and checked for equality.

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x7_nw.py --grid dev
"""

import argparse
import json
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import duckdb  # noqa: E402
import narwhals as nw  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x7_native import (  # noqa: E402
    HS,
    pl_plan,
    run_pandas,
    to_triplet,
)

from common import bench, env_info, make_panel_fast, to_polars_long  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def nw_params(frame, method):
    """Per-series params via narwhals group_by().agg -- order-independent aggs only.

    'last' is expressed as: value at max(time) via a filter on a join (narwhals
    group_by does not guarantee order-dependent .last() across lazy backends).
    """
    df = nw.from_native(frame)
    if method == "last":
        tl = df.group_by("h0").agg(nw.col("time").max().alias("tlast"))
        p = df.join(tl, on="h0").filter(nw.col("time") == nw.col("tlast"))
        return p.select("h0", "tlast", nw.col("c0").alias("a"))
    if method == "ols":
        x = (nw.col("time") - nw.lit(pd.Timestamp("2000-01-01"))).dt.total_seconds() / 86400.0
        d = df.with_columns(x=x).with_columns(
            xy=nw.col("x") * nw.col("c0"), xx=nw.col("x") * nw.col("x"), one=nw.lit(1.0)
        )
        s = d.group_by("h0").agg(
            nw.col("one").sum().alias("n"),
            nw.col("x").sum().alias("sx"),
            nw.col("c0").sum().alias("sy"),
            nw.col("xy").sum().alias("sxy"),
            nw.col("xx").sum().alias("sxx"),
            nw.col("x").max().alias("xl"),
            nw.col("time").max().alias("tlast"),
        )
        b = (nw.col("n") * nw.col("sxy") - nw.col("sx") * nw.col("sy")) / (
            nw.col("n") * nw.col("sxx") - nw.col("sx") * nw.col("sx")
        )
        s = s.with_columns(b=b).with_columns(
            a=(nw.col("sy") - nw.col("b") * nw.col("sx")) / nw.col("n")
        )
        return s.select("h0", "tlast", "a", "b", "xl")
    raise ValueError(method)


def nw_forecast(frame, method):
    p = nw_params(frame, method)
    hor = pd.DataFrame({"h": HS.astype("int64")})
    native = nw.to_native(p)
    # cross join with horizon in the same backend
    if isinstance(native, pd.DataFrame):
        hor_n = hor
    elif isinstance(native, pl.DataFrame):
        hor_n = pl.from_pandas(hor)
    elif isinstance(native, pl.LazyFrame):
        hor_n = pl.from_pandas(hor).lazy()
    else:  # duckdb relation
        hor_n = duckdb.sql("SELECT range::BIGINT AS h FROM range(1, 15)")
    hor_nw = nw.from_native(hor_n)
    j = p.join(hor_nw, how="cross")
    if method == "last":
        val = nw.col("a")
    else:
        val = nw.col("a") + nw.col("b") * (nw.col("xl") + nw.col("h"))
    # duration from integer days: multiply 1-day duration column is not portable;
    # use the backend-agnostic route: keep h, compute timestamps natively after
    out = j.select("h0", "tlast", "h", val.alias("c0"))
    return out


def finish(out):
    """Materialize to polars and compute time = tlast + h days (for checking only)."""
    nat = nw.to_native(out)
    if isinstance(nat, pl.LazyFrame):
        nat = nat.collect()
    elif isinstance(nat, pd.DataFrame):
        nat = pl.from_pandas(nat)
    elif not isinstance(nat, pl.DataFrame):
        # pyarrow.Table: narwhals collects duckdb to arrow by default
        nat = pl.from_arrow(nat) if hasattr(nat, "schema") and not hasattr(
            nat, "pl"
        ) else nat.pl()
    return nat.select(
        "h0", (pl.col("tlast") + pl.duration(days=pl.col("h"))).alias("time"), "c0"
    )


def probe_total_seconds():
    """narwhals dt.total_seconds() on a 3-day duration, per backend."""
    d = pl.DataFrame(
        {"a": [pd.Timestamp("2000-01-04 00:00:05")], "b": [pd.Timestamp("2000-01-01")]}
    )
    out = {}
    for name, native in [
        ("polars", d),
        ("pandas", d.to_pandas()),
        ("duckdb", duckdb.from_arrow(d.to_arrow())),
    ]:
        r = nw.from_native(native).select(
            (nw.col("a") - nw.col("b")).dt.total_seconds().alias("s")
        )
        if hasattr(r, "collect"):
            r = r.collect()
        out[name] = float(r["s"][0])
    out["expected"] = 3 * 86400 + 5
    print("total_seconds probe:", out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    args = ap.parse_args()
    grid = [(1000, 100), (10000, 100)] if args.grid == "dev" else [(10000, 100), (100000, 100)]
    rows = []
    probe = probe_total_seconds()
    for n, T in grid:
        y = make_panel_fast(n, T)
        x0 = float((pd.Timestamp("2000-01-01") - pd.Timestamp(0)) / pd.Timedelta(days=1))
        ypl = to_polars_long(y)
        yflat = y.reset_index()
        # default connection: narwhals cross-join needs both relations on it
        duckdb.execute("SET threads=1")
        _arrow = ypl.to_arrow()
        rel = duckdb.from_arrow(_arrow)
        t0 = pl.lit(np.datetime64("2000-01-01", "ns"))
        for method in ["last", "ols"]:
            ref = to_triplet(run_pandas(y, method, x0))
            fns = {
                "a_pandas_groupby": lambda: run_pandas(y, method, x0),
                "b_polars_handwritten": lambda: pl_plan(ypl, method, t0),
                "nw_pandas_from_multiindex": lambda: nw_forecast(y.reset_index(), method),
                "nw_pandas_flat": lambda: nw_forecast(yflat, method),
                "nw_polars": lambda: nw_forecast(ypl, method),
                "nw_polars_lazy": lambda: nw_forecast(ypl.lazy(), method).collect(),
                "nw_duckdb": lambda: nw_forecast(rel, method).collect(),
            }
            res, chk, otype = {}, {}, {}
            for k, fn in fns.items():
                o = fn()
                otype[k] = type(nw.to_native(o) if k.startswith("nw") else o).__module__
                if k.startswith("nw"):
                    tri = to_triplet(finish(o))
                else:
                    tri = to_triplet(o)
                chk[k] = bool(
                    (tri[0] == ref[0]).all()
                    and (tri[1] == ref[1]).all()
                    and np.allclose(tri[2], ref[2], rtol=1e-7, atol=1e-9)
                )
                res[k] = bench(fn, repeat=5)
            row = {
                "n": n,
                "T": T,
                "method": method,
                "best_s": res,
                "check": chk,
                "native_output_module": otype,
                "ratio_vs_handwritten": {
                    "nw_pandas_flat/a": res["nw_pandas_flat"] / res["a_pandas_groupby"],
                    "nw_pandas_from_multiindex/a": res["nw_pandas_from_multiindex"]
                    / res["a_pandas_groupby"],
                    "nw_polars/b": res["nw_polars"] / res["b_polars_handwritten"],
                },
            }
            rows.append(row)
            print(n, T, method, {k: f"{v * 1e3:.2f}ms" for k, v in res.items()}, chk, otype, flush=True)
    with open(os.path.join(RES, f"x7_nw_{args.grid}.json"), "w") as f:
        json.dump({"env": env_info(), "total_seconds_probe": probe, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
