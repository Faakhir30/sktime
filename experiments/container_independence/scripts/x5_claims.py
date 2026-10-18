"""x5: verify the contested narwhals claims C1 / C2, collect silent-bug repros,
dask flakiness, and LOC.

C1  "Polars lazy collects to Polars by default, but Narwhals' collect also sends
     Dask to pandas and DuckDB to Arrow."  -> true? overridable? avoidable?
C2  "No MultiIndex ... reset_index is a copy ... index came back as RangeIndex,
     level names gone ... current code uses groupby(level=...)."

Run (both envs):
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x5_claims.py [--grid dev|large]
"""

import argparse
import ast
import datetime as dt
import io
import json
import time
import tokenize
import warnings
from pathlib import Path

import narwhals as nw
import numpy as np
import pandas as pd

from common import bench, env_info, make_panel, make_panel_fast
from protos.x5_backends import long_from_mi, make_native

warnings.filterwarnings("ignore")
ED = Path(__file__).resolve().parents[1]
WT = ED.parents[1]
HAS_IBIS = True
try:
    import ibis  # noqa: F401
except ImportError:
    HAS_IBIS = False


def tname(x):
    return type(x).__module__.split(".")[0] + "." + type(x).__name__


# ----------------------------------------------------------------------- C1
def claim_c1(n_series, n_time):
    res = {}
    y = make_panel_fast(n_series, n_time)
    lng = long_from_mi(y)
    lazy_b = ["polars_lazy", "dask", "duckdb"] + (["ibis"] if HAS_IBIS else [])
    for b in lazy_b:
        r = {}
        lf = nw.from_native(make_native(lng, b))
        r["to_native_without_collect"] = tname(lf.to_native())
        r["collect_default"] = tname(lf.collect().to_native())
        for tgt in ["polars", "pandas", "pyarrow"]:
            try:
                r[f"collect({tgt})"] = tname(lf.collect(tgt).to_native())
            except Exception as e:  # noqa: BLE001
                r[f"collect({tgt})"] = f"ERR {type(e).__name__}: {e}"[:120]
        # re-lazify into the original backend
        impl = lf.implementation
        try:
            back = lf.collect().lazy(backend=impl if b != "polars_lazy" else "polars")
            r["collect().lazy(backend=orig)"] = tname(back.to_native())
        except Exception as e:  # noqa: BLE001
            r["collect().lazy(backend=orig)"] = f"ERR {type(e).__name__}: {e}"[:120]
        # timings (relative, same process): default collect vs alternatives
        tt = {}
        tt["collect_default"] = bench(lambda: lf.collect(), repeat=3)
        for tgt in ["polars", "pandas", "pyarrow"]:
            try:
                tt[f"collect({tgt})"] = bench(lambda: lf.collect(tgt), repeat=3)
            except Exception:  # noqa: BLE001
                pass
        if b in ("dask", "duckdb"):
            tt["collect+relazy"] = bench(
                lambda: lf.collect().lazy(backend=impl), repeat=3
            )
        r["time_s"] = tt
        res[b] = r
        print("C1", b, r)

    # does the panel-native forecaster keep lazy inputs lazy?
    from protos.x5_nw_forecasters import NwPanelNaive

    keep = {}
    for b in lazy_b:
        nat = make_native(lng, b)
        t0 = time.perf_counter()
        m = NwPanelNaive("last", sp=7).fit(nat, keys=["h0"], time="time")
        out = m.predict(list(range(1, 13)))
        t_plan = time.perf_counter() - t0
        t0 = time.perf_counter()
        m2 = NwPanelNaive("last", sp=7).fit(nat, keys=["h0"], time="time", freq="D")
        out2 = m2.predict(list(range(1, 13)))
        t_plan_nofreq = time.perf_counter() - t0
        t0 = time.perf_counter()
        nw.from_native(out2).collect()
        t_exec = time.perf_counter() - t0
        keep[b] = {
            "in": tname(nat),
            "out": tname(out),
            "same_type": type(out) is type(nat),
            "fit+predict_s(incl freq inference queries)": t_plan,
            "fit+predict_s(freq given: plan only)": t_plan_nofreq,
            "collect_s": t_exec,
        }
        print("C1 lazy-preserved", b, keep[b])
    res["forecaster_keeps_lazy"] = keep
    return res


# ----------------------------------------------------------------------- C2
def claim_c2(grid_sizes):
    res = {}
    y = make_panel(4, 5, n_levels=2)
    f = nw.from_native(y)
    res["from_native(MI).columns"] = list(f.columns)
    try:
        f.group_by("h0").agg(nw.col("c0").mean())
        res["group_by(level name) on MI frame"] = "works"
    except Exception as e:  # noqa: BLE001
        res["group_by(level name) on MI frame"] = f"ERR {type(e).__name__}"
    mr = nw.to_native(nw.maybe_reset_index(f))
    res["maybe_reset_index"] = {
        "index": type(mr.index).__name__,
        "columns": list(mr.columns),
        "level_values_kept": all(n in mr.columns for n in y.index.names),
    }
    r = y.reset_index()
    res["native reset_index"] = {"columns": list(r.columns)}
    yn = y.copy()
    yn.index = yn.index.set_names([None, None, None])
    res["native reset_index, names None"] = list(yn.reset_index().columns)
    g = nw.from_native(r).group_by("h0", "h1").agg(nw.col("c0").mean()).to_native()
    res["after nw group_by: native index"] = type(g.index).__name__
    w = nw.from_native(r).with_columns(z=nw.col("c0") * 2).to_native()
    res["after nw with_columns: native index"] = type(w.index).__name__
    back = w.set_index(["h0", "h1", "time"])
    res["set_index(names) restores names"] = list(back.index.names) == list(
        y.index.names
    )
    # dtype of key / time columns after reset_index
    res["dtypes after reset_index"] = {c: str(t) for c, t in r.dtypes.items()}
    yp = make_panel(3, 4, n_levels=1, time_kind="period", freq="M")
    rp = yp.reset_index()
    res["period level after reset_index"] = str(rp["time"].dtype)
    try:
        res["narwhals dtype of period column"] = str(
            nw.from_native(rp).collect_schema()["time"]
        )
    except Exception as e:  # noqa: BLE001
        res["narwhals dtype of period column"] = f"ERR {type(e).__name__}"
    try:
        import polars as pl

        pl.from_pandas(rp)
        res["polars.from_pandas(period column)"] = "works"
    except Exception as e:  # noqa: BLE001
        res["polars.from_pandas(period column)"] = f"ERR {type(e).__name__}"

    # cost of reset_index vs current prepare_VectorizedDF / groupby(level)
    from sktime.datatypes._vectorize import prepare_VectorizedDF

    cost = []
    for n_series, n_time in grid_sizes:
        X = make_panel_fast(n_series, n_time)
        mem_in = X.memory_usage(deep=True, index=True).sum()
        mem_vals = X.memory_usage(deep=True, index=False).sum()
        R = X.reset_index()
        mem_out = R.memory_usage(deep=True, index=True).sum()
        t_reset = bench(lambda: X.reset_index(), repeat=5)
        t_prep = bench(
            lambda: prepare_VectorizedDF(X, iterate_as="Series", is_scitype="Panel"),
            repeat=3,
        )

        def it_level():
            return sum(1 for _ in X.groupby(level=0, sort=False))

        def it_col():
            return sum(1 for _ in R.groupby("h0", sort=False))

        t_it_level = bench(it_level, repeat=3)
        t_it_col = bench(it_col, repeat=3)
        # value-column buffer shared after reset_index? (pandas 3 CoW)
        shared = np.shares_memory(R["c0"].to_numpy(), X["c0"].to_numpy())
        row = {
            "n_series": n_series,
            "n_time": n_time,
            "frame_bytes_total": int(mem_in),
            "frame_bytes_values": int(mem_vals),
            "reset_index_result_bytes": int(mem_out),
            "reset_index_new_bytes": int(mem_out - mem_vals) if shared else int(mem_out),
            "value_buffer_shared": bool(shared),
            "t_reset_index_s": t_reset,
            "t_prepare_VectorizedDF_s": t_prep,
            "reset/prepare": t_reset / t_prep,
            "t_iter_groupby_level_s": t_it_level,
            "t_iter_groupby_column_s": t_it_col,
        }
        cost.append(row)
        print("C2 cost", row)
    res["cost"] = cost
    return res


# ----------------------------------------------------------- silent bugs
def silent_bugs():
    import duckdb
    import polars as pl
    import pyarrow as pa

    out = {}
    data = {
        "g": ["a", "a", "a", "b", "b"],
        "t": [1, 2, 3, 1, 2],
        "v": [10.0, 30.0, 20.0, 5.0, 1.0],
        "r": [3, 2, 1, 2, 1],
    }
    makers = {"pandas": pd.DataFrame, "polars": pl.DataFrame, "pyarrow": pa.table}
    row = {}
    for b, mk in makers.items():
        f = nw.from_native(mk(data))
        try:
            got = (
                f.group_by("g")
                .agg(nw.when(nw.col("r") == 1).then(nw.col("v")).max())
                .sort("g")["v"]
                .to_list()
            )
        except Exception as e:  # noqa: BLE001
            got = f"ERR {type(e).__name__}"
        row[b] = {"got": got, "expected": [20.0, 1.0], "correct": got == [20.0, 1.0]}
    out["group_by.agg(when(r==1).then(v).max())"] = row

    d2 = {"g": ["a", "a", "a", "b", "b"], "t": [3, 1, 2, 2, 1], "v": [20.0, 10.0, 30.0, 1.0, 5.0]}
    row = {}
    for b, mk in makers.items():
        f = nw.from_native(mk(d2))
        try:
            got = f.with_columns(l=nw.col("v").last().over("g", order_by="t"))["l"].to_list()
        except Exception as e:  # noqa: BLE001
            got = f"ERR {type(e).__name__}"
        exp = [20.0, 20.0, 20.0, 1.0, 1.0]
        row[b] = {"got": got, "expected": exp, "correct": got == exp}
    out["last().over(g, order_by=t)"] = row

    d3 = {"t": [dt.datetime(2000, 1, 1), dt.datetime(2000, 1, 3), dt.datetime(2000, 1, 1, 0, 0, 5)]}
    row = {}
    lazies = {
        "polars_lazy": nw.from_native(pl.DataFrame(d3).lazy()),
        "duckdb": nw.from_native(duckdb.from_arrow(pa.table(d3))),
        "pandas": nw.from_native(pd.DataFrame(d3)),
    }
    for b, lf in lazies.items():
        q = lf.with_columns(
            d=(nw.col("t") - nw.lit(dt.datetime(2000, 1, 1))).dt.total_seconds()
        )
        q = q.collect() if isinstance(q, nw.LazyFrame) else q
        got = [int(x) for x in q["d"].to_list()]
        row[b] = {"got": got, "expected": [0, 172800, 5], "correct": got == [0, 172800, 5]}
    out["(t - t0).dt.total_seconds()"] = row

    lng = long_from_mi(make_panel(3, 4, n_levels=1))
    row = {}
    for b in ["polars_lazy", "duckdb", "dask"]:
        lf = nw.from_native(make_native(lng, b))
        for arg in ["time", ["time"]]:
            try:
                lf.unique(["h0"], keep="first", order_by=arg).collect()
                row[f"{b} order_by={arg!r}"] = "works"
            except Exception as e:  # noqa: BLE001
                row[f"{b} order_by={arg!r}"] = f"ERR {type(e).__name__}: {e}"[:90]
    out["LazyFrame.unique(order_by=<str>)"] = row
    for k, v in out.items():
        print("BUG", k, v)
    return out


# ------------------------------------------------------------ dask flaky
def dask_flaky(n_trials=20):
    from protos.x5_nw_forecasters import NwPanelNaive

    y = make_panel(20, 60, n_levels=1)
    lng = long_from_mi(y)
    out = {}
    for kw in [dict(strategy="mean", window_length=10), dict(strategy="last"), dict(strategy="drift")]:
        fails, msgs = 0, set()
        for _ in range(n_trials):
            m = NwPanelNaive(**kw).fit(make_native(lng, "dask"), keys=["h0"], time="time", freq="D")
            try:
                m.predict(list(range(1, 13))).compute()
            except Exception as e:  # noqa: BLE001
                fails += 1
                msgs.add(f"{type(e).__name__}: {e}"[:90])
        out[json.dumps(kw)] = {"trials": n_trials, "failures": fails, "msgs": sorted(msgs)}
        print("DASK", kw, out[json.dumps(kw)])
    return out


# --------------------------------------------------------------------- LOC
def sloc(path, start=None, end=None):
    """Lines of code without blank lines, comments and docstrings."""
    src = Path(path).read_text()
    tree = ast.parse(src)
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(
                getattr(body[0], "value", None), ast.Constant
            ) and isinstance(body[0].value.value, str):
                doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    code_lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        for ln in range(tok.start[0], tok.end[0] + 1):
            code_lines.add(ln)
    lines = code_lines - doc_lines
    if start is not None:
        lines = {ln for ln in lines if start <= ln <= end}
    return len(lines)


def class_span(path, cls):
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return node.lineno, node.end_lineno
    raise KeyError(cls)


def loc():
    out = {}
    out["sktime/datatypes/_vectorize.py (whole)"] = sloc(WT / "sktime/datatypes/_vectorize.py")
    out["protos/x5_narwhals.py (whole)"] = sloc(ED / "protos/x5_narwhals.py")
    out["protos/x5_nw_forecasters.py (whole)"] = sloc(ED / "protos/x5_nw_forecasters.py")
    nf = WT / "sktime/forecasting/naive/_naive.py"
    a, b = class_span(nf, "NaiveForecaster")
    out["sktime NaiveForecaster class (excl. base classes)"] = sloc(nf, a, b)
    bw = WT / "sktime/forecasting/base/_sktime.py"
    try:
        a, b = class_span(bw, "_BaseWindowForecaster")
        out["sktime _BaseWindowForecaster class"] = sloc(bw, a, b)
    except Exception:  # noqa: BLE001
        pass
    print("LOC", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    args = ap.parse_args()
    sizes = [(1_000, 100), (10_000, 50)] if args.grid == "dev" else [(10_000, 100), (50_000, 50)]
    out = {"env": env_info()}
    out["C1"] = claim_c1(1_000, 100)
    out["C2"] = claim_c2(sizes)
    out["silent_bugs"] = silent_bugs()
    out["dask_flaky"] = dask_flaky()
    out["loc"] = loc()
    tag = "pd3" if pd.__version__.startswith("3") else "pd2"
    p = ED / "results" / f"x5_claims_{tag}_{args.grid}.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
