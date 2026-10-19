"""g4: probabilistic outputs through each prototype (x4 adapter, x5 narwhals, x6 hub,
x7 hook) and through the stock second vectorization path (BaseProbaForecaster).

For each route: call predict_interval(coverage=[.8,.9]), predict_quantiles(alpha=[.1,.5,.9])
and predict_var on Panel (and Hierarchical where the route supports it), record
(a) output type, (b) equality with stock pandas, (c) time (best of N, same process).

Usage (own venv with statsmodels + skpro, see report):
  PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
    $G4_PY -W ignore experiments/container_independence/scripts/g4_proto_probe.py [--n 200]
"""

import argparse
import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.join(HERE, "..")
sys.path.insert(0, ED)
sys.path.insert(0, os.path.join(ED, "protos"))

import sktime.utils.warnings as _sw  # noqa: E402

_sw.warn = lambda *a, **k: None
from common import bench, make_panel, make_panel_fast  # noqa: E402

from sktime.datatypes import convert_to  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402

FH = [1, 2, 3]
COV = [0.8, 0.9]
ALPHA = [0.1, 0.5, 0.9]
OUT = {}


def rec(key, **kw):
    OUT[key] = kw
    print(key, json.dumps(kw, default=str))


def to_pd(obj):
    """Normalise a proba output to pandas for comparison."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pl.DataFrame):
        return convert_to(obj, "pd-multiindex")
    raise TypeError(type(obj))


def eq(a, b):
    try:
        pd.testing.assert_frame_equal(to_pd(a), to_pd(b), check_freq=False)
        return True
    except Exception as e:  # noqa: BLE001
        return f"NOT EQUAL: {str(e)[:200]}"


def naive():
    return NaiveForecaster(strategy="last").set_config(
        warnings="off", remember_data=True
    )


def run_methods(f):
    t = {}
    res = {}
    for name, fn in [
        ("predict_interval", lambda: f.predict_interval(coverage=COV)),
        ("predict_quantiles", lambda: f.predict_quantiles(alpha=ALPHA)),
        ("predict_var", lambda: f.predict_var()),
    ]:
        try:
            res[name] = fn()
            t[name] = bench(fn, repeat=3, warmup=0)
        except Exception as e:  # noqa: BLE001
            res[name] = e
    return res, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    n = args.n
    y_pd = make_panel_fast(n, 60)
    y_pl = convert_to(y_pd, "polars_panel")
    y_h = make_panel(n_series=n // 4 if n >= 8 else 2, n_time=60, n_levels=3)

    # ---------------- stock, pandas in -------------------------------------
    f0 = naive().fit(y_pd, fh=FH)
    ref, t_ref = run_methods(f0)
    rec(
        "stock_pd",
        types={k: type(v).__name__ for k, v in ref.items()},
        cols={k: [str(c) for c in v.columns[:4]] for k, v in ref.items()},
        sec=t_ref,
    )
    fh_ = naive().fit(y_h, fh=FH)
    ref_h, t_h = run_methods(fh_)
    rec(
        "stock_hier_pd",
        types={k: type(v).__name__ for k, v in ref_h.items()},
        index_names={k: list(v.index.names) for k, v in ref_h.items()},
        sec=t_h,
    )

    # ---------------- stock, polars in -------------------------------------
    f1 = naive().fit(y_pl, fh=FH)
    r1, t1 = run_methods(f1)
    rec(
        "stock_pl",
        types={k: type(v).__name__ for k, v in r1.items()},
        equal={k: eq(v, ref[k]) for k, v in r1.items()},
        predict_type=type(f1.predict()).__name__,
        sec=t1,
    )

    # ---------------- x4 adapter: native polars iteration ------------------
    from x4_instrument import patched_base

    try:
        with patched_base("native", cutoff=True):
            f4 = naive().fit(y_pl, fh=FH)
            r4, t4 = run_methods(f4)
        rec(
            "x4_native_pl",
            types={k: type(v).__name__ for k, v in r4.items()},
            equal={
                k: (eq(v, ref[k]) if not isinstance(v, Exception) else repr(v)[:200])
                for k, v in r4.items()
            },
            sec=t4,
            note="per-instance leaves are pandas-inner; reconstruct dispatches on "
            "OUTPUTS -> PandasAdapter -> pd.concat(keys) keeps column MultiIndex",
        )
    except Exception as e:  # noqa: BLE001
        rec("x4_native_pl", error=repr(e)[:300], tb=traceback.format_exc()[-600:])

    # x4 with a polars-inner leaf: can a polars per-instance proba output be
    # concatenated? (simulate: per-instance interval as polars with flat names)
    from x4_adapter import get_adapter

    parts_pd = [
        f.predict_interval(coverage=COV) for f in f0.forecasters_.iloc[:5, 0]
    ]
    try:
        pl.from_pandas(parts_pd[0].reset_index())
        r = "ok"
    except Exception as e:  # noqa: BLE001
        r = repr(e)[:200]
    rec("polars_from_pandas_MIcols", result=r)
    try:
        ad = get_adapter(pl.DataFrame({"a": [1.0]}))
        flat = []
        for p in parts_pd:
            q = p.copy()
            q.columns = ["|".join(map(str, c)) for c in q.columns]
            flat.append(
                pl.from_pandas(q.reset_index().rename(columns={"time": "__index__time"}))
            )
        out = ad.concat_rows_with_keys(
            flat, list(f0.forecasters_.index[:5]), ["h0"]
        )
        rec("x4_polars_adapter_flatcols", ok=True, columns=out.columns, rows=out.height)
    except Exception as e:  # noqa: BLE001
        rec("x4_polars_adapter_flatcols", ok=False, error=repr(e)[:300])

    # ---------------- x5 narwhals reconstruct ------------------------------
    import narwhals as nw
    from x5_narwhals import pandas_to_long, prepare_nw_vectorized

    try:
        nw.from_native(parts_pd[0].reset_index())
        r = "ok"
    except Exception as e:  # noqa: BLE001
        r = repr(e)[:200]
    rec("x5_nw_from_native_MIcols", result=r)
    try:
        flat, meta = pandas_to_long(parts_pd[0])
        rec(
            "x5_pandas_to_long_MIcols",
            flat_columns=list(flat.columns),
            restorable_columns=[str(c) for c in meta.columns],
            note="tuple column labels are renamed __colj__; the MultiIndex is kept "
            "only in PandasIndexMeta (pandas-only round trip)",
        )
    except Exception as e:  # noqa: BLE001
        rec("x5_pandas_to_long_MIcols", error=repr(e)[:300])
    try:
        schema, frame = prepare_nw_vectorized(y_pl)
        sub = [pl.from_pandas(p.reset_index(drop=True)) for p in parts_pd]
        rec("x5_reconstruct_pl_from_MI", result="unexpected ok")
    except Exception as e:  # noqa: BLE001
        rec("x5_reconstruct_pl_from_MI", error=repr(e)[:300])

    # ---------------- x6 hub ----------------------------------------------
    from x6_hub import from_pandas, to_pandas, to_polars

    hub = from_pandas(y_pd.iloc[: 60 * 5])
    outs = [
        (p.index.to_numpy(), p.to_numpy()) for p in parts_pd
    ]
    try:
        h2 = hub.from_outputs(outs, columns=list(parts_pd[0].columns))
        try:
            o_pl = to_polars(h2)
            pl_res = f"ok cols={o_pl.columns[:4]}"
        except Exception as e:  # noqa: BLE001
            pl_res = repr(e)[:200]
        try:
            o_pd = to_pandas(h2)
            pd_res = f"ok cols={[str(c) for c in o_pd.columns[:4]]} " + str(
                isinstance(o_pd.columns, pd.MultiIndex)
            )
        except Exception as e:  # noqa: BLE001
            pd_res = repr(e)[:200]
        rec("x6_from_outputs_tuplecols", to_polars=pl_res, to_pandas=pd_res)
    except Exception as e:  # noqa: BLE001
        rec("x6_from_outputs_tuplecols", error=repr(e)[:300])

    # ---------------- x7 hook ----------------------------------------------
    from x7_hook import NaivePanelNative

    for lab, y in [("pd", y_pd), ("pl", y_pl)]:
        def x7_fit_then_interval(y=y):
            f = NaivePanelNative().set_config(warnings="off").fit(y, fh=FH)
            f.predict()
            return f.predict_interval(coverage=COV)

        try:
            r7 = x7_fit_then_interval()
            t_native_fp = bench(
                lambda y=y: NaivePanelNative().set_config(warnings="off").fit(y, fh=FH).predict(),
                repeat=3,
            )
            t_mat = bench(x7_fit_then_interval, repeat=2, warmup=0)
            t_stock = bench(
                lambda y=y: naive().fit(y, fh=FH).predict_interval(coverage=COV),
                repeat=2,
                warmup=0,
            )
            rec(
                f"x7_hook_{lab}",
                type=type(r7).__name__,
                equal=eq(r7, ref["predict_interval"]),
                sec_native_fit_predict=t_native_fp,
                sec_native_fit_predict_then_interval=t_mat,
                sec_stock_fit_interval=t_stock,
            )
        except Exception as e:  # noqa: BLE001
            rec(f"x7_hook_{lab}", error=repr(e)[:300], tb=traceback.format_exc()[-500:])

    # ---------------- second vectorization path: predict_proba ------------
    try:
        f0.predict_proba()
        rec("stock_panel_predict_proba", result="ok")
    except Exception as e:  # noqa: BLE001
        rec("stock_panel_predict_proba", error=repr(e)[:200])

    from sktime.forecasting.base._base_proba import BaseProbaForecaster

    class NaiveProba(BaseProbaForecaster, NaiveForecaster):
        pass

    try:
        fp = NaiveProba(strategy="last").set_config(warnings="off").fit(y_pd, fh=FH)
        d = fp.predict_proba()
        rec("BaseProbaForecaster_panel_predict_proba", result=type(d).__name__)
    except Exception as e:  # noqa: BLE001
        rec(
            "BaseProbaForecaster_panel_predict_proba",
            error=repr(e)[:300],
            tb=traceback.format_exc()[-700:],
        )

    with open(os.path.join(ED, "results", f"g4_proto_probe_n{n}.json"), "w") as fo:
        json.dump(OUT, fo, indent=1, default=str)


if __name__ == "__main__":
    main()
