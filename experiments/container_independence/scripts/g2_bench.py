"""g2 benchmark: e2e fit+predict (and predict_interval), stock vs np_grouped ports.

Interleaved, same process: for each repeat, stock then g2 run back to back; min over
repeats. Also counts conversions / mtype checks / pandas constructions / public
re-entries for one fit+predict (``--counts``).

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/g2_bench.py [--counts] [--quick]
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ED)

from common import env_info, make_panel_fast  # noqa: E402
from protos.g2_forecasters import (  # noqa: E402
    G2NaiveForecaster,
    G2PolynomialTrendForecaster,
    G2RecursiveReducer,
)
from sklearn.linear_model import LinearRegression  # noqa: E402
from sktime.datatypes import convert_to  # noqa: E402
from sktime.forecasting.compose import make_reduction  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402
from sktime.forecasting.trend import PolynomialTrendForecaster  # noqa: E402

FH = list(range(1, 11))


def q(f):
    f.set_config(remember_data=True)
    return f


FCS = {
    "naive_last": (lambda: NaiveForecaster("last"), lambda: G2NaiveForecaster("last")),
    "naive_drift": (lambda: NaiveForecaster("drift"),
                    lambda: G2NaiveForecaster("drift")),
    "poly_d1": (lambda: PolynomialTrendForecaster(prediction_intervals=True),
                lambda: G2PolynomialTrendForecaster(prediction_intervals=True)),
    "reduce_lr_w10": (lambda: make_reduction(LinearRegression(), window_length=10,
                                             strategy="recursive"),
                      lambda: G2RecursiveReducer(LinearRegression(), 10)),
}
PI = {"naive_last", "poly_d1"}


def to_arrow(y):
    import pyarrow as pa

    flat = y.reset_index()
    flat.columns = [f"__index__{c}" if i < y.index.nlevels else c
                    for i, c in enumerate(flat.columns)]
    return pa.Table.from_pandas(flat, preserve_index=False)


def arrow_user_stock(mk, tab, do_pi=False):
    """Stock sktime has no pyarrow mtype: the user converts in and out."""
    import pyarrow as pa

    df = tab.to_pandas()
    idx = [c for c in df.columns if c.startswith("__index__")]
    df = df.set_index(idx)
    df.index.names = [c[9:] for c in idx]
    f = q(mk()).fit(df, fh=FH)
    out = f.predict().reset_index()
    out.columns = [f"__index__{c}" if c in df.index.names else c for c in out.columns]
    res = pa.Table.from_pandas(out, preserve_index=False)
    if do_pi:
        f.predict_interval(coverage=[0.9])
    return res


def run(kind, mk_stock, mk_g2, yin, do_pi):
    if kind == "stock" and type(yin).__module__.startswith("pyarrow"):
        return arrow_user_stock(mk_stock, yin, do_pi)
    f = q((mk_stock if kind == "stock" else mk_g2)()).fit(yin, fh=FH)
    out = f.predict()
    if do_pi:
        f.predict_interval(coverage=[0.9])
    return out


def timed(fn):
    gc.collect()
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def bench(sizes, repeats, stock_max, out_rows):
    for n_s, n_t in sizes:
        y = make_panel_fast(n_s, n_t)
        ins = {"pandas": y, "polars": convert_to(y, "polars_panel"),
               "pyarrow": to_arrow(y)}
        for fname, (mk_s, mk_g) in FCS.items():
            for do_pi in ([False, True] if fname in PI else [False]):
                for iname, yin in ins.items():
                    rec = dict(n_series=n_s, n_time=n_t, fc=fname, input=iname,
                               with_pi=do_pi)
                    with_stock = n_s <= stock_max
                    reps = repeats if n_s < 1000 else max(1, repeats - 1)
                    ts, tg = [], []
                    run("g2", mk_s, mk_g, yin, do_pi)  # warmup
                    if with_stock and n_s < 1000:
                        run("stock", mk_s, mk_g, yin, do_pi)
                    for _ in range(reps):
                        if with_stock:
                            ts.append(timed(lambda: run("stock", mk_s, mk_g, yin,
                                                        do_pi)))
                        tg.append(timed(lambda: run("g2", mk_s, mk_g, yin, do_pi)))
                    rec["g2_s"] = min(tg)
                    if with_stock:
                        rec["stock_s"] = min(ts)
                        rec["speedup"] = rec["stock_s"] / rec["g2_s"]
                    out_rows.append(rec)
                    print(json.dumps(rec), flush=True)


def counts(n_s=100, n_t=100):
    """Count conversions, mtype checks, pandas constructions, public re-entries."""
    import sktime.datatypes._convert as cv
    from sktime.datatypes._base import BaseDatatype
    from sktime.forecasting.base import BaseForecaster

    C = Counter()
    orig = dict(cv.convert_dict)

    def wrap_conv(key, f):
        def g(obj, store=None):
            C["convert"] += 1
            C[f"convert:{key[0]}->{key[1]}"] += 1
            return f(obj, store=store)
        return g

    for k, f in orig.items():
        if k[0] != k[1]:
            cv.convert_dict[k] = wrap_conv(k, f)
        else:
            cv.convert_dict[k] = (lambda f_: (lambda obj, store=None: (
                C.__setitem__("convert_identity", C["convert_identity"] + 1),
                f_(obj, store=store))[1]))(f)
    ocheck = BaseDatatype.check

    def check(self, obj, *a, **k):
        C["mtype_check"] += 1
        return ocheck(self, obj, *a, **k)

    BaseDatatype.check = check
    ofit, opred = BaseForecaster.fit, BaseForecaster.predict

    def fit(self, *a, **k):
        C["public_fit"] += 1
        return ofit(self, *a, **k)

    def pred(self, *a, **k):
        C["public_predict"] += 1
        return opred(self, *a, **k)

    BaseForecaster.fit, BaseForecaster.predict = fit, pred
    odf, osr = pd.DataFrame.__init__, pd.Series.__init__

    def dfi(self, *a, **k):
        C["pd.DataFrame()"] += 1
        return odf(self, *a, **k)

    def sri(self, *a, **k):
        C["pd.Series()"] += 1
        return osr(self, *a, **k)

    y = make_panel_fast(n_s, n_t)
    ins = {"pandas": y, "polars": convert_to(y, "polars_panel"), "pyarrow": to_arrow(y)}
    rows = []
    for fname, (mk_s, mk_g) in FCS.items():
        for iname, yin in ins.items():
            for kind in ("stock", "g2"):
                C.clear()
                pd.DataFrame.__init__, pd.Series.__init__ = dfi, sri
                try:
                    run(kind, mk_s, mk_g, yin, False)
                finally:
                    pd.DataFrame.__init__, pd.Series.__init__ = odf, osr
                top = {k: v for k, v in C.items() if not k.startswith("convert:")}
                conv = {k[8:]: v for k, v in C.items() if k.startswith("convert:")}
                rec = dict(fc=fname, input=iname, kind=kind, n_series=n_s, n_time=n_t,
                           **top, conversions_by_pair=conv)
                rows.append(rec)
                print(json.dumps(rec), flush=True)
    BaseDatatype.check = ocheck
    BaseForecaster.fit, BaseForecaster.predict = ofit, opred
    cv.convert_dict.update(orig)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    res = dict(env=env_info())
    if a.counts:
        res["counts"] = counts()
        path = os.path.join(ED, "results", "g2_counts.json")
    else:
        sizes = [(100, 100), (1000, 100)] if a.quick else [(100, 100), (1000, 100),
                                                           (10000, 50)]
        rows = []
        bench(sizes, a.repeats, stock_max=1000, out_rows=rows)
        res["bench"] = rows
        path = os.path.join(ED, "results", "g2_bench.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print("->", path)


if __name__ == "__main__":
    main()
