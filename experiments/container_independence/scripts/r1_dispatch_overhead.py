"""r1: per-call overhead of the dispatch styles used by prior-art libraries.

Styles compared on the SAME small frame (one instance of a panel, i.e. what a
per-instance loop in VectorizedDF would touch), interleaved in one process:

  native        : call the backend method directly (df.height / len(df), ...)
  dict-dispatch : sklearn set_output style, adapter = REG[type(X).__module__ root]
  singledispatch: skrub style, functools.singledispatch on the container type
  skrub         : real skrub._dataframe functions (if skrub importable)
  narwhals      : nw.from_native(X) then the narwhals method, then .to_native()
  sklearn       : sklearn's own validators that route dataframes via narwhals
  sktime        : sktime check_is_mtype on the same frame (reference point for
                  what the current framework pays per call; if importable)

Reports microseconds per call and ratio vs native, and the implied overhead for
a 10_000-instance loop (overhead_us * 1e4 / 1e3 = ms).

Run:
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/r1_dispatch_overhead.py --grid dev
  PYTHONPATH=.:experiments/container_independence $R1_PY -W ignore \
      experiments/container_independence/scripts/r1_dispatch_overhead.py --grid dev --tag skrubvenv
"""

import argparse
import json
import os
import warnings
from functools import singledispatch

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from common import bench, make_panel_fast  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="dev", choices=["dev", "large"])
ap.add_argument("--tag", default="exp")
ap.add_argument("--number", type=int, default=2000)
ap.add_argument("--repeat", type=int, default=5)
args = ap.parse_args()
N_TIME = 100 if args.grid == "dev" else 500

import narwhals.stable.v2 as nw  # noqa: E402

try:
    from skrub import _dataframe as sbd
except Exception:  # noqa: BLE001
    sbd = None
from sklearn.utils.validation import _num_samples, check_array  # noqa: E402

try:
    from sktime.datatypes import check_is_mtype
except Exception:  # noqa: BLE001
    check_is_mtype = None

pdf = make_panel_fast(1, N_TIME, n_cols=1).droplevel(0)  # single series, DatetimeIndex
pdf_flat = pdf.reset_index()
pldf = pl.from_pandas(pdf_flat)


# ---- sklearn-style adapter registry keyed on module root
class _PdAd:
    def n_rows(self, X):
        return len(X)

    def col_mean(self, X, c):
        return X[c].mean()

    def to_numpy(self, X):
        return X.to_numpy()


class _PlAd:
    def n_rows(self, X):
        return X.height

    def col_mean(self, X, c):
        return X[c].mean()

    def to_numpy(self, X):
        return X.to_numpy()


REG = {"pandas": _PdAd(), "polars": _PlAd()}


def adapter(X):
    return REG[type(X).__module__.split(".", 1)[0]]


# ---- skrub-style singledispatch
@singledispatch
def sd_n_rows(X):
    raise TypeError


@sd_n_rows.register(pd.DataFrame)
def _(X):
    return len(X)


@sd_n_rows.register(pl.DataFrame)
def _(X):
    return X.height


@singledispatch
def sd_col_mean(X, c):
    raise TypeError


@sd_col_mean.register(pd.DataFrame)
@sd_col_mean.register(pl.DataFrame)
def _(X, c):
    return X[c].mean()


def cases(X, label):
    native_n = (lambda: len(X)) if label == "pandas" else (lambda: X.height)
    c = {
        ("n_rows", "native"): native_n,
        ("n_rows", "dict-dispatch"): lambda: adapter(X).n_rows(X),
        ("n_rows", "singledispatch"): lambda: sd_n_rows(X),
        ("n_rows", "narwhals"): lambda: nw.from_native(X, eager_only=True).shape[0],
        ("n_rows", "sklearn _num_samples"): lambda: _num_samples(X),
        ("col_mean", "native"): lambda: X["c0"].mean(),
        ("col_mean", "dict-dispatch"): lambda: adapter(X).col_mean(X, "c0"),
        ("col_mean", "singledispatch"): lambda: sd_col_mean(X, "c0"),
        ("col_mean", "narwhals"): lambda: nw.from_native(X, eager_only=True)["c0"].mean(),
        ("to_numpy", "native"): lambda: X.to_numpy(),
        ("to_numpy", "dict-dispatch"): lambda: adapter(X).to_numpy(X),
        ("to_numpy", "narwhals"): lambda: nw.from_native(X, eager_only=True).to_numpy(),
        ("to_numpy", "sklearn check_array"): lambda: check_array(X[["c0"]]),
    }
    # decomposition of the narwhals cost: wrap only, wrap via main namespace,
    # and the method on an already-wrapped frame (wrap hoisted out of the loop)
    import narwhals as nw_main

    W = nw.from_native(X, eager_only=True)
    c[("nw-decomp", "from_native only (stable.v2)")] = lambda: nw.from_native(X, eager_only=True)
    c[("nw-decomp", "from_native only (main ns)")] = lambda: nw_main.from_native(X, eager_only=True)
    c[("nw-decomp", "pre-wrapped .shape[0]")] = lambda: W.shape[0]
    c[("nw-decomp", "pre-wrapped ['c0'].mean()")] = lambda: W["c0"].mean()
    c[("nw-decomp", "pre-wrapped .to_native()")] = lambda: W.to_native()
    if sbd is not None:
        c[("n_rows", "skrub")] = lambda: sbd.shape(X)[0]
        c[("col_mean", "skrub")] = lambda: sbd.mean(sbd.col(X, "c0"))
        c[("to_numpy", "skrub")] = lambda: sbd.to_numpy(X)
    if check_is_mtype is not None:
        mt = "pd.DataFrame" if label == "pandas" else "pl.DataFrame"
        XX = pdf if label == "pandas" else X
        c[("check", "sktime check_is_mtype")] = lambda: check_is_mtype(
            XX, mt, return_metadata=False
        )
        c[("check", "narwhals from_native+schema")] = lambda: nw.from_native(
            X, eager_only=True
        ).schema
    return c


rows = []
for label, X in [("pandas", pdf_flat), ("polars", pldf)]:
    cs = cases(X, label)
    # interleave: run every case once per outer repeat, keep min per case
    best = {k: float("inf") for k in cs}
    for _ in range(args.repeat):
        for k, fn in cs.items():
            t = bench(fn, repeat=1, number=args.number, warmup=1 if _ == 0 else 0)
            best[k] = min(best[k], t)
    for (op, style), t in best.items():
        rows.append(dict(backend=label, op=op, style=style, us=t * 1e6))

# ratios vs native for the same backend/op
for r in rows:
    nat = [x for x in rows if x["backend"] == r["backend"] and x["op"] == r["op"] and x["style"] == "native"]
    if nat:
        r["ratio_vs_native"] = round(r["us"] / nat[0]["us"], 2)
        r["overhead_us"] = round(r["us"] - nat[0]["us"], 2)
        r["overhead_ms_per_10k_instances"] = round((r["us"] - nat[0]["us"]) * 1e4 / 1e3, 2)
    r["us"] = round(r["us"], 2)

out = {
    "versions": {
        "pandas": pd.__version__,
        "polars": pl.__version__,
        "narwhals": __import__("narwhals").__version__,
        "skrub": None if sbd is None else __import__("skrub").__version__,
        "sklearn": __import__("sklearn").__version__,
        "sktime_importable": check_is_mtype is not None,
    },
    "n_time": N_TIME,
    "number": args.number,
    "repeat": args.repeat,
    "rows": rows,
}
for r in rows:
    print(
        f"{r['backend']:7s} {r['op']:9s} {r['style']:28s} {r['us']:10.2f} us"
        f"  x{r.get('ratio_vs_native', '-')}  +{r.get('overhead_ms_per_10k_instances', '-')} ms/10k"
    )
os.makedirs("experiments/container_independence/results", exist_ok=True)
with open(
    f"experiments/container_independence/results/r1_dispatch_overhead_{args.tag}_{args.grid}.json", "w"
) as f:
    json.dump(out, f, indent=1)
