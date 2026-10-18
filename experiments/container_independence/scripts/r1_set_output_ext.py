"""r1: sklearn set_output -- (a) extension story, (b) boundary-conversion tax.

(a) Register a third-party pyarrow adapter (protos/r1_sklearn_pyarrow_adapter.py)
    in sklearn's ADAPTERS_MANAGER and check which estimators honour
    set_output(transform="pyarrow"): a plain transformer, ColumnTransformer,
    FeatureUnion, Pipeline, FunctionTransformer.
(b) "numpy inside, wrap at the edge" tax: StandardScaler.fit_transform on the
    DEV/LARGE grid row counts with input {numpy, pandas, polars} x output
    {default, pandas, polars}; ratio vs numpy-in/default-out, same process,
    interleaved min-of-repeats.

Run:
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/r1_set_output_ext.py --grid dev
"""

import argparse
import inspect
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from common import DEV_GRID, LARGE_GRID, bench, peak_rss_delta_mb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="dev", choices=["dev", "large"])
ap.add_argument("--repeat", type=int, default=5)
ap.add_argument("--n-cols", type=int, default=4)
args = ap.parse_args()
GRID = DEV_GRID if args.grid == "dev" else LARGE_GRID

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
import r1_sklearn_pyarrow_adapter as proto  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.pipeline import FeatureUnion, make_pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    FunctionTransformer,
    MinMaxScaler,
    StandardScaler,
)

out = {"sklearn": __import__("sklearn").__version__}

# ------------------------------------------------------------------ (a)
src = inspect.getsource(proto.PyArrowAdapter)
out["adapter_loc"] = len([ln for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#")])
proto.register()
from sklearn.utils._set_output import ADAPTERS_MANAGER  # noqa: E402

out["registered_after"] = sorted(ADAPTERS_MANAGER.adapters)
rng = np.random.default_rng(0)
Xpd = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
Xpl = pl.from_pandas(Xpd)
ests = {
    "StandardScaler": lambda: StandardScaler(),
    "ColumnTransformer": lambda: ColumnTransformer(
        [("s", StandardScaler(), ["a", "b"]), ("m", MinMaxScaler(), ["c"])]
    ),
    "FeatureUnion": lambda: FeatureUnion([("s", StandardScaler()), ("m", MinMaxScaler())]),
    "Pipeline": lambda: make_pipeline(StandardScaler(), MinMaxScaler()),
    "FunctionTransformer(feature_names_out=one-to-one)": lambda: FunctionTransformer(
        np.log1p, feature_names_out="one-to-one"
    ),
}
ext = {}
for name, mk in ests.items():
    for in_name, X in [("pandas", Xpd), ("polars", Xpl)]:
        try:
            # FunctionTransformer(np.log1p) needs non-negative input
            Xin = (X.abs() if in_name == "pandas" else X.select(pl.all().abs())) if "Function" in name else X
            o = mk().set_output(transform="pyarrow").fit_transform(Xin)
            ext[f"{name} <- {in_name}"] = f"ok: {type(o).__module__.split('.')[0]}.{type(o).__name__} cols={list(getattr(o, 'column_names', []))}"
        except Exception as e:  # noqa: BLE001
            ext[f"{name} <- {in_name}"] = f"FAIL {type(e).__name__}: {str(e)[:160]}"
out["extension_results"] = ext

# ------------------------------------------------------------------ (b)
rows = []
for n_series, n_time in GRID:
    n = n_series * n_time
    A = rng.normal(size=(n, args.n_cols))
    cols = [f"c{i}" for i in range(args.n_cols)]
    inputs = {
        "numpy": A,  # C-ordered (row-major)
        "numpy_F": np.asfortranarray(A),  # column-major, what columnar to_numpy yields
        "pandas": pd.DataFrame(A, columns=cols),
        "polars": pl.DataFrame(A, schema=cols, orient="row"),
    }
    combos = {}
    for iname, X in inputs.items():
        for oname in ["default", "pandas", "polars"]:
            if iname == "numpy" and oname != "default":
                # numpy input + df output works too (column names x0..), include
                pass
            combos[(iname, oname)] = (lambda X=X, oname=oname: StandardScaler().set_output(transform=oname).fit_transform(X))
    best = {k: float("inf") for k in combos}
    for r in range(args.repeat):
        for k, fn in combos.items():
            best[k] = min(best[k], bench(fn, repeat=1, warmup=1 if r == 0 else 0))
    base = best[("numpy", "default")]
    for (iname, oname), t in best.items():
        _, mem = peak_rss_delta_mb(combos[(iname, oname)])
        rows.append(
            dict(n_series=n_series, n_time=n_time, n_rows=n, input=iname, output=oname,
                 ms=round(t * 1e3, 3), ratio_vs_numpy_default=round(t / base, 2),
                 peak_rss_mb=round(mem, 1))
        )
    del A, inputs
out["boundary_tax"] = rows
# memory layout of the array each container hands to sklearn (explains ratios < 1)
_A = rng.normal(size=(10, 3))
out["to_numpy_layout"] = {
    "numpy C": "F" if np.asarray(_A).flags["F_CONTIGUOUS"] else "C",
    "pandas.to_numpy()": "F" if pd.DataFrame(_A).to_numpy().flags["F_CONTIGUOUS"] else "C",
    "polars.to_numpy()": "F" if pl.DataFrame(_A, orient="row").to_numpy().flags["F_CONTIGUOUS"] else "C",
}

print(json.dumps({k: v for k, v in out.items() if k != "boundary_tax"}, indent=1))
print("to_numpy_layout", out["to_numpy_layout"])
for r in rows:
    print(f"{r['n_rows']:>8} in={r['input']:7s} out={r['output']:8s} {r['ms']:9.3f} ms  x{r['ratio_vs_numpy_default']}  rss+{r['peak_rss_mb']}MB")
os.makedirs("experiments/container_independence/results", exist_ok=True)
with open(f"experiments/container_independence/results/r1_set_output_ext_{args.grid}.json", "w") as f:
    json.dump(out, f, indent=1)
