"""x2: verify the minimal index-preserving recipe (protos/x2_index_recipe.py).

For every case of x2_nw_roundtrip_fidelity.make_cases() and for two narwhals
backends (pandas, polars):

  long_to_mi(ops(nw.from_native(flat)), schema, sort_like_schema=True)  vs original
  iter_slices(nw.from_native(flat), schema)                              vs VectorizedDF.items

The ops in between are a with_columns + filter, plus (polars only) a sort by a data
column, which scrambles the row order on purpose.

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x2_recipe_fidelity.py
"""

import importlib.util
import json
import os
import sys
import traceback

import narwhals as nw
import numpy as np
import pandas as pd

from common import env_info

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
sys.path.insert(0, os.path.join(HERE, "..", "protos"))
from x2_index_recipe import iter_slices, long_to_mi, mi_to_long  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "fid", os.path.join(HERE, "x2_nw_roundtrip_fidelity.py")
)
fid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fid)


def fstr(ix):
    f = getattr(ix, "freq", None)
    return None if f is None else f.freqstr


def main():
    import polars as pl

    from sktime.datatypes._vectorize import prepare_VectorizedDF

    out = {"env": env_info(), "cases": {}}
    print(f"pandas {pd.__version__} narwhals {nw.__version__}")
    print(f"{'case':18s} {'backend':7s} names dtypes rows vals  eq | slices: n  order vals name dtype freq(sk,recipe)")
    for cname in fid.make_cases():
        for backend in ["pandas", "polars"]:
            scitype, df = fid.make_cases()[cname]
            _, ref = fid.make_cases()[cname]
            rec = {}
            try:
                flat, schema = mi_to_long(df, period_to_timestamp=(backend == "polars"))
                native = flat if backend == "pandas" else pl.from_pandas(flat)
                nf = nw.from_native(native, eager_only=True)
                ops = nf.with_columns(nw.col("c0") * 1.0).filter(nw.col("c0") > -np.inf)
                if backend == "polars":
                    ops = ops.sort("c1")
                back = long_to_mi(ops, schema, sort_like_schema=True)
                c = fid.compare(ref, back)
                rec.update({k: c[k] for k in ["names_ok", "level_dtypes_ok", "row_order_ok", "values_ok", "frame_equals", "level_dtypes"]})

                # slices vs current sktime
                _, ref2 = fid.make_cases()[cname]
                vec, X_mi = prepare_VectorizedDF(ref2, iterate_as="Series", is_scitype=scitype)
                sk = [(k, g) for k, _, g in vec.items(X=X_mi)]
                mine = list(iter_slices(nf, schema))
                rec["n_slices"] = (len(sk), len(mine))
                rec["slice_order_ok"] = [k for k, _ in sk] == [k for k, _ in mine]
                rec["slice_values_ok"] = all(
                    np.allclose(a.to_numpy(), b.to_numpy()) for (_, a), (_, b) in zip(sk, mine)
                )
                rec["slice_index_values_ok"] = all(
                    a.index.equals(b.index) for (_, a), (_, b) in zip(sk, mine)
                )
                rec["slice_name_ok"] = all(a.index.name == b.index.name for (_, a), (_, b) in zip(sk, mine))
                rec["slice_dtype_ok"] = all(a.index.dtype == b.index.dtype for (_, a), (_, b) in zip(sk, mine))
                rec["slice_freq"] = (fstr(sk[0][1].index), fstr(mine[0][1].index))
                rec["slice_freq_ok"] = all(fstr(a.index) == fstr(b.index) for (_, a), (_, b) in zip(sk, mine))
                print(
                    f"{cname:18s} {backend:7s} {str(rec['names_ok'])[0]:5s} {str(rec['level_dtypes_ok'])[0]:6s} "
                    f"{str(rec['row_order_ok'])[0]:4s} {str(rec['values_ok'])[0]:4s} {str(rec['frame_equals'])[0]:3s} | "
                    f"{rec['n_slices'][1]:>8d} {str(rec['slice_order_ok'])[0]:5s} "
                    f"{str(rec['slice_values_ok'] and rec['slice_index_values_ok'])[0]:4s} "
                    f"{str(rec['slice_name_ok'])[0]:4s} {str(rec['slice_dtype_ok'])[0]:5s} {rec['slice_freq']}"
                    + ("" if rec["level_dtypes_ok"] else f"  dtypes={rec['level_dtypes']}")
                )
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["tb"] = traceback.format_exc().splitlines()[-4:]
                print(f"{cname:18s} {backend:7s} ERROR {rec['error'][:150]}")
            out["cases"][f"{cname}|{backend}"] = rec
    tag = "pd" + pd.__version__.split(".")[0]
    path = os.path.join(RES, f"x2_recipe_fidelity_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
