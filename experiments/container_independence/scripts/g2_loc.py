"""g2 LOC: logical lines (no blank / comment / docstring lines) of ported vs stock code.

    PYTHONPATH=.:experiments/container_independence $EXP_PY \
        experiments/container_independence/scripts/g2_loc.py
"""

import ast
import inspect
import json
import os
import sys

ED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ED)


def _doc_lines(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef,
                          ast.Module)) and n.body:
            b = n.body[0]
            if isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None),
                                                       ast.Constant) and isinstance(
                    b.value.value, str):
                out.update(range(b.lineno, b.end_lineno + 1))
    return out


def lloc(obj, skip=("get_test_params",)):
    src = inspect.getsource(obj)
    src = inspect.cleandoc("\n" + src) if src.startswith(" ") else src
    tree = ast.parse(src)
    lines = src.splitlines()
    keep = set()
    top = tree.body[0]
    body = top.body if isinstance(top, ast.ClassDef) else [top]
    if isinstance(top, ast.ClassDef):
        keep.add(top.lineno)
    for n in body:
        if getattr(n, "name", None) in skip:
            continue
        keep.update(range(n.lineno, n.end_lineno + 1))
    keep -= _doc_lines(tree)
    return sum(1 for i in keep if lines[i - 1].strip() and
               not lines[i - 1].strip().startswith("#"))


def file_lloc(path):
    src = open(path).read()
    tree = ast.parse(src)
    doc = _doc_lines(tree)
    return sum(1 for i, ln in enumerate(src.splitlines(), 1)
               if i not in doc and ln.strip() and not ln.strip().startswith("#"))


def main():
    import protos.g2_forecasters as g
    import protos.x6_hub as x6
    from sktime.forecasting.base import _sktime as sk
    from sktime.forecasting.compose import _reduce as red
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.naive import _naive as nv
    from sktime.forecasting.trend import PolynomialTrendForecaster
    from sktime.forecasting.trend import _util as tu

    res = {
        "g2": {
            "shared _G2Base (grid, output hub, proba frame, _update)": lloc(g._G2Base),
            "G2NaiveForecaster": lloc(g.G2NaiveForecaster) + lloc(g._seg_last_valid)
            + lloc(g._seg_nanmean),
            "G2PolynomialTrendForecaster": lloc(g.G2PolynomialTrendForecaster)
            + lloc(g._lstsq_cond),
            "G2RecursiveReducer": lloc(g.G2RecursiveReducer),
        },
        "stock": {
            "NaiveForecaster (+_pivot_sp/_unpivot_sp helpers)": lloc(NaiveForecaster)
            + lloc(nv._pivot_sp) + lloc(nv._unpivot_sp),
            "_BaseWindowForecaster (shared by Naive + reducers)":
                lloc(sk._BaseWindowForecaster),
            "PolynomialTrendForecaster (+_get_X_numpy_int_from_pandas)":
                lloc(PolynomialTrendForecaster) + lloc(tu._get_X_numpy_int_from_pandas),
            "_RecursiveReducer + _Reducer + sliding-window helpers":
                lloc(red._RecursiveReducer) + lloc(red._Reducer)
                + lloc(red._sliding_window_transform)
                + lloc(red._sliding_window_transform_local) + lloc(red._concat_y_X)
                + lloc(red._check_fh),
        },
        "framework": {
            "protos/g2_np_grouped.py (container, step inference, time ops, "
            "converters, mtype classes, registration)":
                file_lloc(os.path.join(ED, "protos", "g2_np_grouped.py")),
            "reused x6 hand-written spokes pandas/polars/arrow (+core)":
                sum(lloc(f) for f in [x6._pd_time_meta, x6.from_pandas, x6.to_pandas,
                                      x6._pl_time_meta, x6.from_polars, x6.to_polars,
                                      x6._np_from_arrow, x6._arrow_codes,
                                      x6.from_arrow, x6.to_arrow,
                                      x6.time_to_pandas_index, x6.infer_step_numpy,
                                      x6._run_starts_from_codes, x6._composite,
                                      x6._first_appearance_perm, x6._starts_to_indptr,
                                      x6._stack_cols, x6.PanelData]),
            "reused a3 Step + NumpyTimeOps + PandasTimeOps.to_offset":
                lloc(__import__("protos.a3_timeindex", fromlist=["x"]).Step)
                + lloc(__import__("protos.a3_timeindex", fromlist=["x"]).NumpyTimeOps),
        },
    }
    print(json.dumps(res, indent=1))
    with open(os.path.join(ED, "results", "g2_loc.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
