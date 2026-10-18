"""a1 verify0: the 2112 grid failures in test_vectorize_est stop at
`assert isinstance(result, pd.DataFrame)` (test_vectorize.py:535); the shape and
cell-type assertions after it never run. Here we run the SAME fixture loop as
test_vectorize_est (subset) and check the masked assertions on the grid:
shape == (n_rows, n_cols), all cells are fitted NaiveForecaster, and row/col keys
equal the baseline DataFrame's index/columns.

Run from WT root:
  PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
     $EXP_PY -W ignore experiments/container_independence/scripts/a1_core_coupling_verify0_grid_masked.py
"""
import itertools
import json
import warnings

warnings.filterwarnings("ignore")

import a1_estimator_grid as G  # noqa: E402

from sktime.datatypes import prepare_VectorizedDF  # noqa: E402
from sktime.datatypes._examples import get_examples  # noqa: E402
from sktime.datatypes._vectorize import VectorizedDF  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402

ORIG = VectorizedDF.vectorize_est
CASES = []
for scitype, mtypes, ias in [
    ("Panel", ["pd-multiindex", "numpy3D", "df-list"], ["Series"]),
    ("Hierarchical", ["pd_multiindex_hier"], ["Series", "Panel"]),
]:
    for mt, it_as, it_cols in itertools.product(mtypes, ias, [False, True]):
        for fi in range(3):
            fx = get_examples(mtype=mt, as_scitype=scitype).get(fi)
            if fx is not None:
                CASES.append((scitype, mt, fi, it_as, it_cols, fx))


def run(fx, it_as, it_cols):
    X_vect, X_mi = prepare_VectorizedDF(
        X=fx, iterate_as=it_as, is_scitype=None, iterate_cols=it_cols
    )
    kw = {"fh": [1, 2], "X": (X_mi, X_vect), "y": (X_mi, X_vect)}
    clones = X_vect.vectorize_est(NaiveForecaster(), method="clone")
    res = X_vect.vectorize_est(clones, method="fit", **kw)
    rows, cols = X_vect.get_iter_indices()
    nr = 1 if rows is None else len(rows)
    nc = 1 if cols is None else len(cols)
    return res, (nr, nc)


out = []
for scitype, mt, fi, it_as, it_cols, fx in CASES:
    tag = f"{scitype}|{mt}|{fi}|{it_as}|cols={it_cols}"
    rec = {"case": tag}
    try:
        VectorizedDF.vectorize_est = ORIG
        df, exp = run(fx, it_as, it_cols)
        G.install("strict")
        g, exp2 = run(fx, it_as, it_cols)
        rec["baseline_shape_ok"] = df.shape == exp
        rec["grid_shape_ok"] = tuple(g.shape) == exp2
        rec["grid_cells_fitted_naive"] = all(
            isinstance(c, NaiveForecaster) and c.is_fitted for c in g.cells()
        )
        rec["keys_equal"] = list(df.index) == list(g.row_keys) and list(
            df.columns
        ) == list(g.col_keys)
        rec["ok"] = all(v for k, v in rec.items() if k != "case")
    except Exception as e:  # noqa: BLE001
        rec["ok"] = False
        rec["exc"] = f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        VectorizedDF.vectorize_est = ORIG
    out.append(rec)
    print(rec)

n_ok = sum(r["ok"] for r in out)
print(f"masked-assertion check: {n_ok}/{len(out)} cases ok")
json.dump(out, open("experiments/container_independence/results/a1_core_coupling_verify0_grid_masked.json", "w"), indent=1)
