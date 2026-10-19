"""design_minimal: end-to-end cost of Arrow-valued pandas (hybrid MI: numpy index, double[pyarrow] values)
through stock VectorizedDF + NaiveForecaster / PolynomialTrendForecaster (critic G8, e2e unmeasured).

Interleaved in one process, min over reps. Ratio = time(arrow) / time(numpy) (>1 means Arrow is slower).
Also records the output dtype (does Arrow survive) and value equality.

Usage: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
       experiments/container_independence/scripts/design_minimal_arrow.py --sizes 300x100
"""

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from common import make_panel_fast  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402
from sktime.forecasting.trend import PolynomialTrendForecaster  # noqa: E402

ESTS = {
    "naive": lambda: NaiveForecaster(strategy="last").set_config(remember_data=True),
    "poly": lambda: PolynomialTrendForecaster(degree=1).set_config(remember_data=True),
}


def fp(est, y):
    f = ESTS[est]().fit(y, fh=[1, 2, 3])
    return f.predict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="300x100")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="experiments/container_independence/results/design_minimal_arrow.json")
    a = ap.parse_args()
    rows = []
    for sz in a.sizes.split(","):
        n, t = map(int, sz.split("x"))
        ynp = make_panel_fast(n, t)
        yar = ynp.astype("double[pyarrow]")
        for est in ESTS:
            pn, pa_ = fp(est, ynp), fp(est, yar)
            eq = bool(np.allclose(pn.to_numpy(dtype=float), pa_.to_numpy(dtype=float), rtol=1e-12))
            tn = ta = float("inf")
            for _ in range(a.reps):
                t0 = time.perf_counter(); fp(est, ynp); tn = min(tn, time.perf_counter() - t0)
                t0 = time.perf_counter(); fp(est, yar); ta = min(ta, time.perf_counter() - t0)
            row = {"size": sz, "est": est, "numpy_s": tn, "arrow_s": ta, "arrow_over_numpy": ta / tn,
                   "values_equal": eq, "out_dtype_arrow_input": str(pa_.dtypes.iloc[0]),
                   "index_equal": bool(pn.index.equals(pa_.index))}
            print(json.dumps(row), flush=True)
            rows.append(row)
    with open(a.out, "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
