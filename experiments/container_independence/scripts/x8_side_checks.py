"""x8 (aux): side measurements supporting the e2e profile.

(a) input-check cost vs n for pandas MI vs sktime polars_panel
    (polars check has a python loop over groups in ``is_monotonically_increasing``)
(b) conversion cost vs n (polars_panel <-> pd-multiindex) via convert_to
(c) VectorizedDF slicing: per-group cost of the groupby iteration vs n (linearity)
(d) FutureWarning lines printed per vectorized fit+predict (filter reset by
    ``_SuppressWarningPattern.__exit__``), with the user's filter set to "ignore"

Run (from worktree root)::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x8_side_checks.py --grid dev
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))

GRIDS = {"dev": [100, 1000, 5000, 20000], "large": [1000, 10000, 50000, 100000]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="dev", choices=sorted(GRIDS))
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    warnings.simplefilter("ignore")
    from common import bench, env_info, make_panel_fast

    from sktime.datatypes import check_is_scitype, convert_to
    from sktime.datatypes._adapter.polars import is_monotonically_increasing
    from sktime.datatypes._vectorize import prepare_VectorizedDF

    res = {"env": env_info(), "T": a.T, "checks": [], "warnings": {}}
    meta_req = ["n_features", "feature_names", "feature_kind", "is_univariate", "has_nans"]
    for n in GRIDS[a.grid]:
        y = make_panel_fast(n, a.T)
        ypl = convert_to(y, "polars_panel")
        rep = 3 if n <= 5000 else 1
        row = {"n_series": n}
        row["check_pd"] = bench(
            lambda: check_is_scitype(y, scitype=["Series", "Panel", "Hierarchical"],
                                     return_metadata=meta_req), repeat=rep)
        row["check_pl"] = bench(
            lambda: check_is_scitype(ypl, scitype=["Series", "Panel", "Hierarchical"],
                                     return_metadata=meta_req), repeat=rep)
        row["pl_is_monotonic_loop"] = bench(lambda: is_monotonically_increasing(ypl), repeat=rep)
        row["conv_pl_to_pd"] = bench(lambda: convert_to(ypl, "pd-multiindex"), repeat=rep)
        row["conv_pd_to_pl"] = bench(lambda: convert_to(y, "polars_panel"), repeat=rep)
        vec, ymi = prepare_VectorizedDF(y, iterate_as="Series", is_scitype="Panel")

        def iterate():
            k = 0
            for _ in vec.items(X=ymi):
                k += 1
            return k

        row["slice_iter_total"] = bench(iterate, repeat=rep)
        row["slice_iter_per_group_us"] = 1e6 * row["slice_iter_total"] / n
        row["prepare_VectorizedDF"] = bench(
            lambda: prepare_VectorizedDF(y, iterate_as="Series", is_scitype="Panel"),
            repeat=rep)
        res["checks"].append(row)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()},
              flush=True)

    # (d) warnings printed per vectorized fit/predict, user filter = ignore
    from common import make_panel_fast as mpf
    from sktime.forecasting.naive import NaiveForecaster

    for n in [50, 200]:
        y = mpf(n, 50)
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            t0 = time.perf_counter()
            NaiveForecaster().fit(y, fh=[1, 2, 3]).predict()
            el = time.perf_counter() - t0
        txt = buf.getvalue()
        res["warnings"][str(n)] = {
            "futurewarning_lines": txt.count("FutureWarning"),
            "remember_data_warnings": txt.count("remember_data"),
            "stderr_bytes": len(txt),
            "top_filter_after": str(warnings.filters[0][:3]),
            "elapsed": el,
        }
        print("warnings", n, res["warnings"][str(n)], flush=True)

    out = a.out or os.path.join(os.path.dirname(HERE), "results", f"x8_side_checks_{a.grid}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
