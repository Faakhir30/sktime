"""x4: cost of getting per-instance data into a forecaster's working representation.

Input is a polars Panel (``polars_panel``), univariate. For each grid size, the cost
of producing N per-instance objects in the representation a forecaster computes in:

  pd_once      current: convert whole frame pl->pd-multiindex once, VectorizedDF
               pandas slices, then slice -> pd.Series (inner mtype of NaiveForecaster)
  pd_per_slice adapter-native: polars slices, sktime convert_to(slice, "pd.Series")
  np_once      current route to numpy: whole pl->pd once, pandas slices, .to_numpy()
  np_per_slice adapter-native: polars slices, PolarsAdapter.to_numpy (zero-copy col)
  pl_per_slice adapter-native: polars slices as-is (polars-inner forecaster)
  np_reshape   ceiling for equal-length sorted panels: one zero-copy column ->
               reshape (n_series, n_time); no per-instance objects at all

Also registers (at runtime, in-process only) a direct ``pl.DataFrame -> np.ndarray``
Series converter in sktime's convert_dict, to show the mtype registry can carry
adapter-backed edges that skip pandas (sktime has no such edge today).

Usage:
  PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
    $EXP_PY -W ignore experiments/container_independence/scripts/x4_slice_extract.py [--grid dev|large]
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from common import DEV_GRID, LARGE_GRID, bench, env_info, make_panel_fast  # noqa: E402

from sktime.datatypes import convert_to  # noqa: E402
from sktime.datatypes._vectorize import prepare_VectorizedDF  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
from x4_adapter import PolarsAdapter, prepare_VectorizedDF2  # noqa: E402
from x4_instrument import ConvCounter  # noqa: E402

ED = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    pa_ = PolarsAdapter()
    res = {"env": env_info(), "grid": grid, "rows": []}

    # runtime-registered direct edge (no file change): pl.DataFrame -> np.ndarray
    import sktime.datatypes._convert as C

    key = ("pl.DataFrame", "np.ndarray", "Series")
    res["sktime_has_pl_to_np_edge"] = key in C.convert_dict
    C.convert_dict[key] = lambda obj, store=None: pa_.to_numpy(obj)
    y1 = convert_to(make_panel_fast(1, 50, 1).droplevel(0), "pl.DataFrame")
    with ConvCounter() as cc:
        arr = convert_to(y1, "np.ndarray")
    res["runtime_edge_pl_to_np"] = {"conversions": cc.as_dict(), "shape": list(arr.shape)}
    print("runtime edge:", res["runtime_edge_pl_to_np"], "pre-existing:", res["sktime_has_pl_to_np_edge"])

    for n_series, n_time in grid:
        rep = 5 if n_series <= 100 else 3
        y_pd = make_panel_fast(n_series, n_time, 1)
        y_pl = convert_to(y_pd, "polars_panel")
        vc, _ = prepare_VectorizedDF(y_pd, iterate_as="Series", is_scitype="Panel")
        vn, _ = prepare_VectorizedDF2(y_pl, iterate_as="Series", is_scitype="Panel", X_orig_mtype="polars_panel")

        def pd_once():
            Xp = convert_to(y_pl, "pd-multiindex")
            return [convert_to(s, "pd.Series") for s in vc.as_list(Xp)]

        def pd_per_slice():
            return [convert_to(s, "pd.Series") for s in vn.as_list(y_pl)]

        def np_once():
            Xp = convert_to(y_pl, "pd-multiindex")
            return [s.to_numpy() for s in vc.as_list(Xp)]

        def np_per_slice():
            return [pa_.to_numpy(s) for s in vn.as_list(y_pl)]

        def pl_per_slice():
            return vn.as_list(y_pl)

        def np_reshape():
            return y_pl.get_column("c0").to_numpy().reshape(n_series, n_time)

        # sanity: same numbers
        a = np.stack([x.ravel() for x in np_per_slice()])
        b = np.stack([x.ravel() for x in np_once()])
        assert np.allclose(a, b) and np.allclose(a, np_reshape())

        row = {"n_series": n_series, "n_time": n_time}
        # interleave the 6 variants, min over rounds
        fns = {
            "pd_once": pd_once,
            "pd_per_slice": pd_per_slice,
            "np_once": np_once,
            "np_per_slice": np_per_slice,
            "pl_per_slice": pl_per_slice,
            "np_reshape": np_reshape,
        }
        best = {k: float("inf") for k in fns}
        for r in range(rep):
            for k, fn in (list(fns.items())[r % 6 :] + list(fns.items())[: r % 6]):
                best[k] = min(best[k], bench(fn, repeat=1, warmup=1 if r == 0 else 0))
        row.update(best)
        res["rows"].append(row)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    out = os.path.join(ED, "results", f"x4_slice_extract_{args.grid}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    main()
