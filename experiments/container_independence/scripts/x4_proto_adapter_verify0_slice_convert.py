"""x4 verify0: is the per-slice conversion penalty (claim x4.4) intrinsic or sktime-overhead?

Independent code path (no x4_adapter): for a polars_panel y, produce N pandas
pd.Series slices (what a pd.Series-inner forecaster gets) four ways:

  once_sktime   : convert_to(whole, pd-multiindex) -> groupby(level=0) -> convert_to(slice, pd.Series)
                  (== current sktime path, same as x4 "C_once")
  slice_sktime  : partition_by -> convert_to(slice, pd.Series)  (== x4 "C_per_slice")
  once_raw      : whole.to_pandas() + set_index + groupby(level=0) + droplevel + squeeze
  slice_numpy   : partition_by -> pd.Series(col.to_numpy(), index=pd.DatetimeIndex(time.to_numpy()))
                  (hand-written per-slice converter, no sktime convert_to)

All slices are checked equal (values + time index) to once_sktime.
Interleaved min-of-repeats via common.bench; ratios reported vs once_sktime.

Run: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
     experiments/container_independence/scripts/x4_proto_adapter_verify0_slice_convert.py --grid dev
"""

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from common import DEV_GRID, LARGE_GRID, bench, env_info, make_panel_fast  # noqa: E402

from sktime.datatypes import convert_to  # noqa: E402

ED = os.path.join(os.path.dirname(__file__), "..")
T = "__index__time"


def once_sktime(y_pl):
    Xp = convert_to(y_pl, "pd-multiindex")
    return [convert_to(g.droplevel(0), "pd.Series") for _, g in Xp.groupby(level=0, sort=False)]


def slice_sktime(y_pl, keys):
    parts = y_pl.partition_by(keys[:-1], maintain_order=True, include_key=False)
    return [convert_to(p, "pd.Series") for p in parts]


def once_raw(y_pl, keys):
    Xp = y_pl.to_pandas().set_index(keys)
    return [g.droplevel(0).iloc[:, 0] for _, g in Xp.groupby(level=0, sort=False)]


def slice_numpy(y_pl, keys):
    parts = y_pl.partition_by(keys[:-1], maintain_order=True, include_key=False)
    vcol = [c for c in y_pl.columns if c not in keys][0]
    return [
        pd.Series(p.get_column(vcol).to_numpy(), index=pd.DatetimeIndex(p.get_column(keys[-1]).to_numpy()))
        for p in parts
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--max-series", type=int, default=10**9)
    args = ap.parse_args()
    grid = [g for g in (DEV_GRID if args.grid == "dev" else LARGE_GRID) if g[0] <= args.max_series]
    out = {"env": env_info(), "rows": []}
    for n, t in grid:
        y_pl = convert_to(make_panel_fast(n, t, 1), "polars_panel")
        keys = [c for c in y_pl.columns if c.startswith("__index__")]
        ref = once_sktime(y_pl)
        for name, fn in [("slice_sktime", lambda: slice_sktime(y_pl, keys)),
                         ("once_raw", lambda: once_raw(y_pl, keys)),
                         ("slice_numpy", lambda: slice_numpy(y_pl, keys))]:
            got = fn()
            assert len(got) == len(ref)
            for a, b in zip(ref, got):
                np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
                assert (pd.DatetimeIndex(a.index) == pd.DatetimeIndex(b.index)).all()
        rep = 5 if n <= 100 else 3
        row = {"n_series": n, "n_time": t}
        # interleave: two passes, min across both
        fns = {
            "once_sktime": lambda: once_sktime(y_pl),
            "slice_sktime": lambda: slice_sktime(y_pl, keys),
            "once_raw": lambda: once_raw(y_pl, keys),
            "slice_numpy": lambda: slice_numpy(y_pl, keys),
        }
        for _ in range(2):
            for k, f in fns.items():
                v = bench(f, repeat=rep)
                row[k] = min(row.get(k, float("inf")), v)
        for k in ["slice_sktime", "once_raw", "slice_numpy"]:
            row[f"{k}/once_sktime"] = row[k] / row["once_sktime"]
        out["rows"].append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    with open(os.path.join(ED, "results", f"x4_proto_adapter_verify0_slice_convert_{args.grid}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
