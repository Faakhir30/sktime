"""design_hub: cost of making hub ingress *safe* (wrap_critic g12 "hub guarded").

Measures, interleaved in one process (min of repeats):
  U   x6 hub ingress, unguarded (from_pandas / from_polars)
  G   U + guards: NaN/null instance key, strictly increasing time within instance
      (catches unsorted + duplicate + NaT), all-instance Step inference (g2 infer_step,
      raises on mixed freq)
  S   stock sktime validation of the same input (check_is_mtype; pandas: pd-multiindex,
      polars: polars_panel) -- the work the guards replace
  GT  G on time-major input (rows sorted by time, not instance)
Also checks that each guard fires on an adversarial fixture.

Run: PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
     $EXP_PY -W ignore experiments/container_independence/scripts/design_hub_guard_bench.py
"""
import json
import sys

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "experiments/container_independence/protos")
from common import bench, make_panel_fast  # noqa: E402
from g2_np_grouped import _within_mask, infer_step  # noqa: E402
from x6_hub import from_pandas, from_polars  # noqa: E402

from sktime.datatypes import check_is_mtype, convert_to  # noqa: E402


class HubGuardError(ValueError):
    pass


def guard_pandas_keys(obj):
    idx = obj.index
    if isinstance(idx, pd.MultiIndex):
        for c in idx.codes[:-1]:
            if (np.asarray(c) == -1).any():
                raise HubGuardError("NaN in instance key")


def guard_polars_keys(df, key_cols):
    if df.select(pl.sum_horizontal(pl.col(key_cols).null_count())).item() > 0:
        raise HubGuardError("null in instance key")


def guard_hub(hub):
    n = hub.n_rows
    if n > 1:
        d = np.diff(hub.time)[_within_mask(hub.indptr, n)]
        zero = d.dtype.type(0) if d.dtype.kind != "m" else np.timedelta64(0)
        if not (d > zero).all():
            raise HubGuardError("time not strictly increasing within instance")
    st = infer_step(hub)
    if st is None:
        raise HubGuardError("no regular step")
    return st


def run_U_pd(y):
    return from_pandas(y)


def run_G_pd(y):
    guard_pandas_keys(y)
    h = from_pandas(y)
    guard_hub(h)
    return h


def run_S_pd(y):
    return check_is_mtype(y, "pd-multiindex", return_metadata=["is_equally_spaced"])


def run_U_pl(df):
    return from_polars(df)


def run_G_pl(df):
    h = from_polars(df)
    guard_polars_keys(df, h.key_names)
    guard_hub(h)
    return h


def run_S_pl(df):
    return check_is_mtype(df, "polars_panel", return_metadata=["is_equally_spaced"])


def adversarial():
    out = {}
    y = make_panel_fast(4, 10)
    # NaN key
    yk = y.copy()
    lv0 = yk.index.get_level_values(0).to_numpy(dtype=object).copy()
    lv0[10:20] = np.nan
    yk.index = pd.MultiIndex.from_arrays([lv0, yk.index.get_level_values(1)],
                                         names=yk.index.names)
    # mixed freq: instance s0000001 hourly
    t = y.index.get_level_values(1).to_numpy().copy()
    t[10:20] = np.datetime64("2000-01-01") + np.arange(10) * np.timedelta64(1, "h")
    ym = y.copy()
    ym.index = pd.MultiIndex.from_arrays([y.index.get_level_values(0), t],
                                         names=y.index.names)
    # duplicate timestamp in one instance
    td = y.index.get_level_values(1).to_numpy().copy()
    td[11] = td[10]
    yd = y.copy()
    yd.index = pd.MultiIndex.from_arrays([y.index.get_level_values(0), td],
                                         names=y.index.names)
    for name, obj in [("nan_key", yk), ("mixed_freq", ym), ("dup_time", yd)]:
        r = {}
        try:
            from_pandas(obj)
            r["unguarded"] = "accepted (silent)"
        except Exception as e:  # noqa: BLE001
            r["unguarded"] = f"raised {type(e).__name__}"
        try:
            run_G_pd(obj)
            r["guarded"] = "accepted"
        except HubGuardError as e:
            r["guarded"] = f"raised: {e}"
        r["stock_check_valid"] = bool(check_is_mtype(obj, "pd-multiindex"))
        out[name] = r
    return out


def main():
    res = {"adversarial": adversarial(), "timing": []}
    print(json.dumps(res["adversarial"], indent=1))
    for n_series, n_time in [(1000, 100), (10000, 50), (50000, 20)]:
        y = make_panel_fast(n_series, n_time)
        ytm = y.sort_index(level="time", sort_remaining=True)
        ypl = convert_to(y, "polars_panel")
        ypl_tm = ypl.sort("__index__time", maintain_order=True)
        reps = 5 if n_series <= 1000 else 3
        row = {"size": f"{n_series}x{n_time}"}
        # interleave: each bench is min-of-repeats; run all variants back to back twice
        for _ in range(2):
            cand = {
                "pd_U": lambda: run_U_pd(y),
                "pd_G": lambda: run_G_pd(y),
                "pd_S": lambda: run_S_pd(y),
                "pd_GT": lambda: run_G_pd(ytm),
                "pl_U": lambda: run_U_pl(ypl),
                "pl_G": lambda: run_G_pl(ypl),
                "pl_S": lambda: run_S_pl(ypl),
                "pl_GT": lambda: run_G_pl(ypl_tm),
            }
            for k, fn in cand.items():
                t = bench(fn, repeat=reps)
                row[k] = min(row.get(k, float("inf")), t)
        for p in ("pd", "pl"):
            row[f"{p}_G_over_U"] = row[f"{p}_G"] / row[f"{p}_U"]
            row[f"{p}_S_over_G"] = row[f"{p}_S"] / row[f"{p}_G"]
            row[f"{p}_GT_over_G"] = row[f"{p}_GT"] / row[f"{p}_G"]
        res["timing"].append(row)
        print(json.dumps({k: (round(v * 1e3, 3) if k.split("_")[-1] in ("U", "G", "S", "GT")
                              and "_over_" not in k else
                              (round(v, 2) if isinstance(v, float) else v))
                          for k, v in row.items()}))
    with open("experiments/container_independence/results/design_hub_guard_bench.json", "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
