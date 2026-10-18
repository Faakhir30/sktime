"""r3_ts_libs_verify0: independent check of claim r3_ts_libs.3 (indptr vs groupby iteration).

The original E1 benchmark (r3_nixtla_boundary.py) compared iteration over a
PRE-BUILT numpy indptr hub (build cost excluded) to pandas groupby(level=0)
on a MultiIndex frame, and called groupby+droplevel "closest to VectorizedDF".

This script uses a different code path (no utilsforecast; the hub is built by
hand from pandas MultiIndex codes and from a polars frame) and adds the
variants the original left out, all interleaved in ONE process:

  hub_iter_only          : iterate prebuilt (data, indptr)  [original metric]
  hub_build_pd_codes+iter: build indptr from the pandas MultiIndex codes, iterate
                           (pandas input, no conversion out of pandas)
  hub_build_polars+iter  : build indptr from a polars long frame (sorted check +
                           rle lengths), iterate
  pd_groupby             : mi.groupby(level=0, sort=False)       [original]
  pd_groupby_droplevel   : + droplevel(0)                         [original]
  sktime_vdf_items       : real sktime VectorizedDF.items(X_mi)  (groupby +
                           droplevel + _enforce_index_freq), the true baseline
  hub_iter_build_pandas  : iterate the hub but hand each forecaster a pandas
                           DataFrame with freq-carrying DatetimeIndex (what an
                           unchanged pandas-inner forecaster would need)
  sktime_prepare_from_polars : sktime prepare_VectorizedDF on a polars_panel
                           input (the convert_to boundary), for scale

Each variant reads the last value of each series (same work as the original).

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/r3_ts_libs_verify0_iter.py --grid dev
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from common import DEV_GRID, LARGE_GRID, bench, env_info, make_panel_fast, rss_mb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--max-series", type=int, default=10**9)
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    grid = [g for g in grid if g[0] <= args.max_series and g[0] >= 100]

    from sktime.datatypes import convert_to
    from sktime.datatypes._vectorize import prepare_VectorizedDF

    res = {"env": env_info(), "grid": args.grid, "rows": []}
    for n_series, n_time in grid:
        mi = make_panel_fast(n_series, n_time)
        flat = mi.reset_index().rename(columns={"h0": "unique_id", "time": "ds", "c0": "y"})
        plf = pl.from_pandas(flat)
        pl_panel = convert_to(mi, "polars_panel")
        vec, X_mi = prepare_VectorizedDF(mi, iterate_as="Series", is_scitype="Panel")
        rep = 3 if n_series >= 10_000 else 5

        # prebuilt hub (original metric)
        codes = mi.index.codes[0]
        bnd = np.flatnonzero(np.diff(codes)) + 1
        indptr0 = np.concatenate([[0], bnd, [len(codes)]])
        data0 = mi.to_numpy()
        tidx = mi.index.get_level_values(1)

        def hub_iter_only():
            s = 0.0
            d, ip = data0, indptr0
            for i in range(len(ip) - 1):
                s += d[ip[i]:ip[i + 1], 0][-1]
            return s

        def hub_build_pd_codes_iter():
            c = mi.index.codes[0]
            if not (np.diff(c) >= 0).all():  # need grouped rows
                raise RuntimeError("unsorted")
            b = np.flatnonzero(np.diff(c)) + 1
            ip = np.concatenate([[0], b, [len(c)]])
            d = mi.to_numpy()
            s = 0.0
            for i in range(len(ip) - 1):
                s += d[ip[i]:ip[i + 1], 0][-1]
            return s

        def hub_build_polars_iter():
            ids = plf["unique_id"]
            lens = ids.rle()  # struct(len, value); assumes grouped (sorted) rows
            ln = lens.struct.field("len").to_numpy().astype(np.int64)
            ip = np.concatenate([[0], np.cumsum(ln)])
            d = plf.select("y").to_numpy()
            s = 0.0
            for i in range(len(ip) - 1):
                s += d[ip[i]:ip[i + 1], 0][-1]
            return s

        def pd_groupby():
            s = 0.0
            for _, g in mi.groupby(level=0, sort=False):
                s += g["c0"].to_numpy()[-1]
            return s

        def pd_groupby_droplevel():
            s = 0.0
            for _, g in mi.groupby(level=0, sort=False):
                g = g.droplevel(0)
                s += g["c0"].to_numpy()[-1]
            return s

        def sktime_vdf_items():
            s = 0.0
            for _, _, g in vec.items(X_mi):
                s += g["c0"].to_numpy()[-1]
            return s

        def hub_iter_build_pandas():
            s = 0.0
            d, ip = data0, indptr0
            for i in range(len(ip) - 1):
                a, b = ip[i], ip[i + 1]
                ti = pd.DatetimeIndex(tidx[a:b], freq="D")
                g = pd.DataFrame(d[a:b], index=ti, columns=["c0"])
                s += g["c0"].to_numpy()[-1]
            return s

        def sktime_prepare_from_polars():
            return prepare_VectorizedDF(pl_panel, iterate_as="Series", is_scitype="Panel")

        ref = hub_iter_only()
        for fn in [hub_build_pd_codes_iter, hub_build_polars_iter, pd_groupby,
                   pd_groupby_droplevel, sktime_vdf_items, hub_iter_build_pandas]:
            assert np.isclose(fn(), ref), fn.__name__

        row = {"n_series": n_series, "n_time": n_time}
        fns = [hub_iter_only, hub_build_pd_codes_iter, hub_build_polars_iter, pd_groupby,
               pd_groupby_droplevel, sktime_vdf_items, hub_iter_build_pandas,
               sktime_prepare_from_polars]
        # interleave: two rounds over all variants, keep the min
        for rnd in range(2):
            for fn in fns:
                t = bench(fn, repeat=rep if fn is not sktime_vdf_items else max(2, rep - 2))
                k = fn.__name__ + "_s"
                row[k] = min(row.get(k, float("inf")), t)
        b = row["hub_iter_only_s"]
        for k in list(row):
            if k.endswith("_s") and k != "hub_iter_only_s":
                row["ratio_" + k[:-2] + "_over_hub_iter_only"] = row[k] / b
        row["ratio_pd_groupby_over_hub_build_pd_codes_iter"] = (
            row["pd_groupby_s"] / row["hub_build_pd_codes_iter_s"])
        row["ratio_sktime_vdf_items_over_hub_build_pd_codes_iter"] = (
            row["sktime_vdf_items_s"] / row["hub_build_pd_codes_iter_s"])
        row["ratio_sktime_vdf_items_over_hub_iter_build_pandas"] = (
            row["sktime_vdf_items_s"] / row["hub_iter_build_pandas_s"])
        row["rss_mb"] = rss_mb()
        print(json.dumps(row), flush=True)
        res["rows"].append(row)

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"r3_ts_libs_verify0_iter_{args.grid}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    sys.exit(main())
