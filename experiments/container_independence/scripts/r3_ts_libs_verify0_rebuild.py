"""r3_ts_libs_verify0: adversarial check of claim r3_ts_libs.3.

Claim: "a hub that must rebuild a freq-carrying pandas DataFrame per series is
~2.2x SLOWER than today's VectorizedDF.items()". The original rebuild used
``pd.DatetimeIndex(tidx[a:b], freq="D")``, which makes pandas VALIDATE the freq
against a freshly generated date_range for every series. That is one specific
(slow) way to rebuild. This script tries cheaper, still-correct rebuilds and
compares them, interleaved in one process, against the real sktime
``VectorizedDF.items()``:

  vdf_items          : real sktime VectorizedDF.items (groupby+droplevel+infer_freq)
  rebuild_validate   : original: pd.DatetimeIndex(slice, freq="D")   [claim .3]
  rebuild_date_range : pd.date_range(start=t0, periods=n, freq=freq) (freq known
                       from the stateless schema; valid for regular series only)
  rebuild_slice_freq : slice of the time level (a view) + set ``_freq`` without
                       validation (trusting the schema), pd.DataFrame(ndarray)
  rebuild_slice_infer: slice + pd.infer_freq per series (no schema trust; this is
                       what _enforce_index_freq does today)

Every variant hands the consumer a pd.DataFrame whose index equals the
groupby slice and whose ``index.freq`` is "D"; checked for every series once.

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/r3_ts_libs_verify0_rebuild.py --grid dev
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import DEV_GRID, LARGE_GRID, bench, env_info, make_panel_fast, rss_mb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--max-series", type=int, default=10**9)
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    grid = [g for g in grid if 100 <= g[0] <= args.max_series]

    from sktime.datatypes._vectorize import prepare_VectorizedDF

    res = {"env": env_info(), "grid": args.grid, "rows": []}
    for n_series, n_time in grid:
        mi = make_panel_fast(n_series, n_time)
        vec, X_mi = prepare_VectorizedDF(mi, iterate_as="Series", is_scitype="Panel")
        codes = mi.index.codes[0]
        b = np.flatnonzero(np.diff(codes)) + 1
        ip = np.concatenate([[0], b, [len(codes)]])
        data = mi.to_numpy()
        tidx = mi.index.get_level_values(1)
        tvals = tidx.to_numpy()
        freq = "D"
        off = pd.tseries.frequencies.to_offset(freq)
        cols = pd.Index(["c0"])

        def consume(g):
            return g["c0"].to_numpy()[-1]

        def vdf_items(check=None):
            s = 0.0
            for _, _, g in vec.items(X_mi):
                if check is not None:
                    check(g)
                s += consume(g)
            return s

        def _mk(builder):
            def run(check=None):
                s = 0.0
                for i in range(len(ip) - 1):
                    a, z = ip[i], ip[i + 1]
                    g = builder(a, z)
                    if check is not None:
                        check(g)
                    s += consume(g)
                return s
            run.__name__ = builder.__name__
            return run

        def rebuild_validate(a, z):
            return pd.DataFrame(data[a:z], index=pd.DatetimeIndex(tidx[a:z], freq=freq),
                                columns=cols)

        def rebuild_date_range(a, z):
            return pd.DataFrame(data[a:z],
                                index=pd.date_range(tvals[a], periods=z - a, freq=off),
                                columns=cols)

        def rebuild_slice_freq(a, z):
            ix = tidx[a:z]
            ix._data._freq = off  # trust schema freq; no validation (private API)
            return pd.DataFrame(data[a:z], index=ix, columns=cols)

        def rebuild_slice_infer(a, z):
            ix = tidx[a:z]
            ix._data._freq = pd.tseries.frequencies.to_offset(pd.infer_freq(ix))
            return pd.DataFrame(data[a:z], index=ix, columns=cols)

        variants = [vdf_items] + [_mk(f) for f in (rebuild_validate, rebuild_date_range,
                                                   rebuild_slice_freq, rebuild_slice_infer)]

        # correctness: same per-series frames as the groupby slice, freq == "D"
        ref = [g for _, _, g in vec.items(X_mi)]
        for fn in variants[1:]:
            k = [0]

            def chk(g, k=k):
                r = ref[k[0]]
                assert g.index.equals(r.index) and g.index.freqstr == "D", fn.__name__
                assert np.array_equal(g.to_numpy(), r.to_numpy()), fn.__name__
                k[0] += 1
            fn(check=chk)
            assert k[0] == n_series

        rep = 3 if n_series >= 10_000 else 5
        row = {"n_series": n_series, "n_time": n_time}
        for _ in range(2):
            for fn in variants:
                t = bench(fn, repeat=rep)
                row[fn.__name__ + "_s"] = min(row.get(fn.__name__ + "_s", 1e9), t)
        base = row["vdf_items_s"]
        for fn in variants[1:]:
            row["ratio_" + fn.__name__ + "_over_vdf_items"] = row[fn.__name__ + "_s"] / base
        row["rss_mb"] = rss_mb()
        print(json.dumps(row), flush=True)
        res["rows"].append(row)

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"r3_ts_libs_verify0_rebuild_{args.grid}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    sys.exit(main())
