"""r4: "keep the pandas API, swap the engine" (Modin) -- fidelity + cost for the
pandas operations VectorizedDF actually uses (MultiIndex-heavy).

For each op we record: result equal to pandas? index names kept? how many
"defaulting to pandas" warnings fired? time relative to pandas.

Run with the r4 private venv (modin[dask] installed there), e.g.:
  MODIN_ENGINE=dask MODIN_CPUS=2 PYTHONPATH=experiments/container_independence \
    $SCRATCH/venvs/r4/bin/python -W ignore \
    experiments/container_independence/scripts/r4_modin_fidelity.py --grid dev
or the serial debugging engine (shared env):
  MODIN_ENGINE=python PYTHONPATH=.:experiments/container_independence $EXP_PY \
    experiments/container_independence/scripts/r4_modin_fidelity.py --grid dev
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from common import DEV_GRID, LARGE_GRID, make_panel, make_panel_fast, rss_mb

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")


def to_pd(x):
    if hasattr(x, "_to_pandas"):
        return x._to_pandas()
    return x


def run_op(fn, *args):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        res = fn(*args)
        res = to_pd(res) if not isinstance(res, (list, dict, tuple)) else res
        dt = time.perf_counter() - t0
    n_default = sum("default" in str(x.message).lower() for x in w)
    return res, dt, n_default


def ops():
    def iter_groups(X, mpd):
        # VectorizedDF._iter_indices: for name, group in X.groupby(level=..., sort=False)
        lv = list(range(X.index.nlevels - 1))
        lv = lv[0] if len(lv) == 1 else lv
        out = []
        for name, g in X.groupby(level=lv, sort=False):
            g2 = g.droplevel(lv)
            out.append((name, to_pd(g2)))
        return out

    def xs_first(X, mpd):
        return X.xs(X.index.droplevel(-1)[0])

    def row_ix(X, mpd):
        return X.index.droplevel(-1).unique()

    def concat_keys(X, mpd):
        lv = list(range(X.index.nlevels - 1))
        lv = lv[0] if len(lv) == 1 else lv
        groups = [(k, g.droplevel(lv)) for k, g in X.groupby(level=lv, sort=False)]
        keys = [k for k, _ in groups]
        lib = mpd if mpd is not None else pd
        return lib.concat([g for _, g in groups], keys=keys, axis=0)

    def agg_level(X, mpd):
        lv = list(range(X.index.nlevels - 1))
        return X.groupby(level=lv, sort=False).mean()

    def apply_level(X, mpd):
        lv = list(range(X.index.nlevels - 1))
        return X.groupby(level=lv, sort=False)["c0"].apply(
            lambda s: float(s.mean() + s.iloc[-1])
        )

    def reset_set(X, mpd):
        names = list(X.index.names)
        return X.reset_index().set_index(names)

    def loc_tuple(X, mpd):
        k = X.index[0][:-1]
        k = k[0] if len(k) == 1 else k
        return X.loc[k]

    return {
        "groupby(level, sort=False) iterate + droplevel": iter_groups,
        "xs(instance key)": xs_first,
        "index.droplevel(-1).unique()": row_ix,
        "concat(keys=instance keys)": concat_keys,
        "groupby(level).mean()": agg_level,
        "groupby(level)[c].apply(f)": apply_level,
        "reset_index().set_index(names) roundtrip": reset_set,
        "loc[instance key]": loc_tuple,
    }


def compare(a, b):
    try:
        if isinstance(a, list):
            return all(
                ka == kb and va.equals(vb) and list(va.index.names) == list(vb.index.names)
                for (ka, va), (kb, vb) in zip(a, b)
            ) and len(a) == len(b)
        if isinstance(a, pd.Index):
            return a.equals(b) and list(a.names) == list(b.names)
        ok = a.equals(b)
        names = list(a.index.names) == list(b.index.names)
        return bool(ok and names)
    except Exception as e:  # noqa: BLE001
        return f"compare-error {e!r}"[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])
    args = ap.parse_args()
    import modin
    import modin.pandas as mpd

    engine = os.environ.get("MODIN_ENGINE", "default")
    out = {
        "env": {"modin": modin.__version__, "pandas": pd.__version__, "engine": engine},
        "python": sys.version.split()[0],
        "isinstance(modin_df, pd.DataFrame)": None,
        "datasets": [],
    }
    datasets = [
        ("panel datetime 2-level", make_panel(20, 30, n_levels=1)),
        ("panel period 2-level", make_panel(20, 30, n_levels=1, time_kind="period", freq="M")),
        ("hierarchical 3-level", make_panel(20, 30, n_levels=2)),
    ]
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    # timing rows on the largest dev size that is still quick
    for n_series, n_time in grid[1:3]:
        datasets.append(
            (f"panel datetime {n_series}x{n_time} (timing)", make_panel_fast(n_series, n_time))
        )
    for name, X in datasets:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mX = mpd.DataFrame(X)
        out["isinstance(modin_df, pd.DataFrame)"] = isinstance(mX, pd.DataFrame)
        rec = {
            "dataset": name,
            "construct_defaults": sum("default" in str(x.message).lower() for x in w),
            "index_type_after_construct": type(mX.index).__name__,
            "index_names_after_construct": list(mX.index.names),
            "ops": {},
        }
        for opname, fn in ops().items():
            try:
                ref, t_pd, _ = run_op(fn, X, None)
                got, t_md, n_def = run_op(fn, mX, mpd)
                rec["ops"][opname] = {
                    "equal_incl_names": compare(ref, got),
                    "n_defaulting_to_pandas_warnings": n_def,
                    "t_pandas_ms": round(t_pd * 1e3, 2),
                    "t_modin_ms": round(t_md * 1e3, 2),
                    "ratio": round(t_md / t_pd, 1) if t_pd > 0 else None,
                }
            except Exception as e:  # noqa: BLE001
                rec["ops"][opname] = {"error": repr(e)[:300]}
        rec["rss_mb"] = rss_mb()
        out["datasets"].append(rec)
        print(f"\n== {name}: construct_defaults={rec['construct_defaults']} "
              f"index={rec['index_type_after_construct']} names={rec['index_names_after_construct']}",
              flush=True)
        for k, v in rec["ops"].items():
            print(f"  {k:48s} {v}", flush=True)
    print("isinstance(modin_df, pd.DataFrame) =", out["isinstance(modin_df, pd.DataFrame)"])
    path = os.path.join(RES, f"r4_modin_fidelity_{engine}_{args.grid}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
