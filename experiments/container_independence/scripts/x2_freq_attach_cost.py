"""x2 (4/5): what does it cost to give every slice a time index *with freq*?

The current VectorizedDF re-infers freq per slice (_enforce_index_freq ->
pd.infer_freq) whenever the slice comes out of groupby without one. That happens
for MultiIndexes built with from_tuples/from_arrays, but not for from_product
over a date_range. A narwhals/columns path always loses freq and has to attach it
again. Compared per slice, in the same process:

  sk_items_levelfreq   VectorizedDF.items on a frame whose time level has freq
  sk_items_nofreq      same data, MultiIndex rebuilt with from_arrays, so the
                       level has no freq and every slice runs pd.infer_freq
  attach_infer         DatetimeIndex(values) + pd.infer_freq (current approach)
  attach_validate      DatetimeIndex(values, freq=schema_freq) (pandas validates)
  attach_date_range    date_range(start, periods, freq) + array_equal check
  attach_grid          slice of one global regular grid + array_equal check
                       (protos/x2_index_recipe.py)

Usage::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x2_freq_attach_cost.py --grid dev
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from common import DEV_GRID, LARGE_GRID, bench, env_info, make_panel_fast

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    from sktime.datatypes._vectorize import prepare_VectorizedDF

    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    rows = []
    for n_series, n_time in grid:
        df = make_panel_fast(n_series, n_time)
        df_nf = df.copy()
        df_nf.index = pd.MultiIndex.from_arrays(
            [df.index.get_level_values(0), np.asarray(df.index.get_level_values(1))],
            names=df.index.names,
        )
        assert df.index.levels[-1].freq is not None
        assert df_nf.index.levels[-1].freq is None
        v1, X1 = prepare_VectorizedDF(df, iterate_as="Series", is_scitype="Panel")
        v2, X2 = prepare_VectorizedDF(df_nf, iterate_as="Series", is_scitype="Panel")

        tvals = df.index.get_level_values(1).asi8.reshape(n_series, n_time)
        unit = df.index.levels[-1].unit
        slices = [np.asarray(r).view(f"datetime64[{unit}]") for r in tvals]
        f = pd.tseries.frequencies.to_offset("D")
        g = pd.date_range(slices[0][0], periods=n_time, freq=f, unit=unit)

        def attach_infer():
            out = []
            for s in slices:
                ix = pd.DatetimeIndex(s)
                ix.freq = pd.infer_freq(ix)
                out.append(ix)
            return out

        def attach_validate():
            return [pd.DatetimeIndex(s, freq=f) for s in slices]

        def attach_date_range():
            out = []
            for s in slices:
                ix = pd.DatetimeIndex(s)
                c = pd.date_range(ix[0], periods=len(ix), freq=f, unit=ix.unit)
                out.append(c if np.array_equal(c.asi8, ix.asi8) else ix)
            return out

        def attach_grid():
            out = []
            for s in slices:
                ix = pd.DatetimeIndex(s)
                a = g.searchsorted(ix[0])
                c = g[a : a + len(ix)]
                out.append(c if np.array_equal(c.asi8, ix.asi8) else ix)
            return out

        methods = {
            "sk_items_levelfreq": lambda: list(v1.items(X=X1)),
            "sk_items_nofreq": lambda: list(v2.items(X=X2)),
            "attach_infer": attach_infer,
            "attach_validate": attach_validate,
            "attach_date_range": attach_date_range,
            "attach_grid": attach_grid,
        }
        for name in ["attach_infer", "attach_validate", "attach_date_range", "attach_grid"]:
            assert all(ix.freqstr == "D" for ix in methods[name]()), name
        assert all(x[2].index.freqstr == "D" for x in methods["sk_items_nofreq"]())

        t = dict.fromkeys(methods, float("inf"))
        rep = args.repeat if n_series * n_time <= 200_000 else max(2, args.repeat - 2)
        for _ in range(rep):
            for k, fn in methods.items():
                t[k] = min(t[k], bench(fn, repeat=1, warmup=0))
        row = {
            "n_series": n_series,
            "n_time": n_time,
            "t_ms": {k: round(v * 1e3, 3) for k, v in t.items()},
            "us_per_slice": {k: round(v * 1e6 / n_series, 2) for k, v in t.items()},
            "ratio_sk_nofreq/levelfreq": round(
                t["sk_items_nofreq"] / t["sk_items_levelfreq"], 3
            ),
            "ratio_infer/grid": round(t["attach_infer"] / t["attach_grid"], 2),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    tag = "pd" + pd.__version__.split(".")[0]
    path = os.path.join(RES, f"x2_freq_attach_cost_{tag}_{args.grid}.json")
    with open(path, "w") as fh:
        json.dump({"env": env_info(), "python": sys.version.split()[0], "rows": rows}, fh, indent=1)
    print(f"\npandas {pd.__version__}: microseconds per slice")
    ks = list(rows[0]["us_per_slice"])
    print(f"{'size':>12s} " + " ".join(f"{k:>19s}" for k in ks) + "  sk_nofreq/levelfreq")
    for r in rows:
        print(
            f"{r['n_series']:>6d}x{r['n_time']:<5d} "
            + " ".join(f"{r['us_per_slice'][k]:>19.1f}" for k in ks)
            + f"  {r['ratio_sk_nofreq/levelfreq']:>8.3f}"
        )
    print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
