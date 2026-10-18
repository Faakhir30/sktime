"""a1: cost of the estimator-grid construction (contract C1) - pandas pivot vs EstimatorGrid.

Times exactly the post-processing that ``VectorizedDF.vectorize_est`` does on the
``(row_key, col_key, estimator)`` triples returned by ``parallelize`` (pd.DataFrame(long)
-> unique -> pivot -> 2x reindex -> clear names -> MultiIndex.from_tuples), against
``EstimatorGrid.from_long`` (protos/a1_estimator_grid.py), plus the consumer side:
iterating cells row-major (``iterrows`` + ``.values`` today vs ``grid.cells()``) and the
get_fitted_params key loop (``product(index, columns)`` + ``.loc`` vs ``grid.items()``).

Estimators are cheap placeholder objects (``object()``), so only container cost is timed.
Row keys: Panel -> str instance names; Hierarchical -> 2-tuples. n_cols in {1, 3}.

Run (from WT root):
    PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
        $EXP_PY -W ignore experiments/container_independence/scripts/a1_grid_bench.py --grid dev
Output: results/a1_grid_bench_<grid>.json
"""

import argparse
import itertools
import json
import warnings

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
from a1_estimator_grid import EstimatorGrid  # noqa: E402
from common import DEV_GRID, LARGE_GRID, bench, env_info  # noqa: E402


def pandas_pivot(ret):
    """Verbatim logic of VectorizedDF.vectorize_est, return_type='pd.DataFrame'."""
    df_long = pd.DataFrame(ret)
    cols_right_order = df_long.loc[:, 1].unique()
    rows_right_order = df_long.loc[:, 0].unique()
    df = df_long.pivot(index=0, columns=1, values=2)
    df = df.reindex(cols_right_order, axis=1)
    df = df.reindex(rows_right_order, axis=0)
    df.index.names = [None] * len(df.index.names)
    df.columns.name = None
    try:
        df.index = pd.MultiIndex.from_tuples(df.index)
    except (TypeError, ValueError):
        pass
    return df


def consume_pd(df):
    cells = [c for _, row in df.iterrows() for c in row.values]
    keys = [(ix, col, df.loc[ix, col]) for ix, col in itertools.product(df.index, df.columns)]
    return len(cells) + len(keys)


def consume_grid(g):
    cells = g.cells()
    keys = list(g.items())
    return len(cells) + len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])
    a = ap.parse_args()
    grid = DEV_GRID if a.grid == "dev" else LARGE_GRID
    out = {"env": env_info(), "grid": a.grid, "rows": []}
    for n_series, _ in grid:
        for kind, n_cols in [("panel", 1), ("hier", 1), ("panel", 3)]:
            if kind == "panel":
                rkeys = [f"s{i}" for i in range(n_series)]
            else:
                rkeys = [(f"a{i % 10}", f"b{i}") for i in range(n_series)]
            ckeys = ["forecasters"] if n_cols == 1 else [f"c{j}" for j in range(n_cols)]
            ret = [(r, c, object()) for r in rkeys for c in ckeys]
            df = pandas_pivot(ret)
            g = EstimatorGrid.from_long(ret)
            # equivalence: same cells in same order, same keys
            assert [c for _, row in df.iterrows() for c in row.values] == g.cells()
            assert list(df.columns) == g.col_keys
            assert list(df.index) == g.row_keys
            rep = 3 if n_series >= 10_000 else 5
            t_pd = bench(lambda: pandas_pivot(ret), repeat=rep)
            t_g = bench(lambda: EstimatorGrid.from_long(ret), repeat=rep)
            t_cpd = bench(lambda: consume_pd(df), repeat=rep)
            t_cg = bench(lambda: consume_grid(g), repeat=rep)
            row = dict(
                n_series=n_series, kind=kind, n_cols=n_cols,
                build_pandas_s=t_pd, build_grid_s=t_g, build_ratio=t_pd / t_g,
                consume_pandas_s=t_cpd, consume_grid_s=t_cg, consume_ratio=t_cpd / t_cg,
            )
            out["rows"].append(row)
            print(
                f"{kind:5s} n={n_series:>6} cols={n_cols}  build pd={t_pd*1e3:8.2f}ms "
                f"grid={t_g*1e3:7.2f}ms ({row['build_ratio']:.0f}x)  consume pd={t_cpd*1e3:9.2f}ms "
                f"grid={t_cg*1e3:7.2f}ms ({row['consume_ratio']:.0f}x)",
                flush=True,
            )
    with open(
        f"experiments/container_independence/results/a1_grid_bench_{a.grid}.json", "w"
    ) as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
