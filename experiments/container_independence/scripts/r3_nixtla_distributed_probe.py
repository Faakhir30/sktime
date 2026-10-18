"""r3: probe Nixtla's "partition-and-delegate" path for lazy/distributed frames.

statsforecast routes non-(pandas|polars) inputs through fugue: the frame is
partitioned by id_col, each partition is materialized as a pandas DataFrame,
the numpy GroupedArray core runs per partition, and results come back in the
input engine with an output schema declared up front.

Checks: output type per input engine (dask, duckdb relation), whether the
result stays lazy, and timing vs the eager pandas path at small sizes.

    PYTHONPATH=.:experiments/container_independence $R3_PY -W ignore \
        experiments/container_independence/scripts/r3_nixtla_distributed_probe.py --grid dev
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from common import bench, env_info, make_panel_fast  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    args = ap.parse_args()
    sizes = [(100, 100), (1000, 100)] if args.grid == "dev" else [(1000, 500), (10_000, 100)]

    import dask
    import dask.dataframe as dd
    import duckdb
    import statsforecast
    from statsforecast import StatsForecast
    from statsforecast.models import Naive

    res = {"env": {**env_info(), "statsforecast": statsforecast.__version__,
                   "dask": dask.__version__, "duckdb": duckdb.__version__},
           "rows": [], "types": {}}
    sf = StatsForecast(models=[Naive()], freq="D", n_jobs=1)

    def flat(n, t):
        return (make_panel_fast(n, t).reset_index()
                .rename(columns={"h0": "unique_id", "time": "ds", "c0": "y"}))

    f = flat(20, 30)
    ddf = dd.from_pandas(f, npartitions=4)
    dask.config.set({"scheduler": "threads", "dataframe.shuffle.method": "tasks"})
    out = sf.forecast(h=3, df=ddf)
    res["types"]["dask"] = f"{type(out).__module__}.{type(out).__name__}"
    res["types"]["dask_is_lazy(has .compute)"] = hasattr(out, "compute")
    res["types"]["dask_computed_type"] = type(out.compute()).__name__
    try:
        con = duckdb.connect()
        rel = con.from_df(f)
        out_d = sf.forecast(h=3, df=rel)
        res["types"]["duckdb"] = f"{type(out_d).__module__}.{type(out_d).__name__}"
    except Exception as e:  # noqa: BLE001
        res["types"]["duckdb"] = f"ERROR {type(e).__name__}: {str(e)[:150]}"
    print("types:", json.dumps(res["types"], indent=1), flush=True)

    # p2p shuffle needs a distributed client; use task shuffle + threads (no cluster)
    with dask.config.set({"scheduler": "threads", "dataframe.shuffle.method": "tasks"}):
        for n, t in sizes:
            f = flat(n, t)
            ddf = dd.from_pandas(f, npartitions=4)
            row = {"n_series": n, "n_time": t}
            row["pandas_eager_s"] = bench(lambda: sf.forecast(h=5, df=f), repeat=3)
            row["dask_fugue_compute_s"] = bench(
                lambda: sf.forecast(h=5, df=ddf).compute(), repeat=3)
            row["ratio_dask_over_pandas"] = row["dask_fugue_compute_s"] / row["pandas_eager_s"]
            print(json.dumps(row), flush=True)
            res["rows"].append(row)

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"r3_nixtla_distributed_probe_{args.grid}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    sys.exit(main())
