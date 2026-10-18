"""x5: repro - narwhals pyarrow ``expr.agg().over(keys)`` silently misaligns rows.

narwhals 2.27.1 implements over() for pyarrow as group_by + ``pa.Table.join``;
pyarrow's hash join does not preserve left row order, so the broadcast values
are assigned to the wrong rows (no error). Also tests the join-based workaround
used in protos/x5_nw_forecasters.py (``_group_broadcast``).

Run:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x5_pa_over_bug.py [--grid dev|large]
"""

import argparse
import json
from pathlib import Path

import narwhals as nw
import numpy as np
import pyarrow as pa

from common import env_info
from protos.x5_nw_forecasters import _group_broadcast

ED = Path(__file__).resolve().parents[1]


def trial(n, T, seed):
    rng = np.random.default_rng(seed)
    g = np.repeat([f"s{i:06d}" for i in range(n)], T)
    v = rng.normal(size=n * T)
    t = pa.table({"g": g, "v": v})
    ref = t.to_pandas().groupby("g")["v"].transform("min").to_numpy()
    over = nw.from_native(t).with_columns(m=nw.col("v").min().over("g")).to_native()
    bad_over = int((over.column("m").to_numpy() != ref).sum())
    wk = _group_broadcast(nw.from_native(t), ["g"], [nw.col("v").min().alias("m")])
    wk = wk.to_native().to_pandas()
    ref_wk = wk.groupby("g")["v"].transform("min").to_numpy()
    bad_wk = int((wk["m"].to_numpy() != ref_wk).sum())
    return bad_over, bad_wk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--trials", type=int, default=10)
    a = ap.parse_args()
    sizes = [10, 100, 300, 500, 1000, 10000] if a.grid == "dev" else [1000, 10000, 50000]
    out = {"env": env_info(), "rows": []}
    for n in sizes:
        T = 100 if n <= 1000 else 20
        res = [trial(n, T, s) for s in range(a.trials)]
        row = {
            "n_groups": n,
            "rows": n * T,
            "trials": a.trials,
            "trials_with_wrong_rows_over": sum(r[0] > 0 for r in res),
            "mean_frac_wrong_over": float(np.mean([r[0] for r in res]) / (n * T)),
            "trials_with_wrong_rows_workaround": sum(r[1] > 0 for r in res),
        }
        out["rows"].append(row)
        print(row)
    p = ED / "results" / f"x5_pa_over_bug_{a.grid}.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
