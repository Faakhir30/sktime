"""x6 verifier: peak memory of hub ingress vs current, inputs READ FROM PARQUET.

Differences vs x6_mem.py: (a) the input is read from a parquet file in the child, so
the input buffers are laid out as a real reader produces them (separate column
buffers for polars/arrow), not derived from one pandas 2D block; (b) the expected
copy size is computed analytically from hub arrays that do not alias the source
(address-range check) and reported next to the measured VmHWM delta.

Run: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
       experiments/container_independence/scripts/x6_proto_hub_verify0_mem.py --grid dev
"""

import argparse
import gc
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ED, "protos"))
SCR = os.environ.get("X6V0_SCRATCH", "/tmp/claude-0/-home-user-sktime/"
                     "c9329e73-89f5-514d-ae5a-9180bf0e4a71/scratchpad/x6v0")


def _kb(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])


def child(case, path):
    import warnings

    warnings.simplefilter("ignore")
    import numpy as np
    import pandas as pd
    import polars as pl
    import pyarrow.parquet as pq

    import x6_hub as H
    from sktime.datatypes import convert_to
    from sktime.datatypes._vectorize import prepare_VectorizedDF

    src, kind = case.split(":")
    kw = {"key_cols": ["h0"], "time_col": "time"}
    if src == "pd":
        data = pd.read_parquet(path).set_index(["h0", "time"])
    elif src == "pl":
        data = pl.read_parquet(path)
    else:
        data = pq.read_table(path)

    def op():
        if kind == "hub":
            return H.from_native(data, **({} if src == "pd" else kw))
        if src == "pd":
            return prepare_VectorizedDF(data, iterate_as="Series", is_scitype="Panel")
        return prepare_VectorizedDF(data.to_pandas().set_index(["h0", "time"]),
                                    iterate_as="Series", is_scitype="Panel")

    # warm-up on a tiny slice of the same type
    small = data.iloc[:100] if src == "pd" else data.slice(0, 100)
    _ = (H.from_native(small, **({} if src == "pd" else kw)) if kind == "hub" else
         None)
    gc.collect()
    rss0 = _kb("VmRSS")
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")
    res = op()
    hwm = _kb("VmHWM")
    rec = {"case": case, "peak_delta_mb": (hwm - rss0) / 1024}
    if kind == "hub":
        hub = res
        # analytic: bytes of hub arrays not aliasing source buffers
        if src == "pd":
            srcs = [data.to_numpy(copy=False)]
        elif src == "pl":
            srcs = [data[c].to_numpy() for c in data.columns
                    if data.schema[c].is_numeric() or data.schema[c].is_temporal()]
        else:
            srcs = []
            for c in data.column_names:
                for ch in data.column(c).chunks:
                    try:
                        srcs.append(ch.to_numpy(zero_copy_only=True))
                    except Exception:  # noqa: BLE001
                        pass
        own = 0
        for arr in (hub.values, hub.time, hub.indptr):
            if not any(np.shares_memory(arr, s) for s in srcs):
                own += arr.nbytes
        rec["hub_owned_mb_analytic"] = own / 2**20
        rec["values_zero_copy"] = any(np.shares_memory(hub.values, s) for s in srcs)
    print(json.dumps(rec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--child", nargs=2)
    a = ap.parse_args()
    if a.child:
        child(*a.child)
        return
    import pandas as pd

    from common import env_info, make_panel_fast

    n = 10_000 if a.grid == "dev" else 50_000
    os.makedirs(SCR, exist_ok=True)
    rows = []
    for cols in (1, 4):
        path = os.path.join(SCR, f"x6v0_{n}x100_c{cols}.parquet")
        if not os.path.exists(path):
            make_panel_fast(n, 100, n_cols=cols).reset_index().to_parquet(path)
        for case in ["pd:current", "pd:hub", "pl:current", "pl:hub", "pa:current",
                     "pa:hub"]:
            reps = []
            for _ in range(3):
                o = subprocess.run([sys.executable, "-W", "ignore", __file__,
                                    "--child", case, path], capture_output=True,
                                   text=True)
                line = [x for x in o.stdout.splitlines() if x.startswith("{")]
                if not line:
                    print("FAIL", case, o.stderr[-800:])
                    break
                reps.append(json.loads(line[-1]))
            reps.sort(key=lambda r: r["peak_delta_mb"])
            r = reps[len(reps) // 2]
            r["cols"] = cols
            r["reps"] = [round(x["peak_delta_mb"], 1) for x in reps]
            rows.append(r)
            print(json.dumps(r), flush=True)
    tag = "pd" + pd.__version__.split(".")[0]
    with open(os.path.join(ED, "results", f"x6_proto_hub_verify0_mem_{tag}_{a.grid}.json"),
              "w") as f:
        json.dump({"env": env_info(), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
