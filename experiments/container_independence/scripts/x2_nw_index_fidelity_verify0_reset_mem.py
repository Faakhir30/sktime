"""x2 verify0: independent re-measurement of the reset_index copy / cost claim.

Different methods from scripts/x2_reset_index_cost.py:

memory  - one FRESH subprocess per config, glibc dynamic mmap threshold disabled
          (MALLOC_MMAP_THRESHOLD_=131072 so freed big buffers go back to the OS),
          kernel peak-RSS counter reset via /proc/self/clear_refs ("5") right before
          the call; reports VmHWM-delta (peak) and VmRSS-delta (retained, result alive).
          On pd2 env additionally memray (native allocations, incl. arrow) peak.
sharing - buffer-address overlap of EVERY block of the reset frame against EVERY
          block of the original (not just same-named column pairs), and the number
          of bytes in reset-frame blocks that overlap nothing (= newly allocated).
timing  - paired, alternating rounds (reset, items, reset, items, ...) with
          perf_counter_ns; reports the MEDIAN of per-round ratios (not min-of-repeats).

Usage::

    PYTHONPATH=.:experiments/container_independence $EXP_PY  -W ignore \
        experiments/container_independence/scripts/x2_nw_index_fidelity_verify0_reset_mem.py --grid dev
    ... $EXP_PY3 ... (same)
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def _status(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1]) / 1024  # MB
    return float("nan")


def _block_ranges(df):
    import numpy as np

    out = []
    for b in df._mgr.blocks:
        v = b.values
        if isinstance(v, np.ndarray):
            arr = v
        else:  # extension array (DatetimeArray, ArrowStringArray, ...)
            arr = getattr(v, "_ndarray", None)
            if arr is None:
                out.append((None, None, getattr(v, "nbytes", 0), str(b.dtype)))
                continue
        lo, hi = np.lib.array_utils.byte_bounds(arr)
        out.append((lo, hi, arr.nbytes, str(b.dtype)))
    return out


def child(n_series, n_time, n_cols, cow, mode):
    import gc
    import time

    import numpy as np
    import pandas as pd

    if cow:
        pd.options.mode.copy_on_write = True
    from common import make_panel_fast

    df = make_panel_fast(n_series, n_time, n_cols=n_cols)
    res = {"pandas": pd.__version__, "cow": bool(cow or pd.__version__ >= "3")}
    if mode == "mem":
        flat0 = df.reset_index()  # warm-up any lazy caches (MI codes/levels)
        del flat0
        gc.collect()
        open("/proc/self/clear_refs", "w").write("5")
        rss0 = _status("VmRSS")
        hwm0 = _status("VmHWM")
        flat = df.reset_index()
        rss1 = _status("VmRSS")
        hwm1 = _status("VmHWM")
        res.update(
            rss_retained_MB=round(rss1 - rss0, 2),
            hwm_peak_delta_MB=round(hwm1 - max(rss0, hwm0) if hwm1 >= rss0 else 0, 2),
            hwm_minus_rss0_MB=round(hwm1 - rss0, 2),
        )
        # sharing by buffer address over all blocks
        orig = _block_ranges(df)
        new = _block_ranges(flat)
        new_bytes = 0
        shared_blocks = []
        for lo, hi, nb, dt in new:
            ov = lo is not None and any(
                o_lo is not None and lo < o_hi and o_lo < hi for o_lo, o_hi, _, _ in orig
            )
            shared_blocks.append((dt, nb, bool(ov)))
            if not ov:
                new_bytes += nb
        res["blocks(dtype,nbytes,shared)"] = shared_blocks
        res["new_block_bytes_MB"] = round(new_bytes / 2**20, 2)
        res["data_MB"] = round(df.memory_usage(index=False).sum() / 2**20, 2)
        # per-column shares_memory against the whole original data block(s)
        per_col = []
        for c in df.columns:
            a = flat[c].to_numpy()
            per_col.append(
                any(np.shares_memory(a, b.values) for b in df._mgr.blocks
                    if isinstance(b.values, np.ndarray))
            )
        res["n_data_cols_shared"] = int(sum(per_col))
        res["n_data_cols"] = len(per_col)
        # memray (only if available)
        try:
            import tempfile

            import memray

            del flat
            gc.collect()
            p = tempfile.mktemp(suffix=".bin")
            with memray.Tracker(p, native_traces=False):
                flat = df.reset_index()
            r = memray.FileReader(p)
            res["memray_peak_MB"] = round(r.metadata.peak_memory / 2**20, 2)
            os.remove(p)
        except ImportError:
            res["memray_peak_MB"] = None
    else:  # timing: paired alternating rounds
        from sktime.datatypes._vectorize import prepare_VectorizedDF

        vec, X_mi = prepare_VectorizedDF(df, iterate_as="Series", is_scitype="Panel")
        flat = df.reset_index()

        def f_reset():
            return df.reset_index()

        def f_set():
            return flat.set_index(["h0", "time"])

        def f_items():
            return list(vec.items(X=X_mi))

        def f_lvl():
            return [g for _, g in df.groupby(level=0, sort=False)]

        fns = {"reset": f_reset, "set": f_set, "items": f_items, "gb_level": f_lvl}
        for f in fns.values():
            f()
        rounds = 7 if len(df) <= 200_000 else 4
        ts = {k: [] for k in fns}
        for _ in range(rounds):
            for k, f in fns.items():
                gc.collect()
                t0 = time.perf_counter_ns()
                f()
                ts[k].append((time.perf_counter_ns() - t0) / 1e6)
        med = lambda xs: float(np.median(xs))  # noqa: E731
        res.update({f"{k}_ms_median": round(med(v), 3) for k, v in ts.items()})
        res["ratio_reset/items_median_paired"] = round(
            med([a / b for a, b in zip(ts["reset"], ts["items"])]), 4
        )
        res["ratio_reset+set/items_median_paired"] = round(
            med([(a + s) / b for a, s, b in zip(ts["reset"], ts["set"], ts["items"])]), 4
        )
        res["ratio_reset/gb_level_median_paired"] = round(
            med([a / b for a, b in zip(ts["reset"], ts["gb_level"])]), 4
        )
    print("RESULT " + json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--child", nargs=5)
    args = ap.parse_args()
    if args.child:
        ns, nt, nc, cow, mode = args.child
        child(int(ns), int(nt), int(nc), cow == "1", mode)
        return
    import pandas as pd

    grid = [(1000, 100), (10000, 50)] if args.grid == "dev" else [(10000, 100), (50000, 50)]
    cows = [0] if pd.__version__ >= "3" else [0, 1]
    env = dict(os.environ, MALLOC_MMAP_THRESHOLD_="131072")
    rows = []
    for cow in cows:
        for ns, nt in grid:
            for nc in (1, 10):
                row = {"n_series": ns, "n_time": nt, "n_cols": nc, "rows": ns * nt}
                for mode in ("mem", "time"):
                    out = subprocess.run(
                        [sys.executable, "-W", "ignore", __file__, "--child",
                         str(ns), str(nt), str(nc), str(cow), mode],
                        env=env, capture_output=True, text=True, check=False,
                    )
                    line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
                    if not line:
                        row[mode + "_error"] = out.stderr[-800:]
                        continue
                    row.update(json.loads(line[0][7:]))
                rows.append(row)
                print(json.dumps(row), flush=True)
    tag = "pd" + pd.__version__.split(".")[0]
    path = os.path.join(RES, f"x2_nw_index_fidelity_verify0_reset_mem_{tag}_{args.grid}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
