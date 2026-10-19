"""g3: combined-architecture waterfall C0..C5, end to end, with equality per step.

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/g3_stack_bench.py \
        --sizes 100x100,1000x100 --est naive,poly --cont pd,pl,pl_lazy \
        --out experiments/container_independence/results/g3_stack_part.json

    # one fresh-process memory case:
    ... g3_stack_bench.py --mem 1000x100:naive:pl:3 [--no-g2]

Workload per (size, estimator, container, level): fit(y, fh=[1,2,3]) + predict(),
fresh estimator (remember_data=True set explicitly), timed from the native input to
the native output. Containers: pandas pd-multiindex ("pd"), eager polars in the
sktime polars_panel layout ("pl"), polars LazyFrame from ``pl.scan_parquet``
("pl_lazy"; sktime has no lazy Series/Panel mtype, so every level collects once at
the boundary, inside the timed region). Levels C0..C5 are interleaved round-robin
within each repeat in one process; min over repeats; ratios are vs C0 and vs the
previous level. Peak RSS: VmHWM reset via /proc/self/clear_refs before each run,
delta vs RSS before the run (in-process); ``--mem`` gives the fresh-process number.
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ED)
sys.path.insert(0, os.path.join(ED, "protos"))

from common import env_info, make_panel_fast  # noqa: E402

FH = [1, 2, 3]
SCRATCH = os.environ.get(
    "G3_SCRATCH",
    "/tmp/claude-0/-home-user-sktime/c9329e73-89f5-514d-ae5a-9180bf0e4a71/scratchpad",
)
LEVELS = [0, 1, 2, 3, 4, 5]


def _status_kb(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1])
    return -1


def _reset_hwm():
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def make_inputs(n, T, conts):
    from sktime.datatypes import convert_to

    y = make_panel_fast(n, T)
    ins = {}
    if "pd" in conts:
        ins["pd"] = y
    if "pl" in conts or "pl_lazy" in conts:
        ypl = convert_to(y, "polars_panel")
        if "pl" in conts:
            ins["pl"] = ypl
        if "pl_lazy" in conts:
            path = os.path.join(SCRATCH, f"g3_stack_{n}x{T}.parquet")
            if not os.path.exists(path):
                ypl.write_parquet(path)
            ins["pl_lazy"] = path
    return ins


def get_input(cont, inp):
    if cont == "pl_lazy":
        import polars as pl

        # sktime has no lazy Panel mtype: collect once at the boundary (all levels)
        return pl.scan_parquet(inp).collect()
    return inp


def run_once(g3, level, est, cont, inp):
    g3.configure(level)
    f = g3.make_estimator(est, level)
    gc.collect()
    hwm_ok = _reset_hwm()
    rss0 = _status_kb("VmRSS")
    t0 = time.perf_counter()
    y = get_input(cont, inp)
    f.fit(y, fh=FH)
    p = f.predict()
    dt = time.perf_counter() - t0
    peak = (_status_kb("VmHWM") - rss0) / 1024 if hwm_ok else None
    g3.configure(0)
    return p, f, dt, peak


def to_pd(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    import polars as pl

    if isinstance(obj, pl.LazyFrame):
        obj = obj.collect()
    # independent of sktime / proto converters
    df = obj.to_pandas()
    idx = [c for c in df.columns if c.startswith("__index__")]
    df = df.set_index(idx)
    df.index.names = [c[9:] for c in idx]
    return df


def compare(p_ref, f_ref, p, f, level):
    r = {"type_same": type(p_ref) is type(p)}
    a, b = to_pd(p_ref), to_pd(p)
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        r["pred_exact"] = True
    except AssertionError as e:
        r["pred_exact"] = False
        r["pred_exact_msg"] = str(e).splitlines()[0][:200]
    try:
        ok_idx = a.index.equals(b.index) and list(a.columns) == list(b.columns)
        va, vb = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
        rel = np.max(np.abs(va - vb) / np.maximum(np.abs(va), 1e-300)) if va.size else 0
        r["pred_close_1e9"] = bool(ok_idx and np.allclose(va, vb, rtol=1e-9, atol=0))
        r["pred_max_rel_err"] = float(rel)
    except Exception as e:  # noqa: BLE001
        r["pred_close_1e9"] = False
        r["pred_err"] = repr(e)[:200]
    if isinstance(p_ref, pd.DataFrame) and isinstance(p_ref.index, pd.MultiIndex):
        r["time_level_freq_same"] = (
            getattr(p_ref.index.levels[-1], "freq", None)
            == getattr(p.index.levels[-1], "freq", None)
        )
    else:
        try:
            r["schema_same"] = p_ref.schema == p.schema
        except Exception:  # noqa: BLE001
            r["schema_same"] = None
    c0, c1 = f_ref.cutoff, f.cutoff
    r["cutoff_same"] = bool(c0.equals(c1) and getattr(c0, "freq", None)
                            == getattr(c1, "freq", None))
    if level < 5:
        F0, F1 = f_ref.forecasters_, f.forecasters_
        r["forecasters_index_same"] = bool(F0.index.equals(F1.index)
                                           and F0.shape == F1.shape
                                           and list(F0.columns) == list(F1.columns))
        bad = []
        for pos in (0, F0.shape[0] - 1):
            e0, e1 = F0.iloc[pos, 0], F1.iloc[pos, 0]
            if type(e0) is not type(e1):
                bad.append("type")
                continue
            if not (e0._cutoff.equals(e1._cutoff) and e0._cutoff.freq == e1._cutoff.freq):
                bad.append("_cutoff")
            try:
                if isinstance(e0._y, pd.Series):
                    pd.testing.assert_series_equal(e0._y, e1._y, check_exact=True)
                else:
                    pd.testing.assert_frame_equal(e0._y, e1._y, check_exact=True)
                if getattr(e0._y.index, "freq", None) != getattr(e1._y.index, "freq",
                                                                 None):
                    bad.append("_y.freq")
            except AssertionError:
                bad.append("_y")
            if e0._y_metadata != e1._y_metadata:
                bad.append("_y_metadata")
            if e0._state != e1._state or e0._is_vectorized != e1._is_vectorized:
                bad.append("_state")
            if not np.array_equal(np.asarray(e0._fh), np.asarray(e1._fh)):
                bad.append("_fh")
        r["slice_state_mismatch"] = bad
    else:
        r["forecasters_index_same"] = None  # g2: no per-instance forecasters_ grid
        r["slice_state_mismatch"] = None
    return r


def reps_for(n, T):
    if n <= 100:
        return 3
    if n <= 1000 and T <= 100:
        return 2
    return 1


def bench(args):
    import g3_stack as g3

    g3._g2()  # register np_grouped before any level is configured
    sizes = [tuple(int(x) for x in s.split("x")) for s in args.sizes.split(",")]
    ests = args.est.split(",")
    conts = args.cont.split(",")
    levels = [int(x) for x in args.levels.split(",")]
    rows = []

    # warm-up (imports, caches) for every (est, cont, level) on a tiny panel
    win = make_inputs(20, 30, conts)
    for est in ests:
        for cont in conts:
            for lv in levels:
                run_once(g3, lv, est, cont, win[cont])
    print("warm-up done", flush=True)

    for n, T in sizes:
        ins = make_inputs(n, T, conts)
        reps = args.reps or reps_for(n, T)
        for est in ests:
            for cont in conts:
                times = {lv: [] for lv in levels}
                peaks = {lv: [] for lv in levels}
                outs = {}
                stats = {}
                for rep in range(reps):
                    for lv in levels:
                        g3.reset_stats()
                        p, f, dt, pk = run_once(g3, lv, est, cont, ins[cont])
                        times[lv].append(dt)
                        peaks[lv].append(pk)
                        if rep == 0:
                            stats[lv] = dict(g3.STATS)
                            if lv == levels[0]:
                                outs["ref"] = (p, f)
                            else:  # compare now, keep only the reference alive
                                outs[lv] = compare(*outs["ref"], p, f, lv)
                        del p, f
                        gc.collect()
                rec_lv = {}
                prev = None
                for lv in levels:
                    t = min(times[lv])
                    eq = outs.get(lv) if lv != levels[0] else None
                    rec_lv[f"C{lv}"] = {
                        "time_s": t,
                        "times_s": times[lv],
                        "peak_rss_delta_mb_inproc": max(x for x in peaks[lv]
                                                        if x is not None)
                        if any(x is not None for x in peaks[lv]) else None,
                        "x_vs_C0": min(times[levels[0]]) / t,
                        "x_vs_prev": (prev / t) if prev else None,
                        "equality_vs_C0": eq,
                        "stats": stats[lv],
                    }
                    prev = t
                row = {"n_series": n, "n_time": T, "est": est, "cont": cont,
                       "reps": reps, "levels": rec_lv}
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("n_series", "n_time", "est",
                                                       "cont")}),
                      " ".join(f"C{lv}={rec_lv[f'C{lv}']['time_s']:.3f}s"
                               f"(x{rec_lv[f'C{lv}']['x_vs_C0']:.2f})"
                               for lv in levels), flush=True)
                for lv in levels[1:]:
                    eq = rec_lv[f"C{lv}"]["equality_vs_C0"]
                    print(f"   C{lv} eq:", {k: v for k, v in eq.items()
                                            if k not in ("pred_exact_msg",)},
                          flush=True)
                del outs
                gc.collect()
                if args.out:
                    _dump(args.out, rows)
    return rows


def _dump(path, rows):
    with open(path, "w") as f:
        json.dump({"env": env_info(), "argv": sys.argv, "rows": rows}, f, indent=1,
                  default=str)


def mem_case(spec, no_g2):
    """Fresh process: one run of one (size, est, cont, level); prints JSON."""
    size, est, cont, lv = spec.split(":")
    n, T = (int(x) for x in size.split("x"))
    lv = int(lv)
    import g3_stack as g3

    if not no_g2:
        g3._g2()
    ins = make_inputs(n, T, [cont])
    gc.collect()
    rss_base = _status_kb("VmRSS")
    p, f, dt, pk = run_once(g3, lv, est, cont, ins[cont])
    print(json.dumps({"size": size, "est": est, "cont": cont, "level": lv,
                      "no_g2": no_g2, "time_s": dt, "peak_rss_delta_mb": pk,
                      "rss_base_mb": rss_base / 1024,
                      "maxrss_mb": __import__("resource").getrusage(
                          __import__("resource").RUSAGE_SELF).ru_maxrss / 1024}))


def mem_sweep(args):
    sizes = args.sizes.split(",")
    out = []
    for size in sizes:
        for est in args.est.split(","):
            for cont in args.cont.split(","):
                for lv in [int(x) for x in args.levels.split(",")]:
                    variants = [False]
                    if lv == 0 and args.control_no_g2:
                        variants = [False, True]
                    for no_g2 in variants:
                        cmd = [sys.executable, "-W", "ignore", __file__, "--mem",
                               f"{size}:{est}:{cont}:{lv}"] + (["--no-g2"] if no_g2
                                                               else [])
                        res = subprocess.run(cmd, capture_output=True, text=True,
                                             env=os.environ.copy())
                        line = [x for x in res.stdout.splitlines() if x.startswith("{")]
                        if not line:
                            print("FAILED", cmd, res.stderr[-2000:], flush=True)
                            continue
                        rec = json.loads(line[-1])
                        out.append(rec)
                        print(rec, flush=True)
                        if args.out:
                            with open(args.out, "w") as f:
                                json.dump({"env": env_info(), "argv": sys.argv,
                                           "mem": out}, f, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100x100")
    ap.add_argument("--est", default="naive,poly")
    ap.add_argument("--cont", default="pd,pl,pl_lazy")
    ap.add_argument("--levels", default="0,1,2,3,4,5")
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--mem", default="")
    ap.add_argument("--no-g2", action="store_true")
    ap.add_argument("--mem-sweep", action="store_true")
    ap.add_argument("--control-no-g2", action="store_true")
    a = ap.parse_args()
    if a.mem:
        mem_case(a.mem, a.no_g2)
    elif a.mem_sweep:
        mem_sweep(a)
    else:
        bench(a)
