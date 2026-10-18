"""x9 verifier: vectorized monotonic check — equivalence on adversarial inputs and the
END-TO-END effect on convert_to (monkeypatched), instead of only the check itself.

Run::
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x9_cheap_paths_verify0_mono.py --grid dev
"""
import argparse
import sys
import warnings

warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from common import bench  # noqa: E402

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from x9_arrow_convert import make_pl_panel  # noqa: E402

import sktime.datatypes._adapter.polars as adp  # noqa: E402
from sktime.datatypes import convert_to  # noqa: E402

ORIG = adp.is_monotonically_increasing


def is_mono_vec(obj):
    cols = [c for c in obj.columns if c.startswith("__index__")]
    if len(cols) <= 1:
        return obj[cols[0]].is_sorted() if cols else True
    t = pl.col(cols[-1])
    ok = (t >= t.shift(1).over(cols[:-1])).fill_null(True)
    return bool(obj.select(ok.all()).item())


ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="dev")
a = ap.parse_args()

# ---- 1) equivalence on random / adversarial inputs
rng = np.random.default_rng(1)
mismatch = []
n_cases = 0
for trial in range(300):
    ns = int(rng.integers(1, 8))
    nt = int(rng.integers(1, 12))
    nlev = int(rng.integers(1, 3))  # instance levels
    d = {}
    keys = rng.integers(0, ns, size=ns * nt)
    for lv in range(nlev):
        d[f"__index__h{lv}"] = [f"k{(k + lv) % ns}" for k in keys]
    tt = rng.integers(0, 20, size=ns * nt)
    mode = trial % 4
    if mode == 0:  # sorted within groups
        df = pl.DataFrame(d).with_columns(__index__time=pl.Series(tt)).sort(
            [f"__index__h{lv}" for lv in range(nlev)] + ["__index__time"])
    elif mode == 1:  # random
        df = pl.DataFrame(d).with_columns(__index__time=pl.Series(tt))
    elif mode == 2:  # sorted within group but groups interleaved
        df = pl.DataFrame(d).with_columns(__index__time=pl.Series(tt)).sort("__index__time")
    else:  # with nulls in time
        tn = [None if rng.random() < 0.2 else int(x) for x in np.sort(tt)]
        df = pl.DataFrame(d).with_columns(__index__time=pl.Series(tn, dtype=pl.Int64))
    df = df.with_columns(c0=pl.Series(rng.normal(size=df.height)))
    n_cases += 1
    r1, r2 = ORIG(df), is_mono_vec(df)
    if r1 != r2:
        mismatch.append((trial, mode, nlev, r1, r2))
print(f"equivalence: {n_cases} cases, {len(mismatch)} mismatches; by mode: "
      f"{ {m: sum(1 for x in mismatch if x[1]==m) for m in range(4)} }")
for m in mismatch[:5]:
    print("  mismatch (trial,mode,nlev,orig,vec):", m)

# ---- 2) end-to-end convert_to with the check patched out vs vectorized
ns, nt = (10_000, 50) if a.grid == "dev" else (50_000, 50)
df = make_pl_panel(ns, nt)
convert_to(make_pl_panel(3, 5), "pd-multiindex")


def run_with(fn):
    adp.is_monotonically_increasing = fn
    try:
        return convert_to(df, "pd-multiindex")
    finally:
        adp.is_monotonically_increasing = ORIG


r_o = run_with(ORIG)
r_v = run_with(is_mono_vec)
assert r_o.equals(r_v)
times = {}
for rep in range(2):  # interleave
    times.setdefault("orig", []).append(bench(lambda: run_with(ORIG), repeat=2))
    times.setdefault("vec", []).append(bench(lambda: run_with(is_mono_vec), repeat=3))
    times.setdefault("nocheck", []).append(bench(lambda: run_with(lambda o: True), repeat=3))
t = {k: min(v) for k, v in times.items()}
print(f"convert_to {ns}x{nt}: orig {t['orig']*1e3:.1f} ms | vectorized check "
      f"{t['vec']*1e3:.1f} ms | no check {t['nocheck']*1e3:.1f} ms | "
      f"end-to-end speedup x{t['orig']/t['vec']:.1f}")
