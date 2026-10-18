"""Verifier: is the canonical-Arrow speedup due to Arrow, or to offsets+numpy views?
Compares on the same data, interleaved in one process:
  A pandas MI groupby loop (status quo)
  B pandas MI -> index codes -> np.split (NO arrow)            [pandas-native offsets]
  C polars -> pa.table capsule -> dict-encode -> np.split     [r4 canonical arrow]
  D polars direct: rle on key col -> to_numpy -> np.split     [no pyarrow]
  E C + per-instance pd.Series construction w/ DatetimeIndex  [cost for pandas-inner estimators]
  F polars input SHUFFLED, r4 arrow_runs(assume_contiguous=True) -> correctness check
"""
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd, polars as pl, pyarrow as pa
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from common import DEV_GRID, LARGE_GRID, bench, make_panel_fast
from r4_canonical_arrow import arrow_runs, to_arrow
def f(a): return float(a.mean() + a[-1])
ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="dev", choices=["dev","large"])
grid = DEV_GRID if ap.parse_args().grid == "dev" else LARGE_GRID
rows = []
for ns, nt in grid:
    pdf = make_panel_fast(ns, nt); flat = pdf.reset_index()[["h0","time","c0"]]; pldf = pl.from_pandas(flat)
    ref = {k: f(g["c0"].to_numpy()) for k, g in pdf.groupby(level=0, sort=False)}
    def A(): return {k: f(g["c0"].to_numpy()) for k, g in pdf.groupby(level=0, sort=False)}
    def B():
        codes = pdf.index.codes[0]; lv = pdf.index.levels[0]; v = pdf["c0"].to_numpy()
        b = np.flatnonzero(codes[1:] != codes[:-1]) + 1; st = np.r_[0, b]
        return {k: f(a) for k, a in zip(lv[codes[st]], np.split(v, b))}
    def C(): return {k: f(a) for k, a in arrow_runs(to_arrow(pldf))}
    def D():
        r = pldf.select(pl.col("h0").rle()).unnest("h0")
        lens = r["len"].to_numpy(); b = np.cumsum(lens)[:-1]
        return {k: f(a) for k, a in zip(r["value"].to_list(), np.split(pldf["c0"].to_numpy(), b))}
    tn = pldf["time"].to_numpy()
    def E():
        tbl = to_arrow(pldf); t = tbl["time"].to_numpy(); out = {}
        keys = tbl["h0"].combine_chunks(); de = pa.compute.dictionary_encode(keys)
        enc = de.indices.to_numpy(zero_copy_only=False); b = np.flatnonzero(enc[1:] != enc[:-1]) + 1
        st = np.r_[0, b]; uq = de.dictionary.to_pylist()
        for k, a, ti in zip([uq[i] for i in enc[st]], np.split(tbl["c0"].to_numpy(), b), np.split(t, b)):
            s = pd.Series(a, index=pd.DatetimeIndex(ti), name="c0"); out[k] = f(s.to_numpy())
        return out
    res = {}
    for nm, fn in [("A_pd_groupby", A), ("B_pd_codes_split", B), ("C_arrow_canon", C), ("D_polars_rle", D), ("E_arrow+pd.Series", E)]:
        got0 = fn(); ok = all(np.isclose(ref[k], got0[k]) for k in ref)
        res[nm] = (bench(fn, repeat=3), ok)
    base = res["A_pd_groupby"][0]
    sh = pldf.sample(fraction=1.0, shuffle=True, seed=0)
    got = {}; nk = 0
    for k, a in arrow_runs(to_arrow(sh)): nk += 1; got[k] = f(a)
    wrong = sum(not np.isclose(ref[k], got.get(k, np.nan)) for k in ref)
    row = {"n_series": ns, "n_time": nt, **{k: {"ms": v[0]*1e3, "ratio": v[0]/base, "correct": v[1]} for k, v in res.items()},
           "shuffled_contiguous_assumption": {"runs_yielded": nk, "wrong_keys": int(wrong), "n_keys": ns}}
    print(json.dumps(row)); rows.append(row)
json.dump(rows, open(os.path.join(os.path.dirname(__file__), "..", "results", "r4_dispatch_arch_verify0_offsets_dev.json"), "w"), indent=1)
