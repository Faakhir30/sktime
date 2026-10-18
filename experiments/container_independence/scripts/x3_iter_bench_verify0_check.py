"""x3_iter_bench_verify0: independent checks of x3_iter_bench claims.

Different code paths from x3_iter_bench.py:
 (1) FULL slice-equivalence across ALL instances (values, time index, freq) for
     S1 (VectorizedDF.items), S2_iloc, S2_build, S5 (polars slice), S6, S8, S7a.
 (2) freq-less control built a different way (time level from raw datetime64
     values via set_index on a flat frame) -> slice freq before/after enforce, and
     cost of VectorizedDF.items on it vs the freq-carrying frame.
 (3) end-to-end: real sktime NaiveForecaster fit+predict on a panel (vectorized)
     vs n_inst * single-slice cost vs pure S1 iteration -> share of iteration.
 (4) symmetric accounting: polars-native S5 iteration INCLUDING the setup
     (rle offsets) and INCLUDING input conversion pd->polars, vs S1 incl. schema.

Usage: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
   experiments/container_independence/scripts/x3_iter_bench_verify0_check.py --grid dev
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from common import DEV_GRID, LARGE_GRID, bench, make_panel_fast, to_polars_long  # noqa

ED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def vec(X, freq=None):
    from sktime.datatypes._vectorize import VectorizedDF

    kw = {"freq": freq} if freq else {}
    return VectorizedDF(
        X, iterate_as="Series", is_scitype="Panel", X_orig_mtype="pd-multiindex", **kw
    )


def full_equiv(X):
    import narwhals as nw

    from protos.x3_strategies import (
        GroupedArray,
        PandasPositional,
        pa_offsets,
        pl_offsets,
    )

    v = vec(X, "D")
    ref = [inst for _, _, inst in v.items(X)]
    keys = list(X.index.names[:-1])
    pdpos = PandasPositional(X, 1)
    ga = GroupedArray.from_pandas_mi(X, 1)
    pl_df = to_polars_long(X)
    pk, poff, plen, _ = pl_offsets(pl_df, keys)
    body = pl_df.drop(keys)
    pa_t = pl_df.to_arrow()
    _, aoff, alen, _ = pa_offsets(pa_t, keys)
    abody = pa_t.drop_columns(keys)
    nwpd = nw.from_native(X.reset_index(), eager_only=True)
    res = {}

    def cmp_vals_time(name, it):
        ok_v = ok_t = True
        n = 0
        for r, (t, vals) in zip(ref, it):
            n += 1
            ok_v &= np.array_equal(r.to_numpy(), np.asarray(vals))
            ok_t &= np.array_equal(r.index.to_numpy(), np.asarray(t))
        res[name] = {"n": n, "values_equal": bool(ok_v), "time_equal": bool(ok_t)}

    def cmp_frames(name, it):
        eq = freq_eq = 0
        n = 0
        for r, f in zip(ref, it):
            n += 1
            eq += bool(r.equals(f) and r.index.equals(f.index))
            freq_eq += r.index.freq == f.index.freq
        res[name] = {"n": n, "frame_equal": eq, "freq_equal": int(freq_eq)}

    cmp_frames("S2_iloc", pdpos.iter_iloc())
    cmp_frames("S2_build", pdpos.iter_build())
    cmp_frames("S2_build_freqD", pdpos.iter_build(freq="D"))
    cmp_vals_time("S6", ga)
    cmp_vals_time(
        "S5",
        (
            (s["time"].to_numpy(), s.drop("time").to_numpy())
            for s in (body.slice(o, n) for o, n in zip(poff, plen))
        ),
    )
    cmp_vals_time(
        "S8",
        (
            (
                s.column("time").to_numpy(),
                np.column_stack(
                    [s.column(c).to_numpy() for c in s.column_names if c != "time"]
                ),
            )
            for s in (abody.slice(o, n) for o, n in zip(aoff, alen))
        ),
    )
    cmp_vals_time(
        "S7a",
        (
            (g["time"].to_numpy(), g.drop([*keys, "time"]).to_numpy())
            for _, g in nwpd.group_by(keys)
        ),
    )
    res["S1_ref_freq"] = str(ref[0].index.freq)
    res["n_ref"] = len(ref)
    return res


def nofreq_control(X):
    from sktime.datatypes._vectorize import _enforce_index_freq

    flat = X.reset_index()
    flat["time"] = flat["time"].to_numpy()  # raw datetime64, no freq anywhere
    Xnf = flat.set_index(["h0", "time"])
    g = next(iter(Xnf.groupby(level=0, sort=False)))[1].droplevel(0)
    out = {
        "slice_freq_before_enforce": str(g.index.freq),
        "slice_freq_after_enforce": str(_enforce_index_freq(g).index.freq),
    }
    v1, v2 = vec(X, "D"), vec(Xnf)
    _, _, i2 = next(iter(v2.items(Xnf)))
    out["items_slice_freq_nofreq_frame"] = str(i2.index.freq)

    def it(v, F):
        return lambda: sum(i.shape[0] for _, _, i in v.items(F))

    tf = bench(it(v1, X), repeat=3)
    tn = bench(it(v2, Xnf), repeat=3)
    tf2 = bench(it(v1, X), repeat=3)
    out["items_freq_s"] = min(tf, tf2)
    out["items_nofreq_s"] = tn
    out["ratio_nofreq_vs_freq"] = tn / min(tf, tf2)
    return out


def end_to_end(X):
    from sktime.forecasting.naive import NaiveForecaster

    v = vec(X, "D")
    _, _, inst = next(iter(v.items(X)))
    n = len(v.get_iter_indices()[0])

    def one():
        f = NaiveForecaster()
        f.fit(inst, fh=[1, 2])
        return f.predict()

    def panel():
        f = NaiveForecaster()
        f.fit(X, fh=[1, 2])
        return f.predict()

    def iter_only():
        return sum(i.shape[0] for _, _, i in v.items(X))

    t1 = bench(one, repeat=3)
    tp = bench(panel, repeat=2)
    ti = bench(iter_only, repeat=3)
    return {
        "n_inst": n,
        "single_slice_fit_predict_s": t1,
        "panel_fit_predict_s": tp,
        "n_x_single_s": n * t1,
        "panel_overhead_vs_n_x_single": tp / (n * t1),
        "S1_iteration_s (one pass)": ti,
        "iteration_share_of_panel_fit_predict (one pass)": ti / tp,
    }


def symmetric(X):
    import polars as pl  # noqa: F401

    from protos.x3_strategies import pl_offsets

    keys = list(X.index.names[:-1])

    def s1_incl_schema():
        v = vec(X, "D")
        return sum(i.shape[0] for _, _, i in v.items(X))

    def s5_incl_setup_from_polars(pl_df):
        def f():
            _, off, ln, _ = pl_offsets(pl_df, keys)
            body = pl_df.drop(keys)
            return sum(body.slice(o, n).height for o, n in zip(off, ln))

        return f

    def s5_incl_conv_from_pandas():
        pl_df = to_polars_long(X)
        return s5_incl_setup_from_polars(pl_df)()

    pl_df = to_polars_long(X)
    ops = {
        "S1 incl VectorizedDF schema": s1_incl_schema,
        "S5 incl rle setup (polars input)": s5_incl_setup_from_polars(pl_df),
        "S5 incl pd->polars conv + setup (pandas input)": s5_incl_conv_from_pandas,
    }
    best = {k: float("inf") for k in ops}
    for _ in range(3):
        for k, f in ops.items():
            best[k] = min(best[k], bench(f, repeat=1))
    base = best["S1 incl VectorizedDF schema"]
    return {"s": best, "ratio_vs_S1": {k: v / base for k, v in best.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-e2e", action="store_true")
    a = ap.parse_args()
    grid = DEV_GRID if a.grid == "dev" else LARGE_GRID
    if a.only:
        grid = [tuple(int(x) for x in a.only.split("x"))]
    out = {"pandas": pd.__version__, "cmd": " ".join(sys.argv), "points": []}
    for ns, nt in grid:
        X = make_panel_fast(ns, nt)
        rec = {"n_series": ns, "n_time": nt}
        rec["equiv"] = full_equiv(X)
        rec["nofreq"] = nofreq_control(X)
        rec["symmetric"] = symmetric(X)
        if not a.skip_e2e and ns <= 1000:
            rec["e2e"] = end_to_end(X)
        print(json.dumps(rec, indent=1, default=str), flush=True)
        out["points"].append(rec)
    tag = f"pd{pd.__version__.split('.')[0]}"
    with open(
        os.path.join(ED, "results", f"x3_iter_bench_verify0_check_{a.grid}_{tag}.json"),
        "w",
    ) as fh:
        json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
