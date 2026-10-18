"""x4 verify0: adversarial correctness checks for protos/x4_adapter.py.

Independent of x4_test_adapter.py: builds inputs that the original suite never
uses (it only used make_panel: equal length, sorted, contiguous, string keys)
and compares, with strict pandas.testing equality,

  P: VectorizedDF2(target="pandas")  vs current sktime VectorizedDF
  L: VectorizedDF2 on native polars   vs current sktime VectorizedDF on the
     equivalent pandas frame (slices converted to pandas only for comparison)
  C: adapter_get_cutoff(polars)       vs sktime get_cutoff(pandas)

Cases: unsorted/non-contiguous rows, unequal lengths, ragged end times,
int instance keys, Hierarchical->Panel iteration, iterate_cols=True,
unnamed index levels, NaNs, empty frame, single instance.

Run (worktree root):
  PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
    $EXP_PY -W ignore experiments/container_independence/scripts/x4_proto_adapter_verify0_adversarial.py
"""

import json
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from sktime.datatypes import convert_to, get_cutoff  # noqa: E402
from sktime.datatypes._adapter.polars import convert_polars_to_pandas  # noqa: E402
from sktime.datatypes._vectorize import prepare_VectorizedDF  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
from x4_adapter import adapter_get_cutoff, prepare_VectorizedDF2  # noqa: E402

ED = os.path.join(os.path.dirname(__file__), "..")
OUT = {"pandas_version": pd.__version__, "polars_version": pl.__version__, "results": []}


def rec(name, ok, detail=""):
    OUT["results"].append({"name": name, "ok": ok, "detail": str(detail)[:600]})
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -- {str(detail)[:400]}"))


def mk(lengths, keys=None, n_cols=2, levels=1, starts=None, names=None, nan=False, seed=1):
    """Build a pandas multiindex frame with given per-instance lengths."""
    rng = np.random.default_rng(seed)
    tups = []
    for i, L in enumerate(lengths):
        k = keys[i] if keys is not None else f"s{i}"
        st = pd.Timestamp("2000-01-01") + pd.Timedelta(days=(starts[i] if starts else 0))
        t = pd.date_range(st, periods=L, freq="D")
        kk = k if isinstance(k, tuple) else (k,)
        tups += [(*kk, ti) for ti in t]
    if names is None:
        names = [f"h{j}" for j in range(levels)] + ["time"]
    ix = pd.MultiIndex.from_tuples(tups, names=names)
    vals = rng.standard_normal((len(ix), n_cols))
    if nan:
        vals[::7, 0] = np.nan
    return pd.DataFrame(vals, index=ix, columns=[f"c{j}" for j in range(n_cols)])


def to_pl(X):
    """pandas multiindex -> sktime polars convention, preserving row order."""
    phys = [f"__index__{n}" if n is not None else f"__index__{i}" for i, n in enumerate(X.index.names)]
    Xr = X.copy()
    Xr.index = Xr.index.set_names(phys)
    return pl.from_pandas(Xr.reset_index())


def mk_int_time(lengths, n_cols=2):
    X = mk(lengths, n_cols=n_cols)
    lv0 = X.index.get_level_values(0)
    t = np.concatenate([np.arange(L) for L in lengths])
    X.index = pd.MultiIndex.from_arrays([lv0, t], names=["h0", "time"])
    return X


def shuffle_rows(X, seed=0):
    perm = np.random.default_rng(seed).permutation(len(X))
    return X.iloc[perm]


def interleave(X):
    """Non-contiguous instances, time still increasing within instance."""
    t = X.index.get_level_values(-1)
    order = np.lexsort((X.index.get_level_values(0).astype(str), t))
    return X.iloc[order]


def cmp_pandas(tag, X, sci, it_as, it_cols):
    v0, X0 = prepare_VectorizedDF(X, iterate_as=it_as, is_scitype=sci, iterate_cols=it_cols)
    v1, X1 = prepare_VectorizedDF2(
        X, iterate_as=it_as, is_scitype=sci, iterate_cols=it_cols, target="pandas"
    )
    s0 = list(v0.items(X0))
    s1 = list(v1.items(X1))
    assert len(s0) == len(s1)
    for a, b in zip(s0, s1):
        assert a[0] == b[0] and a[1] == b[1]
        pd.testing.assert_frame_equal(a[2], b[2], check_freq=True)
    assert v0.freq == v1.freq, (v0.freq, v1.freq)
    r0 = v0.reconstruct([s * 2 for s in v0.as_list(X0)], convert_back=True)
    r1 = v1.reconstruct([s * 2 for s in v1.as_list(X1)], convert_back=True)
    pd.testing.assert_frame_equal(r0, r1)


def cmp_polars(tag, X, sci, it_as, it_cols):
    """Native polars iteration vs current pandas path on the same data."""
    pl_mtype = {"Panel": "polars_panel", "Hierarchical": "polars_hierarchical"}[sci]
    # build the polars frame from the *same row order* as X (no sorting)
    Xpl = to_pl(X)
    v0, X0 = prepare_VectorizedDF(X, iterate_as=it_as, is_scitype=sci, iterate_cols=it_cols)
    v1, X1 = prepare_VectorizedDF2(
        Xpl, iterate_as=it_as, is_scitype=sci, iterate_cols=it_cols,
        target="native", X_orig_mtype=pl_mtype,
    )
    k0 = [(g, c) for g, c, _ in v0.items(X0)]
    k1 = [(g, c) for g, c, _ in v1.items(X1)]
    assert k0 == k1, ("keys differ", k0[:4], k1[:4])
    for (_, _, a), (_, _, b) in zip(v0.items(X0), v1.items(X1)):
        bp = convert_polars_to_pandas(b)
        pd.testing.assert_frame_equal(
            a, bp, check_freq=False, check_index_type=False, check_names=True
        )
    # reconstruct
    r0 = v0.reconstruct([s * 2 for s in v0.as_list(X0)])
    outs = []
    for s in v1.as_list(X1):
        vc = [c for c in s.columns if not c.startswith("__index__")]
        outs.append(s.with_columns([pl.col(c) * 2 for c in vc]))
    r1 = convert_polars_to_pandas(v1.reconstruct(outs))
    pd.testing.assert_frame_equal(r0, r1, check_freq=False, check_index_type=False)
    # freq agreement (VectorizedDF metadata)
    return {"freq_pd": v0.freq, "freq_pl": v1.freq}


def cmp_cutoff(tag, X, sci):
    pl_mtype = {"Panel": "polars_panel", "Hierarchical": "polars_hierarchical"}[sci]
    Xpl = to_pl(X)
    c_pd = get_cutoff(X, return_index=True)
    c_pl = adapter_get_cutoff(Xpl, return_index=True)
    c_pd_first = get_cutoff(X, return_index=True, reverse_order=True)
    c_pl_first = adapter_get_cutoff(Xpl, return_index=True, reverse_order=True)
    d = {
        "pd": repr(c_pd), "pl": repr(c_pl),
        "pd_first": repr(c_pd_first), "pl_first": repr(c_pl_first),
    }
    ok = c_pd.equals(c_pl) and c_pd.dtype == c_pl.dtype
    d["freq_equal"] = getattr(c_pd, "freq", None) == getattr(c_pl, "freq", None)
    ok_first = c_pd_first.equals(c_pl_first)
    return ok, ok_first, d


def run(tag, fn, *a):
    try:
        r = fn(tag, *a)
        rec(tag, True, r if r else "")
        return True
    except Exception as e:  # noqa: BLE001
        rec(tag, False, f"{type(e).__name__}: {e} | {traceback.format_exc()[-300:]}")
        return False


CASES = {
    "unequal_len": (mk([5, 9, 3, 12]), "Panel"),
    "ragged_start_end": (mk([5, 9, 3], starts=[0, 4, 10]), "Panel"),
    "int_keys": (mk([6, 6, 6], keys=[3, 1, 2]), "Panel"),
    "int_time_unequal": (mk_int_time([5, 8, 3]), "Panel"),
    "interleaved_instances": (interleave(mk([6, 6, 6])), "Panel"),
    "nan_values": (mk([10, 10], nan=True), "Panel"),
    "single_instance": (mk([8]), "Panel"),
    "unnamed_levels": (mk([5, 5], names=[None, None]), "Panel"),
    "hier_unequal": (
        mk([4, 6, 5, 7], keys=[("a", "x"), ("a", "y"), ("b", "x"), ("b", "z")], levels=2),
        "Hierarchical",
    ),
    "hier_int_keys": (
        mk([4, 4, 4], keys=[(2, 10), (1, 10), (1, 11)], levels=2), "Hierarchical"
    ),
    "hier_interleaved": (
        interleave(mk([5, 5, 5, 5], keys=[("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")], levels=2)),
        "Hierarchical",
    ),
}
ITER = {"Panel": ["Series"], "Hierarchical": ["Series", "Panel"]}


def main():
    summary = {"P": [0, 0], "L": [0, 0], "C": [0, 0], "Cfirst": [0, 0]}
    for name, (X, sci) in CASES.items():
        for it_as in ITER[sci]:
            for it_cols in [False, True]:
                t = f"{name}|{sci}->{it_as}|cols={it_cols}"
                summary["P"][1] += 1
                summary["P"][0] += run("P " + t, cmp_pandas, X, sci, it_as, it_cols)
                summary["L"][1] += 1
                summary["L"][0] += run("L " + t, cmp_polars, X, sci, it_as, it_cols)
        try:
            ok, ok_first, d = cmp_cutoff(name, X, sci)
        except Exception as e:  # noqa: BLE001
            ok, ok_first, d = False, False, f"{type(e).__name__}: {e}"
        summary["C"][1] += 1
        summary["C"][0] += ok
        summary["Cfirst"][1] += 1
        summary["Cfirst"][0] += ok_first
        rec(f"C cutoff {name}", ok, d)
        rec(f"Cfirst reverse_order {name}", ok_first, d)

    # empty frame
    Xe = mk([5, 5]).iloc[:0]
    for lbl, fn in [("P", cmp_pandas), ("L", cmp_polars)]:
        run(f"{lbl} empty|Panel->Series", fn, Xe, "Panel", "Series", False)

    # does current sktime accept the adversarial polars frames as valid mtypes?
    from sktime.datatypes import check_is_mtype

    validity = {}
    for name, (X, sci) in CASES.items():
        pl_mtype = {"Panel": "polars_panel", "Hierarchical": "polars_hierarchical"}[sci]
        Xpl = to_pl(X)
        validity[name] = {
            "pd_valid": bool(check_is_mtype(X, {"Panel": "pd-multiindex", "Hierarchical": "pd_multiindex_hier"}[sci])),
            "pl_valid": bool(check_is_mtype(Xpl, pl_mtype)),
        }
    OUT["mtype_validity"] = validity
    print("mtype validity", validity)
    OUT["summary"] = summary
    print("SUMMARY", summary)
    tag = "pd3" if pd.__version__.startswith("3") else "pd2"
    with open(os.path.join(ED, "results", f"x4_proto_adapter_verify0_adversarial_{tag}.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=str)


if __name__ == "__main__":
    main()
