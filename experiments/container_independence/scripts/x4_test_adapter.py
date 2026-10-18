"""x4: correctness tests for the adapter-based VectorizedDF2 prototype.

Run (from worktree root):
    PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
        $EXP_PY -W ignore experiments/container_independence/scripts/x4_test_adapter.py

Tests
-----
T1 pandas equivalence: VectorizedDF2(PandasAdapter) vs current sktime VectorizedDF on
   Series/Panel/Hierarchical, iterate_as Series/Panel, iterate_cols True/False,
   datetime/period/int time index: keys, slices, freq, reconstruct (3 col modes,
   convert_back), vectorize_est grid.
T2 polars native: polars_panel / polars_hierarchical through VectorizedDF2 with ZERO
   pandas<->polars conversions (instrumented), values equal to the pandas reference.
T3 polars lazy: schema without collecting data, slices lazy or collect-once.
T4 pyarrow: slices + reconstruct round trip (no sktime mtype exists).
T5 plain-key polars frames (Nixtla style) via rename to __index__ convention.
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
import pyarrow as pa  # noqa: E402
from common import env_info, make_panel  # noqa: E402

from sktime.datatypes import check_is_scitype, convert_to  # noqa: E402
from sktime.datatypes._vectorize import prepare_VectorizedDF  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
from x4_adapter import (  # noqa: E402
    PolarsAdapter,
    PolarsLazyAdapter,
    VectorizedDF2,
    get_adapter,
    prepare_VectorizedDF2,
)

ED = os.path.join(os.path.dirname(__file__), "..")
RESULTS = {"env": env_info(), "tests": []}


def record(name, ok, detail=""):
    RESULTS["tests"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:500]})
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -- {detail}"))


# ----------------------------------------------------------------------------
# conversion counter: every pandas<->polars conversion goes through these two
# ----------------------------------------------------------------------------
class ConvCounter:
    def __init__(self):
        self.to_pandas = 0
        self.from_pandas = 0
        self.rows = 0
        self._orig_tp = pl.DataFrame.to_pandas
        self._orig_fp = pl.from_pandas

    def __enter__(self):
        cnt = self
        orig_tp, orig_fp = self._orig_tp, self._orig_fp

        def tp(df, *a, **k):
            cnt.to_pandas += 1
            cnt.rows += df.height
            return orig_tp(df, *a, **k)

        def fp(obj, *a, **k):
            cnt.from_pandas += 1
            cnt.rows += len(obj)
            return orig_fp(obj, *a, **k)

        pl.DataFrame.to_pandas = tp
        pl.from_pandas = fp
        return self

    def __exit__(self, *exc):
        pl.DataFrame.to_pandas = self._orig_tp
        pl.from_pandas = self._orig_fp

    @property
    def total(self):
        return self.to_pandas + self.from_pandas


def transform(df):
    """Per-slice op used for reconstruct tests (backend-specific, same math)."""
    if isinstance(df, pd.DataFrame):
        return df * 2.0 + 1.0
    if isinstance(df, pl.DataFrame):
        ad = get_adapter(df)
        return df.with_columns([(pl.col(c) * 2.0 + 1.0) for c in ad.value_columns(df)])
    if isinstance(df, pa.Table):
        import pyarrow.compute as pc

        ad = get_adapter(df)
        for c in ad.value_columns(df):
            i = df.column_names.index(c)
            df = df.set_column(i, c, pc.add(pc.multiply(df.column(c), 2.0), 1.0))
        return df
    raise TypeError(type(df))


def transform_multi(df):
    """Per-slice op producing 2 output cols per input col (forces 'flat')."""
    if isinstance(df, pd.DataFrame):
        out = pd.concat([df, df * 2.0], axis=1)
        out.columns = [f"{c}_{j}" for j in range(2) for c in df.columns]
        return out
    ad = get_adapter(df)
    vc = ad.value_columns(df)
    keys = [c for c in df.columns if c not in vc]
    exprs = [pl.col(c).alias(f"{c}_0") for c in vc] + [
        (pl.col(c) * 2.0).alias(f"{c}_1") for c in vc
    ]
    return df.select(keys + exprs)


# ----------------------------------------------------------------------------
# T1: pandas equivalence
# ----------------------------------------------------------------------------
def make_series(n_time=30, n_cols=3, time_kind="datetime"):
    X = make_panel(1, n_time, n_cols=n_cols, n_levels=1, time_kind=time_kind)
    return X.droplevel(0)


CASES = []
for tk in ["datetime", "period", "int"]:
    for n_cols in [1, 3]:
        CASES.append(("Series", make_series(30, n_cols, tk), tk, n_cols))
        CASES.append(("Panel", make_panel(7, 20, n_cols, 1, time_kind=tk), tk, n_cols))
        CASES.append(
            ("Hierarchical", make_panel(12, 15, n_cols, 2, time_kind=tk), tk, n_cols)
        )
# 3 instance levels hierarchical
CASES.append(("Hierarchical", make_panel(27, 10, 2, 3), "datetime", 2))

ITER_AS = {
    "Series": ["Series"],
    "Panel": ["Series", "Panel"],
    "Hierarchical": ["Series", "Panel", "Hierarchical"],
}


class _Est:
    """Minimal estimator for vectorize_est: clone + fit storing slice length."""

    def clone(self):
        return _Est()

    def fit(self, y=None, X=None):
        self.n_ = len(y) if y is not None else None
        return self


def t1():
    n_ok = 0
    n_total = 0
    for sci, X, tk, n_cols in CASES:
        for it_as in ITER_AS[sci]:
            for it_cols in [False, True]:
                tag = f"T1[{sci},{tk},cols={n_cols},iter_as={it_as},iter_cols={it_cols}]"
                n_total += 1
                try:
                    v0, X0 = prepare_VectorizedDF(
                        X, iterate_as=it_as, is_scitype=sci, iterate_cols=it_cols
                    )
                    v1, X1 = prepare_VectorizedDF2(
                        X,
                        iterate_as=it_as,
                        is_scitype=sci,
                        iterate_cols=it_cols,
                        target="pandas",
                    )
                    pd.testing.assert_frame_equal(X0, X1)
                    assert len(v0) == len(v1) and v0.shape == v1.shape
                    assert v0.freq == v1.freq, (v0.freq, v1.freq)
                    r0, c0 = v0.get_iter_indices()
                    r1, c1 = v1.get_iter_indices()
                    if r0 is None:
                        assert r1 is None
                    else:
                        pd.testing.assert_index_equal(r0, r1, exact=True)
                    if c0 is None:
                        assert c1 is None
                    else:
                        pd.testing.assert_index_equal(c0, c1, exact=True)
                    it0 = list(v0.items(X0))
                    it1 = list(v1.items(X1))
                    assert len(it0) == len(it1)
                    for (g0, cc0, s0), (g1, cc1, s1) in zip(it0, it1):
                        assert g0 == g1 and cc0 == cc1, (g0, g1, cc0, cc1)
                        pd.testing.assert_frame_equal(s0, s1, check_freq=True)
                    # schema-only items
                    assert [x[:2] for x in v0.items()] == [x[:2] for x in v1.items()]
                    # reconstruct in all modes
                    for fn in [transform, transform_multi]:
                        outs0 = [fn(s) for s in v0.as_list(X0)]
                        outs1 = [fn(s) for s in v1.as_list(X1)]
                        for mode in ["none", "flat", "multiindex"]:
                            for cb in [False, True]:
                                a = v0.reconstruct(
                                    [o.copy() for o in outs0],
                                    col_multiindex=mode,
                                    convert_back=cb,
                                )
                                b = v1.reconstruct(
                                    [o.copy() for o in outs1],
                                    col_multiindex=mode,
                                    convert_back=cb,
                                )
                                if isinstance(a, pd.DataFrame):
                                    pd.testing.assert_frame_equal(a, b)
                                else:
                                    pd.testing.assert_series_equal(a, b)
                    # vectorize_est (inherited, unchanged) -> same estimator grid
                    g0 = v0.vectorize_est(_Est(), method="clone")
                    g1 = v1.vectorize_est(_Est(), method="clone")
                    g0 = v0.vectorize_est(g0, method="fit", y=(X0, v0), varname_of_self="y")
                    g1 = v1.vectorize_est(g1, method="fit", y=(X1, v1), varname_of_self="y")
                    pd.testing.assert_index_equal(g0.index, g1.index)
                    pd.testing.assert_index_equal(g0.columns, g1.columns)
                    assert [e.n_ for e in g0.values.ravel()] == [
                        e.n_ for e in g1.values.ravel()
                    ]
                    n_ok += 1
                except Exception as e:  # noqa: BLE001
                    record(tag, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}")
    record(f"T1 pandas equivalence: {n_ok}/{n_total} configs identical", n_ok == n_total)
    return n_ok, n_total


# ----------------------------------------------------------------------------
# T2: polars native, zero conversion
# ----------------------------------------------------------------------------
def _pl_to_pd_slice(s):
    """Reference conversion of a polars slice (any nlevels) for comparison only."""
    from sktime.datatypes._adapter.polars import convert_polars_to_pandas

    return convert_polars_to_pandas(s)


def t2():
    n_ok = 0
    n_total = 0
    conv_counts = []
    for sci, X, tk, n_cols in CASES:
        if sci == "Series" or tk == "period":
            # period -> polars converts to timestamps (sktime polars limitation)
            continue
        if tk == "int":
            continue  # polars mtypes + int time are tested implicitly via datetime
        pl_mtype = {"Panel": "polars_panel", "Hierarchical": "polars_hierarchical"}[sci]
        Xpl = convert_to(X, pl_mtype)
        for it_as in ITER_AS[sci]:
            for it_cols in [False, True]:
                n_total += 1
                tag = f"T2[{sci},cols={n_cols},iter_as={it_as},iter_cols={it_cols}]"
                try:
                    with ConvCounter() as cc:
                        v, Xn = prepare_VectorizedDF2(
                            Xpl,
                            iterate_as=it_as,
                            is_scitype=sci,
                            iterate_cols=it_cols,
                            target="native",
                            X_orig_mtype=pl_mtype,
                        )
                        slices = v.as_list(Xn)
                        keys = [(g, c) for g, c, _ in v.items(Xn)]
                        outs = [transform(s) for s in slices]
                        rec = v.reconstruct(outs, col_multiindex="flat")
                        rec_id = v.reconstruct(slices)
                    conv_counts.append(cc.total)
                    assert isinstance(Xn, pl.DataFrame) and Xn is Xpl
                    assert all(isinstance(s, pl.DataFrame) for s in slices)
                    assert cc.total == 0, f"{cc.total} conversions"
                    # reference: current pandas path
                    v0, X0 = prepare_VectorizedDF(
                        X, iterate_as=it_as, is_scitype=sci, iterate_cols=it_cols
                    )
                    keys0 = [(g, c) for g, c, _ in v0.items(X0)]
                    assert keys == keys0, (keys[:3], keys0[:3])
                    for s, s0 in zip(slices, v0.as_list(X0)):
                        sp = _pl_to_pd_slice(s)
                        np.testing.assert_allclose(sp.to_numpy(), s0.to_numpy())
                        assert list(sp.index.get_level_values(-1)) == list(
                            s0.index.get_level_values(-1)
                        )
                        assert list(sp.index.names) == list(s0.index.names)
                    # identity reconstruct round trip == input frame
                    assert rec_id.equals(Xpl), "identity reconstruct != input"
                    # transformed reconstruct equals pandas reconstruct
                    ref = v0.reconstruct(
                        [transform(s) for s in v0.as_list(X0)], col_multiindex="flat"
                    )
                    rec_pd = convert_to(rec, "pd-multiindex" if sci == "Panel" else "pd_multiindex_hier")
                    np.testing.assert_allclose(rec_pd.to_numpy(), ref.to_numpy())
                    assert list(rec_pd.columns) == list(ref.columns), (
                        list(rec_pd.columns),
                        list(ref.columns),
                    )
                    assert list(rec_pd.index.names) == list(ref.index.names)
                    assert rec_pd.index.equals(ref.index)
                    n_ok += 1
                except Exception as e:  # noqa: BLE001
                    record(tag, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}")
    record(
        f"T2 polars native: {n_ok}/{n_total} configs equal to pandas ref, "
        f"max pandas<->polars conversions={max(conv_counts) if conv_counts else 'NA'}",
        n_ok == n_total,
    )
    return n_ok, n_total, conv_counts


# ----------------------------------------------------------------------------
# T3: polars lazy
# ----------------------------------------------------------------------------
class CollectCounter:
    def __init__(self):
        self.n = 0
        self._orig = pl.LazyFrame.collect

    def __enter__(self):
        cnt = self
        orig = self._orig

        def collect(lf, *a, **k):
            cnt.n += 1
            return orig(lf, *a, **k)

        pl.LazyFrame.collect = collect
        return self

    def __exit__(self, *exc):
        pl.LazyFrame.collect = self._orig


def t3():
    X = make_panel(9, 12, 2, 2)
    Xpl = convert_to(X, "polars_hierarchical")
    Xlz = Xpl.lazy()
    out = {}
    # is a LazyFrame panel an sktime mtype? (expected: no)
    valid, _, _ = check_is_scitype(Xlz, scitype=["Series", "Panel", "Hierarchical"], return_metadata=[])
    out["lazy_panel_is_sktime_mtype"] = bool(valid)
    record("T3 polars LazyFrame hierarchical is NOT a recognised sktime mtype", not valid)
    ok = True
    for mode in ["lazy", "collect_once"]:
        from x4_adapter import REGISTRY

        REGISTRY.by_lib["polars_lazy"].slice_mode = mode
        with CollectCounter() as cc_schema:
            v = VectorizedDF2(Xlz, iterate_as="Series", is_scitype="Hierarchical")
        with CollectCounter() as cc_iter:
            sl = v.as_list(Xlz)
        ve, _ = prepare_VectorizedDF2(Xpl, iterate_as="Series", is_scitype="Hierarchical", X_orig_mtype="polars_hierarchical")
        se = ve.as_list(Xpl)
        sl_eager = [s.collect() if isinstance(s, pl.LazyFrame) else s for s in sl]
        ok_slices = all(a.equals(b) for a, b in zip(sl_eager, se)) and len(sl) == len(se)
        with CollectCounter() as cc_rec:
            rec = v.reconstruct([s.lazy() for s in sl])
        rec_lazy = isinstance(rec, pl.LazyFrame)
        ok_rec = rec.collect().equals(Xpl)
        out[mode] = {
            "collects_schema": cc_schema.n,
            "collects_iterate": cc_iter.n,
            "collects_reconstruct": cc_rec.n,
            "slices_lazy": all(isinstance(s, pl.LazyFrame) for s in sl),
            "reconstruct_lazy": rec_lazy,
            "slices_equal_eager": ok_slices,
            "reconstruct_equal": ok_rec,
        }
        ok = ok and ok_slices and ok_rec
        record(f"T3 lazy[{mode}] {out[mode]}", ok_slices and ok_rec)
    REGISTRY.by_lib["polars_lazy"].slice_mode = "collect_once"
    return out


# ----------------------------------------------------------------------------
# T4: pyarrow
# ----------------------------------------------------------------------------
def t4():
    res = {}
    for sci, X in [("Panel", make_panel(8, 10, 2, 1)), ("Hierarchical", make_panel(12, 8, 2, 2))]:
        pl_mtype = {"Panel": "polars_panel", "Hierarchical": "polars_hierarchical"}[sci]
        Xpl = convert_to(X, pl_mtype)
        Xpa = Xpl.to_arrow()
        for it_as in ITER_AS[sci][:2]:
            if it_as == sci:
                continue
            vpa = VectorizedDF2(Xpa, iterate_as=it_as, is_scitype=sci)
            vpl = VectorizedDF2(Xpl, iterate_as=it_as, is_scitype=sci)
            a = vpa.as_list(Xpa)
            b = vpl.as_list(Xpl)
            ok = len(a) == len(b) and all(pl.from_arrow(x).equals(y) for x, y in zip(a, b))
            ok = ok and list(vpa.iter_indices[0]) == list(vpl.iter_indices[0])
            rec = vpa.reconstruct(a)
            ok_rec = pl.from_arrow(rec).equals(Xpl)
            # shuffled (non-contiguous) instances -> sort fallback
            perm = np.random.default_rng(0).permutation(Xpa.num_rows)
            Xsh = Xpa.take(pa.array(perm))
            vsh = VectorizedDF2(Xsh, iterate_as=it_as, is_scitype=sci)
            ok_sh = len(vsh) == len(vpa)
            res[f"{sci}/{it_as}"] = {"slices": ok, "reconstruct": ok_rec, "noncontiguous_groups": ok_sh}
            record(f"T4 pyarrow {sci}/{it_as}: slices={ok} reconstruct={ok_rec} shuffled_ngroups_ok={ok_sh}", ok and ok_rec and ok_sh)
    return res


# ----------------------------------------------------------------------------
# T5: plain key columns -> __index__ convention by rename
# ----------------------------------------------------------------------------
def t5():
    from common import to_polars_long

    X = make_panel(50, 20, 1, 1)
    long = to_polars_long(X)  # columns h0, time, c0
    valid_plain, _, md = check_is_scitype(long, scitype=["Series", "Panel", "Hierarchical"], return_metadata=["n_features", "feature_names"])
    renamed = PolarsAdapter.from_key_columns(long, ["h0", "time"])
    valid_ren, _, md2 = check_is_scitype(renamed, scitype=["Series", "Panel", "Hierarchical"], return_metadata=[])
    # Is the plain frame silently mis-read?
    out = {
        "plain_valid": bool(valid_plain),
        "plain_mtype": md.get("mtype") if valid_plain else None,
        "plain_scitype": md.get("scitype") if valid_plain else None,
        "plain_n_features": md.get("n_features") if valid_plain else None,
        "plain_feature_names": md.get("feature_names") if valid_plain else None,
        "renamed_mtype": md2.get("mtype"),
        "renamed_scitype": md2.get("scitype"),
    }
    # zero-copy: buffers shared?
    a = long.get_column("c0").to_arrow()
    b = renamed.get_column("c0").to_arrow()
    out["rename_shares_buffers"] = a.buffers()[1].address == b.buffers()[1].address
    record(f"T5 plain keys: {out}", md2.get("mtype") == "polars_panel")
    # a slice of a plain-key frame (time col + value) is mis-read as 2-var series
    sl = long.filter(pl.col("h0") == "s0").drop("h0")
    v, _, mds = check_is_scitype(sl, scitype=["Series"], return_metadata=["n_features", "feature_names"])
    out["plain_slice"] = {"valid": bool(v), "n_features": mds.get("n_features") if v else None, "feature_names": mds.get("feature_names") if v else None}
    record(f"T5 plain-key slice read as: {out['plain_slice']}", True)
    return out


if __name__ == "__main__":
    RESULTS["t1"] = t1()
    RESULTS["t2"] = t2()
    RESULTS["t3"] = t3()
    RESULTS["t4"] = t4()
    RESULTS["t5"] = t5()
    n_fail = sum(not t["ok"] for t in RESULTS["tests"])
    print(f"\n{len(RESULTS['tests']) - n_fail} passed, {n_fail} failed")
    with open(os.path.join(ED, "results", "x4_test_adapter.json"), "w") as f:
        json.dump(RESULTS, f, indent=1, default=str)
