"""g2 equality: ported np_grouped forecasters vs stock sktime, on pandas/polars/pyarrow.

Run (from worktree root):
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/g2_equality.py
"""

import json
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ED)

from common import env_info, make_panel, make_panel_fast  # noqa: E402
from protos.g2_forecasters import (  # noqa: E402
    G2NaiveForecaster,
    G2PolynomialTrendForecaster,
    G2RecursiveReducer,
)
from protos.g2_np_grouped import COUNTS  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sktime.datatypes import convert_to  # noqa: E402
from sktime.forecasting.compose import make_reduction  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402
from sktime.forecasting.trend import PolynomialTrendForecaster  # noqa: E402

PD_MAJOR = int(pd.__version__.split(".")[0])


def quiet(f):
    f.set_config(remember_data=True)
    return f


def to_arrow(y):
    """pandas sktime frame -> pyarrow table with __index__ columns (user-side)."""
    import pyarrow as pa

    if isinstance(y, pd.Series):
        y = y.to_frame(name=y.name if y.name is not None else 0)
    flat = y.reset_index()
    nidx = y.index.nlevels
    flat.columns = [f"__index__{c}" if i < nidx else c
                    for i, c in enumerate(flat.columns)]
    return pa.Table.from_pandas(flat, preserve_index=False)


def arrow_to_pandas(tab, like):
    """pyarrow / polars output -> pandas, independently of the proto code."""
    df = tab.to_pandas()
    idx = [c for c in df.columns if c.startswith("__index__")]
    df = df.set_index(idx)
    df.index.names = [c[len("__index__"):] for c in idx]
    if len(idx) == 1:  # stock single-series predictions have an unnamed index
        df.index.name = None
    if isinstance(like, pd.Series):
        s = df.iloc[:, 0]
        s.name = like.name
        return s
    return df


def make_cases():
    rng = np.random.default_rng(1)
    c = {}
    c["panel_D"] = make_panel_fast(20, 50)
    c["hier_D"] = make_panel(n_series=12, n_time=40, n_levels=2)
    s = make_panel_fast(1, 60)["c0"].droplevel(0)
    s.index.freq = "D"
    c["series_D"] = s
    c["frame_h"] = make_panel_fast(1, 60).droplevel(0).asfreq("D")
    p = make_panel_fast(10, 40, time_kind="period")
    c["panel_periodM"] = p
    c["panel_int"] = make_panel_fast(10, 40, time_kind="int")
    c["panel_MS"] = make_panel(10, 40, freq="MS", n_levels=1)
    c["panel_W"] = make_panel(10, 40, freq="W-SUN", n_levels=1)
    # ragged: unequal lengths, same end
    r = make_panel_fast(8, 50)
    keep = np.ones(len(r), bool)
    for i in range(0, 8, 2):
        keep[i * 50: i * 50 + 5 * (i + 1)] = False
    c["ragged_same_end"] = r[keep]
    # ragged, different ends
    keep2 = np.ones(len(r), bool)
    for i in range(1, 8, 3):
        keep2[(i + 1) * 50 - 3 * i: (i + 1) * 50] = False
    c["ragged_diff_end"] = r[keep2]
    n = make_panel_fast(10, 50).copy()
    v = n["c0"].to_numpy().copy()
    v[rng.choice(len(v), 25, replace=False)] = np.nan
    v[49] = np.nan  # trailing NaN in instance 0
    n["c0"] = v
    c["panel_nan"] = n
    c["panel_2col"] = make_panel_fast(10, 50, n_cols=2)
    return c


def inputs_for(y):
    out = {"pandas": y}
    try:
        if isinstance(y, pd.Series) or y.index.nlevels == 1:
            out["polars"] = convert_to(y if isinstance(y, pd.DataFrame)
                                       else y.to_frame(), "pl.DataFrame")
        elif y.index.nlevels == 2:
            out["polars"] = convert_to(y, "polars_panel")
        else:
            out["polars"] = convert_to(y, "polars_hierarchical")
    except Exception as e:  # noqa: BLE001
        out["polars_err"] = repr(e)[:120]
    try:
        out["pyarrow"] = to_arrow(y)
    except Exception as e:  # noqa: BLE001
        out["pyarrow_err"] = repr(e)[:120]
    return out


FH = [1, 2, 3, 5]


def configs():
    return [
        ("naive_last", lambda: NaiveForecaster("last"),
         lambda: G2NaiveForecaster("last"), True),
        ("naive_mean", lambda: NaiveForecaster("mean"),
         lambda: G2NaiveForecaster("mean"), True),
        ("naive_mean_w5", lambda: NaiveForecaster("mean", window_length=5),
         lambda: G2NaiveForecaster("mean", window_length=5), True),
        ("naive_drift", lambda: NaiveForecaster("drift"),
         lambda: G2NaiveForecaster("drift"), True),
        ("naive_drift_w10", lambda: NaiveForecaster("drift", window_length=10),
         lambda: G2NaiveForecaster("drift", window_length=10), True),
        ("naive_last_sp7", lambda: NaiveForecaster("last", sp=7),
         lambda: G2NaiveForecaster("last", sp=7), False),
        ("naive_mean_sp7_w14", lambda: NaiveForecaster("mean", sp=7, window_length=14),
         lambda: G2NaiveForecaster("mean", sp=7, window_length=14), False),
        ("poly_d1_pi", lambda: PolynomialTrendForecaster(prediction_intervals=True),
         lambda: G2PolynomialTrendForecaster(prediction_intervals=True), True),
        ("poly_d2_noint_pi", lambda: PolynomialTrendForecaster(
            degree=2, with_intercept=False, prediction_intervals=True),
         lambda: G2PolynomialTrendForecaster(degree=2, with_intercept=False,
                                             prediction_intervals=True), True),
        ("poly_d2_ridge", lambda: PolynomialTrendForecaster(Ridge(1.0), degree=2),
         lambda: G2PolynomialTrendForecaster(Ridge(1.0), degree=2), False),
        ("reduce_lr_w5", lambda: make_reduction(LinearRegression(), window_length=5,
                                                strategy="recursive"),
         lambda: G2RecursiveReducer(LinearRegression(), window_length=5), False),
    ]


def norm(x, like):
    mod = type(x).__module__.split(".")[0]
    if mod in ("pyarrow", "polars"):
        return arrow_to_pandas(x, like)
    return x


def compare(a, b, rtol):
    """a (g2), b (stock), both pandas. Returns (ok, max_rel_err, msg)."""
    try:
        if isinstance(b, pd.Series):
            pd.testing.assert_series_equal(a, b, rtol=rtol, check_freq=False)
        else:
            pd.testing.assert_frame_equal(a, b, rtol=rtol, check_freq=False)
        av, bv = np.asarray(a, float), np.asarray(b, float)
        m = np.isfinite(bv)
        err = float(np.max(np.abs(av[m] - bv[m]) / np.maximum(np.abs(bv[m]), 1e-300))
                    ) if m.any() else 0.0
        return True, err, ""
    except AssertionError as e:
        return False, None, str(e).replace("\n", " ")[:300]


def main():
    cases = make_cases()
    results = []
    for cname, y in cases.items():
        ins = inputs_for(y)
        for fname, mk_stock, mk_g2, has_pi in configs():
            if fname.startswith("naive_last_sp7") or fname.startswith("naive_mean_sp7"):
                if cname not in ("panel_D", "series_D", "ragged_same_end"):
                    continue
            if fname == "reduce_lr_w5" and cname in ("panel_nan",):
                continue
            if fname.startswith("poly") and cname == "panel_nan":
                continue
            # stock reference on pandas input
            try:
                fs = quiet(mk_stock()).fit(y, fh=FH)
                ref = fs.predict()
                ref_pi = fs.predict_interval(coverage=[0.8, 0.95]) if has_pi else None
                ref_cut = fs.cutoff
            except Exception as e:  # noqa: BLE001
                results.append(dict(case=cname, fc=fname, input="pandas",
                                    stock_error=repr(e)[:200]))
                continue
            for iname, yin in ins.items():
                if iname.endswith("_err"):
                    results.append(dict(case=cname, fc=fname, input=iname[:-4],
                                        input_error=yin))
                    continue
                rec = dict(case=cname, fc=fname, input=iname)
                if cname == "panel_periodM" and iname == "pyarrow":
                    rec["na"] = "pandas Period has no arrow type (int64 + metadata)"
                    results.append(rec)
                    continue
                ref_i, ref_pi_i, ref_cut_i = ref, ref_pi, ref_cut
                if cname == "panel_periodM" and iname == "polars":
                    # polars has no Period: input is already datetime; stock polars
                    # run is the reference
                    fsp = quiet(mk_stock()).fit(yin, fh=FH)
                    ref_i = arrow_to_pandas(fsp.predict(), ref)
                    ref_pi_i = (fsp.predict_interval(coverage=[0.8, 0.95])
                                if has_pi else None)
                    ref_cut_i = fsp.cutoff
                try:
                    f = quiet(mk_g2()).fit(yin, fh=FH)
                    out = f.predict()
                    rec["out_type"] = type(out).__module__.split(".")[0]
                    rec["same_type_as_input"] = type(out) is type(yin)
                    if iname == "polars":  # compare natively against stock polars run
                        fsp = quiet(mk_stock()).fit(yin, fh=FH)
                        sp_out = fsp.predict()
                        try:
                            rec["polars_native_equal"] = bool(
                                out.select(sp_out.columns).equals(sp_out))
                        except Exception as e:  # noqa: BLE001
                            rec["polars_native_equal"] = False
                            rec["stock_polars_columns"] = list(sp_out.columns)
                            rec["g2_polars_columns"] = list(out.columns)
                            raise_skip = repr(e)[:80]  # noqa: F841
                        if not rec["polars_native_equal"] and \
                                "stock_polars_columns" not in rec:
                            import polars.testing as plt

                            try:
                                plt.assert_frame_equal(out, sp_out, rel_tol=1e-9)
                                rec["polars_native_close"] = True
                            except AssertionError as e:
                                rec["polars_native_close"] = str(e)[:200]
                    ok, err, msg = compare(norm(out, ref), ref_i, 1e-9)
                    rec.update(pred_ok=ok, pred_maxrel=err, pred_msg=msg)
                    c = f.cutoff
                    rec["cutoff_ok"] = bool(len(c) == 1 and c[0] == ref_cut_i[0])
                    rec["cutoff_freq_same"] = bool(
                        getattr(c, "freq", None) == getattr(ref_cut_i, "freq", None))
                    if not (rec["cutoff_ok"] and rec["cutoff_freq_same"]):
                        rec["cutoff"] = f"{c!r} vs {ref_cut_i!r}"
                    if has_pi:
                        pi = f.predict_interval(coverage=[0.8, 0.95])
                        ok, err, msg = compare(pi, ref_pi_i, 1e-7)
                        rec.update(pi_ok=ok, pi_maxrel=err, pi_msg=msg)
                except Exception as e:  # noqa: BLE001
                    rec["g2_error"] = repr(e)[:300]
                    rec["tb"] = traceback.format_exc()[-600:]
                results.append(rec)

    # absolute fh + update, on panel_D (pandas + polars)
    extra = []
    y = cases["panel_D"]
    fh_abs = pd.date_range(y.index.get_level_values(-1).max(), periods=4, freq="D")[1:]
    for iname, yin in inputs_for(y).items():
        if iname.endswith("_err"):
            continue
        for strat in ("last", "drift"):
            try:
                a = quiet(G2NaiveForecaster(strat)).fit(yin).predict(fh=fh_abs)
                b = quiet(NaiveForecaster(strat)).fit(y).predict(fh=fh_abs)
                ok, err, msg = compare(norm(a, b), b, 1e-9)
                extra.append(dict(test="absolute_fh", input=iname, strategy=strat,
                                  ok=ok, maxrel=err, msg=msg))
            except Exception as e:  # noqa: BLE001
                extra.append(dict(test="absolute_fh", input=iname, strategy=strat,
                                  error=repr(e)[:200]))
    yfull = make_panel_fast(20, 60)
    tl = yfull.index.get_level_values(-1)
    y1, y2 = yfull[tl < tl.unique()[50]], yfull[tl >= tl.unique()[50]]
    for iname in ("pandas", "polars", "pyarrow"):
        try:
            i1, i2 = inputs_for(y1)[iname], inputs_for(y2)[iname]
            for strat in ("last", "drift", "mean"):
                fg = quiet(G2NaiveForecaster(strat)).fit(i1, fh=FH)
                fg.update(i2)
                a = fg.predict()
                fs = quiet(NaiveForecaster(strat)).fit(y1, fh=FH)
                fs.update(y2)
                b = fs.predict()
                ok, err, msg = compare(norm(a, b), b, 1e-9)
                extra.append(dict(test="update_then_predict", input=iname,
                                  strategy=strat, ok=ok, maxrel=err, msg=msg,
                                  cutoff_ok=bool(fg.cutoff[0] == fs.cutoff[0])))
        except Exception as e:  # noqa: BLE001
            extra.append(dict(test="update_then_predict", input=iname,
                              error=repr(e)[:300], tb=traceback.format_exc()[-500:]))
    # mixed frequency panel must fail loudly (x6 verify found a silent error)
    a = make_panel_fast(1, 10)
    b = make_panel_fast(1, 10).rename(index={"s0000000": "s1"}, level=0)
    b.index = b.index.set_levels(pd.date_range("2000-01-01", periods=10, freq="h"),
                                 level=1)
    mixed = pd.concat([a, b])
    try:
        quiet(G2NaiveForecaster()).fit(mixed, fh=[1, 2]).predict()
        extra.append(dict(test="mixed_freq", result="no error (SILENT)"))
    except Exception as e:  # noqa: BLE001
        extra.append(dict(test="mixed_freq", result="raises", error=repr(e)[:160]))
    try:
        quiet(NaiveForecaster()).fit(mixed, fh=[1, 2]).predict()
        extra.append(dict(test="mixed_freq_stock", result="no error"))
    except Exception as e:  # noqa: BLE001
        extra.append(dict(test="mixed_freq_stock", result="raises",
                          error=repr(e)[:160]))
    extra.append(dict(test="counts", **COUNTS))

    summ = {}
    for r in results:
        k = r["fc"]
        s = summ.setdefault(k, {"n": 0, "pred_ok": 0, "pi_ok": 0, "pi_n": 0,
                                "errors": 0, "cutoff_ok": 0, "cutoff_freq_same": 0,
                                "na": 0})
        s["na"] += "na" in r
        if "pred_ok" in r or "g2_error" in r:
            s["n"] += 1
            s["pred_ok"] += bool(r.get("pred_ok"))
            s["errors"] += "g2_error" in r
            s["cutoff_ok"] += bool(r.get("cutoff_ok"))
            s["cutoff_freq_same"] += bool(r.get("cutoff_freq_same"))
            if "pi_ok" in r:
                s["pi_n"] += 1
                s["pi_ok"] += bool(r["pi_ok"])
    out = dict(env=env_info(), summary=summ, results=results, extra=extra)
    tag = f"pd{PD_MAJOR}"
    path = os.path.join(ED, "results", f"g2_equality_{tag}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps(summ, indent=0))
    for r in results:
        bad = (not r.get("pred_ok", True)) or (r.get("pi_ok") is False) or (
            "g2_error" in r) or r.get("cutoff_ok") is False or "stock_error" in r or (
            "input_error" in r)
        if bad:
            print({k: v for k, v in r.items() if k != "tb"})
    for e in extra:
        print({k: v for k, v in e.items() if k != "tb"})
    print("->", path)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time() - t0:.1f}s")
