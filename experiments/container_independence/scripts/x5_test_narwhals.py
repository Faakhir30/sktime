"""x5: correctness tests for the narwhals VectorizedDF + narwhals forecasters.

A. pandas equivalence with current sktime VectorizedDF (items + reconstruct):
   group names, order, slice index (names, dtype, freq), values, dtypes.
B. round-trip fidelity for polars / polars lazy / pyarrow / dask / duckdb / ibis:
   items -> identity -> reconstruct; output native type == input type? values?
C. narwhals panel-native NaiveForecaster vs sktime NaiveForecaster (vectorized):
   numeric equality per backend x strategy; NaN; unequal length; monthly.

Run (both pandas 2 and pandas 3 envs):
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x5_test_narwhals.py
"""

import json
import warnings
from pathlib import Path

import narwhals as nw
import numpy as np
import pandas as pd

from common import env_info, make_panel
from protos.x5_backends import BACKENDS, long_from_mi, make_native, to_pandas_any
from protos.x5_narwhals import prepare_nw_vectorized
from protos.x5_nw_forecasters import NwPanelNaive

warnings.filterwarnings("ignore")
ED = Path(__file__).resolve().parents[1]
RES = {"env": env_info(), "A": {}, "B": {}, "C": {}}


def _sk_items(X, iterate_as, iterate_cols):
    from sktime.datatypes._vectorize import prepare_VectorizedDF

    is_scitype = {1: "Series", 2: "Panel"}.get(X.index.nlevels, "Hierarchical")
    vec, X_mi = prepare_VectorizedDF(
        X, iterate_as=iterate_as, is_scitype=is_scitype, iterate_cols=iterate_cols
    )
    return vec, X_mi, list(vec.items(X_mi))


# --------------------------------------------------------------------------- A
def case_frames():
    out = {}
    out["panel_dt_D"] = make_panel(6, 12, n_levels=1)
    out["panel_period_M"] = make_panel(6, 12, n_levels=1, time_kind="period", freq="M")
    out["panel_int"] = make_panel(6, 12, n_levels=1, time_kind="int")
    out["hier_3lvl"] = make_panel(8, 10, n_levels=2)
    mv = make_panel(5, 10, n_cols=3, n_levels=1)
    out["panel_multivar"] = mv
    ic = mv.copy()
    ic.columns = [0, 1, 2]
    out["panel_int_colnames"] = ic
    nn = make_panel(4, 10, n_levels=1)
    nn.index = nn.index.set_names([None, None])
    out["panel_none_names"] = nn
    # instance order not sorted: s3 first
    un = make_panel(4, 10, n_levels=1)
    order = ["s3", "s0", "s2", "s1"]
    un = pd.concat([un.loc[[k]] for k in order])
    out["panel_unsorted_instances"] = un
    return out


def test_A():
    cases = case_frames()
    configs = [("Series", False), ("Series", True), ("Panel", False)]
    for name, X in cases.items():
        for iterate_as, iterate_cols in configs:
            if iterate_as == "Panel" and X.index.nlevels < 3:
                continue
            key = f"{name}|{iterate_as}|cols={iterate_cols}"
            try:
                vec, X_mi, sk = _sk_items(X, iterate_as, iterate_cols)
                schema, f = prepare_nw_vectorized(
                    X, iterate_as=iterate_as, iterate_cols=iterate_cols
                )
                mine = list(schema.items(f))
                res = {"n": len(sk), "n_nw": len(mine)}
                assert len(sk) == len(mine), "length"
                res["names_equal"] = all(a[0] == b[0] for a, b in zip(sk, mine))
                res["colnames_equal"] = all(a[1] == b[1] for a, b in zip(sk, mine))
                n_fail, first_err = 0, None
                for a, b in zip(sk, mine):
                    try:
                        pd.testing.assert_frame_equal(a[2], b[2], check_freq=True)
                    except AssertionError as e:
                        n_fail += 1
                        first_err = first_err or str(e).replace("\n", " ")[:300]
                res["slices_equal"] = n_fail == 0
                fast = list(schema.items(f, fast_pandas=True))
                n_fail_fast = 0
                for a, b in zip(sk, fast):
                    try:
                        pd.testing.assert_frame_equal(a[2], b[2], check_freq=True)
                    except AssertionError:
                        n_fail_fast += 1
                res["fast_slices_equal"] = n_fail_fast == 0 and len(fast) == len(sk)
                res["slice_fail"] = n_fail
                res["first_err"] = first_err
                res["freq_sk"] = vec.freq
                res["freq_nw"] = schema.freq
                # reconstruct (rows only; column-iteration reconstruct not in proto)
                if not iterate_cols:
                    slices = [s for _, _, s in sk]
                    r_sk = vec.reconstruct(slices)
                    r_nw = schema.reconstruct(f, slices)
                    try:
                        pd.testing.assert_frame_equal(r_sk, r_nw)
                        res["reconstruct_equal"] = True
                    except AssertionError as e:
                        res["reconstruct_equal"] = False
                        res["reconstruct_err"] = str(e).replace("\n", " ")[:300]
                    # restore to the full original frame
                    r_full = schema.reconstruct(f, schema.as_list(f))
                    try:
                        pd.testing.assert_frame_equal(r_full, X, check_freq=False)
                        res["roundtrip_equals_input"] = True
                    except AssertionError as e:
                        res["roundtrip_equals_input"] = False
                        res["roundtrip_err"] = str(e).replace("\n", " ")[:300]
            except Exception as e:  # noqa: BLE001
                res = {"error": f"{type(e).__name__}: {e}"[:300]}
            RES["A"][key] = res
            ok = (
                res.get("names_equal")
                and res.get("slices_equal")
                and res.get("reconstruct_equal", True)
            )
            print(f"A {key}: {'PASS' if ok else 'FAIL'} {res}")


# --------------------------------------------------------------------------- B
def test_B():
    X = make_panel(6, 12, n_cols=2, n_levels=1)
    lng = long_from_mi(X)
    for b in BACKENDS:
        if b == "pandas":
            continue
        native = make_native(lng, b)
        res = {"in_type": type(native).__module__ + "." + type(native).__name__}
        try:
            schema, f = prepare_nw_vectorized(native, keys=["h0"], time="time")
            res["freq"] = schema.freq
            res["n_instances"] = schema.shape(f)[0]
            items = list(schema.items(f))
            res["slice_type"] = type(items[0][2]).__module__ + "." + type(
                items[0][2]
            ).__name__
            out = schema.reconstruct(f, [s for _, _, s in items], relazy=True)
            res["out_type"] = type(out).__module__ + "." + type(out).__name__
            res["type_roundtrip"] = type(out) is type(native)
            got = to_pandas_any(out).sort_values(["h0", "time"]).reset_index(drop=True)
            exp = lng.sort_values(["h0", "time"]).reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(got, exp, check_dtype=False)
                res["values_equal"] = True
            except AssertionError as e:
                res["values_equal"] = False
                res["err"] = str(e).replace("\n", " ")[:200]
            in_schema = dict(nw.from_native(native).collect_schema())
            out_schema = dict(nw.from_native(out).collect_schema())
            res["dtype_changes"] = {
                c: f"{in_schema[c]} -> {out_schema.get(c)}"
                for c in in_schema
                if str(in_schema[c]) != str(out_schema.get(c))
            }
            res["schema_order"] = [str(r) for r in schema.instance_keys(f).rows()][:3]
        except Exception as e:  # noqa: BLE001
            res["error"] = f"{type(e).__name__}: {e}"[:300]
        RES["B"][b] = res
        print("B", b, res)

    # sktime __index__ convention via sktime's own converter
    from sktime.datatypes import convert_to

    pl_sk = convert_to(X, to_type="polars_panel")
    schema, f = prepare_nw_vectorized(pl_sk)
    out = schema.reconstruct(f, schema.as_list(f))
    back = convert_to(out, to_type="pd-multiindex")
    try:
        pd.testing.assert_frame_equal(back, X, check_freq=False)
        ok = True
    except AssertionError as e:
        ok = str(e)[:200]
    RES["B"]["polars_panel(__index__)"] = {
        "keys": schema.keys,
        "time": schema.time,
        "type_roundtrip": type(out) is type(pl_sk),
        "values_equal_after_sktime_convert": ok,
    }
    print("B polars_panel(__index__)", RES["B"]["polars_panel(__index__)"])


# --------------------------------------------------------------------------- C
STRATS = [
    dict(strategy="last"),
    dict(strategy="last", sp=7),
    dict(strategy="mean"),
    dict(strategy="mean", sp=7),
    dict(strategy="mean", window_length=10),
    dict(strategy="drift"),
    dict(strategy="drift", window_length=15),
]


def _sk_predict(y, kw, fh):
    from sktime.forecasting.naive import NaiveForecaster

    f = NaiveForecaster(**kw)
    f.set_config(remember_data=True)
    return f.fit(y).predict(fh)


def _cmp(sk, got_pd):
    """sk: pd-multiindex frame; got_pd: long pandas frame (h0,time,c0)."""
    s = sk.reset_index()
    s.columns = ["h0", "time", "c0"]
    g = got_pd.copy()
    g["time"] = pd.to_datetime(g["time"]).astype("datetime64[ns]")
    s["time"] = pd.to_datetime(s["time"]).astype("datetime64[ns]")
    g = g.sort_values(["h0", "time"]).reset_index(drop=True)[["h0", "time", "c0"]]
    s = s.sort_values(["h0", "time"]).reset_index(drop=True)
    if len(g) != len(s):
        return {"ok": False, "why": f"len {len(g)} vs {len(s)}"}
    t_ok = bool((g["time"].values == s["time"].values).all())
    k_ok = bool((g["h0"].astype(str).values == s["h0"].astype(str).values).all())
    a, b = g["c0"].to_numpy(float), s["c0"].to_numpy(float)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.where(both_nan, 0.0, np.abs(a - b))
    maxdiff = float(np.nanmax(diff)) if len(diff) else 0.0
    nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
    return {
        "ok": t_ok and k_ok and maxdiff < 1e-9 and nan_mismatch == 0,
        "time_ok": t_ok,
        "keys_ok": k_ok,
        "max_abs_diff": maxdiff,
        "nan_mismatch": nan_mismatch,
    }


def test_C():
    fh = list(range(1, 15))
    y = make_panel(20, 60, n_levels=1)
    lng = long_from_mi(y)
    for kw in STRATS:
        sk = _sk_predict(y, kw, fh)
        row = {}
        for b in ["pandas_mi", *BACKENDS]:
            try:
                if b == "pandas_mi":
                    got = NwPanelNaive(**kw).fit(y).predict(fh)
                    try:
                        pd.testing.assert_frame_equal(sk, got, check_freq=False)
                        row[b] = {"ok": True, "exact_frame_equal": True}
                    except AssertionError as e:
                        row[b] = {"ok": False, "err": str(e)[:200]}
                    continue
                m = NwPanelNaive(**kw).fit(make_native(lng, b), keys=["h0"], time="time")
                out = m.predict(fh)
                row[b] = _cmp(sk, to_pandas_any(out))
                row[b]["out_type_same_as_in"] = type(out) is type(make_native(lng, b))
            except Exception as e:  # noqa: BLE001
                row[b] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
        key = json.dumps(kw)
        RES["C"][key] = row
        print("C", key, {b: ("ok" if r.get("ok") else r) for b, r in row.items()})

    # SES vs pandas groupby ewm reference (sktime ExponentialSmoothing uses
    # statsmodels with estimated init -> not comparable)
    alpha = 0.3
    ref_level = (
        lng.sort_values(["h0", "time"])
        .groupby("h0")["c0"]
        .apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean().iloc[-1])
    )
    row = {}
    for b in ["pandas_mi", *BACKENDS]:
        try:
            if b == "pandas_mi":
                out = NwPanelNaive("ses", alpha=alpha).fit(y).predict(fh).reset_index()
            else:
                m = NwPanelNaive("ses", alpha=alpha).fit(
                    make_native(lng, b), keys=["h0"], time="time"
                )
                out = to_pandas_any(m.predict(fh))
            lv = out.groupby("h0")["c0"].first()
            row[b] = {"ok": bool(np.allclose(lv.loc[ref_level.index], ref_level))}
        except Exception as e:  # noqa: BLE001
            row[b] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:150]}
    RES["C"]["ses(alpha=0.3) vs pandas ewm"] = row
    print("C ses", {b: ("ok" if r.get("ok") else r) for b, r in row.items()})

    # NaN robustness: trailing NaNs in some series ("last" is NaN-robust in sktime)
    yn = y.copy()
    vals = yn["c0"].to_numpy().copy()
    vals[59] = np.nan  # last value of s0
    vals[60 + 57 : 60 + 60] = np.nan  # last 3 of s1
    yn["c0"] = vals
    lngn = long_from_mi(yn)
    for kw in [dict(strategy="last"), dict(strategy="last", sp=7), dict(strategy="mean")]:
        sk = _sk_predict(yn, kw, fh)
        row = {}
        for b in BACKENDS:
            try:
                m = NwPanelNaive(**kw).fit(make_native(lngn, b), keys=["h0"], time="time")
                row[b] = _cmp(sk, to_pandas_any(m.predict(fh)))
            except Exception as e:  # noqa: BLE001
                row[b] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:150]}
        RES["C"]["NaN " + json.dumps(kw)] = row
        print("C NaN", kw, {b: ("ok" if r.get("ok") else r) for b, r in row.items()})

    # unequal length / unequal end: drop last 5 obs of s0, first 7 of s1
    yu = y.copy()
    yu = yu[~((yu.index.get_level_values(0) == "s0") & (yu.index.get_level_values(1) >= y.loc["s0"].index[-5]))]
    yu = yu[~((yu.index.get_level_values(0) == "s1") & (yu.index.get_level_values(1) < y.loc["s1"].index[7]))]
    lngu = long_from_mi(yu)
    for kw in [dict(strategy="last"), dict(strategy="drift"), dict(strategy="last", sp=7)]:
        row = {}
        try:
            sk = _sk_predict(yu, kw, fh)
            row["sktime_rows"] = len(sk)
            for b in ["polars", "duckdb"]:
                m = NwPanelNaive(**kw).fit(make_native(lngu, b), keys=["h0"], time="time")
                row[b] = _cmp(sk, to_pandas_any(m.predict(fh)))
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"[:300]
        RES["C"]["unequal_end " + json.dumps(kw)] = row
        print("C unequal", kw, row)

    # monthly, month-end anchored, ending in February (calendar offsets)
    ym = make_panel(5, 26, n_levels=1, freq="ME" if pd.__version__ >= "2.2" else "M")
    lngm = long_from_mi(ym)
    row = {"last_time": str(ym.index.get_level_values(-1).max())}
    sk = _sk_predict(ym, dict(strategy="last"), [1, 2, 3])
    row["sktime_future"] = [str(t) for t in sk.loc["s0"].index]
    for b in ["pandas_mi", "polars", "duckdb", "pyarrow", "dask"]:
        try:
            if b == "pandas_mi":
                out = NwPanelNaive("last").fit(ym).predict([1, 2, 3]).reset_index()
            else:
                m = NwPanelNaive("last").fit(make_native(lngm, b), keys=["h0"], time="time")
                out = to_pandas_any(m.predict([1, 2, 3]))
            row[b] = [str(t) for t in out[out["h0"] == "s0"]["time"]]
            row[b + "_matches"] = row[b] == row["sktime_future"]
        except Exception as e:  # noqa: BLE001
            row[b] = f"ERR {type(e).__name__}: {e}"[:150]
    RES["C"]["monthly_ME"] = row
    print("C monthly", row)


if __name__ == "__main__":
    test_A()
    test_B()
    test_C()
    tag = "pd3" if pd.__version__.startswith("3") else "pd2"
    p = ED / "results" / f"x5_test_{tag}.json"
    p.write_text(json.dumps(RES, indent=1, default=str))
    print("wrote", p)
