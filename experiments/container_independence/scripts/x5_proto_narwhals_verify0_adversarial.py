"""x5 verifier: adversarial correctness checks of NwPanelNaive / NwVectorizedDF vs sktime.

Independent comparator: outer-merge on (keys, time) of sktime output vs proto output,
all value columns, NaN-aware -- not positional sort-and-compare on c0 only.

Cases
  F1 period index (M, Q) pandas-MI path
  F2 int time index
  F3 hierarchical 3-level, all backends
  F4 multi-column with scattered NaN (incl. NaN at end of series), all backends
  F5 unequal lengths + unequal ends, n_series >= 1000 (pyarrow over-bug zone), all backends
  F6 rows shuffled (within and between series) for long-format backends
  F7 series shorter than sp, all-NaN series
  F8 freq coverage: h, W-SUN, B, MS, QE, YE (pandas_mi + polars)
  F9 drift with NaN at window ends (sktime raises, proto?)
  V1 NwVectorizedDF items vs sktime: hierarchical, interleaved rows, unequal lengths
  V2 lazy instance order vs eager order

Run:
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
     experiments/container_independence/scripts/x5_proto_narwhals_verify0_adversarial.py [--n-big 1000]
"""

import argparse
import json
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common import env_info, make_panel
from protos.x5_backends import BACKENDS, long_from_mi, make_native, to_pandas_any
from protos.x5_narwhals import prepare_nw_vectorized
from protos.x5_nw_forecasters import NwPanelNaive

warnings.filterwarnings("ignore")
ED = Path(__file__).resolve().parents[1]
FH = list(range(1, 13))
RES = {"env": env_info(), "cases": {}}


def sk_predict(y, kw, fh=FH):
    from sktime.forecasting.naive import NaiveForecaster

    f = NaiveForecaster(**kw)
    f.set_config(remember_data=False)
    return f.fit(y).predict(fh)


def compare(sk, got, keys, time="time"):
    """Outer merge on keys+time; compare all value columns NaN-aware."""
    s = sk.reset_index()
    s.columns = [*keys, time, *[str(c) for c in sk.columns]]
    g = got.reset_index() if isinstance(got.index, pd.MultiIndex) else got.copy()
    g.columns = [str(c) for c in g.columns]
    for d in (s, g):
        if isinstance(d[time].dtype, pd.PeriodDtype):
            d[time] = d[time].dt.to_timestamp()
        if not pd.api.types.is_integer_dtype(d[time].dtype):
            d[time] = pd.to_datetime(d[time]).astype("datetime64[ns]")
        for k in keys:
            d[k] = d[k].astype(str)
    vcols = [c for c in s.columns if c not in [*keys, time]]
    m = s.merge(g, on=[*keys, time], how="outer", suffixes=("_sk", "_nw"), indicator=True)
    only_sk = int((m["_merge"] == "left_only").sum())
    only_nw = int((m["_merge"] == "right_only").sum())
    out = {"rows_sk": len(s), "rows_nw": len(g), "only_sk": only_sk, "only_nw": only_nw}
    md, nanmis = 0.0, 0
    for v in vcols:
        a = m[v + "_sk"].to_numpy(float)
        b = m[v + "_nw"].to_numpy(float) if v + "_nw" in m else np.full(len(m), np.nan)
        both = m["_merge"] == "both"
        a, b = a[both], b[both]
        nanmis += int((np.isnan(a) != np.isnan(b)).sum())
        d = np.abs(a - b)
        d = d[~np.isnan(d)]
        if len(d):
            md = max(md, float(d.max()))
    out["max_abs_diff"] = md
    out["nan_mismatch"] = nanmis
    out["ok"] = only_sk == 0 and only_nw == 0 and md < 1e-9 and nanmis == 0
    return out


def run_backends(name, y_mi, kw, keys, backends, fh=FH, freq=None, shuffle_long=False):
    try:
        sk = sk_predict(y_mi, kw, fh)
    except Exception as e:  # noqa: BLE001
        sk = None
        sk_err = f"{type(e).__name__}: {e}"[:200]
    row = {}
    if sk is None:
        row["sktime_error"] = sk_err
    lng = long_from_mi(y_mi)
    lng.columns = [*keys, "time", *[str(c) for c in y_mi.columns]]
    if shuffle_long:
        lng = lng.sample(frac=1.0, random_state=1).reset_index(drop=True)
    for b in backends:
        try:
            if b == "pandas_mi":
                got = NwPanelNaive(**kw).fit(y_mi, freq=freq).predict(fh)
                gotp = got
            else:
                nat = make_native(lng, b)
                m = NwPanelNaive(**kw).fit(nat, keys=keys, time="time", freq=freq)
                out = m.predict(fh)
                gotp = to_pandas_any(out)
            if sk is None:
                row[b] = {"ok": None, "proto_rows": len(gotp),
                          "proto_nan_frac": float(gotp.select_dtypes("number").isna().mean().mean())}
            else:
                row[b] = compare(sk, gotp, keys)
        except Exception as e:  # noqa: BLE001
            row[b] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    RES["cases"][name] = row
    summ = {b: (r.get("ok") if isinstance(r, dict) else r) for b, r in row.items()}
    print(name, summ, flush=True)
    bad = {b: r for b, r in row.items() if isinstance(r, dict) and r.get("ok") is False}
    if bad:
        print("   fails:", json.dumps(bad, default=str)[:600], flush=True)
    return row


STRATS = {
    "last": dict(strategy="last"),
    "last_sp3": dict(strategy="last", sp=3),
    "mean": dict(strategy="mean"),
    "mean_sp3": dict(strategy="mean", sp=3),
    "mean_wl5": dict(strategy="mean", window_length=5),
    "drift": dict(strategy="drift"),
}
ALL_B = ["pandas_mi", *BACKENDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-big", type=int, default=1000)
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    args = ap.parse_args()
    nbig = args.n_big if args.grid == "dev" else max(args.n_big, 3000)

    # F1 period
    for fq in ["M", "Q"]:
        y = make_panel(8, 30, n_levels=1, time_kind="period", freq=fq)
        for s in ["last", "last_sp3", "mean", "drift"]:
            run_backends(f"F1_period_{fq}|{s}", y, STRATS[s], ["h0"],
                         ["pandas_mi", "polars", "duckdb"])

    # F2 int
    y = make_panel(8, 30, n_levels=1, time_kind="int")
    for s in ["last", "last_sp3", "mean", "drift"]:
        run_backends(f"F2_int|{s}", y, STRATS[s], ["h0"], ALL_B)

    # F3 hierarchical
    y = make_panel(27, 30, n_levels=3)
    for s in ["last", "last_sp3", "mean_sp3", "drift"]:
        run_backends(f"F3_hier3|{s}", y, STRATS[s], ["h0", "h1", "h2"], ALL_B)

    # F4 multi-column, scattered NaN incl. at series ends, different per column
    rng = np.random.default_rng(3)
    y = make_panel(30, 40, n_cols=3, n_levels=1)
    v = y.to_numpy().copy()
    mask = rng.random(v.shape) < 0.15
    v[mask] = np.nan
    # last 2 obs NaN in col 1 of series s0..s4
    for i in range(5):
        v[i * 40 + 38: i * 40 + 40, 1] = np.nan
    y = pd.DataFrame(v, index=y.index, columns=y.columns)
    for s in ["last", "last_sp3", "mean", "mean_sp3", "mean_wl5"]:
        run_backends(f"F4_multicol_nan|{s}", y, STRATS[s], ["h0"], ALL_B)

    # F5 unequal lengths/ends at big n (pyarrow over-bug zone)
    yb = make_panel(nbig, 30, n_levels=1)
    keep = []
    rng = np.random.default_rng(5)
    for i, k in enumerate(yb.index.get_level_values(0).unique()):
        start = int(rng.integers(0, 8))
        stop = 30 - int(rng.integers(0, 8))
        keep.append(yb.loc[[k]].iloc[start:stop])
    yu = pd.concat(keep)
    for s in ["last", "last_sp3", "mean_wl5", "drift"]:
        run_backends(f"F5_unequal_n{nbig}|{s}", yu, STRATS[s], ["h0"], ALL_B)

    # F6 shuffled rows (long backends) - small n and big n
    y = make_panel(40, 30, n_levels=1)
    for s in ["last", "last_sp3", "drift"]:
        run_backends(f"F6_shuffled|{s}", y, STRATS[s], ["h0"], BACKENDS, shuffle_long=True)
    for s in ["last", "drift"]:
        run_backends(f"F6_shuffled_n{nbig}|{s}", yb, STRATS[s], ["h0"],
                     ["pandas", "polars", "pyarrow", "duckdb"], shuffle_long=True)

    # F7 short series (< sp) and all-NaN series
    y = make_panel(6, 20, n_levels=1)
    y = pd.concat([y.loc[["s0"]].iloc[:4], y.loc[["s1", "s2", "s3", "s4", "s5"]]])
    y.loc["s5", "c0"] = np.nan
    for s, kw in [("last_sp7", dict(strategy="last", sp=7)), ("mean_sp7", dict(strategy="mean", sp=7)), ("last", STRATS["last"])]:
        run_backends(f"F7_short_allnan|{s}", y, kw, ["h0"], ALL_B)

    # F8 frequency coverage
    for fq in ["h", "W-SUN", "B", "MS", "QE", "YE"]:
        y = make_panel(4, 20, n_levels=1, freq=fq)
        run_backends(f"F8_freq_{fq}|last", y, STRATS["last"], ["h0"],
                     ["pandas_mi", "polars", "polars_lazy", "duckdb", "pyarrow"])

    # F9 drift with NaN at the window end
    y = make_panel(5, 20, n_levels=1)
    y.iloc[19, 0] = np.nan
    run_backends("F9_drift_nan_end|drift", y, STRATS["drift"], ["h0"], ["pandas_mi", "polars"])

    # V1 NwVectorizedDF vs sktime VectorizedDF on interleaved rows / unequal lengths / hier
    from sktime.datatypes._vectorize import prepare_VectorizedDF

    def vcase(name, X, iterate_as="Series"):
        res = {}
        try:
            is_sc = {1: "Series", 2: "Panel"}.get(X.index.nlevels, "Hierarchical")
            vec, X_mi = prepare_VectorizedDF(X, iterate_as=iterate_as, is_scitype=is_sc)
            sk = list(vec.items(X_mi))
            schema, f = prepare_nw_vectorized(X, iterate_as=iterate_as)
            nw_ = list(schema.items(f))
            res["n"] = (len(sk), len(nw_))
            res["names_equal"] = [a[0] for a in sk] == [b[0] for b in nw_]
            nf = 0
            first = None
            for a, b in zip(sk, nw_):
                try:
                    pd.testing.assert_frame_equal(a[2], b[2], check_freq=True)
                except AssertionError as e:
                    nf += 1
                    first = first or str(e)[:200]
            res["slice_fail"] = nf
            res["first_err"] = first
            r_sk = vec.reconstruct([s for *_, s in sk])
            r_nw = schema.reconstruct(f, [s for *_, s in nw_])
            try:
                pd.testing.assert_frame_equal(r_sk, r_nw)
                res["reconstruct_equal"] = True
            except AssertionError as e:
                res["reconstruct_equal"] = str(e)[:200]
        except Exception as e:  # noqa: BLE001
            res["error"] = f"{type(e).__name__}: {e}"[:300]
        RES["cases"][name] = res
        print(name, res, flush=True)

    yi = make_panel(6, 10, n_levels=1)
    inter = yi.sort_index(level=1)  # rows interleaved across instances (time-major)
    vcase("V1_interleaved_rows", inter)
    vcase("V1_unequal", yu.iloc[: 30 * 20])
    vcase("V1_hier_interleaved", make_panel(8, 10, n_levels=2).sort_index(level=-1))
    vcase("V1_hier_as_panel_interleaved", make_panel(8, 10, n_levels=2).sort_index(level=-1), "Panel")

    # V2 lazy instance order vs eager order
    un = make_panel(5, 6, n_levels=1)
    un = pd.concat([un.loc[[k]] for k in ["s3", "s0", "s4", "s2", "s1"]])
    lng = long_from_mi(un)
    order = {}
    for b in ["polars", "polars_lazy", "pyarrow", "duckdb", "dask"]:
        try:
            sc, f = prepare_nw_vectorized(make_native(lng, b), keys=["h0"], time="time")
            order[b] = [r[0] for r in sc.instance_keys(f).rows()]
        except Exception as e:  # noqa: BLE001
            order[b] = f"ERR {e}"[:100]
    RES["cases"]["V2_instance_order"] = order
    print("V2 order", order)

    tag = "pd3" if pd.__version__.startswith("3") else "pd2"
    p = ED / "results" / f"x5_proto_narwhals_verify0_adversarial_{tag}_{args.grid}.json"
    p.write_text(json.dumps(RES, indent=1, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
