"""g4: long/tidy proba schema - correctness vs stock and conversion cost.

A. correctness: stock NaiveForecaster / ThetaForecaster predict_interval,
   predict_quantiles, predict_var (Panel + Hierarchical) -> long (pandas, polars)
   -> today's pandas MultiIndex layout, assert_frame_equal (exact).
   Native polars Naive(last) interval kernel vs stock, interval->quantile relabel.
B. cost at N in {1k, 10k} series (fh=1..12, coverage [.8,.9]), interleaved
   same-process min-of-R timings, reported as ms and ratios.
C. distributions: current _concat_distributions_hierarchical vs one Normal from a
   long (mu, sigma) table.

Usage:
  PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
    $G4_PY -W ignore experiments/container_independence/scripts/g4_proba_bench.py \
    [--ncorr 1000] [--ntheta 200] [--sizes 1000,10000] [--rounds 5]
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.join(HERE, "..")
sys.path.insert(0, ED)
sys.path.insert(0, os.path.join(ED, "protos"))

import sktime.utils.warnings as _sw  # noqa: E402

_sw.warn = lambda *a, **k: None
from common import make_panel, make_panel_fast, rss_mb  # noqa: E402
from g4_proba_schema import (  # noqa: E402
    ProbaSpec,
    canonical_sort,
    flat_names,
    flat_to_pandas,
    interval_long_to_quantiles_long_pl,
    long_to_pandas,
    naive_last_interval_polars,
    pandas_to_long,
)

from sktime.datatypes import convert_to  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402
from sktime.forecasting.theta import ThetaForecaster  # noqa: E402

OUT = {"A": {}, "B": {}, "C": {}}
COV = [0.9, 0.8]  # deliberately non-sorted: order must be preserved
FH3 = [1, 2, 3]


def log(sec, key, val):
    OUT[sec][key] = val
    print(sec, key, json.dumps(val, default=str))


def exact_equal(a, b):
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True, check_freq=False)
        return True
    except AssertionError as e:
        return f"NOT EQUAL: {str(e)[:300]}"


def interleaved(fns, rounds=5):
    """min-of-rounds seconds per fn, fns run round-robin in one process."""
    best = {k: float("inf") for k in fns}
    for k, f in fns.items():  # warmup
        f()
    for _ in range(rounds):
        for k, f in fns.items():
            gc.collect()
            t0 = time.perf_counter()
            f()
            best[k] = min(best[k], time.perf_counter() - t0)
    return best


# ----------------------------------------------------------------------------- A
def roundtrip_checks(tag, f):
    res = {}
    outs = {
        "interval": f.predict_interval(coverage=COV),
        "quantiles": f.predict_quantiles(alpha=[0.05, 0.5, 0.95, 0.1]),
        "var": f.predict_var(),
    }
    for kind, ref in outs.items():
        for be in ["pandas", "polars"]:
            long, spec = pandas_to_long(ref, kind, backend=be)
            back = long_to_pandas(long, spec)
            r = exact_equal(back, ref)
            # shuffled rows -> canonical_sort fallback
            if be == "polars":
                shuf = long.sample(fraction=1.0, shuffle=True, seed=1)
            else:
                shuf = long.sample(frac=1.0, random_state=1)
            back2 = long_to_pandas(shuf, spec)
            r2 = exact_equal(back2, ref)
            res[f"{kind}/{be}"] = {"roundtrip": r, "shuffled": r2}
    log("A", tag, res)
    return outs


def part_a(ncorr, ntheta):
    # Panel, Naive, 1k
    y = make_panel_fast(ncorr, 60)
    t0 = time.perf_counter()
    f = NaiveForecaster(strategy="last").set_config(remember_data=True).fit(y, fh=FH3)
    outs = roundtrip_checks(f"naive_panel_n{ncorr}", f)
    log("A", "naive_stock_fit_plus_3_proba_calls_sec", time.perf_counter() - t0)
    # Hierarchical, Naive
    yh = make_panel(n_series=50, n_time=60, n_levels=3)
    fh_ = NaiveForecaster().set_config(remember_data=True).fit(yh, fh=FH3)
    roundtrip_checks("naive_hier_3lvl", fh_)
    # unnamed levels (None names) + int time
    yu = make_panel_fast(20, 30, time_kind="int")
    yu.index = yu.index.set_names([None, None])
    fu = NaiveForecaster().set_config(remember_data=True).fit(yu, fh=FH3)
    roundtrip_checks("naive_panel_unnamed_int_time", fu)
    # multivariate (2 vars, column vectorization)
    y2 = make_panel_fast(30, 40, n_cols=2)
    f2 = NaiveForecaster().set_config(remember_data=True).fit(y2, fh=FH3)
    roundtrip_checks("naive_panel_2vars", f2)
    # Theta
    yt = make_panel_fast(ntheta, 60) + 1000.0
    t0 = time.perf_counter()
    ft = ThetaForecaster(sp=1).set_config(remember_data=True).fit(yt, fh=FH3)
    roundtrip_checks(f"theta_panel_n{ntheta}", ft)
    log("A", "theta_stock_fit_plus_3_proba_calls_sec", time.perf_counter() - t0)

    # native polars Naive interval kernel vs stock
    ypl = convert_to(y, "polars_panel")
    long_native = naive_last_interval_polars(
        ypl, FH3, COV, ["__index__h0"], "__index__time", ["c0"]
    ).collect()
    spec = ProbaSpec(
        "interval", ["h0"], "time", ["c0"], COV, ["__index__h0"], "__index__time"
    )
    nat_pd = long_to_pandas(long_native, spec)
    ref = outs["interval"]
    idx_eq = nat_pd.index.equals(ref.index)
    cols_eq = nat_pd.columns.equals(ref.columns)
    maxdiff = float(np.nanmax(np.abs(nat_pd.to_numpy() - ref.to_numpy())))
    try:
        pd.testing.assert_frame_equal(nat_pd, ref, check_freq=False, rtol=1e-12)
        close = True
    except AssertionError as e:
        close = str(e)[:300]
    log(
        "A",
        f"native_polars_naive_interval_vs_stock_n{ncorr}",
        {"index_equal": idx_eq, "columns_equal": cols_eq, "max_abs_diff": maxdiff,
         "assert_close_rtol1e-12": close},
    )
    # interval -> quantiles relabel (polars) vs stock predict_quantiles
    q_long = interval_long_to_quantiles_long_pl(long_native)
    alphas = [0.05, 0.1, 0.9, 0.95]
    qspec = ProbaSpec(
        "quantiles", ["h0"], "time", ["c0"], alphas, ["__index__h0"], "__index__time"
    )
    q_pd = long_to_pandas(canonical_sort(q_long, qspec), qspec)
    q_ref = f.predict_quantiles(alpha=alphas)
    try:
        pd.testing.assert_frame_equal(q_pd, q_ref, check_freq=False, rtol=1e-12)
        qclose = True
    except AssertionError as e:
        qclose = str(e)[:300]
    log("A", "relabel_interval_to_quantiles_vs_stock", qclose)
    # unrounded relabel pitfall
    log("A", "float_key_pitfall_0.5-0.9/2", repr(0.5 - 0.9 / 2))
    return f, y


# ----------------------------------------------------------------------------- B
def synth_interval(n, h=12, nvar=1, cov=COV, seed=0):
    rng = np.random.default_rng(seed)
    inst = pd.Index([f"s{i:07d}" for i in range(n)], name="h0")
    t = pd.date_range("2001-01-01", periods=h, freq="D")
    idx = pd.MultiIndex.from_product([inst, t], names=["h0", "time"])
    cols = pd.MultiIndex.from_product([[f"c{j}" for j in range(nvar)], cov,
                                       ["lower", "upper"]])
    return pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx,
                        columns=cols)


def part_b(sizes, rounds, f1k_holder):
    for n in sizes:
        for nvar in ([1, 3] if n >= 10000 else [1]):
            ref = synth_interval(n, nvar=nvar)
            lp, spec = pandas_to_long(ref, "interval", "pandas")
            ll, _ = pandas_to_long(ref, "interval", "polars")
            shuf = ll.sample(fraction=1.0, shuffle=True, seed=1)
            wide = pl.from_pandas(
                ref.set_axis(flat_names(spec), axis=1).reset_index()
            ).rename({"h0": "h0", "time": "time"})
            spec_w = ProbaSpec("interval", ["h0"], "time", spec.variables, COV,
                               ["h0"], "time")
            assert exact_equal(long_to_pandas(ll, spec), ref) is True
            assert exact_equal(flat_to_pandas(wide, spec_w), ref) is True
            # today's reconstruct: per-instance MI frames, pd.concat(keys=...)
            per_inst = [g.droplevel(0) for _, g in ref.groupby(level=0, sort=False)]
            keys = ref.index.get_level_values(0).unique()
            per_np = [p.to_numpy() for p in per_inst]
            times = [p.index.to_numpy() for p in per_inst]
            vnames = list(spec.variables)

            def concat_today():
                return pd.concat(per_inst, keys=keys, axis=0)

            def concat_long_numpy():
                # per-instance numpy blocks -> one polars long table
                m = len(vnames) * len(COV)
                blk = np.concatenate(per_np).reshape(-1, 2)
                lens = np.fromiter((len(a) for a in per_np), np.int64, len(per_np))
                return pl.DataFrame({
                    "h0": np.repeat(np.asarray(keys), lens * m),
                    "time": np.repeat(np.concatenate(times), m),
                    "variable": np.tile(
                        np.repeat(np.asarray(vnames, dtype=object), len(COV)),
                        int(lens.sum())).astype(str),
                    "coverage": np.tile(np.asarray(COV * len(vnames)), int(lens.sum())),
                    "lower": blk[:, 0], "upper": blk[:, 1],
                })

            def get_columns_per_instance():
                for _ in range(n):
                    pd.MultiIndex.from_product([vnames, COV, ["lower", "upper"]])

            fns = {
                "pd_MI_to_long_pandas": lambda: pandas_to_long(ref, "interval", "pandas"),
                "pd_MI_to_long_polars": lambda: pandas_to_long(ref, "interval", "polars"),
                "long_pandas_to_MI_guarded": lambda: long_to_pandas(lp, spec),
                "long_pandas_to_MI_unguarded": lambda: long_to_pandas(lp, spec, guard=False),
                "long_polars_to_MI_guarded": lambda: long_to_pandas(ll, spec),
                "long_polars_to_MI_unguarded": lambda: long_to_pandas(ll, spec, guard=False),
                "long_polars_shuffled_to_MI": lambda: long_to_pandas(shuf, spec),
                "wide_flat_polars_to_MI": lambda: flat_to_pandas(wide, spec_w),
                "polars_long_to_pandas_long": lambda: ll.to_pandas(),
                "today_reconstruct_pd_concat_keys": concat_today,
                "long_from_per_instance_numpy": concat_long_numpy,
                "today_get_columns_x_n_instances": get_columns_per_instance,
            }
            t = interleaved(fns, rounds=rounds if n < 10000 else max(3, rounds - 2))
            ms = {k: round(v * 1e3, 3) for k, v in t.items()}
            base = t["today_reconstruct_pd_concat_keys"]
            ratios = {k: round(v / base, 4) for k, v in t.items()}
            log("B", f"n{n}_vars{nvar}", {"ms": ms, "ratio_vs_today_reconstruct": ratios,
                                          "rows_wide": len(ref), "rows_long": ll.height,
                                          "rss_mb": rss_mb()})

        # native kernel vs stock vectorized predict_interval
        y = make_panel_fast(n, 60)
        ypl = convert_to(y, "polars_panel")
        spec = ProbaSpec("interval", ["h0"], "time", ["c0"], COV,
                         ["__index__h0"], "__index__time")

        def native_long():
            return naive_last_interval_polars(
                ypl, list(range(1, 13)), COV, ["__index__h0"], "__index__time", ["c0"]
            ).collect()

        def native_to_pd():
            return long_to_pandas(native_long(), spec)

        tn = interleaved({"native_polars_long": native_long,
                          "native_polars_plus_to_MI": native_to_pd}, rounds=3)
        rec = {k: round(v * 1e3, 3) for k, v in tn.items()}
        if n <= 1000:
            f = NaiveForecaster(strategy="last").set_config(remember_data=True)
            f.fit(y, fh=list(range(1, 13)))
            t0 = time.perf_counter()
            ref = f.predict_interval(coverage=COV)
            rec["stock_predict_interval_ms_single_run"] = round(
                (time.perf_counter() - t0) * 1e3, 1)
            t0 = time.perf_counter()
            f.predict_var()
            rec["stock_predict_var_ms_single_run"] = round(
                (time.perf_counter() - t0) * 1e3, 1)
            rec["native_equal_stock_rtol1e-12"] = bool(np.allclose(
                native_to_pd().to_numpy(), ref.to_numpy(), rtol=1e-12, atol=0))
        log("B", f"native_kernel_n{n}", rec)


# ----------------------------------------------------------------------------- C
def part_c(f, y, nd):
    from sktime.forecasting.base._base_proba import BaseProbaForecaster

    try:
        from skpro.distributions import Normal
    except ImportError:  # pragma: no cover
        from sktime.base._proba._normal import Normal

    fs = f.forecasters_.iloc[:nd, 0]
    t0 = time.perf_counter()
    dists = [fi.predict_proba() for fi in fs]
    t_per = time.perf_counter() - t0
    row_idx = f.forecasters_.index[:nd]

    class _Schema:
        def get_iter_indices(self):
            return row_idx, None

    helper = BaseProbaForecaster.__new__(BaseProbaForecaster)

    def concat_today():
        return BaseProbaForecaster._concat_distributions_hierarchical(
            helper, dists, _Schema())

    # long (mu, sigma) table -> one Normal
    mu = np.concatenate([d.mu.to_numpy().ravel() if hasattr(d.mu, "to_numpy")
                         else np.asarray(d.mu).ravel() for d in dists])
    sg = np.concatenate([d.sigma.to_numpy().ravel() if hasattr(d.sigma, "to_numpy")
                         else np.asarray(d.sigma).ravel() for d in dists])
    lens = [len(d.index) for d in dists]
    long = pl.DataFrame({
        "h0": np.repeat(np.asarray(row_idx), lens),
        "time": np.concatenate([d.index.to_numpy() for d in dists]),
        "variable": ["c0"] * int(sum(lens)), "mu": mu, "sigma": sg})

    def from_long():
        idx = pd.MultiIndex.from_arrays(
            [long.get_column("h0").to_numpy(),
             long.get_column("time").to_numpy().astype("datetime64[ns]")],
            names=["h0", "time"])
        return Normal(mu=long.get_column("mu").to_numpy().reshape(-1, 1),
                      sigma=long.get_column("sigma").to_numpy().reshape(-1, 1),
                      index=idx, columns=pd.Index(["c0"]))

    try:
        d_today = concat_today()
        today_ok = True
    except Exception as e:  # noqa: BLE001
        d_today, today_ok = None, repr(e)[:300]
    d_long = from_long()
    eq_ = None
    if d_today is not None:
        eq_ = {
            "index_equal": d_today.index.equals(d_long.index),
            "mean_equal": bool(np.allclose(d_today.mean().to_numpy(),
                                           d_long.mean().to_numpy(), rtol=0, atol=0)),
            "var_equal": bool(np.allclose(d_today.var().to_numpy(),
                                          d_long.var().to_numpy(), rtol=1e-15)),
        }
    t = interleaved({"concat_distributions_today": concat_today,
                     "normal_from_long_table": from_long}, rounds=3) if today_ok is True \
        else {"normal_from_long_table": min(
            (lambda: (lambda t0: (from_long(), time.perf_counter() - t0)[1])(
                time.perf_counter()))() for _ in range(3))}
    log("C", f"dist_concat_n{nd}", {
        "dist_type": type(dists[0]).__module__ + "." + type(dists[0]).__name__,
        "per_instance_predict_proba_sec_total": round(t_per, 3),
        "today_concat_ok": today_ok, "equal": eq_,
        "ms": {k: round(v * 1e3, 3) for k, v in t.items()}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncorr", type=int, default=1000)
    ap.add_argument("--ntheta", type=int, default=200)
    ap.add_argument("--sizes", default="1000,10000")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--ndist", type=int, default=1000)
    ap.add_argument("--parts", default="ABC")
    a = ap.parse_args()
    f, y = None, None
    if "A" in a.parts:
        f, y = part_a(a.ncorr, a.ntheta)
    if "B" in a.parts:
        part_b([int(s) for s in a.sizes.split(",")], a.rounds, None)
    if "C" in a.parts:
        if f is None:
            y = make_panel_fast(a.ndist, 60)
            f = NaiveForecaster().set_config(remember_data=True).fit(y, fh=FH3)
        part_c(f, y, min(a.ndist, len(f.forecasters_)))
    fn = os.path.join(ED, "results", f"g4_proba_bench_{a.parts}.json")
    with open(fn, "w") as fo:
        json.dump(OUT, fo, indent=1, default=str)
    print("wrote", fn)


if __name__ == "__main__":
    main()
