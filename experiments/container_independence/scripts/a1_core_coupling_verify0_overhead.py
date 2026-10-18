"""a1 verify0: independent (subtractive, uninstrumented) check of claim a1.5.

a1 claims: vectorized NaiveForecaster.fit on a Panel spends 13.5-21.7% in the
VectorizedDF container layer, 62-84% in per-slice public-API re-entry, ~0.3% inner.
That was measured with an exclusive timer stack (wrappers on ~30 functions).

Here, no instrumentation; we time separately, interleaved, min-of-repeats:
  T_total   NaiveForecaster(strategy="last").fit(y, fh)              (public API)
  T_outer   check_is_scitype(y) on the WHOLE panel (the outer boundary check,
            which a1 buckets as "mtype-check" i.e. "re-entry")
  T_prep    prepare_VectorizedDF(y) (schema build, container)
  T_slices  materialise all per-instance slices via groupby(level, sort=False)
            + droplevel (container iterate, w/o freq enforcement)
  T_trusted "trusted inner path": for each pre-made slice (pd.Series w/ freq),
            clone + set _y/_fh/_cutoff + call _fit directly (no check/convert)
  T_pub1    for each pre-made slice: clone().fit(slice, fh) via the public API
  => re-entry_est = T_pub1 - T_trusted ; container_est = T_prep + T_slices
  => residual     = T_total - (T_outer + container_est + T_pub1)  (pivot, glue...)

Run from WT root:
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
     experiments/container_independence/scripts/a1_core_coupling_verify0_overhead.py --grid dev
"""

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
from common import bench, env_info, make_panel_fast  # noqa: E402

from sktime.datatypes import check_is_scitype, prepare_VectorizedDF  # noqa: E402
from sktime.forecasting.base import ForecastingHorizon  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--grid", choices=["dev", "large"], default="dev")
a = ap.parse_args()
SIZES = [(100, 100), (1000, 100)] if a.grid == "dev" else [(1000, 500), (10_000, 100)]
FH = [1, 2, 3]
out = {"env": env_info(), "grid": a.grid, "runs": []}


def slices_of(y):
    res = []
    for k, g in y.groupby(level=0, sort=False):
        s = g.droplevel(0)
        s.index.freq = pd.infer_freq(s.index)
        res.append(s.iloc[:, 0])
    return res


for ns, nt in SIZES:
    y = make_panel_fast(ns, nt)
    proto = NaiveForecaster(strategy="last")
    sl = slices_of(y)
    fh_abs = None

    def total():
        return NaiveForecaster(strategy="last").fit(y, fh=FH)

    def outer():
        return check_is_scitype(y, scitype=["Series", "Panel", "Hierarchical"],
                                return_metadata=[])

    def prep():
        return prepare_VectorizedDF(y, iterate_as="Series", iterate_cols=True) \
            if "iterate_cols" in prepare_VectorizedDF.__code__.co_varnames \
            else prepare_VectorizedDF(y, iterate_as="Series")

    def slices():
        return slices_of(y)

    def trusted():
        for s in sl:
            f = proto.clone()
            f._y = s
            f._X = None
            f._cutoff = s.index[[-1]]
            f._fh = ForecastingHorizon(FH, is_relative=True, freq=s.index.freq)
            f._fit(y=s, X=None, fh=f._fh)
            f._is_fitted = True

    def pub1():
        for s in sl:
            proto.clone().fit(s, fh=FH)

    fns = {"total": total, "outer_check": outer, "prepare": prep,
           "slices": slices, "trusted": trusted, "pub_per_slice": pub1}
    rep = 3 if ns <= 100 else 2
    # interleave: one round of all, repeated, keep min
    best = {k: float("inf") for k in fns}
    for k, fn in fns.items():
        fn()  # warmup
    for _ in range(rep):
        for k, fn in fns.items():
            best[k] = min(best[k], bench(fn, repeat=1, warmup=0))
    T = best
    container = T["prepare"] + T["slices"]
    reentry = T["pub_per_slice"] - T["trusted"]
    resid = T["total"] - (T["outer_check"] + container + T["pub_per_slice"])
    rec = {
        "n_series": ns, "n_time": nt, "seconds": T,
        "share_of_total_pct": {
            "outer_check (whole panel)": 100 * T["outer_check"] / T["total"],
            "container (prepare+slices)": 100 * container / T["total"],
            "per-slice re-entry (pub - trusted)": 100 * reentry / T["total"],
            "trusted inner (clone+_fit)": 100 * T["trusted"] / T["total"],
            "residual (pivot/glue/other)": 100 * resid / T["total"],
        },
        "ratio_pub_per_slice_over_trusted": T["pub_per_slice"] / T["trusted"],
    }
    out["runs"].append(rec)
    print(f"\n== n_series={ns} n_time={nt}")
    for k, v in T.items():
        print(f"   {k:16s} {v:8.4f}s  {1e3 * v / ns:7.3f} ms/inst")
    for k, v in rec["share_of_total_pct"].items():
        print(f"   share {k:38s} {v:6.1f}%")
    print(f"   pub-per-slice / trusted = {rec['ratio_pub_per_slice_over_trusted']:.1f}x")

json.dump(out, open(f"experiments/container_independence/results/"
                    f"a1_core_coupling_verify0_overhead_{a.grid}.json", "w"), indent=1)
