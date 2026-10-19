"""G1: where does time go per phase, stock vs trusted (cProfile, shares only).

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $PY -W ignore \
        experiments/container_independence/scripts/g1_profile.py --n 300 --est naive
Prints, per phase, the top cumulative functions (cProfile inflates ~2x; use for
shares / call counts only).
"""

import argparse
import cProfile
import io
import os
import pstats
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
sys.path.insert(0, os.path.dirname(__file__))

import g1_trusted as g1  # noqa: E402
from g1_trusted_bench import COV, FH, make_data, make_estimator  # noqa: E402

KEYS = [
    "check_is_scitype", "check_is_mtype", "convert_to", "infer_mtype", "get_cutoff",
    "_check_fh", "check_fh", "clone", "_reset_at", "items", "_enforce_index_freq",
    "infer_freq", "groupby", "_fit", "_predict", "_predict_interval", "_predict_var",
    "_update", "fit_transform", "transform", "reconstruct", "concat", "deepcopy",
    "_slice_metadata", "trusted_fit_slice", "trusted_predict_slice",
    "trusted_update_slice", "_estimators_frame", "update_data", "warn",
]


def prof(fn):
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    st = pstats.Stats(pr)
    tot = max(v[3] for v in st.stats.values())
    agg = {}
    for (fname, _, func), (cc, nc, tt, ct, _) in st.stats.items():
        if func in KEYS:
            a = agg.setdefault(func, [0, 0.0])
            a[0] += nc
            a[1] = max(a[1], ct)  # cumulative of outermost (approx; recursion)
    return tot, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--est", default="naive")
    ap.add_argument("--sc", default="Panel")
    a = ap.parse_args()
    y_tr, y_new = make_data(a.sc, a.n, a.T)
    proto = make_estimator(a.est)
    for mode in ("stock", "trusted"):
        if mode == "trusted":
            g1.install()
        else:
            g1.uninstall()
        f = proto.clone()
        # warm
        proto.clone().fit(y_tr.iloc[:400], fh=FH).predict()
        phases = [
            ("fit", lambda: f.fit(y_tr, fh=FH)),
            ("predict", lambda: f.predict()),
        ]
        if proto.get_tag("capability:pred_int"):
            phases.append(("predict_interval", lambda: f.predict_interval(coverage=COV)))
        phases.append(("update", lambda: f.update(y_new)))
        for ph, fn in phases:
            tot, agg = prof(fn)
            print(f"== {a.est} {a.sc} n={a.n} {mode} {ph}: total(cProfile) {tot:.3f}s")
            for k, (nc, ct) in sorted(agg.items(), key=lambda x: -x[1][1])[:14]:
                print(f"   {k:24s} calls/inst={nc / a.n:6.1f} cum={ct:.3f}s ({100 * ct / tot:4.1f}%)")
        g1.uninstall()


if __name__ == "__main__":
    main()
