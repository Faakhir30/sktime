"""x8 (aux): who calls the per-instance checks / conversions, and how often?

cProfile caller breakdown of ``mtype``, ``check_is_scitype``, ``check_is_mtype``,
``convert_to`` for the vectorized path, normalised per instance.

Run (from worktree root)::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x8_callers.py --n 300
"""

import argparse
import cProfile
import json
import os
import pstats
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from x8_e2e_profile import _short, make_containers, make_data, make_estimator  # noqa


def callers_of(st, fsub, fname):
    out = []
    for (fn, ln, func), (cc, nc, tt, ct, callers) in st.stats.items():
        if func == fname and fn.replace(os.sep, "/").endswith(fsub):
            for (cfn, cln, cfunc), (c_cc, c_nc, c_tt, c_ct) in callers.items():
                out.append(
                    {
                        "caller": f"{_short(cfn)}:{cln}({cfunc})",
                        "ncalls": c_nc,
                        "cumtime": c_ct,
                    }
                )
    out.sort(key=lambda r: -r["cumtime"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    warnings.simplefilter("ignore")
    res = {"n": a.n, "T": a.T, "runs": {}}
    for scitype in ["Panel"]:
        y = make_data(scitype, a.n, a.T)
        conts = make_containers(y, scitype, ["pd", "pl"])
        for est_name in ["naive", "poly"]:
            est = make_estimator(est_name)
            for c, d in conts.items():
                est.clone().fit(d, fh=[1, 2, 3]).predict()
                pr = cProfile.Profile()
                f = est.clone()
                pr.enable()
                f.fit(d, fh=[1, 2, 3])
                f.predict()
                pr.disable()
                st = pstats.Stats(pr)
                key = f"{scitype}/{est_name}/{c}"
                r = {"total_calls": st.total_calls, "total_tt": st.total_tt}
                for lab, (fs, fn) in {
                    "mtype": ("datatypes/_check.py", "mtype"),
                    "check_is_scitype": ("datatypes/_check.py", "check_is_scitype"),
                    "check_is_mtype": ("datatypes/_check.py", "check_is_mtype"),
                    "convert_to": ("datatypes/_convert.py", "convert_to"),
                    "infer_freq": ("pandas/tseries/frequencies.py", "infer_freq"),
                    "get_cutoff": ("datatypes/_utilities.py", "get_cutoff"),
                }.items():
                    rows = callers_of(st, fs, fn)
                    r[lab] = {
                        "ncalls_total": sum(x["ncalls"] for x in rows),
                        "ncalls_per_instance": sum(x["ncalls"] for x in rows) / a.n,
                        "callers": rows[:8],
                    }
                res["runs"][key] = r
                print(key, {k: (round(v["ncalls_per_instance"], 2)) for k, v in r.items() if isinstance(v, dict)})
    out = a.out or os.path.join(os.path.dirname(HERE), "results", "x8_callers.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
