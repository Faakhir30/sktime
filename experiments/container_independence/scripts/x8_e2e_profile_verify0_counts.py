"""x8 verify0: re-count per-instance check/convert calls via sys.setprofile (not cProfile).

Run: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
    experiments/container_independence/scripts/x8_e2e_profile_verify0_counts.py --n 100
"""
import argparse, json, os, sys, warnings
from collections import Counter

def main():
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=100); a = p.parse_args()
    warnings.simplefilter("ignore")
    from common import make_panel_fast
    from sktime.datatypes import convert_to
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.trend import PolynomialTrendForecaster
    targets = {("datatypes/_convert.py", "convert_to"), ("datatypes/_convert.py", "convert"),
               ("datatypes/_check.py", "mtype"), ("datatypes/_check.py", "check_is_mtype"),
               ("datatypes/_check.py", "check_is_scitype"), ("datatypes/_utilities.py", "get_cutoff"),
               ("datatypes/_utilities.py", "infer_freq"), ("base/_base.py", "clone")}
    out = {}
    y = make_panel_fast(a.n, 100); ypl = convert_to(y, "polars_panel")
    for en, est in [("naive", NaiveForecaster(strategy="last")), ("poly", PolynomialTrendForecaster(degree=1))]:
        for cn, d in [("pd", y), ("pl", ypl)]:
            c = Counter()
            def prof(frame, event, arg):
                if event == "call":
                    co = frame.f_code
                    fn = co.co_filename.replace(os.sep, "/")
                    for sub, name in targets:
                        if co.co_name == name and fn.endswith(sub):
                            c[name] += 1
            f = est.clone()
            sys.setprofile(prof)
            try:
                f.fit(d, fh=[1, 2, 3]); f.predict()
            finally:
                sys.setprofile(None)
            out[f"{en}/{cn}"] = {k: v / a.n for k, v in sorted(c.items())}
            print(en, cn, out[f"{en}/{cn}"], flush=True)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "x8_e2e_profile_verify0_counts.json")
    json.dump(out, open(path, "w"), indent=1); print("wrote", path)

if __name__ == "__main__":
    main()
