"""a2_census_verify0: independent recount via a different code path.

- registry count via all_estimators(estimator_types="forecaster") as classes, + BaseForecaster subclass walk
- inner mtype analysis directly from get_class_tag
- fh/cutoff reliance by regex over inspect.getsource of every class in the MRO (below BaseForecaster)
  -- intentionally different from the AST/call-graph method of a2_census.py
- random sample of 25 (seed 20261018) for manual reclassification
Usage: PYTHONPATH=. $EXP_PY -W ignore scripts/a2_census_verify0_recount.py [--grid dev]
"""
import argparse, inspect, re, random, json, sys, csv
from collections import Counter

ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="dev"); ap.parse_args()
from sktime.registry import all_estimators
from sktime.forecasting.base import BaseForecaster

ests = all_estimators("forecaster", return_names=True)
print("all_estimators n =", len(ests))
names = {n for n, _ in ests}

PANDAS = {"pd.Series", "pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"}
nonpd = []
panel = 0
for n, c in ests:
    y = c.get_class_tag("y_inner_mtype")
    y = [y] if isinstance(y, str) else list(y)
    if any(m not in PANDAS for m in y):
        nonpd.append((n, sorted(set(y) - PANDAS)[:4]))
    if any(m in ("pd-multiindex", "pd_multiindex_hier") for m in y):
        panel += 1
print("non-pandas y_inner:", len(nonpd), [n for n, _ in nonpd])
print("panel/hier y_inner:", panel)

# leaf forecasters with any non-pandas mtype in X_inner too
nonpdX = [n for n, c in ests if any(m not in PANDAS for m in ([c.get_class_tag("X_inner_mtype")] if isinstance(c.get_class_tag("X_inner_mtype"), str) else list(c.get_class_tag("X_inner_mtype") or [])))]
print("non-pandas X_inner:", len(nonpdX))

# regex over full class chain source (incl. module-level helpers in the defining module NOT included)
pat_abs = re.compile(r"to_absolute|get_expected_pred_idx")
pat_cut = re.compile(r"\bself\._?cutoff\b|\bcutoff\b")
n_abs = n_cut = n_any = 0
none = []
for n, c in ests:
    src = ""
    for k in c.__mro__:
        if k is BaseForecaster or not k.__module__.startswith("sktime"):
            continue
        if k.__module__.startswith("sktime.base") or k.__module__.startswith("skbase"):
            continue
        try:
            src += inspect.getsource(k)
        except Exception:
            pass
    a = bool(pat_abs.search(src)); cu = bool(pat_cut.search(src))
    n_abs += a; n_cut += cu; n_any += (a or cu)
    if not (a or cu):
        none.append(n)
print(f"regex chain-source: to_absolute|get_expected_pred_idx {n_abs}, cutoff {n_cut}, either {n_any}, neither {len(none)}")
print("neither:", none)

random.seed(20261018)
sample = sorted(random.sample(sorted(names), 25))
print("SAMPLE25:", sample)
