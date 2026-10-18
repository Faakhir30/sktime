"""x4 verify0: independent re-count of forecaster y_inner_mtype families (claim x4.11).
Uses the class tag via get_class_tag (no instantiation). Run:
PYTHONPATH=. $EXP_PY -W ignore <this>
"""
import json, warnings, collections
warnings.filterwarnings("ignore")
from sktime.registry import all_estimators
ests = all_estimators("forecaster", return_names=True)
fam = collections.Counter(); pl_list = []; numpy_list = []
def famof(m):
    if m.startswith("pd") : return "pandas"
    if m.startswith("pl") or m.startswith("polars"): return "polars"
    if m.startswith("np") or m.startswith("numpy"): return "numpy"
    return m
for name, cls in ests:
    m = cls.get_class_tag("y_inner_mtype")
    ms = [m] if isinstance(m, str) else list(m)
    fams = sorted({famof(x) for x in ms})
    fam[tuple(fams)] += 1
    if "polars" in fams: pl_list.append((name, ms))
    if "numpy" in fams: numpy_list.append((name, ms))
print("n_forecasters", len(ests)); print(dict(fam))
print("polars:", pl_list); print("numpy:", [n for n, _ in numpy_list])
json.dump({"n": len(ests), "families": {"+".join(k): v for k, v in fam.items()}, "polars": pl_list, "numpy": numpy_list},
          open("experiments/container_independence/results/x4_proto_adapter_verify0_census.json", "w"), indent=1, default=str)
