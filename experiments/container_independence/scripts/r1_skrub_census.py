"""r1: census of skrub's dataframe dispatch layer (skrub._dataframe + skrub._dispatch).

Counts generic (dispatched) functions in skrub._dataframe._common and, for each
backend type, how many have a registered specialization. Also scans the skrub
package for code outside skrub/_dataframe that still talks to pandas/polars
directly ("leakage" of backend API past the dispatch layer).

Run (needs skrub; use the private r1 venv, NOT the shared envs):
  PYTHONPATH=.:experiments/container_independence $R1_PY -W ignore \
      experiments/container_independence/scripts/r1_skrub_census.py [--grid dev]
"""

import argparse
import json
import os
import re
import warnings

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="dev", choices=["dev", "large"])
args = ap.parse_args()

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import skrub  # noqa: E402
from skrub._dataframe import _common as C  # noqa: E402

types = {
    "pd.DataFrame": pd.DataFrame,
    "pd.Series": pd.Series,
    "pl.DataFrame": pl.DataFrame,
    "pl.LazyFrame": pl.LazyFrame,
    "pl.Series": pl.Series,
}
generic = {}
for name in dir(C):
    f = getattr(C, name)
    if callable(f) and hasattr(f, "registry") and hasattr(f, "specialize"):
        reg = f.registry
        generic[name] = {tn: (t in reg) for tn, t in types.items()}

n_generic = len(generic)
cover = {tn: sum(v[tn] for v in generic.values()) for tn in types}
lazy_supported = sorted(k for k, v in generic.items() if v["pl.LazyFrame"])
public = [n for n in C.__all__]
n_public_generic = sum(1 for n in public if n in generic)

# leakage scan: modules outside skrub/_dataframe that use pandas/polars directly.
# AST-based so docstring doctests (">>> pd.DataFrame") are NOT counted.
import ast  # noqa: E402

root = os.path.dirname(skrub.__file__)
leak = {}
own_dispatch = {}
n_mod = 0
n_uses_ns = 0
for dp, _, fns in os.walk(root):
    if "/tests" in dp or "_dataframe" in dp or "_docs" in dp:
        continue
    for fn in fns:
        if not fn.endswith(".py") or fn == "conftest.py":
            continue
        n_mod += 1
        src = open(os.path.join(dp, fn), encoding="utf8").read()
        rel = os.path.relpath(os.path.join(dp, fn), root)
        if re.search(r"_dataframe import|from \.+_dataframe|sbd\.", src):
            n_uses_ns += 1
        tree = ast.parse(src)
        pd_calls = pl_calls = 0
        n_disp = n_spec = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "pd":
                    pd_calls += 1
                elif node.value.id == "pl":
                    pl_calls += 1
                if node.attr == "specialize":
                    n_spec += 1
            if isinstance(node, (ast.FunctionDef,)):
                for d in node.decorator_list:
                    if isinstance(d, ast.Name) and d.id == "dispatch":
                        n_disp += 1
        if pd_calls or pl_calls:
            leak[rel] = {"pd.*": pd_calls, "pl.*": pl_calls}
        if n_disp:
            own_dispatch[rel] = {"generic": n_disp, "specializations": n_spec}

out = {
    "skrub": skrub.__version__,
    "pandas": pd.__version__,
    "polars": pl.__version__,
    "n_generic_functions": n_generic,
    "n_public_api_names": len(public),
    "n_public_generic": n_public_generic,
    "specializations_per_type": cover,
    "lazyframe_supported_functions": lazy_supported,
    "n_modules_scanned_outside_dataframe": n_mod,
    "n_modules_using_dispatch_namespace": n_uses_ns,
    "n_modules_with_direct_pd_or_pl_calls": len(leak),
    "direct_calls_total": {
        "pd.*": sum(v["pd.*"] for v in leak.values()),
        "pl.*": sum(v["pl.*"] for v in leak.values()),
    },
    "top_leaky_modules": sorted(leak.items(), key=lambda kv: -sum(kv[1].values()))[:10],
    "n_modules_with_own_dispatch": len(own_dispatch),
    "own_dispatch_generic_total": sum(v["generic"] for v in own_dispatch.values()),
    "own_dispatch_specializations_total": sum(
        v["specializations"] for v in own_dispatch.values()
    ),
    "own_dispatch_modules": own_dispatch,
}
print(json.dumps(out, indent=1))
os.makedirs("experiments/container_independence/results", exist_ok=True)
with open("experiments/container_independence/results/r1_skrub_census.json", "w") as f:
    json.dump({**out, "generic_table": generic}, f, indent=1)
