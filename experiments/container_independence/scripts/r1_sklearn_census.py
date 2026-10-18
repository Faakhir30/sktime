"""r1: census of scikit-learn's container-independence machinery (static + runtime).

Measures, on the installed scikit-learn:
  * size of the set_output ContainerAdapterProtocol and the registered adapters
  * how many estimators declare ``array_api_support=True`` (runtime tag census)
  * how many estimators expose ``set_output`` (output-container dispatch)
  * narwhals footprint: files importing narwhals, ``nw.`` call sites, distinct names
  * Array API footprint: files / call sites using ``get_namespace``

Run:
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/r1_sklearn_census.py [--grid dev]
(--grid is accepted for uniformity; this script is size independent.)
"""

import argparse
import ast
import inspect
import json
import os
import re
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="dev", choices=["dev", "large"])
args = ap.parse_args()

import sklearn  # noqa: E402
from sklearn.utils import all_estimators  # noqa: E402
from sklearn.utils._set_output import (  # noqa: E402
    ADAPTERS_MANAGER,
    ContainerAdapterProtocol,
)

out = {"sklearn": sklearn.__version__}

# ---------------------------------------------------------------- set_output
proto_methods = [
    n
    for n, v in vars(ContainerAdapterProtocol).items()
    if callable(v) and not n.startswith("_")
]
proto_attrs = list(getattr(ContainerAdapterProtocol, "__annotations__", {}).keys())
out["set_output_protocol"] = {
    "methods": proto_methods,
    "attributes": proto_attrs,
    "registered_adapters": sorted(ADAPTERS_MANAGER.adapters),
    "supported_outputs": sorted(ADAPTERS_MANAGER.supported_outputs),
}
for name, ad in ADAPTERS_MANAGER.adapters.items():
    src = inspect.getsource(type(ad))
    out["set_output_protocol"][f"loc_{name}_adapter"] = len(
        [ln for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    )

# ---------------------------------------------------------------- tag census
try:
    from sklearn.utils._test_common.instance_generator import _construct_instances
except Exception:  # noqa: BLE001
    _construct_instances = None

from sklearn.utils import get_tags  # noqa: E402

n_total = 0
n_inst = 0
array_api_true = []
set_output_avail = []
failed = []
by_type = Counter()
for name, Est in all_estimators():
    n_total += 1
    inst = None
    try:
        if _construct_instances is not None:
            inst = next(iter(_construct_instances(Est)))
        else:
            inst = Est()
    except Exception as e:  # noqa: BLE001
        failed.append((name, type(e).__name__))
        continue
    n_inst += 1
    try:
        tags = get_tags(inst)
    except Exception as e:  # noqa: BLE001
        failed.append((name, "tags:" + type(e).__name__))
        continue
    if tags.array_api_support:
        array_api_true.append(name)
        by_type[tags.estimator_type or "other"] += 1
    if hasattr(inst, "set_output"):
        set_output_avail.append(name)

out["tag_census"] = {
    "n_estimators_all_estimators": n_total,
    "n_instantiated": n_inst,
    "n_array_api_support_true": len(array_api_true),
    "array_api_support_true": sorted(array_api_true),
    "array_api_by_estimator_type": dict(by_type),
    "n_set_output_available": len(set_output_avail),
    "failed_to_instantiate": failed,
}

# ---------------------------------------------------------------- static scans
root = os.path.dirname(sklearn.__file__)
nw_files, nw_calls, nw_names = [], 0, Counter()
xp_files, xp_calls = [], 0
tag_true_static = 0
for dp, _, fns in os.walk(root):
    if "/tests" in dp or "/externals" in dp:
        continue
    for fn in fns:
        if not fn.endswith(".py"):
            continue
        p = os.path.join(dp, fn)
        rel = os.path.relpath(p, root)
        src = open(p, encoding="utf8").read()
        if re.search(r"^import narwhals|^from narwhals", src, re.M):
            nw_files.append(rel)
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    # collect dotted chains rooted at "nw"
                    chain = []
                    cur = node
                    while isinstance(cur, ast.Attribute):
                        chain.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name) and cur.id == "nw":
                        chain = ".".join(reversed(chain))
                        nw_names[chain] += 1
        n_get_ns = len(re.findall(r"get_namespace(?:_and_device)?\(", src))
        if n_get_ns and rel != "utils/_array_api.py":
            xp_files.append(rel)
            xp_calls += n_get_ns
        tag_true_static += len(re.findall(r"array_api_support\s*=\s*True", src))

# drop sub-chains (nw.dependencies counted inside nw.dependencies.is_into_dataframe)
full_chains = {
    k: v
    for k, v in nw_names.items()
    if not any(o != k and o.startswith(k + ".") for o in nw_names)
}
out["narwhals_footprint"] = {
    "files_importing_narwhals": sorted(nw_files),
    "n_files": len(nw_files),
    "nw_attribute_chains": dict(sorted(full_chains.items(), key=lambda kv: -kv[1])),
    "n_nw_call_sites": sum(full_chains.values()),
}
out["array_api_footprint"] = {
    "n_files_calling_get_namespace": len(xp_files),
    "n_get_namespace_call_sites": xp_calls,
    "n_static_array_api_support_true_literals": tag_true_static,
}

print(json.dumps({k: v for k, v in out.items() if k != "tag_census"}, indent=1))
tc = out["tag_census"]
print(
    "tag_census:",
    {k: v for k, v in tc.items() if k not in ("array_api_support_true", "failed_to_instantiate")},
)
print("array_api_support_true:", tc["array_api_support_true"])
print("failed_to_instantiate:", tc["failed_to_instantiate"])
os.makedirs("experiments/container_independence/results", exist_ok=True)
with open("experiments/container_independence/results/r1_sklearn_census.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
