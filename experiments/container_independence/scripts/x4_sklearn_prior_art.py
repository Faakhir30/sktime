"""x4: prior art in scikit-learn -- boundary adapter vs per-estimator internal dispatch.

* ``set_output`` container adapters (``sklearn.utils._set_output.ADAPTERS_MANAGER``):
  a 4-method protocol applied at the *boundary* (wrapping outputs) of every transformer.
* array API support: estimators *internally* rewritten against a dispatched namespace,
  opted in per estimator via the ``array_api_support`` tag.

Usage: $EXP_PY -W ignore experiments/container_independence/scripts/x4_sklearn_prior_art.py
"""
import inspect
import json
import os
import warnings

warnings.filterwarnings("ignore")

import sklearn  # noqa: E402
from sklearn.utils import all_estimators  # noqa: E402
from sklearn.utils._set_output import ADAPTERS_MANAGER, ContainerAdapterProtocol, _SetOutputMixin  # noqa: E402
from sklearn.utils._tags import get_tags  # noqa: E402

ED = os.path.join(os.path.dirname(__file__), "..")
res = {"sklearn": sklearn.__version__}
proto_methods = [m for m, _ in inspect.getmembers(ContainerAdapterProtocol, inspect.isfunction)]
res["set_output_protocol_methods"] = proto_methods
res["set_output_adapters"] = sorted(ADAPTERS_MANAGER.adapters)
tr = all_estimators(type_filter="transformer")
res["n_transformers"] = len(tr)
res["n_transformers_with_set_output"] = sum(
    1 for _, c in tr if issubclass(c, _SetOutputMixin) and hasattr(c, "set_output")
)
n, yes, skipped = 0, [], 0
for name, cls in all_estimators():
    try:
        est = cls()
        n += 1
        if get_tags(est).array_api_support:
            yes.append(name)
    except Exception:  # noqa: BLE001
        skipped += 1
res["n_estimators_default_instantiable"] = n
res["n_array_api_support_default_params"] = len(yes)
res["array_api_estimators"] = sorted(yes)
res["skipped_not_default_instantiable"] = skipped
print(json.dumps(res, indent=1))
with open(os.path.join(ED, "results", "x4_sklearn_prior_art.json"), "w") as f:
    json.dump(res, f, indent=1)
