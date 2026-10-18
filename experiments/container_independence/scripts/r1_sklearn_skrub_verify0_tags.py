"""verify0: independent array_api_support census via plain default constructors
(not sklearn's _construct_instances test helper)."""
import inspect, json, warnings, sys
warnings.filterwarnings("ignore")
from sklearn.utils import all_estimators
from sklearn.utils._tags import get_tags
true, fail, n = [], [], 0
for name, cls in all_estimators():
    n += 1
    try:
        est = cls()
    except Exception as e:
        fail.append((name, type(e).__name__)); continue
    try:
        if get_tags(est).array_api_support:
            true.append(name)
    except Exception as e:
        fail.append((name, "tags:" + type(e).__name__))
print(json.dumps({"n": n, "n_default_ctor_ok": n - len(fail), "n_true": len(true), "true": true,
                  "failed_ctor": fail}, indent=1))
