"""G1: which registered forecasters would the trusted path serve / fall back on.

For every forecaster in all_estimators("forecaster"): does the class override a
GUARDED public method/helper (-> fallback), which inner mtypes (vectorization
possible at all?), and the tags the trusted path must honour.
"""
import json
import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
import g1_trusted as g1  # noqa: E402
from sktime.registry import all_estimators  # noqa: E402

PANEL_MT = {"pd-multiindex", "pd_multiindex_hier", "numpy3D", "df-list", "nested_univ"}
rows = []
for name, cls in all_estimators("forecaster", return_names=True):
    ok, reason = g1.class_eligible(cls)
    yi = cls.get_class_tag("y_inner_mtype")
    yi = [yi] if isinstance(yi, str) else list(yi)
    rows.append(dict(
        name=name, eligible=ok, reason=reason,
        panel_native=bool(PANEL_MT & set(yi)),
        series_inner=sorted(set(yi) - PANEL_MT),
        requires_fh=cls.get_class_tag("requires-fh-in-fit"),
        insample=cls.get_class_tag("capability:insample"),
        exog=cls.get_class_tag("capability:exogenous"),
        missing=cls.get_class_tag("capability:missing_values"),
        pretrain=cls.get_class_tag("capability:pretrain", False),
        overrides_update_private=cls._update is not g1.BaseForecaster._update,
    ))
c = Counter((r["eligible"], r["reason"]) for r in rows)
print("n forecasters", len(rows))
for k, v in sorted(c.items(), key=lambda x: -x[1]):
    print(v, k)
print("not panel-native (vectorize via VectorizedDF):",
      sum(not r["panel_native"] for r in rows))
print("  of which eligible:", sum((not r["panel_native"]) and r["eligible"] for r in rows))
print("requires-fh-in-fit True:", sum(bool(r["requires_fh"]) for r in rows))
print("insample False:", sum(r["insample"] is False for r in rows))
print("pretrain:", sum(bool(r["pretrain"]) for r in rows))
print("series inner mtypes:", Counter(tuple(r["series_inner"]) for r in rows).most_common(8))
for r in rows:
    if not r["eligible"]:
        print("  fallback:", r["name"], "-", r["reason"], "panel_native" if r["panel_native"] else "")
out = os.path.join(os.path.dirname(__file__), "..", "results", "g1_census.json")
json.dump(rows, open(out, "w"), indent=1, default=str)
print("wrote", out)
