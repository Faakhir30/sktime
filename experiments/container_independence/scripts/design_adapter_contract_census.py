"""design_adapter: static census of public-API contract consumers (G16-lite). No sktime edits."""
import json, re, pathlib, sys
root = pathlib.Path("sktime")
pats = {
  "forecasters_.loc/iloc/values/iterrows/index/columns": r"forecasters_\s*\.\s*(loc|iloc|values|iterrows|index|columns|shape)\b",
  "forecasters_ any": r"\bforecasters_\b",
  "transformers_.loc/iloc/...": r"transformers_\s*\.\s*(loc|iloc|values|iterrows|index|columns)\b",
  "vectorize_est(": r"vectorize_est\(",
  "fitted-param key 'forecasters.loc['": r"forecasters\.loc\[",
  "proba col MultiIndex.from_product/from_tuples in forecasting": r"MultiIndex\.from_(product|tuples)",
  "cutoff.freq / cutoff[0]": r"cutoff(\.freq|\[0\])",
  "get_cutoff(": r"\bget_cutoff\(",
  "hasattr(.,'index')": r"hasattr\([^,]+,\s*['\"]index['\"]\)",
}
res = {}
for name, p in pats.items():
    rx = re.compile(p)
    files = {}; n = 0
    for f in root.rglob("*.py"):
        s = f.read_text(errors="ignore")
        k = len(rx.findall(s))
        if k:
            files[str(f)] = k; n += k
    tests = sum(v for f, v in files.items() if "/tests/" in f)
    res[name] = {"sites": n, "files": len(files), "test_sites": tests, "nontest_sites": n - tests,
                 "top": sorted(files.items(), key=lambda x: -x[1])[:6]}
# proba builders restricted to forecasting leaves
fx = re.compile(r"MultiIndex\.from_(product|tuples)")
leaf = [str(f) for f in pathlib.Path("sktime/forecasting").rglob("*.py") if "/tests/" not in str(f) and fx.search(f.read_text(errors='ignore'))]
res["forecasting non-test files building MultiIndex"] = {"files": len(leaf)}
json.dump(res, open("experiments/container_independence/results/design_adapter_contract_census.json","w"), indent=1)
for k, v in res.items():
    print(k, {kk: vv for kk, vv in v.items() if kk != "top"})
