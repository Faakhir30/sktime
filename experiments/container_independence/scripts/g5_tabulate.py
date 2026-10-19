"""Tabulate results/g5_api_surface.json into the markdown matrix + coverage counts."""
import collections
import json

r = json.load(open("experiments/container_independence/results/g5_api_surface.json"))


def cell(x):
    st = x["status"]
    if st == "FAIL":
        return "FAIL"
    if st.startswith("FAIL(silent)"):
        return "SILENT"
    if "raised" in x:
        return "raises"
    e = x.get("eq_vs_stock_pd")
    if not isinstance(e, dict) or not e:
        return "PASS(noref)"
    lv = [v[0] for v in e.values()]
    if all(v == "strict" for v in lv):
        tag = "="
    elif all(v in ("strict", "values-only") for v in lv):
        tag = "~"
    else:
        tag = "≠"
    bk = "" if x.get("out_backend_matches_in", True) else ",pdOut"
    return "PASS" + tag + bk


rows = collections.OrderedDict()
for x in r:
    rows.setdefault((x["op"], x["scitype"]), {})[(x["approach"], x["backend"])] = cell(x)
apprs = ["stock", "x4", "x4nc", "x4fix", "x6b"]
cols = [(a, b) for a in apprs for b in ["pandas", "polars"]]
print("| op | scitype | " + " | ".join(f"{a} {b[:2]}" for a, b in cols) + " |")
print("|" + "---|" * (len(cols) + 2))
for k, v in rows.items():
    print(f"| {k[0]} | {k[1]} | " + " | ".join(v.get(c, "") for c in cols) + " |")
print()
print("| approach | backend | runs w/o error (incl. correct raise) | strict-equal to stock-pandas & backend-preserving |")
print("|---|---|---|---|")
for a in apprs:
    for b in ["pandas", "polars"]:
        cs = [cell(x) for x in r if x["approach"] == a and x["backend"] == b]
        ok = sum(c.startswith("PASS") or c == "raises" for c in cs)
        strict = sum(c in ("PASS=", "raises") for c in cs)
        print(f"| {a} | {b} | {ok}/{len(cs)} | {strict}/{len(cs)} |")
