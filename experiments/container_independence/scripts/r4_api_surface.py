"""r4: size of the pandas API surface used by VectorizedDF / BaseForecaster.

Sizing input for the "narrow adapter protocol" vs "full API emulation" design
patterns: how many distinct pandas names does the vectorization layer touch?

Heuristic static count (AST): `pd.<name>` references, plus attribute
accesses/method calls whose name is a public pandas DataFrame/Series/Index/
MultiIndex/GroupBy member that is NOT also a generic Python container method.
False positives/negatives are possible -- treat as an order-of-magnitude estimate.

Run (from WT root):
  PYTHONPATH=.:experiments/container_independence $EXP_PY \
      experiments/container_independence/scripts/r4_api_surface.py
"""

import ast
import json
import os
import sys
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

FILES = [
    "sktime/datatypes/_vectorize.py",
    "sktime/forecasting/base/_base.py",
    "sktime/forecasting/base/_fh.py",
    "sktime/forecasting/naive/_naive.py",
]

GENERIC = set(dir(dict)) | set(dir(list)) | set(dir(str)) | set(dir(set)) | {
    "copy", "get", "items", "keys", "values", "update", "pop", "append", "index",
    "count", "sort", "format", "join", "replace", "split", "strip",
}
PANDAS_MEMBERS = set()
for cls in [
    pd.DataFrame, pd.Series, pd.Index, pd.MultiIndex, pd.PeriodIndex,
    pd.DatetimeIndex, pd.core.groupby.DataFrameGroupBy,
]:
    PANDAS_MEMBERS |= {m for m in dir(cls) if not m.startswith("_")}
PANDAS_SPECIFIC = PANDAS_MEMBERS - GENERIC


def scan(path):
    src = open(path).read()
    tree = ast.parse(src)
    pd_refs = Counter()
    meth = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "pd":
                pd_refs[node.attr] += 1
            elif node.attr in PANDAS_SPECIFIC:
                meth[node.attr] += 1
    return pd_refs, meth, len(src.splitlines())


def main():
    out = {"pandas": pd.__version__, "files": {}}
    tot_pd, tot_m = Counter(), Counter()
    for f in FILES:
        if not os.path.exists(f):
            continue
        pd_refs, meth, n = scan(f)
        tot_pd.update(pd_refs)
        tot_m.update(meth)
        out["files"][f] = {
            "lines": n,
            "pd.<name> distinct": len(pd_refs),
            "pd.<name> sites": sum(pd_refs.values()),
            "pandas-specific attr distinct": len(meth),
            "pandas-specific attr sites": sum(meth.values()),
            "pd_refs": dict(pd_refs.most_common()),
            "attrs": dict(meth.most_common()),
        }
        print(f"{f}: {n} lines | pd.* {len(pd_refs)} distinct / {sum(pd_refs.values())} sites"
              f" | pandas-specific attrs {len(meth)} distinct / {sum(meth.values())} sites")
        print("   pd.*  :", dict(pd_refs.most_common(15)))
        print("   attrs :", dict(meth.most_common(25)))
    out["total_pd_distinct"] = len(tot_pd)
    out["total_attr_distinct"] = len(tot_m)
    print("TOTAL distinct pd.* =", len(tot_pd), " distinct pandas-specific attrs =", len(tot_m))
    path = os.path.join(RES, "r4_api_surface.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
