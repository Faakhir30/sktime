"""x1 verify0: re-run the x1 op matrix (part C) through narwhals.stable.v1 / v2.

x1 used top-level ``narwhals``. sktime would pin to a stable API. This checks
whether the "7 ops fail for every backend by design" and the per-backend
coverage numbers hold under stable.v1 / stable.v2, and what nw class each
native maps to.

Run::
  PYTHONPATH=.:experiments/container_independence:experiments/container_independence/scripts \
     $EXP_PY -W ignore experiments/container_independence/scripts/x1_nw_lazy_collect_verify0_stable.py
"""

import json
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

import narwhals.stable.v1 as nw1  # noqa: E402
import narwhals.stable.v2 as nw2  # noqa: E402

import x1_matrix as M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAG = f"pd{pd.__version__.split('.')[0]}"

pdf = M.make_long(20, 30)
nat = M.Natives(pdf)
out = {}
for api_name, api in [("stable.v1", nw1), ("stable.v2", nw2)]:
    M.nw = api  # op_specs reads the module global at call time
    specs = M.op_specs(pdf["h0"].iloc[0])
    res = {}
    for name, build in nat.builders().items():
        native = build()
        try:
            obj = api.from_native(native)
        except Exception as e:  # noqa: BLE001
            res[name] = {"from_native": M.err_str(e)}
            continue
        is_df = isinstance(obj, api.DataFrame)
        lf = obj.lazy() if is_df and hasattr(obj, "lazy") else obj
        row = {"_class": type(obj).__name__, "_lazy_class": type(lf).__name__}
        ok, fails = 0, []
        for op, fn in specs.items():
            try:
                r = fn(lf)
                if isinstance(r, (api.LazyFrame, api.DataFrame)):
                    nr = r.to_native()
                    same = isinstance(nr, type(native))
                    if isinstance(r, api.LazyFrame):
                        r.collect()
                    row[op] = "ok" + ("" if same else f" TYPE->{M.short_type(nr)}")
                else:
                    row[op] = "ok(meta)"
                ok += 1
            except Exception as e:  # noqa: BLE001
                row[op] = "ERR " + M.err_str(e)[:90]
                fails.append(op)
        row["_ok"] = ok
        row["_n"] = len(specs)
        row["_fails"] = fails
        res[name] = row
    out[api_name] = res

os.makedirs(RES, exist_ok=True)
path = os.path.join(RES, f"x1_nw_lazy_collect_verify0_stable_{TAG}.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1, default=str)
for api_name, res in out.items():
    print(f"\n## {api_name}")
    for name, row in res.items():
        if "_ok" not in row:
            print(f"  {name}: {row}")
            continue
        print(f"  {name}: class={row['_class']}/{row['_lazy_class']} ok={row['_ok']}/{row['_n']}"
              f" fails={row['_fails']}")
print("\nwrote", path)
