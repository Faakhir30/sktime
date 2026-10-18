"""x2: check that the narwhals "ctor" recipe yields slices identical to VectorizedDF.items.

Compares values, index name, index dtype and index freq per instance.

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore experiments/container_independence/scripts/x2_verify_recipe.py
"""
import narwhals as nw, numpy as np, pandas as pd
import importlib.util, sys
spec = importlib.util.spec_from_file_location("b", "experiments/container_independence/scripts/x2_nw_groupby_iter_bench.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
from sktime.datatypes._vectorize import prepare_VectorizedDF
df = b.make_permuted(50, 30, 2)
vec, X_mi = prepare_VectorizedDF(df, iterate_as="Series", is_scitype="Panel")
ref = {k: g for k, _, g in vec.items(X=X_mi)}
schema_freq = vec.freq; inst=["h0"]; tname="time"; data_cols=list(df.columns)
nfl = nw.from_native(df.reset_index(), eager_only=True)
ok = True
for k, g in nfl.group_by(inst):
    gn = nw.to_native(g); tv = gn[tname].to_numpy(); ix = pd.DatetimeIndex(tv, name=tname)
    cand = pd.date_range(tv[0], periods=len(tv), freq=schema_freq, name=tname, unit=ix.unit)
    if np.array_equal(cand.asi8, ix.asi8): ix = cand
    s = pd.DataFrame(gn[data_cols].to_numpy(), index=ix, columns=data_cols)
    r = ref[k[0]]
    ok &= s.equals(r) and s.index.freq == r.index.freq and s.index.name == r.index.name and s.index.dtype == r.index.dtype
print(pd.__version__, "ctor recipe == sk_items slices (values, freq, name, dtype):", ok)
