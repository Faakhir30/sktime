"""Verifier: dask groupby.apply within-group order; independent check using row ids,
several sizes/partitionings, and the 'partition-aligned' workaround (repartition so keys don't span)."""
import warnings, json, sys, os; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import dask, dask.dataframe as dd, numpy as np, pandas as pd
from common import make_panel_fast
out = []
for n, nt, npart in [(10, 100, 4), (10, 100, 7), (100, 100, 4), (1000, 50, 8)]:
    flat = make_panel_fast(n, nt).reset_index()[["h0", "time", "c0"]]
    flat["rid"] = np.arange(len(flat))
    d = dd.from_pandas(flat, npartitions=npart)
    meta = ("rid", bool)
    with dask.config.set(scheduler="sync"):
        r = d.groupby("h0")["rid"].apply(lambda g: bool(np.all(np.diff(g.to_numpy()) > 0)), meta=meta).compute()
        # groups spanning partitions in input
        span = d.map_partitions(lambda p: pd.Series(p["h0"].unique())).compute().value_counts()
        # workaround: set_index on key (sorted divisions) then map_partitions + pandas groupby
        d2 = d.set_index("h0", sorted=True)
        r2 = d2.map_partitions(lambda p: p.groupby(level=0, sort=False)["rid"].apply(lambda g: bool(np.all(np.diff(g.to_numpy()) > 0))), meta=meta).compute()
    row = {"n": n, "nt": nt, "npart": npart, "groups_spanning_partitions": int((span > 1).sum()),
           "groupby_apply_out_of_order": int((~r).sum()), "n_groups": len(r),
           "set_index_sorted+map_partitions_out_of_order": int((~r2).sum()), "n_groups2": int(r2.index.nunique())}
    print(json.dumps(row)); out.append(row)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "r4_dispatch_arch_verify0_dask.json"), "w"), indent=1)
