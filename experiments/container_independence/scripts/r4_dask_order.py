import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "experiments/container_independence")
import dask, dask.dataframe as dd, numpy as np
from common import make_panel_fast
for n in [10, 100]:
    flat = make_panel_fast(n, 100).reset_index()[["h0","time","c0"]]
    d = dd.from_pandas(flat, npartitions=4)
    with dask.config.set(scheduler="sync"):
        r = d.groupby("h0")["time"].apply(lambda g: bool(g.is_monotonic_increasing), meta=("time", bool)).compute()
    print(n, "groups with non-monotonic time inside dask groupby.apply:", int((~r).sum()), "of", len(r))
