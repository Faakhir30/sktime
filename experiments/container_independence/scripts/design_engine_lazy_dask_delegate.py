"""design_engine_lazy: partition-and-delegate vs stock per-instance parallelism (critic gap G7).

Stock sktime fit+predict of an UNCHANGED per-instance forecaster on a Panel, compared with:
  - stock + backend:parallel="loky" (n_jobs=4)         -> one joblib task per instance
  - stock + backend:parallel="dask" (processes)       -> one dask task per instance
  - partition-and-delegate: dask.dataframe partitioned by key (set_index sorted -> divisions),
    map_partitions(fn) where fn runs the stock sktime forecaster on the whole partition
    (pandas MI) and returns long predictions; processes scheduler, 4 workers, P partitions.
Equality of predictions vs stock serial is asserted.

Usage: python design_engine_lazy_dask_delegate.py N T est   (est in naive, poly)
"""

import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, "experiments/container_independence")
from common import make_panel_fast  # noqa: E402

FH = [1, 2, 3]


def make_est(name):
    if name == "naive":
        from sktime.forecasting.naive import NaiveForecaster

        e = NaiveForecaster(strategy="last")
    else:
        from sktime.forecasting.trend import PolynomialTrendForecaster

        e = PolynomialTrendForecaster(degree=1)
    e.set_config(remember_data=True) if "remember_data" in e.get_config() else None
    return e


def part_fn(pdf, est_name):
    """Runs on a worker: one stock sktime call per PARTITION (vectorizes inside)."""
    import warnings

    warnings.filterwarnings("ignore")
    if len(pdf) == 0:
        return pd.DataFrame({"h0": pd.Series([], dtype=object), "time": pd.Series([], dtype="datetime64[ns]"),
                             "c0": pd.Series([], dtype=float)})
    y = pdf.reset_index().set_index(["h0", "time"]).sort_index()
    est = make_est(est_name)
    est.fit(y, fh=FH)
    return est.predict().reset_index()


def main(N, T, est_name):
    import dask
    import dask.dataframe as dd

    y = make_panel_fast(N, T)
    y.index.names = ["h0", "time"]
    res = {"N": N, "T": T, "est": est_name}

    def serial():
        e = make_est(est_name)
        e.fit(y, fh=FH)
        return e.predict()

    def stock_backend(backend, params):
        e = make_est(est_name)
        e.set_config(**{"backend:parallel": backend, "backend:parallel:params": params})
        e.fit(y, fh=FH)
        return e.predict()

    flat = y.reset_index()

    def delegate(P):
        # partition-aligned by key (r4 verifier V5: set_index(sorted) + map_partitions keeps order)
        ddf = dd.from_pandas(flat.set_index("h0"), npartitions=P, sort=True)
        meta = pd.DataFrame({"h0": pd.Series([], dtype=object), "time": pd.Series([], dtype="datetime64[ns]"),
                             "c0": pd.Series([], dtype=float)})
        out = ddf.map_partitions(part_fn, est_name, meta=meta)
        with dask.config.set(scheduler="processes", num_workers=4):
            pdf = out.compute()
        return pdf.set_index(["h0", "time"]).sort_index()

    t0 = time.perf_counter()
    ref = serial()
    res["serial_s"] = time.perf_counter() - t0
    ref_s = ref.sort_index()

    cases = [("loky4", lambda: stock_backend("loky", {"n_jobs": 4})),
             ("dask_per_instance", lambda: stock_backend("dask", {"scheduler": "processes", "num_workers": 4}))]
    for P in (4, 16, 64):
        cases.append((f"delegate_P{P}", (lambda P=P: delegate(P))))
    for name, fn in cases:
        try:
            t0 = time.perf_counter()
            out = fn()
            dt = time.perf_counter() - t0
            out_s = out.sort_index()
            key_dtype = str(out_s.index.get_level_values(0).dtype)
            # dask converts object keys to string[pyarrow] (fidelity loss, recorded); compare as object
            out_c = out_s.copy()
            out_c.index = out_c.index.set_levels(out_c.index.levels[0].astype(object), level=0)
            eq = bool(np.allclose(out_c.values, ref_s.values, rtol=1e-9)) and out_c.index.equals(ref_s.index)
            res[name] = {"s": dt, "speedup_vs_serial": res["serial_s"] / dt, "equal_values_index": eq,
                         "key_dtype_out": key_dtype}
        except Exception as ex:
            res[name] = {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}
        print(name, res[name], flush=True)
    print(json.dumps(res))
    with open(f"experiments/container_independence/results/design_engine_lazy_dask_{est_name}_{N}x{T}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
