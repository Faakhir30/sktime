"""x9 verifier: Arrow-backed index acceptance, by level, and cost of coercing an
Arrow timestamp index to a numpy DatetimeIndex (zero-copy?).

Run:: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore <this>
"""
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, pyarrow as pa
from sktime.datatypes import check_is_scitype
from sktime.forecasting.naive import NaiveForecaster
from common import bench
print("pandas", pd.__version__)
t = pd.date_range("2000-01-01", periods=40, freq="D")
keys = [f"s{i}" for i in range(4)]
ts_ar = pa.array(np.tile(t.values, 4).astype("datetime64[ns]"))
k_ar = pa.array(np.repeat(keys, 40))
v = np.random.default_rng(0).normal(size=160).cumsum()
def mk(kind_k, kind_t):
    kk = pd.Index(pd.array(k_ar, dtype=pd.ArrowDtype(pa.string()))) if kind_k == "arrow" else pd.Index(np.repeat(keys, 40))
    tt = pd.Index(pd.array(ts_ar, dtype=pd.ArrowDtype(pa.timestamp("ns")))) if kind_t == "arrow" else pd.DatetimeIndex(np.tile(t.values, 4))
    mi = pd.MultiIndex.from_arrays([kk, tt], names=["h0", "time"])
    return pd.DataFrame({"y": v}, index=mi)
for kk in ["numpy", "arrow"]:
    for tk in ["numpy", "arrow"]:
        X = mk(kk, tk)
        ok, msg, md = check_is_scitype(X, ["Panel"], return_metadata=True)
        try:
            NaiveForecaster().fit(X, fh=[1, 2]).predict(); fr = "OK"
        except Exception as e:
            fr = type(e).__name__
        print(f"panel keys={kk:5} time={tk:5}: check={ok} mtype={md.get('mtype') if md else None} naive={fr} level dtypes={[str(l.dtype) for l in X.index.levels]}")
# series
for tk in ["numpy", "arrow"]:
    idx = pd.Index(pd.array(pa.array(t.values.astype("datetime64[ns]")), dtype=pd.ArrowDtype(pa.timestamp("ns")))) if tk == "arrow" else t
    y = pd.Series(v[:40], index=idx)
    print("series time=", tk, "check:", check_is_scitype(y, "Series"))
# coercion cost + zero-copy
n = 2_000_000
big = pd.Index(pd.array(pa.array(np.arange(n).astype("datetime64[ns]")), dtype=pd.ArrowDtype(pa.timestamp("ns"))))
di = pd.DatetimeIndex(big.array._pa_array.combine_chunks().to_numpy(zero_copy_only=True))
src = big.array._pa_array.chunk(0).buffers()[1].address
print("coerced via to_numpy(zero_copy_only=True): shares buffer:", di.asi8.__array_interface__["data"][0] == src)
di2 = pd.DatetimeIndex(big)
print("pd.DatetimeIndex(arrow_index) shares buffer:", di2.asi8.__array_interface__["data"][0] == src)
print("coercion time 2M: pd.DatetimeIndex(idx) %.2f ms ; zero-copy route %.3f ms" % (
    bench(lambda: pd.DatetimeIndex(big))*1e3,
    bench(lambda: pd.DatetimeIndex(big.array._pa_array.combine_chunks().to_numpy(zero_copy_only=True)))*1e3))
