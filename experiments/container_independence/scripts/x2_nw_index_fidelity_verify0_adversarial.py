"""x2 verify0: adversarial cases for the x2 recipe and the pd3 mutation finding.

(a) recipe iter_slices / long_to_mi vs VectorizedDF.items on inputs NOT in the
    original test set: mixed data dtypes (int64 + float64 + bool), unequal-length
    series, series with gaps (irregular), and NaN in data.
(b) pd3 mutation: can VectorizedDF.items() write a WRONG freq into the caller's
    MI level? (first series regular daily, level = union of irregular times)

Usage::
    PYTHONPATH=.:experiments/container_independence $EXP_PY{,3} -W ignore \
        experiments/container_independence/scripts/x2_nw_index_fidelity_verify0_adversarial.py
"""

import json
import os
import sys

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
from x2_index_recipe import iter_slices, long_to_mi, mi_to_long  # noqa: E402

from sktime.datatypes._vectorize import prepare_VectorizedDF  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
out = {"pandas": pd.__version__}


def mk_mixed():
    t = pd.date_range("2000-01-01", periods=6, freq="D")
    idx = pd.MultiIndex.from_product([["s0", "s1", "s2"], t], names=["h0", "time"])
    n = len(idx)
    return pd.DataFrame(
        {
            "f": np.arange(n, dtype=float),
            "i": np.arange(n, dtype="int64"),
            "b": np.arange(n) % 2 == 0,
        },
        index=idx,
    )


def mk_unequal_gaps():
    rows = []
    d = pd.Timestamp("2000-01-01")
    for k, ts in {
        "s0": [d + pd.Timedelta(days=i) for i in range(8)],  # regular D, len 8
        "s1": [d + pd.Timedelta(days=i) for i in (0, 2, 4, 6, 8, 10)],  # every 2D
        "s2": [d + pd.Timedelta(days=i) for i in (3, 4, 5)],  # short, regular
    }.items():
        rows += [(k, x) for x in ts]
    idx = pd.MultiIndex.from_tuples(rows, names=["h0", "time"])
    vals = np.arange(len(idx), dtype=float)
    vals[3] = np.nan
    return pd.DataFrame({"c0": vals}, index=idx)


def compare(df, label):
    res = {}
    vec, X = prepare_VectorizedDF(df.copy(deep=True), iterate_as="Series", is_scitype="Panel")
    ref = [(t[0], t[-1]) for t in vec.items(X=X)]
    for backend in ("pandas", "polars"):
        flat, schema = mi_to_long(df, period_to_timestamp=(backend == "polars"))
        nf = nw.from_native(flat if backend == "pandas" else pl.from_pandas(flat))
        got = list(iter_slices(nf, schema))
        checks = {
            "n": len(got) == len(ref),
            "values": all(np.allclose(np.asarray(a[1], float), np.asarray(b[1], float), equal_nan=True)
                          for a, b in zip(ref, got)),
            "data_dtypes_equal": all(list(a[1].dtypes) == list(b[1].dtypes) for a, b in zip(ref, got)),
            "index_equal": all(a[1].index.equals(b[1].index) for a, b in zip(ref, got)),
            "freq_equal": [(str(a[1].index.freq), str(b[1].index.freq)) for a, b in zip(ref, got)],
        }
        back = long_to_mi(nf, schema, sort_like_schema=True)
        checks["long_to_mi_equals"] = bool(back.equals(df))
        checks["long_to_mi_dtypes"] = [str(x) for x in back.dtypes]
        checks["ref_dtypes_slice0"] = [str(x) for x in ref[0][1].dtypes]
        checks["got_dtypes_slice0"] = [str(x) for x in got[0][1].dtypes]
        res[backend] = checks
    out[label] = res
    print(label, json.dumps(res, default=str))


compare(mk_mixed(), "mixed_dtypes")
compare(mk_unequal_gaps(), "unequal_gaps_nan")

# (b) can items() write a wrong freq into the caller?
df = mk_unequal_gaps()
lvl_before = df.index.levels[-1].freq
vec, X = prepare_VectorizedDF(df, iterate_as="Series", is_scitype="Panel")
print("X is df:", X is df, "vec.freq:", vec.freq)
items = list(vec.items(X=X))
lvl_after = df.index.levels[-1].freq
s1 = df.xs("s1").index
s1_valid = None
try:
    pd.DatetimeIndex(s1.asi8.astype("M8[ns]") if False else s1, freq=s1.freq)
    s1_valid = True
except ValueError as e:
    s1_valid = f"invalid: {e}"
out["mutation_irregular"] = {
    "level_freq_before": str(lvl_before),
    "level_freq_after": str(lvl_after),
    "slice_freqs": [str(it[-1].index.freq) for it in items],
    "caller_xs_s1_freq_after": str(s1.freq),
    "caller_xs_s1_freq_consistent": s1_valid,
    "caller_level_values": [str(x.date()) for x in df.index.levels[-1]],
}
print("mutation_irregular", json.dumps(out["mutation_irregular"]))

# same with the time level equal for all series (common case): does mutation still happen?
t = pd.date_range("2000-01-01", periods=8, freq="D")
rows = [(k, x) for k in ["a", "b"] for x in t]
df2 = pd.DataFrame({"c0": np.arange(16.0)},
                   index=pd.MultiIndex.from_tuples(rows, names=["h0", "time"]))
f0 = df2.index.levels[-1].freq
vec, X = prepare_VectorizedDF(df2, iterate_as="Series", is_scitype="Panel")
list(vec.items(X=X))
out["mutation_equal_len"] = {"before": str(f0), "after": str(df2.index.levels[-1].freq)}
print("mutation_equal_len", out["mutation_equal_len"])

tag = "pd" + pd.__version__[0]
with open(os.path.join(HERE, "..", "results", f"x2_nw_index_fidelity_verify0_adversarial_{tag}.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)
