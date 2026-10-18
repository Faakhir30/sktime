"""x2 side finding: does freq survive pandas groupby slicing, and does sktime's
``_enforce_index_freq`` mutate the caller's MultiIndex under pandas 3?

1. For make_panel (MultiIndex.from_tuples) and make_panel_fast (from_product with
   date_range): freq of the time level, of ``xs`` slices and of fresh
   ``groupby(level=0, sort=False)`` slices; whether the slice index shares its
   array with the MultiIndex level.
2. Run ``prepare_VectorizedDF`` + ``items`` on a user frame and check whether the
   user frame's time-level freq changed afterwards.

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x2_freq_mutation_probe.py
"""

import pandas as pd

from common import make_panel, make_panel_fast

print("pandas", pd.__version__)
for gen in ["make_panel", "make_panel_fast"]:
    if gen == "make_panel":
        df = make_panel(4, 8, n_cols=2, n_levels=1)
    else:
        df = make_panel_fast(4, 8, n_cols=2)
    k0 = df.index[0][0]
    print(
        f"{gen}: levels[-1].freq={df.index.levels[-1].freq} "
        f"xs-slice freq={df.xs(k0).index.freq}"
    )
    g = next(iter(df.groupby(level=0, sort=False)))[1].droplevel(0)
    print(
        f"  fresh groupby slice freq={g.index.freq}; "
        f"slice index shares DatetimeArray with MI level: "
        f"{g.index._data is df.index.levels[-1]._data}"
    )
    g.index.freq = "D"
    print(
        f"  after `slice.index.freq = 'D'`: MI levels[-1].freq={df.index.levels[-1].freq}"
        f"; xs-slice freq={df.xs(k0).index.freq}"
    )

from sktime.datatypes._vectorize import prepare_VectorizedDF  # noqa: E402

df = make_panel(4, 8, n_cols=2, n_levels=1)
before = (df.index.levels[-1].freq, df.xs("s0").index.freq)
vec, X_mi = prepare_VectorizedDF(df, iterate_as="Series", is_scitype="Panel")
_ = list(vec.items(X=X_mi))
after = (df.index.levels[-1].freq, df.xs("s0").index.freq)
print(f"prepare_VectorizedDF returned the caller's object (X_mi is df): {X_mi is df}")
print(f"caller df (time-level freq, xs freq) before items(): {before}  after: {after}")
print(f"caller frame mutated by VectorizedDF.items(): {before != after}")
