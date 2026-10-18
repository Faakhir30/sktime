"""x3_iter_bench: microbenchmark core VectorizedDF operations across strategies.

Operations
----------
A  list instance keys (+ offsets for positional strategies)
B  iterate all instance slices, materialized as what an inner forecaster gets
C  reconstruct: concat per-instance outputs (last 2 rows of each slice, mimics
   fh=2 predict) with instance keys back into one frame

Strategies
----------
S1    pandas groupby(level=..., sort=False) + pd.concat(keys=)  (current sktime;
      "S1_sktime" = actual VectorizedDF.items/reconstruct incl. per-slice freq
      inference; "S1_core" = groupby + droplevel only)
S2    pandas precomputed positions (MultiIndex codes -> runs -> indptr)
      S2_iloc: X.iloc[s:e].droplevel(...)   S2_build: DataFrame(vals[s:e], t[s:e])
S3    polars partition_by(keys, maintain_order=True, as_dict=True)
S4    polars group_by(keys, maintain_order=True) iteration
S5    polars rle offsets + df.slice(offset, len) (zero-copy)
S5p   S5 + per-slice .to_pandas().set_index(time) (pd-inner forecaster on polars)
S6    numpy GroupedArray (values + indptr, slice = view)
S7a   narwhals group_by iteration over pandas reset_index frame
S7a_pd  S7a + to_native().set_index(time) (what a pd-inner forecaster needs)
S7b   narwhals group_by iteration over polars frame
S8    pyarrow Table offsets + Table.slice (zero-copy)

Run from the worktree root::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x3_iter_bench.py --grid dev

Flags: --grid {dev,large}, --hier (3-level hierarchical index), --n-cols,
--rounds, --skip-mem, --merge (combine all results/x3_iter_bench__*.json).
"""

import argparse
import gc
import glob
import json
import os
import platform
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import common  # noqa: E402
from common import (  # noqa: E402
    DEV_GRID,
    LARGE_GRID,
    bench,
    env_info,
    make_panel,
    make_panel_fast,
    peak_rss_delta_mb,
    rss_mb,
    to_polars_long,
)

KEY = "x3_iter_bench"
ED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ED, "results")


# ---------------------------------------------------------------------------
# data + representations
# ---------------------------------------------------------------------------


def make_data(n_series, n_time, n_cols, hier):
    if hier:
        return make_panel(n_series, n_time, n_cols=n_cols, n_levels=2)
    return make_panel_fast(n_series, n_time, n_cols=n_cols)


class Reps:
    """All representations of one dataset, built once."""

    def __init__(self, X, hier):
        import narwhals as nw
        import polars as pl

        from protos.x3_strategies import pa_offsets, pd_group_positions, pl_offsets
        from sktime.datatypes._vectorize import VectorizedDF

        self.X = X
        self.hier = hier
        self.scitype = "Hierarchical" if hier else "Panel"
        self.k = X.index.nlevels - 1
        self.keys = list(X.index.names[: self.k])
        self.lv = list(range(self.k))
        self.lv_gb = self.lv[0] if self.k == 1 else self.lv
        self.cols = list(X.columns)
        self.index_names = list(X.index.names)
        self.vec = VectorizedDF(
            X,
            iterate_as="Series",
            is_scitype=self.scitype,
            X_orig_mtype="pd-multiindex" if not hier else "pd_multiindex_hier",
            freq="D",
        )
        self.row_ix = self.vec.get_iter_indices()[0]
        # same data, but the time level carries no freq (e.g. MI rebuilt from
        # arrays, ragged/irregular series): VectorizedDF.items then runs
        # pd.infer_freq on every slice in _enforce_index_freq
        nl = X.index.nlevels
        self.X_nf = X.copy()
        self.X_nf.index = pd.MultiIndex.from_arrays(
            [X.index.get_level_values(i) for i in range(nl)], names=X.index.names
        )
        self.vec_nf = VectorizedDF(
            self.X_nf,
            iterate_as="Series",
            is_scitype=self.scitype,
            X_orig_mtype="pd-multiindex" if not hier else "pd_multiindex_hier",
        )
        self.pl = to_polars_long(X)
        self.pl_body = self.pl.drop(self.keys)
        self.pd_flat = X.reset_index()
        self.nw_pd = nw.from_native(self.pd_flat, eager_only=True)
        self.nw_pl = nw.from_native(self.pl, eager_only=True)
        self.pa = self.pl.to_arrow()
        self.pa_body = self.pa.drop_columns(self.keys)
        self.pd_keys, self.pd_indptr, order = pd_group_positions(X, self.k)
        assert order is None
        self.pl_keys, self.pl_off, self.pl_len, c1 = pl_offsets(self.pl, self.keys)
        self.pa_keys, self.pa_off, self.pa_len, c2 = pa_offsets(self.pa, self.keys)
        assert c1 and c2
        self.pl_keys_rows = list(self.pl_keys.iter_rows())
        _ = pl  # noqa


def sizes_mb(R):
    from protos.x3_strategies import GroupedArray

    ga = GroupedArray.from_pandas_mi(R.X, R.k)
    return {
        "pandas_mi_deep": R.X.memory_usage(deep=True, index=True).sum() / 2**20,
        "pandas_flat_deep": R.pd_flat.memory_usage(deep=True, index=True).sum()
        / 2**20,
        "polars_estimated": R.pl.estimated_size() / 2**20,
        "arrow_nbytes": R.pa.nbytes / 2**20,
        "grouped_array_nbytes_excl_keys": ga.nbytes() / 2**20,
    }


# ---------------------------------------------------------------------------
# (A) keys
# ---------------------------------------------------------------------------


def ops_A(R):
    import polars as pl  # noqa: F401

    from protos.x3_strategies import pa_offsets, pd_group_positions, pl_offsets

    X, k, keys = R.X, R.k, R.keys
    pl_df, pa_tab = R.pl, R.pa

    def s1():
        return X.index.droplevel(list(range(k, X.index.nlevels))).unique()

    def s2():
        return pd_group_positions(X, k)

    def s3s4():
        return pl_df.select(keys).unique(maintain_order=True)

    def s5():
        return pl_offsets(pl_df, keys)

    def s7a():
        return R.nw_pd.select(keys).unique(maintain_order=True)

    def s7b():
        return R.nw_pl.select(keys).unique(maintain_order=True)

    def s8():
        return pa_offsets(pa_tab, keys)

    return {
        "S1": s1,
        "S2": s2,
        "S3/S4": s3s4,
        "S5": s5,
        "S6": lambda: (R.pd_keys, R.pd_indptr),  # stored at conversion; O(1)
        "S7a": s7a,
        "S7b": s7b,
        "S8": s8,
    }


# ---------------------------------------------------------------------------
# (B) iterate
# ---------------------------------------------------------------------------


def ops_B(R, ga, pdpos):
    keys = R.keys
    X, lv, lv_gb = R.X, R.lv, R.lv_gb

    def s1_sktime():
        n = 0
        for _, _, inst in R.vec.items(X):
            n += inst.shape[0]
        return n

    def s1_sktime_nofreq():
        n = 0
        for _, _, inst in R.vec_nf.items(R.X_nf):
            n += inst.shape[0]
        return n

    def s1_core():
        n = 0
        for _, g in X.groupby(level=lv_gb, sort=False):
            n += g.droplevel(lv).shape[0]
        return n

    def s2_iloc():
        n = 0
        for inst in pdpos.iter_iloc():
            n += inst.shape[0]
        return n

    def s2_build():
        n = 0
        for inst in pdpos.iter_build():
            n += inst.shape[0]
        return n

    def s3():
        parts = R.pl.partition_by(
            keys, maintain_order=True, as_dict=True, include_key=False
        )
        n = 0
        for inst in parts.values():
            n += inst.height
        return n

    def s4():
        n = 0
        for _, g in R.pl.group_by(keys, maintain_order=True):
            n += g.drop(keys).height
        return n

    def s5():
        body = R.pl_body
        n = 0
        for o, ln in zip(R.pl_off, R.pl_len):
            n += body.slice(o, ln).height
        return n

    def s5p():
        body = R.pl_body
        n = 0
        for o, ln in zip(R.pl_off, R.pl_len):
            n += body.slice(o, ln).to_pandas().set_index("time").shape[0]
        return n

    def s5p_fast():
        # polars slice -> pandas via numpy (no pyarrow/to_pandas, no set_index)
        body = R.pl_body
        cols = R.cols
        n = 0
        for o, ln in zip(R.pl_off, R.pl_len):
            sl = body.slice(o, ln)
            df = pd.DataFrame(
                sl.select(cols).to_numpy(),
                index=pd.DatetimeIndex(sl["time"].to_numpy(), name="time"),
                columns=cols,
                copy=False,
            )
            n += df.shape[0]
        return n

    def s6():
        n = 0
        for t, v in ga:
            n += v.shape[0]
        return n

    def s7a():
        n = 0
        for _, g in R.nw_pd.group_by(keys):
            n += g.drop(keys).shape[0]
        return n

    def s7a_pd():
        n = 0
        for _, g in R.nw_pd.group_by(keys):
            n += g.drop(keys).to_native().set_index("time").shape[0]
        return n

    def s7b():
        n = 0
        for _, g in R.nw_pl.group_by(keys):
            n += g.drop(keys).shape[0]
        return n

    def s8():
        body = R.pa_body
        n = 0
        for o, ln in zip(R.pa_off, R.pa_len):
            n += body.slice(o, ln).num_rows
        return n

    return {
        "S1_sktime": s1_sktime,
        "S1_sktime_nofreq": s1_sktime_nofreq,
        "S1_core": s1_core,
        "S2_iloc": s2_iloc,
        "S2_build": s2_build,
        "S3": s3,
        "S4": s4,
        "S5": s5,
        "S5p": s5p,
        "S5p_fast": s5p_fast,
        "S6": s6,
        "S7a": s7a,
        "S7a_pd": s7a_pd,
        "S7b": s7b,
        "S8": s8,
    }


# ---------------------------------------------------------------------------
# (C) reconstruct
# ---------------------------------------------------------------------------


OUT_GROUPS = ["pd", "pl", "np", "nw_pd", "pa"]


def outputs(R, ga, group):
    """Per-instance outputs (last 2 rows) in one strategy family's native format.

    Built per family (not all at once) to keep peak RSS low: 10k small pandas
    frames alone are ~100MB of Python objects.
    """
    if group == "pd":
        return {"pd": [inst.iloc[-2:] for _, _, inst in R.vec.items(R.X)]}
    if group == "pl":
        return {
            "pl": [R.pl_body.slice(o, ln).tail(2) for o, ln in zip(R.pl_off, R.pl_len)]
        }
    if group == "np":
        return {"np": [(t[-2:], v[-2:]) for t, v in ga]}
    if group == "nw_pd":
        pairs = [(k, g.drop(R.keys).tail(2)) for k, g in R.nw_pd.group_by(R.keys)]
        return {
            "nw_pd_keys": [k for k, _ in pairs],
            "nw_pd": [g for _, g in pairs],
        }
    if group == "pa":
        return {
            "pa": [R.pa_body.slice(o + ln - 2, 2) for o, ln in zip(R.pa_off, R.pa_len)]
        }
    raise ValueError(group)


def ops_C(R, outs, group):
    import narwhals as nw
    import polars as pl  # noqa: F401

    from protos.x3_strategies import (
        GroupedArray,
        pa_reconstruct,
        pd_reconstruct_fast,
        pl_reconstruct_lit,
        pl_reconstruct_vec,
    )

    keys = R.keys

    def s1():
        return R.vec.reconstruct(outs["pd"], overwrite_index=True)

    def s2():
        return pd_reconstruct_fast(outs["pd"], R.pd_keys, R.index_names)

    def s5_vec():
        return pl_reconstruct_vec(outs["pl"], R.pl_keys)

    def s5_lit():
        return pl_reconstruct_lit(outs["pl"], R.pl_keys_rows, keys)

    def s6():
        return GroupedArray.from_outputs(
            outs["np"], R.pd_keys, R.cols, R.index_names
        )

    def s6_topd():
        return GroupedArray.from_outputs(
            outs["np"], R.pd_keys, R.cols, R.index_names
        ).to_pandas_mi()

    def s7a_lit():
        parts = [
            o.with_columns([nw.lit(v).alias(n) for n, v in zip(keys, kk)])
            for kk, o in zip(outs["nw_pd_keys"], outs["nw_pd"])
        ]
        res = nw.concat(parts, how="vertical")
        return res.to_native().set_index(keys + ["time"])

    def s8():
        return pa_reconstruct(outs["pa"], R.pa_keys)

    by_group = {
        "pd": {"S1_sktime": s1, "S2_codes": s2},
        "pl": {"S3/S4/S5_vec": s5_vec, "S3/S4/S5_lit": s5_lit},
        "np": {"S6": s6, "S6_to_pd": s6_topd},
        "nw_pd": {"S7a_lit": s7a_lit},
        "pa": {"S8": s8},
    }
    return by_group[group]


def run_C(R, ga, rounds):
    """Time (C) family by family; interleaved within a family."""
    C, Cres = {}, {}
    for group in OUT_GROUPS:
        outs = outputs(R, ga, group)
        ops = ops_C(R, outs, group)
        C.update(interleaved(ops, rounds, keep=True))
        Cres.update({k: WARM.pop(k) for k in ops})
        del outs, ops
        gc.collect()
    return C, Cres


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------


def conv_inputs(X, hier):
    from sktime.datatypes import convert_to

    pl_mtype = "polars_hierarchical" if hier else "polars_panel"
    return {
        "X": X,
        "pl": to_polars_long(X),
        "pl_sk": convert_to(X, pl_mtype),
        "pl_mtype": pl_mtype,
        "pd_mtype": "pd_multiindex_hier" if hier else "pd-multiindex",
        "k": X.index.nlevels - 1,
        "keys": list(X.index.names[:-1]),
        "scitype": "Hierarchical" if hier else "Panel",
    }


def conv_ops(I):
    import narwhals as nw
    import pyarrow as pa

    from protos.x3_strategies import GroupedArray
    from sktime.datatypes import convert_to
    from sktime.datatypes._adapter.polars import convert_polars_to_pandas
    from sktime.datatypes._vectorize import VectorizedDF, prepare_VectorizedDF

    X, pl_df, pl_sk, k, keys = I["X"], I["pl"], I["pl_sk"], I["k"], I["keys"]
    value_cols = list(X.columns)

    return {
        "pd->polars_long (reset_index+from_pandas)": lambda: to_polars_long(X),
        "pd->polars_sktime (convert_to)": lambda: convert_to(X, I["pl_mtype"]),
        "polars_sktime->pd (convert_to)": lambda: convert_to(pl_sk, I["pd_mtype"]),
        "polars_sktime->pd (raw converter, no check)": lambda: (
            convert_polars_to_pandas(pl_sk)
        ),
        "polars_long->pd MI (to_pandas+set_index)": lambda: pl_df.to_pandas().set_index(
            keys + ["time"]
        ),
        "pd->GroupedArray": lambda: GroupedArray.from_pandas_mi(X, k),
        "polars->GroupedArray": lambda: GroupedArray.from_polars(
            pl_df, keys, "time", value_cols
        ),
        "pd->narwhals[pandas] (reset_index)": lambda: nw.from_native(
            X.reset_index(), eager_only=True
        ),
        "polars->narwhals[polars]": lambda: nw.from_native(pl_df, eager_only=True),
        "pd->arrow (reset_index+from_pandas)": lambda: pa.Table.from_pandas(
            X.reset_index(), preserve_index=False
        ),
        "polars->arrow (to_arrow)": lambda: pl_df.to_arrow(),
        "pd->S1 schema (VectorizedDF.__init__)": lambda: VectorizedDF(
            X,
            iterate_as="Series",
            is_scitype=I["scitype"],
            X_orig_mtype=I["pd_mtype"],
            freq="D",
        ),
        "pd->prepare_VectorizedDF (check+convert+freq+schema)": lambda: (
            prepare_VectorizedDF(X, is_scitype=None)
        ),
        "polars_sktime->prepare_VectorizedDF (current polars path)": lambda: (
            prepare_VectorizedDF(pl_sk, is_scitype=None)
        ),
    }


def mem_case(name, n_series, n_time, n_cols, hier):
    """Subprocess body: measure peak/retained RSS of one conversion."""
    # warm imports / registries on a tiny dataset so they are not counted, and
    # so the measured call does not reuse memory freed by a full-size warmup
    conv_ops(conv_inputs(make_data(4, 5, n_cols, hier), hier))[name]()
    X = make_data(n_series, n_time, n_cols, hier)
    I = conv_inputs(X, hier)
    fn = conv_ops(I)[name]
    gc.collect()
    base = rss_mb()
    res, peak = peak_rss_delta_mb(fn, interval=0.001)
    gc.collect()
    retained = rss_mb() - base
    del res
    return {"peak_mb": peak, "retained_mb": retained, "base_rss_mb": base}


def run_mem(n_series, n_time, n_cols, hier, names):
    out = {}
    for name in names:
        cmd = [
            sys.executable,
            "-W",
            "ignore",
            os.path.abspath(__file__),
            "--mem-case",
            name,
            "--n-series",
            str(n_series),
            "--n-time",
            str(n_time),
            "--n-cols",
            str(n_cols),
        ]
        if hier:
            cmd.append("--hier")
        p = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
        try:
            out[name] = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            out[name] = {"error": (p.stderr or p.stdout)[-800:]}
    return out


# ---------------------------------------------------------------------------
# micro-ops: per single slice, to explain WHY
# ---------------------------------------------------------------------------


def micro(R, pdpos, ga):
    import polars as pl  # noqa: F401

    from sktime.datatypes._vectorize import _enforce_index_freq

    i = len(R.pd_indptr) // 2
    s, e = int(R.pd_indptr[i]), int(R.pd_indptr[i + 1])
    X, lv = R.X, R.lv
    sl_mi = X.iloc[s:e]
    sl = sl_mi.droplevel(lv)
    t = pdpos.time
    vals = pdpos.vals
    body = R.pl_body
    pbody = R.pa_body
    plsl = body.slice(s, e - s)
    gb = X.groupby(level=R.lv_gb, sort=False)
    first_key = R.row_ix[i]
    from pandas.tseries.frequencies import to_offset

    off = to_offset("D")
    t_i8 = t.asi8
    idx_cache = {(t_i8[s], e - s): t[s:e]}

    def fresh_slice():
        return X.iloc[s:e].droplevel(lv)

    out = {}
    N = 300
    cases = {
        "pd: X.iloc[s:e] (MI frame slice)": lambda: X.iloc[s:e],
        "pd: .droplevel(inst levels) on slice": lambda: sl_mi.droplevel(lv),
        "pd: groupby.get_group(key)": lambda: gb.get_group(first_key),
        "pd: _enforce_index_freq (infer_freq) on fresh slice": lambda: (
            _enforce_index_freq(fresh_slice())
        ),
        "pd: fresh slice only (baseline for line above)": fresh_slice,
        "pd: DatetimeIndex(t[s:e], freq='D') (validated)": lambda: pd.DatetimeIndex(
            t[s:e], freq="D"
        ),
        "pd: date_range(start, periods, freq) (trusted)": lambda: pd.date_range(
            t[s], periods=e - s, freq="D"
        ),
        "pd: DataFrame(vals[s:e], index=t[s:e])": lambda: pd.DataFrame(
            vals[s:e], index=t[s:e], columns=R.cols, copy=False
        ),
        "pd: _enforce_index_freq on freq-less built slice": lambda: (
            _enforce_index_freq(
                pd.DataFrame(vals[s:e], index=t[s:e], columns=R.cols, copy=False)
            )
        ),
        "pd: pd.infer_freq(t[s:e]) alone": lambda: pd.infer_freq(t[s:e]),
        "pd: DatetimeIndex._simple_new(arr[s:e], freq) (private, trusted)": lambda: (
            pd.DatetimeIndex._simple_new(
                type(t._data)._simple_new(
                    t._data._ndarray[s:e], freq=off, dtype=t.dtype
                ),
                name=t.name,
            )
        ),
        "pd: DataFrame(vals[s:e], index=cached_idx) (shared time idx)": lambda: (
            pd.DataFrame(
                vals[s:e], index=idx_cache[(t_i8[s], e - s)], columns=R.cols, copy=False
            )
        ),
        "pd: slice.iloc[-2:] (output)": lambda: sl.iloc[-2:],
        "pl: body.slice(o, n)": lambda: body.slice(s, e - s),
        "pl: full.slice(o,n).drop(keys)": lambda: R.pl.slice(s, e - s).drop(R.keys),
        "pl: slice.to_pandas()": lambda: plsl.to_pandas(),
        "pl: slice.to_pandas().set_index('time')": lambda: plsl.to_pandas().set_index(
            "time"
        ),
        "pl: slice.to_numpy()": lambda: plsl.select(R.cols).to_numpy(),
        "pl: df.filter(key==k) (naive select)": lambda: R.pl.filter(
            pl_key_expr(R, i)
        ),
        "np: values[s:e] + time[s:e] views": lambda: (ga.values[s:e], ga.time[s:e]),
        "pa: table.slice(o, n)": lambda: pbody.slice(s, e - s),
        "pa: slice.to_pandas()": lambda: pbody.slice(s, e - s).to_pandas(),
    }
    for name, fn in cases.items():
        out[name] = bench(fn, repeat=5, number=N) * 1e6

    # global freq check, done once on the whole frame (alternative to per-slice
    # infer_freq): all within-instance time deltas equal one step
    def global_freq_check():
        ti = t.asi8
        d = np.diff(ti)
        inner = np.ones(len(d), dtype=bool)
        inner[R.pd_indptr[1:-1] - 1] = False
        dd = d[inner]
        return dd.size == 0 or bool((dd == dd[0]).all())

    out["pd: GLOBAL vectorized freq check, whole frame (us total)"] = (
        bench(global_freq_check, repeat=5) * 1e6
    )
    return out


def pl_key_expr(R, i):
    import polars as pl

    row = R.pl_keys_rows[i]
    expr = None
    for n, v in zip(R.keys, row):
        c = pl.col(n) == v
        expr = c if expr is None else expr & c
    return expr


# ---------------------------------------------------------------------------
# calibration: real forecaster cost per slice
# ---------------------------------------------------------------------------


def calib(R):
    from sktime.forecasting.naive import NaiveForecaster

    _, _, inst = next(iter(R.vec.items(R.X)))

    def one():
        f = NaiveForecaster()
        f.fit(inst, fh=[1, 2])
        return f.predict()

    return {"NaiveForecaster fit+predict on 1 slice (us)": bench(one, repeat=5) * 1e6}


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def checks(R, ga, pdpos, Bops, Cres, B_warm):
    out = {}
    n_rows = len(R.X)
    out["B_rows_equal"] = {k: (B_warm[k] == n_rows) for k in Bops}
    s1_keys = [k if isinstance(k, tuple) else (k,) for k in R.row_ix]
    out["S2_keys_equal_S1"] = bool(R.pd_keys.equals(R.row_ix))
    out["S3_order_equal_S1"] = list(
        R.pl.partition_by(R.keys, maintain_order=True, as_dict=True).keys()
    ) == s1_keys
    out["S4_order_equal_S1"] = [
        k for k, _ in R.pl.group_by(R.keys, maintain_order=True)
    ] == s1_keys
    out["S5_order_equal_S1"] = R.pl_keys_rows == s1_keys
    out["S7a_order_equal_S1"] = [k for k, _ in R.nw_pd.group_by(R.keys)] == s1_keys
    out["S7b_order_equal_S1 (narwhals polars, no maintain_order)"] = [
        k for k, _ in R.nw_pl.group_by(R.keys)
    ] == s1_keys
    out["S8_order_equal_S1"] = list(R.pa_keys.to_pylist()) == [
        dict(zip(R.keys, k)) for k in s1_keys
    ]
    # slice content: S2_build/S2_iloc vs S1_sktime first slice
    _, _, s1_first = next(iter(R.vec.items(R.X)))
    b_first = next(iter(pdpos.iter_build()))
    i_first = next(iter(pdpos.iter_iloc()))
    out["S2_build_slice_values_equal"] = bool(
        np.array_equal(s1_first.to_numpy(), b_first.to_numpy())
        and s1_first.index.equals(b_first.index)
    )
    out["S2_iloc_slice_equal"] = bool(s1_first.equals(i_first))
    out["S1_slice_freq"] = str(s1_first.index.freq)
    out["S2_build_slice_freq"] = str(b_first.index.freq)
    out["S2_iloc_slice_freq"] = str(i_first.index.freq)
    parts = R.pl.partition_by(
        R.keys, maintain_order=True, as_dict=True, include_key=False
    )
    out["S3_partitions_total_estimated_mb"] = (
        sum(p.estimated_size() for p in parts.values()) / 2**20
    )
    out["S3_source_body_estimated_mb"] = R.pl_body.estimated_size() / 2**20
    del parts
    g_nf = next(iter(R.X_nf.groupby(level=R.lv_gb, sort=False)))[1].droplevel(R.lv)
    out["S1_nofreq_slice_freq_before_enforce"] = str(g_nf.index.freq)
    # what a narwhals[pandas] slice looks like natively (C2 evidence)
    _, g = next(iter(R.nw_pd.group_by(R.keys)))
    nat = g.to_native()
    out["S7a_native_slice_index_type"] = type(nat.index).__name__
    out["S7a_native_slice_index_head"] = [int(x) for x in nat.index[:3]]
    out["S7a_native_slice_columns"] = [str(c) for c in nat.columns]
    # reconstruct equality vs S1
    ref = Cres["S1_sktime"]
    out["C_S2_codes_equals_S1"] = bool(ref.equals(Cres["S2_codes"])) and list(
        ref.index.names
    ) == list(Cres["S2_codes"].index.names)
    out["C_S6_to_pd_equals_S1"] = bool(
        np.array_equal(ref.to_numpy(), Cres["S6_to_pd"].to_numpy())
        and ref.index.equals(Cres["S6_to_pd"].index)
    )
    out["C_S7a_lit_equals_S1_values"] = bool(
        np.array_equal(ref.to_numpy(), Cres["S7a_lit"].to_numpy())
    )
    out["C_S7a_lit_index_names"] = [str(x) for x in Cres["S7a_lit"].index.names]
    pv = Cres["S3/S4/S5_vec"]
    out["C_pl_vec_values_equal_S1"] = bool(
        np.array_equal(ref.to_numpy(), pv.select(R.cols).to_numpy())
    )
    out["C_pl_lit_equals_vec"] = bool(
        Cres["S3/S4/S5_lit"].select(pv.columns).equals(pv)
    )
    pa_r = Cres["S8"]
    out["C_pa_values_equal_S1"] = bool(
        np.array_equal(ref.to_numpy(), pa_r.select(R.cols).to_pandas().to_numpy())
    )
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


WARM = {}  # name -> return value of the warmup call (used by checks)
SLOW_S = 3.0  # callables slower than this get at most 2 timed rounds


def interleaved(ops, rounds, warmup=1, keep=False):
    """Interleaved min-of-rounds timing (s) for a dict of zero-arg callables.

    Each round calls every callable once (gc.collect before each, via bench), so
    load from other processes affects all strategies alike; min over rounds.
    """
    best = {k: float("inf") for k in ops}
    warm_t = {}
    for _ in range(warmup):
        for k, fn in ops.items():
            t0 = time.perf_counter()
            res = fn()
            if keep:
                WARM[k] = res
            del res
            warm_t[k] = time.perf_counter() - t0
    for r in range(rounds):
        for k, fn in ops.items():
            if r >= 2 and warm_t.get(k, 0) > SLOW_S:
                continue
            best[k] = min(best[k], bench(fn, repeat=1, number=1, warmup=0))
    return best


def ops_noncontig(R):
    """Setup/iteration cost when instance rows are NOT contiguous (time-major)."""
    from protos.x3_strategies import PandasPositional, pl_offsets

    X_tm = R.X.sort_index(level=-1, sort_remaining=False, kind="stable")
    pl_tm = R.pl.sort("time", maintain_order=True)
    keys, lv, lv_gb = R.keys, R.lv, R.lv_gb

    def s1_core_tm():
        n = 0
        for _, g in X_tm.groupby(level=lv_gb, sort=False):
            n += g.droplevel(lv).shape[0]
        return n

    return {
        "S1_core iterate, contiguous": lambda: sum(
            g.droplevel(lv).shape[0] for _, g in R.X.groupby(level=lv_gb, sort=False)
        ),
        "S1_core iterate, time-major": s1_core_tm,
        "S2 setup (codes->indptr), contiguous": lambda: PandasPositional(R.X, R.k),
        "S2 setup (codes->argsort->take), time-major": lambda: PandasPositional(
            X_tm, R.k
        ),
        "S3 partition_by, contiguous": lambda: R.pl.partition_by(
            keys, maintain_order=True, as_dict=True, include_key=False
        ),
        "S3 partition_by, time-major": lambda: pl_tm.partition_by(
            keys, maintain_order=True, as_dict=True, include_key=False
        ),
        "S5 setup (rle), contiguous": lambda: pl_offsets(R.pl, keys),
        "S5 setup (stable sort by keys + rle), time-major": lambda: pl_offsets(
            pl_tm.sort(keys, maintain_order=True), keys
        ),
    }


def run_point(n_series, n_time, args):
    from protos.x3_strategies import GroupedArray, PandasPositional

    t0 = time.perf_counter()
    X = make_data(n_series, n_time, args.n_cols, args.hier)
    R = Reps(X, args.hier)
    ga = GroupedArray.from_pandas_mi(X, R.k)
    pdpos = PandasPositional(X, R.k)
    n_inst = len(R.row_ix)
    rec = {
        "n_series": n_series,
        "n_time": n_time,
        "n_cols": args.n_cols,
        "n_rows": len(X),
        "n_instances": n_inst,
        "sizes_mb": sizes_mb(R),
    }
    rounds = args.rounds

    A = interleaved(ops_A(R), rounds)
    Bops = ops_B(R, ga, pdpos)
    B = interleaved(Bops, rounds, keep=True)
    B_warm = {k: WARM[k] for k in Bops}
    WARM.clear()
    C, Cres = run_C(R, ga, rounds)

    I = conv_inputs(X, args.hier)
    CV = interleaved(conv_ops(I), max(3, rounds - 2))
    del I

    rec["A_s"] = A
    rec["B_s"] = B
    rec["C_s"] = C
    rec["conv_s"] = CV
    rec["B_per_slice_us"] = {k: v / n_inst * 1e6 for k, v in B.items()}
    rec["C_per_slice_us"] = {k: v / n_inst * 1e6 for k, v in C.items()}
    rec["A_ratio_vs_S1"] = {k: v / A["S1"] for k, v in A.items()}
    rec["B_ratio_vs_S1_sktime"] = {k: v / B["S1_sktime"] for k, v in B.items()}
    rec["B_ratio_vs_S1_core"] = {k: v / B["S1_core"] for k, v in B.items()}
    rec["C_ratio_vs_S1"] = {k: v / C["S1_sktime"] for k, v in C.items()}
    rec["noncontig_s"] = interleaved(ops_noncontig(R), max(3, rounds - 2))
    rec["micro_us"] = micro(R, pdpos, ga)
    if not args.skip_calib and n_series <= 1000:
        rec["calib"] = calib(R)
    rec["checks"] = checks(R, ga, pdpos, Bops, Cres, B_warm)
    WARM.clear()
    del Cres, R, ga, pdpos
    gc.collect()
    if not args.skip_mem:
        rec["conv_mem"] = run_mem(
            n_series,
            n_time,
            args.n_cols,
            args.hier,
            list(conv_ops(conv_inputs(make_data(10, 5, args.n_cols, args.hier), args.hier))),
        )
    rec["rss_mb_end"] = rss_mb()
    import resource

    rec["ru_maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def fmt_table(rec, sect, unit_scale=1e3, unit="ms"):
    d = rec[sect]
    return ", ".join(f"{k}={v * unit_scale:.2f}{unit}" for k, v in d.items())


def merge():
    files = sorted(glob.glob(os.path.join(RES, f"{KEY}__*.json")))
    allres = {}
    for f in files:
        with open(f) as fh:
            allres[os.path.basename(f)[len(KEY) + 2 : -5]] = json.load(fh)
    with open(os.path.join(RES, f"{KEY}.json"), "w") as fh:
        json.dump({"key": KEY, "runs": allres}, fh, indent=1, default=str)
    print("merged", list(allres))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--hier", action="store_true")
    ap.add_argument("--n-cols", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--skip-mem", action="store_true")
    ap.add_argument("--skip-calib", action="store_true")
    ap.add_argument("--only", type=str, default=None, help="e.g. 1000x100")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--mem-case", type=str, default=None)
    ap.add_argument("--n-series", type=int, default=None)
    ap.add_argument("--n-time", type=int, default=None)
    args = ap.parse_args()

    if args.merge:
        merge()
        return
    if args.mem_case:
        r = mem_case(args.mem_case, args.n_series, args.n_time, args.n_cols, args.hier)
        print(json.dumps(r))
        return

    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    if args.only:
        grid = [tuple(int(x) for x in args.only.split("x"))]
    pdmaj = pd.__version__.split(".")[0]
    tag = args.tag or (
        f"{args.grid}_pd{pdmaj}{'_hier' if args.hier else ''}"
        f"{'_c' + str(args.n_cols) if args.n_cols != 1 else ''}"
    )
    result = {
        "key": KEY,
        "tag": tag,
        "env": env_info(),
        "python": platform.python_version(),
        "cmd": " ".join(sys.argv),
        "args": vars(args),
        "grid": grid,
        "points": [],
    }
    for n_series, n_time in grid:
        print(f"=== {tag} {n_series}x{n_time}", flush=True)
        rec = run_point(n_series, n_time, args)
        result["points"].append(rec)
        print("A:", fmt_table(rec, "A_s"), flush=True)
        print("B:", fmt_table(rec, "B_s"), flush=True)
        print(
            "B per slice:",
            ", ".join(f"{k}={v:.1f}us" for k, v in rec["B_per_slice_us"].items()),
        )
        print(
            "B ratio vs S1_sktime:",
            ", ".join(f"{k}={v:.3f}" for k, v in rec["B_ratio_vs_S1_sktime"].items()),
        )
        print("C:", fmt_table(rec, "C_s"), flush=True)
        print(
            "C ratio vs S1:",
            ", ".join(f"{k}={v:.3f}" for k, v in rec["C_ratio_vs_S1"].items()),
        )
        print("conv:", fmt_table(rec, "conv_s"), flush=True)
        print("noncontig:", fmt_table(rec, "noncontig_s"), flush=True)
        print("micro:", ", ".join(f"{k}={v:.1f}" for k, v in rec["micro_us"].items()))
        if "calib" in rec:
            print("calib:", rec["calib"])
        if "conv_mem" in rec:
            print(
                "conv_mem:",
                ", ".join(
                    f"{k}: peak={v.get('peak_mb', float('nan')):.1f}MB "
                    f"ret={v.get('retained_mb', float('nan')):.1f}MB"
                    for k, v in rec["conv_mem"].items()
                ),
            )
        print("checks:", json.dumps(rec["checks"]), flush=True)
        print(
            f"wall={rec['wall_s']:.1f}s rss_end={rec['rss_mb_end']:.0f}MB "
            f"maxrss={rec['ru_maxrss_mb']:.0f}MB",
            flush=True,
        )
        with open(os.path.join(RES, f"{KEY}__{tag}.json"), "w") as fh:
            json.dump(result, fh, indent=1, default=str)
    print("wrote", os.path.join(RES, f"{KEY}__{tag}.json"))


if __name__ == "__main__":
    main()
