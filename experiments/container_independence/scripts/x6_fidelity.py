"""x6: round-trip fidelity + zero-copy matrix + LOC for the hub spokes.

For each fixture (native container) and each spoke style
  hand  = hand-written spoke per library (style i)
  nw    = narwhals generic spoke (style ii-a)
  cap   = Arrow PyCapsule ingress + pyarrow spoke + narwhals.from_arrow egress (ii-b)
do native -> hub -> native and check STRICT equality (values, dtypes, names, freq,
row order, container type incl. lazy/eager), and record zero-copy of ingress.

Run:  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x6_fidelity.py
"""

import ast
import inspect
import json
import os
import sys
import textwrap
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ED, "protos"))

import x6_hub as H  # noqa: E402
from common import env_info, make_panel, to_polars_long  # noqa: E402


# ----------------------------------------------------------------------------- LOC
def loc(funcs):
    """Logical source lines (no blank, comment, or docstring lines)."""
    total = 0
    for f in funcs:
        src = textwrap.dedent(inspect.getsource(f))
        tree = ast.parse(src)
        doc_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(
                    getattr(first, "value", None), ast.Constant
                ) and isinstance(first.value.value, str):
                    doc_lines.update(range(first.lineno, first.end_lineno + 1))
        for i, line in enumerate(src.splitlines(), start=1):
            s = line.strip()
            if s and not s.startswith("#") and i not in doc_lines:
                total += 1
    return total


# ------------------------------------------------------------------------ fixtures
def fixtures():
    import dask.dataframe as dd
    import duckdb
    import polars as pl
    import pyarrow as pa

    from sktime.datatypes import convert_to

    fx = {}
    base = make_panel(6, 12, n_cols=2, n_levels=1)
    fx["pd_mi_datetime_D"] = (base, {})
    fx["pd_mi_period_M"] = (make_panel(6, 12, n_cols=1, n_levels=1,
                                       time_kind="period", freq="M"), {})
    fx["pd_mi_int"] = (make_panel(6, 12, n_cols=1, n_levels=1, time_kind="int"), {})
    fx["pd_hier_3lvl"] = (make_panel(8, 12, n_cols=1, n_levels=2), {})
    s0 = base.xs("s0").copy()
    s0.index = pd.DatetimeIndex(s0.index, freq="D", name="time")
    fx["pd_single_freq_D"] = (s0, {})
    fx["pd_series_named_h"] = (pd.Series(np.arange(20.0), name="y", index=pd.date_range(
        "2000", periods=20, freq="h")), {})
    fx["pd_series_range"] = (pd.Series(np.arange(20.0), name="y"), {})
    mixed = base.copy()
    mixed["f32"] = mixed["c0"].astype("float32")
    mixed["i64"] = np.arange(len(mixed), dtype="int64")
    mixed["cat"] = np.where(np.arange(len(mixed)) % 2, "a", "b")
    fx["pd_mi_mixed_dtypes"] = (mixed, {})
    rng = np.random.default_rng(0)
    fx["pd_mi_unsorted"] = (base.iloc[rng.permutation(len(base))], {})
    fx["pd_mi_unequal"] = (base.drop(index=base.index[[0, 1, 2, 30]]), {})
    tz = base.copy()
    tz.index = tz.index.set_levels(tz.index.levels[1].tz_localize("Europe/Berlin"),
                                   level=1)
    fx["pd_mi_tz"] = (tz, {})
    me = make_panel(4, 12, n_cols=1, n_levels=1, freq="ME")
    fx["pd_mi_monthly_ME"] = (me, {})
    cat = base.copy()
    cat.index = cat.index.set_levels(cat.index.levels[0].astype("category"), level=0)
    fx["pd_mi_categorical_level"] = (cat, {})
    nan = base.copy()
    nan.iloc[[3, 17], 0] = np.nan
    fx["pd_mi_nan"] = (nan, {})

    # polars
    pl_long = to_polars_long(base)
    kw = {"key_cols": ["h0"], "time_col": "time"}
    fx["pl_long_ns"] = (pl_long, kw)
    fx["pl_long_us"] = (pl_long.with_columns(pl.col("time").dt.cast_time_unit("us")),
                        kw)
    fx["pl_sktime_index_cols"] = (convert_to(base, "polars_panel"), {})
    fx["pl_lazy"] = (pl_long.lazy(), kw)
    fx["pl_nulls"] = (pl_long.with_columns(
        pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("c0"))
        .alias("c0")), kw)
    fx["pl_unsorted"] = (pl_long.sample(fraction=1.0, shuffle=True, seed=1), kw)
    fx["pl_categorical_key"] = (pl_long.with_columns(pl.col("h0").cast(pl.Categorical)),
                                kw)
    fx["pl_date_dtype"] = (pl_long.with_columns(pl.col("time").dt.date()), kw)

    # arrow
    pa_long = pa.Table.from_pandas(base.reset_index(), preserve_index=False)
    pa_long = pa_long.replace_schema_metadata(None)
    fx["pa_long"] = (pa_long, kw)
    fx["pa_chunked"] = (pa.concat_tables([pa_long.slice(0, 30), pa_long.slice(30)]), kw)
    fx["pa_dict_key"] = (pa_long.set_column(0, "h0", pa_long.column("h0")
                                            .dictionary_encode()), kw)
    fx["pa_custom_schema_metadata"] = (pa_long.replace_schema_metadata(
        {"origin": "x6"}), kw)
    fx["pa_from_pandas_MI_meta"] = (pa.Table.from_pandas(base), kw)

    # dask
    fx["dask_long"] = (dd.from_pandas(base.reset_index(), npartitions=2), kw)
    fx["dask_sktime_index_cols"] = (convert_to(base, "dask_panel"), {})

    # duckdb
    con = duckdb.connect()
    rel = con.from_arrow(pa_long)
    fx["duckdb_relation"] = (rel, kw)
    fx["_con"] = con  # keep alive
    return fx


# ---------------------------------------------------------------------- equality
def native_equal(a, b):
    """Strict equality; returns (ok, reason)."""
    import polars as pl
    import pyarrow as pa

    mod_a, mod_b = type(a).__module__.split(".")[0], type(b).__module__.split(".")[0]
    if type(a) is not type(b):
        return False, f"type {type(a).__name__} != {type(b).__name__}"
    try:
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b, check_exact=True, check_freq=True)
            for i in range(a.index.nlevels):
                da = a.index.get_level_values(i).dtype
                db = b.index.get_level_values(i).dtype
                if da != db:
                    return False, f"index level {i} dtype {da} != {db}"
            if getattr(a.index, "freq", None) != getattr(b.index, "freq", None):
                return False, f"freq {a.index.freq} != {b.index.freq}"
            if type(a.index) is not type(b.index):
                return False, f"index type {type(a.index).__name__}"
            return True, ""
        if isinstance(a, pd.Series):
            pd.testing.assert_series_equal(a, b, check_exact=True, check_freq=True)
            if type(a.index) is not type(b.index):
                return False, f"index type {type(a.index).__name__}"
            return True, ""
        if isinstance(a, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(a, pl.LazyFrame):
                a, b = a.collect(), b.collect()
            if a.schema != b.schema:
                return False, f"schema {dict(a.schema)} != {dict(b.schema)}"
            return (a.equals(b, null_equal=True), "values differ")
        if isinstance(a, pa.Table):
            if not a.schema.equals(b.schema, check_metadata=True):
                return False, f"schema {a.schema} != {b.schema}"
            return a.equals(b), "values differ"
        if mod_a == "dask":
            if a.npartitions != b.npartitions:
                return False, f"npartitions {a.npartitions} != {b.npartitions}"
            ca, cb = a.compute(), b.compute()
            pd.testing.assert_frame_equal(ca.reset_index(drop=True),
                                          cb.reset_index(drop=True), check_exact=True)
            return True, ""
        if mod_a.lstrip("_") == "duckdb":
            ta, tb = pa.table(a), pa.table(b)
            if not ta.schema.equals(tb.schema):
                return False, f"schema {ta.schema} != {tb.schema}"
            return ta.equals(tb), "values differ"
    except AssertionError as e:
        return False, str(e).strip().splitlines()[0][:160]
    return False, f"no comparator for {mod_a}/{mod_b}"


def src_value_views(native):
    """Zero-copy numpy views of the native value buffers, for shares_memory checks."""
    import polars as pl
    import pyarrow as pa

    mod = type(native).__module__.split(".")[0]
    try:
        if isinstance(native, pd.DataFrame):
            return [native.to_numpy(copy=False)]
        if isinstance(native, pd.Series):
            return [native.to_numpy(copy=False)]
        if isinstance(native, pl.DataFrame):
            return [native[c].to_numpy() for c in native.columns
                    if native.schema[c].is_float()]
        if isinstance(native, pa.Table):
            out = []
            for c in native.column_names:
                if pa.types.is_floating(native.schema.field(c).type):
                    for ch in native.column(c).chunks:
                        out.append(ch.to_numpy(zero_copy_only=False))
            return out
    except Exception:  # noqa: BLE001
        return []
    return []  # lazy / dask / duckdb: nothing materialized to share with


STYLES = {
    "hand": (H.from_native, H.to_native),
    "nw": (H.nw_from_native, H.nw_to_native),
    "cap": (H.capsule_from_native, H.capsule_to_native),
}


def main():
    fx = fixtures()
    fx.pop("_con")
    rows = []
    for name, (native, kw) in fx.items():
        src_views = src_value_views(native)
        for style, (fin, fout) in STYLES.items():
            rec = {"fixture": name, "style": style}
            try:
                if style == "hand" and type(native).__module__.lstrip("_").startswith(
                        "duckdb"):
                    rec["roundtrip_ok"] = None
                    rec["reason"] = "n/a: no hand-written duckdb spoke in prototype"
                    rows.append(rec)
                    print(f"{name:26s} {style:5s} n/a")
                    continue
                kw2 = {} if style == "cap" and not kw else kw
                hub = fin(native, **kw2)
                rec["values_zero_copy"] = bool(src_views) and any(
                    np.shares_memory(hub.values, v) for v in src_views)
                rec["n_instances"] = hub.n_instances
                rec["tmeta"] = {"kind": hub.tmeta.kind, "unit": hub.tmeta.unit,
                                "tz": hub.tmeta.tz, "freq": hub.tmeta.freq}
                out = fout(hub)
                ok, why = native_equal(out, native)
                rec["roundtrip_ok"] = bool(ok)
                rec["reason"] = "" if ok else why
            except Exception as e:  # noqa: BLE001
                rec["roundtrip_ok"] = False
                rec["reason"] = f"ERROR {type(e).__name__}: {str(e).splitlines()[0][:150]}"
                rec["tb"] = traceback.format_exc().splitlines()[-3:]
            rows.append(rec)
            print(f"{name:26s} {style:5s} ok={rec['roundtrip_ok']!s:5s} "
                  f"zc={rec.get('values_zero_copy')!s:5s} {rec['reason'][:110]}")

    locs = {
        "hand_all_libs (pandas MI+long, polars, arrow, dask + dispatch)":
            loc(H.HAND_SPOKES),
        "hand_pandas_only": loc([H._pd_time_meta, H.from_pandas, H.to_pandas]),
        "hand_polars_only": loc([H._pl_time_meta, H.from_polars, H.to_polars]),
        "hand_arrow_only": loc([H._np_from_arrow, H.from_arrow, H.to_arrow]),
        "narwhals_generic": loc(H.NW_SPOKES),
        "capsule_generic (+ arrow spoke it reuses)":
            loc(H.CAPSULE_SPOKES + [H._np_from_arrow, H.from_arrow, H.to_arrow]),
        "shared_numpy_core": loc([H._run_starts_from_codes, H._composite,
                                  H._first_appearance_perm, H._starts_to_indptr,
                                  H._stack_cols, H.infer_step_numpy,
                                  H._detect_long_cols, H._split_value_cols]),
    }
    print(json.dumps(locs, indent=1))
    summ = {}
    for style in STYLES:
        r = [x for x in rows if x["style"] == style]
        summ[style] = {"pass": sum(bool(x["roundtrip_ok"]) for x in r),
                       "n/a": sum(x["roundtrip_ok"] is None for x in r),
                       "total": len(r)}
    print(json.dumps(summ))
    tag = "pd" + pd.__version__.split(".")[0]
    out = os.path.join(ED, "results", f"x6_fidelity_{tag}.json")
    with open(out, "w") as f:
        json.dump({"env": env_info(), "rows": rows, "loc": locs, "summary": summ}, f,
                  indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    main()
