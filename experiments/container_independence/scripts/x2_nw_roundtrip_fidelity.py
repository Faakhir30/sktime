"""x2 (3)+(5): index fidelity of MultiIndex -> columns -> narwhals -> MultiIndex.

For several sktime-style MultiIndex frames (time level = DatetimeIndex with freq,
PeriodIndex(M), RangeIndex/int, tz-aware datetime; categorical instance level;
3-level hierarchical; non-lexsorted instance order) run several code paths and
compare the reconstructed frame with the original:

  A  current sktime: prepare_VectorizedDF -> items -> reconstruct
  A0 pure pandas groupby(level=..., sort=False) slice, *without* _enforce_index_freq
  B  reset_index -> nw group_by(inst) iteration -> per-group set_index(time)
     -> pd.concat(keys=schema row index) [what a narwhals VectorizedDF would do]
  B2 reset_index -> nw group_by iteration -> nw.concat -> set_index(names)
  C  reset_index -> nw filter + sort -> to_native -> set_index(names)
  D  nw.maybe_reset_index (dev repro candidate)
  D2 reset_index(drop=True) by hand
  D3 reset_index -> nw group_by(inst).agg -> to_native (aggregation result)
  D4 reset_index -> nw.from_native -> ... -> to_native() *without* set_index
  E  polars via narwhals: pl.from_pandas(reset frame) -> nw ops -> to_pandas
     -> set_index(names)
  F  sktime convert_pandas_to_polars -> convert_polars_to_pandas (existing adapter)

Recorded per (case, path): level names, level order, per-level dtype, row order,
values, freq of the time level (both the MultiIndex level attribute and
the per-instance slice ``X.xs(key).index.freq``, which is what sktime's
``coerce_to_multiindex.infer_freq`` reads) and the freq of the slices yielded to an
estimator (iteration paths).

Run::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x2_nw_roundtrip_fidelity.py
"""

import json
import os
import sys
import traceback
import warnings

import narwhals as nw
import numpy as np
import pandas as pd

from common import env_info, make_panel, make_panel_fast

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


# ----------------------------------------------------------------------------------
# test frames
# ----------------------------------------------------------------------------------
def make_cases():
    cases = {}
    cases["dt_freqD"] = ("Panel", make_panel(4, 8, n_cols=2, n_levels=1, time_kind="datetime"))
    # from_product(date_range) -> the time *level* carries freq=D (make_panel uses
    # from_tuples, whose levels carry no freq)
    cases["dt_freqD_product"] = ("Panel", make_panel_fast(4, 8, n_cols=2))
    cases["dt_freqMS"] = (
        "Panel",
        make_panel(4, 8, n_cols=2, n_levels=1, time_kind="datetime", freq="MS"),
    )
    cases["period_M"] = (
        "Panel",
        make_panel(4, 8, n_cols=2, n_levels=1, time_kind="period", freq="M"),
    )
    cases["int_range"] = ("Panel", make_panel(4, 8, n_cols=2, n_levels=1, time_kind="int"))

    df = make_panel(4, 8, n_cols=2, n_levels=1, time_kind="datetime", freq="h")
    lv = df.index.levels[-1].tz_localize("Europe/Berlin")
    df.index = df.index.set_levels(lv, level=-1)
    cases["dt_tz_Berlin_h"] = ("Panel", df)

    df = make_panel(4, 8, n_cols=2, n_levels=1, time_kind="datetime")
    df.index = df.index.set_levels(
        pd.CategoricalIndex(df.index.levels[0]), level=0, verify_integrity=False
    )
    cases["categorical_inst"] = ("Panel", df)

    cases["hier3_dt"] = ("Hierarchical", make_panel(8, 6, n_cols=2, n_levels=2))

    # instance order not lexsorted: s2, s0, s3, s1
    df = make_panel(4, 8, n_cols=2, n_levels=1, time_kind="datetime")
    order = ["s2", "s0", "s3", "s1"]
    df = pd.concat([df.xs(k, level=0, drop_level=False) for k in order])
    cases["unsorted_inst"] = ("Panel", df)
    return cases


# ----------------------------------------------------------------------------------
# description / comparison helpers
# ----------------------------------------------------------------------------------
def _freqstr(ix):
    f = getattr(ix, "freq", None)
    return None if f is None else getattr(f, "freqstr", str(f))


def describe(df):
    ix = df.index
    d = {
        "index_type": type(ix).__name__,
        "nlevels": ix.nlevels,
        "names": [None if n is None else str(n) for n in ix.names],
    }
    if isinstance(ix, pd.MultiIndex):
        d["level_dtypes"] = [str(ix.get_level_values(i).dtype) for i in range(ix.nlevels)]
        d["time_level_attr_freq"] = _freqstr(ix.levels[-1])
        try:
            first = ix.droplevel(-1)[0]
            d["xs_first_instance_freq"] = _freqstr(df.xs(first).index)
        except Exception as e:  # noqa: BLE001
            d["xs_first_instance_freq"] = f"err {type(e).__name__}"
    else:
        d["level_dtypes"] = [str(ix.dtype)]
    d["col_dtypes"] = [str(t) for t in df.dtypes]
    return d


def compare(orig, res):
    do, dr = describe(orig), describe(res)
    out = {
        "index_type": dr["index_type"],
        "names_ok": do["names"] == dr["names"],
        "names": dr["names"],
        "nlevels_ok": do["nlevels"] == dr["nlevels"],
        "level_dtypes_ok": do["level_dtypes"] == dr["level_dtypes"],
        "level_dtypes": dr["level_dtypes"],
        "col_dtypes_ok": do["col_dtypes"] == dr["col_dtypes"],
        "time_level_attr_freq": (do.get("time_level_attr_freq"), dr.get("time_level_attr_freq")),
        "xs_first_instance_freq": (
            do.get("xs_first_instance_freq"),
            dr.get("xs_first_instance_freq"),
        ),
    }
    try:
        out["row_order_ok"] = bool(
            len(orig) == len(res) and orig.index.equals(res.index)
        )
    except Exception:  # noqa: BLE001
        out["row_order_ok"] = False
    try:
        out["values_ok"] = bool(
            res.sort_index().reindex(columns=orig.columns).to_numpy().tolist()
            == orig.sort_index().to_numpy().tolist()
        )
    except Exception as e:  # noqa: BLE001
        out["values_ok"] = f"err {type(e).__name__}"
    try:
        out["frame_equals"] = bool(orig.equals(res))
    except Exception:  # noqa: BLE001
        out["frame_equals"] = False
    return out


def slice_info(s):
    ix = s.index
    return {
        "type": type(ix).__name__,
        "name": None if ix.name is None else str(ix.name),
        "dtype": str(ix.dtype),
        "freq": _freqstr(ix),
    }


# ----------------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------------
def path_A_sktime(df, scitype):
    from sktime.datatypes._vectorize import prepare_VectorizedDF

    vec, X_mi = prepare_VectorizedDF(df, iterate_as="Series", is_scitype=scitype)
    items = list(vec.items(X=X_mi))
    rec = vec.reconstruct([x[2] for x in items])
    return rec, slice_info(items[0][2]), repr(items[0][0])


def path_A0_pandas_groupby(df, scitype):
    lv = list(range(df.index.nlevels - 1))
    lv = lv[0] if len(lv) == 1 else lv
    groups = list(df.groupby(level=lv, sort=False))
    first = groups[0][1].droplevel(lv)
    keys = [k for k, _ in groups]
    row_ix = df.index.droplevel(-1).unique()
    rec = pd.concat([g.droplevel(lv) for _, g in groups], keys=row_ix)
    return rec, slice_info(first), repr(keys[0])


def _inst_names(df):
    return list(df.index.names[:-1])


def path_B_nw_iter(df, scitype):
    names = list(df.index.names)
    inst, tname = names[:-1], names[-1]
    row_ix = df.index.droplevel(-1).unique()  # schema, as VectorizedDF stores it
    nf = nw.from_native(df.reset_index(), eager_only=True)
    keys, slices = [], []
    for key, g in nf.group_by(inst):
        gn = nw.to_native(g.drop(inst)).set_index(tname)
        keys.append(key)
        slices.append(gn)
    # align to schema order (group_by order is backend dependent)
    key_to_slice = {tuple(k): s for k, s in zip(keys, slices)}
    row_keys = [k if isinstance(k, tuple) else (k,) for k in row_ix]
    ordered = [key_to_slice[tuple(k)] for k in row_keys]
    rec = pd.concat(ordered, keys=row_ix)
    return rec, slice_info(slices[0]), repr(keys[0])


def path_B2_nw_iter_nwconcat(df, scitype):
    names = list(df.index.names)
    inst = names[:-1]
    nf = nw.from_native(df.reset_index(), eager_only=True)
    groups = [g for _, g in nf.group_by(inst)]
    rec = nw.to_native(nw.concat(groups)).set_index(names)
    return rec, None, None


def path_C_nw_filter_sort(df, scitype):
    names = list(df.index.names)
    nf = nw.from_native(df.reset_index(), eager_only=True)
    out = nf.filter(nw.col("c0") > -np.inf).sort(names)
    rec = nw.to_native(out).set_index(names)
    return rec, None, None


def path_D_maybe_reset_index(df, scitype):
    out = nw.maybe_reset_index(nw.from_native(df, eager_only=True))
    rec = nw.to_native(out)
    return rec, None, f"columns={list(rec.columns)}"


def path_D2_reset_drop(df, scitype):
    rec = df.reset_index(drop=True)
    return rec, None, f"columns={list(rec.columns)}"


def path_D3_nw_agg(df, scitype):
    inst = _inst_names(df)
    nf = nw.from_native(df.reset_index(), eager_only=True)
    rec = nw.to_native(nf.group_by(inst).agg(nw.col("c0").mean()))
    return rec, None, f"columns={list(rec.columns)}"


def path_D4_no_set_index(df, scitype):
    nf = nw.from_native(df.reset_index(), eager_only=True)
    rec = nw.to_native(nf.filter(nw.col("c0") > -np.inf))
    return rec, None, f"columns={list(rec.columns)}"


def path_E_polars(df, scitype):
    import polars as pl

    names = list(df.index.names)
    inst = names[:-1]
    pf = pl.from_pandas(df.reset_index())
    nf = nw.from_native(pf, eager_only=True)
    schema = {k: str(v) for k, v in nf.schema.items()}
    groups = [g for _, g in nf.group_by(inst)]
    out = nw.concat(groups).sort(names)  # polars group order is not stable
    rec = nw.to_native(out).to_pandas().set_index(names)
    return rec, None, f"nw schema={schema}"


def path_F_sktime_polars_adapter(df, scitype):
    from sktime.datatypes._adapter.polars import (
        convert_pandas_to_polars,
        convert_polars_to_pandas,
    )

    pf = convert_pandas_to_polars(df)
    rec = convert_polars_to_pandas(pf)
    return rec, None, f"polars cols={pf.columns[:3]}..."


PATHS = {
    "A_sktime_vectorizeddf": path_A_sktime,
    "A0_pd_groupby_level_noinfer": path_A0_pandas_groupby,
    "B_nw_iter_setindex_schema": path_B_nw_iter,
    "B2_nw_iter_nwconcat_setindex": path_B2_nw_iter_nwconcat,
    "C_nw_filter_sort_setindex": path_C_nw_filter_sort,
    "D_nw_maybe_reset_index": path_D_maybe_reset_index,
    "D2_pd_reset_index_drop": path_D2_reset_drop,
    "D3_nw_groupby_agg": path_D3_nw_agg,
    "D4_nw_ops_no_set_index": path_D4_no_set_index,
    "E_polars_via_nw_setindex": path_E_polars,
    "F_sktime_polars_adapter": path_F_sktime_polars_adapter,
}


# ----------------------------------------------------------------------------------
# narwhals dtype handling of the index-derived columns (part 5)
# ----------------------------------------------------------------------------------
def dtype_probe(cases):
    out = {}
    for name, (_, df) in cases.items():
        flat = df.reset_index()
        tname = df.index.names[-1]
        nf = nw.from_native(flat, eager_only=True)
        rec = {
            "pandas_dtype": str(flat[tname].dtype),
            "nw_dtype": str(nf.schema[tname]),
            "inst_pandas_dtype": str(flat[df.index.names[0]].dtype),
            "inst_nw_dtype": str(nf.schema[df.index.names[0]]),
        }
        probes = {
            "sort(time)": lambda: nf.sort(tname),
            "filter(time == first)": lambda: nf.filter(nw.col(tname) == flat[tname].iloc[0]),
            "time.diff()": lambda: nf.select(nw.col(tname).diff()),
            "time.dt.year()": lambda: nf.select(nw.col(tname).dt.year()),
            "time.max() by inst": lambda: nf.group_by(df.index.names[0]).agg(
                nw.col(tname).max()
            ),
            "to_polars()": lambda: nf.to_polars(),
            "to_arrow()": lambda: nf.to_arrow(),
        }
        for pn, fn in probes.items():
            try:
                r = fn()
                if pn == "to_polars()":
                    rec[pn] = f"ok, polars dtype {r.schema[tname]}"
                elif pn == "to_arrow()":
                    rec[pn] = f"ok, arrow type {r.schema.field(tname).type}"
                elif pn == "time.max() by inst":
                    rec[pn] = f"ok, result dtype {nw.to_native(r)[tname].dtype}"
                else:
                    rec[pn] = "ok"
            except Exception as e:  # noqa: BLE001
                rec[pn] = f"raised {type(e).__name__}: {str(e).splitlines()[0][:100]}"
        try:
            import polars as pl

            p = pl.from_pandas(flat)
            back = p.to_pandas()
            rec["pl.from_pandas"] = f"ok, polars dtype {p.schema[tname]}"
            rec["polars->pandas dtype"] = str(back[tname].dtype)
            rec["polars->pandas inst dtype"] = str(back[df.index.names[0]].dtype)
            # narwhals ops on the polars-backed column (e.g. period -> extension)
            npf = nw.from_native(p, eager_only=True)
            rec["nw(polars) dtype"] = str(npf.schema[tname])
            pprobes = {
                "nw(polars) sort(time)": lambda: npf.sort(tname),
                "nw(polars) filter(time==first)": lambda: npf.filter(
                    nw.col(tname) == npf[tname][0]
                ),
                "nw(polars) time.diff()": lambda: npf.select(nw.col(tname).diff()),
                "nw(polars) time.dt.year()": lambda: npf.select(nw.col(tname).dt.year()),
                "nw(polars) time.max() by inst": lambda: npf.group_by(
                    df.index.names[0]
                ).agg(nw.col(tname).max()),
            }
            for pn, fn in pprobes.items():
                try:
                    fn()
                    rec[pn] = "ok"
                except Exception as e:  # noqa: BLE001
                    rec[pn] = (
                        f"raised {type(e).__name__}: {str(e).splitlines()[0][:100]}"
                    )
        except Exception as e:  # noqa: BLE001
            rec["pl.from_pandas"] = f"raised {type(e).__name__}: {str(e).splitlines()[0][:100]}"
        out[name] = rec
    return out


def main():
    cases = make_cases()
    results = {"env": env_info(), "python": sys.version.split()[0], "cases": {}}
    for cname, (scitype, df) in cases.items():
        results["cases"][cname] = {"orig": describe(df), "paths": {}}
        for pname, fn in PATHS.items():
            # FRESH frame per path: df.copy() shares Index objects, and under
            # pandas 3 sktime's _enforce_index_freq mutates the shared time-level
            # array (see x2_freq_mutation_probe.py), which would leak across paths.
            _, ref = make_cases()[cname]
            _, inp = make_cases()[cname]
            before = describe(inp)
            try:
                rec, sl, extra = fn(inp, scitype)
                r = compare(ref, rec)
                r["first_slice"] = sl
                r["extra"] = extra
                r["input_mutated"] = describe(inp) != before
            except Exception as e:  # noqa: BLE001
                r = {
                    "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}",
                    "tb": traceback.format_exc().splitlines()[-3:],
                }
            results["cases"][cname]["paths"][pname] = r
    results["dtype_probe"] = dtype_probe(cases)

    tag = "pd" + pd.__version__.split(".")[0]
    path = os.path.join(RES, f"x2_nw_roundtrip_fidelity_{tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1, default=str)

    # compact console table
    print(f"pandas {pd.__version__} narwhals {nw.__version__}")
    for cname, c in results["cases"].items():
        o = c["orig"]
        print(
            f"\n### {cname}: names={o['names']} dtypes={o['level_dtypes']} "
            f"lvl_freq={o.get('time_level_attr_freq')} xs_freq={o.get('xs_first_instance_freq')}"
        )
        print(
            f"{'path':32s} {'idx':10s} names lvl_dt rows vals  eq   "
            f"lvl_freq   xs_freq   mut slice(type,dtype,freq)  extra"
        )
        for pname, r in c["paths"].items():
            if "error" in r:
                print(f"{pname:32s} ERROR {r['error']}")
                continue
            sl = r["first_slice"]
            sls = f"{sl['type']},{sl['dtype']},{sl['freq']}" if sl else "-"
            print(
                f"{pname:32s} {r['index_type'][:10]:10s} {str(r['names_ok'])[0]:5s} "
                f"{str(r['level_dtypes_ok'])[0]:6s} {str(r['row_order_ok'])[0]:4s} "
                f"{str(r['values_ok'])[0]:5s} {str(r['frame_equals'])[0]:4s} "
                f"{str(r['time_level_attr_freq'][1]):10s} "
                f"{str(r['xs_first_instance_freq'][1]):9s} {str(r['input_mutated'])[0]:3s} {sls:24s} "
                + ("" if r["level_dtypes_ok"] else f"dt={r['level_dtypes']} ")
                + ("" if r["names_ok"] else f"names={r['names']} ")
                + (str(r["extra"])[:90] if r["extra"] else "")
            )
    print("\n### narwhals dtype probe (index-derived columns)")
    for cname, rec in results["dtype_probe"].items():
        print(cname, json.dumps(rec))
    print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
