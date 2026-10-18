"""x9: does sktime accept a polars LazyFrame as forecasting ``y`` today?

Checks
  1. which registered mtypes (per scitype) have a lazy / out-of-core python type
  2. check_is_scitype / check_is_mtype on a LazyFrame in the sktime ``__index__``
     convention, for Series, Panel and Hierarchical shaped data
  3. convert_to from a LazyFrame (the ("pl.LazyFrame", ..., "Series") converters
     present in sktime/datatypes/_series/_convert.py)
  4. NaiveForecaster().fit(lazy_y) for Series and Panel shaped lazy frames

Run (from WT root)::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x9_lazy_mtype_probe.py
"""

import json
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from common import env_info, make_panel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")


def short_exc(e):
    tb = traceback.extract_tb(e.__traceback__)
    last = tb[-1] if tb else None
    where = f"{os.path.relpath(last.filename)}:{last.lineno}" if last else ""
    msg = str(e).replace("\n", " ")[:300]
    return f"{type(e).__name__}: {msg} @ {where}"


def main():
    from sktime.datatypes import check_is_mtype, check_is_scitype, convert_to
    from sktime.datatypes._registry import generate_mtype_cls_list
    from sktime.forecasting.naive import NaiveForecaster

    out = {"env": env_info(), "python": sys.version.split()[0]}

    # 1. registered mtypes and their python types
    reg = []
    for cls in generate_mtype_cls_list(soft_deps="present"):
        tags = cls.get_class_tags()
        reg.append(
            {
                "scitype": tags.get("scitype"),
                "name": tags.get("name"),
                "python_type": tags.get("python_type"),
            }
        )
    lazy_like = [
        r
        for r in reg
        if any(
            k in str(r["python_type"]).lower()
            for k in ["lazy", "dask", "duckdb", "ibis", "spark", "arrow"]
        )
    ]
    out["n_mtypes_present"] = len(reg)
    out["lazy_or_out_of_core_mtypes"] = lazy_like
    out["polars_mtypes"] = [r for r in reg if "polars" in str(r["python_type"])]
    print("lazy / out-of-core / arrow mtypes registered:")
    for r in lazy_like:
        print("  ", r)
    print("polars mtypes registered:")
    for r in out["polars_mtypes"]:
        print("  ", r)

    # 2. lazy frames in __index__ convention
    y_ser = make_panel(1, 30, n_levels=1).droplevel(0)  # single series
    y_ser.index.name = "time"
    y_pan = make_panel(4, 30, n_levels=1)
    y_hier = make_panel(8, 30, n_levels=2)
    cases = {
        "Series": convert_to(y_ser, "pl.DataFrame"),
        "Panel": convert_to(y_pan, "polars_panel"),
        "Hierarchical": convert_to(y_hier, "polars_hierarchical"),
    }
    checks = {}
    for scit, eager in cases.items():
        lazy = eager.lazy()
        rec = {"eager_columns": eager.columns}
        for label, obj in [("eager", eager), ("lazy", lazy)]:
            try:
                # forecasting y scitypes (as in BaseForecaster._check_X_y)
                ok, msg, meta = check_is_scitype(
                    obj,
                    scitype=["Series", "Panel", "Hierarchical"],
                    return_metadata=["mtype"],
                )
                rec[f"{label}_check_is_scitype[forecasting]"] = {
                    "valid": bool(ok),
                    "mtype": meta.get("mtype") if isinstance(meta, dict) else None,
                    "msg": None if ok else str(msg)[:300],
                }
            except Exception as e:  # noqa: BLE001
                rec[f"{label}_check_is_scitype[forecasting]"] = {
                    "error": short_exc(e)
                }
            try:
                ok, msg, meta = check_is_scitype(
                    obj, scitype="Table", return_metadata=["mtype"]
                )
                rec[f"{label}_check_is_scitype[Table]"] = {
                    "valid": bool(ok),
                    "mtype": meta.get("mtype") if isinstance(meta, dict) else None,
                }
            except Exception as e:  # noqa: BLE001
                rec[f"{label}_check_is_scitype[Table]"] = {"error": short_exc(e)}
        for mt in ["pl.DataFrame", "polars_panel", "polars_hierarchical",
                   "polars_lazy_table", "polars_eager_table"]:
            try:
                rec[f"lazy_is_mtype[{mt}]"] = bool(check_is_mtype(lazy, mt))
            except Exception as e:  # noqa: BLE001
                rec[f"lazy_is_mtype[{mt}]"] = short_exc(e)
        # 3. convert_to from lazy
        for target in ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]:
            try:
                res = convert_to(lazy, target)
                rec[f"convert_to({target})"] = f"OK -> {type(res).__name__}"
            except Exception as e:  # noqa: BLE001
                rec[f"convert_to({target})"] = short_exc(e)
        # 4. forecaster fit on lazy
        for label, obj in [("eager", eager), ("lazy", lazy)]:
            try:
                f = NaiveForecaster(strategy="last").fit(obj, fh=[1, 2])
                pred = f.predict()
                rec[f"NaiveForecaster.fit({label})"] = (
                    f"OK -> predict type {type(pred).__module__}."
                    f"{type(pred).__name__}"
                )
            except Exception as e:  # noqa: BLE001
                rec[f"NaiveForecaster.fit({label})"] = short_exc(e)
        checks[scit] = rec
        print(f"\n== {scit} ==")
        for k, v in rec.items():
            print(f"  {k}: {v}")
    out["checks"] = checks

    # the converter table entries that mention LazyFrame
    from sktime.datatypes._convert import convert_dict as cd

    lazy_keys = [list(k) for k in cd if any("Lazy" in str(x) or "lazy" in str(x)
                                            for x in k)]
    out["converter_keys_mentioning_lazy"] = lazy_keys
    print("\nconverter keys mentioning lazy:", lazy_keys)
    registered_names = {r["name"] for r in reg}
    out["lazy_converter_keys_with_unregistered_mtype"] = [
        k for k in lazy_keys if k[0] not in registered_names
        or k[1] not in registered_names
    ]
    print(
        "lazy converter keys with an UNREGISTERED mtype:",
        out["lazy_converter_keys_with_unregistered_mtype"],
    )

    tag = "pd3" if pd.__version__.startswith("3") else "pd2"
    path = os.path.join(RES, f"x9_lazy_mtype_probe_{tag}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
