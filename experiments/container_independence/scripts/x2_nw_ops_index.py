"""x2 (1): what does narwhals do with a pandas MultiIndex?

For a pandas frame in sktime ``pd-multiindex`` layout (index h0, time), wrap it
with ``nw.from_native`` and run a set of narwhals operations. For each op record

- whether it raised / warned
- the type, nlevels and names of the index of ``.to_native()``
- whether the index still *aligns with the row data* (checked via a row-id
  column ``rid`` added before the op: result.index must equal
  original.index.take(result.rid))

Also probes "leaks": pandas resolves index level *names* in groupby / sort_values,
so narwhals ops that pass a string straight through to pandas may silently work
on index levels on the pandas backend only.

Run (from worktree root)::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x2_nw_ops_index.py
"""

import json
import os
import sys
import warnings

import narwhals as nw
import numpy as np
import pandas as pd

from common import env_info, make_panel

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def describe_index(native, orig):
    """Describe index of a native pandas result, check alignment with rid col."""
    out = {
        "result_type": type(native).__name__,
        "index_type": type(native.index).__name__,
        "nlevels": int(native.index.nlevels),
        "names": [None if n is None else str(n) for n in native.index.names],
    }
    if isinstance(native, pd.DataFrame) and "rid" in native.columns:
        rid = native["rid"].to_numpy()
        try:
            expected = orig.index.take(rid)
            out["index_aligned_with_rows"] = bool(native.index.equals(expected))
        except Exception as e:  # noqa: BLE001
            out["index_aligned_with_rows"] = f"n/a ({type(e).__name__})"
        out["row_order_preserved"] = bool(np.all(np.diff(rid) > 0))
    return out


def run_op(name, fn, orig):
    rec = {"op": name}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = fn()
            native = nw.to_native(res) if not isinstance(res, (pd.DataFrame, pd.Series)) else res
            rec.update(describe_index(native, orig))
            rec["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"raised {type(e).__name__}: {str(e).splitlines()[0][:160]}"
    rec["warnings"] = sorted({f"{x.category.__name__}: {str(x.message)[:120]}" for x in w})
    return rec


def main():
    df = make_panel(n_series=4, n_time=6, n_cols=2, n_levels=1)
    df["rid"] = np.arange(len(df))
    out = {"env": env_info(), "python": sys.version.split()[0]}
    out["pandas_cow"] = (
        True
        if pd.__version__ >= "3"
        else bool(getattr(pd.options.mode, "copy_on_write", False))
    )

    # from_native itself
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ndf = nw.from_native(df, eager_only=True)
    out["from_native"] = {
        "raised": False,
        "warnings": [str(x.message) for x in w],
        "nw_columns": list(ndf.columns),
        "nw_schema": {k: str(v) for k, v in ndf.schema.items()},
        "maybe_get_index_type": type(nw.maybe_get_index(ndf)).__name__,
        "maybe_get_index_names": list(nw.maybe_get_index(ndf).names),
        "to_native_is_same_object": nw.to_native(ndf) is df,
    }

    other = pd.DataFrame({"c0_key": df["c0"].to_numpy()[:10], "extra": np.arange(10)})
    nother = nw.from_native(other, eager_only=True)
    ndf_k = ndf.with_columns(c0_key=nw.col("c0"))
    # index level names that ALSO exist as columns (e.g. set_index(..., drop=False)):
    # narwhals' pandas group_by then calls native.reset_index(drop=True) internally
    # (narwhals/_pandas_like/group_by.py, issue #1907)
    df_both = df.reset_index().set_index(["h0", "time"], drop=False)
    ndf_both = nw.from_native(df_both, eager_only=True)

    ops = {
        # row/column preserving or row-subsetting ops
        "select('c0','rid')": lambda: ndf.select("c0", "rid"),
        "select(expr c0*2, rid)": lambda: ndf.select(nw.col("c0") * 2, "rid"),
        "with_columns(c2=c0*2)": lambda: ndf.with_columns(c2=nw.col("c0") * 2),
        "rename(c0->x)": lambda: ndf.rename({"c0": "x"}),
        "drop('c1')": lambda: ndf.drop("c1"),
        "filter(c0>0)": lambda: ndf.filter(nw.col("c0") > 0),
        "sort('c0')": lambda: ndf.sort("c0"),
        "head(5)": lambda: ndf.head(5),
        "tail(5)": lambda: ndf.tail(5),
        "df[2:7] (row slice)": lambda: ndf[2:7],
        "df[[0,3,5]] (gather)": lambda: ndf[[0, 3, 5]],
        "drop_nulls()": lambda: ndf.drop_nulls(),
        "unique() (all cols)": lambda: ndf.unique(),
        "with_row_index('r')": lambda: ndf.with_row_index("r"),
        "lazy().collect()": lambda: ndf.lazy().collect(),
        # reshaping / combining
        "concat([df, df]) vertical": lambda: nw.concat([ndf, ndf], how="vertical"),
        "concat horizontal": lambda: nw.concat(
            [ndf.select("c0", "rid"), ndf.select(c1b=nw.col("c1"))], how="horizontal"
        ),
        "join(other, on c0_key) inner": lambda: ndf_k.join(nother, on="c0_key"),
        "join(other, on c0_key) left": lambda: ndf_k.join(
            nother, on="c0_key", how="left"
        ),
        "unpivot(index=rid)": lambda: ndf.unpivot(index="rid"),
        # group-by
        "group_by(col).agg(mean)": lambda: ndf.with_columns(
            g=nw.col("rid") % 2
        )
        .group_by("g")
        .agg(nw.col("c0").mean()),
        "group_by(col) iter first group": lambda: next(
            iter(ndf.with_columns(g=nw.col("rid") % 2).group_by("g"))
        )[1],
        "over(): c0.mean().over(g)": lambda: ndf.with_columns(g=nw.col("rid") % 2)
        .with_columns(m=nw.col("c0").mean().over("g")),
        "OVERLAP idx names==cols: group_by('h0') iter first group": lambda: next(
            iter(ndf_both.group_by("h0"))
        )[1],
        "OVERLAP idx names==cols: filter(c0>0)": lambda: ndf_both.filter(
            nw.col("c0") > 0
        ),
        # index level used by NAME: works only because pandas resolves level names
        "LEAK group_by('h0') [index level name].agg": lambda: ndf.group_by("h0").agg(
            nw.col("c0").mean()
        ),
        "LEAK group_by('h0') iter first group": lambda: next(iter(ndf.group_by("h0")))[1],
        "LEAK sort('h0','time') [index level names]": lambda: ndf.sort("h0", "time"),
        "nw.col('h0') [index level name]": lambda: ndf.select(nw.col("h0")),
        "filter(nw.col('time')>...)": lambda: ndf.filter(
            nw.col("time") > pd.Timestamp("2000-01-03")
        ),
        # helpers
        "maybe_reset_index": lambda: nw.maybe_reset_index(ndf),
        "maybe_set_index(['rid'])": lambda: nw.maybe_set_index(ndf, ["rid"]),
    }
    recs = [run_op(k, f, df) for k, f in ops.items()]

    # Series with MultiIndex
    s = nw.from_native(df["c0"], series_only=True)
    srecs = [
        run_op("Series.filter(s>0)", lambda: s.filter(s > 0), df),
        run_op("Series.sort()", lambda: s.sort(), df),
        run_op("Series*2", lambda: s * 2, df),
        run_op("Series.cum_sum()", lambda: s.cum_sum(), df),
        run_op("Series.shift(1)", lambda: s.shift(1), df),
    ]
    # the LEAK group_by('h0') key + columns of the group
    k, g = next(iter(ndf.group_by("h0")))
    out["leak_group_by_index_level"] = {
        "key": repr(k),
        "group_nw_columns": list(g.columns),
        "polars_equivalent": None,
    }
    try:
        import polars as pl

        pdf = nw.from_native(pl.from_pandas(df.reset_index(drop=True)), eager_only=True)
        next(iter(pdf.group_by("h0")))
        out["leak_group_by_index_level"]["polars_equivalent"] = "ok?!"
    except Exception as e:  # noqa: BLE001
        out["leak_group_by_index_level"]["polars_equivalent"] = (
            f"raised {type(e).__name__}"
        )

    out["ops"] = recs
    out["series_ops"] = srecs

    tag = "pd" + pd.__version__.split(".")[0]
    path = os.path.join(RES, f"x2_nw_ops_index_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)

    # human-readable table
    print(f"pandas {pd.__version__}  narwhals {nw.__version__}  cow={out['pandas_cow']}")
    print("from_native:", json.dumps(out["from_native"], default=str))
    print("leak:", out["leak_group_by_index_level"])
    hdr = f"{'op':48s} {'status':10s} {'index':12s} {'names':22s} aligned  order"
    print(hdr)
    for r in recs + srecs:
        st = "ok" if r["status"] == "ok" else r["status"][:60]
        if r["status"] == "ok":
            print(
                f"{r['op']:48s} {st:10s} {r['index_type']:12s} "
                f"{str(r['names']):22s} {str(r.get('index_aligned_with_rows', '-')):8s}"
                f" {r.get('row_order_preserved', '-')}"
                + (f"  WARN {r['warnings']}" if r["warnings"] else "")
            )
        else:
            print(f"{r['op']:48s} {st}")
    print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
