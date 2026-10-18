"""x5: probe which narwhals operations needed by panel forecasters work per backend.

For every (operation, backend) pair, run the operation through narwhals,
materialize to pandas, and compare with the polars-eager result.

Output: results/x5_probe.json and a markdown table on stdout.

Run:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x5_probe_narwhals.py
"""

import json
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import narwhals as nw
import numpy as np
import pandas as pd

from common import env_info, make_panel
from protos.x5_backends import BACKENDS, LAZY_BACKENDS, long_from_mi, make_native

ED = Path(__file__).resolve().parents[1]
K = ["h0"]
T = "time"
V = "c0"


def steps_frame(f, H):
    """Small frame with step column k=1..H, on the same backend as ``f``."""
    impl = f.implementation
    data = {"k": list(range(1, H + 1))}
    if isinstance(f, nw.DataFrame):
        return nw.from_dict(data, backend=impl)
    # lazy: build eagerly then move to the lazy backend
    eager = nw.from_dict(data, backend="pyarrow")
    if impl.is_polars():
        return nw.from_dict(data, backend="polars").lazy()
    return eager.lazy(backend=impl)


def r_desc(f):
    """Descending time rank per series (1 = last observation)."""
    return f.with_columns(
        nw.col(T).rank("ordinal", descending=True).over(*K).alias("r")
    )


OPS = {
    # order-independent aggregations
    "group_by.agg(mean)": lambda f: f.group_by(*K).agg(nw.col(V).mean()),
    "mean().over(keys)": lambda f: f.with_columns(m=nw.col(V).mean().over(*K)),
    "filter(time==max over keys)": lambda f: f.filter(
        nw.col(T) == nw.col(T).max().over(*K)
    ),
    "rank(ordinal,desc).over(keys)": r_desc,
    # order-dependent
    "group_by.agg(last) [no order]": lambda f: f.group_by(*K).agg(nw.col(V).last()),
    "sort+group_by.agg(last)": lambda f: f.sort(*K, T).group_by(*K).agg(
        nw.col(V).last()
    ),
    "last().over(keys,order_by)": lambda f: f.with_columns(
        l=nw.col(V).last().over(*K, order_by=T)
    ),
    "shift(1).over(keys,order_by)": lambda f: f.with_columns(
        s=nw.col(V).shift(1).over(*K, order_by=T)
    ),
    "diff().over(keys,order_by)": lambda f: f.with_columns(
        d=nw.col(V).diff().over(*K, order_by=T)
    ),
    "cum_count().over(keys,order_by)": lambda f: f.with_columns(
        n=nw.col(V).cum_count().over(*K, order_by=T)
    ),
    "cum_sum().over(keys,order_by)": lambda f: f.with_columns(
        n=nw.col(V).cum_sum().over(*K, order_by=T)
    ),
    "rolling_mean(3).over(keys,order_by)": lambda f: f.with_columns(
        rm=nw.col(V).rolling_mean(3).over(*K, order_by=T)
    ),
    "ewm_mean(alpha,adjust=False).over(keys,order_by)": lambda f: f.with_columns(
        e=nw.col(V).ewm_mean(alpha=0.3, adjust=False).over(*K, order_by=T)
    ),
    "ewm_mean(alpha,adjust=False).over(keys) [no order_by]": lambda f: f.sort(
        *K, T
    ).with_columns(e=nw.col(V).ewm_mean(alpha=0.3, adjust=False).over(*K)),
    "unique(keys, order_by=time, keep=first)": lambda f: f.unique(
        K, keep="first", order_by=T
    ).select(*K)
    if isinstance(f, nw.LazyFrame)
    else f.unique(K, keep="first", maintain_order=True).select(*K),
    "with_row_index(order_by)": lambda f: f.with_row_index("ri", order_by=[*K, T]),
    # complex aggregations
    "group_by.agg(when(r==1).then(v).max())": lambda f: r_desc(f)
    .group_by(*K)
    .agg(nw.when(nw.col("r") == 1).then(nw.col(V)).otherwise(None).max()),
    "group_by.agg(col.filter(r==1).max())": lambda f: r_desc(f)
    .group_by(*K)
    .agg(nw.col(V).filter(nw.col("r") == 1).max()),
    "int modulo (r-1)%sp": lambda f: r_desc(f).with_columns(
        c=(nw.col("r") - 1) % 7
    ),
    # time arithmetic for future timestamps
    "dt.offset_by('1d')": lambda f: f.with_columns(t2=nw.col(T).dt.offset_by("1d")),
    "dt.offset_by('1mo')": lambda f: f.with_columns(
        t2=nw.col(T).dt.offset_by("1mo")
    ),
    "time + lit(timedelta)": lambda f: f.with_columns(
        t2=nw.col(T) + nw.lit(timedelta(days=1))
    ),
    "time + lit(timedelta) * int col": lambda f: r_desc(f).with_columns(
        t2=nw.col(T) + nw.lit(timedelta(days=1)) * nw.col("r")
    ),
    "(time - time).dt.total_seconds()": lambda f: f.with_columns(
        d=(nw.col(T) - nw.col(T).min().over(*K)).dt.total_seconds()
    ),
    "cross join with steps frame": lambda f: f.filter(
        nw.col(T) == nw.col(T).max().over(*K)
    ).join(steps_frame(f, 3), how="cross"),
    "left join on keys": lambda f: f.join(
        f.group_by(*K).agg(nw.col(V).mean().alias("m")), on=K, how="left"
    ),
    "concat vertical (lazy/eager)": lambda f: nw.concat([f, f], how="vertical"),
    "concat horizontal": lambda f: nw.concat(
        [f.select(*K), f.select(V)], how="horizontal"
    ),
    "group_by iteration (python)": lambda f: [g for _, g in f.group_by(*K)][0],
}

# ops whose result order is not defined: compare after sorting
SORT_COLS = None


def _canon(pdf):
    pdf = pdf.copy()
    cols = list(pdf.columns)
    if "ri" in cols:
        pdf = pdf.drop(columns="ri")
    by = [c for c in [*K, T, "k"] if c in pdf.columns]
    if by:
        pdf = pdf.sort_values(by, kind="stable")
    pdf = pdf.reset_index(drop=True)
    for c in pdf.columns:
        if pd.api.types.is_datetime64_any_dtype(pdf[c]):
            pdf[c] = pd.to_datetime(pdf[c]).dt.tz_localize(None).astype("datetime64[ns]")
        elif pd.api.types.is_numeric_dtype(pdf[c]):
            pdf[c] = pdf[c].astype("float64")
        else:
            pdf[c] = pdf[c].astype(str)
    return pdf


def materialize(res):
    if isinstance(res, nw.LazyFrame):
        res = res.collect()
    return res.to_pandas()


def main():
    df_mi = make_panel(n_series=5, n_time=12, n_cols=1, n_levels=1)
    df_long = long_from_mi(df_mi)
    natives = {b: make_native(df_long, b) for b in BACKENDS}

    out = {"env": env_info(), "results": {}}
    for op, fn in OPS.items():
        ref = None
        try:
            ref = _canon(materialize(fn(nw.from_native(natives["polars"]))))
        except Exception as e:  # noqa: BLE001
            ref = None
            print("REF FAILED", op, repr(e)[:200], file=sys.stderr)
        row = {}
        for b in BACKENDS:
            f = nw.from_native(natives[b])
            try:
                got = _canon(materialize(fn(f)))
                if ref is None:
                    row[b] = "ok (no ref)"
                else:
                    try:
                        pd.testing.assert_frame_equal(
                            got, ref, check_dtype=False, check_exact=False, rtol=1e-9
                        )
                        row[b] = "ok"
                    except AssertionError as ae:
                        row[b] = "MISMATCH: " + str(ae).replace("\n", " ")[:160]
            except Exception as e:  # noqa: BLE001
                row[b] = f"ERR {type(e).__name__}: " + str(e).split("\n")[0][:140]
        out["results"][op] = row

    # print table
    hdr = "| op | " + " | ".join(BACKENDS) + " |"
    print(hdr)
    print("|" + "---|" * (len(BACKENDS) + 1))
    for op, row in out["results"].items():
        cells = []
        for b in BACKENDS:
            v = row[b]
            cells.append("ok" if v == "ok" else ("MISMATCH" if v.startswith("MIS") else ("ERR" if v.startswith("ERR") else v)))
        print(f"| {op} | " + " | ".join(cells) + " |")
    print()
    for op, row in out["results"].items():
        for b, v in row.items():
            if v != "ok":
                print(f"- {op} / {b}: {v}")

    p = ED / "results" / "x5_probe.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
