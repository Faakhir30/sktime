"""verify0: independent timeit check of r1 claim 7 -- is from_native the only narwhals per-call
cost, or do ops on PRE-WRAPPED frames also carry large overhead? Uses timeit (not common.bench).
Run: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore <this> --grid dev"""
import argparse, json, timeit, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, polars as pl
import narwhals.stable.v2 as nw
ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="dev", choices=["dev", "large"])
n = 100 if ap.parse_args().grid == "dev" else 500
idx = pd.date_range("2000-01-01", periods=n, freq="D")
pdf = pd.DataFrame({"c0": np.random.default_rng(0).normal(size=n)}, index=idx)
pdf_flat = pdf.reset_index(drop=True)
plf = pl.from_pandas(pdf_flat)
def t(stmt, g, number=3000):
    return min(timeit.repeat(stmt, globals=g, number=number, repeat=7)) / number * 1e6
out = {}
for name, df in [("pandas_dtidx", pdf), ("pandas_range", pdf_flat), ("polars", plf)]:
    w = nw.from_native(df)
    g = dict(df=df, w=w, nw=nw, pl=pl)
    native_mean = "df['c0'].mean()" if name.startswith("pandas") else "df['c0'].mean()"
    r = {
        "native_len": t("len(df)", g),
        "native_mean": t(native_mean, g),
        "from_native": t("nw.from_native(df)", g),
        "prewrapped_len": t("len(w)", g),
        "prewrapped_getitem_mean": t("w['c0'].mean()", g),
        "prewrapped_select_expr_mean": t("w.select(nw.col('c0').mean())", g),
        "prewrapped_get_column_mean": t("w.get_column('c0').mean()", g),
    }
    r["prewrapped_mean_overhead_us"] = r["prewrapped_getitem_mean"] - r["native_mean"]
    r["prewrapped_mean_ratio"] = r["prewrapped_getitem_mean"] / r["native_mean"]
    out[name] = {k: round(v, 2) for k, v in r.items()}
    print(name, json.dumps(out[name]))
