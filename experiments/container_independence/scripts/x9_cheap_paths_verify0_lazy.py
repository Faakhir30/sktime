"""x9 verifier (B): decompose the lazy-pushdown gain into
  (i)  "don't convert everything to pandas" (filter in polars first, eager read), and
  (ii) "push the filter into the parquet scan" (true laziness / pushdown),
and re-baseline "sktime today" after the cheap vectorized monotonic fix.

All variants return a pandas pd-multiindex frame of the same slice (column y, 1000
series, last 50 of 200 points), so output containers are symmetric.

  V_today      : read all -> convert_to (current sktime)                 -> select
  V_today_fix  : read all -> convert_to with VECTORIZED monotonic check  -> select
  V_adapter    : read all -> raw adapter (no check)                      -> select   (= B1c)
  E_filter     : eager read all -> polars filter -> raw adapter          (no laziness)
  E_proj       : eager read_parquet(columns=3) -> polars filter -> raw adapter
  L_push       : scan_parquet.filter.select.collect -> raw adapter        (= B5b)
  L_push_fix   : L_push but via convert_to with vectorized check (realistic boundary)

Time: min of repeats, interleaved rounds, same process. RSS: fresh subprocess per
variant (result kept alive). Writes its own parquet and deletes it.

Run::
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x9_cheap_paths_verify0_lazy.py --grid dev
"""
import argparse
import json
import os
import subprocess
import sys
import warnings

os.environ.setdefault("POLARS_MAX_THREADS", "4")
warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from common import bench, peak_rss_delta_mb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
DATA = os.path.join(RES, "x9_cheap_paths_verify0_data.parquet")
GRIDS = {"dev": (10_000, 200), "large": (50_000, 200)}
N_SEL, LAST = 1000, 50

import sktime.datatypes._adapter.polars as adp  # noqa: E402
from sktime.datatypes import convert_to  # noqa: E402
from sktime.datatypes._adapter.polars import convert_polars_to_pandas  # noqa: E402

ORIG = adp.is_monotonically_increasing


def is_mono_vec(obj):
    cols = [c for c in obj.columns if c.startswith("__index__")]
    if len(cols) <= 1:
        return obj[cols[0]].is_sorted() if cols else True
    t = pl.col(cols[-1])
    return bool(obj.select((t >= t.shift(1).over(cols[:-1])).fill_null(True).all()).item())


def write(ns, nt):
    rng = np.random.default_rng(3)
    sid = np.repeat(np.array([f"s{i:07d}" for i in range(ns)]), nt)
    t = np.tile(pd.date_range("2000-01-01", periods=nt, freq="D").values, ns)
    n = ns * nt
    pl.DataFrame({"__index__series_id": sid, "__index__time": t,
                  "y": rng.normal(size=n).cumsum(), "x1": rng.normal(size=n),
                  "x2": rng.normal(size=n)}).write_parquet(DATA, row_group_size=100_000)


def sel(ns, nt):
    rng = np.random.default_rng(7)
    ids = [f"s{i:07d}" for i in sorted(rng.choice(ns, N_SEL, replace=False))]
    cut = pd.Timestamp("2000-01-01") + pd.Timedelta(days=nt - LAST)
    return ids, cut


def _pd_select(y, ids):
    s = y[y.index.get_level_values(0).isin(ids)]
    return s.groupby(level=0, sort=False).tail(LAST)[["y"]]


def _flt(ids, cut):
    return pl.col("__index__series_id").is_in(ids) & (pl.col("__index__time") >= cut)


def V_today(ids, cut):
    return _pd_select(convert_to(pl.read_parquet(DATA), "pd-multiindex"), ids)


def V_today_fix(ids, cut):
    adp.is_monotonically_increasing = is_mono_vec
    try:
        return _pd_select(convert_to(pl.read_parquet(DATA), "pd-multiindex"), ids)
    finally:
        adp.is_monotonically_increasing = ORIG


def V_adapter(ids, cut):
    return _pd_select(convert_polars_to_pandas(pl.read_parquet(DATA)), ids)


COLS = ["__index__series_id", "__index__time", "y"]


def E_filter(ids, cut):
    return convert_polars_to_pandas(pl.read_parquet(DATA).filter(_flt(ids, cut)).select(COLS))


def E_proj(ids, cut):
    return convert_polars_to_pandas(pl.read_parquet(DATA, columns=COLS).filter(_flt(ids, cut)))


def L_push(ids, cut):
    return convert_polars_to_pandas(pl.scan_parquet(DATA).filter(_flt(ids, cut)).select(COLS).collect())


def L_push_fix(ids, cut):
    adp.is_monotonically_increasing = is_mono_vec
    try:
        df = pl.scan_parquet(DATA).filter(_flt(ids, cut)).select(COLS).collect()
        return convert_to(df, "pd-multiindex")
    finally:
        adp.is_monotonically_increasing = ORIG


V = {f.__name__: f for f in [V_today, V_today_fix, V_adapter, E_filter, E_proj, L_push, L_push_fix]}


def canon(df):
    df = df[["y"]].sort_index()
    return len(df), round(float(df["y"].sum()), 6), df.index[0], df.index[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev")
    ap.add_argument("--child", default=None)
    a = ap.parse_args()
    ns, nt = GRIDS[a.grid]
    ids, cut = sel(ns, nt)
    if a.child:
        _, d = peak_rss_delta_mb(lambda: V[a.child](ids, cut), interval=0.002)
        print(json.dumps({"rss": d}))
        sys.exit()
    write(ns, nt)
    try:
        c = {k: canon(f(ids, cut)) for k, f in V.items()}
        print("agree:", len({v for v in c.values()}) == 1, c["L_push"][:2])
        tt = {k: [] for k in V}
        for _ in range(2):
            for k, f in V.items():
                tt[k].append(bench(lambda f=f: f(ids, cut), repeat=2 if k == "V_today" else 4))
        t = {k: min(v) for k, v in tt.items()}
        out = {}
        for k in V:
            p = subprocess.run([sys.executable, "-W", "ignore", __file__, "--grid", a.grid,
                                "--child", k], capture_output=True, text=True, env=os.environ)
            rss = json.loads(p.stdout.strip().splitlines()[-1])["rss"]
            out[k] = {"ms": t[k] * 1e3, "rss": rss}
            print(f"  {k:<12} {t[k]*1e3:8.1f} ms  x{t['V_today']/t[k]:6.1f} vs today  "
                  f"x{t['L_push']/t[k]:5.2f} of L_push(time ratio inverse)  peakRSS {rss:6.1f} MB")
        print(f"pushdown-only gain (E_filter / L_push): time x{t['E_filter']/t['L_push']:.1f}, "
              f"rss x{out['E_filter']['rss']/max(out['L_push']['rss'],0.1):.1f}")
        print(f"after cheap fix (V_today_fix / L_push_fix): time x{t['V_today_fix']/t['L_push_fix']:.1f}, "
              f"rss x{out['V_today_fix']['rss']/max(out['L_push_fix']['rss'],0.1):.1f}")
        with open(os.path.join(RES, f"x9_cheap_paths_verify0_lazy_{a.grid}_pd{pd.__version__[0]}.json"), "w") as fh:
            json.dump(out, fh, indent=1)
    finally:
        os.remove(DATA)
