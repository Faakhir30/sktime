"""a3 experiment (c): can fh.to_absolute be expressed without pandas?

Ground truth: sktime ForecastingHorizon(1..H).to_absolute(cutoff) for NC
consecutive cutoffs per pandas frequency.  Candidates: rules expressed with
polars expressions (vectorised over all (cutoff, k) pairs), numpy datetime64 /
busday functions, duckdb SQL interval arithmetic, and pyarrow compute.
Reports the number of mismatching (cutoff, k) pairs per rule; "NA" means the
library has no expression for the rule.

Also checks sktime's answer against pandas' own date_range continuation.

Output: results/a3_offset_equiv_<pdmajor>.json + printed table.

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/a3_offset_equiv.py
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import polars as pl

from common import env_info

warnings.filterwarnings("ignore")
PD3 = int(pd.__version__.split(".")[0]) >= 3
H = 24
NC = 150
TZ = "Europe/Berlin"

# name -> (kind, pandas freq, start, tz)
FREQS = {
    "D": ("dt", "D", "2000-01-01", None),
    "h": ("dt", "h", "2000-01-01", None),
    "2h": ("dt", "2h", "2000-01-01", None),
    "15min": ("dt", "15min", "2000-01-01", None),
    "B": ("dt", "B", "2000-01-03", None),
    "W-SUN": ("dt", "W-SUN", "2000-01-02", None),
    "W-WED": ("dt", "W-WED", "2000-01-05", None),
    "MS": ("dt", "MS", "2000-01-01", None),
    "ME": ("dt", "ME", "2000-01-31", None),
    "2ME": ("dt", "2ME", "2000-01-31", None),
    "QS": ("dt", "QS", "2000-01-01", None),
    "QE": ("dt", "QE", "2000-03-31", None),
    "YS": ("dt", "YS", "2000-01-01", None),
    "YE": ("dt", "YE", "2000-12-31", None),
    "SME": ("dt", "SME", "2000-01-15", None),
    "BMS": ("dt", "BMS", "2000-01-03", None),
    "BME": ("dt", "BME", "2000-01-31", None),
    "D tz=Berlin (DST)": ("dt", "D", "2000-03-01 12:00", TZ),
    "h tz=Berlin (DST)": ("dt", "h", "2000-03-25 12:00", TZ),
    "Period M": ("period", "M", "2000-01", None),
    "Period Q": ("period", "Q", "2000Q1", None),
    "Period W-SUN": ("period", "W-SUN", "2000-01-03", None),
    "Period D": ("period", "D", "2000-01-01", None),
}


def ground_truth(kind, freq, start, tz):
    from sktime.forecasting.base import ForecastingHorizon

    if kind == "dt":
        cut = pd.date_range(start, periods=NC, freq=freq, tz=tz)
    else:
        cut = pd.period_range(start, periods=NC, freq=freq)
    sk_rows, dr_rows = [], []
    for i in range(NC):
        c = cut[[i]]
        if kind == "dt":
            c = pd.DatetimeIndex(c, freq=freq)
        fh = ForecastingHorizon(list(range(1, H + 1)))
        a = fh.to_absolute(c).to_pandas()
        if kind == "period":
            a = a.to_timestamp()  # compare on period START timestamps
        sk_rows.append(np.asarray(a.tz_localize(None) if tz else a, dtype="M8[ns]"))
        if kind == "dt":
            dr = pd.date_range(cut[i], periods=H + 1, freq=freq)[1:]
        else:
            dr = pd.period_range(cut[i], periods=H + 1, freq=freq)[1:].to_timestamp()
        dr_rows.append(np.asarray(dr.tz_localize(None) if tz else dr, dtype="M8[ns]"))
    if kind == "period":
        cut_ts = cut.to_timestamp()
    else:
        cut_ts = cut
    return cut_ts, np.stack(sk_rows), np.stack(dr_rows)


# ---------------------------------------------------------------- polars rules
def _pl_frame(cut_ts, tz):
    c = pl.Series("c", np.asarray(cut_ts.tz_localize(None) if tz else cut_ts,
                                  dtype="M8[us]"))
    if tz:
        c = c.dt.replace_time_zone("UTC").dt.convert_time_zone(tz) if False else (
            pl.Series("c", cut_ts.tz_convert("UTC").tz_localize(None).values)
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone(tz)
        )
    k = pl.Series("k", np.arange(1, H + 1), dtype=pl.Int64)
    return c.to_frame().join(k.to_frame(), how="cross")


def _mo(n):
    return pl.format("{}mo", pl.col("k") * n)


def _d(n):
    return pl.format("{}d", pl.col("k") * n)


C = pl.col("c")
POLARS_RULES = {
    "D": {"offset_by(kd)": C.dt.offset_by(_d(1))},
    "h": {"+duration(h)": C + pl.duration(hours=pl.col("k"))},
    "2h": {"+duration(2h)": C + pl.duration(hours=pl.col("k") * 2)},
    "15min": {"+duration(15m)": C + pl.duration(minutes=pl.col("k") * 15)},
    "B": {
        "add_business_days(k)": C.dt.add_business_days(pl.col("k")),
        "NAIVE offset_by(kd)": C.dt.offset_by(_d(1)),
    },
    "W-SUN": {"offset_by(7k d)": C.dt.offset_by(_d(7))},
    "W-WED": {"offset_by(7k d)": C.dt.offset_by(_d(7))},
    "MS": {"offset_by(k mo)": C.dt.offset_by(_mo(1))},
    "ME": {
        "offset_by(k mo).month_end()": C.dt.offset_by(_mo(1)).dt.month_end(),
        "NAIVE offset_by(k mo)": C.dt.offset_by(_mo(1)),
    },
    "2ME": {"offset_by(2k mo).month_end()": C.dt.offset_by(_mo(2)).dt.month_end()},
    "QS": {"offset_by(3k mo)": C.dt.offset_by(_mo(3))},
    "QE": {
        "offset_by(3k mo).month_end()": C.dt.offset_by(_mo(3)).dt.month_end(),
        "NAIVE offset_by(k q)": C.dt.offset_by(pl.format("{}q", pl.col("k"))),
    },
    "YS": {"offset_by(k y)": C.dt.offset_by(pl.format("{}y", pl.col("k")))},
    "YE": {
        "offset_by(k y).month_end()": C.dt.offset_by(
            pl.format("{}y", pl.col("k"))
        ).dt.month_end()
    },
    "SME": {
        # half-month ordinal: 2*(12*y+m-1) + (day>15) ; add k ; back
        "half-month ordinal": None,  # built below
    },
    "BMS": {
        "month_start+k mo, roll fwd bday": C.dt.month_start()
        .dt.offset_by(_mo(1))
        .dt.add_business_days(0, roll="forward"),
    },
    "BME": {
        "month_start+k mo, month_end, roll back bday": C.dt.month_start()
        .dt.offset_by(_mo(1))
        .dt.month_end()
        .dt.add_business_days(0, roll="backward"),
    },
    "D tz=Berlin (DST)": {
        "offset_by(kd) [calendar day]": C.dt.offset_by(_d(1)),
        "+duration(24k h) [absolute]": C + pl.duration(hours=pl.col("k") * 24),
    },
    "h tz=Berlin (DST)": {"+duration(h)": C + pl.duration(hours=pl.col("k"))},
    "Period M": {"start.offset_by(k mo)": C.dt.offset_by(_mo(1))},
    "Period Q": {"start.offset_by(3k mo)": C.dt.offset_by(_mo(3))},
    "Period W-SUN": {"start.offset_by(7k d)": C.dt.offset_by(_d(7))},
    "Period D": {"start.offset_by(kd)": C.dt.offset_by(_d(1))},
}


def _sme_expr():
    o = (C.dt.year() * 12 + C.dt.month() - 1) * 2 + (C.dt.day() > 15).cast(pl.Int32)
    o2 = o + pl.col("k")
    m = o2 // 2
    first = pl.datetime(m // 12, m % 12 + 1, 1)
    return pl.when(o2 % 2 == 0).then(first.dt.offset_by("14d")).otherwise(
        first.dt.month_end()
    )


POLARS_RULES["SME"]["half-month ordinal"] = _sme_expr()


def eval_polars(cut_ts, tz, expr):
    df = _pl_frame(cut_ts, tz).with_columns(out=expr)
    out = df["out"]
    if tz:
        out = out.dt.replace_time_zone(None)
    return out.cast(pl.Datetime("ns")).to_numpy().reshape(len(cut_ts), H)


# ----------------------------------------------------------------- numpy rules
def np_rules(name):
    k = np.arange(1, H + 1)[None, :]
    D = np.timedelta64(1, "D")

    def tod(c):  # time of day part
        return c - c.astype("M8[D]")

    r = {}
    if name in ("D", "Period D"):
        r["+k days"] = lambda c: c[:, None] + k * D
    elif name == "h":
        r["+k h"] = lambda c: c[:, None] + k * np.timedelta64(1, "h")
    elif name == "2h":
        r["+2k h"] = lambda c: c[:, None] + 2 * k * np.timedelta64(1, "h")
    elif name == "15min":
        r["+15k m"] = lambda c: c[:, None] + 15 * k * np.timedelta64(1, "m")
    elif name == "B":
        r["busday_offset"] = lambda c: np.busday_offset(
            c.astype("M8[D]")[:, None], k, roll="forward"
        ).astype("M8[ns]") + tod(c)[:, None]
    elif name in ("W-SUN", "W-WED", "Period W-SUN"):
        r["+7k days"] = lambda c: c[:, None] + 7 * k * D
    elif name in ("MS", "Period M"):
        r["M8[M]+k"] = lambda c: (c.astype("M8[M]")[:, None] + k).astype("M8[ns]")
    elif name == "ME":
        r["(M8[M]+k+1)->D -1"] = lambda c: (
            (c.astype("M8[M]")[:, None] + k + 1).astype("M8[D]") - D
        ).astype("M8[ns]")
    elif name == "2ME":
        r["(M8[M]+2k+1)->D -1"] = lambda c: (
            (c.astype("M8[M]")[:, None] + 2 * k + 1).astype("M8[D]") - D
        ).astype("M8[ns]")
    elif name in ("QS", "Period Q"):
        r["M8[M]+3k"] = lambda c: (c.astype("M8[M]")[:, None] + 3 * k).astype("M8[ns]")
    elif name == "QE":
        r["(M8[M]+3k+1)->D -1"] = lambda c: (
            (c.astype("M8[M]")[:, None] + 3 * k + 1).astype("M8[D]") - D
        ).astype("M8[ns]")
    elif name == "YS":
        r["M8[Y]+k"] = lambda c: (c.astype("M8[Y]")[:, None] + k).astype("M8[ns]")
    elif name == "YE":
        r["(M8[Y]+k+1)->D -1"] = lambda c: (
            (c.astype("M8[Y]")[:, None] + k + 1).astype("M8[D]") - D
        ).astype("M8[ns]")
    elif name == "BMS":
        r["busday_offset(M8[M]+k, roll fwd)"] = lambda c: np.busday_offset(
            (c.astype("M8[M]")[:, None] + k).astype("M8[D]"), 0, roll="forward"
        ).astype("M8[ns]")
    elif name == "BME":
        r["busday_offset(month_end, roll back)"] = lambda c: np.busday_offset(
            (c.astype("M8[M]")[:, None] + k + 1).astype("M8[D]") - D, 0,
            roll="backward",
        ).astype("M8[ns]")
    elif name == "SME":
        def sme(c):
            m = c.astype("M8[M]")
            day = (c.astype("M8[D]") - m.astype("M8[D]")).astype(int) + 1
            o = m.astype(int) * 2 + (day > 15)
            o2 = o[:, None] + k
            mm = (o2 // 2).astype("M8[M]")
            first = mm.astype("M8[D]")
            end = (mm + 1).astype("M8[D]") - D
            return np.where(o2 % 2 == 0, first + 14 * D, end).astype("M8[ns]")

        r["half-month ordinal"] = sme
    elif name == "h tz=Berlin (DST)":
        r["UTC + k h (no tz in numpy)"] = None  # handled specially
    elif name == "D tz=Berlin (DST)":
        r["UTC + 24k h (no tz in numpy)"] = None
    return r


def eval_numpy(name, cut_ts, tz, fn):
    if tz:
        utc = np.asarray(cut_ts.tz_convert("UTC").tz_localize(None), dtype="M8[ns]")
        step = np.timedelta64(1, "h") if name.startswith("h") else np.timedelta64(
            24, "h"
        )
        out_utc = utc[:, None] + np.arange(1, H + 1)[None, :] * step
        out = pd.DatetimeIndex(out_utc.ravel()).tz_localize("UTC").tz_convert(tz)
        return np.asarray(out.tz_localize(None), dtype="M8[ns]").reshape(-1, H)
    return fn(np.asarray(cut_ts, dtype="M8[ns]"))


# ---------------------------------------------------------------- duckdb rules
DUCK = {
    "D": "c + to_days(k::INT)",
    "Period D": "c + to_days(k::INT)",
    "h": "c + to_hours(k)",
    "2h": "c + to_hours(2*k)",
    "15min": "c + to_minutes(15*k)",
    "W-SUN": "c + to_days((7*k)::INT)",
    "W-WED": "c + to_days((7*k)::INT)",
    "Period W-SUN": "c + to_days((7*k)::INT)",
    "MS": "c + to_months(k::INT)",
    "Period M": "c + to_months(k::INT)",
    "ME": "last_day(c + to_months(k::INT))::TIMESTAMP",
    "2ME": "last_day(c + to_months((2*k)::INT))::TIMESTAMP",
    "QS": "c + to_months((3*k)::INT)",
    "Period Q": "c + to_months((3*k)::INT)",
    "QE": "last_day(c + to_months((3*k)::INT))::TIMESTAMP",
    "YS": "c + to_years(k::INT)",
    "YE": "last_day(c + to_years(k::INT))::TIMESTAMP",
}


def eval_duckdb(name, cut_ts, tz):
    import duckdb
    import pyarrow as pa

    expr = DUCK[name]
    tbl = pa.table({"c": pa.array(np.asarray(cut_ts, dtype="M8[us]"))})
    ks = pa.table({"k": pa.array(np.arange(1, H + 1, dtype="int64"))})
    con = duckdb.connect()
    con.register("cut", tbl)
    con.register("ks", ks)
    res = con.execute(
        f"SELECT row_number() OVER () AS rn, c, k, ({expr})::TIMESTAMP AS out "
        "FROM cut CROSS JOIN ks ORDER BY c, k"
    ).arrow()
    if hasattr(res, "read_all"):
        res = res.read_all()
    con.close()
    out = np.asarray(res.column("out").to_numpy(), dtype="M8[ns]")
    return out.reshape(len(cut_ts), H)


def eval_pyarrow(name, cut_ts, tz):
    import pyarrow as pa
    import pyarrow.compute as pc

    fixed = {"D": 24 * 3600, "Period D": 24 * 3600, "h": 3600, "2h": 7200,
             "15min": 900, "W-SUN": 7 * 86400, "W-WED": 7 * 86400,
             "Period W-SUN": 7 * 86400}
    if name in fixed:
        c = pa.array(np.repeat(np.asarray(cut_ts, dtype="M8[ns]"), H))
        k = np.tile(np.arange(1, H + 1), len(cut_ts)) * fixed[name] * 10**9
        out = pc.add(c, pa.array(k.astype("m8[ns]")))
        return np.asarray(out.to_numpy(zero_copy_only=False), dtype="M8[ns]").reshape(
            len(cut_ts), H
        )
    # calendar units: try month_day_nano interval add (known to be unsupported)
    c = pa.array(np.asarray(cut_ts, dtype="M8[ns]"))
    pc.add(c, pa.scalar((1, 0, 0), type=pa.month_day_nano_interval()))
    raise AssertionError("unexpected: arrow month add worked")


def mism(a, b):
    return int((np.asarray(a, dtype="M8[ns]") != np.asarray(b, dtype="M8[ns]")).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])  # unused
    ap.parse_args()
    res = {"env": env_info(), "H": H, "NC": NC, "rows": {}}
    total = NC * H
    print(f"pairs checked per rule: {total} (NC={NC} cutoffs x H={H})")
    for name, (kind, freq, start, tz) in FREQS.items():
        row = {}
        try:
            cut_ts, sk, dr = ground_truth(kind, freq, start, tz)
        except Exception as e:  # noqa: BLE001
            res["rows"][name] = {"ground_truth_error": f"{type(e).__name__}: {e}"}
            print(f"{name:20s} GROUND TRUTH ERROR {e}")
            continue
        row["sktime_vs_date_range_mismatch"] = mism(sk, dr)
        for rname, expr in POLARS_RULES.get(name, {}).items():
            try:
                row[f"polars: {rname}"] = mism(eval_polars(cut_ts, tz, expr), sk)
            except Exception as e:  # noqa: BLE001
                row[f"polars: {rname}"] = f"ERR {type(e).__name__}: {str(e)[:80]}"
        for rname, fn in np_rules(name).items():
            try:
                row[f"numpy: {rname}"] = mism(eval_numpy(name, cut_ts, tz, fn), sk)
            except Exception as e:  # noqa: BLE001
                row[f"numpy: {rname}"] = f"ERR {type(e).__name__}: {str(e)[:80]}"
        if name in DUCK and not tz:
            try:
                row[f"duckdb: {DUCK[name]}"] = mism(eval_duckdb(name, cut_ts, tz), sk)
            except Exception as e:  # noqa: BLE001
                row[f"duckdb: {DUCK[name]}"] = f"ERR {type(e).__name__}: {str(e)[:80]}"
        else:
            row["duckdb"] = "NA (no builtin rule tested)"
        if not tz:
            try:
                row["pyarrow: pc.add"] = mism(eval_pyarrow(name, cut_ts, tz), sk)
            except Exception as e:  # noqa: BLE001
                row["pyarrow: pc.add"] = f"NA {type(e).__name__}: {str(e)[:70]}"
        res["rows"][name] = row
        print(f"--- {name}  (sktime vs pd.date_range mismatches: "
              f"{row['sktime_vs_date_range_mismatch']}/{total})")
        for k, v in row.items():
            if k != "sktime_vs_date_range_mismatch":
                print(f"      {k:55s} {v}")
    tag = "pd3" if PD3 else "pd2"
    path = f"experiments/container_independence/results/a3_offset_equiv_{tag}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
