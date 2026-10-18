"""a3 prototype: a minimal, container-neutral "TimeIndex" abstraction.

The forecasting framework needs only five operations on the time axis:

    cutoff(times)                 last time point (per series: group max)
    add_steps(cutoff, k)          cutoff shifted by k steps   (fh.to_absolute)
    steps_between(a, b)           integer number of steps a->b (fh.to_relative)
    range(start, n)               n consecutive time points    (index generation)
    infer_step(times)             step spec from observed time points (freq)

and one value object, ``Step``, that replaces the pandas offset / freq string as
the *semantic* description of the step.  ``Step`` is plain python (no pandas),
can be mapped to/from pandas offsets, polars interval strings, numpy units.

Backends implement the operations on their native containers:

* ``PandasTimeOps``  - delegates to pandas offsets (reference semantics)
* ``PolarsTimeOps``  - polars expressions (vectorised; Date/Datetime/Int)
* ``NumpyTimeOps``   - numpy datetime64 + busday functions (no tz support)

Step kinds
----------
int         integer index, step n                         (RangeIndex/Index[int])
fixed       absolute duration (nanos), e.g. h, 15min, D (naive), W-xxx
calday      calendar day(s) in a time zone (D on tz-aware data, pandas>=3)
month       n calendar months, anchor "start" | "end"       (MS/ME/QS/QE/YS/YE)
bday        n business days, weekmask/holidays             (B, C)
bmonth      n months, anchor start|end, rolled to business day (BMS/BME)
halfmonth   semi-month: 15th + month end                   (SME)
period      like month/fixed but values are pandas Periods -> stored as ordinals

Not representable without a calendar engine: pandas custom holiday calendars
(CustomBusinessDay with calendar objects) beyond a static holiday list,
fiscal-year offsets (FY5253), Easter, business hours (BH).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NS = {"ns": 1, "us": 10**3, "ms": 10**6, "s": 10**9, "m": 60 * 10**9,
      "h": 3600 * 10**9, "d": 86400 * 10**9, "w": 7 * 86400 * 10**9}


@dataclass(frozen=True)
class Step:
    """Library-neutral description of a time step."""

    kind: str  # int|fixed|calday|month|bday|bmonth|halfmonth
    n: int = 1  # multiplier: int step, n months, n business days, n days
    nanos: int = 0  # for kind == "fixed"
    anchor: str | None = None  # "start" | "end" for month/bmonth
    weekmask: str = "1111100"
    holidays: tuple = field(default_factory=tuple)
    tz: str | None = None
    week_anchor: int | None = None  # 0=Mon..6=Sun for W-xxx (informational)

    # ------------------------------------------------------------ pandas map
    @classmethod
    def from_pandas(cls, freq, tz=None):
        """Map a pandas offset / freq string to a Step (None if unsupported)."""
        import pandas as pd
        from pandas.tseries import offsets as o

        if freq is None:
            return None
        if isinstance(freq, str):
            try:
                freq = pd.tseries.frequencies.to_offset(freq)
            except ValueError:
                freq = pd.PeriodDtype(freq).freq
        n = freq.n
        if isinstance(freq, o.Day):
            if tz is not None and int(pd.__version__.split(".")[0]) >= 3:
                return cls("calday", n=n, tz=tz)
            return cls("fixed", nanos=n * NS["d"], tz=tz)
        if isinstance(freq, o.Tick):
            return cls("fixed", nanos=int(pd.Timedelta(freq).value), tz=tz)
        if isinstance(freq, o.Week) and freq.weekday is not None:
            return cls("fixed", nanos=n * NS["w"], week_anchor=freq.weekday, tz=tz)
        if isinstance(freq, o.Week):
            return cls("fixed", nanos=n * NS["w"], tz=tz)
        if isinstance(freq, o.CustomBusinessDay):  # subclass of BusinessDay
            hol = tuple(np.asarray(freq.holidays, dtype="M8[D]").astype(str))
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            wm = "".join("1" if d in freq.weekmask.split() else "0" for d in days)
            return cls("bday", n=n, holidays=hol, weekmask=wm, tz=tz)
        if isinstance(freq, o.BusinessDay):
            return cls("bday", n=n, tz=tz)
        months = {o.MonthBegin: (1, "start"), o.MonthEnd: (1, "end"),
                  o.QuarterBegin: (3, "start"), o.QuarterEnd: (3, "end"),
                  o.YearBegin: (12, "start"), o.YearEnd: (12, "end")}
        for typ, (m, anchor) in months.items():
            if type(freq) is typ:
                return cls("month", n=n * m, anchor=anchor, tz=tz)
        bmonths = {o.BusinessMonthBegin: (1, "start"), o.BusinessMonthEnd: (1, "end")}
        for typ, (m, anchor) in bmonths.items():
            if type(freq) is typ:
                return cls("bmonth", n=n * m, anchor=anchor, tz=tz)
        if isinstance(freq, o.SemiMonthEnd) and freq.day_of_month == 15:
            return cls("halfmonth", n=n, anchor="end", tz=tz)
        return None  # FY5253, Easter, BusinessHour, custom calendars, ...

    def to_polars_interval(self):
        """Polars interval string for pl.datetime_range / offset_by, or None."""
        if self.kind == "fixed":
            return f"{self.nanos}ns"
        if self.kind == "calday":
            return f"{self.n}d"
        if self.kind == "month":
            return f"{self.n}mo"
        if self.kind == "int":
            return f"{self.n}i"
        return None  # bday / bmonth / halfmonth need expression rules


# ======================================================================= pandas
class PandasTimeOps:
    """Reference implementation via pandas offsets."""

    @staticmethod
    def to_offset(step: Step):
        import pandas as pd
        from pandas.tseries import offsets as o

        if step.kind == "fixed":
            return pd.Timedelta(step.nanos, "ns")
        if step.kind == "calday":
            return o.Day(step.n)
        if step.kind == "bday":
            if step.holidays:
                return o.CustomBusinessDay(step.n, holidays=list(step.holidays))
            return o.BusinessDay(step.n)
        if step.kind == "month":
            return (o.MonthBegin if step.anchor == "start" else o.MonthEnd)(step.n)
        if step.kind == "bmonth":
            cls = o.BusinessMonthBegin if step.anchor == "start" else o.BusinessMonthEnd
            return cls(step.n)
        if step.kind == "halfmonth":
            return o.SemiMonthEnd(step.n)
        raise NotImplementedError(step.kind)

    @classmethod
    def add_steps(cls, cutoffs, k, step: Step):
        """cutoffs: DatetimeIndex/Index[int] (U,), k: int array (H,) -> (U, H)."""
        import pandas as pd

        if step.kind == "int":
            return np.asarray(cutoffs)[:, None] + step.n * np.asarray(k)[None, :]
        off = cls.to_offset(step)
        cols = [np.asarray(pd.DatetimeIndex(cutoffs) + int(kk) * off) for kk in k]
        return np.stack(cols, axis=1)


# ======================================================================= polars
class PolarsTimeOps:
    """Polars-expression implementation; operates on columns, vectorised."""

    @staticmethod
    def add_steps_expr(c, k, step: Step):
        """Expression: time expr ``c`` shifted by integer expr ``k`` steps."""
        import polars as pl

        if step.kind == "int":
            return c + k * step.n
        if step.kind == "fixed":
            return c + pl.duration(nanoseconds=k * step.nanos)
        if step.kind == "calday":
            return c.dt.offset_by(pl.format("{}d", k * step.n))
        if step.kind == "bday":
            return c.dt.add_business_days(
                k * step.n, week_mask=[b == "1" for b in step.weekmask],
                holidays=[np.datetime64(h).astype(object) for h in step.holidays],
            )
        if step.kind == "month":
            out = c.dt.offset_by(pl.format("{}mo", k * step.n))
            return out.dt.month_end() if step.anchor == "end" else out
        if step.kind == "bmonth":
            base = c.dt.month_start().dt.offset_by(pl.format("{}mo", k * step.n))
            if step.anchor == "start":
                return base.dt.add_business_days(0, roll="forward")
            return base.dt.month_end().dt.add_business_days(0, roll="backward")
        if step.kind == "halfmonth":
            o = (c.dt.year() * 12 + c.dt.month() - 1) * 2 + (c.dt.day() > 15).cast(
                pl.Int32
            )
            o2 = o + k * step.n
            m = o2 // 2
            first = pl.datetime(m // 12, m % 12 + 1, 1)
            return pl.when(o2 % 2 == 0).then(first.dt.offset_by("14d")).otherwise(
                first.dt.month_end()
            )
        raise NotImplementedError(step.kind)

    @staticmethod
    def steps_between_expr(a, b, step: Step):
        """Expression: integer number of steps from a to b (b - a)."""
        import polars as pl

        if step.kind == "int":
            return (b - a) // step.n
        if step.kind == "fixed":
            return (b - a).dt.total_nanoseconds() // step.nanos
        if step.kind == "calday":
            return (b.dt.date() - a.dt.date()).dt.total_days() // step.n
        if step.kind in ("month", "bmonth"):
            ym = lambda x: x.dt.year() * 12 + x.dt.month()  # noqa: E731
            return (ym(b) - ym(a)) // step.n
        if step.kind == "bday":
            return pl.business_day_count(
                a.dt.date(), b.dt.date(),
                week_mask=[x == "1" for x in step.weekmask],
                holidays=[np.datetime64(h).astype(object) for h in step.holidays],
            ) // step.n
        if step.kind == "halfmonth":
            o = lambda x: (x.dt.year() * 12 + x.dt.month() - 1) * 2 + (  # noqa: E731
                x.dt.day() > 15
            ).cast(pl.Int32)
            return (o(b) - o(a)) // step.n
        raise NotImplementedError(step.kind)

    @staticmethod
    def range(start, n, step: Step):
        import polars as pl

        k = pl.int_range(0, n, eager=True).alias("k")
        df = pl.DataFrame({"k": k}).with_columns(c=pl.lit(start))
        return df.select(
            PolarsTimeOps.add_steps_expr(pl.col("c"), pl.col("k"), step).alias("t")
        )["t"]

    @staticmethod
    def infer_step(times):
        """Infer a Step from a sorted polars Series of times, None if irregular."""
        import polars as pl

        s = times.drop_nulls().unique().sort()
        if len(s) < 2:
            return None
        if s.dtype.is_integer():
            d = s.diff().drop_nulls().unique()
            return Step("int", n=int(d[0])) if len(d) == 1 else None
        tz = getattr(s.dtype, "time_zone", None)
        d = s.diff().drop_nulls().dt.total_nanoseconds().unique()
        if len(d) == 1:
            nanos = int(d[0])
            if nanos % NS["w"] == 0:
                return Step("fixed", nanos=nanos, tz=tz,
                            week_anchor=int(s.dt.weekday()[0]) - 1)
            return Step("fixed", nanos=nanos, tz=tz)
        # calendar days in a time zone (DST makes the duration differ)
        if tz is not None:
            dd = s.dt.date().diff().drop_nulls().dt.total_days().unique()
            same_tod = s.dt.time().n_unique() == 1
            if len(dd) == 1 and same_tod:
                return Step("calday", n=int(dd[0]), tz=tz)
        # months
        ym = s.dt.year() * 12 + s.dt.month()
        dm = ym.diff().drop_nulls().unique()
        if len(dm) == 1 and s.dt.time().n_unique() == 1:
            n = int(dm[0])
            if (s.dt.day() == 1).all():
                return Step("month", n=n, anchor="start", tz=tz)
            if (s.dt.date() == s.dt.month_end().dt.date()).all():
                return Step("month", n=n, anchor="end", tz=tz)
            # business month start / end
            bms = s.dt.month_start().dt.add_business_days(0, roll="forward")
            if (s == bms).all():
                return Step("bmonth", n=n, anchor="start", tz=tz)
            bme = s.dt.month_end().dt.add_business_days(0, roll="backward")
            if (s == bme).all():
                return Step("bmonth", n=n, anchor="end", tz=tz)
        # business days
        if s.dt.is_business_day().all():
            nxt = s.head(len(s) - 1).dt.add_business_days(1)
            if (nxt == s.tail(len(s) - 1)).all():
                return Step("bday", n=1, tz=tz)
        return None


# ======================================================================== numpy
class NumpyTimeOps:
    """numpy datetime64 implementation (naive times only - numpy has no tz)."""

    D = np.timedelta64(1, "D")

    @classmethod
    def add_steps(cls, cutoffs, k, step: Step):
        c = np.asarray(cutoffs)
        k = np.asarray(k)[None, :]
        if step.kind == "int":
            return c[:, None] + step.n * k
        if step.tz is not None:
            raise NotImplementedError("numpy datetime64 has no time zones")
        c = c.astype("M8[ns]")
        tod = c - c.astype("M8[D]")
        if step.kind in ("fixed", "calday"):
            nanos = step.nanos if step.kind == "fixed" else step.n * NS["d"]
            return c[:, None] + k * np.timedelta64(nanos, "ns")
        if step.kind == "bday":
            hol = np.asarray(step.holidays, dtype="M8[D]") if step.holidays else []
            return np.busday_offset(
                c.astype("M8[D]")[:, None], k * step.n, roll="forward",
                weekmask=step.weekmask, holidays=hol,
            ).astype("M8[ns]") + tod[:, None]
        if step.kind == "month":
            m = c.astype("M8[M]")[:, None] + k * step.n
            if step.anchor == "start":
                return m.astype("M8[ns]") + tod[:, None]
            return ((m + 1).astype("M8[D]") - cls.D).astype("M8[ns]") + tod[:, None]
        if step.kind == "bmonth":
            m = c.astype("M8[M]")[:, None] + k * step.n
            if step.anchor == "start":
                return np.busday_offset(m.astype("M8[D]"), 0, roll="forward").astype(
                    "M8[ns]"
                )
            return np.busday_offset(
                (m + 1).astype("M8[D]") - cls.D, 0, roll="backward"
            ).astype("M8[ns]")
        if step.kind == "halfmonth":
            m = c.astype("M8[M]")
            day = (c.astype("M8[D]") - m.astype("M8[D]")).astype(int) + 1
            o2 = (m.astype(int) * 2 + (day > 15))[:, None] + k * step.n
            mm = (o2 // 2).astype("M8[M]")
            return np.where(
                o2 % 2 == 0, mm.astype("M8[D]") + 14 * cls.D,
                (mm + 1).astype("M8[D]") - cls.D,
            ).astype("M8[ns]")
        raise NotImplementedError(step.kind)

    @classmethod
    def steps_between(cls, a, b, step: Step):
        a = np.asarray(a)
        b = np.asarray(b)
        if step.kind == "int":
            return (b - a) // step.n
        if step.kind == "fixed":
            return (b - a).astype("m8[ns]").astype(np.int64) // step.nanos
        if step.kind == "calday":
            return (b.astype("M8[D]") - a.astype("M8[D]")).astype(int) // step.n
        if step.kind in ("month", "bmonth"):
            return (b.astype("M8[M]").astype(int) - a.astype("M8[M]").astype(int)) // (
                step.n
            )
        if step.kind == "bday":
            hol = np.asarray(step.holidays, dtype="M8[D]") if step.holidays else []
            return np.busday_count(
                a.astype("M8[D]"), b.astype("M8[D]"), weekmask=step.weekmask,
                holidays=hol,
            ) // step.n
        if step.kind == "halfmonth":

            def o(x):
                m = x.astype("M8[M]")
                day = (x.astype("M8[D]") - m.astype("M8[D]")).astype(int) + 1
                return m.astype(int) * 2 + (day > 15)

            return (o(b) - o(a)) // step.n
        raise NotImplementedError(step.kind)
