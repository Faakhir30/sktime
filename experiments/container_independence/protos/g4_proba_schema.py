"""g4: a backend-neutral long ("tidy") schema for probabilistic forecast outputs.

Prototype only, NOT part of sktime.

Today ``predict_interval`` / ``predict_quantiles`` return a pandas frame whose ROW
index is (instance levels..., time) and whose COLUMN index is a MultiIndex
``(variable, coverage, "lower"/"upper")`` resp. ``(variable, alpha)`` built by
``BaseForecaster._get_columns`` (``_base.py:2797/2805``). Only pandas has a column
MultiIndex, and only pandas has a row MultiIndex, so non-pandas backends cannot hold
that layout. The schema below can be held by any columnar backend:

* ``interval`` (semi-long):  key_1..key_k | time | variable | coverage | lower | upper
* ``quantiles`` (long):      key_1..key_k | time | variable | alpha    | value
* ``var`` (long):            key_1..key_k | time | variable | var

Rows are ordered (instance, time, variable, coverage/alpha) with variable and
coverage/alpha in the REQUESTED order (sktime keeps the user's coverage order, so
we keep an ordinal, not a sort). A small ``ProbaSpec`` travels with the frame: it
holds the key/time names (including ``None`` names), the variable labels in their
original (possibly non-str) type, and the coverage/alpha list. With the spec, the
conversion to today's pandas layout never parses strings.

Interval <-> quantiles is a pure relabelling: lower = q(0.5 - c/2), upper =
q(0.5 + c/2) (as in ``_PredictProbaMixin._predict_interval``).
"""

from dataclasses import dataclass, field

import numpy as np

VAR, COV, ALPHA, LOW, UPP, VAL, VARIANCE = (
    "variable",
    "coverage",
    "alpha",
    "lower",
    "upper",
    "value",
    "var",
)


@dataclass
class ProbaSpec:
    """Metadata that turns a long proba frame back into the sktime pandas layout."""

    kind: str  # "interval" | "quantiles" | "var"
    key_names: list  # instance level names, may contain None
    time_name: object  # time level name, may be None
    variables: list  # original variable labels (any hashable)
    levels: list = field(default_factory=list)  # coverage or alpha list (req. order)
    # physical column names used in the long frame for keys/time
    phys_keys: list = field(default_factory=list)
    phys_time: str = "time"

    @property
    def n_per_row(self):
        """Long rows per (instance, time) row of the wide output."""
        return len(self.variables) * max(1, len(self.levels))

    def pandas_columns(self):
        import pandas as pd

        if self.kind == "interval":
            return pd.MultiIndex.from_product(
                [self.variables, self.levels, ["lower", "upper"]]
            )
        if self.kind == "quantiles":
            return pd.MultiIndex.from_product([self.variables, self.levels])
        return pd.Index(self.variables)


def _phys(names, prefix="__key"):
    return [n if isinstance(n, str) else f"{prefix}{i}" for i, n in enumerate(names)]


# ---------------------------------------------------------------------------
# pandas (today's layout) -> long
# ---------------------------------------------------------------------------
def spec_from_pandas(df, kind):
    """Read a ProbaSpec off a stock pandas proba output."""
    import pandas as pd

    idx = df.index
    names = list(idx.names) if isinstance(idx, pd.MultiIndex) else [idx.name]
    key_names, time_name = names[:-1], names[-1]
    cols = df.columns
    if kind == "var":
        variables, levels = list(cols), []
    else:
        variables = list(cols.get_level_values(0).unique())
        levels = list(cols.get_level_values(1).unique())
    phys = _phys(key_names)
    ptime = time_name if isinstance(time_name, str) else "time"
    return ProbaSpec(kind, key_names, time_name, variables, levels, phys, ptime)


def pandas_to_long(df, kind, backend="pandas"):
    """sktime pandas proba frame -> (long frame, ProbaSpec). O(n) numpy reshapes."""
    import pandas as pd

    spec = spec_from_pandas(df, kind)
    n = len(df)
    V, L = len(spec.variables), max(1, len(spec.levels))
    m = V * L
    idx = df.index
    cols = {}
    if isinstance(idx, pd.MultiIndex):
        for j, p in enumerate(spec.phys_keys):
            cols[p] = np.repeat(idx.get_level_values(j).to_numpy(), m)
        t = idx.get_level_values(-1)
    else:
        t = idx
    cols[spec.phys_time] = np.repeat(t.to_numpy(), m)
    var_codes = np.repeat(np.arange(V), L)
    cols[VAR] = np.asarray(spec.variables, dtype=object)[np.tile(var_codes, n)]
    if kind == "var":
        cols[VARIANCE] = df.to_numpy().reshape(n * m)
    else:
        lev = np.tile(np.tile(np.asarray(spec.levels, dtype=float), V), n)
        if kind == "interval":
            vals = df.to_numpy().reshape(n * m, 2)
            cols[COV] = lev
            cols[LOW] = vals[:, 0]
            cols[UPP] = vals[:, 1]
        else:
            cols[ALPHA] = lev
            cols[VAL] = df.to_numpy().reshape(n * m)
    if backend == "pandas":
        out = pd.DataFrame(cols)
        out[VAR] = out[VAR].astype(str)
        return out, spec
    import polars as pl

    cols[VAR] = cols[VAR].astype(str)
    return pl.DataFrame(cols), spec


# ---------------------------------------------------------------------------
# long -> pandas (today's layout)
# ---------------------------------------------------------------------------
def _np_cols(long, names):
    """Column -> numpy, for pandas or polars long frames."""
    if hasattr(long, "get_column"):  # polars
        return [long.get_column(c).to_numpy() for c in names]
    return [long[c].to_numpy() for c in names]


def _check_canonical(var_col, lev_col, spec, n):
    """Guard for the reshape fast path: (variable, level) pattern repeats per row."""
    V, L = len(spec.variables), max(1, len(spec.levels))
    var_str = np.asarray([str(v) for v in spec.variables], dtype=object)
    exp_var = np.tile(var_str[np.repeat(np.arange(V), L)], n)
    if not np.array_equal(np.asarray(var_col, dtype=object), exp_var):
        return False
    if lev_col is not None:
        exp_lev = np.tile(np.tile(np.asarray(spec.levels, float), V), n)
        if not np.array_equal(np.asarray(lev_col, float), exp_lev):
            return False
    return True


def _check_row_blocks(long, spec, m):
    """Guard: each m-row block shares (keys, time); instances contiguous; time
    strictly increasing within an instance. Any instance ORDER is accepted."""
    import pandas as pd

    keys = _np_cols(long, spec.phys_keys)
    t = _np_cols(long, [spec.phys_time])[0]
    if m > 1:
        for a in keys + [t]:
            if not np.array_equal(np.repeat(a[::m], m), a):
                return False
    keys = [a[::m] for a in keys]
    t = t[::m]
    if keys:
        codes = (
            pd.factorize(keys[0])[0]
            if len(keys) == 1
            else pd.MultiIndex.from_arrays(keys).factorize()[0]
        )
        change = codes[1:] != codes[:-1]
        if change.sum() + 1 != codes.max() + 1:
            return False
        same = ~change
    else:
        same = np.ones(len(t) - 1, dtype=bool)
    return bool(np.all(t[1:][same] > t[:-1][same]))


def long_to_pandas(long, spec, guard=True, time_unit="ns"):
    """Long frame (pandas or polars) + spec -> sktime pandas proba frame.

    Fast path: rows are in canonical order, so the value block is a reshape and the
    row index is every ``n_per_row``-th key/time tuple. ``guard=True`` checks the
    canonical pattern (O(rows)); on failure the frame is re-ordered first.
    """
    import pandas as pd

    m = spec.n_per_row
    nrows_long = long.height if hasattr(long, "height") else len(long)
    if nrows_long % m:
        raise ValueError("long frame is not complete: rows % (n_var*n_levels) != 0")
    n = nrows_long // m
    lev_name = {"interval": COV, "quantiles": ALPHA, "var": None}[spec.kind]
    if guard:
        var_col = _np_cols(long, [VAR])[0]
        lev_col = _np_cols(long, [lev_name])[0] if lev_name else None
        ok = _check_canonical(var_col, lev_col, spec, n)
        ok = ok and _check_row_blocks(long, spec, m)
        if not ok:
            long = canonical_sort(long, spec)
    sel = slice(None, None, m)
    key_arrays = [a[sel] for a in _np_cols(long, spec.phys_keys)]
    t = _np_cols(long, [spec.phys_time])[0][sel]
    if np.issubdtype(t.dtype, np.datetime64):
        t = t.astype(f"datetime64[{time_unit}]")
    if spec.key_names:
        index = pd.MultiIndex.from_arrays(
            key_arrays + [t], names=spec.key_names + [spec.time_name]
        )
    else:
        index = pd.Index(t, name=spec.time_name)
    if spec.kind == "interval":
        lo, up = _np_cols(long, [LOW, UPP])
        vals = np.stack([lo, up], axis=1).reshape(n, m * 2)
    elif spec.kind == "quantiles":
        vals = _np_cols(long, [VAL])[0].reshape(n, m)
    else:
        vals = _np_cols(long, [VARIANCE])[0].reshape(n, m)
    return pd.DataFrame(vals, index=index, columns=spec.pandas_columns())


def canonical_sort(long, spec):
    """Re-order a long frame to (instance first-appearance, time, var ord, level ord)."""
    lev_name = {"interval": COV, "quantiles": ALPHA, "var": None}[spec.kind]
    var_map = {str(v): i for i, v in enumerate(spec.variables)}
    lev_map = {float(v): i for i, v in enumerate(spec.levels)}
    if hasattr(long, "get_column"):
        import polars as pl

        ex = [
            pl.col(VAR).replace_strict(var_map, return_dtype=pl.Int64).alias("__v"),
        ]
        if lev_name:
            ex.append(
                pl.col(lev_name).replace_strict(lev_map, return_dtype=pl.Int64).alias("__l")
            )
        # instance order: lexicographic by key values (sktime's sorted
        # MultiIndex convention); a long frame carries no "first appearance" order
        out = long.with_columns(ex)
        by = list(spec.phys_keys) + [spec.phys_time, "__v"]
        by += ["__l"] if lev_name else []
        return out.sort(by).drop([c for c in ["__v", "__l"] if c in out.columns])
    out = long.copy()
    out["__v"] = out[VAR].map(var_map)
    if lev_name:
        out["__l"] = out[lev_name].map(lev_map)
    by = list(spec.phys_keys) + [spec.phys_time, "__v"]
    by += ["__l"] if lev_name else []
    out = out.sort_values(by, kind="stable")
    return out.drop(columns=[c for c in ["__v", "__l"] if c in out.columns])


# ---------------------------------------------------------------------------
# relabelling interval <-> quantiles (backend native, polars shown)
# ---------------------------------------------------------------------------
def interval_long_to_quantiles_long_pl(long):
    """polars: (coverage, lower, upper) -> (alpha, value), 2 rows per coverage."""
    import polars as pl

    base = [c for c in long.columns if c not in (COV, LOW, UPP)]
    # float level keys: round like _PredictProbaMixin (np.round(..., 12)), else
    # 0.5 - 0.9/2 = 0.04999999999999999 != 0.05 and joins on alpha fail
    lo_a = (0.5 - pl.col(COV) / 2).round(12).alias(ALPHA)
    up_a = (0.5 + pl.col(COV) / 2).round(12).alias(ALPHA)
    lo = long.select(base + [lo_a, pl.col(LOW).alias(VAL)])
    up = long.select(base + [up_a, pl.col(UPP).alias(VAL)])
    return pl.concat([lo, up])  # unordered; canonical_sort restores order


# ---------------------------------------------------------------------------
# flat-column ("wide") alternative: names encoded, spec carries the labels
# ---------------------------------------------------------------------------
def flat_names(spec):
    """Physical names for a wide flat layout; the spec keeps the real labels."""
    if spec.kind == "interval":
        return [
            f"{v}__{c}__{b}" for v in spec.variables for c in spec.levels
            for b in ("lower", "upper")
        ]
    if spec.kind == "quantiles":
        return [f"{v}__{a}" for v in spec.variables for a in spec.levels]
    return [str(v) for v in spec.variables]


def flat_to_pandas(wide, spec, time_unit="ns"):
    """Wide flat frame (polars or pandas, keys/time as columns) -> sktime pandas."""
    import pandas as pd

    names = flat_names(spec)
    key_arrays = _np_cols(wide, spec.phys_keys)
    t = _np_cols(wide, [spec.phys_time])[0]
    if np.issubdtype(t.dtype, np.datetime64):
        t = t.astype(f"datetime64[{time_unit}]")
    if hasattr(wide, "get_column"):
        vals = wide.select(names).to_numpy()
    else:
        vals = wide[names].to_numpy()
    index = pd.MultiIndex.from_arrays(
        key_arrays + [t], names=spec.key_names + [spec.time_name]
    )
    return pd.DataFrame(vals, index=index, columns=spec.pandas_columns())


# ---------------------------------------------------------------------------
# a native panel proba kernel (x7-style): NaiveForecaster(strategy="last", sp=1)
# ---------------------------------------------------------------------------
def naive_last_interval_polars(y, fh, coverage, key_cols, time_col, value_cols,
                               step="1d"):
    """All instances at once, polars, long interval output.

    Mirrors ``NaiveForecaster._predict_var`` (strategy="last", sp=1):
    mse = sum((y_t - y_{t-1})^2) / (T - n_nan(res)), var_h = mse * max(h, 1),
    and ``_predict_quantiles``: q = y_T + sqrt(var_h) * Phi^-1(alpha).
    Assumes rows sorted by time within instance and no NaN in y (the sortedness /
    NaN guards are the x7/G13 open items).
    """
    import polars as pl
    from scipy.stats import norm

    lf = y.lazy() if isinstance(y, pl.DataFrame) else y
    aggs = []
    for v in value_cols:
        d = pl.col(v) - pl.col(v).shift(1)
        aggs += [
            pl.col(v).last().alias(f"{v}__last"),
            ((d**2).sum() / (pl.len() - d.is_null().sum())).alias(f"{v}__mse"),
        ]
    aggs.append(pl.col(time_col).last().alias("__cut"))
    params = lf.group_by(key_cols, maintain_order=True).agg(aggs)
    h = np.asarray(fh, dtype=np.int64)
    hz = pl.LazyFrame({"__h": h})
    grid = params.join(hz, how="cross")
    zl = {c: norm.ppf(0.5 - c / 2) for c in coverage}
    zu = {c: norm.ppf(0.5 + c / 2) for c in coverage}
    unit = {"1d": "days"}[step]
    parts = []
    for v in value_cols:
        for c in coverage:
            sd = (pl.col(f"{v}__mse") * pl.col("__h").clip(lower_bound=1)).sqrt()
            parts.append(
                grid.select(
                    key_cols
                    + [
                        (pl.col("__cut") + pl.duration(**{unit: pl.col("__h")})).alias(
                            time_col
                        ),
                        pl.lit(str(v)).alias(VAR),
                        pl.lit(float(c)).alias(COV),
                        (pl.col(f"{v}__last") + sd * zl[c]).alias(LOW),
                        (pl.col(f"{v}__last") + sd * zu[c]).alias(UPP),
                        pl.col("__h"),
                        pl.lit(len(parts)).alias("__o"),
                    ]
                )
            )
    # canonical order: instance keys, h, (variable, coverage) ordinal
    out = pl.concat(parts).sort(key_cols + ["__h", "__o"])
    return out.drop(["__o", "__h"])
