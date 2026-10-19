"""g3: cumulative "combined architecture" stack C0..C5 for vectorized forecasters.

Each level adds one layer on top of the previous one; nothing under ``sktime/`` is
edited (monkeypatches, installed/removed by ``configure(level)``).

C0  stock sktime.
C1  + cheap framework fixes, no architecture change (x9 / x4 / critic section 5):
      (a) polars ``is_monotonically_increasing`` as one vectorized expression
          (null guard: falls back to stock if any key/time null);
      (b) known-mtype pass-through: the mtype found by ``check_is_scitype`` (and the
          mtype produced by ``convert`` / ``VectorizedDF.reconstruct``) is remembered
          per object, and ``infer_mtype`` / ``_vectorize.mtype`` return it instead of
          re-running the mtype checks (``convert_to`` re-inference skipped).
C2  + g1 trusted per-slice path (``protos/g1_trusted.py``, ``install()``).
C3  + time axis once per panel and vectorized cutoff (a1 P10, a3 Step):
      (a) outer ``get_cutoff`` on a pandas MultiIndex frame: one max over the time
          level codes instead of groupby().nth(-1) (a1 P10);
      (b) the panel step is inferred and verified once (numpy over all within-instance
          diffs); per-slice ``_enforce_index_freq`` (``pd.infer_freq`` per slice) is
          replaced by setting that verified freq without re-validation; fallback to
          stock for calendar/irregular/short series;
      (c) ``fh.to_absolute`` / ``to_relative`` computed once per unique
          (fh, cutoff) (cheap key; the stock lru_cache never hits because
          ``_HashIndex`` has no ``__eq__``).
C4  + hub / positional iteration (x6 hub, g2 ``NpGrouped`` with a3 Step):
      ``prepare_VectorizedDF`` is replaced (for univariate Panel/Hierarchical,
      pandas MultiIndex or eager polars input) by one hub ingress (no pandas
      MultiIndex built for polars input, no groupby); each instance is handed to the
      *unchanged* forecaster as a trusted pandas view (zero-copy values, trusted
      freq) through the g1 trusted entries; predictions are concatenated in numpy
      and egressed once to the native container (no ``reconstruct`` / convert).
C5  + panel-native kernel where available: g2 ``G2NaiveForecaster`` /
      ``G2PolynomialTrendForecaster`` (``np_grouped`` inner mtype, batched numpy),
      used through the public ``BaseForecaster`` API with the C1 fixes active.
      C2..C4 do not apply (no vectorization happens).
"""

from __future__ import annotations

import os
import sys
import weakref
from collections import OrderedDict

import numpy as np
import pandas as pd

ED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ED)
sys.path.insert(0, os.path.join(ED, "protos"))

import sktime.datatypes._adapter.polars as _sk_pl  # noqa: E402
import sktime.datatypes._check as _sk_check  # noqa: E402
import sktime.datatypes._convert as _sk_convert  # noqa: E402
import sktime.datatypes._vectorize as _sk_vec  # noqa: E402
import sktime.forecasting.base._base as _sk_base  # noqa: E402
import sktime.forecasting.base._fh as _sk_fh  # noqa: E402
import sktime.transformations.base._base as _sk_tbase  # noqa: E402
from sktime.datatypes._convert import _get_first_mtype_of_same_scitype  # noqa: E402
from sktime.datatypes._vectorize import VectorizedDF  # noqa: E402
from sktime.forecasting.base._base import BaseForecaster  # noqa: E402

_G2 = {}


def _g2():
    """Import g2 lazily (registers np_grouped mtypes + wraps get_cutoff/update_data)."""
    if not _G2:
        import g1_trusted as g1  # noqa: F401

        from protos import g2_forecasters as g2f
        from protos import g2_np_grouped as g2n

        _G2["n"], _G2["f"] = g2n, g2f
    return _G2["n"], _G2["f"]


STATS = {"memo_hit": 0, "memo_miss": 0, "fh_hit": 0, "fh_miss": 0,
         "freq_trusted": 0, "freq_present": 0, "freq_fallback": 0, "hub_fit": 0, "hub_predict": 0,
         "cutoff_fast": 0}


def reset_stats():
    for k in STATS:
        STATS[k] = 0


# =============================================================================
# patch bookkeeping
# =============================================================================
_PATCHES = []  # (obj, attr, original)


def _patch(obj, attr, new):
    _PATCHES.append((obj, attr, getattr(obj, attr)))
    setattr(obj, attr, new)


def uninstall_all():
    import g1_trusted as g1

    g1.uninstall()
    while _PATCHES:
        obj, attr, orig = _PATCHES.pop()
        setattr(obj, attr, orig)
    _MEMO.clear()
    _FH_MEMO.clear()


LEVEL = {"level": 0}


def configure(level):
    """Install layers C1..C<level> (cumulative). level 0 = stock."""
    import g1_trusted as g1

    uninstall_all()
    LEVEL["level"] = level
    if level >= 1:
        _install_c1()
    if level >= 2:
        g1.install("schema_items")
    if level >= 3:
        _install_c3()
    if level >= 4:
        _install_c4()
    # C5 is a different estimator class (g2) + C1 fixes; see make_estimator


# =============================================================================
# C1 (a): vectorized polars monotonic check
# =============================================================================
_ORIG_MONO = _sk_pl.is_monotonically_increasing


def mono_vectorized(obj):
    """Same answer as stock ``is_monotonically_increasing`` for eager frames."""
    import polars as pl

    if not isinstance(obj, pl.DataFrame):
        return _ORIG_MONO(obj)
    idx = _sk_pl.get_mi_cols(obj)
    if len(idx) < 2:
        return _ORIG_MONO(obj)
    t, keys = idx[-1], idx[:-1]
    if any(obj[c].null_count() for c in idx):  # null guard
        return _ORIG_MONO(obj)
    ok = obj.select(
        (pl.col(t) >= pl.col(t).shift(1).over(keys)).fill_null(True).all()
    ).item()
    return bool(ok)


# =============================================================================
# C1 (b): known-mtype pass-through
# =============================================================================
_MEMO = OrderedDict()  # id(obj) -> (ref, mtype)
_MEMO_MAX = 64


def _ref(obj):
    try:
        return weakref.ref(obj)
    except TypeError:
        return lambda o=obj: o


def remember_mtype(obj, mt):
    if obj is None or not isinstance(mt, str):
        return
    _MEMO[id(obj)] = (_ref(obj), mt)
    _MEMO.move_to_end(id(obj))
    while len(_MEMO) > _MEMO_MAX:
        _MEMO.popitem(last=False)


def known_mtype(obj):
    e = _MEMO.get(id(obj))
    if e is None or e[0]() is not obj:
        return None
    return e[1]


def _as_list(x):
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


def _wrap_check_is_scitype(orig):
    def check_is_scitype(obj, *args, **kwargs):
        res = orig(obj, *args, **kwargs)
        if isinstance(res, tuple) and len(res) == 3 and res[0]:
            md = res[2]
            if isinstance(md, dict) and isinstance(md.get("mtype"), str):
                remember_mtype(obj, md["mtype"])
        return res

    return check_is_scitype


def _wrap_infer(orig):
    from sktime.datatypes._registry import mtype_to_scitype

    def infer(obj, as_scitype=None, *args, **kwargs):
        mt = known_mtype(obj)
        if mt is not None:
            sc = _as_list(as_scitype)
            if sc is None or mtype_to_scitype(mt) in sc:
                STATS["memo_hit"] += 1
                return mt
        STATS["memo_miss"] += 1
        mt = orig(obj, as_scitype, *args, **kwargs)
        remember_mtype(obj, mt)
        return mt

    return infer


def _wrap_convert(orig):
    def convert(obj, from_type, to_type, as_scitype=None, store=None,
                store_behaviour=None, return_to_mtype=False):
        res, to = orig(obj, from_type, to_type, as_scitype=as_scitype, store=store,
                       store_behaviour=store_behaviour, return_to_mtype=True)
        remember_mtype(res, to)
        return (res, to) if return_to_mtype else res

    return convert


def _wrap_reconstruct(orig):
    def reconstruct(self, df_list, convert_back=False, overwrite_index=True,
                    col_multiindex="none"):
        out = orig(self, df_list, convert_back=convert_back,
                   overwrite_index=overwrite_index, col_multiindex=col_multiindex)
        if not convert_back and col_multiindex == "none":
            remember_mtype(out, _sk_vec._PANDAS_MTYPE_BY_SCITYPE[self.is_scitype])
        return out

    return reconstruct


def _install_c1():
    _patch(_sk_pl, "is_monotonically_increasing", mono_vectorized)
    _patch(_sk_base, "check_is_scitype", _wrap_check_is_scitype(_sk_base.check_is_scitype))
    _patch(_sk_vec, "check_is_scitype", _wrap_check_is_scitype(_sk_vec.check_is_scitype))
    # nested transformers (e.g. Lag inside NaiveForecaster) check, then convert_to
    _patch(_sk_tbase, "check_is_scitype",
           _wrap_check_is_scitype(_sk_tbase.check_is_scitype))
    _patch(_sk_tbase, "check_is_mtype", _wrap_check_is_scitype(_sk_tbase.check_is_mtype))
    _patch(_sk_convert, "infer_mtype", _wrap_infer(_sk_convert.infer_mtype))
    _patch(_sk_vec, "mtype", _wrap_infer(_sk_vec.mtype))
    _patch(_sk_convert, "convert", _wrap_convert(_sk_convert.convert))
    _patch(VectorizedDF, "reconstruct", _wrap_reconstruct(VectorizedDF.reconstruct))


# =============================================================================
# C3 (a): vectorized outer cutoff
# =============================================================================
def _wrap_get_cutoff(orig):
    def get_cutoff(obj, cutoff=None, return_index=False, reverse_order=False,
                   convert_input=True):
        if (
            isinstance(obj, pd.DataFrame)
            and isinstance(obj.index, pd.MultiIndex)
            and len(obj) > 0
            and not reverse_order
        ):
            STATS["cutoff_fast"] += 1
            idx = obj.index
            lv = idx.levels[-1]
            codes = idx.codes[-1]
            if lv.is_monotonic_increasing:
                v = lv[int(codes.max())]
            else:
                v = lv.take(codes).max()
            if not return_index:
                return v
            from pandas.core.indexes.base import ensure_index

            res = ensure_index([v])
            if getattr(lv, "freq", None) is not None and res.freq != lv.freq:
                res.freq = lv.freq
            return res
        return orig(obj, cutoff, return_index=return_index,
                    reverse_order=reverse_order, convert_input=convert_input)

    return get_cutoff


# =============================================================================
# C3 (b): panel step once, trusted per-slice freq
# =============================================================================
_ORIG_ENFORCE = _sk_vec._enforce_index_freq
_FREQ_STACK = []


def panel_offset(X_mi):
    """Verified pandas offset of the panel's time axis, or None (-> stock per slice).

    Applies only to tz-naive DatetimeIndex time levels whose within-instance diffs are
    all equal over the whole panel, every instance longer than 2 (stock leaves freq
    None for <= 2), and whose ``pd.infer_freq`` on the first instance is a Tick
    (fixed) offset; for Tick offsets equal diffs <=> the same inferred freq on every
    slice, so the result equals per-slice ``pd.infer_freq``.
    """
    idx = X_mi.index
    if not isinstance(idx, pd.MultiIndex):
        return None
    lv = idx.levels[-1]
    if not isinstance(lv, pd.DatetimeIndex) or lv.tz is not None:
        return None
    n = len(idx)
    if n < 3:
        return None
    inst_codes = [np.asarray(c) for c in idx.codes[:-1]]
    chg = np.zeros(n - 1, dtype=bool)
    for c in inst_codes:
        chg |= c[1:] != c[:-1]
    starts = np.concatenate([[0], np.flatnonzero(chg) + 1])
    lens = np.diff(np.concatenate([starts, [n]]))
    if (lens <= 2).any():
        return None
    t = lv.asi8.take(idx.codes[-1])
    d = np.diff(t)[~chg]
    if len(d) == 0 or not (d == d[0]).all() or d[0] <= 0:
        return None
    first = pd.DatetimeIndex(lv.take(idx.codes[-1][: lens[0]]))
    try:
        f = pd.infer_freq(first)
    except (TypeError, ValueError):
        return None
    if f is None:
        return None
    off = pd.tseries.frequencies.to_offset(f)
    # fixed-length offsets only (Tick; pandas 3: Day is no longer a Tick)
    if not isinstance(off, (pd.offsets.Tick, pd.offsets.Day)):
        return None
    if int((first[1] - first[0]).value) != int(d[0]):
        return None
    return off


def enforce_index_freq_trusted(item):
    off = _FREQ_STACK[-1] if _FREQ_STACK else None
    ix = item.index
    if (
        off is not None
        and type(ix) is pd.DatetimeIndex
        and ix.freq is None
        and len(ix) > 2
    ):
        ix._data._freq = off  # trusted: verified once for the whole panel
        STATS["freq_trusted"] += 1
        return item
    if getattr(ix, "freq", None) is not None:
        STATS["freq_present"] += 1  # slice already carries freq: stock is a no-op
        return item
    STATS["freq_fallback"] += 1
    return _ORIG_ENFORCE(item)


def _wrap_items(orig):
    def items(self, X=None, iterate_as=None, iterate_cols=None):
        off = getattr(self, "_g3_off", None) if X is not None else None
        _FREQ_STACK.append(off)
        try:
            yield from orig(self, X=X, iterate_as=iterate_as, iterate_cols=iterate_cols)
        finally:
            _FREQ_STACK.pop()

    return items


def _wrap_prepare_c3(orig):
    def prepare_VectorizedDF(X, iterate_as="Series", is_scitype="Panel",
                             iterate_cols=False, store=None, store_behaviour=None):
        vec, X_mi = orig(X, iterate_as=iterate_as, is_scitype=is_scitype,
                         iterate_cols=iterate_cols, store=store,
                         store_behaviour=store_behaviour)
        vec._g3_off = panel_offset(X_mi) if iterate_as == "Series" else None
        return vec, X_mi

    return prepare_VectorizedDF


# =============================================================================
# C3 (c): fh.to_absolute / to_relative once per unique (fh, cutoff)
# =============================================================================
_FH_MEMO = OrderedDict()


def _fh_key(fh, cutoff_idx, tag):
    v = fh._values
    c = cutoff_idx
    cvals = c.asi8.tobytes() if hasattr(c, "asi8") else tuple(np.asarray(c).tolist())
    return (tag, fh.is_relative, str(v.dtype), np.asarray(v).tobytes(),
            str(fh.freq), type(c).__name__, cvals, getattr(c, "freqstr", None),
            str(getattr(c, "tz", None)))


def _wrap_fh(orig, tag):
    raw = orig.__wrapped__ if hasattr(orig, "__wrapped__") else orig

    def f(fh, cutoff=None):
        if cutoff is None or not isinstance(getattr(cutoff, "index", None), pd.Index):
            return orig(fh, cutoff)
        try:
            key = _fh_key(fh, cutoff.index, tag)
        except Exception:  # noqa: BLE001
            return orig(fh, cutoff)
        res = _FH_MEMO.get(key)
        if res is None:
            STATS["fh_miss"] += 1
            res = raw(fh, cutoff)
            _FH_MEMO[key] = res
            if len(_FH_MEMO) > 4096:
                _FH_MEMO.popitem(last=False)
        else:
            STATS["fh_hit"] += 1
        return res

    return f


def _install_c3():
    import g1_trusted as g1

    _patch(_sk_base, "get_cutoff", _wrap_get_cutoff(_sk_base.get_cutoff))
    _patch(_sk_vec, "_enforce_index_freq", enforce_index_freq_trusted)
    _patch(VectorizedDF, "items", _wrap_items(VectorizedDF.items))
    _patch(_sk_base, "prepare_VectorizedDF",
           _wrap_prepare_c3(_sk_base.prepare_VectorizedDF))
    _patch(_sk_fh, "_to_absolute", _wrap_fh(_sk_fh._to_absolute, "abs"))
    _patch(_sk_fh, "_to_relative", _wrap_fh(_sk_fh._to_relative, "rel"))
    _ = g1  # g1 already installed at C2


# =============================================================================
# C4: hub ingress + positional iteration + trusted pandas views + native egress
# =============================================================================
class HubSchema:
    """Stands in for VectorizedDF when the hub route is taken."""

    def __init__(self, npg, is_scitype, columns):
        self.npg = npg
        self.is_scitype = is_scitype
        self.columns = columns  # pandas columns Index of the y slices
        self.shape = (npg.n_instances, 1)
        self.iterate_cols = False
        self.iterate_as = "Series"


def _strip(name):
    if isinstance(name, str) and name.startswith("__index__"):
        return name[9:]
    return name


def _hub_ingress(X, is_scitype):
    """X -> (NpGrouped, columns Index) or None if the hub route does not apply."""
    g2n, _ = _g2()
    import polars as pl

    if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.MultiIndex):
        if not all(dt.kind == "f" for dt in X.dtypes):
            return None
        if not isinstance(X.index.levels[-1], pd.DatetimeIndex):
            return None
        hub = g2n.from_pandas(X)
        cols = X.columns
    elif isinstance(X, pl.DataFrame):
        idx = _sk_pl.get_mi_cols(X)
        if len(idx) < 2:
            return None
        hub = g2n.from_polars(X, key_cols=idx[:-1], time_col=idx[-1])
        if hub.tmeta.kind != "datetime":
            return None
        # stock converts to pandas: index names / columns as pandas would have them
        cols = pd.Index([c for c in X.columns if c not in idx])
    else:
        return None
    if hub.aux or hub.meta.get("perm") is not None or hub.tmeta.tz is not None:
        return None
    npg = g2n._wrap(hub, is_scitype)
    if npg.step is None or npg.step.kind != "fixed" or (npg.lengths <= 2).any():
        return None
    return npg, cols


def _wrap_prepare_c4(orig):
    def prepare_VectorizedDF(X, iterate_as="Series", is_scitype="Panel",
                             iterate_cols=False, store=None, store_behaviour=None):
        if iterate_as == "Series" and not iterate_cols and is_scitype in (
                "Panel", "Hierarchical"):
            r = _hub_ingress(X, is_scitype)
            if r is not None:
                npg, cols = r
                return HubSchema(npg, is_scitype, cols), npg
        return orig(X, iterate_as=iterate_as, is_scitype=is_scitype,
                    iterate_cols=iterate_cols, store=store,
                    store_behaviour=store_behaviour)

    return prepare_VectorizedDF


def _hub_offset(npg):
    # same freq string the slices would carry in stock (pandas freq of the source,
    # else from the a3 Step); pandas 3 distinguishes Day from 24h
    g2n, _ = _g2()
    return pd.tseries.frequencies.to_offset(g2n.step_freqstr(npg))


def _trusted_dt_index(t, off, name):
    from pandas.core.arrays import DatetimeArray

    # keep the source unit (pandas 3 defaults to datetime64[us])
    arr = DatetimeArray._simple_new(t, freq=off, dtype=t.dtype)
    return pd.DatetimeIndex._simple_new(arr, name=name)


def _row_key(npg, i):
    if len(npg.key_values) == 1:
        return npg.key_values[0][i]
    return tuple(kv[i] for kv in npg.key_values)


def hub_vectorize(self, methodname, **kwargs):
    import g1_trusted as g1

    y = kwargs.get("y", None)
    if methodname == "fit":
        schema = y[1] if y else None
    else:
        schema = getattr(self, "_y_schema", None)
    X = kwargs.get("X", None)
    has_X = X is not None and isinstance(X, tuple) and X[0] is not None
    if not isinstance(schema, HubSchema) or has_X or methodname not in ("fit",
                                                                        "predict"):
        if isinstance(schema, HubSchema):
            raise NotImplementedError(f"g3 C4 hub route: {methodname} / X not ported")
        return _PREV_VECTORIZE["f"](self, methodname, **kwargs)
    ok, reason = g1.estimator_eligible(self)
    if not ok:
        raise NotImplementedError(f"g3 C4 hub route: estimator not eligible: {reason}")

    npg = schema.npg
    ctx = g1._make_ctx(self)
    fh = kwargs.get("fh", None)
    if methodname == "fit":
        STATS["hub_fit"] += 1
        off = _hub_offset(npg)
        tname = _strip(npg.time_name)
        cols = schema.columns
        ip, t, v = npg.indptr, npg.time, npg.values
        rows = []
        for i in range(npg.n_instances):
            a, b = ip[i], ip[i + 1]
            y_i = pd.DataFrame(v[a:b], index=_trusted_dt_index(t[a:b], off, tname),
                               columns=cols, copy=False)
            est = g1.trusted_fit_slice(self.clone(), y_i, None, fh, ctx)
            rows.append((_row_key(npg, i), "forecasters", est))
        self._y_schema = schema
        self.forecasters_ = g1._estimators_frame(rows)
        return self

    STATS["hub_predict"] += 1
    ests = self.forecasters_.iloc[:, 0].to_numpy()
    times, vals = [], []
    for est in ests:
        p = g1.trusted_predict_slice(est, "predict", None, fh, ctx, {})
        times.append(p.index.values)
        vals.append(p.to_numpy())
    return _hub_egress(self, npg, times, vals, schema)


def _hub_egress(self, npg, times, vals, schema):
    import polars as pl

    lens = np.fromiter((len(x) for x in times), dtype=np.int64, count=len(times))
    T = np.concatenate(times)
    V = np.concatenate(vals).reshape(len(T), -1)
    src = npg.meta.get("source")
    if src == "polars":
        m = npg.meta
        sch = m["schema"]
        inst = pl.Series(np.repeat(np.arange(npg.n_instances, dtype=np.uint32), lens))
        cols = []
        for k, kv in zip(m["key_cols"], npg.key_values):
            cols.append(pl.Series(k, kv, dtype=sch.get(k)).gather(inst))
        cols.append(pl.Series(m["time_col"], T).cast(sch[m["time_col"]]))
        for j, c in enumerate(schema.columns):
            cols.append(pl.Series(c, V[:, j]).cast(sch.get(c, pl.Float64)))
        out = pl.DataFrame(cols)
    else:
        levels, codes = [], []
        kdt = npg.meta.get("key_dtypes") or [None] * len(npg.key_values)
        for kv, dt in zip(npg.key_values, kdt):
            kc, ku = pd.factorize(kv)
            ku = pd.Index(ku)
            if dt is not None and ku.dtype != dt:
                ku = ku.astype(dt)
            levels.append(ku)
            codes.append(np.repeat(kc, lens))
        tc, tu = pd.factorize(T, sort=True)
        tlev = pd.DatetimeIndex(tu)
        if len(tu) > 2:
            dd = np.diff(tu.view(np.int64))
            if (dd == dd[0]).all():
                tlev = _trusted_dt_index(tu, _hub_offset(npg), None)
        levels.append(tlev)
        codes.append(tc)
        mi = pd.MultiIndex(levels=levels, codes=codes,
                           names=[*npg.key_names, npg.time_name],
                           verify_integrity=False)
        out = pd.DataFrame(V, index=mi, columns=schema.columns, copy=False)
    # egress already produced the input container: identity in predict's convert_to
    remember_mtype(out, self._y_metadata["mtype"])
    return out


_PREV_VECTORIZE = {}


def _install_c4():
    _PREV_VECTORIZE["f"] = BaseForecaster._vectorize  # g1 trusted_vectorize
    _patch(BaseForecaster, "_vectorize", hub_vectorize)
    _patch(_sk_base, "prepare_VectorizedDF",
           _wrap_prepare_c4(_sk_base.prepare_VectorizedDF))
    # cutoff on the hub: delegate NpGrouped to g2's get_cutoff wrapper (already in
    # the chain below the C3 wrapper); nothing to add.


# =============================================================================
# estimators
# =============================================================================
def make_estimator(name, level):
    """Stock estimator for C0..C4, g2 port for C5. remember_data=True explicit."""
    if level >= 5:
        _, g2f = _g2()
        if name == "naive":
            f = g2f.G2NaiveForecaster(strategy="last")
        elif name == "poly":
            f = g2f.G2PolynomialTrendForecaster(degree=1, prediction_intervals=True)
        else:
            raise ValueError(name)
    else:
        if name == "naive":
            from sktime.forecasting.naive import NaiveForecaster

            f = NaiveForecaster(strategy="last")
        elif name == "poly":
            from sktime.forecasting.trend import PolynomialTrendForecaster

            f = PolynomialTrendForecaster(degree=1, prediction_intervals=True)
        else:
            raise ValueError(name)
    f.set_config(remember_data=True)
    return f
