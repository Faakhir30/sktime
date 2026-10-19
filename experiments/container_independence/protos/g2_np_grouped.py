"""g2: proto inner mtype "np_grouped" (numpy values + indptr + keys + time axis).

Prototype only, not part of sktime. A forecaster opts in per estimator by declaring

    "y_inner_mtype": ["np_grouped_series", "np_grouped", "np_grouped_hier"]

(one mtype name per scitype, all backed by the same python class ``NpGrouped``).
``BaseForecaster`` is not edited: ``register_np_grouped()`` extends sktime's mtype
registry, the converter dictionary and two dispatch points from the outside
(monkeypatch), see ``register_np_grouped`` for the exact list.

Container (``NpGrouped``, a subclass of the x6 hub ``PanelData``)
-------------------------------------------------------------------
* ``values``   2D float64 (n_rows, n_cols), a view of the source buffer where possible
* ``indptr``   int64 offsets, instance i = rows indptr[i]:indptr[i+1]
* ``key_names`` / ``key_values``  instance key table (one 1D array per level)
* ``time``     per-row datetime64 (naive UTC) / int64 (period ordinals, ints)
* ``step``     a3 ``Step`` (library-neutral time step), inferred over *all* instances
* ``scitype``  "Series" | "Panel" | "Hierarchical"
* ``meta``     egress metadata of the source container (x6 hand-written spokes)

Converters (hand-written spokes of x6): pandas (pd.Series / pd.DataFrame /
pd-multiindex / pd_multiindex_hier), polars (pl.DataFrame / polars_panel /
polars_hierarchical, sktime ``__index__`` convention) and pyarrow (new proto
mtypes ``pa_series`` / ``pa_panel`` / ``pa_hier``: a ``pyarrow.Table`` with
``__index__``-prefixed index columns, the same convention as sktime's polars mtypes).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protos.a3_timeindex import NS, NumpyTimeOps, Step  # noqa: E402
from protos.x6_hub import (  # noqa: E402
    INDEX_PREFIX,
    PanelData,
    TimeMeta,
    from_arrow,
    from_pandas,
    from_polars,
    time_to_pandas_index,
    to_arrow,
    to_pandas,
    to_polars,
)

NPG = {"Series": "np_grouped_series", "Panel": "np_grouped",
       "Hierarchical": "np_grouped_hier"}
PA = {"Series": "pa_series", "Panel": "pa_panel", "Hierarchical": "pa_hier"}
PANDAS = {"Series": ["pd.Series", "pd.DataFrame"], "Panel": ["pd-multiindex"],
          "Hierarchical": ["pd_multiindex_hier"]}
POLARS = {"Series": ["pl.DataFrame"], "Panel": ["polars_panel"],
          "Hierarchical": ["polars_hierarchical"]}
NPG_MTYPES = list(NPG.values())

# conversion counters (instrumentation for the report)
COUNTS = {"ingress": 0, "egress": 0}


# =============================================================================
# container
# =============================================================================
@dataclass
class NpGrouped(PanelData):
    """x6 PanelData + a3 Step + scitype. See module docstring."""

    step: Step | None = None
    scitype: str = "Panel"

    # ------------------------------------------------------------ derived views
    @property
    def ends(self):
        return self.indptr[1:]

    def inst_cutoffs(self):
        """Per-instance last time point (time is sorted within instance)."""
        return self.time[self.indptr[1:] - 1]

    def with_rows(self, indptr, time, values, columns=None):
        """Same keys / egress metadata / step, new rows (e.g. predictions)."""
        meta = dict(self.meta)
        meta["perm"] = None
        meta["aux_order"] = None
        tm = self.tmeta
        if tm.is_range:  # new rows need not be a contiguous range
            tm = TimeMeta(tm.kind, unit=tm.unit, tz=tm.tz, freq=tm.freq,
                          step=tm.step, is_range=False)
        return NpGrouped(
            indptr=np.asarray(indptr, dtype=np.int64), time=time, values=values,
            columns=list(self.columns if columns is None else columns),
            key_names=self.key_names, key_values=self.key_values,
            time_name=self.time_name, tmeta=tm, aux={}, meta=meta,
            step=self.step, scitype=self.scitype,
        )


def _wrap(hub, scitype, freq_hint=None):
    COUNTS["ingress"] += 1
    if hub.aux:
        raise TypeError("np_grouped holds float columns only (non-float y columns)")
    step = infer_step(hub, freq_hint)
    d = {k: getattr(hub, k) for k in PanelData.__dataclass_fields__}
    return NpGrouped(**d, step=step, scitype=scitype)


# =============================================================================
# step inference over all instances (numpy only; loud None instead of silent)
# =============================================================================
def _within_mask(indptr, n):
    """Boolean mask over np.diff(time) positions that lie inside one instance."""
    m = np.ones(max(n - 1, 0), dtype=bool)
    b = indptr[1:-1] - 1
    m[b[(b >= 0) & (b < n - 1)]] = False
    return m


def infer_step(hub, freq_hint=None):
    """a3 ``Step`` from the whole time array; None if irregular / inconsistent."""
    t, tm, n = hub.time, hub.tmeta, hub.n_rows
    if tm.kind in ("int", "period"):
        if n < 2:
            return Step("int", n=int(tm.step or 1))
        d = np.diff(t)[_within_mask(hub.indptr, n)]
        if len(d) == 0:
            return Step("int", n=int(tm.step or 1))
        return Step("int", n=int(d[0])) if (d == d[0]).all() else None
    tz = tm.tz
    if freq_hint is not None:  # trusted pandas freq of a plain (single) index
        st = Step.from_pandas(freq_hint, tz=tz)
        if st is not None:
            return st
    if n < 2:
        return None
    d = np.diff(t)[_within_mask(hub.indptr, n)]
    if len(d) == 0:
        return None
    if (d == d[0]).all():
        nanos = int(d[0].astype("m8[ns]").astype(np.int64))
        return Step("fixed", nanos=nanos, tz=tz)
    mask = _within_mask(hub.indptr, n)
    months = t.astype("M8[M]")
    md = np.diff(months.astype(np.int64))[mask]
    if (md == md[0]).all() and (t == t.astype("M8[D]")).all():
        if (t == months.astype(t.dtype)).all():
            return Step("month", n=int(md[0]), anchor="start", tz=tz)
        if (t == ((months + 1).astype("M8[D]") - np.timedelta64(1, "D"))
                .astype(t.dtype)).all():
            return Step("month", n=int(md[0]), anchor="end", tz=tz)
    days = t.astype("M8[D]")
    if (t == days.astype(t.dtype)).all() and np.is_busday(days).all():
        bd = np.busday_count(days[:-1], days[1:])[mask]
        if (bd == bd[0]).all():
            return Step("bday", n=int(bd[0]), tz=tz)
    return None


def step_freqstr(npg):
    """pandas freq string for the (small) public index objects."""
    if npg.tmeta.freq is not None:
        return npg.tmeta.freq
    st = npg.step
    if st is None or npg.tmeta.kind != "datetime":
        return None
    import pandas as pd

    from protos.a3_timeindex import PandasTimeOps

    off = PandasTimeOps.to_offset(st)
    if isinstance(off, pd.Timedelta):
        off = pd.tseries.frequencies.to_offset(off)
    return off.freqstr


# =============================================================================
# time arithmetic on the hub (numpy only)
# =============================================================================
def _naive_step(st):
    """numpy has no tz: fixed steps are tz-independent on UTC-naive storage."""
    if st.tz is None:
        return st
    if st.kind == "fixed":
        return Step("fixed", nanos=st.nanos)
    raise NotImplementedError(f"calendar step {st.kind} on tz-aware time in numpy")


def add_steps(npg, cut, k):
    """cut (U,) time points, k (H,) ints -> (U, H) time points, hub dtype."""
    st = npg.step
    if st is None:
        raise ValueError("time axis has no regular step (irregular or mixed "
                         "frequencies across instances); cannot place fh")
    if npg.tmeta.kind in ("int", "period"):
        return cut[:, None] + st.n * np.asarray(k)[None, :]
    out = NumpyTimeOps.add_steps(cut, k, _naive_step(st))
    return out.astype(cut.dtype)


def steps_between(npg, a, b):
    st = npg.step
    if st is None:
        raise ValueError("time axis has no regular step")
    if npg.tmeta.kind in ("int", "period"):
        return (b - a) // st.n
    return NumpyTimeOps.steps_between(a, b, _naive_step(st))


def time_as_float_days(npg, t):
    """Trend regressor: days since epoch (datetime), ordinals (period), ints."""
    if npg.tmeta.kind == "datetime":
        return t.astype("M8[ns]").view(np.int64) / 86400e9
    return t.astype(np.int64)


# =============================================================================
# small pandas objects only: cutoff index, fh <-> numpy, proba output index
# =============================================================================
def pandas_time_index(npg, t):
    """numpy time array -> pandas index, with freq (small arrays only)."""
    import pandas as pd

    tm = npg.tmeta
    freq = step_freqstr(npg)
    if tm.kind == "datetime":
        idx = pd.DatetimeIndex(t)
        if tm.tz is not None:
            idx = idx.tz_localize("UTC").tz_convert(tm.tz)
        if freq is not None and len(t) > 0:
            try:
                idx = pd.DatetimeIndex(idx, freq=freq)
            except ValueError:
                pass
        return idx
    tm2 = TimeMeta(tm.kind, unit=tm.unit, tz=tm.tz, freq=tm.freq, step=tm.step,
                   is_range=False)
    return time_to_pandas_index(np.asarray(t), tm2, freq_mode="public")


def cutoff_index(npg, reverse_order=False):
    """Global cutoff as the 1-element pandas Index BaseForecaster expects."""
    if npg.n_rows == 0:
        return None
    t = npg.time
    v = t.min() if reverse_order else t.max()
    return pandas_time_index(npg, np.asarray([v], dtype=t.dtype))


def fh_to_grid(npg, fh, cutoff):
    """ForecastingHorizon -> (K (U,H) int steps per instance, T (U,H) times).

    Relative fh: every instance steps from its own cutoff (= what vectorization
    over per-instance clones does). Absolute fh: same absolute times for every
    instance, K computed per instance. Only the H-element fh touches pandas.
    """
    cut = npg.inst_cutoffs()
    U = len(cut)
    if fh.is_relative:
        k = np.asarray(fh.to_pandas(), dtype=np.int64)
        return np.broadcast_to(k, (U, len(k))), add_steps(npg, cut, k)
    abs_idx = fh.to_pandas()
    if npg.tmeta.kind == "datetime":
        if getattr(abs_idx, "tz", None) is not None:
            abs_idx = abs_idx.tz_convert("UTC").tz_localize(None)
        T = np.asarray(abs_idx.values).astype(npg.time.dtype)
    elif npg.tmeta.kind == "period":
        T = np.asarray(abs_idx.asi8)
    else:
        T = np.asarray(abs_idx, dtype=npg.time.dtype)
    K = steps_between(npg, cut[:, None], T[None, :])
    return K, np.broadcast_to(T, (U, len(T)))


def pandas_row_index(npg):
    """Row index of a hub as pandas (for proba outputs, which are pandas types)."""
    import pandas as pd

    names = [_strip(k) for k in npg.key_names]
    tname = _strip(npg.time_name)
    tidx = pandas_time_index(npg, npg.time) if not npg.key_names else None
    if not npg.key_names:
        tidx.name = tname
        return tidx
    levels, codes = [], []
    for kv in npg.key_values:
        kc, ku = pd.factorize(kv)
        levels.append(pd.Index(ku))
        codes.append(np.repeat(kc, npg.lengths))
    tc, tu = pd.factorize(npg.time, sort=True)
    lv = pandas_time_index(npg, tu)
    lv = lv._with_freq(None) if hasattr(lv, "_with_freq") else lv
    levels.append(lv)
    codes.append(tc)
    return pd.MultiIndex(levels=levels, codes=codes, names=[*names, tname],
                         verify_integrity=False)


def _strip(name):
    if isinstance(name, str) and name.startswith(INDEX_PREFIX):
        return name[len(INDEX_PREFIX):]
    return name


# =============================================================================
# converters (registered into sktime.datatypes._convert.convert_dict)
# =============================================================================
def _pd_to_npg(scitype):
    def f(obj, store=None):
        import pandas as pd

        hint = None
        if not isinstance(obj.index, pd.MultiIndex):
            hint = getattr(obj.index, "freq", None)
        return _wrap(from_pandas(obj), scitype, freq_hint=hint)

    return f


def _pl_to_npg(scitype):
    return lambda obj, store=None: _wrap(from_polars(obj), scitype)


def _pa_to_npg(scitype):
    return lambda obj, store=None: _wrap(from_arrow(obj), scitype)


def _pandas_meta(npg, series):
    return {"source": "pandas", "container": "series" if series else "frame",
            "layout": "multiindex" if npg.key_names else "index",
            "series_name": npg.columns[0] if series else None,
            "columns_order": list(npg.columns), "col_dtypes": None, "perm": None}


def _npg_to_pd(target):
    def f(obj, store=None):
        COUNTS["egress"] += 1
        src = obj.meta.get("source")
        if src != "pandas":
            obj = obj.with_rows(obj.indptr, obj.time, obj.values)
            obj.meta = _pandas_meta(obj, target == "pd.Series")
            obj.key_names = [_strip(k) for k in obj.key_names]
            obj.time_name = _strip(obj.time_name)
        elif target == "pd.Series" and obj.meta.get("container") != "series":
            obj = obj.with_rows(obj.indptr, obj.time, obj.values)
            obj.meta["container"] = "series"
            obj.meta["series_name"] = obj.columns[0]
        elif target == "pd.DataFrame" and obj.meta.get("container") == "series":
            obj = obj.with_rows(obj.indptr, obj.time, obj.values)
            obj.meta["container"] = "frame"
        return to_pandas(obj)

    return f


def _idx_names(npg):
    ks = [k if str(k).startswith(INDEX_PREFIX) else f"{INDEX_PREFIX}{k}"
          for k in npg.key_names]
    t = npg.time_name
    t = t if str(t).startswith(INDEX_PREFIX) else f"{INDEX_PREFIX}{t}"
    return ks, t


def _npg_to_pl(obj, store=None):
    COUNTS["egress"] += 1
    if obj.meta.get("source") != "polars":
        ks, t = _idx_names(obj)
        obj = obj.with_rows(obj.indptr, obj.time, obj.values)
        obj.meta = {"source": "polars", "lazy": False, "schema": {}, "key_cols": ks,
                    "time_col": t, "columns_order": [*ks, t, *obj.columns],
                    "null_cols": [], "perm": None}
    return to_polars(obj)


def _npg_to_pa(obj, store=None):
    COUNTS["egress"] += 1
    if obj.meta.get("source") != "arrow":
        import pyarrow as pa

        ks, t = _idx_names(obj)
        arrays = [pa.array(np.repeat(kv, obj.lengths)) for kv in obj.key_values]
        arrays.append(pa.array(obj.time))
        arrays += [pa.array(np.ascontiguousarray(obj.values[:, j]))
                   for j in range(len(obj.columns))]
        return pa.Table.from_arrays(arrays, names=[*ks, t, *map(str, obj.columns)])
    return to_arrow(obj)


def _identity(obj, store=None):
    return obj


def _chain(f, g):
    return lambda obj, store=None: g(f(obj))


# =============================================================================
# mtype (checker) classes
# =============================================================================
def _metadata(npg_like, return_metadata, scitype):
    """Cheap metadata for an NpGrouped: O(n_instances) + one isnan pass."""
    from sktime.datatypes._base._common import _req
    from sktime.datatypes._dtypekind import DtypeKind

    v, ip = npg_like.values, npg_like.indptr
    lens = np.diff(ip)
    md = {}
    nf = v.shape[1]
    md["is_univariate"] = nf == 1
    md["n_features"] = nf
    md["feature_names"] = list(npg_like.columns)
    md["dtypekind_dfip"] = [DtypeKind.FLOAT] * nf
    md["feature_kind"] = [DtypeKind.FLOAT] * nf
    md["is_empty"] = len(v) == 0 or nf == 0
    if _req("has_nans", return_metadata):
        md["has_nans"] = bool(np.isnan(v).any())
    md["is_equally_spaced"] = npg_like.step is not None
    if scitype != "Series":
        md["n_instances"] = len(lens)
        md["is_one_series"] = len(lens) == 1
        md["is_equal_length"] = bool((lens == lens[0]).all()) if len(lens) else True
    if scitype == "Hierarchical":
        top = npg_like.key_values[:-1]
        npan = len(set(zip(*top))) if top else 1
        md["n_panels"] = npan
        md["is_one_panel"] = npan == 1
    return md


def _make_mtype_classes():
    from sktime.datatypes._base._common import _ret
    from sktime.datatypes._hierarchical._base import ScitypeHierarchical
    from sktime.datatypes._panel._base import ScitypePanel
    from sktime.datatypes._series._base import ScitypeSeries

    bases = {"Series": ScitypeSeries, "Panel": ScitypePanel,
             "Hierarchical": ScitypeHierarchical}
    out = []
    for sc, base in bases.items():

        def _check_npg(self, obj, return_metadata=False, var_name="obj", _sc=sc):
            if obj.scitype != _sc:
                return _ret(False, f"{var_name} is np_grouped of scitype "
                            f"{obj.scitype}, not {_sc}", None, return_metadata)
            return _ret(True, None, _metadata(obj, return_metadata, _sc),
                        return_metadata)

        tags = dict(base._tags)
        tags.update({"scitype": sc, "name": NPG[sc], "name_python": NPG[sc],
                     "description": f"g2 proto numpy grouped {sc}",
                     "python_type": "NpGrouped", "python_dependencies": None,
                     "capability:multivariate": True,
                     "capability:unequally_spaced": True,
                     "capability:missing_values": True})
        out.append(type(f"G2NpGrouped{sc}", (base,),
                        {"_tags": tags, "_check": _check_npg,
                         "__module__": __name__}))

        def _check_pa(self, obj, return_metadata=False, var_name="obj", _sc=sc):
            names = obj.column_names
            idx = [c for c in names if c.startswith(INDEX_PREFIX)]
            nkeys = {"Series": 0, "Panel": 1}.get(_sc)
            ok = len(idx) >= 1 and (len(idx) - 1 == nkeys if nkeys is not None
                                    else len(idx) - 1 >= 2)
            if not ok:
                return _ret(False, f"{var_name}: pyarrow table without the "
                            f"__index__ columns of a {_sc}", None, return_metadata)
            import pyarrow as pa

            vals = [c for c in names if c not in idx]
            if not all(pa.types.is_floating(obj.schema.field(c).type)
                       or pa.types.is_integer(obj.schema.field(c).type) for c in vals):
                return _ret(False, f"{var_name}: non-numeric value columns", None,
                            return_metadata)
            if not _req_any(return_metadata):
                return _ret(True, None, {}, return_metadata)
            npg = _wrap(from_arrow(obj), _sc)
            COUNTS["ingress"] -= 1  # metadata probe, not a conversion
            return _ret(True, None, _metadata(npg, return_metadata, _sc),
                        return_metadata)

        tags = dict(base._tags)
        tags.update({"scitype": sc, "name": PA[sc], "name_python": PA[sc],
                     "description": f"g2 proto pyarrow.Table {sc} (__index__ cols)",
                     "python_type": "pyarrow.Table",
                     "python_dependencies": "pyarrow",
                     "capability:multivariate": True,
                     "capability:unequally_spaced": True,
                     "capability:missing_values": True})
        out.append(type(f"G2PyArrow{sc}", (base,),
                        {"_tags": tags, "_check": _check_pa,
                         "__module__": __name__}))
    return out


def _req_any(return_metadata):
    return not isinstance(return_metadata, bool) or return_metadata


# =============================================================================
# registration from outside sktime (the only "framework" change in this proto)
# =============================================================================
_REGISTERED = {}


def register_np_grouped(pyarrow_to_pandas=True):
    """Extend sktime's datatypes registry + 2 dispatch points, from the outside.

    1. mtype classes: appended to the lru_cached class lists that
       ``sktime.datatypes._registry._generate_mtype_cls_list`` returns (the registry
       is discovered by crawling the ``sktime.datatypes`` package; there is no
       public plugin hook), then dependent caches are cleared
       (``_generate_mtype_register``, ``_generate_mtype_register_subset``,
       ``generate_check_dict``).
    2. converters: inserted into the plain module dict
       ``sktime.datatypes._convert.convert_dict`` (public-ish, no cache).
    3. ``get_cutoff`` and ``update_data`` as imported into
       ``sktime.forecasting.base._base``: wrapped to dispatch ``NpGrouped``
       (otherwise ``get_cutoff`` converts any unknown mtype to pandas).
    """
    if _REGISTERED:
        return _REGISTERED
    import sktime.datatypes._check as chk
    import sktime.datatypes._convert as cv
    import sktime.datatypes._registry as reg
    import sktime.forecasting.base._base as fb

    classes = _make_mtype_classes()
    for sd in ("present", "all", "exclude"):
        lst = reg._generate_mtype_cls_list(soft_deps=sd)
        lst.extend(c for c in classes
                   if sd != "exclude" or c.get_class_tag("python_dependencies")
                   is None)
    reg._generate_mtype_register.cache_clear()
    reg._generate_mtype_register_subset.cache_clear()
    chk.generate_check_dict.cache_clear()

    cd = cv.convert_dict
    n_conv = 0
    for sc in ("Series", "Panel", "Hierarchical"):
        npg = NPG[sc]
        cd[(npg, npg, sc)] = _identity
        cd[(PA[sc], PA[sc], sc)] = _identity
        for m in PANDAS[sc]:
            cd[(m, npg, sc)] = _pd_to_npg(sc)
            cd[(npg, m, sc)] = _npg_to_pd(m)
            n_conv += 2
        for m in POLARS[sc]:
            cd[(m, npg, sc)] = _pl_to_npg(sc)
            cd[(npg, m, sc)] = _npg_to_pl
            n_conv += 2
        cd[(PA[sc], npg, sc)] = _pa_to_npg(sc)
        cd[(npg, PA[sc], sc)] = _npg_to_pa
        n_conv += 2
        if pyarrow_to_pandas:  # lets *stock* forecasters accept pyarrow input too
            for m in PANDAS[sc]:
                cd[(PA[sc], m, sc)] = _chain(_pa_to_npg(sc), _npg_to_pd(m))
                cd[(m, PA[sc], sc)] = _chain(_pd_to_npg(sc), _npg_to_pa)
                n_conv += 2

    orig_get_cutoff, orig_update = fb.get_cutoff, fb.update_data

    def get_cutoff(obj, cutoff=0, return_index=False, reverse_order=False, **kw):
        if isinstance(obj, NpGrouped):
            idx = cutoff_index(obj, reverse_order=reverse_order)
            if idx is None:
                return cutoff
            return idx if return_index else idx[0]
        return orig_get_cutoff(obj, cutoff, return_index=return_index,
                               reverse_order=reverse_order, **kw)

    def update_data(X, X_new=None):
        if isinstance(X, NpGrouped) or isinstance(X_new, NpGrouped):
            return npg_append(X, X_new)
        return orig_update(X, X_new)

    fb.get_cutoff, fb.update_data = get_cutoff, update_data
    _REGISTERED.update(classes=classes, n_converters=n_conv)
    return _REGISTERED


def npg_append(old, new):
    """update_data for NpGrouped: append new rows per instance (time-ordered)."""
    if old is None:
        return new
    if new is None:
        return old
    ko = list(zip(*old.key_values)) if old.key_values else [()]
    kn = list(zip(*new.key_values)) if new.key_values else [()]
    pos = {k: i for i, k in enumerate(ko)}
    keys = list(ko)
    for k in kn:
        if k not in pos:
            pos[k] = len(keys)
            keys.append(k)
    inst_old = np.repeat(np.arange(len(ko)), old.lengths)
    inst_new = np.repeat(np.array([pos[k] for k in kn]), new.lengths)
    inst = np.concatenate([inst_old, inst_new])
    order = np.argsort(inst, kind="stable")
    time = np.concatenate([old.time, new.time.astype(old.time.dtype)])[order]
    vals = np.concatenate([old.values, new.values])[order]
    inst = inst[order]
    counts = np.bincount(inst, minlength=len(keys))
    indptr = np.zeros(len(keys) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    d = np.diff(time)[_within_mask(indptr, len(time))]
    if len(d) and not (d > np.zeros(1, dtype=d.dtype)).all():
        raise NotImplementedError("np_grouped update with overlapping/unsorted times")
    out = old.with_rows(indptr, time, vals)
    out.key_values = [np.array([k[j] for k in keys], dtype=object if not len(keys)
                               else np.asarray(old.key_values[j]).dtype)
                      for j in range(len(old.key_values))]
    out.step = infer_step(out)
    return out


__all__ = ["NpGrouped", "register_np_grouped", "NPG_MTYPES", "fh_to_grid",
           "cutoff_index", "pandas_row_index", "time_as_float_days", "COUNTS", "NS"]
