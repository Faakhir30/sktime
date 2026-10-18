"""a2_trace: dynamic census of the pandas API that forecasters call on their inputs.

For each runnable forecaster (soft deps present), we fit/predict/(predict_interval)
/update/predict on a small monthly series, and replace the y / X that reach the
inner methods (``_fit``, ``_predict``, ``_update``, ``_predict_interval``,
``_predict_quantiles``, ``_predict_var``, ``_predict_proba``) by *instrumented*
pandas subclasses. Every attribute access / special-method call on those
objects (and on pandas objects derived from them via ``_constructor``) is
recorded with the calling file, if the caller is not pandas/numpy itself.

Callers are attributed to:
    class      -> a module defining a class in the forecaster's own MRO chain
    framework  -> sktime/forecasting/base/_base.py, _fh.py, sktime/datatypes,
                  sktime/utils (conversion, update_data, checks)
    sktime     -> any other sktime module (e.g. a child forecaster/transformer)
    thirdparty -> any non-sktime package (statsmodels, pmdarima, ...)

This is ground truth for "which pandas API does this forecaster rely on, on its
input data", complementary to the static AST census (a2_census.py) which also
sees code paths not exercised here but cannot know receiver types.

Usage (from worktree root)::

    PYTHONPATH=.:experiments/container_independence <python> -W ignore \
        experiments/container_independence/scripts/a2_trace.py [--names A,B] \
        [--out results/a2_trace_<tag>.json] [--n 48] [--timeout 90]
"""

import argparse
import gc
import json
import os
import signal
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from a2_census import ATTR2CAT, HARD_CATS  # noqa: E402

PD_DIR = os.path.dirname(pd.__file__)
NP_DIR = os.path.dirname(np.__file__)
SELF_FILE = str(HERE)


class _T:
    active = 0
    rec = None  # Counter of (attr, file, line)
    allowed = None  # None = record only; else set of allowed names (enforce)
    chain_files = set()


class PandasAPIViolation(BaseException):
    """Raised (enforce mode) when the forecaster's own code uses a pandas API
    outside the allowed "values + time-index metadata" surface.

    BaseException so that ``except Exception`` fallbacks in estimator code do
    not swallow it.
    """


# surface a "numpy values + time-index metadata" container would offer
ALLOWED = {
    "strict": {
        "to_numpy", "values", "__array__", "index", "columns", "name", "shape",
        "ndim", "dtype", "dtypes", "size", "__len__", "copy", "empty",
    },
}
ALLOWED["numpylike"] = ALLOWED["strict"] | {
    "iloc", "iat", "__getitem__", "__setitem__", "__iter__", "__arith__",
    "__cmp__", "__array_ufunc__", "mean", "sum", "std", "var", "min", "max",
    "abs", "round", "clip", "cumsum", "cumprod", "isna", "notna", "isnull",
    "notnull", "astype", "any", "all", "median", "quantile", "T", "tolist",
    "to_list", "item", "squeeze", "flatten", "reshape",
}
# numpylike + elementwise power (VAR/VARMAX/VECM hit .pow first in numpylike)
ALLOWED["numpylike_pow"] = ALLOWED["numpylike"] | {"pow"}


def _skip(fn):
    return fn.startswith(PD_DIR) or fn.startswith(NP_DIR) or fn == SELF_FILE


def _record(name, depth=2, walk=False):
    if not _T.active or _T.rec is None:
        return
    fr = sys._getframe(depth)
    if walk:
        while fr is not None and _skip(fr.f_code.co_filename):
            fr = fr.f_back
        if fr is None:
            return
    fn = fr.f_code.co_filename
    if _skip(fn):
        return
    _T.rec[(name, fn, fr.f_lineno)] += 1
    if _T.allowed is not None and fn in _T.chain_files and name not in _T.allowed:
        raise PandasAPIViolation(name, fn, fr.f_lineno)


class _TraceMixin:
    def __getattribute__(self, name):
        if _T.active and not (name.startswith("__") and name.endswith("__")):
            _record(name)
        return object.__getattribute__(self, name)

    def __getitem__(self, key):
        _record("__getitem__")
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        _record("__setitem__")
        return super().__setitem__(key, value)

    def __array__(self, *args, **kwargs):
        _record("__array__")
        return super().__array__(*args, **kwargs)

    def __array_ufunc__(self, *args, **kwargs):
        _record("__array_ufunc__", walk=True)
        return super().__array_ufunc__(*args, **kwargs)

    def __len__(self):
        _record("__len__")
        return super().__len__()

    def __iter__(self):
        _record("__iter__")
        return super().__iter__()

    def _arith_method(self, other, op):
        _record("__arith__", walk=True)
        return super()._arith_method(other, op)

    def _cmp_method(self, other, op):
        _record("__cmp__", walk=True)
        return super()._cmp_method(other, op)


class TSeries(_TraceMixin, pd.Series):
    @property
    def _constructor(self):
        return TSeries

    @property
    def _constructor_expanddim(self):
        return TFrame


class TFrame(_TraceMixin, pd.DataFrame):
    @property
    def _constructor(self):
        return TFrame

    @property
    def _constructor_sliced(self):
        return TSeries


# sktime's mtype checks require type(obj).__module__ to start with "pandas"
# (sktime/datatypes/_base/_base.py:217); masquerade so the framework and child
# estimators accept the traced objects.
TSeries.__module__ = "pandas.traced"
TFrame.__module__ = "pandas.traced"
_TraceMixin.__module__ = "pandas.traced"


def wrap(obj):
    if obj is None or isinstance(obj, _TraceMixin):
        return obj
    if isinstance(obj, pd.Series):
        return TSeries(obj)
    if isinstance(obj, pd.DataFrame):
        return TFrame(obj)
    return obj


INNER = [
    "_fit",
    "_predict",
    "_update",
    "_predict_interval",
    "_predict_quantiles",
    "_predict_var",
    "_predict_proba",
]


def instrument(est):
    """Wrap inner methods on the instance so their y/X inputs are traced."""

    def mk(name, orig):
        def w(*args, **kwargs):
            args = [wrap(a) if isinstance(a, (pd.Series, pd.DataFrame)) else a for a in args]
            for k in ("y", "X"):
                if k in kwargs:
                    kwargs[k] = wrap(kwargs[k])
            for attr in ("_y", "_X"):
                v = est.__dict__.get(attr)
                if isinstance(v, (pd.Series, pd.DataFrame)):
                    est.__dict__[attr] = wrap(v)
            _T.active += 1
            try:
                return orig(*args, **kwargs)
            finally:
                _T.active -= 1

        return w

    for m in INNER:
        if hasattr(est, m):
            est.__dict__[m] = mk(m, getattr(est, m))


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def make_y(n, ncols=1, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000-01", periods=n, freq="M")
    vals = 10 + np.abs(rng.normal(size=(n, ncols)).cumsum(axis=0)) + 1.0
    if ncols == 1:
        return pd.Series(vals[:, 0], index=idx, name="y")
    return pd.DataFrame(vals, index=idx, columns=[f"y{i}" for i in range(ncols)])


def classify_caller(fn, chain_files):
    if fn in chain_files:
        return "class"
    if "/sktime/" in fn:
        rel = fn.rsplit("/sktime/", 1)[1]
        if rel.startswith(("forecasting/base/_base.py", "forecasting/base/_fh.py")) or rel.startswith(
            ("datatypes/", "utils/", "base/")
        ):
            return "framework"
        return "sktime"
    return "thirdparty"


def attr_cat(name):
    if name in ("__array__", "__array_ufunc__"):
        return "to_numpy"
    if name in ("__getitem__", "__setitem__"):
        return "getitem"
    if name in ("__arith__", "__cmp__"):
        return "arith"
    if name in ("__len__", "__iter__"):
        return "generic"
    return ATTR2CAT.get(name, "generic")


def run_one(cls, n, timeout, mode="record"):
    from sktime.forecasting.base import BaseForecaster

    chain = [k for k in cls.__mro__ if k not in set(BaseForecaster.__mro__)]
    chain_files = set()
    import inspect

    for k in chain:
        try:
            chain_files.add(inspect.getsourcefile(k))
        except TypeError:
            pass

    _T.chain_files = chain_files
    _T.allowed = ALLOWED.get(mode)

    params = cls.get_test_params()
    if isinstance(params, list):
        params = params[0]
    est = cls(**params)
    multiv = est.get_tag("capability:multivariate")
    out = {"name": cls.__name__, "status": "ok", "steps": []}
    _T.rec = Counter()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    t0 = time.time()
    # line coverage of sktime code (excluding datatypes), to tell "static op not
    # executed in this run" apart from "executed but receiver not traced"
    import sktime as _sk

    sk_root = os.path.dirname(_sk.__file__) + os.sep
    sk_skip = sk_root + "datatypes" + os.sep
    executed = defaultdict(set)

    def _local(frame, event, arg):
        if event == "line":
            executed[frame.f_code.co_filename].add(frame.f_lineno)
        return _local

    def _global(frame, event, arg):
        fn = frame.f_code.co_filename
        if fn.startswith(sk_root) and not fn.startswith(sk_skip):
            return _local
        return None

    sys.settrace(_global)
    try:
        y = make_y(n)
        instrument(est)
        try:
            est.fit(y, fh=[1, 2, 3])
            y_new = make_y(n + 3).iloc[-3:]
        except Exception:  # noqa: BLE001  - multivariate-only estimators
            if not multiv:
                raise
            est = cls(**params)
            instrument(est)
            y = make_y(n, ncols=2)
            est.fit(y, fh=[1, 2, 3])
            y_new = make_y(n + 3, ncols=2).iloc[-3:]
        out["steps"].append("fit")
        est.predict()
        out["steps"].append("predict")
        if est.get_tag("capability:pred_int"):
            try:
                est.predict_interval(coverage=0.9)
                out["steps"].append("predict_interval")
            except Exception as e:  # noqa: BLE001
                out["pi_error"] = f"{type(e).__name__}: {e}"[:200]
        try:
            est.update(y_new)
            est.predict()
            out["steps"].append("update+predict")
        except Exception as e:  # noqa: BLE001
            out["update_error"] = f"{type(e).__name__}: {e}"[:200]
    except PandasAPIViolation as v:
        out["status"] = "violation"
        name, fn, line = v.args
        out["violation"] = {
            "attr": name,
            "site": f"{fn.rsplit('/sktime/', 1)[-1]}:{line}",
            "step": len(out["steps"]),
        }
    except Timeout:
        out["status"] = "timeout"
    except Exception as e:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"[:300]
        out["tb"] = traceback.format_exc()[-1500:]
    finally:
        sys.settrace(None)
        signal.alarm(0)
        _T.active = 0
        _T.allowed = None
    out["seconds"] = round(time.time() - t0, 2)
    out["executed"] = {
        fn.rsplit("/sktime/", 1)[1]: sorted(lines) for fn, lines in executed.items()
    }

    agg = defaultdict(Counter)  # caller_kind -> Counter(attr)
    where = defaultdict(set)  # (kind, attr) -> set(file:line)
    for (name, fn, line), c in _T.rec.items():
        kind = classify_caller(fn, chain_files)
        agg[kind][name] += c
        rel = fn.rsplit("/sktime/", 1)[1] if "/sktime/" in fn else fn.split("site-packages/")[-1]
        where[(kind, name)].add(f"{rel}:{line}")
    out["attrs"] = {k: dict(v) for k, v in agg.items()}
    out["where"] = {
        f"{k}|{a}": sorted(s)[:5]
        for (k, a), s in where.items()
        if attr_cat(a) in HARD_CATS or a in ("__getitem__", "loc", "iloc")
    }
    cats = defaultdict(Counter)
    for kind, cnt in agg.items():
        for a, c in cnt.items():
            cats[kind][attr_cat(a)] += c
    out["cats"] = {k: dict(v) for k, v in cats.items()}
    own = Counter()
    for kind in ("class", "sktime"):
        own.update(cats.get(kind, {}))
    hard_own = sum(own.get(c, 0) for c in HARD_CATS)
    out["dyn_style"] = (
        "pandas-semantic-input"
        if hard_own > 0
        else "numpy-boundary"
        if own.get("to_numpy", 0) > 0
        else "no-input-pandas-use"
        if sum(own.values()) == 0
        else "light-pandas"
    )
    tp = cats.get("thirdparty", {})
    out["thirdparty_hard"] = sum(tp.get(c, 0) for c in HARD_CATS)
    out["thirdparty_any"] = sum(tp.values())
    _T.rec = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])  # unused
    ap.add_argument("--list-runnable", action="store_true")
    ap.add_argument("--mode", default="record", choices=["record", "strict", "numpylike", "numpylike_pow"])
    args = ap.parse_args()

    from sktime.registry import all_estimators
    from sktime.utils.dependencies import _check_estimator_deps

    ests = dict(all_estimators("forecaster", return_names=True))
    runnable = [n for n, c in ests.items() if _check_estimator_deps(c, severity="none")]
    if args.list_runnable:
        print(json.dumps(runnable))
        return
    names = args.names.split(",") if args.names else runnable
    import psutil

    proc = psutil.Process()
    results = []
    for nm in names:
        if nm not in runnable:
            results.append({"name": nm, "status": "deps_missing"})
            continue
        r = run_one(ests[nm], args.n, args.timeout, mode=args.mode)
        r["rss_mb"] = round(proc.memory_info().rss / 2**20, 1)
        results.append(r)
        print(
            f"{nm:45s} {r['status']:8s} {r.get('dyn_style', ''):24s} "
            f"steps={'/'.join(r.get('steps', []))} {r['seconds']}s rss={r['rss_mb']}MB "
            f"{r.get('error', '')[:100]} {r.get('violation', '')}",
            flush=True,
        )
        gc.collect()
    out = Path(args.out) if args.out else HERE.parents[1] / "results" / "a2_trace.json"
    meta = {
        "python": sys.executable,
        "versions": {m: getattr(__import__(m), "__version__", None) for m in ["pandas", "numpy"]},
        "n": args.n,
        "mode": args.mode,
        "allowed": sorted(ALLOWED.get(args.mode, [])),
    }
    for m in ["statsmodels", "pmdarima", "statsforecast", "arch", "tbats", "skforecast"]:
        try:
            meta["versions"][m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            meta["versions"][m] = None
    out.write_text(json.dumps({"meta": meta, "results": results}, indent=1, default=str))


if __name__ == "__main__":
    main()
