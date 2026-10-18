"""x9 verifier: census of ALL installable sktime forecasters on Arrow-valued pandas y
(the "hybrid" layout: numpy DatetimeIndex + double[pyarrow] values), vs a numpy
baseline. Tests generalisation of the 13/14 claim beyond the hand-picked list.

For each forecaster whose soft deps are importable: create_test_instance(), fit on
a univariate daily pd.Series (numpy) and the same values as double[pyarrow],
fh=[1,2,3], predict; record OK/ERR, and whether predictions match numpy.

Run (statsmodels from a --target dir, as in x9_arrow_compat)::
    PYTHONPATH=.:experiments/container_independence:$SM $EXP_PY -W ignore \
        experiments/container_independence/scripts/x9_cheap_paths_verify0_census.py
"""
import collections
import json
import os
import signal
import traceback
import warnings

warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sktime.registry import all_estimators  # noqa: E402
from sktime.utils.dependencies import _check_estimator_deps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

idx = pd.date_range("2000-01-01", periods=60, freq="D")
vals = 10 + np.random.default_rng(0).normal(size=60).cumsum()
vals = vals - vals.min() + 1.0  # positive (some forecasters need it)
y_np = pd.Series(vals, index=idx, name="y")
y_ar = pd.Series(pd.array(vals, dtype="double[pyarrow]"), index=idx, name="y")
FH = [1, 2, 3]


class TO(Exception):
    pass


def _alarm(*a):
    raise TO("timeout")


signal.signal(signal.SIGALRM, _alarm)


def run(est, y):
    signal.alarm(20)
    try:
        e = est.clone()
        e.fit(y, fh=FH)
        return e.predict()
    finally:
        signal.alarm(0)


def where(e):
    tb = traceback.extract_tb(e.__traceback__)
    sk = [f for f in tb if "/sktime/" in f.filename and "experiments" not in f.filename]
    f = sk[-1] if sk else (tb[-1] if tb else None)
    return f"{f.filename.split('sktime/')[-1]}:{f.lineno}" if f else ""


res = {}
for name, cls in all_estimators("forecaster"):
    if not _check_estimator_deps(cls, severity="none"):
        continue
    if cls.get_class_tag("scitype:y", "univariate") == "multivariate":
        continue
    if cls.get_class_tag("requires-fh-in-fit", False) is None:
        pass
    try:
        est = cls.create_test_instance()
    except Exception as e:  # noqa: BLE001
        res[name] = {"status": "skip_instance", "err": str(e)[:100]}
        continue
    try:
        p0 = run(est, y_np)
    except BaseException as e:  # noqa: BLE001
        res[name] = {"status": "skip_baseline_fails", "err": f"{type(e).__name__}"}
        continue
    try:
        p1 = run(est, y_ar)
    except BaseException as e:  # noqa: BLE001
        res[name] = {"status": "ERR", "err": f"{type(e).__name__}: {str(e)[:120]}", "at": where(e)}
        continue
    try:
        v0 = np.asarray(p0, dtype="float64")
        v1 = np.asarray(pd.DataFrame(p1).astype("float64").to_numpy(), dtype="float64").reshape(v0.shape)
        match = bool(np.allclose(v0, v1, equal_nan=True, rtol=1e-6, atol=1e-8))
    except Exception as e:  # noqa: BLE001
        match = f"cmp-error {type(e).__name__}"
    dt = str(getattr(p1, "dtype", None) or list(map(str, p1.dtypes)))
    res[name] = {"status": "OK" if match is True else "OK_MISMATCH", "out_dtype": dt,
                 "arrow_kept": "pyarrow" in dt}

cnt = collections.Counter(r["status"] for r in res.values())
print("pandas", pd.__version__, "| status counts:", dict(cnt))
tested = [k for k, r in res.items() if r["status"] in ("OK", "ERR", "OK_MISMATCH")]
print(f"tested (baseline OK): {len(tested)}; OK {cnt['OK']}, ERR {cnt['ERR']}, mismatch {cnt['OK_MISMATCH']}")
kept = sum(1 for k in tested if res[k].get("arrow_kept"))
print(f"outputs still Arrow among OK: {kept}/{cnt['OK'] + cnt['OK_MISMATCH']}")
errs = collections.Counter((r["err"][:70], r.get("at")) for r in res.values() if r["status"] == "ERR")
for (e, at), n in errs.most_common(25):
    print(f"  {n:3d}x {e} @ {at}")
for k, r in res.items():
    if r["status"] in ("ERR", "OK_MISMATCH"):
        print("   ", r["status"], k)
with open(os.path.join(RES, f"x9_cheap_paths_verify0_census_pd{pd.__version__[0]}.json"), "w") as fh:
    json.dump(res, fh, indent=1)
