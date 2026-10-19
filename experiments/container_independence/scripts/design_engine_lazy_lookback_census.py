"""design_engine_lazy: empirical fit-lookback census (critic gaps G9/G14).

For each registered forecaster whose deps import, fit on a full univariate series y
(len T) and on y.tail(k) for a ladder of k, predict fh=1..3, and record the smallest
k at which predictions equal the full-history fit (rtol 1e-9). Also records the
smallest k for predict_interval where supported.

This is an *empirical* lookback: equality on 2 seeds/two signals is evidence, not a proof.
A forecaster that is equal at k for both seeds is classed "finite(k)"; otherwise "all".

Run (from WT root):
  PYTHONPATH=.:experiments/container_independence $PY -W ignore \
     experiments/container_independence/scripts/design_engine_lazy_lookback_census.py
"""

import json
import os
import signal
import sys
import time
import traceback
import warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sktime.registry import all_estimators

T = 96
LADDER = [1, 2, 3, 4, 7, 8, 12, 13, 14, 24, 25, 36, 48, 72]
FH = [1, 2, 3]
OUT = "experiments/container_independence/results/design_engine_lazy_lookback_census.json"
PER_EST_TIMEOUT = 40


class _TO(Exception):
    pass


def _alarm(*a):
    raise _TO()


signal.signal(signal.SIGALRM, _alarm)


def make_y(seed):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    vals = 50 + 0.3 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, T)
    idx = pd.period_range("2000-01", periods=T, freq="M")
    return pd.Series(vals, index=idx, name="y")


def eq(a, b):
    try:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.shape != b.shape:
            return False
        return bool(np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True))
    except Exception:
        return False


def run_one(cls, params, y, k, method):
    est = cls(**params)
    yy = y if k is None else y.iloc[-k:]
    est.fit(yy, fh=FH)
    if method == "predict":
        return est.predict().values
    return est.predict_interval(coverage=0.8).values


def min_k(cls, params, y, method):
    ref = run_one(cls, params, y, None, method)
    for k in LADDER:
        if k >= T:
            break
        try:
            out = run_one(cls, params, y, k, method)
        except _TO:
            raise
        except Exception:
            continue
        if eq(out, ref):
            return k
    return None  # needs all (within ladder)


def main():
    ests = all_estimators("forecaster", return_names=True)
    rows = []
    t_all = time.time()
    for name, cls in ests:
        row = {"name": name}
        try:
            tags = cls.get_class_tags()
            row["y_inner_mtype"] = str(tags.get("y_inner_mtype"))
            row["requires_fh_in_fit"] = tags.get("requires-fh-in-fit")
            row["pred_int"] = tags.get("capability:pred_int")
            row["univariate_ok"] = tags.get("scitype:y") in ("univariate", "both")
            try:
                plist = cls.get_test_params()
            except Exception as e:
                row["status"] = "no_test_params"
                rows.append(row)
                continue
            params = plist[0] if isinstance(plist, list) else plist
            row["params"] = {k: repr(v)[:60] for k, v in params.items()}
            row["has_window_length"] = "window_length" in cls.__init__.__code__.co_varnames
            signal.alarm(PER_EST_TIMEOUT)
            t0 = time.time()
            ks = []
            for seed in (0, 1):
                y = make_y(seed)
                ks.append(min_k(cls, params, y, "predict"))
            row["k_predict_seeds"] = ks
            row["k_predict"] = (
                None if any(k is None for k in ks) else max(ks)
            )
            if row["pred_int"]:
                try:
                    kis = [min_k(cls, params, make_y(s), "interval") for s in (0, 1)]
                    row["k_interval"] = None if any(k is None for k in kis) else max(kis)
                except _TO:
                    raise
                except Exception as e:
                    row["k_interval"] = "error"
            signal.alarm(0)
            row["status"] = "ok"
            row["secs"] = round(time.time() - t0, 2)
        except _TO:
            row["status"] = "timeout"
        except Exception as e:
            signal.alarm(0)
            msg = f"{type(e).__name__}: {str(e)[:120]}"
            row["status"] = "error"
            row["err"] = msg
        finally:
            signal.alarm(0)
        rows.append(row)
        print(name, row.get("status"), row.get("k_predict"), row.get("k_interval"),
              row.get("err", ""), flush=True)
    ok = [r for r in rows if r["status"] == "ok"]
    finite = [r for r in ok if r["k_predict"] is not None]
    summ = {
        "n_registered": len(rows),
        "n_runnable": len(ok),
        "n_finite_predict_lookback": len(finite),
        "finite_hist": {},
        "n_timeout": sum(r["status"] == "timeout" for r in rows),
        "n_error": sum(r["status"] == "error" for r in rows),
        "wall_s": round(time.time() - t_all, 1),
        "T": T,
        "ladder": LADDER,
    }
    for r in finite:
        summ["finite_hist"][str(r["k_predict"])] = summ["finite_hist"].get(str(r["k_predict"]), 0) + 1
    pi = [r for r in ok if r.get("pred_int") and r.get("k_interval") not in (None, "error")]
    summ["n_pred_int_runnable"] = sum(1 for r in ok if r.get("pred_int"))
    summ["n_finite_interval_lookback"] = len(pi)
    print(json.dumps(summ, indent=1))
    with open(OUT, "w") as f:
        json.dump({"summary": summ, "rows": rows}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
