"""design_minimal: measure the "trusted nested transformer" lever (g1 §3.3, unmeasured).

Stack levels, interleaved in one process, fit(y, fh=[1,2,3]) + predict(), NaiveForecaster("last"):
  S0  stock
  S1  g1 trusted outer path (protos/g1_trusted.install)
  S2  S1 + trusted nested-transformer entry for Lag (skips BaseTransformer check/convert,
      calls _fit/_transform directly when the input already is the inner pandas type)
Also: predict-only timing (fitted once per level) to isolate the predict-side lever.
Equality: assert_frame_equal(check_exact=True) of predictions, plus cutoff equality.

Usage: PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
       $EXP_PY -W ignore experiments/container_independence/scripts/design_minimal_nested.py --sizes 300x100,1000x100
"""

import argparse
import json
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from common import bench, make_panel_fast  # noqa: E402

import sktime.transformations.lag as lagmod  # noqa: E402
from sktime.datatypes import convert_to  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402

_StockLag = lagmod.Lag
NSTATS = {"trusted": 0, "fallback": 0}


class TrustedLag(_StockLag):
    """Lag whose fit_transform takes a trusted entry when X is already inner-typed pandas."""

    def fit_transform(self, X, y=None):
        if y is not None or type(X) not in (pd.Series, pd.DataFrame) or isinstance(
            X.index, pd.MultiIndex
        ):
            NSTATS["fallback"] += 1
            return super().fit_transform(X, y)
        NSTATS["trusted"] += 1
        was_series = type(X) is pd.Series
        Xf = X.to_frame() if was_series else X
        if self.get_tag("remember_data", False):
            self._X = Xf
        self._fit(Xf)
        self._is_fitted = True
        Xt = self._transform(Xf)
        if was_series and Xt.shape[1] == 1:
            Xt = Xt.iloc[:, 0]
            Xt.name = X.name
        return Xt


import g3_stack  # noqa: E402

# level ids: 0=C0 stock, 1=C1 (mono+known mtype), 2=C2 (+g1 trusted), 3=C3 (+fh memo/cutoff),
# 4=C3+N (+trusted nested Lag entry). C0..C3 are g3_stack.configure(level).
LEVELS = (0, 1, 2, 3, 4)


def set_level(level):
    g3_stack.uninstall_all()
    lagmod.Lag = _StockLag
    g3_stack.configure(min(level, 3))
    if level >= 4:
        lagmod.Lag = TrustedLag


def run(y, level):
    set_level(level)
    f = NaiveForecaster(strategy="last").set_config(remember_data=True)
    f.fit(y, fh=[1, 2, 3])
    return f, f.predict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="300x100,1000x100")
    ap.add_argument("--conts", default="pd,pl")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="experiments/container_independence/results/design_minimal_nested.json")
    a = ap.parse_args()
    rows = []
    for sz in a.sizes.split(","):
        n, t = map(int, sz.split("x"))
        ypd = make_panel_fast(n, t)
        for cont in a.conts.split(","):
            y = ypd if cont == "pd" else convert_to(ypd, "polars_panel")
            # equality
            ref_f, ref = run(y, 0)
            outs = {}
            for lv in LEVELS[1:]:
                f, p = run(y, lv)
                if cont == "pd":
                    pd.testing.assert_frame_equal(p, ref, check_exact=True)
                else:
                    assert p.equals(ref) and type(p) is type(ref), "polars output differs"
                assert f.cutoff.equals(ref_f.cutoff) and f.cutoff.freq == ref_f.cutoff.freq
                outs[lv] = True
            # fit+predict, interleaved
            times = {lv: float("inf") for lv in LEVELS}
            for _ in range(a.reps):
                for lv in LEVELS:
                    t0 = time.perf_counter()
                    run(y, lv)
                    times[lv] = min(times[lv], time.perf_counter() - t0)
            # predict only (fitted per level)
            fitted = {}
            for lv in LEVELS:
                set_level(lv)
                fitted[lv] = NaiveForecaster(strategy="last").set_config(remember_data=True).fit(y, fh=[1, 2, 3])
            ptimes = {lv: float("inf") for lv in LEVELS}
            NSTATS.update(trusted=0, fallback=0)
            for _ in range(a.reps):
                for lv in LEVELS:
                    set_level(lv)
                    t0 = time.perf_counter()
                    fitted[lv].predict()
                    ptimes[lv] = min(ptimes[lv], time.perf_counter() - t0)
            set_level(0)
            row = {
                "size": sz, "cont": cont, "equal": outs,
                "fp_s": times, "predict_s": ptimes,
                "fp_cum_vs_C0": {lv: times[0] / times[lv] for lv in LEVELS},
                "fp_marginal": {lv: times[lv - 1] / times[lv] for lv in LEVELS[1:]},
                "pred_cum_vs_C0": {lv: ptimes[0] / ptimes[lv] for lv in LEVELS},
                "pred_marginal": {lv: ptimes[lv - 1] / ptimes[lv] for lv in LEVELS[1:]},
                "nested_trusted_calls_predict": NSTATS["trusted"] // a.reps,
                "nested_fallbacks": NSTATS["fallback"],
            }
            print(json.dumps(row), flush=True)
            rows.append(row)
    with open(a.out, "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
