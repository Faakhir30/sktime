"""a3: does freq-alias drift after a polars round trip break absolute fh?

pd.infer_freq names quarter-start data "QS-OCT" (any of QS-JAN/APR/JUL/OCT is
equivalent); after polars -> pandas conversion the cutoff carries QS-OCT.
Probe: fit on polars quarterly-start data, predict with an absolute fh whose
DatetimeIndex carries freq QS-JAN (what pd.date_range(..., freq="QS") gives).
Same for annual "YS" and weekly W-SUN, and pandas-input controls.

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/a3_freq_alias_probe.py
"""

import json
import warnings

import numpy as np
import pandas as pd

from common import env_info

warnings.filterwarnings("ignore")
PD3 = int(pd.__version__.split(".")[0]) >= 3


def main():
    from sktime.datatypes import convert_to
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.naive import NaiveForecaster

    res = {"env": env_info(), "cases": {}}
    for freq in ["QS", "YS", "QE", "W-SUN", "MS"]:
        idx = pd.date_range("2000-01-01", periods=20, freq=freq)
        y_pd = pd.DataFrame({"y": np.arange(20.0)}, index=idx)
        fh_idx = pd.date_range(idx[-1], periods=4, freq=freq)[1:]
        for kind in ["pandas", "polars(pl.DataFrame)"]:
            y = y_pd if kind == "pandas" else convert_to(y_pd, "pl.DataFrame")
            rec = {"fh_freq": fh_idx.freqstr}
            try:
                f = NaiveForecaster().fit(y)
                rec["cutoff_freq"] = getattr(f.cutoff, "freqstr", None)
                yp = f.predict(fh=ForecastingHorizon(fh_idx, is_relative=False))
                rec["ok"] = True
                rec["n_pred"] = len(yp)
            except Exception as e:  # noqa: BLE001
                rec["ok"] = False
                rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            res["cases"][f"{freq} {kind}"] = rec
            print(f"{freq:6s} {kind:22s} {rec}")
    tag = "pd3" if PD3 else "pd2"
    with open(f"experiments/container_independence/results/a3_freq_alias_probe_{tag}"
              ".json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
