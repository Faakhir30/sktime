"""x6 verifier: NaiveForecaster numpy kernels vs sktime on harder inputs + speed ratio.

Fixtures beyond the original (make_panel_fast, 1 col, regular daily, no NaN):
NaN at series end, unequal lengths, 2 value columns, month-end freq, PeriodIndex.
Equality: assert_frame_equal with check_freq=True (original used check_freq=False).

Run: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
       experiments/container_independence/scripts/x6_proto_hub_verify0_forecast.py --grid dev
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ED, "protos"))
warnings.simplefilter("ignore")
warnings.showwarning = lambda *a, **k: None
import x6_hub as H  # noqa: E402
from common import bench, env_info, make_panel, make_panel_fast  # noqa: E402

FH = np.arange(1, 6)


def fixtures():
    fx = {}
    b = make_panel(5, 20, n_cols=1, n_levels=1)
    nan = b.copy()
    nan.iloc[[19, 18, 39], 0] = np.nan  # last two of s0, last of s1
    fx["nan_at_end"] = nan
    fx["unequal_len"] = b.drop(index=b.index[:7])
    fx["two_cols"] = make_panel(5, 20, n_cols=2, n_levels=1)
    fx["month_end"] = make_panel(5, 20, n_cols=1, n_levels=1, freq="ME")
    fx["period_M"] = make_panel(5, 20, n_cols=1, n_levels=1, time_kind="period",
                                freq="M")
    return fx


def main():
    from sktime.forecasting.naive import NaiveForecaster

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    a = ap.parse_args()
    res = {"env": env_info(), "equiv": [], "speed": []}
    for name, X in fixtures().items():
        for strat in ["last", "mean", "drift"]:
            try:
                ref = NaiveForecaster(strategy=strat).fit(X, fh=FH).predict()
            except Exception as e:  # noqa: BLE001
                ref = None
                ref_err = f"sktime raises {type(e).__name__}: {str(e)[:90]}"
            for kind in ["loop", "batched"]:
                rec = {"fixture": name, "strategy": strat, "kind": kind}
                if ref is None:
                    hub = H.from_native(X)
                    o = H.to_native(H.hub_naive_batched(hub, FH, strat))
                    rec.update(ok=None, reason=ref_err + "; hub returns NaN rows: "
                               + str(int(o.isna().any(axis=1).sum())))
                    res["equiv"].append(rec)
                    print(json.dumps(rec), flush=True)
                    continue
                try:
                    hub = H.from_native(X)
                    if kind == "loop":
                        out = H.to_native(H.hub_vectorize(
                            hub, lambda s=strat: H.HubNaive(s), FH))
                    else:
                        out = H.to_native(H.hub_naive_batched(hub, FH, strat))
                    pd.testing.assert_frame_equal(out, ref, check_freq=True,
                                                  check_exact=False, rtol=1e-10)
                    rec["ok"] = True
                except Exception as e:  # noqa: BLE001
                    rec["ok"] = False
                    rec["reason"] = f"{type(e).__name__}: " + str(e).strip(
                        ).splitlines()[0][:140] if str(e) else type(e).__name__
                res["equiv"].append(rec)
                print(json.dumps(rec), flush=True)

    sizes = [(200, 100), (1000, 100)] if a.grid == "dev" else [(5000, 100)]
    for n, t in sizes:
        X = make_panel_fast(n, t)
        cur = bench(lambda: NaiveForecaster(strategy="last").fit(X, fh=FH).predict(),
                    repeat=1, warmup=0)
        bat = bench(lambda: H.to_native(H.hub_naive_batched(H.from_native(X), FH,
                                                             "last")), repeat=5)
        loop = bench(lambda: H.to_native(H.hub_vectorize(
            H.from_native(X), lambda: H.HubNaive("last"), FH)), repeat=3)
        r = {"n": n, "t": t, "current_s": cur, "hub_loop_s": loop, "hub_batched_s": bat,
             "ratio_loop": cur / loop, "ratio_batched": cur / bat,
             "current_ms_per_instance": cur / n * 1e3}
        res["speed"].append(r)
        print(json.dumps(r), flush=True)
    tag = "pd" + pd.__version__.split(".")[0]
    with open(os.path.join(ED, "results",
                           f"x6_proto_hub_verify0_forecast_{tag}_{a.grid}.json"),
              "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
