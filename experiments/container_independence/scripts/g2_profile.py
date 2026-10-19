"""g2 attribution: where does g2 fit+predict time go (cProfile, cumulative).

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/g2_profile.py
"""

import cProfile
import json
import os
import pstats
import sys
import warnings

warnings.filterwarnings("ignore")
ED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ED)

from common import bench, make_panel_fast  # noqa: E402
from protos.g2_forecasters import G2NaiveForecaster  # noqa: E402
from sktime.datatypes import convert_to  # noqa: E402

FUNCS = {
    "check_is_scitype (input y check, BaseForecaster._check_X_y)": "check_is_scitype",
    "mtype (2nd input check inside convert_to)": "mtype",
    "ingress spoke (from_pandas/from_polars/from_arrow)": ("from_pandas", "from_polars",
                                                           "from_arrow"),
    "infer_step (numpy, all instances)": "infer_step",
    "_fit": "_fit",
    "_predict (kernel + numpy time grid)": "_predict",
    "fh_to_grid": "fh_to_grid",
    "get_cutoff (1-elem pandas index)": "get_cutoff",
    "_check_fh": "_check_fh",
    "egress spoke (to_pandas/to_polars/to_arrow)": ("to_pandas", "to_polars",
                                                    "to_arrow"),
}


def run(yin):
    f = G2NaiveForecaster("last")
    f.set_config(remember_data=True)
    return f.fit(yin, fh=list(range(1, 11))).predict()


def main():
    import pyarrow as pa

    y = make_panel_fast(10000, 50)
    flat = y.reset_index()
    flat.columns = ["__index__h0", "__index__time", "c0"]
    ins = {"pandas": y, "polars": convert_to(y, "polars_panel"),
           "pyarrow": pa.Table.from_pandas(flat, preserve_index=False)}
    out = {}
    for name, yin in ins.items():
        t = bench(lambda: run(yin), repeat=5)
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(3):
            run(yin)
        pr.disable()
        st = pstats.Stats(pr)
        tot = sum(v[3] for k, v in st.stats.items() if k[2] == "run") / 3
        rows = {}
        for label, fn in FUNCS.items():
            fns = fn if isinstance(fn, tuple) else (fn,)
            ct = sum(v[3] for k, v in st.stats.items() if k[2] in fns
                     and "site-packages" not in k[0]) / 3
            rows[label] = round(ct / tot, 3) if tot else None
        out[name] = dict(min_s=t, profiled_total_s=tot, share=rows)
        print(name, json.dumps(out[name], indent=1))
    with open(os.path.join(ED, "results", "g2_profile.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
