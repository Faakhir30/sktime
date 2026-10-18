"""r4: pandera as prior art for "one API, per-backend implementation".

Probes (pandera 0.34.x, r4 private venv):
  * which schema classes exist per backend (separate classes vs one class)
  * registry dispatch: which backend class serves (schema_cls, frame_type)
  * native vs narwhals backend (PANDERA_USE_NARWHALS_BACKEND) for pandas / polars:
    validation time ratio on identical data, and MultiIndex handling
  * dispatch overhead: get_backend() cost per call

Run:
  PYTHONPATH=experiments/container_independence \
    $SCRATCH/venvs/r4/bin/python -W ignore \
    experiments/container_independence/scripts/r4_pandera_backends.py --grid dev
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

from common import DEV_GRID, LARGE_GRID, bench, make_panel_fast

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])
    args = ap.parse_args()
    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    import pandera
    import pandera.config as pcfg
    import pandera.pandas as pa_pd
    import pandera.polars as pa_pl
    import polars as pl

    out = {
        "env": {"pandera": pandera.__version__, "pandas": pd.__version__,
                "polars": pl.__version__},
        "python": sys.version.split()[0],
        "schema_classes_distinct": pa_pd.DataFrameSchema is not pa_pl.DataFrameSchema,
        "dispatch": {},
        "rows": [],
        "multiindex": {},
    }

    def schemas():
        s_pd = pa_pd.DataFrameSchema(
            {"c0": pa_pd.Column(float, pa_pd.Check.in_range(-1e9, 1e9))},
            index=pa_pd.MultiIndex(
                [pa_pd.Index(str, name="h0"), pa_pd.Index("datetime64[ns]", name="time")]
            ),
        )
        s_pd_flat = pa_pd.DataFrameSchema(
            {"h0": pa_pd.Column(str), "time": pa_pd.Column("datetime64[ns]"),
             "c0": pa_pd.Column(float, pa_pd.Check.in_range(-1e9, 1e9))}
        )
        s_pl = pa_pl.DataFrameSchema(
            {"h0": pa_pl.Column(pl.String), "time": pa_pl.Column(pl.Datetime("ns")),
             "c0": pa_pl.Column(pl.Float64, pa_pl.Check.in_range(-1e9, 1e9))}
        )
        return s_pd, s_pd_flat, s_pl

    for mode in [False, True]:
        pcfg.set_config(use_narwhals_backend=mode) if hasattr(pcfg, "set_config") else None
        tag = "narwhals" if mode else "native"
        s_pd, s_pd_flat, s_pl = schemas()
        small = make_panel_fast(3, 4)
        flat = small.reset_index()
        plf = pl.from_pandas(flat)
        d = {}
        for name, sch, obj in [
            ("pandas MI", s_pd, small), ("pandas flat", s_pd_flat, flat),
            ("polars eager", s_pl, plf), ("polars lazy", s_pl, plf.lazy()),
        ]:
            try:
                b = type(sch.get_backend(obj))
                d[name] = f"{b.__module__}.{b.__name__}"
            except Exception as e:  # noqa: BLE001
                d[name] = "ERR " + repr(e)[:150]
        out["dispatch"][tag] = d
        # MultiIndex validation result
        try:
            r = s_pd.validate(small)
            out["multiindex"][tag] = {
                "ok": True, "index_type": type(r.index).__name__,
                "names": list(r.index.names),
            }
        except Exception as e:  # noqa: BLE001
            out["multiindex"][tag] = {"ok": False, "err": repr(e)[:200]}
        # bad data detected?
        bad = small.copy()
        bad.iloc[0, 0] = 1e12
        try:
            s_pd.validate(bad)
            out["multiindex"][tag]["detects_bad"] = False
        except Exception:  # noqa: BLE001
            out["multiindex"][tag]["detects_bad"] = True
        out["dispatch"][tag + " get_backend us/call"] = bench(
            lambda: [s_pl.get_backend(plf) for _ in range(1000)], repeat=3
        ) * 1e3
        print(tag, json.dumps(out["dispatch"][tag], indent=1), out["multiindex"][tag])

    for n_series, n_time in grid:
        pdf = make_panel_fast(n_series, n_time)
        flat = pdf.reset_index()
        plf = pl.from_pandas(flat)
        row = {"n_series": n_series, "n_time": n_time}
        for mode in [False, True]:
            pcfg.set_config(use_narwhals_backend=mode)
            tag = "narwhals" if mode else "native"
            s_pd, s_pd_flat, s_pl = schemas()
            for name, sch, obj in [
                ("pandas MI", s_pd, pdf), ("pandas flat", s_pd_flat, flat),
                ("polars eager", s_pl, plf),
                ("polars lazy (+collect)", s_pl, plf.lazy()),
            ]:
                key = f"{tag}: {name}"
                try:
                    if "lazy" in name:
                        fn = lambda: sch.validate(obj).collect()  # noqa: E731
                    else:
                        fn = lambda: sch.validate(obj)  # noqa: E731
                    row[key] = bench(fn, repeat=3)
                except Exception as e:  # noqa: BLE001
                    row[key] = "ERR " + repr(e)[:150]
        out["rows"].append(row)
        print(n_series, n_time, {k: (f"{v*1e3:.2f}ms" if isinstance(v, float) else v)
                                 for k, v in row.items() if k not in ("n_series", "n_time")},
              flush=True)
    path = os.path.join(RES, f"r4_pandera_backends_{args.grid}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
