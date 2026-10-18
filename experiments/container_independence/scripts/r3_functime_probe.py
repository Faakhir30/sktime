"""r3: functime container probe -- which inputs it accepts and what it returns.

    PYTHONPATH=.:experiments/container_independence $R3F_PY -W ignore \
        experiments/container_independence/scripts/r3_functime_probe.py
"""
import json
import warnings

warnings.filterwarnings("ignore")
import polars as pl  # noqa: E402

from common import make_panel_fast  # noqa: E402
from functime.forecasting import naive  # noqa: E402

flat = make_panel_fast(5, 20).reset_index()
plf = pl.from_pandas(flat)
out = {}
for name, y in [("polars_eager", plf), ("polars_lazy", plf.lazy()), ("pandas", flat),
                ("pyarrow", plf.to_arrow())]:
    try:
        r = naive(freq="1d")(y=y, fh=3)
        out[name] = f"{type(r).__module__}.{type(r).__name__}"
    except Exception as e:  # noqa: BLE001
        out[name] = f"ERROR {type(e).__name__}: {str(e)[:120]}"
print(json.dumps(out, indent=1))
