"""x8 verify0: per-instance check+convert cost if slices stayed polars (native per-instance path).

Run: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
    experiments/container_independence/scripts/x8_e2e_profile_verify0_slicecheck.py
"""
import json, os, warnings
warnings.simplefilter("ignore")
from common import bench, make_panel_fast
from sktime.datatypes import check_is_scitype, convert_to
y = make_panel_fast(1, 100).droplevel(0); y.index.freq = "D"
ypl = convert_to(y, "pl.DataFrame")
meta = ["n_features", "feature_names", "feature_kind", "is_univariate", "has_nans"]
r = {
    "check_series_pd_us": 1e6 * bench(lambda: check_is_scitype(y, scitype="Series", return_metadata=meta), repeat=20, number=20),
    "check_series_pl_us": 1e6 * bench(lambda: check_is_scitype(ypl, scitype="Series", return_metadata=meta), repeat=20, number=20),
    "convert_to_pd_frame_noop_us": 1e6 * bench(lambda: convert_to(y, "pd.DataFrame"), repeat=20, number=20),
    "convert_to_pl_to_pd_us": 1e6 * bench(lambda: convert_to(ypl, "pd.DataFrame"), repeat=20, number=20),
}
print(r)
json.dump(r, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "x8_e2e_profile_verify0_slicecheck.json"), "w"), indent=1)
