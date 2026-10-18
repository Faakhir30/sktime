"""r3_ts_libs_verify0: side check of C2 as it relates to prior-art patterns.

Does a pandas MultiIndex frame survive narwhals? Checks:
  a) nw.from_native(mi_frame).to_native()        -> index preserved? (no-op passthrough)
  b) nw.maybe_reset_index(mi_frame)              -> level names kept as columns?
  c) group_by over the reset frame + to_native   -> index of result
  d) round trip: reset -> narwhals op -> set_index(level names) recovers the MultiIndex?
  e) copy cost of reset_index vs groupby(level) iteration (one process, min-of-repeats)

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/r3_ts_libs_verify0_c2.py
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")
import narwhals as nw  # noqa: E402
import numpy as np  # noqa: E402

from common import bench, make_panel_fast  # noqa: E402

out = {}
mi = make_panel_fast(1000, 100)
a = nw.from_native(mi, eager_only=True).to_native()
out["a_passthrough_index_is_multiindex"] = bool(a.index.equals(mi.index)
                                                and list(a.index.names) == ["h0", "time"])
out["a_nw_columns"] = list(nw.from_native(mi, eager_only=True).columns)
r = nw.maybe_reset_index(nw.from_native(mi, eager_only=True)).to_native()
out["b_maybe_reset_index_type"] = type(r).__name__
out["b_maybe_reset_index_index"] = type(r.index).__name__
out["b_maybe_reset_index_columns"] = list(r.columns)
rr = mi.reset_index()
g = (nw.from_native(rr, eager_only=True).group_by("h0").agg(nw.col("c0").last())
     .to_native())
out["c_groupby_result_index"] = type(g.index).__name__
out["c_groupby_result_columns"] = list(g.columns)
back = (nw.from_native(rr, eager_only=True).with_columns(c1=nw.col("c0") * 2)
        .to_native().set_index(["h0", "time"]))
out["d_roundtrip_multiindex_equal"] = bool(back.index.equals(mi.index)
                                           and list(back.index.names) == ["h0", "time"])
out["e_reset_index_s"] = bench(lambda: mi.reset_index(), repeat=5)
out["e_reset_set_index_s"] = bench(lambda: mi.reset_index().set_index(["h0", "time"]), repeat=5)


def it():
    s = 0.0
    for _, gg in mi.groupby(level=0, sort=False):
        s += gg["c0"].to_numpy()[-1]
    return s


out["e_groupby_level_iter_s"] = bench(it, repeat=3)
out["e_ratio_reset_over_groupby_iter"] = out["e_reset_index_s"] / out["e_groupby_level_iter_s"]
out["narwhals"] = nw.__version__
out["pandas"] = __import__("pandas").__version__
print(json.dumps(out, indent=1))
p = os.path.join(os.path.dirname(__file__), "..", "results", "r3_ts_libs_verify0_c2.json")
with open(p, "w") as fh:
    json.dump(out, fh, indent=1)
