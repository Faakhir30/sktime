"""x9: where does sktime convert_to(polars_panel -> pd-multiindex) spend time?

Run::
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x9_profile_convert.py --grid dev
"""
import argparse, cProfile, pstats, io, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from x9_arrow_convert import make_pl_panel  # noqa: E402
from sktime.datatypes import convert_to  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="dev"); a = ap.parse_args()
ns, nt = (10_000, 50) if a.grid == "dev" else (50_000, 50)
df = make_pl_panel(ns, nt)
convert_to(make_pl_panel(3, 5), "pd-multiindex")
pr = cProfile.Profile(); pr.enable(); convert_to(df, "pd-multiindex"); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
print("\n".join(l[:200] for l in s.getvalue().splitlines()))

# --- vectorized alternative to sktime/datatypes/_adapter/polars.py::is_monotonically_increasing
import polars as pl  # noqa: E402
from common import bench  # noqa: E402
from sktime.datatypes._adapter.polars import is_monotonically_increasing  # noqa: E402


def is_mono_vec(obj):
    cols = [c for c in obj.columns if c.startswith("__index__")]
    if len(cols) <= 1:
        return obj[cols[0]].is_sorted() if cols else True
    t = pl.col(cols[-1])
    ok = (t >= t.shift(1).over(cols[:-1])).fill_null(True)
    return bool(obj.select(ok.all()).item())


bad = df.with_columns(
    pl.when(pl.int_range(pl.len()) == 7).then(pl.col("__index__time") - pl.duration(days=30))
    .otherwise(pl.col("__index__time")).alias("__index__time"))
for name, d in [("sorted", df), ("unsorted_at_row7", bad)]:
    r_old, r_new = is_monotonically_increasing(d), is_mono_vec(d)
    t_old = bench(lambda: is_monotonically_increasing(d), repeat=3)
    t_new = bench(lambda: is_mono_vec(d), repeat=5)
    print(f"[{name}] rows={d.height} current={r_old} {t_old*1e3:.1f} ms | vectorized={r_new} "
          f"{t_new*1e3:.2f} ms | speedup x{t_old/t_new:.0f}")
t_full = bench(lambda: convert_to(df, "pd-multiindex"), repeat=3)
t_chk = bench(lambda: is_monotonically_increasing(df), repeat=3)
print(f"convert_to total {t_full*1e3:.1f} ms; current monotonic check {t_chk*1e3:.1f} ms = {t_chk/t_full:.0%} of convert_to")
