import cProfile, pstats, io, warnings, time
warnings.filterwarnings("ignore")
import common
from common import bench
from sktime.datatypes._vectorize import VectorizedDF
from sktime.forecasting.naive import NaiveForecaster
X = common.make_panel_fast(10, 100)
v = VectorizedDF(X, iterate_as="Series", is_scitype="Panel", X_orig_mtype="pd-multiindex", freq="D")
_, _, inst = next(iter(v.items(X)))
def one(cfg=None):
    f = NaiveForecaster()
    if cfg: f.set_config(**cfg)
    f.fit(inst, fh=[1, 2]); return f.predict()
print("fit+predict us (default config):", bench(one, repeat=7) * 1e6)
print("fit+predict us (remember_data=False):", bench(lambda: one({"remember_data": False}), repeat=7) * 1e6)
print("fit+predict us (input_conversion/output_conversion off):", bench(lambda: one({"input_conversion": "off", "output_conversion": "off"}), repeat=7) * 1e6)
pr = cProfile.Profile(); pr.enable()
for _ in range(20): one()
pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25); print(s.getvalue()[:6000])
