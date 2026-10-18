"""x9 verifier: zero-copy claims via pyarrow buffer addresses (independent code path:
uses polars .to_arrow() buffers and the pandas ArrowExtensionArray's pa.ChunkedArray
buffers, not polars' internal pointer helpers), plus np.shares_memory, plus
multi-chunk input (as produced by read_parquet / concat), plus the hybrid layout and
the reverse direction, plus memory delta of a large conversion as a sanity check.

Run::
    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x9_cheap_paths_verify0_zerocopy.py --grid dev
"""
import argparse
import sys
import warnings

warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402

from common import peak_rss_delta_mb  # noqa: E402

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from x9_arrow_convert import make_pl_panel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="dev")
a = ap.parse_args()
print("pandas", pd.__version__, "polars", pl.__version__, "pyarrow", pa.__version__)


def data_addrs_pl(s):
    """addresses of the data buffers (last buffer of each chunk) of a polars Series."""
    arr = s.to_arrow()
    chunks = arr.chunks if isinstance(arr, pa.ChunkedArray) else [arr]
    return [c.buffers()[-1].address for c in chunks if c.buffers()[-1] is not None]


def data_addrs_pd(col):
    arr = col.array
    if hasattr(arr, "_pa_array"):
        ca = arr._pa_array
        return [c.buffers()[-1].address for c in ca.chunks]
    return [np.asarray(col).__array_interface__["data"][0]]


def report(tag, pls, pdc):
    a1, a2 = data_addrs_pl(pls), data_addrs_pd(pdc)
    print(f"  {tag:<34} same_addr={a1 == a2}  (pl {len(a1)} chunk(s), pd {len(a2)})")
    return a1 == a2


for chunked in [False, True]:
    df = make_pl_panel(1000, 100, n_cols=1)
    if chunked:
        df = pl.concat([df.slice(0, 50_000), df.slice(50_000)], rechunk=False)
    print(f"--- input n_chunks={df.n_chunks()} ---")
    arrow = df.to_pandas(use_pyarrow_extension_array=True)
    npd = df.to_pandas()
    for c in ["c0", "__index__time"]:
        report(f"arrow {c}", df[c], arrow[c])
        report(f"numpy {c}", df[c], npd[c])
    report("arrow __index__h0 (string)", df["__index__h0"], arrow["__index__h0"])
    # shares_memory on value column
    if not chunked:
        x_pl = df["c0"].to_numpy(allow_copy=False)
        x_ar = arrow["c0"].array._pa_array.chunk(0).to_numpy(zero_copy_only=True)
        print("  np.shares_memory(c0 polars, arrow pandas):", np.shares_memory(x_pl, x_ar),
              "| vs numpy route:", np.shares_memory(x_pl, npd["c0"].to_numpy()))
    # after a pandas op that sktime does: set_index for hybrid / arrow MI
    hyb = arrow[["c0"]].copy(deep=False)
    hyb.index = pd.MultiIndex.from_arrays(
        [npd["__index__h0"], npd["__index__time"]], names=["h0", "time"])
    report("hybrid c0 after index assign", df["c0"], hyb["c0"])
    # sktime-ish ops on the hybrid: slice per instance (loc), .iloc tail
    sl = hyb.loc["s0000001"]
    print("  hybrid loc slice dtype", sl["c0"].dtype)
    # reverse
    back = pl.from_pandas(arrow)
    for c in ["c0", "__index__time"]:
        a1 = data_addrs_pl(back[c])
        print(f"  back {c}: same_addr_as_src={a1 == data_addrs_pl(df[c])}")

# memory sanity: 2M rows x 4 float cols -> arrow vs numpy; fresh-ish within process
n_s, n_t = (10_000, 200) if a.grid == "dev" else (50_000, 200)
big = make_pl_panel(n_s, n_t, n_cols=4)
print(f"--- memory: {big.height} rows, polars est {big.estimated_size('mb'):.1f} MB ---")
_, d_ar = peak_rss_delta_mb(lambda: big.select(pl.exclude("__index__h0")).to_pandas(
    use_pyarrow_extension_array=True))
_, d_np = peak_rss_delta_mb(lambda: big.select(pl.exclude("__index__h0")).to_pandas())
print(f"  peak RSS delta excl. string key: arrow {d_ar:.1f} MB, numpy {d_np:.1f} MB")
_, d_ar2 = peak_rss_delta_mb(lambda: big.to_pandas(use_pyarrow_extension_array=True))
print(f"  peak RSS delta incl. string key: arrow {d_ar2:.1f} MB")
