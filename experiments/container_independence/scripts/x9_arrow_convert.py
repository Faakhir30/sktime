"""x9 (A1): polars -> pandas with Arrow-backed dtypes vs numpy-backed (sktime today).

Input: a polars frame in the sktime ``polars_panel`` convention
(``__index__h0`` String, ``__index__time`` Datetime[ns], value cols Float64),
generated directly in polars (no pandas detour, so the RSS baseline is clean).

Forward variants (polars -> pandas):
  sktime_convert_to   : sktime.datatypes.convert_to(pl_df, "pd-multiindex")
                        (what BaseForecaster._check_X_y does: mtype inference/check
                        + convert_polars_to_pandas + infer_freq)
  sktime_adapter      : sktime.datatypes._adapter.polars.convert_polars_to_pandas
                        (the conversion alone, without the mtype check)
  np_flat             : pl_df.to_pandas()                      (numpy, no index)
  arrow_flat          : pl_df.to_pandas(use_pyarrow_extension_array=True)
  np_mi               : np_flat + set_index(index cols)        (numpy + MultiIndex)
  arrow_mi            : arrow_flat + set_index(index cols)     (Arrow everywhere;
                        NOT accepted by sktime, see x9_arrow_compat)
  hybrid_mi           : MultiIndex from numpy-converted index cols + Arrow values
                        (the variant sktime DOES accept, see x9_arrow_compat)

Return variants (pandas -> polars):
  ret_sktime_convert_to : convert_to(np_mi, "polars_panel")
  ret_np_flat           : pl.from_pandas(np_flat)
  ret_arrow_flat        : pl.from_pandas(arrow_flat)
  ret_hybrid_mi         : pl.from_pandas(hybrid_mi.reset_index())

Measures, per grid size: min-of-repeats wall time (interleaved, same process) and
peak RSS delta (one fresh subprocess per (size, variant), result kept alive), plus
zero-copy evidence: buffer addresses of the value / time / key columns before and
after conversion.

Run::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x9_arrow_convert.py --grid dev
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import warnings

warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from common import DEV_GRID, LARGE_GRID, bench, env_info, peak_rss_delta_mb, rss_mb  # noqa: E402,E501

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
IDX = ["__index__h0", "__index__time"]


def make_pl_panel(n_series, n_time, n_cols=1, seed=0):
    """Polars frame in sktime polars_panel convention, built in polars."""
    rng = np.random.default_rng(seed)
    n = n_series * n_time
    base = pl.select(pl.int_range(0, n, dtype=pl.Int64).alias("i"))
    df = base.select(
        (pl.lit("s") + (pl.col("i") // n_time).cast(pl.String).str.zfill(7)).alias(
            "__index__h0"
        ),
        (
            pl.lit(pd.Timestamp("2000-01-01")).cast(pl.Datetime("ns"))
            + pl.duration(days=pl.col("i") % n_time)
        )
        .cast(pl.Datetime("ns"))
        .alias("__index__time"),
    )
    vals = {}
    for j in range(n_cols):
        vals[f"c{j}"] = pl.Series(rng.normal(size=n).cumsum())
    return df.with_columns(**vals).rechunk()


# ----------------------------------------------------------------------------
# variants
# ----------------------------------------------------------------------------
def _value_cols(pl_df):
    return [c for c in pl_df.columns if c not in IDX]


def v_sktime_convert_to(pl_df):
    from sktime.datatypes import convert_to

    return convert_to(pl_df, "pd-multiindex")


def v_sktime_adapter(pl_df):
    from sktime.datatypes._adapter.polars import convert_polars_to_pandas

    return convert_polars_to_pandas(pl_df)


def v_np_flat(pl_df):
    return pl_df.to_pandas()


def v_arrow_flat(pl_df):
    return pl_df.to_pandas(use_pyarrow_extension_array=True)


def v_np_mi(pl_df):
    return pl_df.to_pandas().set_index(IDX)


def v_arrow_mi(pl_df):
    return pl_df.to_pandas(use_pyarrow_extension_array=True).set_index(IDX)


def v_hybrid_mi(pl_df):
    mi = pd.MultiIndex.from_frame(pl_df.select(IDX).to_pandas(), names=["h0", "time"])
    vals = pl_df.select(_value_cols(pl_df)).to_pandas(use_pyarrow_extension_array=True)
    vals.index = mi
    return vals


FORWARD = {
    "sktime_convert_to": v_sktime_convert_to,
    "sktime_adapter": v_sktime_adapter,
    "np_flat": v_np_flat,
    "arrow_flat": v_arrow_flat,
    "np_mi": v_np_mi,
    "arrow_mi": v_arrow_mi,
    "hybrid_mi": v_hybrid_mi,
}


def r_sktime_convert_to(pd_obj):
    from sktime.datatypes import convert_to

    return convert_to(pd_obj, "polars_panel")


def r_from_pandas(pd_obj):
    return pl.from_pandas(pd_obj)


def r_hybrid(pd_obj):
    return pl.from_pandas(pd_obj.reset_index())


# return variant -> (input builder, fn)
RETURN = {
    "ret_sktime_convert_to": (v_np_mi, r_sktime_convert_to),
    "ret_np_flat": (v_np_flat, r_from_pandas),
    "ret_arrow_flat": (v_arrow_flat, r_from_pandas),
    "ret_hybrid_mi": (v_hybrid_mi, r_hybrid),
}


# ----------------------------------------------------------------------------
# zero-copy evidence
# ----------------------------------------------------------------------------
def _pl_addr(s):
    a = s.to_arrow()
    if hasattr(a, "combine_chunks"):
        a = a.combine_chunks()
    bufs = a.buffers()
    return bufs[-1].address if a.type in ("large_string", "string") else bufs[1].address


def _pd_addr(col):
    arr = col.array
    if hasattr(arr, "_pa_array"):
        ch = arr._pa_array
        if ch.num_chunks != 1:
            return f"{ch.num_chunks} chunks"
        bufs = ch.chunks[0].buffers()
        return bufs[-1].address if str(ch.type) in ("large_string", "string") else bufs[1].address
    npa = np.asarray(col.to_numpy())
    if npa.dtype == object:
        return "object (python str per row)"
    return npa.__array_interface__["data"][0]


def zero_copy_report(n_series=1000, n_time=100):
    pl_df = make_pl_panel(n_series, n_time, n_cols=1)
    src = {c: _pl_addr(pl_df[c]) for c in pl_df.columns}
    out = {"source_polars": {c: str(a) for c, a in src.items()}}
    for name in ["np_flat", "arrow_flat"]:
        pdf = FORWARD[name](pl_df)
        rec = {}
        for c in pl_df.columns:
            a = _pd_addr(pdf[c])
            rec[c] = {"addr": str(a), "zero_copy": a == src[c]}
        out[name] = rec
        back = pl.from_pandas(pdf)
        rec_b = {}
        for c in pl_df.columns:
            a_pd = _pd_addr(pdf[c])
            a_b = _pl_addr(back[c])
            rec_b[c] = {"zero_copy_from_pandas": a_b == a_pd}
        out[name + "->pl.from_pandas"] = rec_b
    # hybrid: values Arrow, index via numpy MultiIndex (levels are a factorized copy)
    hy = v_hybrid_mi(pl_df)
    out["hybrid_mi"] = {
        "c0": {"addr": str(_pd_addr(hy["c0"])), "zero_copy": _pd_addr(hy["c0"]) == src["c0"]},
        "index": "MultiIndex levels+codes (factorized copy by construction)",
    }
    # sktime adapter (numpy): values copied?
    sk = v_sktime_adapter(pl_df)
    out["sktime_adapter"] = {
        "c0": {"addr": str(_pd_addr(sk["c0"])), "zero_copy": _pd_addr(sk["c0"]) == src["c0"]}
    }
    # numpy shares_memory check where meaningful (float column)
    pl_view = pl_df["c0"].to_numpy(allow_copy=False)
    out["np.shares_memory(polars c0 view, np_flat c0)"] = bool(
        np.shares_memory(pl_view, v_np_flat(pl_df)["c0"].to_numpy())
    )
    out["np.shares_memory(polars c0 view, arrow_flat c0 -> to_numpy)"] = bool(
        np.shares_memory(
            pl_view,
            v_arrow_flat(pl_df)["c0"].array._pa_array.chunks[0].to_numpy(
                zero_copy_only=True
            ),
        )
    )
    return out


# ----------------------------------------------------------------------------
# memory worker (fresh process per measurement)
# ----------------------------------------------------------------------------
def mem_worker(variant, n_series, n_time, n_cols):
    pl_df = make_pl_panel(n_series, n_time, n_cols)
    if variant in FORWARD:
        fn = FORWARD[variant]
        arg = pl_df
        # warm up imports / code paths on a tiny frame so imports are not counted
        fn(make_pl_panel(3, 5, n_cols))
    else:
        build, fn = RETURN[variant]
        arg = build(pl_df)
        fn(build(make_pl_panel(3, 5, n_cols)))
    gc.collect()
    rss0 = rss_mb()
    res, delta = peak_rss_delta_mb(lambda: fn(arg))
    nbytes_src = pl_df.estimated_size() / 2**20
    print(
        json.dumps(
            {"variant": variant, "peak_delta_mb": delta, "rss0_mb": rss0,
             "src_polars_mb": nbytes_src}
        )
    )
    del res


def run_mem(variant, n_series, n_time, n_cols):
    cmd = [
        sys.executable, "-W", "ignore", os.path.abspath(__file__),
        "--mem-worker", variant, str(n_series), str(n_time), str(n_cols),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    for line in p.stdout.splitlines()[::-1]:
        if line.startswith("{"):
            return json.loads(line)
    return {"variant": variant, "error": p.stderr[-500:]}


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--mem-worker", nargs=4, default=None)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--no-mem", action="store_true")
    args = ap.parse_args()

    if args.mem_worker:
        v, a, b, c = args.mem_worker
        mem_worker(v, int(a), int(b), int(c))
        return

    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    cases = [(ns, nt, 1) for ns, nt in grid]
    # wide case: more value columns -> value copy dominates
    cases += [(1_000, 100, 10)] if args.grid == "dev" else [(10_000, 100, 10)]

    out = {"env": env_info(), "python": sys.version.split()[0], "grid": args.grid}
    out["zero_copy"] = zero_copy_report()
    print("zero-copy evidence:")
    print(json.dumps(out["zero_copy"], indent=1))

    variants_all = list(FORWARD) + list(RETURN)
    rows = []
    for ns, nt, nc in cases:
        pl_df = make_pl_panel(ns, nt, nc)
        inputs = {v: pl_df for v in FORWARD}
        for v, (build, _) in RETURN.items():
            inputs[v] = build(pl_df)
        fns = {**FORWARD, **{v: f for v, (_, f) in RETURN.items()}}
        best = {v: float("inf") for v in variants_all}
        rep = 3 if ns * nt <= 200_000 else 2
        for _ in range(args.rounds):
            for v in variants_all:
                if v == "sktime_convert_to" and ns * nt > 2_000_000:
                    continue
                t = bench(lambda v=v: fns[v](inputs[v]), repeat=rep, warmup=1)
                best[v] = min(best[v], t)
        del inputs
        gc.collect()
        mem = {}
        if not args.no_mem:
            for v in variants_all:
                mem[v] = run_mem(v, ns, nt, nc)
        row = {
            "n_series": ns, "n_time": nt, "n_cols": nc, "rows": ns * nt,
            "src_polars_mb": pl_df.estimated_size() / 2**20,
            "time_s": best, "mem": mem,
        }
        ref_f = best["sktime_adapter"]
        ref_r = best["ret_sktime_convert_to"]
        row["ratio_vs_sktime_adapter"] = {
            v: best[v] / ref_f for v in FORWARD if np.isfinite(best[v])
        }
        row["ratio_vs_sktime_return"] = {v: best[v] / ref_r for v in RETURN}
        rows.append(row)
        print(f"\n== n_series={ns} n_time={nt} n_cols={nc} rows={ns*nt} "
              f"(polars src {row['src_polars_mb']:.1f} MB)")
        for v in variants_all:
            m = mem.get(v, {})
            pk = m.get("peak_delta_mb")
            r = row["ratio_vs_sktime_adapter"].get(v) or row["ratio_vs_sktime_return"].get(v)
            print(
                f"  {v:24s} {best[v]*1e3:10.2f} ms  ratio={r if r is None else round(r, 3)}"
                f"  peakRSSdelta={'n/a' if pk is None else f'{pk:7.1f} MB'}"
            )
        del pl_df
        gc.collect()

    out["rows"] = rows
    tag = "pd3" if pd.__version__.startswith("3") else "pd2"
    path = os.path.join(RES, f"x9_arrow_convert_{args.grid}_{tag}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
