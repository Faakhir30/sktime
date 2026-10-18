"""x2 (4): narwhals group_by *iteration* vs pandas groupby(level=..., sort=False).

Input: Panel with instance order deliberately NOT sorted (permuted), so that
"first-appearance order" (pandas sort=False) differs from sorted order.

Methods (all produce one slice per instance):

  pd_level        [g.droplevel(0) for _, g in df.groupby(level=0, sort=False)]
  sk_items        list(vec.items(X=X_mi))      current sktime VectorizedDF (adds freq
                                               inference via pd.infer_freq per slice)
  pd_col          [g for _, g in flat.groupby("h0", sort=False)]   flat = reset_index()
  nw_pd           [g for _, g in nw.from_native(flat).group_by("h0")]
  nw_pd_native    same, + nw.to_native(g)
  nw_pd_recipe    full "index-preserving recipe" equivalent of sk_items, INCLUDING
                  reset_index: reset_index -> nw group_by -> to_native ->
                  drop(h0) -> set_index(time) -> index.freq = schema freq
  nw_pd_recipe_infer   same, but pd.infer_freq per slice (like sktime does)
  pl_iter_mo      polars df.group_by("h0", maintain_order=True) iteration
  pl_partition    polars df.partition_by("h0", maintain_order=True)
  nw_pl           [g for _, g in nw.from_native(pl_df).group_by("h0")]

Also records: iteration order == first-appearance order? (and stable over 5 runs
for the polars-backed variants that do not request maintain_order), the type of
the group key, and the type/freq of the yielded slice index.

Usage::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/x2_nw_groupby_iter_bench.py --grid dev
"""

import argparse
import gc
import json
import os
import sys

import narwhals as nw
import numpy as np
import pandas as pd

from common import DEV_GRID, LARGE_GRID, bench, env_info, make_panel_fast, to_polars_long

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def make_permuted(n_series, n_time, n_cols, seed=1):
    df = make_panel_fast(n_series, n_time, n_cols=n_cols)
    perm = np.random.default_rng(seed).permutation(n_series)
    rows = (perm[:, None] * n_time + np.arange(n_time)).ravel()
    return df.iloc[rows]


def first_appearance(df):
    return list(pd.unique(df.index.get_level_values(0)))


def norm_key(k):
    if isinstance(k, tuple) and len(k) == 1:
        return k[0]
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["dev", "large"], default="dev")
    ap.add_argument("--ncols", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    import polars as pl

    from sktime.datatypes._vectorize import prepare_VectorizedDF

    grid = DEV_GRID if args.grid == "dev" else LARGE_GRID
    rows = []
    for n_series, n_time in grid:
        df = make_permuted(n_series, n_time, args.ncols)
        expected = first_appearance(df)
        flat = df.reset_index()
        nf = nw.from_native(flat, eager_only=True)
        pl_df = to_polars_long(df)
        npl = nw.from_native(pl_df, eager_only=True)
        vec, X_mi = prepare_VectorizedDF(df, iterate_as="Series", is_scitype="Panel")
        names = list(df.index.names)
        inst, tname = names[:-1], names[-1]
        # the schema freq the VectorizedDF already stores
        schema_freq = vec.freq

        def pd_level():
            return [(k, g.droplevel(0)) for k, g in df.groupby(level=0, sort=False)]

        def sk_items():
            return [(k, g) for k, _, g in vec.items(X=X_mi)]

        def pd_col():
            return list(flat.groupby("h0", sort=False))

        def nw_pd():
            return list(nf.group_by("h0"))

        def nw_pd_native():
            return [(k, nw.to_native(g)) for k, g in nf.group_by("h0")]

        def _recipe(infer):
            nfl = nw.from_native(df.reset_index(), eager_only=True)
            out = []
            for k, g in nfl.group_by(inst):
                s = nw.to_native(g.drop(inst)).set_index(tname)
                if infer:
                    if len(s.index) > 2:
                        s.index.freq = pd.infer_freq(s.index)
                elif schema_freq is not None:
                    s.index.freq = schema_freq
                out.append((k, s))
            return out

        def nw_pd_recipe():
            return _recipe(False)

        def nw_pd_recipe_infer():
            return _recipe(True)

        data_cols = list(df.columns)

        def nw_pd_recipe_ctor():
            # cheaper per-slice rebuild: no pandas drop()/set_index(); build the
            # DatetimeIndex directly and attach the schema freq via date_range when
            # the slice is exactly regular (O(n) numpy compare, no infer_freq)
            nfl = nw.from_native(df.reset_index(), eager_only=True)
            out = []
            for k, g in nfl.group_by(inst):
                gn = nw.to_native(g)
                tv = gn[tname].to_numpy()
                ix = pd.DatetimeIndex(tv, name=tname)
                if schema_freq is not None and len(tv) > 0:
                    cand = pd.date_range(tv[0], periods=len(tv), freq=schema_freq, name=tname, unit=ix.unit)
                    if np.array_equal(cand.asi8, ix.asi8):
                        ix = cand
                out.append(
                    (k, pd.DataFrame(gn[data_cols].to_numpy(), index=ix, columns=data_cols))
                )
            return out

        sys.path.insert(0, os.path.join(HERE, "..", "protos"))
        from x2_index_recipe import iter_slices, mi_to_long

        def nw_pd_recipe_fast():
            # protos/x2_index_recipe.py: mi_to_long + narwhals group_by + per-column
            # numpy -> DataFrame(copy=False) + freq via slices of one global grid
            flat_, schema_ = mi_to_long(df)
            return list(iter_slices(nw.from_native(flat_, eager_only=True), schema_))

        def nw_offsets_iloc():
            # narwhals only computes group boundaries (one group_by.agg), slices are
            # taken from the ORIGINAL MultiIndex frame by position (contiguous groups)
            nfl = nw.from_native(df.reset_index(), eager_only=True).with_row_index(
                "__r", order_by=None
            )
            b = nw.to_native(
                nfl.group_by(inst).agg(
                    nw.col("__r").min().alias("a"), nw.col("__r").max().alias("b")
                )
            )
            out = []
            for k, a, e in zip(b[inst[0]].to_numpy(), b["a"].to_numpy(), b["b"].to_numpy()):
                out.append((k, X_mi.iloc[a : e + 1].droplevel(0)))
            return out

        def pl_iter_mo():
            return list(pl_df.group_by("h0", maintain_order=True))

        def pl_partition():
            d = pl_df.partition_by("h0", maintain_order=True, as_dict=True)
            return list(d.items())

        def nw_pl():
            return list(npl.group_by("h0"))

        methods = {
            "pd_level": pd_level,
            "sk_items": sk_items,
            "pd_col": pd_col,
            "nw_pd": nw_pd,
            "nw_pd_native": nw_pd_native,
            "nw_pd_recipe": nw_pd_recipe,
            "nw_pd_recipe_infer": nw_pd_recipe_infer,
            "nw_pd_recipe_ctor": nw_pd_recipe_ctor,
            "nw_pd_recipe_fast": nw_pd_recipe_fast,
            "nw_offsets_iloc": nw_offsets_iloc,
            "pl_iter_mo": pl_iter_mo,
            "pl_partition": pl_partition,
            "nw_pl": nw_pl,
        }

        rec = {"n_series": n_series, "n_time": n_time, "n_cols": args.ncols}
        rec["n_rows"] = len(df)
        rec["schema_freq"] = str(schema_freq)
        info = {}
        for name, fn in methods.items():
            res = fn()
            keys = [norm_key(k) for k, _ in res]
            k0, g0 = res[0]
            ix = None
            if isinstance(g0, pd.DataFrame):
                ix = g0.index
            entry = {
                "order_first_appearance": keys == expected,
                "order_sorted": keys == sorted(expected),
                "key_type": type(k0).__name__
                + (f"[{type(k0[0]).__name__}]" if isinstance(k0, tuple) else ""),
                "slice_type": type(g0).__name__,
                "slice_index": None
                if ix is None
                else f"{type(ix).__name__}(name={ix.name}, freq={getattr(ix, 'freqstr', None)})",
            }
            if name in ("nw_pl",):
                runs = [[norm_key(k) for k, _ in fn()] for _ in range(5)]
                entry["order_stable_5_runs"] = all(r == runs[0] for r in runs)
                entry["order_first_appearance_all_runs"] = all(
                    r == expected for r in runs
                )
            info[name] = entry
            del res
        rec["semantics"] = info

        rep = args.repeat if len(df) <= 200_000 else max(2, args.repeat - 2)
        t = {}
        # interleave: run every method once per round, keep min
        for name in methods:
            t[name] = float("inf")
        for _ in range(rep):
            for name, fn in methods.items():
                t[name] = min(t[name], bench(fn, repeat=1, warmup=0))
        rec["t_ms"] = {k: round(v * 1e3, 3) for k, v in t.items()}
        rec["ratios"] = {
            "nw_pd/pd_col": round(t["nw_pd"] / t["pd_col"], 3),
            "nw_pd/pd_level": round(t["nw_pd"] / t["pd_level"], 3),
            "pd_col/pd_level": round(t["pd_col"] / t["pd_level"], 3),
            "nw_pd_native/pd_level": round(t["nw_pd_native"] / t["pd_level"], 3),
            "nw_pd_recipe/sk_items": round(t["nw_pd_recipe"] / t["sk_items"], 3),
            "nw_pd_recipe_infer/sk_items": round(
                t["nw_pd_recipe_infer"] / t["sk_items"], 3
            ),
            "nw_pd_recipe_ctor/sk_items": round(t["nw_pd_recipe_ctor"] / t["sk_items"], 3),
            "nw_pd_recipe_fast/sk_items": round(t["nw_pd_recipe_fast"] / t["sk_items"], 3),
            "nw_offsets_iloc/sk_items": round(t["nw_offsets_iloc"] / t["sk_items"], 3),
            "pl_partition/pd_level": round(t["pl_partition"] / t["pd_level"], 4),
            "nw_pl/pl_iter_mo": round(t["nw_pl"] / t["pl_iter_mo"], 3),
        }
        rows.append(rec)
        print(json.dumps(rec), flush=True)
        del df, flat, nf, pl_df, npl, vec, X_mi
        gc.collect()

    tag = "pd" + pd.__version__.split(".")[0]
    out = {"env": env_info(), "python": sys.version.split()[0], "rows": rows}
    path = os.path.join(RES, f"x2_nw_groupby_iter_bench_{tag}_{args.grid}_c{args.ncols}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\npandas {pd.__version__} narwhals {nw.__version__} grid={args.grid} ncols={args.ncols}")
    meths = list(rows[0]["t_ms"].keys())
    print(f"{'size':>12s} " + " ".join(f"{m[:12]:>12s}" for m in meths))
    for r in rows:
        print(
            f"{r['n_series']:>6d}x{r['n_time']:<5d} "
            + " ".join(f"{r['t_ms'][m]:>12.2f}" for m in meths)
        )
    print("\nratios")
    for r in rows:
        print(f"{r['n_series']:>6d}x{r['n_time']:<5d}", json.dumps(r["ratios"]))
    print("\nsemantics (largest size)")
    for m, e in rows[-1]["semantics"].items():
        print(f"  {m:20s} {json.dumps(e)}")
    print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
