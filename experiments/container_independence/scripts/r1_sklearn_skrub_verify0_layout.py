"""verify0: independent check of r1 claim 8 (layout, not container, drives StandardScaler speed;
egress to polars cost is layout-dependent). Different code path: direct timing of
(a) polars->to_numpy F vs forced-C copy fed to StandardScaler, (b) pl.DataFrame(arr) construction
from C vs F arrays, (c) the PolarsAdapter.create_container directly.
Run: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore <this> --grid dev"""
import argparse, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, polars as pl
from common import bench
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import ADAPTERS_MANAGER
ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="dev", choices=["dev", "large"])
a = ap.parse_args()
sizes = [100_000, 500_000] if a.grid == "dev" else [500_000, 5_000_000]
pa = ADAPTERS_MANAGER.adapters["polars"]
res = []
rng = np.random.default_rng(1)
for n in sizes:
    A = rng.normal(size=(n, 4)); cols = [f"c{i}" for i in range(4)]
    P = pl.DataFrame(A, schema=cols, orient="row")
    AF = np.asfortranarray(A)
    out_from_polars = StandardScaler().fit_transform(P)
    fns = {
        "SS(numpy C)": lambda: StandardScaler().fit_transform(A),
        "SS(numpy F)": lambda: StandardScaler().fit_transform(AF),
        "SS(polars)": lambda: StandardScaler().fit_transform(P),
        "SS(ascontig(polars.to_numpy()))": lambda: StandardScaler().fit_transform(np.ascontiguousarray(P.to_numpy())),
        "pl.DataFrame(C, orient=row)": lambda: pl.DataFrame(A, orient="row"),
        "pl.DataFrame(F, orient=row)": lambda: pl.DataFrame(AF, orient="row"),
        "PolarsAdapter.create_container(C)": lambda: pa.create_container(A, None, cols),
        "PolarsAdapter.create_container(F)": lambda: pa.create_container(AF, None, cols),
    }
    best = {k: float("inf") for k in fns}
    for r in range(5):
        for k, f in fns.items():
            best[k] = min(best[k], bench(f, repeat=1, warmup=1 if r == 0 else 0))
    base = best["SS(numpy C)"]
    row = {"n": n, "polars_output_is_F": bool(out_from_polars.flags["F_CONTIGUOUS"]),
           **{k: {"ms": round(v * 1e3, 2), "x": round(v / base, 2)} for k, v in best.items()}}
    res.append(row); print(json.dumps(row))
