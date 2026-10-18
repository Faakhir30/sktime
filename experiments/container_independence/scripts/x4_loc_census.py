"""x4: code-size and coupling census for the adapter approach.

1. LOC per adapter (logical source lines: no blanks, comments, docstrings),
   protocol method count, protocol conformance (runtime_checkable isinstance).
2. pandas data-API touch points in BaseForecaster vectorization/cutoff code and in
   the current VectorizedDF (regex census, with line numbers).
3. Census of forecaster inner mtypes via the sktime registry (which backend family
   do the forecasters compute in? -> where the adapter approach stops).

Usage:
  PYTHONPATH=.:experiments/container_independence:experiments/container_independence/protos \
    $EXP_PY -W ignore experiments/container_independence/scripts/x4_loc_census.py [--no-registry]
"""

import argparse
import ast
import io
import json
import os
import re
import sys
import tokenize
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

ED = os.path.join(os.path.dirname(__file__), "..")
WT = os.path.abspath(os.path.join(ED, "..", ".."))
sys.path.insert(0, os.path.join(ED, "protos"))


def logical_loc(src):
    """Count source lines that carry code (exclude blank, comments, docstrings)."""
    tree = ast.parse(src)
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    code_lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        for ln in range(tok.start[0], tok.end[0] + 1):
            if ln not in doc_lines:
                code_lines.add(ln)
    return len(code_lines)


def class_sources(path):
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines()
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            seg = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            import textwrap

            seg = textwrap.dedent(seg)
            n_methods = sum(isinstance(b, ast.FunctionDef) for b in node.body) if isinstance(node, ast.ClassDef) else None
            out[node.name] = {"loc": logical_loc(seg), "methods": n_methods, "lines": [node.lineno, node.end_lineno]}
    return out


PANDAS_PATTERNS = {
    "pd.<api>": r"\bpd\.(concat|DataFrame|Series|Index|MultiIndex|infer_freq|RangeIndex|DatetimeIndex|PeriodIndex)\b",
    ".groupby(": r"\.groupby\(",
    ".droplevel(": r"\.droplevel\(",
    ".loc/.iloc/.xs": r"\.(loc|iloc|xs)\[|\.xs\(",
    ".index (attr)": r"\.index\b(?!\()",
    ".columns (attr)": r"\.columns\b",
    ".set_names/.set_index/.reset_index": r"\.(set_names|set_index|reset_index)\(",
    ".iterrows/.pivot/.reindex": r"\.(iterrows|pivot|reindex)\(",
    "isinstance(..., pd.X)": r"isinstance\([^)]*pd\.",
}


def pandas_touch(path, line_range=None):
    lines = open(path).read().splitlines()
    lo, hi = line_range or (1, len(lines))
    hits = []
    counts = Counter()
    in_doc = False
    for i in range(lo - 1, hi):
        ln = lines[i]
        s = ln.strip()
        # crude docstring/comment skip
        if s.startswith("#"):
            continue
        if s.count('"""') == 1:
            in_doc = not in_doc
            continue
        if in_doc or (s.startswith('"""') and s.endswith('"""')):
            continue
        for name, pat in PANDAS_PATTERNS.items():
            n = len(re.findall(pat, ln))
            if n:
                counts[name] += n
                hits.append((i + 1, name, s[:110]))
    return {"counts": dict(counts), "total": sum(counts.values()), "lines_with_hits": len({h[0] for h in hits}), "hits": hits}


def func_range(path, cls, func):
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for b in node.body:
                if isinstance(b, ast.FunctionDef) and b.name == func:
                    return b.lineno, b.end_lineno
    return None


def registry_census():
    import psutil

    from sktime.registry import all_estimators

    ests = all_estimators("forecaster", return_names=True, return_tags=["y_inner_mtype", "X_inner_mtype"])
    fam = Counter()
    y_mt = Counter()
    per = []
    for name, _cls, y_inner, x_inner in ests:
        ys = y_inner if isinstance(y_inner, list) else [y_inner]
        y_mt.update(ys)
        libs = set()
        for m in ys:
            if m is None:
                continue
            if m.startswith("pd") or m in ("pd-multiindex", "pd_multiindex_hier"):
                libs.add("pandas")
            elif m.startswith("pl") or m.startswith("polars"):
                libs.add("polars")
            elif m.startswith("np") or m == "numpy3D":
                libs.add("numpy")
            else:
                libs.add(m)
        key = "+".join(sorted(libs))
        fam[key] += 1
        per.append((name, ys))
    accepts_panel_natively = sum(
        1 for _, ys in per if any(m in ("pd-multiindex", "pd_multiindex_hier", "polars_panel", "polars_hierarchical", "numpy3D") for m in ys)
    )
    return {
        "n_forecasters": len(ests),
        "y_inner_family_counts": dict(fam.most_common()),
        "y_inner_mtype_counts": dict(y_mt.most_common()),
        "n_with_any_polars_inner": sum(1 for _, ys in per if any(str(m).startswith(("pl", "polars")) for m in ys)),
        "n_pandas_only": fam.get("pandas", 0),
        "n_declaring_panel_or_hier_inner": accepts_panel_natively,
        "rss_mb_after_import_all": psutil.Process().memory_info().rss / 2**20,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-registry", action="store_true")
    args = ap.parse_args()
    from x4_adapter import ADAPTER_CLASSES, PROTOCOL_METHODS, REGISTRY, ContainerAdapter

    res = {}
    proto = os.path.join(ED, "protos", "x4_adapter.py")
    cs = class_sources(proto)
    res["proto_classes"] = cs
    res["protocol_n_methods"] = len(PROTOCOL_METHODS)
    res["protocol_conformance"] = {
        lib: isinstance(ad, ContainerAdapter) for lib, ad in REGISTRY.by_lib.items()
    }
    res["adapter_loc"] = {c.__name__: cs[c.__name__]["loc"] for c in ADAPTER_CLASSES}
    res["generic_loc"] = {k: cs[k]["loc"] for k in ["AdapterRegistry", "VectorizedDF2", "prepare_VectorizedDF2", "adapter_get_cutoff", "family_of_inner_mtype"]}
    cur = class_sources(os.path.join(WT, "sktime", "datatypes", "_vectorize.py"))
    res["current_vectorize_loc"] = {k: v["loc"] for k, v in cur.items()}

    # existing sktime polars *mtype family* (Series/Panel/Hierarchical; Table excluded)
    dt = os.path.join(WT, "sktime", "datatypes")
    fam = {}
    fam["_adapter/polars.py (shared check/convert helpers)"] = logical_loc(open(os.path.join(dt, "_adapter", "polars.py")).read())
    for rel, names in [
        ("_series/_check.py", ["SeriesPolarsEager"]),
        ("_panel/_check.py", ["PanelPolarsEager", "_check_polars_panel"]),
        ("_hierarchical/_check.py", ["HierarchicalPolarsEager"]),
    ]:
        cs_ = class_sources(os.path.join(dt, rel))
        for nm in names:
            fam[f"{rel}:{nm}"] = cs_[nm]["loc"]
    for rel in ["_series/_convert.py", "_panel/_convert.py", "_hierarchical/_convert.py"]:
        src = open(os.path.join(dt, rel)).read()
        tree = ast.parse(src)
        lines = src.splitlines()
        for node in tree.body:
            if isinstance(node, ast.If) and "polars" in ast.unparse(node.test):
                import textwrap

                seg = textwrap.dedent("\n".join(lines[node.body[0].lineno - 1 : node.end_lineno]))
                fam[f"{rel}:polars converter block"] = logical_loc(seg)
    res["polars_mtype_family_loc"] = fam
    res["polars_mtype_family_loc_total"] = sum(fam.values())
    print("polars mtype family LOC:", fam, "total", res["polars_mtype_family_loc_total"])

    base = os.path.join(WT, "sktime", "forecasting", "base", "_base.py")
    res["pandas_touch"] = {
        "_vectorize.py (whole)": pandas_touch(os.path.join(WT, "sktime", "datatypes", "_vectorize.py")),
        "BaseForecaster._vectorize": pandas_touch(base, func_range(base, "BaseForecaster", "_vectorize")),
        "BaseForecaster._check_X_y": pandas_touch(base, func_range(base, "BaseForecaster", "_check_X_y")),
        "BaseForecaster._set_cutoff": pandas_touch(base, func_range(base, "BaseForecaster", "_set_cutoff")),
        "BaseForecaster._check_fh": pandas_touch(base, func_range(base, "BaseForecaster", "_check_fh")),
        "_base.py (whole)": pandas_touch(base),
    }
    for k, v in res["pandas_touch"].items():
        print(f"{k:32s} total={v['total']:4d} lines={v['lines_with_hits']:4d} {v['counts']}")
    for h in res["pandas_touch"]["BaseForecaster._vectorize"]["hits"]:
        print("   _vectorize hit:", h)
    print("adapter LOC:", res["adapter_loc"])
    print("generic LOC:", res["generic_loc"])
    print("current _vectorize.py LOC:", res["current_vectorize_loc"])
    print("protocol methods:", res["protocol_n_methods"], res["protocol_conformance"])

    if not args.no_registry:
        try:
            res["registry_census"] = registry_census()
            print("registry census:", json.dumps(res["registry_census"], indent=1))
        except Exception as e:  # noqa: BLE001
            res["registry_census"] = {"error": repr(e)}
            print("registry census failed", e)

    with open(os.path.join(ED, "results", "x4_loc_census.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
