"""a1 verify0: independent census cross-check (claims a1.1 / a1.2).

Method (deliberately different from a1_static_touchpoints.py):
 1. For each of the 229 rows of results/a1_touchpoints.json-derived table
    (results/a1_touchpoints_table.md), check the token text actually occurs
    on the cited source line (file:line accuracy).
 2. Token-level regex census (not AST) of `pd.<name>` references and
    `.<pandas-attr>` accesses for EVERY function in _base.py / _vectorize.py /
    _utilities.py, split into "in a1 scope" vs "out of a1 scope", to see how
    sensitive the 229 number is to the scope choice.

Run from WT root:
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
     experiments/container_independence/scripts/a1_core_coupling_verify0_census.py
"""

import ast
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "experiments/container_independence/scripts")
from a1_static_touchpoints import SCOPE  # noqa: E402  (only the scope list)

ED = "experiments/container_independence"

# ---- 1. line accuracy ------------------------------------------------------
rows = []
for ln in open(f"{ED}/results/a1_touchpoints_table.md"):
    m = re.match(r"\| (\d+) \| `([^`]+):(\d+)` \| `([^`]*)` \| `(.*?)` \|", ln)
    if m:
        rows.append((m.group(2), int(m.group(3)), m.group(4), m.group(5)))
src_cache = {}


def src(f):
    full = [k for k in SCOPE if k.endswith(f)][0]
    if full not in src_cache:
        src_cache[full] = open(full).read().splitlines()
    return src_cache[full]


bad = []
for f, line, func, tok in rows:
    text = src(f)[line - 1]
    probe = tok
    m = re.match(r"hasattr\(\.\., '(\w+)'\)", tok)
    if m:
        probe = m.group(1)
    probe = probe.strip("'").lstrip(".")
    # implicit/manual entries are free text; take first identifier
    ident = re.findall(r"[A-Za-z_][\w\-\.]*", probe)
    ok = any(i.split(".")[-1] in text for i in ident) if ident else False
    if not ok:
        bad.append((f, line, func, tok, text.strip()[:90]))
print(f"[line-accuracy] rows parsed: {len(rows)}  token not found on cited line: {len(bad)}")
for b in bad:
    print("   ", b)

# ---- 2. regex census per function, in-scope vs out-of-scope ---------------
PD_ATTR = (
    r"\.(groupby|droplevel|xs|loc|iloc|index|columns|names|nlevels|levels|pivot|"
    r"reindex|unique|iterrows|set_names|freq|freqstr|combine_first|isna|to_frame|nth|"
    r"agg|get_level_values|is_monotonic_increasing|sort_index|equals|isin|difference|"
    r"symmetric_difference|dtypes|to_list|from_product|from_tuples|from_arrays|"
    r"sort_values|stack|unstack|melt|merge|join|set_index|reset_index|swaplevel|"
    r"to_numpy|assign|apply|where|shift|astype)\b"
)
PD_REF = r"\bpd\.[A-Za-z_][\w\.]*"


def func_spans(path):
    tree = ast.parse(open(path).read())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append((f"{node.name}.{b.name}", b))
        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            out.append((node.name, node))
    return out


def strip_doc(lines, fn):
    body = fn.body
    skip = set()
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        skip = set(range(body[0].lineno, body[0].end_lineno + 1))
    return [
        (i, lines[i - 1])
        for i in range(fn.lineno, fn.end_lineno + 1)
        if i not in skip and not lines[i - 1].strip().startswith("#")
    ]


summary = {}
for path in [
    "sktime/forecasting/base/_base.py",
    "sktime/datatypes/_vectorize.py",
    "sktime/datatypes/_utilities.py",
]:
    lines = open(path).read().splitlines()
    wanted = SCOPE[path]
    per = defaultdict(Counter)
    for name, fn in func_spans(path):
        inscope = (
            "*" in wanted
            or name in wanted
            or name.split(".")[-1] in wanted
            and "." not in name
        )
        n_pd = n_attr = 0
        for _, t in strip_doc(lines, fn):
            code = t.split("#")[0]
            n_pd += len(re.findall(PD_REF, code))
            n_attr += len([m for m in re.findall(PD_ATTR, code)])
            # remove self.<attr> false positives
            n_attr -= len(re.findall(r"self\.(index|columns|names|freq|levels|name)\b", code))
        key = "in" if inscope else "out"
        per[key][name] = n_pd + max(n_attr, 0)
    tot_in = sum(per["in"].values())
    tot_out = sum(per["out"].values())
    top_out = per["out"].most_common(12)
    summary[path] = {"in_scope_hits": tot_in, "out_of_scope_hits": tot_out, "top_out": top_out}
    print(f"\n[{path}] regex hits in a1 scope: {tot_in}   out of a1 scope: {tot_out}")
    for n, c in top_out:
        if c:
            print(f"    out-of-scope {n}: {c}")

json.dump(
    {"line_accuracy_bad": bad, "n_rows": len(rows), "regex_census": summary},
    open(f"{ED}/results/a1_core_coupling_verify0_census.json", "w"),
    indent=1,
)
