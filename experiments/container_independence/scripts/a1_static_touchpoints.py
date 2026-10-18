"""a1: static (AST) census of pandas touchpoints in the vectorization + BaseForecaster
boundary layer, with a hand-curated classification of every hit.

What is detected (syntactically, in the in-scope functions only, docstrings excluded):

1. ``pd.<...>`` references (module-level pandas API, incl. inside ``isinstance``)
2. attribute / method accesses whose name is pandas-specific API
   (``groupby``, ``droplevel``, ``loc``, ``iloc``, ``index``, ``columns``, ``names``,
   ``nlevels``, ``pivot``, ``reindex``, ``iterrows``, ``freq``, ... see PANDAS_ATTRS)
3. ``hasattr/getattr(obj, "<pandas attr name>")`` duck-typing on pandas attributes
4. string literals that pin a pandas mtype ("pd.DataFrame", "pd-multiindex", ...)
   in code (not docstrings) -> "mtype-pinning"
5. a short list of IMPLICIT pandas semantics that are not visible by name
   (index-aligned arithmetic, ``len()`` of a frame, ...) -> MANUAL entries

Every detected hit must have an entry in CLASSIFY (keyed by file, line, api);
unclassified hits are reported and make the script exit non-zero, so the census
stays complete if the code changes.

Run (from WT root):
    PYTHONPATH=.:experiments/container_independence $EXP_PY \
        experiments/container_independence/scripts/a1_static_touchpoints.py
Outputs:
    results/a1_touchpoints.json, results/a1_touchpoints_table.md
"""

import ast
import json
import os
import sys
from collections import Counter, defaultdict

ED = "experiments/container_independence"

# ---------------------------------------------------------------------------
# scope: file -> list of function qualnames ("*" = whole file)
# ---------------------------------------------------------------------------
SCOPE = {
    "sktime/datatypes/_vectorize.py": ["*"],
    "sktime/forecasting/base/_base.py": [
        "BaseForecaster.fit",
        "BaseForecaster.predict",
        "BaseForecaster.fit_predict",
        "BaseForecaster.predict_quantiles",
        "BaseForecaster.predict_interval",
        "BaseForecaster.predict_var",
        "BaseForecaster.predict_proba",
        "BaseForecaster.pretrain",
        "BaseForecaster.update",
        "BaseForecaster.update_predict",
        "BaseForecaster.update_predict_single",
        "BaseForecaster.predict_residuals",
        "BaseForecaster.get_fitted_params",
        "BaseForecaster._check_X_y",
        "BaseForecaster._check_X",
        "BaseForecaster._update_X",
        "BaseForecaster._update_y_X",
        "BaseForecaster.cutoff",
        "BaseForecaster._set_cutoff",
        "BaseForecaster._set_cutoff_from_y",
        "BaseForecaster._check_fh",
        "BaseForecaster._vectorize",
        "BaseForecaster._update",
        "BaseForecaster._predict_moving_cutoff",
        "BaseForecaster._get_varnames",
        "BaseForecaster._get_columns",
        "_format_moving_cutoff_predictions",
        "<class-attrs>",  # _tags defaults (y_inner_mtype etc.)
    ],
    "sktime/datatypes/_convert.py": [
        "convert",
        "convert_to",
        "_get_first_mtype_of_same_scitype",
        "_check_str_or_list_of_str",
    ],
    "sktime/datatypes/_check.py": [
        "check_is_scitype",
        "mtype",
        "_coerce_list_of_str",
        "check_is_error_msg",
    ],
    "sktime/utils/multiindex.py": ["flatten_multiindex", "underscore_join"],
    "sktime/datatypes/_utilities.py": [
        "_get_index",
        "get_time_index",
        "get_index_for_series",
        "_get_cutoff_from_index",
        "get_cutoff",
        "update_data",
        "<module-consts>",
    ],
    "sktime/utils/validation/series.py": ["check_equal_time_index"],
    "sktime/utils/parallel.py": ["*"],
}

PANDAS_ATTRS = {
    "groupby", "droplevel", "xs", "loc", "iloc", "index", "columns", "names",
    "nlevels", "levels", "pivot", "reindex", "unique", "iterrows", "set_names",
    "freq", "freqstr", "combine_first", "isna", "to_frame", "nth", "agg",
    "get_level_values", "is_monotonic_increasing", "sort_index", "equals", "isin",
    "difference", "symmetric_difference", "values", "dtypes", "to_list",
    "from_product", "from_tuples", "from_arrays", "name", "sort_values",
}
# attribute names that are pandas-specific only when NOT accessed on self/meta/etc.
NON_PANDAS_BASES = {"self", "np", "pd", "meta", "metadata", "kwargs", "args", "d"}

PANDAS_MTYPE_STRINGS = {
    "pd.Series", "pd.DataFrame", "pd-multiindex", "pd_multiindex_hier",
    "pd-wide", "pd-long", "df-list", "nested_univ",
}
DUCK_ATTRS = {"index", "freq", "columns", "X", "freqstr", "levels"}


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _base_name(node):
    """Leftmost Name id of an attribute chain, or None."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.value if not isinstance(node, ast.Call) else node.func
    return node.id if isinstance(node, ast.Name) else None


class Scanner(ast.NodeVisitor):
    def __init__(self, relpath, wanted, src_lines):
        self.relpath = relpath
        self.wanted = wanted
        self.stack = []
        self.hits = []
        self.src = src_lines
        self.docstring_nodes = set()

    # ---- scope handling -------------------------------------------------
    def _qual(self):
        return ".".join(self.stack)

    def _in_scope(self):
        if "*" in self.wanted:
            return True
        q = self._qual()
        return any(q == w or q.startswith(w + ".") for w in self.wanted)

    def _mark_docstring(self, node):
        body = getattr(node, "body", [])
        if body and isinstance(body[0], ast.Expr) and isinstance(
            body[0].value, ast.Constant
        ):
            if isinstance(body[0].value.value, str):
                self.docstring_nodes.add(id(body[0].value))

    def visit_Module(self, node):
        self._mark_docstring(node)
        # module-level constants
        if "<module-consts>" in self.wanted:
            for st in node.body:
                if isinstance(st, ast.Assign):
                    self.stack = ["<module-consts>"]
                    self._scan(st)
                    self.stack = []
        for st in node.body:
            if isinstance(st, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                self.visit(st)
            elif "*" in self.wanted and not isinstance(st, ast.Assign):
                self._scan(st)
            elif "*" in self.wanted and isinstance(st, ast.Assign):
                self.stack = ["<module>"]
                self._scan(st)
                self.stack = []

    def visit_ClassDef(self, node):
        self._mark_docstring(node)
        self.stack.append(node.name)
        if "<class-attrs>" in self.wanted:
            for st in node.body:
                if isinstance(st, ast.Assign):
                    saved = list(self.stack)
                    self.stack = ["<class-attrs>"]
                    self._scan(st)
                    self.stack = saved
        for st in node.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(st)
            elif "*" in self.wanted:
                self._scan(st)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self._mark_docstring(node)
        self.stack.append(node.name)
        if self._in_scope():
            for st in node.body:
                self._scan(st)
        else:
            # nested defs might still be in scope (not used here)
            pass
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    # ---- detection ------------------------------------------------------
    def _add(self, node, api, kind):
        self.hits.append(
            {
                "file": self.relpath,
                "line": node.lineno,
                "func": self._qual(),
                "api": api,
                "kind": kind,
                "code": self.src[node.lineno - 1].strip()[:110],
            }
        )

    def _scan(self, root):
        # mark nested function docstrings
        for n in ast.walk(root):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._mark_docstring(n)
        seen_pd_chain = set()
        for n in ast.walk(root):
            # 1. pd.<...> chains (only report the outermost chain)
            if isinstance(n, ast.Attribute):
                d = _dotted(n)
                if d and d.startswith("pd.") and id(n) not in seen_pd_chain:
                    # mark inner chain nodes as seen
                    inner = n.value
                    while isinstance(inner, ast.Attribute):
                        seen_pd_chain.add(id(inner))
                        inner = inner.value
                    self._add(n, d, "pd-ref")
                    continue
                if id(n) in seen_pd_chain:
                    continue
                # 2. pandas-specific attribute names
                if n.attr in PANDAS_ATTRS:
                    base = _base_name(n)
                    direct = isinstance(n.value, ast.Name) and n.value.id in (
                        NON_PANDAS_BASES
                    )
                    if not direct and base not in ("pd", "np"):
                        self._add(n, "." + n.attr, "attr")
            # 3. hasattr/getattr duck typing
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in ("hasattr", "getattr")
                and len(n.args) >= 2
                and isinstance(n.args[1], ast.Constant)
                and n.args[1].value in DUCK_ATTRS
            ):
                self._add(n, f"{n.func.id}(.., '{n.args[1].value}')", "duck")
            # 4. mtype string pinning
            if (
                isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and n.value in PANDAS_MTYPE_STRINGS
                and id(n) not in self.docstring_nodes
            ):
                self._add(n, f"'{n.value}'", "mtype-str")


def scan_all():
    hits = []
    for rel, wanted in SCOPE.items():
        with open(rel) as f:
            src = f.read()
        tree = ast.parse(src)
        sc = Scanner(rel, wanted, src.splitlines())
        sc.visit(tree)
        hits.extend(sc.hits)
    return hits


# ---------------------------------------------------------------------------
# classification. Categories:
#   G  container-op: group/iterate
#   S  container-op: slice/select
#   C  container-op: concat/reconstruct
#   P  container-op: pivot/reshape
#   L  index-semantics: MultiIndex levels/names
#   T  index-semantics: time index/freq/cutoff
#   M  metadata: columns/dtypes/shape/nans
#   O  output-construction (building pd objects: interval/quantile cols, results)
#   D  type-dispatch: isinstance / duck-typing checks
#   K  mtype-pinning (pandas mtype names hard-coded as conversion targets)
#   X  not-pandas (false positive of the name heuristic; kept for transparency)
# value: (category, abstract operation, api-contract?)  api-contract=True means the
# touchpoint produces/consumes pandas as a value type in the PUBLIC API.
# key: (file-suffix, line, api)
# ---------------------------------------------------------------------------
V = "_vectorize.py"
B = "_base.py"
CV = "_convert.py"
U = "_utilities.py"
MI = "multiindex.py"
SE = "series.py"

CLASSIFY = {
    # ---------------- _vectorize.py ----------------
    (V, 25, "'pd-multiindex'"): ("K", "target mtype for Panel frame", 0),
    (V, 26, "'pd_multiindex_hier'"): ("K", "target mtype for Hierarchical frame", 0),
    (V, 27, "'pd_multiindex_hier'"): ("K", "target mtype for Hierarchical frame", 0),
    (V, 100, ".freqstr"): ("T", "freq(time_index)", 0),
    (V, 369, "inst[[col]]"): ("S", "select_columns(slice, [col])", 0),
    (V, 767, ".shape"): ("P", "estimator grid shape check", 1),
    (B, 1942, "'pd-wide'"): ("X", "mtype listed in error message only", 0),
    (B, 1947, ".difference"): ("X", "set.difference (python set)", 0),
    (B, 2645, ".values"): ("X", "dict.values()", 0),
    (B, 2722, "_shift"): ("T", "shift cutoff by -1 period (pandas freq arithmetic)", 0),
    (B, 2725, "cv.split(y)"): ("T", "splitter windows over y's pandas time index", 0),
    (B, 2860, "cutoff[0]"): ("T", "unwrap length-1 pd.Index cutoff", 1),
    (U, 153, "ss[ix]"): ("T", "positional element of time index", 0),
    (U, 165, "time_idx[ix]"): ("T", "positional element of time index", 0),
    (U, 166, "time_idx[[ix]]"): ("T", "length-1 index slice (keeps type)", 1),
    (U, 247, "cutoff[0]"): ("T", "unwrap length-1 pd.Index cutoff", 1),
    (U, 278, "idx[ix]"): ("T", "positional element of time index", 0),
    (U, 279, "idx[[ix]]"): ("T", "length-1 index slice (keeps type)", 1),
    (U, 414, "'pd.DataFrame'"): ("X", "get_window consts (out of scope)", 0),
    (U, 415, "'pd-multiindex'"): ("X", "get_window consts (out of scope)", 0),
    (U, 416, "'pd_multiindex_hier'"): ("X", "get_window consts (out of scope)", 0),
    (V, 24, "'pd.DataFrame'"): ("K", "target mtype for Series frame", 0),
    (V, 27, "'pd.DataFrame'"): ("K", "target mtype for Series frame", 0),
    (V, 27, "'pd-multiindex'"): ("K", "target mtype for Panel frame", 0),
    (V, 90, "len"): ("M", "n_rows(frame)", 0),
    (V, 93, ".index"): ("L", "row keys of frame", 0),
    (V, 94, "pd.MultiIndex"): ("D", "is_multiindexed(frame)", 0),
    (V, 95, ".xs"): ("S", "select_instance(frame, key) -> time slice", 0),
    (V, 95, ".droplevel"): ("L", "instance_keys(frame)", 0),
    (V, 95, ".index"): ("T", "time_index(slice)", 0),
    (V, 99, "getattr(.., 'freqstr')"): ("T", "freq(time_index)", 0),
    (V, 102, "pd.infer_freq"): ("T", "infer_freq(time_index)", 0),
    (V, 241, ".columns"): ("M", "column_names(frame)", 0),
    (V, 242, ".names"): ("L", "index_level_names(frame)", 0),
    (V, 242, ".index"): ("L", "row keys of frame", 0),
    (V, 243, ".nlevels"): ("L", "n_index_levels(frame)", 0),
    (V, 243, ".index"): ("L", "row keys of frame", 0),
    (V, 244, ".index"): ("L", "row keys of frame", 0),
    (V, 292, ".droplevel"): ("L", "instance_keys(frame, levels) (unique)", 0),
    (V, 292, ".unique"): ("L", "instance_keys(frame, levels) (unique)", 0),
    (V, 294, ".droplevel"): ("L", "instance_keys(frame, levels) (unique)", 0),
    (V, 294, ".unique"): ("L", "instance_keys(frame, levels) (unique)", 0),
    (V, 368, ".columns"): ("M", "column_names(slice)", 0),
    (V, 396, ".groupby"): ("G", "iter_groups(frame, levels) -> (key, slice)", 0),
    (V, 397, ".droplevel"): ("L", "drop instance levels from slice", 0),
    (V, 516, "pd.DataFrame"): ("D", "is_native_frame(x)", 0),
    (V, 536, "pd.DataFrame"): ("C", "identity reconstruct (single slice)", 0),
    (V, 538, "pd.concat"): ("C", "concat_rows(slices, keys=instance_keys)", 0),
    (V, 545, "pd.concat"): ("C", "concat_cols(slices, keys=col_keys)", 0),
    (V, 557, "pd.concat"): ("C", "concat_cols(slices, keys=col_keys)", 0),
    (V, 559, "pd.concat"): ("C", "concat_rows(slices, keys=instance_keys)", 0),
    (V, 527, ".columns"): ("M", "column_names(slice)", 0),
    (V, 529, ".columns"): ("M", "column_names(slice)", 0),
    (V, 561, ".index"): ("L", "row keys of frame", 0),
    (V, 564, "pd.MultiIndex"): ("D", "has_multiindex_columns(frame)", 0),
    (V, 564, ".columns"): ("M", "column_names(frame)", 0),
    (V, 565, ".columns"): ("O", "flatten column names 'a__b'", 0),
    (V, 567, ".names"): ("L", "index_level_names(frame)", 0),
    (V, 568, ".index"): ("L", "set index level names", 0),
    (V, 568, ".set_names"): ("L", "set index level names", 0),
    (V, 726, "'pd.DataFrame'"): ("K", "return_type selector of vectorize_est", 1),
    (V, 766, "pd.DataFrame"): ("D", "is estimator grid a DataFrame", 1),
    (V, 773, ".iterrows"): ("P", "iterate estimator grid cells row-major", 1),
    (V, 773, ".values"): ("P", "iterate estimator grid cells row-major", 1),
    (V, 801, "'pd.DataFrame'"): ("K", "return_type selector of vectorize_est", 1),
    (V, 802, "pd.DataFrame"): ("P", "long list of (row,col,est) -> table", 1),
    (V, 803, ".loc"): ("P", "unique col keys in order", 1),
    (V, 803, ".unique"): ("P", "unique col keys in order", 1),
    (V, 804, ".loc"): ("P", "unique row keys in order", 1),
    (V, 804, ".unique"): ("P", "unique row keys in order", 1),
    (V, 806, ".pivot"): ("P", "pivot long -> estimator grid", 1),
    (V, 810, ".reindex"): ("P", "restore column order after pivot", 1),
    (V, 811, ".reindex"): ("P", "restore row order after pivot", 1),
    (V, 814, ".index"): ("O", "clear index names of grid", 1),
    (V, 814, ".names"): ("O", "clear index names of grid", 1),
    (V, 815, ".columns"): ("O", "clear column name of grid", 1),
    (V, 815, ".name"): ("O", "clear column name of grid", 1),
    (V, 819, ".index"): ("O", "grid row index -> MultiIndex of tuples", 1),
    (V, 819, "pd.MultiIndex.from_tuples"): ("O", "grid row index -> MultiIndex", 1),
    (V, 866, "hasattr(.., 'freq')"): ("T", "freq(time_index of slice)", 0),
    (V, 866, ".index"): ("T", "time_index(slice)", 0),
    (V, 866, ".freq"): ("T", "freq(time_index of slice)", 0),
    (V, 867, ".index"): ("T", "time_index(slice)", 0),
    (V, 868, ".index"): ("T", "time_index(slice)", 0),
    (V, 868, ".freq"): ("T", "set freq on slice time index (per slice!)", 0),
    (V, 868, "pd.infer_freq"): ("T", "infer_freq per slice", 0),
    # ---------------- _base.py ----------------
    (B, 115, "'pd.Series'"): ("K", "default y_inner_mtype tag", 1),
    (B, 116, "'pd.DataFrame'"): ("K", "default X_inner_mtype tag", 1),
    (B, 603, "convert_to"): ("K", "back-convert prediction to last-seen mtype", 0),
    (B, 1187, "'pd-multiindex'"): ("K", "pretrain forces pandas panel mtypes", 0),
    (B, 1187, "'pd_multiindex_hier'"): ("K", "pretrain forces pandas panel mtypes", 0),
    (B, 1715, "pd.DataFrame"): ("D", "is y index-carrying (pandas)", 0),
    (B, 1715, "pd.Series"): ("D", "is y index-carrying (pandas)", 0),
    (B, 1716, ".index"): ("T", "fh := time index of y (absolute)", 1),
    (B, 1735, "y - y_pred"): ("C", "index-ALIGNED subtraction (residuals)", 1),
    (B, 1831, ".index"): ("P", "grid row keys", 1),
    (B, 1831, ".columns"): ("P", "grid col keys", 1),
    (B, 1832, ".loc"): ("P", "grid cell lookup by (row,col) key", 1),
    (B, 1833, "'forecasters.loc[...]'"): (
        "O",
        "fitted-param KEY naming embeds pandas .loc syntax",
        1,
    ),
    (B, 2073, "check_equal_time_index"): ("T", "X time index contains y's", 0),
    (B, 2214, "pd.Index"): ("D", "is cutoff a pd.Index", 1),
    (B, 2215, "pd.Index"): ("T", "cutoff stored as length-1 pd.Index", 1),
    (B, 2459, ".columns"): ("P", "n_cols of estimator grid", 1),
    (B, 2466, ".columns"): ("O", "drop helper column level of predictions", 0),
    (B, 2466, ".droplevel"): ("O", "drop helper column level of predictions", 0),
    (B, 2726, ".iloc"): ("S", "positional row slice of y (cv window)", 0),
    (B, 2797, "pd.MultiIndex.from_product"): (
        "O",
        "predict_interval columns (var, coverage, lower/upper)",
        1,
    ),
    (B, 2805, "pd.MultiIndex.from_product"): (
        "O",
        "predict_quantiles columns (var, alpha)",
        1,
    ),
    (B, 2836, "pd.DataFrame"): ("O", "empty moving-cutoff result", 1),
    (B, 2837, "pd.DataFrame"): ("D", "is native frame/series", 0),
    (B, 2837, "pd.Series"): ("D", "is native frame/series", 0),
    (B, 2841, "pd.DataFrame"): ("D", "is native frame", 0),
    (B, 2842, ".columns"): ("M", "column_names", 0),
    (B, 2851, "pd.DataFrame"): ("D", "is native frame", 0),
    (B, 2853, ".columns"): ("M", "column_names", 0),
    (B, 2853, ".equals"): ("M", "column_names equal", 0),
    (B, 2858, "pd.concat"): ("C", "concat_rows(predictions)", 1),
    (B, 2861, "pd.concat"): ("C", "concat_cols(predictions, keys=cutoffs)", 1),
    (B, 2863, ".index"): ("T", "is time index sorted", 0),
    (B, 2863, ".is_monotonic_increasing"): ("T", "is time index sorted", 0),
    (B, 2864, ".sort_index"): ("S", "sort rows by index", 0),
    (B, 2866, "hasattr(.., 'columns')"): ("D", "duck: has columns", 0),
    (B, 2866, "pd.MultiIndex"): ("D", "has_multiindex_columns", 0),
    (B, 2866, ".columns"): ("M", "column_names", 0),
    (B, 2867, ".columns"): ("M", "column_names", 0),
    (B, 2868, ".loc"): ("S", "select_columns (reorder to original order)", 0),
    # ---------------- _utilities.py ----------------
    (U, 9, "hasattr(.., 'index')"): ("D", "duck: has index", 0),
    (U, 10, ".index"): ("T", "time_index", 0),
    (U, 13, "pd.RangeIndex"): ("T", "synthesize RangeIndex for numpy", 0),
    (U, 15, "pd.RangeIndex"): ("T", "synthesize RangeIndex for numpy", 0),
    (U, 17, "pd.RangeIndex"): ("T", "synthesize RangeIndex for numpy", 0),
    (U, 38, "pd.DataFrame"): ("D", "is pandas", 0),
    (U, 38, "pd.Series"): ("D", "is pandas", 0),
    (U, 40, ".index"): ("L", "row keys", 0),
    (U, 40, "pd.MultiIndex"): ("D", "is multiindexed", 0),
    (U, 41, ".loc"): ("S", "select first instance", 0),
    (U, 41, ".index"): ("T", "time_index of first instance", 0),
    (U, 43, ".iloc"): ("S", "first cell (nested)", 0),
    (U, 43, "pd.DataFrame"): ("D", "is nested frame", 0),
    (U, 44, ".iloc"): ("S", "first cell (nested)", 0),
    (U, 46, ".iloc"): ("S", "first cell (nested)", 0),
    (U, 46, "pd.DataFrame"): ("D", "is nested frame", 0),
    (U, 46, "pd.Series"): ("D", "is nested series", 0),
    (U, 47, ".iloc"): ("S", "first cell (nested)", 0),
    (U, 50, ".index"): ("T", "time_index", 0),
    (U, 55, "pd.RangeIndex"): ("T", "synthesize RangeIndex for numpy", 0),
    (U, 58, "pd.RangeIndex"): ("T", "synthesize RangeIndex for numpy", 0),
    (U, 59, "hasattr(.., 'X')"): ("D", "duck: legacy VectorizedDF.X (stale)", 0),
    (U, 87, "hasattr(.., 'index')"): ("D", "duck: has index (else RangeIndex)", 0),
    (U, 88, ".index"): ("T", "time_index", 0),
    (U, 90, "pd.RangeIndex"): ("T", "synthesize RangeIndex for numpy", 0),
    (U, 94, "'pd.DataFrame'"): ("K", "get_cutoff internal mtypes", 0),
    (U, 95, "'pd.Series'"): ("K", "get_cutoff internal mtypes", 0),
    (U, 97, "'pd-multiindex'"): ("K", "get_cutoff internal mtypes", 0),
    (U, 99, "'nested_univ'"): ("K", "get_cutoff internal mtypes", 0),
    (U, 100, "'df-list'"): ("K", "get_cutoff internal mtypes", 0),
    (U, 101, "'pd_multiindex_hier'"): ("K", "get_cutoff internal mtypes", 0),
    (U, 130, "pd.Index"): ("D", "is pd.Index", 0),
    (U, 141, "pd.MultiIndex"): ("D", "is multiindexed", 0),
    (U, 142, "pd.DataFrame"): ("S", "frame from index for .loc lookups", 0),
    (U, 143, ".droplevel"): ("L", "instance keys", 0),
    (U, 147, ".loc"): ("S", "select instance (per instance loop)", 0),
    (U, 147, ".index"): ("T", "time_index of instance", 0),
    (U, 148, "hasattr(.., 'freq')"): ("T", "freq", 0),
    (U, 148, ".freq"): ("T", "freq", 0),
    (U, 149, ".freq"): ("T", "freq", 0),
    (U, 154, ".get_level_values"): ("T", "time level values (dead store)", 0),
    (U, 154, ".sort_values"): ("T", "time level values (dead store)", 0),
    (U, 155, "pd.Index"): ("T", "cutoff as length-1 pd.Index", 1),
    (U, 156, ".freq"): ("T", "set freq on cutoff", 1),
    (U, 159, "hasattr(.., 'freq')"): ("T", "freq", 0),
    (U, 159, ".freq"): ("T", "freq", 0),
    (U, 160, ".freq"): ("T", "freq", 0),
    (U, 167, "hasattr(.., 'freq')"): ("T", "freq", 0),
    (U, 167, ".freq"): ("T", "freq", 0),
    (U, 168, ".freq"): ("T", "freq", 0),
    (U, 221, "hasattr(.., 'X')"): ("D", "duck: legacy VectorizedDF.X (stale)", 0),
    (U, 224, "pd.Index"): ("D", "is pd.Index", 0),
    (U, 235, "convert_to"): ("K", "convert input to pandas/numpy for cutoff", 0),
    (U, 239, "pd.Index"): ("D", "is cutoff pd.Index", 1),
    (U, 261, "pd.RangeIndex"): ("T", "cutoff for numpy as RangeIndex", 1),
    (U, 280, "hasattr(.., 'freq')"): ("T", "freq", 0),
    (U, 281, ".freq"): ("T", "freq", 0),
    (U, 283, ".freq"): ("T", "freq", 0),
    (U, 283, "pd.infer_freq"): ("T", "infer_freq", 0),
    (U, 290, ".freq"): ("T", "freq", 0),
    (U, 291, ".freq"): ("T", "freq", 0),
    (U, 294, "pd.Series"): ("D", "is pd.Series", 0),
    (U, 295, ".index"): ("T", "time_index", 0),
    (U, 298, "pd.DataFrame"): ("D", "is pd.DataFrame", 0),
    (U, 298, ".index"): ("L", "row keys", 0),
    (U, 298, "pd.MultiIndex"): ("D", "is multiindexed", 0),
    (U, 299, ".columns"): ("M", "column_names", 0),
    (U, 299, ".dtypes"): ("M", "dtypes (object -> nested)", 0),
    (U, 302, ".index"): ("T", "time_index", 0),
    (U, 306, ".index"): ("T", "time_index (nested)", 0),
    (U, 311, "pd.DataFrame"): ("D", "is pd.DataFrame", 0),
    (U, 311, ".index"): ("L", "row keys", 0),
    (U, 311, "pd.MultiIndex"): ("D", "is multiindexed", 0),
    (U, 314, ".index"): ("L", "row keys", 0),
    (U, 314, ".nlevels"): ("L", "n_index_levels", 0),
    (U, 316, ".index"): ("L", "row keys", 0),
    (U, 316, ".to_frame"): ("G", "groupwise last time point", 0),
    (U, 316, ".groupby"): ("G", "groupwise last time point", 0),
    (U, 316, ".nth"): ("G", "groupwise last time point", 0),
    (U, 316, ".iloc"): ("G", "groupwise last time point", 0),
    (U, 316, ".agg"): ("G", "global max of groupwise last", 0),
    (U, 324, ".index"): ("T", "time level", 0),
    (U, 324, ".levels"): ("T", "time level", 0),
    (U, 325, "hasattr(.., 'freq')"): ("T", "freq", 0),
    (U, 325, ".freq"): ("T", "freq", 0),
    (U, 326, ".freq"): ("T", "freq", 0),
    (U, 327, ".freq"): ("T", "freq", 0),
    (U, 334, ".index"): ("T", "time_index (df-list)", 0),
    (U, 339, "'pd.DataFrame'"): ("K", "update_data internal mtypes", 0),
    (U, 340, "'pd.Series'"): ("K", "update_data internal mtypes", 0),
    (U, 342, "'pd-multiindex'"): ("K", "update_data internal mtypes", 0),
    (U, 344, "'pd_multiindex_hier'"): ("K", "update_data internal mtypes", 0),
    (U, 379, "convert_to"): ("K", "convert to pandas/numpy for update", 0),
    (U, 400, "pd.DataFrame"): ("D", "is pandas", 0),
    (U, 400, "pd.Series"): ("D", "is pandas", 0),
    (U, 405, ".combine_first"): ("C", "upsert rows by index (append/overwrite)", 0),
    (U, 406, "hasattr(.., 'freq')"): ("T", "freq", 0),
    (U, 406, ".index"): ("T", "time_index", 0),
    (U, 407, ".index"): ("T", "time_index", 0),
    (U, 407, ".freq"): ("T", "freq", 0),
    (U, 408, ".index"): ("T", "time_index", 0),
    (U, 408, ".freqstr"): ("T", "freq", 0),
    (U, 409, ".index"): ("T", "time_index", 0),
    (U, 409, ".freq"): ("T", "freq", 0),
    # ---------------- multiindex.py ----------------
    (MI, 39, "pd.Index"): ("O", "flat str column names", 0),
    # ---------------- validation/series.py ----------------
    (SE, 268, ".equals"): ("T", "time index equality", 0),
    (SE, 271, ".symmetric_difference"): ("T", "time index diff (msg)", 0),
    (SE, 274, ".isin"): ("T", "time index containment", 0),
    (SE, 277, ".difference"): ("T", "time index diff (msg)", 0),
}

# implicit semantics not visible by name - added manually after reading the code
MANUAL = [
    (B, 1735, "y - y_pred", "BaseForecaster.predict_residuals",
     "y_res = y - y_pred  (index-aligned arithmetic)"),
    (B, 2073, "check_equal_time_index", "BaseForecaster._check_X_y",
     "check_equal_time_index(X, y, mode='contains')  (called on RAW inputs)"),
    (B, 603, "convert_to", "BaseForecaster.predict",
     "convert_to(y_pred, self._y_metadata['mtype'], ...)"),
    (V, 90, "len", "coerce_to_multiindex.infer_freq", "if len(X) == 0"),
    (U, 235, "convert_to", "get_cutoff",
     "obj = convert_to(obj, GET_CUTOFF_SUPPORTED_MTYPES)"),
    (U, 379, "convert_to", "update_data",
     "X = convert_to(X, to_type=UPDATE_DATA_INTERNAL_MTYPES)"),
    (B, 1833, "'forecasters.loc[...]'", "BaseForecaster.get_fitted_params",
     'fcst_key = f"forecasters.loc[{..},{..}]"'),
    (V, 369, "inst[[col]]", "VectorizedDF.items._iter_cols",
     "_enforce_index_freq(inst[[col]])  (frame.__getitem__ with list)"),
    (V, 767, ".shape", "VectorizedDF.vectorize_est",
     "if estimator.shape != self.shape"),
    (B, 2722, "_shift", "BaseForecaster._predict_moving_cutoff",
     "self_copy._set_cutoff(_shift(y_first_index, by=-1, return_index=True))"),
    (B, 2725, "cv.split(y)", "BaseForecaster._predict_moving_cutoff",
     "for new_window, _ in cv.split(y)"),
    (B, 2860, "cutoff[0]", "_format_moving_cutoff_predictions",
     "cutoffs = [cutoff[0] for cutoff in cutoffs]"),
    (U, 153, "ss[ix]", "_get_cutoff_from_index", "cutoff = agg(cutoff, ss[ix])"),
    (U, 165, "time_idx[ix]", "_get_cutoff_from_index", "return time_idx[ix]"),
    (U, 166, "time_idx[[ix]]", "_get_cutoff_from_index", "res = time_idx[[ix]]"),
    (U, 247, "cutoff[0]", "get_cutoff", "cutoff = cutoff[0]"),
    (U, 278, "idx[ix]", "get_cutoff.sub_idx", "return idx[ix]"),
    (U, 279, "idx[[ix]]", "get_cutoff.sub_idx", "res = idx[[ix]]"),
]

# canonical pandas API name per raw token (for "distinct pandas API" counting).
# None = not a pandas API (sktime helper encapsulating pandas, generic protocol,
# naming contract, mtype string) -> counted as touchpoint, not as distinct pandas API.
CANON = {
    ".index": "DataFrame/Series.index", "hasattr(.., 'index')": "DataFrame/Series.index",
    ".columns": "DataFrame.columns", "hasattr(.., 'columns')": "DataFrame.columns",
    ".names": "Index.names", ".name": "Index.name", ".nlevels": "Index.nlevels",
    ".levels": "MultiIndex.levels", ".droplevel": "Index.droplevel",
    ".unique": "Index/Series.unique", ".set_names": "Index.set_names",
    ".groupby": "DataFrame.groupby", ".xs": "DataFrame.xs", ".loc": "DataFrame.loc",
    ".iloc": "DataFrame/Series.iloc", "inst[[col]]": "DataFrame.__getitem__(list)",
    ".pivot": "DataFrame.pivot", ".reindex": "DataFrame.reindex",
    ".iterrows": "DataFrame.iterrows", ".values": "Series.values",
    ".shape": "DataFrame.shape", ".dtypes": "DataFrame.dtypes",
    ".equals": "Index.equals", ".isin": "Index.isin", ".difference": "Index.difference",
    ".symmetric_difference": "Index.symmetric_difference",
    ".freq": "Index.freq", "hasattr(.., 'freq')": "Index.freq",
    ".freqstr": "Index.freqstr", "getattr(.., 'freqstr')": "Index.freqstr",
    ".get_level_values": "MultiIndex.get_level_values",
    ".sort_values": "Index.sort_values", ".sort_index": "DataFrame.sort_index",
    ".is_monotonic_increasing": "Index.is_monotonic_increasing",
    ".combine_first": "DataFrame.combine_first", ".to_frame": "Index.to_frame",
    ".nth": "GroupBy.nth", ".agg": "Series.agg",
    "idx[ix]": "Index.__getitem__(int)", "ss[ix]": "Index.__getitem__(int)",
    "time_idx[ix]": "Index.__getitem__(int)", "cutoff[0]": "Index.__getitem__(int)",
    "idx[[ix]]": "Index.__getitem__(list)", "time_idx[[ix]]": "Index.__getitem__(list)",
    "y - y_pred": "DataFrame.__sub__ (index-aligned)",
    "pd.concat": "pd.concat", "pd.DataFrame": "pd.DataFrame",
    "pd.Series": "pd.Series", "pd.Index": "pd.Index", "pd.MultiIndex": "pd.MultiIndex",
    "pd.RangeIndex": "pd.RangeIndex", "pd.infer_freq": "pd.infer_freq",
    "pd.MultiIndex.from_tuples": "pd.MultiIndex.from_tuples",
    "pd.MultiIndex.from_product": "pd.MultiIndex.from_product",
}


PROTOCOL = {
    "P1": "backend_of(obj) -> adapter  [replaces isinstance(pd.*) / hasattr duck typing]",
    "P2": "index_spec(frame) -> (level_names, n_levels)",
    "P3": "column_names(frame) [+ dtypes kind]",
    "P4": "instance_keys(frame, n_inst_levels) -> ordered unique key tuples",
    "P5": "time_freq(frame|slice) -> freq or None  [infer/attach freq]",
    "P6": "iter_groups(frame, n_inst_levels) -> (key_tuple, slice w/o inst levels)",
    "P7": "select_columns(slice, cols)",
    "P8": "concat_rows(slices, keys, level_names) -> frame",
    "P9": "concat_cols(slices, col_keys|None, flatten) -> frame",
    "P10": "cutoff(frame) -> max over instances of last time point",
    "P11": "upsert_rows(frame, new) -> frame  [update_data]",
    "P12": "take_rows / sort_rows(frame, positions|by time)",
    "P13": "time_contains(X, y) -> bool  [X/y index compatibility]",
    "P14": "aligned_binary_op(a, b, op)  [residuals]",
    "P15": "convert(obj, to_mtype)  [existing convert_to dispatch table]",
    "C1": "CONTRACT estimator grid forecasters_ (container-free possible)",
    "C2": "CONTRACT cutoff / fh / cv value types (pandas Index, freq)",
    "C3": "CONTRACT proba output columns (interval/quantile MultiIndex cols)",
    "C4": "CONTRACT get_fitted_params key naming 'forecasters.loc[..]'",
    "N": "non-hot/legacy path (nested_univ/df-list cells, stale VectorizedDF.X, dead store)",
}


def protocol_of(h):
    cat, op, contract = h["cat"], h["op"], h["api_contract"]
    if cat == "K":
        return "P15"
    if "stale" in op or "nested" in op or "dead store" in op or "df-list" in op:
        return "N"
    if cat == "D":
        return "P1"
    if cat == "P":
        return "C1"
    if cat == "O":
        if "interval" in op or "quantiles" in op:
            return "C3"
        if "KEY naming" in op:
            return "C4"
        if "grid" in op:
            return "C1"
        if "empty moving-cutoff" in op:
            return "P8"
        return "P9"  # flatten / drop helper col level
    if cat == "G":
        return "P10" if ("groupwise" in op) else "P6"
    if cat == "C":
        if "concat_cols" in op:
            return "P9"
        if "upsert" in op:
            return "P11"
        if "ALIGNED" in op:
            return "P14"
        return "P8"
    if cat == "S":
        if "select_columns" in op:
            return "P7"
        if "select_instance" in op or "select instance" in op or "first instance" in op:
            return "P6"
        if "frame from index" in op:
            return "P10"
        return "P12"
    if cat == "M":
        return "P3"
    if cat == "L":
        if "instance" in op:
            return "P4" if "drop" not in op else "P6"
        return "P2"
    if cat == "T":
        if contract or "shift" in op or "splitter" in op:
            return "C2"
        if "freq" in op:
            return "P5"
        if "containment" in op or "equality" in op or "diff" in op or "contains" in op:
            return "P13"
        if "sorted" in op:
            return "P12"
        if "RangeIndex" in op:
            return "P2"
        if "positional element" in op or "time level" in op or "instance" in op:
            return "P10"
        if h["file"].endswith("_utilities.py") and h["func"] in (
            "get_cutoff", "_get_cutoff_from_index", "get_cutoff.sub_idx"
        ):
            return "P10"
        if h["func"] == "get_index_for_series":
            return "P13"
        if h["file"].endswith("_utilities.py") and h["func"] == "update_data":
            return "P11"
        return "P5"  # time_index(slice) accesses feeding freq logic
    return "?"


CATNAME = {
    "G": "container-op: group/iterate",
    "S": "container-op: slice/select",
    "C": "container-op: concat/reconstruct",
    "P": "container-op: pivot/reshape (estimator grid)",
    "L": "index-semantics: MultiIndex levels/names",
    "T": "index-semantics: time index/freq/cutoff",
    "M": "metadata: columns/dtypes/shape/nans",
    "O": "output-construction",
    "D": "type-dispatch: isinstance/duck-typing",
    "K": "mtype-pinning (pandas mtype as conversion target)",
    "X": "false positive (not pandas)",
}


def main():
    hits = scan_all()
    for f, line, api, func, code in MANUAL:
        full = [k for k in SCOPE if k.endswith(f)][0]
        hits.append(
            {"file": full, "line": line, "func": func, "api": api, "kind": "manual",
             "code": code}
        )
    # de-duplicate identical (file,line,api)
    uniq = {}
    for h in hits:
        uniq.setdefault((h["file"], h["line"], h["api"]), h)
    hits = list(uniq.values())

    unclassified = []
    for h in hits:
        key = (os.path.basename(h["file"]).replace("series.py", "series.py"),
               h["line"], h["api"])
        # normalise basename keys
        bn = os.path.basename(h["file"])
        k = (bn, h["line"], h["api"])
        if k not in CLASSIFY:
            unclassified.append(h)
            h["cat"], h["op"], h["api_contract"] = "?", "?", 0
        else:
            h["cat"], h["op"], h["api_contract"] = CLASSIFY[k]
        del key

    stale_keys = set(CLASSIFY) - {
        (os.path.basename(h["file"]), h["line"], h["api"]) for h in hits
    }

    real = [h for h in hits if h["cat"] not in ("X", "?")]
    by_cat = Counter(h["cat"] for h in real)
    # distinct pandas API names (strip args), per category
    for h in hits:
        h["canon"] = CANON.get(h["api"])
    distinct_total = sorted({h["canon"] for h in real if h["canon"]})
    distinct_by_cat = defaultdict(set)
    for h in real:
        if h["canon"]:
            distinct_by_cat[h["cat"]].add(h["canon"])
    non_api = Counter(h["api"] for h in real if not h["canon"])
    n_lines = len({(h["file"], h["line"]) for h in real})
    hot = None
    by_file = Counter(h["file"] for h in real)
    by_file_cat = defaultdict(Counter)
    for h in real:
        by_file_cat[h["file"]][h["cat"]] += 1
    api_contract = [h for h in real if h["api_contract"]]
    abstract_ops = defaultdict(list)
    for h in real:
        abstract_ops[h["op"]].append(f'{os.path.basename(h["file"])}:{h["line"]}')
    by_proto = defaultdict(list)
    for h in real:
        h["proto"] = protocol_of(h)
        by_proto[h["proto"]].append(f'{os.path.basename(h["file"])}:{h["line"]}')

    out = {
        "n_touchpoints": len(real),
        "n_distinct_lines": n_lines,
        "non_pandas_api_touchpoints (helpers/strings/protocol)": dict(non_api),
        "n_false_positive": sum(h["cat"] == "X" for h in hits),
        "n_unclassified": len(unclassified),
        "stale_classify_keys": sorted(map(str, stale_keys)),
        "by_category": {CATNAME[k]: v for k, v in sorted(by_cat.items())},
        "n_distinct_pandas_api": len(distinct_total),
        "distinct_pandas_api": distinct_total,
        "distinct_api_by_category": {
            CATNAME[k]: sorted(v) for k, v in sorted(distinct_by_cat.items())
        },
        "by_file": dict(by_file),
        "by_file_category": {f: dict(c) for f, c in by_file_cat.items()},
        "n_api_contract_touchpoints": len(api_contract),
        "n_abstract_ops": len(abstract_ops),
        "abstract_ops": {k: v for k, v in sorted(abstract_ops.items())},
        "protocol_ops": PROTOCOL,
        "touchpoints_by_protocol_op": {
            k: {"n": len(v), "desc": PROTOCOL.get(k, "?"), "where": sorted(set(v))}
            for k, v in sorted(by_proto.items(), key=lambda kv: (kv[0][0], int(kv[0][1:]) if kv[0][1:].isdigit() else 0))
        },
        "touchpoints": sorted(real, key=lambda h: (h["file"], h["line"], h["api"])),
        "unclassified": unclassified,
    }
    os.makedirs(f"{ED}/results", exist_ok=True)
    with open(f"{ED}/results/a1_touchpoints.json", "w") as f:
        json.dump(out, f, indent=1, default=str)

    # markdown table
    lines = [
        "| # | file:line | function | token | canonical pandas API | cat | abstract op | protocol op | API-contract |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, h in enumerate(out["touchpoints"], 1):
        lines.append(
            f'| {i} | `{h["file"].replace("sktime/", "")}:{h["line"]}` | '
            f'`{h["func"]}` | `{h["api"]}` | {h["canon"] or "-"} | {h["cat"]} | {h["op"]} | {h["proto"]} | '
            f'{"yes" if h["api_contract"] else ""} |'
        )
    with open(f"{ED}/results/a1_touchpoints_table.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("distinct source lines:", n_lines)
    print("non-pandas-API touchpoints:", dict(non_api))
    print("touchpoints:", out["n_touchpoints"], " false positives:",
          out["n_false_positive"], " unclassified:", len(unclassified))
    print("by category:")
    for k, v in out["by_category"].items():
        print(f"  {v:4d}  {k}")
    print("distinct canonical pandas API:", len(distinct_total))
    for k, v in out["distinct_api_by_category"].items():
        print(f"  {k}: {len(v)} -> {v}")
    print("by file:", dict(by_file))
    print("API-contract touchpoints:", len(api_contract))
    print("distinct abstract ops (free text):", len(abstract_ops))
    print("touchpoints by protocol op:")
    for k, v in out["touchpoints_by_protocol_op"].items():
        print(f"  {k:4s} n={v['n']:3d}  {v['desc']}")
    if stale_keys:
        print("STALE classify keys (no matching hit):", sorted(stale_keys))
    for h in unclassified:
        print("UNCLASSIFIED", h["file"], h["line"], h["api"], h["func"], "|", h["code"])
    sys.exit(1 if unclassified else 0)


if __name__ == "__main__":
    main()
