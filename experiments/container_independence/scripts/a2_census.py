"""a2_census: static census of pandas coupling in all sktime forecasters.

For every forecaster in the sktime registry, this script

1. reads tags (y_inner_mtype, X_inner_mtype, capability:multivariate,
   capability:exogenous, capability:pred_int, python_dependencies),
2. detects composites / meta-estimators (meta base classes in the MRO, and
   estimator-valued parameters in ``get_test_params``),
3. AST-scans the code the class actually runs in its core methods
   (``_fit``, ``_predict``, ``_update``, ``_predict_interval``,
   ``_predict_quantiles``, ``_predict_var``, ``_predict_proba``,
   ``_pretrain``, ``_pretrain_update``), resolved through the MRO up to (but
   excluding) ``BaseForecaster``, following ``self.<method>`` calls into the
   class chain and bare-name calls into same-module / sktime helper functions
   (depth-limited), and counts pandas API usage by category,
4. assigns a heuristic bucket B1..B5 (see ``assign_bucket``).

Run from the worktree root::

    PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
        experiments/container_independence/scripts/a2_census.py

Outputs ``results/a2_census.csv`` and ``results/a2_census.json``.
"""

import argparse
import ast
import collections
import csv
import importlib
import inspect
import json
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ED = Path(__file__).resolve().parents[1]
RES = ED / "results"

CORE_METHODS = [
    "_fit",
    "_predict",
    "_update",
    "_predict_interval",
    "_predict_quantiles",
    "_predict_var",
    "_predict_proba",
    "_pretrain",
    "_pretrain_update",
]

# ---------------------------------------------------------------------------
# pandas API categories
# ---------------------------------------------------------------------------

# attribute names on arbitrary receivers, by category.
# "hard" = index/label-aligned or time-semantic dataframe operations: these
#   need a rewrite (or a pandas shim) if the inner data is numpy values.
# "soft" = pandas-only methods that have a trivial positional numpy
#   equivalent (iloc, dropna, diff, ...) - boundary-level edits.
# "meta" = reading index/columns metadata (kept by "values + time-index" repr).
ATTR_CATS = {
    "label_select": {
        "loc",
        "at",
        "xs",
        "get_level_values",
        "droplevel",
        "swaplevel",
        "reorder_levels",
        "truncate",
        "between_time",
        "at_time",
    },
    "time_ops": {
        "resample",
        "shift",
        "rolling",
        "expanding",
        "ewm",
        "asfreq",
        "to_period",
        "to_timestamp",
        "tz_localize",
        "tz_convert",
        "dt",
        "tshift",
        "first_valid_index",
        "last_valid_index",
    },
    "frame_ops": {
        "groupby",
        "reindex",
        "reindex_like",
        "merge",
        "join",
        "pivot",
        "pivot_table",
        "melt",
        "stack",
        "unstack",
        "set_index",
        "reset_index",
        "sort_index",
        "combine_first",
        "align",
        "interpolate",
        "isin",
        "explode",
        "query",
    },
    "soft_ops": {
        "iloc",
        "iat",
        "head",
        "tail",
        "dropna",
        "fillna",
        "ffill",
        "bfill",
        "diff",
        "pct_change",
        "apply",
        "applymap",
        "agg",
        "aggregate",
        "isna",
        "isnull",
        "notna",
        "notnull",
        "drop",
        "to_frame",
        "squeeze",
        "rename",
        "rename_axis",
        "set_axis",
        "sort_values",
        "value_counts",
        "nunique",
        "duplicated",
        "drop_duplicates",
        "hasnans",
        "iterrows",
        "itertuples",
        "assign",
        "add_prefix",
        "add_suffix",
        "to_dict",
    },
    "meta": {
        "index",
        "columns",
        "names",
        "nlevels",
        "levels",
        "freq",
        "freqstr",
        "inferred_freq",
        "dtypes",
        "is_monotonic_increasing",
    },
    "to_numpy": {"values", "to_numpy"},
}
ATTR2CAT = {a: c for c, s in ATTR_CATS.items() for a in s}
HARD_CATS = ("label_select", "time_ops", "frame_ops")

# pd.<name> top-level usage categories
PD_WRAP = {"DataFrame", "Series", "Index", "RangeIndex", "MultiIndex", "Categorical"}
PD_TIME = {
    "date_range",
    "period_range",
    "timedelta_range",
    "to_datetime",
    "to_timedelta",
    "Timestamp",
    "Timedelta",
    "Period",
    "PeriodIndex",
    "DatetimeIndex",
    "TimedeltaIndex",
    "offsets",
    "DateOffset",
    "infer_freq",
    "tseries",
    "NaT",
    "PeriodDtype",
    "DatetimeTZDtype",
}
PD_OPS = {
    "concat",
    "merge",
    "merge_asof",
    "melt",
    "pivot",
    "pivot_table",
    "get_dummies",
    "crosstab",
    "cut",
    "qcut",
    "Grouper",
    "wide_to_long",
    "factorize",
}
PD_SOFT = {"isna", "isnull", "notna", "notnull", "to_numeric", "unique", "api", "NA"}

FH_ATTRS = {
    "to_absolute",
    "to_absolute_index",
    "to_absolute_int",
    "to_relative",
    "to_indexer",
    "get_expected_pred_idx",
    "to_in_sample",
    "to_out_of_sample",
    "is_all_in_sample",
    "is_all_out_of_sample",
    "to_pandas",
}
SELF_STATE = {
    "_y": "hist",
    "_X": "hist",
    "_cur_y": "hist",
    "_cur_X": "hist",
    "_cutoff": "cutoff",
    "cutoff": "cutoff",
    "_fh": "fh_attr",
    "fh": "fh_attr",
}
FRAMEWORK_CALLS = {
    "convert_to",
    "convert",
    "update_data",
    "check_is_mtype",
    "check_is_scitype",
    "get_cutoff",
    "_shift",
    "get_window",
    "ForecastingHorizon",
    "_check_cutoff",
    "_coerce_to_period",
}

# receivers matching this regex are index-level metadata, not data
INDEX_RECV = re.compile(
    r"(\.index\b|\.columns\b|index$|_index\b|\bidx|cutoff|\bfh\b|_fh\b|fh_|\.freq\b|"
    r"\bnames?\b|\bcols?\b|columns$)",
    re.IGNORECASE,
)

STDLIB = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()
NONTHIRD = {
    "sktime",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "skbase",
    "joblib",
    "packaging",
    "__future__",
}

# python_dependencies that do not mean "data goes into a third-party model"
INFRA_DEPS = {
    "scipy",
    "numba",
    "joblib",
    "scikit-learn",
    "tqdm",
    "setuptools",
    "numpy",
    "pandas",
    "matplotlib",
    "typer",
    "wandb",
    "hydra-core",
    "huggingface-hub",
    "huggingface_hub",
    "hf-xet",
    "safetensors",
    "accelerate",
    "einops",
    "einshape",
}
# deps of composites (tuners, proba regressors) - data does not go into them
COMPOSITE_INFRA_DEPS = {"optuna", "hyperactive", "scikit-optimize", "skpro"}

# family of a third-party model library, by dependency name
LIB_FAMILY = {
    "statsmodels": "statsmodels",
    "pmdarima": "pmdarima",
    "statsforecast": "statsforecast",
    "prophet": "prophet",
    "neuralprophet": "neuralprophet",
    "darts": "darts",
    "u8darts": "darts",
    "neuralforecast": "neuralforecast",
    "pytorch-forecasting": "pytorch-forecasting",
    "gluonts": "gluonts",
    "arch": "arch",
    "skforecast": "skforecast",
    "autots": "autots",
    "greykite": "greykite",
    "hcrystalball": "hcrystalball",
    "prophetverse": "prophetverse",
    "rpy2": "R (rpy2/fable)",
    "litellm": "LLM (litellm)",
    "hypertrees-forecasting": "hypertrees",
    "tbats": "tbats",
    "xgboost": "xgboost",
    "tsbootstrap": "tsbootstrap",
    "requests": "remote HTTP API",
}
TORCHISH = {
    "torch",
    "transformers",
    "lightning",
    "jax",
    "jaxlib",
    "paxml",
    "timesfm",
    "chronos-forecasting",
    "granite-tsfm",
    "mantis-tsfm",
    "tirex-2",
    "tirex-ts",
    "toto-models",
    "toto-ts",
    "aurora-model",
    "tfc-t0",
    "tafsut",
    "pykan",
    "FrEIA",
    "torchdiffeq",
    "torchvision",
    "cisco-tsm",
    "utilsforecast",
}

# what the wrapped library consumes, per family (manual, from adapter code;
# evidence = file:line in the sktime adapter that hands data to the library)
LIB_INPUT = {
    "statsmodels": ("numpy-capable", "base/adapters/_statsmodels.py:_fit passes pd.Series; _predict uses integer start/end (to_absolute_int)"),
    "pmdarima": ("numpy-capable", "base/adapters/_pmdarima.py:75 y.reset_index(drop=True) before fit"),
    "statsforecast": ("numpy-capable", "base/adapters/_generalised_statsforecast.py:172 y.to_numpy(copy=False)"),
    "tbats": ("numpy-capable", "base/adapters/_tbats.py:115 fit(y), tbats takes array-like"),
    "arch": ("numpy-capable", "arch/_uarch.py:286 arch_model(y=y, x=X), accepts ndarray"),
    "torch/foundation": ("numpy-capable", "tensors built from y.values / to_numpy (e.g. chronos.py:596, mira.py:316, adapters/_pytorch.py:514)"),
    "vendored(sktime.libs)": ("numpy-capable", "vendored torch model code fed numpy"),
    "LLM (litellm)": ("numpy-capable", "prompt built from values"),
    "remote HTTP API": ("numpy-capable", "falcon_x.py:459 context.tolist() -> JSON"),
    "tsbootstrap": ("numpy-capable", "bootstrap on arrays"),
    "xgboost": ("numpy-capable", "regressor"),
    "prophet": ("pandas-bound", "base/adapters/_fbprophet.py:93-97,144 DataFrame with 'ds' column"),
    "neuralprophet": ("pandas-bound", "neuralprophet.py:338 fit(df)"),
    "darts": ("pandas-bound", "base/adapters/_darts.py:194 darts.TimeSeries.from_dataframe (own type)"),
    "neuralforecast": ("pandas-bound", "base/adapters/_neuralforecast.py:407-420 long df (unique_id, ds, y)"),
    "pytorch-forecasting": ("pandas-bound", "base/adapters/_pytorchforecasting.py:584 TimeSeriesDataSet(pandas)"),
    "gluonts": ("pandas-bound", "moirai.py:678 / lagllama.py:575 gluonts PandasDataset.from_long_dataframe"),
    "skforecast": ("pandas-bound", "compose/_skforecast_reduce.py:781 fit(pd.Series with Datetime/RangeIndex)"),
    "autots": ("pandas-bound", "autots.py:369 fit(df=y_date) wide df"),
    "greykite": ("pandas-bound", "greykite.py:466 run_forecast_config(df, fc)"),
    "hcrystalball": ("pandas-bound", "hcrystalball sklearn-API on pandas"),
    "prophetverse": ("pandas-bound", "prophetverse.py delegates to prophetverse sktime-API estimator"),
    "R (rpy2/fable)": ("pandas-bound", "pyfable_arima.py:309-310 pandas2ri.py2rpy"),
    "hypertrees": ("pandas-bound", "hypertrees/_ar.py _fit builds long pd.DataFrame from numpy values + synthetic dates"),
}

ALGO_OPS = {
    "groupby", "rolling", "expanding", "ewm", "resample", "asfreq", "shift",
    "reindex", "combine_first", "interpolate", "merge", "join", "pivot",
    "pivot_table", "melt", "stack", "unstack", "align",
}

META_BASES = {
    "_HeterogenousMetaEstimator": "hetero_meta",
    "_DelegatedForecaster": "delegate",
    "BaseGridSearch": "tuning",
    "_Pipeline": "pipeline",
    "_HeterogenousEnsembleForecaster": "ensemble",
    "_Reducer": "reducer",
    "_ReducerMixin": "reducer",
    "_StreamDataPoolMixin": "stream",
}


def dep_name(spec):
    """Package name from a PEP 508 requirement string."""
    return re.split(r"[<>=!~;\[ ]", spec.strip())[0]


def flat(d):
    if d is None:
        return []
    if isinstance(d, str):
        return [d]
    out = []
    for x in d:
        out += flat(x)
    return out


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

_AST_CACHE = {}


def module_ast(path):
    if path not in _AST_CACHE:
        src = Path(path).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child._parent = node
        _AST_CACHE[path] = tree
    return _AST_CACHE[path]


def find_classdef(klass):
    path = inspect.getsourcefile(klass)
    tree = module_ast(path)
    try:
        lineno = inspect.getsourcelines(klass)[1]
    except Exception:  # noqa: BLE001
        lineno = None
    cands = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == klass.__name__
    ]
    if not cands:
        return path, None
    if lineno is not None:
        for c in cands:
            first = min([c.lineno] + [d.lineno for d in c.decorator_list])
            if first == lineno or c.lineno == lineno:
                return path, c
    return path, cands[0]


def find_funcdef(func):
    """AST node of a module-level function object."""
    try:
        path = inspect.getsourcefile(func)
        lineno = inspect.getsourcelines(func)[1]
    except Exception:  # noqa: BLE001
        return None, None
    tree = module_ast(path)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == (
            func.__name__
        ):
            first = min([n.lineno] + [d.lineno for d in n.decorator_list])
            if first == lineno or n.lineno == lineno:
                return path, n
    return path, None


def recv_str(node):
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return ""


def is_inside_isinstance(node):
    p = getattr(node, "_parent", None)
    depth = 0
    while p is not None and depth < 4:
        if (
            isinstance(p, ast.Call)
            and isinstance(p.func, ast.Name)
            and p.func.id in ("isinstance", "issubclass")
        ):
            return True
        p = getattr(p, "_parent", None)
        depth += 1
    return False


INPUT_PARAM = re.compile(
    r"^(y|X|Z|df|data|series|ts|endog|exog|target|context|_y|_X|y_\w+|X_\w+|\w+_y|\w+_X)$"
)
SELF_INPUT_ATTR = re.compile(r"^(_y|_X|_cur_y|_cur_X|_y_\w+|_X_\w+|y_|X_)$")
NUMPY_CONV_CALLS = {"to_numpy", "asarray", "array", "values", "ravel", "flatten", "tolist"}


def _root(expr):
    """Root of an attribute/subscript/call chain: Name id or 'self.<attr>'."""
    while True:
        if isinstance(expr, ast.Attribute):
            if isinstance(expr.value, ast.Name) and expr.value.id == "self":
                return f"self.{expr.attr}"
            expr = expr.value
        elif isinstance(expr, ast.Subscript):
            expr = expr.value
        elif isinstance(expr, ast.Call):
            expr = expr.func
        elif isinstance(expr, ast.Name):
            return expr.id
        else:
            return None


def _is_pd_ctor(expr):
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and isinstance(expr.func.value, ast.Name)
        and expr.func.value.id in ("pd", "pandas")
        and expr.func.attr in (PD_WRAP | {"concat"})
    )


def _is_numpy_conv(expr):
    """Expression chain converts to numpy: x.to_numpy()..., np.f(...), x.values..."""
    while True:
        if isinstance(expr, ast.Call):
            f = expr.func
            if isinstance(f, ast.Attribute):
                if isinstance(f.value, ast.Name) and f.value.id in ("np", "numpy"):
                    return True
                if f.attr in NUMPY_CONV_CALLS:
                    return True
            expr = f
        elif isinstance(expr, ast.Attribute):
            if expr.attr == "values":
                return True
            expr = expr.value
        elif isinstance(expr, ast.Subscript):
            expr = expr.value
        else:
            return False


# heuristic v2 refinements (see report): prediction-like names are outputs,
# and results of predict/forecast calls are outputs even if X was an argument
PRED_NAME = re.compile(r"pred|hat|fcst|forecast|in_sample|out_sample|_ins$|_oos$", re.I)
PRED_CALL = re.compile(r"^_?(predict|forecast)", re.I)
HEURISTIC_VERSION = 2


def _is_pred_call(expr):
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and PRED_CALL.match(expr.func.attr) is not None
    )


def origin_map(fnode):
    """Name -> origin in {"input", "wrapped", "numpy"} by a 2-pass taint over assigns."""
    org = {}
    args = fnode.args
    for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        if (
            a.arg not in ("self", "cls")
            and INPUT_PARAM.match(a.arg)
            and not (HEURISTIC_VERSION >= 2 and PRED_NAME.search(a.arg))
        ):
            org[a.arg] = "input"

    def _walk_no_pred(expr):
        # do not descend into predict/forecast calls: their results are outputs
        stack = [expr]
        while stack:
            n = stack.pop()
            if HEURISTIC_VERSION >= 2 and _is_pred_call(n):
                continue
            yield n
            stack.extend(ast.iter_child_nodes(n))

    def refs(expr):
        out = set()
        for n in _walk_no_pred(expr):
            if isinstance(n, ast.Name) and n.id in org:
                out.add(org[n.id])
            elif (
                isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name)
                and n.value.id == "self"
                and SELF_INPUT_ATTR.match(n.attr)
            ):
                out.add("input")
        return out

    assigns = sorted(
        [n for n in ast.walk(fnode) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.For))],
        key=lambda n: n.lineno,
    )
    for _ in range(2):
        for n in assigns:
            if isinstance(n, ast.For):
                val, targets = n.iter, [n.target]
            else:
                val = n.value
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if val is None:
                continue
            if _is_pd_ctor(val) or (HEURISTIC_VERSION >= 2 and _is_pred_call(val)):
                o = "wrapped"
            elif _is_numpy_conv(val):
                o = "numpy"
            else:
                r = refs(val)
                o = "input" if "input" in r else "wrapped" if "wrapped" in r else None
            if o is None:
                continue
            for t in targets:
                # only plain names (and tuple/list unpacking) are (re)bound;
                # "pred.index.names = ..." must not re-taint "pred"
                stack = [t]
                while stack:
                    nn = stack.pop()
                    if isinstance(nn, ast.Name):
                        org[nn.id] = o
                    elif isinstance(nn, (ast.Tuple, ast.List)):
                        stack.extend(nn.elts)
                    elif isinstance(nn, ast.Starred):
                        stack.append(nn.value)
    return org


def node_origin(expr, org):
    root = _root(expr)
    if root is None:
        return "other"
    if root.startswith("self."):
        return "input" if SELF_INPUT_ATTR.match(root[5:]) else "other"
    o = org.get(root)
    return {"input": "input", "wrapped": "output", "numpy": "other"}.get(o, "other")


def scan_function(fnode, relpath):
    """Count features in one function body. Returns (Counter, details list)."""
    cnt = collections.Counter()
    det = []
    org = origin_map(fnode)

    def add(cat, name, node):
        cnt[cat] += 1
        cnt[f"{cat}:{name}"] += 1
        det.append((cat, name, f"{relpath}:{getattr(node, 'lineno', 0)}"))
        if cat in HARD_CATS:
            o = node_origin(node.value, org)
            cnt[f"hard_{o}"] += 1
            det.append((f"hard_{o}", name, f"{relpath}:{getattr(node, 'lineno', 0)}"))
        elif cat == "pd_ops":
            call = getattr(node, "_parent", None)
            o = "other"
            if isinstance(call, ast.Call):
                os_ = set()
                for a in call.args:
                    elts = a.elts if isinstance(a, (ast.List, ast.Tuple)) else [a]
                    for e in elts:
                        if HEURISTIC_VERSION >= 2 and _is_pred_call(e):
                            os_.add("output")
                            continue
                        os_.add(node_origin(e, org))
                        for nn in ast.walk(e):
                            if HEURISTIC_VERSION >= 2 and _is_pred_call(nn):
                                break
                            if isinstance(nn, ast.Name) and org.get(nn.id) == "input":
                                os_.add("input")
                o = "input" if "input" in os_ else "output" if "output" in os_ else "other"
            cnt[f"hard_{o}"] += 1
            det.append((f"hard_{o}", name, f"{relpath}:{getattr(node, 'lineno', 0)}"))

    for node in ast.walk(fnode):
        if isinstance(node, ast.Attribute):
            val = node.value
            parent = getattr(node, "_parent", None)
            is_called = isinstance(parent, ast.Call) and parent.func is node
            # pd.<x>
            if isinstance(val, ast.Name) and val.id in ("pd", "pandas"):
                a = node.attr
                if is_inside_isinstance(node):
                    add("pd_isinstance", a, node)
                elif a in PD_WRAP:
                    add("pd_wrap", a, node)
                elif a in PD_TIME:
                    add("pd_time", a, node)
                elif a in PD_OPS:
                    add("pd_ops", a, node)
                elif a in PD_SOFT:
                    add("pd_soft", a, node)
                else:
                    add("pd_other", a, node)
                continue
            if isinstance(val, ast.Name) and val.id in ("np", "numpy"):
                if node.attr in ("asarray", "array", "asanyarray", "ascontiguousarray"):
                    add("to_numpy", f"np.{node.attr}", node)
                continue
            # self.<state>
            if isinstance(val, ast.Name) and val.id == "self":
                if node.attr in SELF_STATE:
                    add(SELF_STATE[node.attr], node.attr, node)
                continue
            a = node.attr
            r = recv_str(val)
            if a in FH_ATTRS:
                # to_pandas / to_numpy are fh-specific only on fh-like receivers
                if a == "to_pandas" and not re.search(r"fh|horizon", r, re.I):
                    pass
                else:
                    add("fh_ops", a, node)
                    continue
            cat = ATTR2CAT.get(a)
            if cat is None:
                continue
            if a == "index" and is_called:  # list.index(...)
                continue
            if a == "values" and is_called:  # dict.values()
                continue
            if a in ("names",) and not re.search(r"index|idx|columns", r, re.I):
                continue
            if cat in HARD_CATS or cat == "soft_ops":
                if INDEX_RECV.search(r):
                    add("index_level_ops", a, node)
                    continue
            add(cat, a, node)
        elif isinstance(node, ast.Call):
            f = node.func
            name = None
            if isinstance(f, ast.Name):
                name = f.id
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                if f.value.id not in ("self", "pd", "np"):
                    name = f.attr
            if name in FRAMEWORK_CALLS:
                add("framework_call", name, node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for m in mods:
                top = m.split(".")[0]
                if m.startswith("sktime.libs"):
                    add("vendored_import", m, node)
                elif top and top not in NONTHIRD and top not in STDLIB:
                    add("thirdparty_import", top, node)
    return cnt, det


def calls_in(fnode):
    """self.<name> references, super().<name> calls and bare-name calls."""
    selfnames, barenames, localimports = set(), set(), {}
    for node in ast.walk(fnode):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
                selfnames.add(node.attr)
            elif (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "super"
            ):
                selfnames.add(("super", node.attr))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            barenames.add(node.func.id)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                localimports[a.asname or a.name] = (node.module, a.name)
    return selfnames, barenames, localimports


NOFOLLOW_MOD = (
    "sktime.datatypes",
    "sktime.utils.dependencies",
    "sktime.utils.warnings",
    "sktime.utils.validation",
    "sktime.base",
    "sktime.forecasting.base._fh",
    "sktime.forecasting.base._base",
    "sktime.registry",
    "sktime.libs",
    "sktime.utils.singleton",
)


def chain_of(cls, BaseForecaster):
    base_mro = set(BaseForecaster.__mro__)
    return [k for k in cls.__mro__ if k not in base_mro]


def resolve_method(chain, name, after=None):
    started = after is None
    for k in chain:
        if not started:
            if k is after:
                started = True
            continue
        if name in k.__dict__:
            return k
    return None


def scan_class(cls, BaseForecaster, wt_root, max_helper_depth=2):
    chain = chain_of(cls, BaseForecaster)
    classdefs = {}
    for k in chain:
        try:
            classdefs[k] = find_classdef(k)
        except (TypeError, OSError):
            classdefs[k] = (None, None)

    def method_node(k, name):
        path, cnode = classdefs.get(k, (None, None))
        if cnode is None:
            return None, None
        for n in cnode.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
                return path, n
        return path, None

    total = collections.Counter()
    by_origin = {
        "class": collections.Counter(),
        "module_helper": collections.Counter(),
        "sktime_helper": collections.Counter(),
    }
    details = []
    visited = set()
    scanned = []
    core_defined = []
    chain_modules = {k.__module__ for k in chain}
    fit_numpy = False

    queue = []
    owners = {}
    for m in CORE_METHODS:
        k = resolve_method(chain, m)
        if k is not None:
            core_defined.append(f"{m}@{k.__name__}")
            owners[m] = k.__name__
            queue.append(("method", k, m, 0, m))

    while queue:
        kind, owner, name, depth, root = queue.pop(0)
        key = (kind, getattr(owner, "__name__", owner), name)
        if key in visited:
            continue
        visited.add(key)
        if kind == "method":
            path, fnode = method_node(owner, name)
            mod = owner.__module__
            origin = "class"
        else:  # function object
            path, fnode = find_funcdef(owner)
            mod = owner.__module__
            origin = "module_helper" if mod in chain_modules else "sktime_helper"
        if fnode is None:
            continue
        rel = str(Path(path).resolve().relative_to(wt_root)) if path else "?"
        cnt, det = scan_function(fnode, rel)
        total.update(cnt)
        by_origin[origin].update(cnt)
        details += det
        scanned.append(f"{origin}:{getattr(owner, '__name__', '?')}.{name}"
                       if kind == "method" else f"{origin}:{mod}.{name}")
        if root == "_fit" and cnt.get("to_numpy", 0) > 0:
            fit_numpy = True

        selfnames, barenames, localimports = calls_in(fnode)
        for s in selfnames:
            if isinstance(s, tuple):
                k2 = resolve_method(chain, s[1], after=owner) if kind == "method" else None
                if k2 is not None:
                    queue.append(("method", k2, s[1], depth, root))
                continue
            if s in CORE_METHODS:
                continue
            k2 = resolve_method(chain, s)
            if k2 is not None:
                queue.append(("method", k2, s, depth, root))
        if depth >= max_helper_depth:
            continue
        modobj = sys.modules.get(mod)
        g = vars(modobj) if modobj is not None else {}
        for b in barenames:
            obj = g.get(b)
            if obj is None and b in localimports:
                lm, ln = localimports[b]
                if lm.startswith("sktime") and not lm.startswith(NOFOLLOW_MOD):
                    try:
                        obj = getattr(importlib.import_module(lm), ln, None)
                    except Exception:  # noqa: BLE001
                        obj = None
            if obj is None or not inspect.isfunction(obj):
                continue
            fm = getattr(obj, "__module__", "") or ""
            if not fm.startswith("sktime") or fm.startswith(NOFOLLOW_MOD):
                continue
            queue.append(("func", obj, obj.__name__, depth + 1, root))

    return {
        "chain": [k.__name__ for k in chain],
        "core_defined": core_defined,
        "scanned": scanned,
        "counts": total,
        "by_origin": by_origin,
        "details": details,
        "fit_numpy": fit_numpy,
        "owners": owners,
        "module_imports": module_level_imports(chain),
    }


def module_level_imports(chain):
    """Third-party top-level imports + _safe_import targets in chain modules."""
    out = set()
    for k in chain:
        try:
            path = inspect.getsourcefile(k)
        except TypeError:
            continue
        tree = module_ast(path)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for m in mods:
                    top = m.split(".")[0]
                    if m.startswith("sktime.libs"):
                        out.add(m)
                    elif top and top not in NONTHIRD and top not in STDLIB:
                        out.add(top)
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                f = node.value.func
                if isinstance(f, ast.Name) and f.id == "_safe_import" and node.value.args:
                    a0 = node.value.args[0]
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                        out.add(a0.value.split(".")[0])
    return sorted(out)


# ---------------------------------------------------------------------------
# composite detection
# ---------------------------------------------------------------------------


def estimator_params(cls):
    """Params that hold estimators in any get_test_params set.

    Returns (names, kinds, err); kinds is a set of "sktime" (sktime/skbase
    object, e.g. forecaster or transformer child) and/or "sklearn" (sklearn
    or other non-sktime estimator, e.g. a tabular regressor).
    """
    names, kinds = set(), set()
    err = None
    try:
        params = cls.get_test_params()
    except Exception as e:  # noqa: BLE001
        return names, kinds, f"get_test_params failed: {type(e).__name__}"
    if isinstance(params, dict):
        params = [params]

    def is_est(v):
        if isinstance(v, type):
            return False
        return hasattr(v, "fit") and hasattr(v, "get_params")

    def kind(v):
        m = type(v).__module__ or ""
        return "sktime" if m.startswith(("sktime", "skbase")) else "sklearn"

    for p in params or []:
        for k, v in p.items():
            ests = []
            if is_est(v):
                ests = [v]
            elif isinstance(v, (list, tuple)):
                for e in v:
                    if is_est(e):
                        ests.append(e)
                    elif isinstance(e, tuple):
                        ests += [x for x in e if is_est(x)]
            elif isinstance(v, dict):
                ests = [x for x in v.values() if is_est(x)]
            if ests:
                names.add(k)
                kinds.update(kind(e) for e in ests)
    return names, kinds, err


# ---------------------------------------------------------------------------
# bucket assignment
# ---------------------------------------------------------------------------


def assign_bucket(row):
    """Heuristic primary bucket.

    Precedence: B3 (data handed to an external model library) > B4 (composite)
    > B2 (hard pandas ops on data in core) > B1 (numpy conversion present)
    > B5.
    """
    if row["third_party_libs"]:
        return "B3"
    if row["is_composite"]:
        return "B4"
    if row["n_hard_data_ops"] > 0:
        return "B2"
    if row["n_to_numpy"] > 0:
        return "B1"
    return "B5"


def core_style(row):
    """Static glue-style verdict from origin-tagged hard pandas ops.

    pandas-semantic-input : >=1 hard (label/time/frame/concat) op whose receiver
                            traces back to the input data (y, X, self._y, ...)
    output-assembly-only  : hard ops only on freshly wrapped outputs or
                            untraced objects (e.g. library return values)
    numpy-or-passthrough  : no hard pandas ops at all
    """
    if row["n_hard_input"] > 0:
        return "pandas-semantic-input"
    if row["n_hard_data_ops"] > 0:
        return "output-assembly-only"
    return "numpy-or-passthrough"


def static_verdict(row, input_ops):
    """Rule-based estimate of the change needed for a numpy-values +
    time-index-metadata inner representation (see report for definitions).

    V1 boundary-only, V2 glue rewrite, V3 core rewrite, V4 third-party pandas-bound.
    """
    b = row["bucket"]
    algo = bool(set(input_ops) & ALGO_OPS)
    if b == "B3":
        fams = [f for f in row["lib_family"].split("|") if f]
        if any(LIB_INPUT.get(f, ("?",))[0] == "pandas-bound" for f in fams):
            return "V4"
        return "V2" if row["core_style"] == "pandas-semantic-input" else "V1"
    if b == "B1":
        return "V1"
    if b == "B2":
        return "V3" if (algo or row["n_hard_input"] >= 3) else "V2"
    if b == "B4":
        if row["core_style"] == "numpy-or-passthrough":
            return "V1"
        if row["core_style"] == "output-assembly-only":
            return "V2"
        return "V3" if algo else "V2"
    return "V2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="dev", choices=["dev", "large"])  # unused
    ap.add_argument("--max-helper-depth", type=int, default=2)
    args = ap.parse_args()

    t0 = time.time()
    wt_root = Path(__file__).resolve().parents[3]
    from sktime.forecasting.base import BaseForecaster
    from sktime.registry import all_estimators

    ests = all_estimators("forecaster", return_names=True)
    rows, details = [], {}
    for name, cls in ests:
        tags = {
            t: cls.get_class_tag(t)
            for t in [
                "y_inner_mtype",
                "X_inner_mtype",
                "capability:multivariate",
                "capability:exogenous",
                "capability:pred_int",
                "capability:global_forecasting",
                "python_dependencies",
                "requires-fh-in-fit",
                "fit_is_empty",
            ]
        }
        deps = [dep_name(d) for d in flat(tags["python_dependencies"])]
        lib_deps = sorted(
            {d for d in deps if d not in INFRA_DEPS and d not in COMPOSITE_INFRA_DEPS}
        )
        sc = scan_class(cls, BaseForecaster, wt_root, args.max_helper_depth)
        c = sc["counts"]
        vendored = sorted(
            {m for m in sc["module_imports"] if m.startswith("sktime.libs")}
            | {n.split(":", 1)[1] for n in c if n.startswith("vendored_import:")}
        )
        third = list(lib_deps)
        if vendored:
            third.append("vendored:" + ",".join(sorted({v.split(".")[2] for v in vendored})))
        fam = sorted(
            {LIB_FAMILY.get(d, "torch/foundation" if d in TORCHISH else d) for d in lib_deps}
        )
        # tags of BATS/TBATS list only "pmdarima"; the adapter imports tbats
        if "tbats" in sc["module_imports"] or any(
            n == "thirdparty_import:tbats" for n in c
        ):
            fam = sorted((set(fam) - {"pmdarima"}) | {"tbats"})
        if vendored and not fam:
            fam = ["vendored(sktime.libs)"]

        est_par, est_kinds, gtp_err = estimator_params(cls)
        meta_kinds = sorted({META_BASES[k] for k in sc["chain"] if k in META_BASES})
        is_composite = bool(est_par) or bool(meta_kinds)

        def ym(t):
            v = tags[t]
            return v if isinstance(v, str) else "|".join(v) if v else str(v)

        y_m = tags["y_inner_mtype"]
        y_list = [y_m] if isinstance(y_m, str) else list(y_m or [])
        non_pd_inner = sorted(m for m in y_list if not m.startswith("pd"))

        n_hard = sum(c.get(k, 0) for k in HARD_CATS) + c.get("pd_ops", 0)
        row = {
            "name": name,
            "module": cls.__module__,
            "chain": ">".join(sc["chain"]),
            "y_inner_mtype": ym("y_inner_mtype"),
            "X_inner_mtype": ym("X_inner_mtype"),
            "y_inner_n": len(y_list),
            "y_inner_non_pandas": "|".join(non_pd_inner),
            "y_inner_accepts_all": len(y_list) >= 7,
            "multivariate": tags["capability:multivariate"],
            "exogenous": tags["capability:exogenous"],
            "pred_int": tags["capability:pred_int"],
            "global": tags["capability:global_forecasting"],
            "python_dependencies": "|".join(deps),
            "third_party_libs": "|".join(third),
            "lib_family": "|".join(fam),
            "is_composite": is_composite,
            "composite_kinds": "|".join(meta_kinds),
            "estimator_params": "|".join(sorted(est_par)),
            "estimator_param_kinds": "|".join(sorted(est_kinds)),
            "fit_owner": sc["owners"].get("_fit", ""),
            "predict_owner": sc["owners"].get("_predict", ""),
            "gtp_error": gtp_err or "",
            "core_defined": "|".join(sc["core_defined"]),
            "n_scanned_funcs": len(sc["scanned"]),
            "n_hard_data_ops": n_hard,
            "n_label_select": c.get("label_select", 0),
            "n_time_ops": c.get("time_ops", 0),
            "n_frame_ops": c.get("frame_ops", 0),
            "n_pd_ops": c.get("pd_ops", 0),
            "n_hard_input": c.get("hard_input", 0),
            "n_hard_output": c.get("hard_output", 0),
            "n_hard_other": c.get("hard_other", 0),
            "n_soft_ops": c.get("soft_ops", 0) + c.get("pd_soft", 0),
            "n_meta": c.get("meta", 0),
            "n_index_level_ops": c.get("index_level_ops", 0),
            "n_pd_wrap": c.get("pd_wrap", 0),
            "n_pd_time": c.get("pd_time", 0),
            "n_pd_isinstance": c.get("pd_isinstance", 0),
            "n_pd_other": c.get("pd_other", 0),
            "n_to_numpy": c.get("to_numpy", 0),
            "fit_converts_numpy": sc["fit_numpy"],
            "n_fh_ops": c.get("fh_ops", 0),
            "n_fh_to_absolute": c.get("fh_ops:to_absolute", 0)
            + c.get("fh_ops:to_absolute_index", 0)
            + c.get("fh_ops:to_absolute_int", 0),
            "n_get_expected_pred_idx": c.get("fh_ops:get_expected_pred_idx", 0),
            "n_cutoff": c.get("cutoff", 0),
            "n_fh_attr": c.get("fh_attr", 0),
            "n_hist": c.get("hist", 0),
            "n_framework_call": c.get("framework_call", 0),
            "n_class_hard": sum(sc["by_origin"]["class"].get(k, 0) for k in HARD_CATS)
            + sc["by_origin"]["class"].get("pd_ops", 0),
            "n_helper_hard": sum(
                sc["by_origin"][o].get(k, 0)
                for o in ("module_helper", "sktime_helper")
                for k in (*HARD_CATS, "pd_ops")
            ),
            "thirdparty_imports_in_core": "|".join(
                sorted({n.split(":", 1)[1] for n in c if n.startswith("thirdparty_import:")})
            ),
        }
        row["bucket"] = assign_bucket(row)
        row["core_style"] = core_style(row)
        input_ops = [d[1] for d in sc["details"] if d[0] == "hard_input"]
        row["input_hard_ops"] = "|".join(sorted(set(input_ops)))
        fams = [f for f in row["lib_family"].split("|") if f]
        row["lib_input"] = "|".join(sorted({LIB_INPUT.get(f, ("unknown",))[0] for f in fams}))
        row["static_verdict"] = static_verdict(row, input_ops)
        rows.append(row)
        details[name] = {
            "scanned": sc["scanned"],
            "module_imports": sc["module_imports"],
            "attr_counts": {k: v for k, v in sorted(c.items()) if ":" in k},
            "hard_evidence": [d for d in sc["details"] if d[0] in (*HARD_CATS, "pd_ops")],
            "hard_origin": [d for d in sc["details"] if d[0].startswith("hard_")],
            "numpy_evidence": [d for d in sc["details"] if d[0] == "to_numpy"],
            "fh_evidence": [d for d in sc["details"] if d[0] in ("fh_ops", "cutoff")],
        }

    RES.mkdir(exist_ok=True)
    with open(RES / "a2_census.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(RES / "a2_census.json", "w") as f:
        json.dump({"rows": rows, "details": details}, f, indent=1, default=str)

    # ---------------- summary ----------------
    n = len(rows)
    print(f"n_forecasters={n}  runtime={time.time() - t0:.1f}s")
    bc = collections.Counter(r["bucket"] for r in rows)
    print("bucket counts:", {k: (v, f"{100 * v / n:.1f}%") for k, v in sorted(bc.items())})
    print("composite flag:", sum(r["is_composite"] for r in rows))
    print("third-party flag:", sum(bool(r["third_party_libs"]) for r in rows))
    print("composite & third-party:", sum(r["is_composite"] and bool(r["third_party_libs"]) for r in rows))
    ymc = collections.Counter(r["y_inner_mtype"] if not r["y_inner_accepts_all"] else "<all/most mtypes>" for r in rows)
    print("y_inner_mtype:", ymc.most_common())
    print("non-pandas in y_inner:", sum(bool(r["y_inner_non_pandas"]) for r in rows))
    fam = collections.Counter(f for r in rows if r["bucket"] == "B3" for f in r["lib_family"].split("|"))
    print("B3 families:", fam.most_common())
    print("uses fh.to_absolute*:", sum(r["n_fh_to_absolute"] > 0 for r in rows))
    print("uses get_expected_pred_idx:", sum(r["n_get_expected_pred_idx"] > 0 for r in rows))
    print("uses cutoff:", sum(r["n_cutoff"] > 0 for r in rows))
    print("uses any fh op or cutoff:", sum((r["n_fh_ops"] + r["n_cutoff"]) > 0 for r in rows))
    print("uses stored history (_y/_X/_cur_y/_cur_X):", sum(r["n_hist"] > 0 for r in rows))
    impl = collections.Counter((r["fit_owner"], r["predict_owner"]) for r in rows)
    print(f"distinct (_fit owner, _predict owner) implementations: {len(impl)}")
    print("  shared by >1 forecaster:", [(k, v) for k, v in impl.most_common() if v > 1])
    for b in sorted(bc):
        sub = [r for r in rows if r["bucket"] == b]
        nimpl = len({(r["fit_owner"], r["predict_owner"]) for r in sub})
        st = collections.Counter(r["core_style"] for r in sub)
        print(f"{b}: n={len(sub)} distinct_impls={nimpl} core_style={dict(st)}")
    comp = [r for r in rows if r["is_composite"]]
    print("composite param kinds:", collections.Counter(r["estimator_param_kinds"] or "<meta-base only>" for r in comp).most_common())
    panel = [r for r in rows if "multiindex" in r["y_inner_mtype"] and not r["y_inner_accepts_all"]]
    print("native panel/hier y_inner (pd-multiindex listed):", len(panel),
          collections.Counter(r["bucket"] for r in panel))
    sv = collections.Counter(r["static_verdict"] for r in rows)
    print("static verdict:", {k: (v, f"{100 * v / n:.1f}%") for k, v in sorted(sv.items())})
    print("static verdict x bucket:", sorted(collections.Counter((r["bucket"], r["static_verdict"]) for r in rows).items()))
    li = collections.Counter(r["lib_input"] for r in rows if r["bucket"] == "B3")
    print("B3 lib_input:", li.most_common())
    for b in sorted(bc):
        print(b, [r["name"] for r in rows if r["bucket"] == b])


if __name__ == "__main__":
    main()
