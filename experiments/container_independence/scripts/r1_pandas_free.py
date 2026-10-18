"""r1: can a polars-only workflow run with pandas NOT importable?

Each case runs in a fresh subprocess of the interpreter given by ``--py``.
Run it once with an env WITHOUT pandas installed (the real test: any hidden pandas
dependency on the code path surfaces as ImportError), and once with a normal env
(records whether the code path imports pandas anyway: ``"pandas" in sys.modules``).

NOTE: an earlier version used a meta-path finder raising ImportError for pandas;
that is invalid because polars probes optional deps with importlib.util.find_spec,
which then raises, so it was replaced by a genuinely pandas-free venv.

Run (from the worktree root):
  # pandas-free venv: uv venv $V && uv pip install scikit-learn==1.9.1 polars==2.0.0
  #                   narwhals==2.27.1 scikit-base==1.2.0 packaging psutil
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/r1_pandas_free.py --py $V/bin/python --tag nopandas
  PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore \
      experiments/container_independence/scripts/r1_pandas_free.py --tag exp
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap

ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="dev", choices=["dev", "large"])
ap.add_argument("--py", default=sys.executable, help="interpreter to run the cases in")
ap.add_argument("--only", default=None, help="comma separated case-name prefixes")
ap.add_argument("--tag", default="exp")
args = ap.parse_args()

PRE = textwrap.dedent(
    """
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, polars as pl
    rng = np.random.default_rng(0)
    X = pl.DataFrame({f"c{i}": rng.normal(size=200) for i in range(3)})
    y = pl.Series("y", rng.normal(size=200))
    """
)

CASES = {
    "sklearn.StandardScaler set_output=polars": """
from sklearn.preprocessing import StandardScaler
out = StandardScaler().set_output(transform="polars").fit_transform(X)
RESULT = type(out).__module__ + "." + type(out).__name__
""",
    "sklearn.ColumnTransformer set_output=polars": """
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
ct = ColumnTransformer([("a", StandardScaler(), ["c0", "c1"]), ("b", MinMaxScaler(), ["c2"])])
out = ct.set_output(transform="polars").fit_transform(X)
RESULT = type(out).__name__ + str(out.columns)
""",
    "sklearn.Ridge fit/predict polars": """
from sklearn.linear_model import Ridge
p = Ridge().fit(X, y).predict(X)
RESULT = type(p).__name__ + str(p.shape)
""",
    "sklearn.train_test_split polars": """
from sklearn.model_selection import train_test_split
a, b = train_test_split(X, random_state=0)
RESULT = type(a).__name__ + str(a.shape)
""",
    "sklearn.StandardScaler on polars LazyFrame": """
from sklearn.preprocessing import StandardScaler
out = StandardScaler().fit_transform(X.lazy())
RESULT = type(out).__name__
""",
    "skrub import": """
import skrub
RESULT = skrub.__version__
""",
    "skrub.TableVectorizer polars": """
import skrub
out = skrub.TableVectorizer().fit_transform(X)
RESULT = type(out).__module__ + "." + type(out).__name__
""",
    "narwhals ops on polars": """
import narwhals.stable.v2 as nw
d = nw.from_native(X)
d = d.with_columns(g=(nw.col("c0") > 0), z=nw.col("c0") * 2)
out = d.group_by("g").agg(nw.col("z").mean())
RESULT = type(out.to_native()).__name__
""",
    "sktime import sktime.datatypes": """
import sktime.datatypes
RESULT = "ok"
""",
    "sktime check_is_mtype pl.DataFrame": """
from sktime.datatypes import check_is_mtype
RESULT = str(check_is_mtype(X, "pl.DataFrame", return_metadata=False))
""",
    "sktime NaiveForecaster import": """
from sktime.forecasting.naive import NaiveForecaster
RESULT = "ok"
""",
}


def run(case, code):
    prog = PRE + code + textwrap.dedent(
        """
        import sys, json
        print("__R__" + json.dumps({"result": RESULT, "pandas_loaded": "pandas" in sys.modules}))
        """
    )
    env = dict(os.environ)
    p = subprocess.run(
        [args.py, "-W", "ignore", "-c", prog], capture_output=True, text=True, env=env, timeout=600
    )
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("__R__")]
    if p.returncode == 0 and line:
        d = json.loads(line[-1][5:])
        return {"ok": True, **d}
    err = (p.stderr.strip().splitlines() or ["?"])[-1][:240]
    return {"ok": False, "error": err}


res = {}
for case, code in CASES.items():
    if args.only and not any(case.startswith(o) for o in args.only.split(",")):
        continue
    res[case] = run(case, code)
    print(case, json.dumps(res[case]))

os.makedirs("experiments/container_independence/results", exist_ok=True)
with open(f"experiments/container_independence/results/r1_pandas_free_{args.tag}.json", "w") as f:
    json.dump({"python": args.py, "results": res}, f, indent=1)
