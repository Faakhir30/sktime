"""verify0: value-equality check of r1's third-party pyarrow set_output adapter (r1 checked only type+names),
plus protocol introspection via typing.get_protocol_members-like scan of source text.
Run: PYTHONPATH=.:experiments/container_independence $EXP_PY -W ignore <this>"""
import sys, os, inspect, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))
import numpy as np, pandas as pd, polars as pl
import r1_sklearn_pyarrow_adapter as proto
from sklearn.utils import _set_output as so
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
src = inspect.getsource(so.ContainerAdapterProtocol)
defs = [l.strip().split("(")[0][4:] for l in src.splitlines() if l.strip().startswith("def ")]
proto.register()
rng = np.random.default_rng(0)
Xpd = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc")); Xpl = pl.from_pandas(Xpd)
mk = {"SS": lambda: StandardScaler(),
      "CT": lambda: ColumnTransformer([("s", StandardScaler(), ["a", "b"]), ("m", MinMaxScaler(), ["c"])]),
      "FU": lambda: FeatureUnion([("s", StandardScaler()), ("m", MinMaxScaler())]),
      "PL": lambda: make_pipeline(StandardScaler(), MinMaxScaler())}
res = {}
for k, f in mk.items():
    for nm, X in [("pd", Xpd), ("pl", Xpl)]:
        ref = f().fit_transform(Xpd.to_numpy() if k != "CT" else Xpd)
        ref = np.asarray(ref)
        o = f().set_output(transform="pyarrow").fit_transform(X)
        got = np.column_stack([c.to_numpy() for c in o.columns])
        res[f"{k}<-{nm}"] = bool(np.allclose(ref, got))
# does set_output validate unknown names? (extension only works after register)
print(json.dumps({"protocol_defs": defs, "values_equal": res, "n_ok": sum(res.values())}, indent=1))
