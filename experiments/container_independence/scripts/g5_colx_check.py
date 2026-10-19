import numpy as np, pandas as pd
from sktime.datatypes import convert_to
from sktime.forecasting.naive import NaiveForecaster
from x4_instrument import patched_base
idx=pd.date_range("2000",periods=20,freq="D",name="time")
y=pd.concat({f"s{i}":pd.DataFrame({"x":np.arange(20.)+i},index=idx) for i in range(3)},names=["h0"])
ref=NaiveForecaster().fit(y,fh=[1,2,3]).predict()
yX=convert_to(y.rename(columns={"x":"X"}),"polars_panel")
with patched_base("native",cutoff=True):
    out=NaiveForecaster().fit(yX,fh=[1,2,3]).predict()
o=convert_to(out,"pd-multiindex")
print(list(o.columns), np.allclose(o.values, ref.values), list(o.index.names), list(ref.index.names))
