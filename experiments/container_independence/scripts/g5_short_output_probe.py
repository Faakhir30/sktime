import numpy as np, pandas as pd
from sktime.datatypes import convert_to
from sktime.forecasting.naive import NaiveForecaster
idx=pd.date_range("2000",periods=20,freq="D",name="time")
y=pd.concat({f"s{i}":pd.DataFrame({"x":np.arange(20.)+i},index=idx) for i in range(3)},names=["h0"])
for fh in ([1],[1,2],[1,2,3]):
  for mt in ["polars_panel"]:
    try:
        out=NaiveForecaster().fit(convert_to(y,mt),fh=fh).predict(); 
        try: convert_to(out,"pd-multiindex"); print(fh,"predict ok, convert back ok")
        except Exception as e: print(fh,"predict ok; convert_to(pd-multiindex) FAILS:",str(e)[:60])
    except Exception as e: print(fh,"predict FAILS",str(e)[:80])
ys=pd.DataFrame({"x":np.arange(20.)},index=idx)
for fh in ([1],[1,2]):
    out=NaiveForecaster().fit(convert_to(ys,"pl.DataFrame"),fh=fh).predict()
    try: convert_to(out,"pd.DataFrame"); print("series",fh,"convert back ok")
    except Exception as e: print("series",fh,"convert FAILS",str(e)[:60])
# update with a 1- or 2-row polars y_new (streaming-style update)
import traceback
for n in (1, 2, 3):
    for mt, obj in [("pl.DataFrame", ys), ("polars_panel", y)]:
        f = NaiveForecaster().fit(convert_to(obj.iloc[:0] if False else obj, mt), fh=[1, 2, 3])
        new_idx = pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=n, freq="D", name="time")
        if mt == "pl.DataFrame":
            new = pd.DataFrame({"x": np.arange(n) + 100.0}, index=new_idx)
        else:
            new = pd.concat({f"s{i}": pd.DataFrame({"x": np.arange(n) + 100.0 + i}, index=new_idx) for i in range(3)}, names=["h0"])
        try:
            f.update(convert_to(new, mt)); p = f.predict()
            print("update", mt, f"n_new={n}", "ok", type(p).__name__)
        except Exception as e:
            fr = [f"{x.filename.split('/sktime/sktime/')[-1]}:{x.lineno}" for x in traceback.extract_tb(e.__traceback__) if "/sktime/sktime/" in x.filename]
            print("update", mt, f"n_new={n}", "FAILS", str(e)[:50], fr[-3:])
