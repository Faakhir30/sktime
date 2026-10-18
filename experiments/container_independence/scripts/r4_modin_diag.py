import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "experiments/container_independence")
from common import make_panel
def main():
    import modin.pandas as mpd, pandas as pd
    X = make_panel(20, 30, n_levels=1); mX = mpd.DataFrame(X)
    a = [(k, g.droplevel(0)) for k, g in X.groupby(level=0, sort=False)]
    b = [(k, g.droplevel(0)._to_pandas()) for k, g in mX.groupby(level=0, sort=False)]
    print("keys equal", [k for k,_ in a]==[k for k,_ in b], [k for k,_ in a][:4], [k for k,_ in b][:4])
    for (ka,va),(kb,vb) in zip(a,b):
        if not va.equals(vb):
            print(ka, kb, va.index[:2].tolist(), vb.index[:2].tolist(), va.dtypes.tolist(), vb.dtypes.tolist(), va.index.freq, vb.index.freq); break
    ra = X.groupby(level=0, sort=False)["c0"].apply(lambda s: float(s.mean()+s.iloc[-1]))
    rb = mX.groupby(level=0, sort=False)["c0"].apply(lambda s: float(s.mean()+s.iloc[-1]))._to_pandas()
    print("apply idx", ra.index[:4].tolist(), rb.index[:4].tolist(), ra.name, rb.name)
    
if __name__=='__main__':
    main()
