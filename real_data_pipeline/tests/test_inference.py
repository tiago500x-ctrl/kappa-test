import numpy as np,pandas as pd
from src.inference import fit_model
C={"counts_obs":"N","signal_pred":"s","background_pred":"b","energy":"E","dec":"dec"}; CFG={"efficiency_uncertainty":.05,"plasma_scale_uncertainty":.1,"kappa_max":.5}
def make(seed=1,k=.08,n=1500):
    r=np.random.default_rng(seed); cp=r.uniform(0,10,n); s=np.full(n,8.); b=np.full(n,1.); N=r.poisson(b+s*np.exp(-k*cp)); return pd.DataFrame({"N":N,"s":s,"b":b,"E":100.,"dec":r.uniform(-80,80,n),"plasma_column_scaled":cp})
def test_recovers_injected_kappa():
    out,_,_=fit_model(make(),C,CFG); assert abs(out["kappa_hat"]-.08)<.03; assert abs(out["significance_sigma_asymptotic"]-np.sqrt(out["TS"]))<1e-12
def test_null_is_not_forced_positive():
    out,_,_=fit_model(make(k=0),C,CFG); assert out["kappa_hat"]<.04
