from pathlib import Path
import numpy as np,pandas as pd
rng=np.random.default_rng(42); n=400; k=.06
ra=rng.uniform(0,360,n); dec=np.degrees(np.arcsin(rng.uniform(-1,1,n))); z=rng.uniform(.05,1.2,n); E=10**rng.uniform(1,3.5,n)
C=10**rng.uniform(-2,1.2,n); s=np.clip(6*(E/100)**-.25,0.2,30); b=np.clip(2*(E/100)**-.6,.05,15); mu=b+s*np.exp(-k*C)
ice=pd.DataFrame({"ra_deg":ra,"dec_deg":dec,"redshift":z,"energy_tev":E,"counts_obs":rng.poisson(mu),"counts_signal_pred":s,"counts_background_pred":b})
pl=pd.DataFrame({"ra_deg":ra+rng.normal(0,.05,n),"dec_deg":dec+rng.normal(0,.05,n),"redshift":z+rng.normal(0,.02,n),"dm_pc_cm3":C*1e21/3.085677581491367e18})
Path("data/raw").mkdir(parents=True,exist_ok=True); ice.to_csv("data/raw/icecube.csv",index=False); pl.to_csv("data/raw/plasma.csv",index=False)
print("Dados sintéticos criados; κ injetado=0.06. Não são dados do IceCube.")
