import numpy as np, pandas as pd
from .inference import fit_model, expected_under_null

def bootstrap(df,c,cfg):
    rng=np.random.default_rng(int(cfg["random_seed"])); rows=[]
    for i in range(int(cfg["bootstrap_samples"])):
        sample=df.iloc[rng.integers(0,len(df),len(df))].reset_index(drop=True)
        try: r,_,_=fit_model(sample,c,cfg); rows.append({"replicate":i,"kappa_hat":r["kappa_hat"],"TS":r["TS"]})
        except RuntimeError: rows.append({"replicate":i,"kappa_hat":np.nan,"TS":np.nan})
    return pd.DataFrame(rows)

def calibrate_null(df,c,cfg,observed_ts):
    rng=np.random.default_rng(int(cfg["random_seed"])+1); mu=expected_under_null(df,c,cfg); rows=[]
    for i in range(int(cfg["null_simulations"])):
        sim=df.copy(); sim[c["counts_obs"]]=rng.poisson(mu)
        try: r,_,_=fit_model(sim,c,cfg); rows.append({"replicate":i,"TS":r["TS"],"kappa_hat":r["kappa_hat"]})
        except RuntimeError: rows.append({"replicate":i,"TS":np.nan,"kappa_hat":np.nan})
    out=pd.DataFrame(rows); good=out["TS"].dropna(); p=(1+(good>=observed_ts).sum())/(1+len(good))
    return out,float(p)

def subgroup_fits(df,c,cfg):
    e=df[c["energy"]]; bins=cfg["energy_bins_tev"]; results={}
    for lo,hi in zip(bins[:-1],bins[1:]):
        sub=df[(e>=lo)&(e<hi)]
        if len(sub)>=10: results[f"energy_{lo:g}_{hi:g}_TeV"]=fit_model(sub,c,cfg)[0]
    for name,sub in {"north":df[df[c["dec"]]>=0],"south":df[df[c["dec"]]<0]}.items():
        if len(sub)>=10: results[name]=fit_model(sub,c,cfg)[0]
    return results
