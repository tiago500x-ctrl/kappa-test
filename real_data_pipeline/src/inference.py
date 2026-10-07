import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

def _arrays(df,c):
    return tuple(df[c[k]].to_numpy(float) for k in ["counts_obs","signal_pred","background_pred"])+(df["plasma_column_scaled"].to_numpy(float),)

def fit_model(df,c,cfg):
    n,s,b,C=_arrays(df,c); se=float(cfg["efficiency_uncertainty"]); sp=float(cfg["plasma_scale_uncertainty"]); kmax=float(cfg["kappa_max"])
    def nll(x,fixed=None,counts=n):
        if fixed is None: la,k,le,lp=x
        else: la,le,lp=x; k=fixed
        A,ee,ep=np.exp(la),np.exp(le),np.exp(lp)
        mu=np.clip(b+A*s*ee*np.exp(-k*ep*C),1e-12,None)
        return np.sum(mu-counts*np.log(mu)+gammaln(counts+1))+.5*((ee-1)/se)**2+.5*((ep-1)/sp)**2
    alt=minimize(nll,[0,.02,0,0],method="L-BFGS-B",bounds=[(-5,5),(0,kmax),(-1,1),(-1,1)])
    null=minimize(lambda x:nll(x,0),[0,0,0],method="L-BFGS-B",bounds=[(-5,5),(-1,1),(-1,1)])
    if not alt.success or not null.success: raise RuntimeError("Falha de convergência")
    ts=max(0.,2*(null.fun-alt.fun)); grid=np.linspace(0,kmax,121)
    vals=[]
    for k in grid:
        r=minimize(lambda x:nll(x,k),alt.x[[0,2,3]],method="L-BFGS-B",bounds=[(-5,5),(-1,1),(-1,1)])
        vals.append(2*(r.fun-alt.fun))
    vals=np.asarray(vals); inside=grid[vals<=3.841458820694124]
    res={"kappa_hat":float(alt.x[1]),"TS":float(ts),"significance_sigma_asymptotic":float(np.sqrt(ts)),"p_one_sided_asymptotic":float(norm.sf(np.sqrt(ts))),"ci95_grid":[float(inside.min()),float(inside.max())] if len(inside) else [None,None],"A_hat":float(np.exp(alt.x[0])),"eta_eff_hat":float(np.exp(alt.x[2])),"eta_plasma_hat":float(np.exp(alt.x[3])),
         "A_hat_h0":float(np.exp(null.x[0])),"eta_eff_hat_h0":float(np.exp(null.x[1])),"n_rows":len(df)}
    return res,grid,vals

def expected_under_null(df,c,cfg):
    # Usa os nuisances ajustados SOB H0 (kappa=0), nao os do ajuste livre:
    # A e kappa sao parcialmente degenerados, entao usar A_hat do ajuste livre
    # enviesaria a calibracao da distribuicao nula de TS.
    n,s,b,C=_arrays(df,c)
    res,_,_=fit_model(df,c,cfg)
    return b+res["A_hat_h0"]*s*res["eta_eff_hat_h0"]
