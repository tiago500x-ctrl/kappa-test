import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
PC_TO_CM=3.085677581491367e18

def crossmatch(ice,plasma,ci,cp,max_radius_deg=2.0,max_redshift_difference=0.15,require_plasma_behind_source=False):
    a=SkyCoord(ra=ice[ci["ra"]].to_numpy()*u.deg,dec=ice[ci["dec"]].to_numpy()*u.deg)
    b=SkyCoord(ra=plasma[cp["ra"]].to_numpy()*u.deg,dec=plasma[cp["dec"]].to_numpy()*u.deg)
    idx,sep,_=a.match_to_catalog_sky(b)
    out=ice.copy(); matched=plasma.iloc[idx].reset_index(drop=True)
    out["match_sep_deg"]=sep.deg
    out["plasma_redshift"]=matched[cp["redshift"]].to_numpy()
    out["dm_pc_cm3"]=matched[cp["dm"]].to_numpy()
    dz=np.abs(out[ci["redshift"]]-out["plasma_redshift"])
    mask=(out["match_sep_deg"]<=max_radius_deg)&(dz<=max_redshift_difference)
    if require_plasma_behind_source: mask &= out["plasma_redshift"]>=out[ci["redshift"]]
    out=out[mask].copy()
    out["plasma_column_cm2"]=out["dm_pc_cm3"]*PC_TO_CM
    out["plasma_column_scaled"]=out["plasma_column_cm2"]/1e21
    return out
