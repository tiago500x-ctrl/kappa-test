import argparse,yaml
from pathlib import Path
from src.io_utils import download,read_csv
from src.validate import validate_icecube,validate_plasma
from src.crossmatch import crossmatch
from src.inference import fit_model
from src.robustness import bootstrap,calibrate_null,subgroup_fits
from src.reporting import save_outputs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml"); ap.add_argument("--skip-download",action="store_true"); a=ap.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); p=cfg["paths"]
    if not a.skip_download:
        download(cfg["sources"]["icecube_url"],p["icecube_raw"]); download(cfg["sources"]["plasma_url"],p["plasma_raw"])
    ice,pl=read_csv(p["icecube_raw"]),read_csv(p["plasma_raw"]); ci,cp=cfg["columns"]["icecube"],cfg["columns"]["plasma"]
    validate_icecube(ice,ci); validate_plasma(pl,cp); m=cfg["matching"]
    merged=crossmatch(ice,pl,ci,cp,m["max_radius_deg"],m["max_redshift_difference"],m["require_plasma_behind_source"])
    if len(merged)<10: raise ValueError("Menos de 10 correspondências válidas")
    Path(p["merged"]).parent.mkdir(parents=True,exist_ok=True); merged.to_csv(p["merged"],index=False)
    result,grid,vals=fit_model(merged,ci,cfg["model"])
    boot=bootstrap(merged,ci,cfg["model"]); Path(p["bootstrap_csv"]).parent.mkdir(parents=True,exist_ok=True); boot.to_csv(p["bootstrap_csv"],index=False)
    null,p_emp=calibrate_null(merged,ci,cfg["model"],result["TS"]); null.to_csv(p["null_csv"],index=False)
    result["p_empirical_null"]=p_emp; result["bootstrap_ci95_percentile"]=boot["kappa_hat"].quantile([.025,.975]).tolist(); result["subgroups"]=subgroup_fits(merged,ci,cfg["model"])
    save_outputs(result,grid,vals,p["results"],p["profile_plot"]); print(result)
if __name__=="__main__": main()
