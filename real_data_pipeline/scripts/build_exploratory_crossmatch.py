"""
Cruzamento exploratorio entre eventos reais do ICECAT-1 e bursts reais do
CHIME/FRB Catalog 1: para cada evento IceCube, acha o FRB mais proximo
angularmente e usa o excesso de DM (DM total - DM galactica media via
NE2001/YMW16) como proxy de Cp.

NAO ajusta kappa -- e so a tabela e os graficos exploratorios da Milestone 3
(issues #20-23). Ver DATA_PROVENANCE.md para o que esse proxy e e nao e.

Uso:
    python scripts/build_exploratory_crossmatch.py [-o results] [--plots-dir outputs]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
ICE_PATH = ROOT / "data/real_samples/IceCube_Gold_Bronze_Tracks.tab"
FRB_PATH = ROOT / "data/real_samples/CHIME_FRB_Catalog1.csv"


def load_frb(path):
    frb = pd.read_csv(path)
    for c in ["RAJ2000", "DEJ2000", "DM", "DMeNE2001", "DMeYMW16"]:
        frb[c] = pd.to_numeric(frb[c], errors="coerce")
    frb = frb.dropna(subset=["RAJ2000", "DEJ2000", "DM", "DMeNE2001", "DMeYMW16"]).reset_index(drop=True)
    frb["dm_galactic"] = frb[["DMeNE2001", "DMeYMW16"]].mean(axis=1)
    frb["dm_excess"] = (frb["DM"] - frb["dm_galactic"]).clip(lower=0)
    return frb


def load_icecube(path):
    ice = pd.read_csv(path, sep="\t")
    ice["I3TYPE"] = ice["I3TYPE"].str.strip('"')
    ice["NAME"] = ice["NAME"].str.strip('"')
    return ice


def crossmatch(ice, frb):
    ice_c = SkyCoord(ra=ice["RA"].to_numpy() * u.deg, dec=ice["DEC"].to_numpy() * u.deg)
    frb_c = SkyCoord(ra=frb["RAJ2000"].to_numpy() * u.deg, dec=frb["DEJ2000"].to_numpy() * u.deg)
    idx, sep, _ = ice_c.match_to_catalog_sky(frb_c)
    return pd.DataFrame({
        "event": ice["NAME"], "i3type": ice["I3TYPE"], "energy_tev": ice["ENERGY"],
        "far_per_yr": ice["FAR"], "signal": ice["SIGNAL"],
        "nearest_frb": frb.loc[idx, "Name"].to_numpy(), "sep_deg": sep.deg,
        "dm_total_pc_cm3": frb.loc[idx, "DM"].to_numpy(),
        "dm_galactic_pc_cm3": frb.loc[idx, "dm_galactic"].to_numpy(),
        "cp_proxy_dm_excess_pc_cm3": frb.loc[idx, "dm_excess"].to_numpy(),
    })


def make_plots(out, plots_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(out["sep_deg"], bins=30, color="#1f77b4", edgecolor="white")
    ax.axvline(out["sep_deg"].median(), color="#d62728", ls="--", label=f"mediana = {out['sep_deg'].median():.1f}°")
    ax.set_xlabel("separação angular ao FRB mais próximo (graus)")
    ax.set_ylabel("contagem")
    ax.set_title("ICECAT-1 x CHIME/FRB Cat.1: separação angular ao FRB mais próximo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{plots_dir}/exploratory_separation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(out["cp_proxy_dm_excess_pc_cm3"], out["energy_tev"], c=out["sep_deg"], cmap="viridis_r", s=18, alpha=0.8)
    ax.set_yscale("log")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("separação angular (graus)")
    ax.set_xlabel("Cp proxy — excesso de DM do FRB mais próximo (pc cm⁻³)")
    ax.set_ylabel("energia do evento IceCube (TeV)")
    ax.set_title("Exploratório: energia do neutrino vs. Cp proxy (sem ajuste de κ)")
    fig.tight_layout()
    fig.savefig(f"{plots_dir}/exploratory_energy_vs_cp.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--outdir", default=str(ROOT / "results"))
    ap.add_argument("--plots-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--ice", default=str(ICE_PATH))
    ap.add_argument("--frb", default=str(FRB_PATH))
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    ice = load_icecube(args.ice)
    frb = load_frb(args.frb)
    out = crossmatch(ice, frb)
    out.to_csv(f"{args.outdir}/exploratory_icecat1_chimefrb.csv", index=False)
    make_plots(out, args.plots_dir)

    print(f"n eventos: {len(out)}")
    print(out["sep_deg"].describe())
    corr = out[["energy_tev", "far_per_yr", "signal", "sep_deg", "cp_proxy_dm_excess_pc_cm3"]].corr(method="spearman")
    print("\nCorrelação de Spearman com Cp proxy:")
    print(corr["cp_proxy_dm_excess_pc_cm3"])


if __name__ == "__main__":
    main()
