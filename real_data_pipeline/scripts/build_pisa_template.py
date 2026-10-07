"""
Template de dados reais (observado, sem reweighting) em binning multidimensional
PISA (energia x declinacao x Cp proxy), a partir do cruzamento exploratorio
ICECAT-1 x CHIME/FRB (ver scripts/build_exploratory_crossmatch.py).

Isso usa so a infraestrutura de binning/container do PISA (MultiDimBinning,
Map, MapSet) para histogramar dados JA REAIS em formato padrao reconhecido
por outras ferramentas PISA -- nao usa os estagios de fisica (fluxo,
oscilacao, area efetiva), que exigiriam MC verdade e tabelas oficiais do
IceCube que nao temos (ver issue #17, decisao "Rejected for current
milestone" para o Pipeline/Stages completo).

Requer o pacote `pisa` instalado (ver real_data_pipeline/README.md ou rode
dentro do container oficial ghcr.io/icecube/pisa).

Uso:
    python scripts/build_pisa_template.py [-o results/pisa_template.json]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet

ROOT = Path(__file__).resolve().parents[1]
ICE_PATH = ROOT / "data/real_samples/IceCube_Gold_Bronze_Tracks.tab"
CROSSMATCH_PATH = ROOT / "results/exploratory_icecat1_chimefrb.csv"


def load_merged():
    ice = pd.read_csv(ICE_PATH, sep="\t")
    ice["NAME"] = ice["NAME"].str.strip('"')
    cross = pd.read_csv(CROSSMATCH_PATH)
    merged = cross.merge(ice[["NAME", "RA", "DEC"]], left_on="event", right_on="NAME", how="left")
    if merged["RA"].isna().any():
        raise ValueError("Eventos sem RA/DEC apos o merge -- verifique build_exploratory_crossmatch.py")
    return merged


def build_binning():
    return MultiDimBinning([
        OneDimBinning(name="energy", tex=r"E_\nu", units="TeV", is_log=True, num_bins=6, domain=[50, 25000]),
        OneDimBinning(name="declination", tex=r"\delta", units="deg", is_lin=True, num_bins=6, domain=[-90, 90]),
        OneDimBinning(name="cp_proxy", tex=r"C_p\,{\rm proxy}", units="pc/cm^3", is_log=True, num_bins=5, domain=[5, 500]),
    ])


def make_maps(df, binning):
    """Histograma os dados reais (contagem de eventos e, separadamente, a
    soma de SIGNAL por bin) no binning 3D. Sem reweighting -- e so o dado
    observado, como o lado 'Data' de uma analise PISA."""
    e = np.clip(df["energy_tev"].to_numpy(), 50.001, 24999.999)
    d = df["DEC"].to_numpy()
    cp = np.clip(df["cp_proxy_dm_excess_pc_cm3"].to_numpy(), 5.001, 499.999)

    e_edges = binning["energy"].bin_edges.m
    d_edges = binning["declination"].bin_edges.m
    cp_edges = binning["cp_proxy"].bin_edges.m

    counts, _ = np.histogramdd((e, d, cp), bins=(e_edges, d_edges, cp_edges))
    signal_sum, _ = np.histogramdd((e, d, cp), bins=(e_edges, d_edges, cp_edges), weights=df["signal"].to_numpy())

    m_counts = Map(name="icecat1_counts", hist=counts, binning=binning,
                   tex=r"N_{\rm eventos}")
    m_signal = Map(name="icecat1_signal_sum", hist=signal_sum, binning=binning,
                   tex=r"\sum {\rm SIGNAL}")
    return MapSet([m_counts, m_signal], name="ICECAT-1 x CHIME/FRB (dados reais, sem reweighting)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(ROOT / "results/pisa_template.json"))
    args = ap.parse_args()

    df = load_merged()
    binning = build_binning()
    maps = make_maps(df, binning)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    maps.to_json(args.out)

    print(f"n eventos: {len(df)}")
    print(binning)
    print(f"\nTotal no histograma de contagens: {maps['icecat1_counts'].hist.sum()}")
    print(f"Template salvo em {args.out} (formato PISA MapSet, to_json/from_json)")


if __name__ == "__main__":
    main()
