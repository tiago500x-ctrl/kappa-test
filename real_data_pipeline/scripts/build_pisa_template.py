"""
Template de dados reais (observado, sem reweighting) em binning multidimensional
PISA (energia x signalness x FAR x declinacao [x Cp proxy]), a partir do
cruzamento exploratorio ICECAT-1 x CHIME/FRB
(ver scripts/build_exploratory_crossmatch.py).

Usa so a infraestrutura de binning/container do PISA (icecat_binning_compact.py
-> MultiDimBinning/Map), nao os estagios de fisica (fluxo, oscilacao, area
efetiva), que exigiriam MC verdade e tabelas oficiais do IceCube que nao
temos (ver issue #17, decisao "Rejected for current milestone" para o
Pipeline/Stages completo).

Requer o pacote `pisa` instalado (ver real_data_pipeline/README.md ou rode
dentro do container oficial ghcr.io/icecube/pisa).

Uso:
    python scripts/build_pisa_template.py [--with-plasma] [-o results/pisa_template.csv]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from icecat_binning_compact import IcecatBinningConfig, histogram, to_dataframe
from pisa.core.map import MapSet
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ICE_PATH = ROOT / "data/real_samples/IceCube_Gold_Bronze_Tracks.tab"
CROSSMATCH_PATH = ROOT / "results/exploratory_icecat1_chimefrb.csv"


def load_merged():
    """Junta o cruzamento exploratorio (sem RA/DEC) de volta com o catalogo
    ICECAT-1 original, para recuperar a declinacao de cada evento."""
    ice = pd.read_csv(ICE_PATH, sep="\t")
    ice["NAME"] = ice["NAME"].str.strip('"')
    cross = pd.read_csv(CROSSMATCH_PATH)
    merged = cross.merge(ice[["NAME", "DEC"]], left_on="event", right_on="NAME", how="left")
    merged = merged.rename(columns={"DEC": "dec_deg", "far_per_yr": "far_per_year"})
    if merged["dec_deg"].isna().any():
        raise ValueError("Eventos sem DEC apos o merge -- verifique build_exploratory_crossmatch.py")
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(ROOT / "results/pisa_template.csv"))
    ap.add_argument("--with-plasma", action="store_true", help="inclui Cp proxy como dimensao extra do binning")
    args = ap.parse_args()

    df = load_merged()
    cfg = IcecatBinningConfig(plasma_edges=np.geomspace(5, 500, 11) if args.with_plasma else None)
    event_map = histogram(df, cfg, weights=None, name="icecat1_x_chimefrb")
    out = to_dataframe(event_map)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    json_path = str(Path(args.out).with_suffix(".json"))
    MapSet([event_map]).to_json(json_path)

    print(f"n eventos: {len(df)}")
    print(f"Conteúdo retido no histograma: {float(np.asarray(event_map.hist).sum()):.6g} / {len(df)}")
    print(f"Bins preenchidos: {len(out)}")
    print(f"Saída (CSV, bins não-vazios): {args.out}")
    print(f"Saída (formato nativo PISA, MapSet.to_json): {json_path}")


if __name__ == "__main__":
    main()
