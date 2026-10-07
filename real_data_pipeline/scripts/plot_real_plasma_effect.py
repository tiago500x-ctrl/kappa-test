"""
Grafico de "efeito do plasma" com dados REAIS (ICECAT-1 x CHIME/FRB),
analogo ao da validacao sintetica mas SEM curva de kappa ajustada -- o Cp
proxy atual (nearest-neighbor) nao isola associacao fisica (ver
Real-Data-Analysis na wiki e issue #21), entao ajustar uma curva aqui seria
enganoso. So mostra o SIGNAL medio por bin de Cp proxy vs a media geral.

Uso:
    python scripts/plot_real_plasma_effect.py [-o outputs/real_plasma_effect.png]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results/exploratory_icecat1_chimefrb.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(ROOT / "outputs/real_plasma_effect.png"))
    ap.add_argument("--nbins", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    cp = df["cp_proxy_dm_excess_pc_cm3"].to_numpy()
    sig = df["signal"].to_numpy()

    edges = np.quantile(cp, np.linspace(0, 1, args.nbins + 1))
    edges[-1] += 1e-9
    idx = np.digitize(cp, edges) - 1

    centers, means, sems = [], [], []
    for b in range(args.nbins):
        m = idx == b
        if m.sum() == 0:
            continue
        centers.append(cp[m].mean())
        means.append(sig[m].mean())
        sems.append(sig[m].std(ddof=1) / np.sqrt(m.sum()))

    overall_mean = sig.mean()

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.errorbar(centers, means, yerr=sems, fmt="o", color="#1f77b4", capsize=4, ms=7,
                label="SIGNAL médio por bin de Cp proxy (binado por quantil)")
    ax.axhline(overall_mean, color="gray", ls="--", lw=1.2,
               label=f"média geral (sem binning) = {overall_mean:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Cp proxy — excesso de DM do FRB mais próximo (pc cm⁻³)")
    ax.set_ylabel("SIGNAL (probabilidade de origem astrofísica)")
    ax.set_title("Dados REAIS (ICECAT-1 × CHIME/FRB): nenhuma curva ajustada —\nsem evidência de atenuação com Cp proxy atual")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"média geral: {overall_mean:.4f}")
    print(f"Salvo em {args.out}")


if __name__ == "__main__":
    main()
