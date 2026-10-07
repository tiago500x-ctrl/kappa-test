"""Binning compacto do ICECAT-1 com PISA (sem Pipeline/Stages físicos)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.core.map import Map

ALIASES = {
    "energy_tev": ("energy_tev", "ENERGY_TEV", "energy", "ENERGY"),
    "signalness": ("signalness", "SIGNALNESS", "signal", "SIGNAL"),
    "far_per_year": ("far_per_year", "FAR_PER_YEAR", "far", "FAR", "far_per_yr"),
    "dec_deg": ("dec_deg", "DEC_DEG", "dec", "DEC", "declination"),
    # cp_proxy_dm_excess_pc_cm3: excesso de DM do FRB mais proximo (ver
    # scripts/build_exploratory_crossmatch.py e DATA_PROVENANCE.md para as
    # ressalvas sobre esse proxy nao ser um Cp fisicamente validado ainda.
    "plasma_column_scaled": ("plasma_column_scaled", "cp_scaled", "C_p_scaled", "cp", "cp_proxy_dm_excess_pc_cm3"),
}


@dataclass(frozen=True)
class IcecatBinningConfig:
    energy_edges_tev: np.ndarray = field(default_factory=lambda: np.geomspace(10, 1e4, 21))
    signalness_edges: np.ndarray = field(default_factory=lambda: np.linspace(0, 1, 11))
    far_edges_per_year: np.ndarray = field(default_factory=lambda: np.geomspace(1e-4, 1e3, 15))
    dec_edges_deg: np.ndarray = field(default_factory=lambda: np.linspace(-90, 90, 19))
    plasma_edges: np.ndarray | None = None


def normalize(df: pd.DataFrame, require_plasma: bool = False) -> pd.DataFrame:
    required = ["energy_tev", "signalness", "far_per_year", "dec_deg"]
    if require_plasma:
        required.append("plasma_column_scaled")
    rename = {}
    missing = []
    for canonical in required:
        found = next((x for x in ALIASES[canonical] if x in df.columns), None)
        if found is None:
            missing.append(canonical)
        elif found != canonical:
            rename[found] = canonical
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")
    out = df.rename(columns=rename).copy()
    values = out[required].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Valores ausentes ou não finitos nas dimensões obrigatórias")
    if (out.energy_tev <= 0).any() or (out.far_per_year <= 0).any():
        raise ValueError("Energia e FAR devem ser positivos")
    if not out.signalness.between(0, 1).all() or not out.dec_deg.between(-90, 90).all():
        raise ValueError("Signalness ou declinação fora do domínio")
    if require_plasma and (out.plasma_column_scaled < 0).any():
        raise ValueError("A coluna de plasma não pode ser negativa")
    return out


def make_binning(cfg: IcecatBinningConfig | None = None) -> MultiDimBinning:
    cfg = cfg or IcecatBinningConfig()
    dims = [
        OneDimBinning("energy_tev", bin_edges=cfg.energy_edges_tev, is_log=True),
        OneDimBinning("signalness", bin_edges=cfg.signalness_edges),
        OneDimBinning("far_per_year", bin_edges=cfg.far_edges_per_year, is_log=True),
        OneDimBinning("dec_deg", bin_edges=cfg.dec_edges_deg),
    ]
    if cfg.plasma_edges is not None:
        dims.append(OneDimBinning("plasma_column_scaled", bin_edges=cfg.plasma_edges))
    return MultiDimBinning(dims)


def histogram(
    events: pd.DataFrame,
    cfg: IcecatBinningConfig | None = None,
    weights: str | Iterable[float] | None = None,
    name: str = "icecat1_events",
) -> Map:
    cfg = cfg or IcecatBinningConfig()
    df = normalize(events, require_plasma=cfg.plasma_edges is not None)
    binning = make_binning(cfg)
    names = [dim.name for dim in binning]
    edges = [np.asarray(dim.bin_edges, float) for dim in binning]
    if isinstance(weights, str):
        w = df[weights].to_numpy(float)
    elif weights is None:
        w = None
    else:
        w = np.asarray(list(weights), float)
        if len(w) != len(df):
            raise ValueError("Pesos e eventos devem ter o mesmo tamanho")
    hist, _ = np.histogramdd(df[names].to_numpy(float), bins=edges, weights=w)
    return Map(name=name, hist=hist, binning=binning)


def to_dataframe(event_map: Map) -> pd.DataFrame:
    hist = np.asarray(event_map.hist)
    rows = []
    for idx in np.argwhere(hist != 0):
        index = tuple(idx)
        row = {"count": float(hist[index])}
        for axis, dim in enumerate(event_map.binning):
            edges = np.asarray(dim.bin_edges, float)
            i = index[axis]
            row[f"{dim.name}_min"] = float(edges[i])
            row[f"{dim.name}_max"] = float(edges[i + 1])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Binning PISA compacto para ICECAT-1")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-csv", type=Path, default=Path("icecat1_binned.csv"))
    parser.add_argument("--weights")
    parser.add_argument("--with-plasma", action="store_true")
    args = parser.parse_args()

    cfg = IcecatBinningConfig(
        plasma_edges=np.linspace(0, 10, 21) if args.with_plasma else None
    )
    event_map = histogram(pd.read_csv(args.input_csv), cfg, args.weights)
    output = to_dataframe(event_map)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Conteúdo retido: {np.asarray(event_map.hist).sum():.6g}")
    print(f"Bins preenchidos: {len(output)}")
    print(f"Saída: {args.output_csv}")


if __name__ == "__main__":
    main()
