#!/usr/bin/env python3
"""Gera um CSV sintético para testar kappa_test.py."""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--kappa", type=float, default=0.0, help="kappa verdadeiro")
ap.add_argument("--n", type=int, default=200, help="número de fontes")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("-o", "--out", default="synthetic_sources.csv")
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
mu0 = rng.uniform(5, 50, a.n)
Cp = rng.exponential(1.0, a.n)
N = rng.poisson(mu0 * np.exp(-a.kappa * Cp))
pd.DataFrame({"counts_obs": N, "counts_pred": mu0, "plasma_column": Cp}).to_csv(a.out, index=False)
print(f"{a.out}: {a.n} fontes, kappa={a.kappa}")
