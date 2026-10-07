#!/usr/bin/env python3
"""
Teste de razão de verossimilhança (Poisson) para atenuação exponencial
de contagens de neutrinos por coluna de plasma:

    mu_i(kappa) = mu0_i * exp(-kappa * Cp_i),   kappa >= 0

H0: kappa = 0   vs   H1: kappa > 0

Como kappa = 0 está na borda do espaço de parâmetros, a distribuição
assintótica de TS sob H0 é 1/2 chi2_0 + 1/2 chi2_1 (Chernoff, 1954).
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import chi2, norm

REQUIRED_COLS = ("counts_obs", "counts_pred", "plasma_column")


def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Colunas ausentes em {path}: {missing}")
    df = df[list(REQUIRED_COLS)].dropna()
    N = df["counts_obs"].to_numpy(dtype=float)
    mu0 = df["counts_pred"].to_numpy(dtype=float)
    Cp = df["plasma_column"].to_numpy(dtype=float)
    if np.any(N < 0) or np.any(mu0 <= 0) or np.any(Cp < 0):
        sys.exit("Dados inválidos: requer counts_obs >= 0, counts_pred > 0, plasma_column >= 0")
    return N, mu0, Cp


def log_likelihood(kappa, N, mu0, Cp):
    log_mu = np.log(mu0) - kappa * Cp
    return np.sum(N * log_mu - np.exp(log_mu) - gammaln(N + 1))


def fit_kappa(N, mu0, Cp):
    # Normaliza a escala para o otimizador não depender das unidades de Cp
    scale = np.max(Cp) if np.max(Cp) > 0 else 1.0
    nll = lambda k_s: -log_likelihood(k_s / scale, N, mu0, Cp)

    # dlogL/dkappa em 0: se <= 0, o máximo está na borda (logL é côncava em kappa)
    grad0 = np.sum(Cp * (mu0 - N))
    if grad0 <= 0:
        return 0.0

    hi = 1.0
    while -nll(hi) > -nll(hi / 2) and hi < 1e12:
        hi *= 2
    res = minimize_scalar(nll, bounds=(0.0, hi), method="bounded",
                          options={"xatol": 1e-10})
    return res.x / scale


def lr_test(N, mu0, Cp):
    kappa_hat = fit_kappa(N, mu0, Cp)
    logL1 = log_likelihood(kappa_hat, N, mu0, Cp)
    logL0 = log_likelihood(0.0, N, mu0, Cp)
    TS = max(2.0 * (logL1 - logL0), 0.0)
    p_value = 0.5 * chi2.sf(TS, df=1)          # mistura 1/2 chi2_0 + 1/2 chi2_1
    sigma = norm.isf(p_value) if p_value < 0.5 else 0.0   # = sqrt(TS)
    return kappa_hat, TS, p_value, sigma


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV com colunas: " + ", ".join(REQUIRED_COLS))
    args = ap.parse_args()

    N, mu0, Cp = load_data(args.csv)
    kappa_hat, TS, p, sigma = lr_test(N, mu0, Cp)

    print("=" * 31)
    print("RESULTADOS")
    print("=" * 31)
    print(f"fontes    = {len(N)}")
    print(f"kappa_hat = {kappa_hat:.6e}")
    print(f"TS        = {TS:.3f}")
    print(f"p-value   = {p:.3e}  (unilateral, Chernoff)")
    print(f"sigma     = {sigma:.2f}")
    if sigma > 5:
        print("Descoberta (>5 sigma)")
    elif sigma > 3:
        print("Evidência (>3 sigma)")
    else:
        print("Compatível com kappa = 0")


if __name__ == "__main__":
    main()
