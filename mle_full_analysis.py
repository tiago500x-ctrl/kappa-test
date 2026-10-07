#!/usr/bin/env python3
"""
Máxima verossimilhança Poissoniana com nuisance parameters para o teste de
atenuação de neutrinos por coluna de plasma.

Modelo:

    Cp_i(eta_plasma)  = plasma_column_scaled_i * (1 + eta_plasma * sigma_plasma_frac_i)
    eff_i(eta_eff)    = detector_efficiency_i  * (1 + eta_eff   * sigma_eff_frac_i)
    mu_i(A, kappa, eta_eff, eta_plasma) =
        A * counts_signal_pred_i * exp(-kappa * Cp_i(eta_plasma)) * eff_i(eta_eff)
        + counts_background_pred_i

    N_i ~ Poisson(mu_i)

eta_eff e eta_plasma são nuisances gaussianas padrão N(0,1) (penalidade
0.5*eta^2 na log-verossimilhança, perfiladas em cada hipótese).

H0: kappa = 0   vs   H1: kappa > 0

Como kappa = 0 está na borda do espaço de parâmetros, a distribuição
assintótica de TS sob H0 é 1/2 chi2_0 + 1/2 chi2_1 (Chernoff, 1954).
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import chi2, norm

REQUIRED_COLS = (
    "counts_obs",
    "counts_signal_pred",
    "counts_background_pred",
    "plasma_column_scaled",
    "plasma_column_unc_frac",
    "detector_efficiency",
    "detector_eff_unc_frac",
)


def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Colunas ausentes em {path}: {missing}")
    df = df[list(REQUIRED_COLS) + (["synthetic_kappa_true"] if "synthetic_kappa_true" in df.columns else [])].dropna()
    return df


def mu_model(theta, S, B, Cp, sig_plasma, eff0, sig_eff):
    A, kappa, eta_eff, eta_plasma = theta
    Cp_shift = Cp * (1.0 + eta_plasma * sig_plasma)
    eff_shift = eff0 * (1.0 + eta_eff * sig_eff)
    return A * S * np.exp(-kappa * Cp_shift) * eff_shift + B


def neg_log_poisson(mu, N):
    mu = np.clip(mu, 1e-12, None)
    return -np.sum(N * np.log(mu) - mu - gammaln(N + 1))


def nll(theta, N, S, B, Cp, sig_plasma, eff0, sig_eff):
    A, kappa = theta[0], theta[1]
    if A <= 0 or kappa < 0:
        return 1e18
    mu = mu_model(theta, S, B, Cp, sig_plasma, eff0, sig_eff)
    eta_eff, eta_plasma = theta[2], theta[3]
    return neg_log_poisson(mu, N) + 0.5 * eta_eff**2 + 0.5 * eta_plasma**2


def fit(data, fix_kappa=None, x0=None):
    N = data["counts_obs"].to_numpy(float)
    S = data["counts_signal_pred"].to_numpy(float)
    B = data["counts_background_pred"].to_numpy(float)
    Cp = data["plasma_column_scaled"].to_numpy(float)
    sig_plasma = data["plasma_column_unc_frac"].to_numpy(float)
    eff0 = data["detector_efficiency"].to_numpy(float)
    sig_eff = data["detector_eff_unc_frac"].to_numpy(float)
    args = (N, S, B, Cp, sig_plasma, eff0, sig_eff)

    if x0 is None:
        x0 = np.array([1.0, 0.05, 0.0, 0.0])

    if fix_kappa is None:
        bounds = [(1e-8, None), (0.0, None), (None, None), (None, None)]

        def obj(x):
            return nll(x, *args)

        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
    else:
        bounds = [(1e-8, None), (None, None), (None, None)]

        def obj3(x3):
            x = np.array([x3[0], fix_kappa, x3[1], x3[2]])
            return nll(x, *args)

        x03 = np.array([x0[0], x0[2], x0[3]])
        res = minimize(obj3, x03, method="L-BFGS-B", bounds=bounds)
        res.x = np.array([res.x[0], fix_kappa, res.x[1], res.x[2]])

    return res.x, res.fun, args


def profile_kappa(kappa_grid, args, x0):
    out = np.empty_like(kappa_grid)
    xcur = x0.copy()
    for i, k in enumerate(kappa_grid):
        N, S, B, Cp, sig_plasma, eff0, sig_eff = args

        def obj3(x3):
            x = np.array([x3[0], k, x3[1], x3[2]])
            return nll(x, N, S, B, Cp, sig_plasma, eff0, sig_eff)

        x03 = np.array([xcur[0], xcur[2], xcur[3]])
        bounds = [(1e-8, None), (None, None), (None, None)]
        res = minimize(obj3, x03, method="L-BFGS-B", bounds=bounds)
        out[i] = res.fun
        xcur = np.array([res.x[0], k, res.x[1], res.x[2]])
    return out


def confidence_interval_95(kappa_hat, nll_hat, args, x0):
    # varre kappa ate a diferenca de -2logL cruzar o limiar chi2_1(95%) = 3.84
    thresh = chi2.ppf(0.95, df=1)
    lo, hi = kappa_hat, kappa_hat
    step = max(kappa_hat * 0.2, 1e-3)

    # limite superior
    k = kappa_hat
    prev_nll = nll_hat
    while True:
        k_next = k + step
        nll_k = profile_kappa(np.array([k_next]), args, x0)[0]
        if 2 * (nll_k - nll_hat) >= thresh:
            # interpola entre k e k_next
            lo_k, hi_k = k, k_next
            lo_val, hi_val = 2 * (prev_nll - nll_hat), 2 * (nll_k - nll_hat)
            frac = (thresh - lo_val) / (hi_val - lo_val)
            hi = lo_k + frac * (hi_k - lo_k)
            break
        k = k_next
        prev_nll = nll_k
        step *= 1.5
        if k > kappa_hat + 50:
            hi = k
            break

    # limite inferior (so relevante se kappa_hat > 0)
    if kappa_hat <= 1e-10:
        lo = 0.0
    else:
        k = kappa_hat
        prev_nll = nll_hat
        step0 = max(kappa_hat * 0.2, 1e-3)
        step = step0
        while k - step > 0:
            k_next = k - step
            nll_k = profile_kappa(np.array([k_next]), args, x0)[0]
            if 2 * (nll_k - nll_hat) >= thresh:
                lo_k, hi_k = k_next, k
                lo_val, hi_val = 2 * (nll_k - nll_hat), 2 * (prev_nll - nll_hat)
                frac = (thresh - lo_val) / (hi_val - lo_val)
                lo = lo_k + frac * (hi_k - lo_k)
                break
            k = k_next
            prev_nll = nll_k
            step *= 1.2
        else:
            lo = 0.0
    return max(lo, 0.0), hi


def run_analysis(csv_path, outdir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_data(csv_path)

    theta1, nll1, args = fit(data, fix_kappa=None)
    theta0, nll0, _ = fit(data, fix_kappa=0.0)

    A_hat, kappa_hat, eta_eff_hat, eta_plasma_hat = theta1
    TS = max(2.0 * (nll0 - nll1), 0.0)
    p_value = 0.5 * chi2.sf(TS, df=1)
    sigma = norm.isf(p_value) if p_value < 0.5 else 0.0

    ci_lo, ci_hi = confidence_interval_95(kappa_hat, nll1, args, theta1)

    print("=" * 50)
    print("MLE POISSONIANA — ATENUAÇÃO POR COLUNA DE PLASMA")
    print("=" * 50)
    print(f"fontes         = {len(data)}")
    print(f"A_hat          = {A_hat:.4f}")
    print(f"kappa_hat      = {kappa_hat:.6e}")
    print(f"eta_eff_hat    = {eta_eff_hat:.4f}  (pull de eficiência)")
    print(f"eta_plasma_hat = {eta_plasma_hat:.4f}  (pull de coluna de plasma)")
    print("-" * 50)
    print(f"TS (LR)        = {TS:.3f}")
    print(f"p-value        = {p_value:.3e}  (unilateral, Chernoff)")
    print(f"significância  = {sigma:.2f} sigma")
    print(f"IC 95% kappa   = [{ci_lo:.6e}, {ci_hi:.6e}]")
    if "synthetic_kappa_true" in data.columns:
        print(f"kappa_true     = {data['synthetic_kappa_true'].iloc[0]:.6e}  (referência sintética)")
    print("-" * 50)
    if sigma > 5:
        print("Descoberta (>5 sigma)")
    elif sigma > 3:
        print("Evidência (>3 sigma)")
    else:
        print("Compatível com kappa = 0")
    print("=" * 50)

    N, S, B, Cp, sig_plasma, eff0, sig_eff = args
    mu_hat = mu_model(theta1, S, B, Cp, sig_plasma, eff0, sig_eff)

    # 1) perfil da likelihood
    span = max(ci_hi - kappa_hat, kappa_hat - ci_lo, kappa_hat * 0.5, 0.01)
    k_grid = np.linspace(max(0.0, kappa_hat - 2.5 * span), kappa_hat + 2.5 * span, 60)
    nll_grid = profile_kappa(k_grid, args, theta1)
    dts_grid = 2 * (nll_grid - nll1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_grid, dts_grid, color="#1f77b4", lw=2)
    ax.axhline(chi2.ppf(0.95, df=1), color="gray", ls="--", lw=1, label="limiar 95% (Δ=3.84)")
    ax.axvline(kappa_hat, color="#d62728", ls=":", lw=1.5, label=f"κ̂ = {kappa_hat:.4f}")
    ax.axvspan(ci_lo, ci_hi, color="#d62728", alpha=0.1, label="IC 95%")
    if "synthetic_kappa_true" in data.columns:
        ax.axvline(data["synthetic_kappa_true"].iloc[0], color="#2ca02c", ls="-", lw=1.5, label="κ verdadeiro")
    ax.set_xlabel("κ")
    ax.set_ylabel("Δ(-2 ln L)")
    ax.set_title("Perfil da verossimilhança para κ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/perfil_likelihood.png", dpi=150)
    plt.close(fig)

    # 2) observado vs previsto
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(mu_hat, N, yerr=np.sqrt(N), fmt="o", ms=3, alpha=0.4, color="#1f77b4", ecolor="#1f77b4")
    lims = [0, max(mu_hat.max(), N.max()) * 1.05]
    ax.plot(lims, lims, "--", color="gray", lw=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("contagens previstas (μ̂)")
    ax.set_ylabel("contagens observadas (N)")
    ax.set_title("Observado vs. previsto (melhor ajuste)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/observado_vs_previsto.png", dpi=150)
    plt.close(fig)

    # 3) efeito do plasma (fundo subtraído, para isolar o sinal de atenuação)
    signal_no_atten = A_hat * S * eff0
    ratio = (N - B) / np.clip(signal_no_atten, 1e-9, None)
    nbins = 12
    edges = np.quantile(Cp, np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    bin_idx = np.digitize(Cp, edges) - 1
    bin_centers, bin_means, bin_errs = [], [], []
    for b in range(nbins):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        bin_centers.append(Cp[m].mean())
        bin_means.append(ratio[m].mean())
        bin_errs.append(ratio[m].std(ddof=1) / np.sqrt(m.sum()))

    cp_line = np.geomspace(max(Cp.min(), 1e-3), Cp.max(), 200)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(bin_centers, bin_means, yerr=bin_errs, fmt="o", color="#1f77b4", label="dados (binados por quantil, fundo subtraído)")
    ax.plot(cp_line, np.exp(-kappa_hat * cp_line), color="#d62728", lw=2, label=f"ajuste: exp(-κ̂·Cp), κ̂={kappa_hat:.4f}")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="H0: sem atenuação")
    ax.set_xscale("log")
    ax.set_xlabel("coluna de plasma (escalada, log)")
    ax.set_ylabel("(N − B) / (A·S·eff)  [sinal normalizado]")
    ax.set_title("Efeito da coluna de plasma sobre o sinal (fundo subtraído)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/efeito_plasma.png", dpi=150)
    plt.close(fig)

    # 4) recuperação de kappa
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.errorbar([1], [kappa_hat], yerr=[[kappa_hat - ci_lo], [ci_hi - kappa_hat]],
                fmt="o", ms=8, color="#1f77b4", capsize=5, label="κ̂ ± IC 95%")
    if "synthetic_kappa_true" in data.columns:
        k_true = data["synthetic_kappa_true"].iloc[0]
        ax.axhline(k_true, color="#2ca02c", ls="--", lw=1.5, label=f"κ verdadeiro = {k_true:.4f}")
    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([])
    ax.set_ylabel("κ")
    ax.set_title("Recuperação de κ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/recuperacao_kappa.png", dpi=150)
    plt.close(fig)

    print()
    print("Gráficos salvos em:")
    for name in ("perfil_likelihood.png", "observado_vs_previsto.png", "efeito_plasma.png", "recuperacao_kappa.png"):
        print(f"  {outdir}/{name}")

    return dict(A_hat=A_hat, kappa_hat=kappa_hat, eta_eff_hat=eta_eff_hat,
                eta_plasma_hat=eta_plasma_hat, TS=TS, p_value=p_value, sigma=sigma,
                ci_lo=ci_lo, ci_hi=ci_hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV com colunas: " + ", ".join(REQUIRED_COLS))
    ap.add_argument("-o", "--outdir", default=".", help="diretório de saída para os gráficos")
    args = ap.parse_args()
    run_analysis(args.csv, args.outdir)


if __name__ == "__main__":
    main()
