import numpy as np
from iminuit import Minuit
from scipy.special import gammaln
from scipy.stats import norm

def _arrays(df, c):
    return tuple(df[c[k]].to_numpy(float) for k in ["counts_obs", "signal_pred", "background_pred"]) + (df["plasma_column_scaled"].to_numpy(float),)


def _make_nll(n, s, b, C, se, sp):
    def nll(la, k, le, lp):
        A, ee, ep = np.exp(la), np.exp(le), np.exp(lp)
        mu = np.clip(b + A * s * ee * np.exp(-k * ep * C), 1e-12, None)
        return np.sum(mu - n * np.log(mu) + gammaln(n + 1)) + .5 * ((ee - 1) / se) ** 2 + .5 * ((ep - 1) / sp) ** 2
    return nll


def _migrad(nll, k_fixed, kmax, start=(0.0, 0.02, 0.0, 0.0)):
    """MIGRAD de (la,k,le,lp); se k_fixed for dado, k fica fixo nesse valor."""
    la0, k0, le0, lp0 = start
    if k_fixed is None:
        m = Minuit(nll, la=la0, k=k0, le=le0, lp=lp0)
        m.limits["k"] = (0, kmax)
    else:
        m = Minuit(lambda la, le, lp: nll(la, k_fixed, le, lp), la=la0, le=le0, lp=lp0)
    m.errordef = Minuit.LIKELIHOOD  # cost = -logL (nao -2logL): 1 "sigma" <=> Delta(-logL)=0.5
    m.limits["la"] = (-5, 5); m.limits["le"] = (-1, 1); m.limits["lp"] = (-1, 1)
    m.print_level = 0
    m.migrad()
    return m


def fit_model(df, c, cfg, compute_ci=True):
    """Ajusta o modelo via iminuit (MIGRAD). Se compute_ci=True, roda MINOS
    (Delta(-2lnL)=3.84, cl=0.95) para o IC perfilado de kappa -- mais preciso
    que um grid fixo, mas mais caro. Em loops (bootstrap/calibracao
    nula/estudo de injecao), chamar com compute_ci=False."""
    n, s, b, C = _arrays(df, c)
    se = float(cfg["efficiency_uncertainty"]); sp = float(cfg["plasma_scale_uncertainty"]); kmax = float(cfg["kappa_max"])
    nll = _make_nll(n, s, b, C, se, sp)

    alt = _migrad(nll, None, kmax)
    if not alt.valid:
        raise RuntimeError("Falha de convergência (MIGRAD, H1)")
    null = _migrad(nll, 0.0, kmax)
    if not null.valid:
        raise RuntimeError("Falha de convergência (MIGRAD, H0)")

    kh = float(alt.values["k"])
    ts = max(0., 2 * (null.fval - alt.fval))

    ci = [None, None]
    if compute_ci:
        alt.minos("k", cl=0.95)
        me = alt.merrors["k"]
        if me.is_valid:
            ci = [max(0.0, kh + me.lower), kh + me.upper]
        elif kh <= 1e-9:
            # kappa_hat na fronteira: MINOS pode nao validar o lado inferior;
            # 0 e o limite natural, o lado superior geralmente ainda e valido.
            ci = [0.0, kh + me.upper if np.isfinite(me.upper) else None]

    # grid leve so para o grafico de perfil (MIGRAD em cada ponto; nao usado p/ IC)
    grid = np.linspace(0, kmax, 61)
    vals = np.empty_like(grid)
    for i, kk in enumerate(grid):
        mk = _migrad(nll, kk, kmax, start=(alt.values["la"], 0, alt.values["le"], alt.values["lp"]))
        vals[i] = 2 * (mk.fval - alt.fval)

    res = {
        "kappa_hat": kh, "TS": float(ts),
        "significance_sigma_asymptotic": float(np.sqrt(ts)),
        "p_one_sided_asymptotic": float(norm.sf(np.sqrt(ts))),
        "ci95_grid": [float(x) if x is not None else None for x in ci],
        "A_hat": float(np.exp(alt.values["la"])), "eta_eff_hat": float(np.exp(alt.values["le"])), "eta_plasma_hat": float(np.exp(alt.values["lp"])),
        "A_hat_h0": float(np.exp(null.values["la"])), "eta_eff_hat_h0": float(np.exp(null.values["le"])),
        "n_rows": len(df),
    }
    return res, grid, vals


def expected_under_null(df, c, cfg):
    # Usa os nuisances ajustados SOB H0 (kappa=0), nao os do ajuste livre:
    # A e kappa sao parcialmente degenerados, entao usar A_hat do ajuste livre
    # enviesaria a calibracao da distribuicao nula de TS.
    n, s, b, C = _arrays(df, c)
    res, _, _ = fit_model(df, c, cfg, compute_ci=False)
    return b + res["A_hat_h0"] * s * res["eta_eff_hat_h0"]
