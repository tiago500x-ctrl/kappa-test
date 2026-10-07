"""
Estudo de injecao: para kappa_true = 0.00, 0.01, ..., 0.10, gera N replicas
sinteticas (mesmo desenho de fontes, Poisson resampleado) e ajusta o modelo,
medindo vies, cobertura do IC 95% e poder estatistico (fracao que atinge
3 sigma / 5 sigma). Fase 3 do roadmap (cobertura dos intervalos) e Fase 5
(sensibilidade).

Uso:
    python scripts/injection_study.py [--n-reps 200] [--out outputs/injection]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import fit_model
from src.robustness import _limit_threads, _n_jobs

COLS = {"counts_obs": "counts_obs", "signal_pred": "signal_pred", "background_pred": "background_pred"}
MODEL_CFG = {"kappa_max": 2.0, "efficiency_uncertainty": 0.10, "plasma_scale_uncertainty": 0.20, "n_jobs": None}
KAPPA_VALUES = [round(0.01 * i, 2) for i in range(11)]  # 0.00 .. 0.10
SIGMA_THRESHOLDS = {"3sigma": 3.0, "5sigma": 5.0}

def _base_design(n=400, seed=123):
    rng = np.random.default_rng(seed)
    E = 10 ** rng.uniform(1, 3.5, n)
    C = 10 ** rng.uniform(-2, 1.2, n)
    s = np.clip(6 * (E / 100) ** -.25, 0.2, 30)
    b = np.clip(2 * (E / 100) ** -.6, .05, 15)
    return s, b, C

_W = {}

def _init_worker(shared):
    _limit_threads()
    _W.update(shared)

def _task(args):
    ki, kappa_true, rep, seed = args
    s, b, C, cfg = _W["s"], _W["b"], _W["C"], _W["cfg"]
    mu = b + s * np.exp(-kappa_true * C)
    counts = np.random.default_rng(seed).poisson(mu)
    df = pd.DataFrame({"counts_obs": counts, "signal_pred": s, "background_pred": b, "plasma_column_scaled": C})
    try:
        r, _, _ = fit_model(df, COLS, cfg)
        lo, hi = r["ci95_grid"]
        covered = (lo is not None) and (lo <= kappa_true <= hi)
        return {"kappa_true": kappa_true, "rep": rep, "kappa_hat": r["kappa_hat"], "TS": r["TS"],
                "sigma": r["significance_sigma_asymptotic"], "ci_lo": lo, "ci_hi": hi, "covered": covered}
    except RuntimeError:
        return {"kappa_true": kappa_true, "rep": rep, "kappa_hat": np.nan, "TS": np.nan,
                "sigma": np.nan, "ci_lo": None, "ci_hi": None, "covered": False}

def run(n_reps=200, out_dir="outputs/injection", base_seed=777):
    s, b, C = _base_design()
    shared = {"s": s, "b": b, "C": C, "cfg": MODEL_CFG}
    seeds = np.random.SeedSequence(base_seed).spawn(len(KAPPA_VALUES) * n_reps)
    tasks = []
    si = 0
    for ki, kv in enumerate(KAPPA_VALUES):
        for rep in range(n_reps):
            tasks.append((ki, kv, rep, seeds[si])); si += 1
    total = len(tasks)
    print(f"Rodando {total} ajustes ({len(KAPPA_VALUES)} valores de kappa x {n_reps} replicas)...", flush=True)
    t0 = time.time(); rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(_n_jobs(MODEL_CFG), initializer=_init_worker, initargs=(shared,)) as pool:
        for k, res in enumerate(pool.imap_unordered(_task, tasks, chunksize=4), start=1):
            rows.append(res)
            if k % 100 == 0 or k == total:
                el = time.time() - t0; rate = k / el if el > 0 else 0
                eta = (total - k) / rate if rate > 0 else float("nan")
                print(f"[injection] {k}/{total}  ({el:6.1f}s decorridos, ETA {eta:6.1f}s)", file=sys.stderr, flush=True)

    df = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out_dir}/injection_raw.csv", index=False)

    summary = []
    for kv in KAPPA_VALUES:
        sub = df[df["kappa_true"] == kv]
        khat = sub["kappa_hat"].dropna()
        row = {
            "kappa_true": kv, "n_reps": len(sub), "n_ok": len(khat),
            "kappa_hat_mean": float(khat.mean()), "kappa_hat_std": float(khat.std(ddof=1)),
            "bias": float(khat.mean() - kv),
            "coverage_95": float(sub["covered"].mean()),
        }
        for label, thr in SIGMA_THRESHOLDS.items():
            row[f"power_{label}"] = float((sub["sigma"] >= thr).mean())
        summary.append(row)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{out_dir}/injection_summary.csv", index=False)
    print(summary_df.to_string(index=False))
    return df, summary_df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=200)
    ap.add_argument("--out", default="outputs/injection")
    a = ap.parse_args()
    run(n_reps=a.n_reps, out_dir=a.out)
