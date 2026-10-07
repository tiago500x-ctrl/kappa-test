import os, sys, time, numpy as np, pandas as pd
import multiprocessing as mp
from .inference import fit_model, expected_under_null

def _limit_threads():
    # Cada tarefa de bootstrap/calibracao e um ajuste de 4 parametros: BLAS
    # multi-thread nao ajuda aqui, so faz processos competirem por CPU.
    # Paralelizamos nas replicas (entre processos), nao dentro de cada ajuste.
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

def _n_jobs(cfg):
    cfg_jobs = cfg.get("n_jobs")
    if cfg_jobs:
        return int(cfg_jobs)
    return max(1, (os.cpu_count() or 2) - 1)

def _progress(label, done, total, t0, every=25):
    if done % every == 0 or done == total:
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else float("nan")
        print(f"[{label}] {done}/{total}  ({elapsed:6.1f}s decorridos, ETA {eta:6.1f}s)", file=sys.stderr, flush=True)

_W = {}

def _init_worker(shared):
    _limit_threads()
    _W.update(shared)

def _bootstrap_task(args):
    i, idx = args
    c, cfg = _W["c"], _W["cfg"]
    df = pd.DataFrame({
        c["counts_obs"]: _W["n"][idx], c["signal_pred"]: _W["s"][idx],
        c["background_pred"]: _W["b"][idx], "plasma_column_scaled": _W["C"][idx],
    })
    try:
        r, _, _ = fit_model(df, c, cfg)
        return {"replicate": i, "kappa_hat": r["kappa_hat"], "TS": r["TS"]}
    except RuntimeError:
        return {"replicate": i, "kappa_hat": np.nan, "TS": np.nan}

def bootstrap(df, c, cfg):
    rng = np.random.default_rng(int(cfg["random_seed"]))
    total = int(cfg["bootstrap_samples"])
    n_rows = len(df)
    shared = {
        "n": df[c["counts_obs"]].to_numpy(float), "s": df[c["signal_pred"]].to_numpy(float),
        "b": df[c["background_pred"]].to_numpy(float), "C": df["plasma_column_scaled"].to_numpy(float),
        "c": c, "cfg": cfg,
    }
    tasks = [(i, rng.integers(0, n_rows, n_rows)) for i in range(total)]
    t0 = time.time(); rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(_n_jobs(cfg), initializer=_init_worker, initargs=(shared,)) as pool:
        for k, res in enumerate(pool.imap_unordered(_bootstrap_task, tasks, chunksize=4), start=1):
            rows.append(res); _progress("bootstrap", k, total, t0)
    rows.sort(key=lambda r: r["replicate"])
    return pd.DataFrame(rows)

def _null_task(args):
    i, seed = args
    c, cfg, mu = _W["c"], _W["cfg"], _W["mu"]
    counts = np.random.default_rng(seed).poisson(mu)
    df = pd.DataFrame({
        c["counts_obs"]: counts, c["signal_pred"]: _W["s"],
        c["background_pred"]: _W["b"], "plasma_column_scaled": _W["C"],
    })
    try:
        r, _, _ = fit_model(df, c, cfg)
        return {"replicate": i, "TS": r["TS"], "kappa_hat": r["kappa_hat"]}
    except RuntimeError:
        return {"replicate": i, "TS": np.nan, "kappa_hat": np.nan}

def calibrate_null(df, c, cfg, observed_ts):
    base_seed = int(cfg["random_seed"]) + 1
    mu = expected_under_null(df, c, cfg)
    total = int(cfg["null_simulations"])
    shared = {
        "s": df[c["signal_pred"]].to_numpy(float), "b": df[c["background_pred"]].to_numpy(float),
        "C": df["plasma_column_scaled"].to_numpy(float), "mu": mu, "c": c, "cfg": cfg,
    }
    # SeedSequence.spawn garante streams independentes e reprodutiveis por tarefa,
    # mesmo com conclusao fora de ordem entre processos (imap_unordered).
    seeds = np.random.SeedSequence(base_seed).spawn(total)
    tasks = list(enumerate(seeds))
    t0 = time.time(); rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(_n_jobs(cfg), initializer=_init_worker, initargs=(shared,)) as pool:
        for k, res in enumerate(pool.imap_unordered(_null_task, tasks, chunksize=4), start=1):
            rows.append(res); _progress("calibracao_nula", k, total, t0)
    rows.sort(key=lambda r: r["replicate"])
    out = pd.DataFrame(rows)
    good = out["TS"].dropna()
    p = (1 + (good >= observed_ts).sum()) / (1 + len(good))
    return out, float(p)

def subgroup_fits(df, c, cfg):
    e = df[c["energy"]]; bins = cfg["energy_bins_tev"]; results = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = df[(e >= lo) & (e < hi)]
        if len(sub) >= 10: results[f"energy_{lo:g}_{hi:g}_TeV"] = fit_model(sub, c, cfg)[0]
    for name, sub in {"north": df[df[c["dec"]] >= 0], "south": df[df[c["dec"]] < 0]}.items():
        if len(sub) >= 10: results[name] = fit_model(sub, c, cfg)[0]
    return results
