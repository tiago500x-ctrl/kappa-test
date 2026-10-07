"""
Fixtures compartilhadas por toda a suíte de testes.

Compatível com:
- src/inference.py
- src/robustness.py
- src/crossmatch.py

Objetivos:
1. Eliminar duplicação de generate_dataset().
2. Centralizar colunas e configurações.
3. Fornecer datasets padrão: H0 (kappa=0), sinal fraco/moderado/forte.
4. Fornecer catálogos mínimos para crossmatch (compatível e incompatível em redshift).
"""

import numpy as np
import pandas as pd
import pytest


# ============================================================
# Configuração usada pelo fit_model()
# ============================================================

@pytest.fixture(scope="session")
def inference_config():
    return {
        "efficiency_uncertainty": 0.05,
        "plasma_scale_uncertainty": 0.10,
        "kappa_max": 0.50,
        "bootstrap_samples": 20,
        "null_simulations": 50,
        "random_seed": 42,
        "energy_bins_tev": [10, 100, 1000],
    }


# ============================================================
# Mapeamento de colunas usado pela inference.py
# ============================================================

@pytest.fixture(scope="session")
def inference_columns():
    return {
        "counts_obs": "N",
        "signal_pred": "s",
        "background_pred": "b",
        "energy": "E",
        "dec": "dec",
    }


# ============================================================
# Mapeamento usado pelo crossmatch.py
# ============================================================

@pytest.fixture(scope="session")
def crossmatch_columns():
    ci = {
        "ra": "ra",
        "dec": "dec",
        "redshift": "z",
        "energy": "E",
        "counts_obs": "N",
        "signal_pred": "s",
        "background_pred": "b",
    }
    cp = {
        "ra": "ra",
        "dec": "dec",
        "redshift": "z",
        "dm": "dm",
    }
    return ci, cp


# ============================================================
# RNG compartilhado
# ============================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ============================================================
# Gerador sintético principal
# ============================================================

def generate_dataset(kappa, seed, n_events=500, signal=8.0, background=1.0):
    rng = np.random.default_rng(seed)
    cp = rng.uniform(0, 10, n_events)
    signal_arr = np.full(n_events, signal)
    background_arr = np.full(n_events, background)
    mu = background_arr + signal_arr * np.exp(-kappa * cp)
    counts = rng.poisson(mu)
    return pd.DataFrame({
        "N": counts,
        "s": signal_arr,
        "b": background_arr,
        "E": np.full(n_events, 100.0),
        "dec": rng.uniform(-80, 80, n_events),
        "plasma_column_scaled": cp,
    })


# ============================================================
# Dataset H0
# ============================================================

@pytest.fixture
def null_dataset():
    return generate_dataset(kappa=0.0, seed=12345, n_events=500)


# ============================================================
# Dataset com sinal fraco / moderado / forte
# ============================================================

@pytest.fixture
def weak_signal_dataset():
    return generate_dataset(kappa=0.02, seed=12345, n_events=500)


@pytest.fixture
def injected_dataset():
    return generate_dataset(kappa=0.08, seed=12345, n_events=500)


@pytest.fixture
def strong_signal_dataset():
    return generate_dataset(kappa=0.15, seed=12345, n_events=500)


# ============================================================
# Catálogo IceCube mínimo
# ============================================================

@pytest.fixture
def mock_icecube_catalog():
    return pd.DataFrame({
        "ra": [10.0],
        "dec": [20.0],
        "z": [0.30],
        "E": [100.0],
        "N": [10],
        "s": [8.0],
        "b": [1.0],
    })


# ============================================================
# Catálogo plasma compatível / incompatível em redshift
# ============================================================

@pytest.fixture
def mock_plasma_catalog():
    return pd.DataFrame({
        "ra": [10.01],
        "dec": [20.01],
        "z": [0.31],
        "dm": [250.0],
    })


@pytest.fixture
def mismatched_plasma_catalog():
    return pd.DataFrame({
        "ra": [10.01],
        "dec": [20.01],
        "z": [2.0],
        "dm": [250.0],
    })


# ============================================================
# Helper para estudos de cobertura (dataset sob medida)
# ============================================================

@pytest.fixture
def dataset_factory():
    return generate_dataset
