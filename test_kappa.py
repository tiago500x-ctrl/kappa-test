import numpy as np

from kappa_test import lr_test


def _sample(rng, kappa, n=200):
    mu0 = rng.uniform(5, 50, n)
    Cp = rng.exponential(1.0, n)
    N = rng.poisson(mu0 * np.exp(-kappa * Cp)).astype(float)
    return N, mu0, Cp


def test_recovers_kappa():
    rng = np.random.default_rng(0)
    k, TS, p, s = lr_test(*_sample(rng, 0.1, n=2000))
    assert abs(k - 0.1) < 0.02
    assert s > 5


def test_sigma_matches_sqrt_ts():
    rng = np.random.default_rng(1)
    k, TS, p, s = lr_test(*_sample(rng, 0.03))
    assert np.isclose(s, np.sqrt(TS), atol=1e-8)


def test_null_calibration():
    rng = np.random.default_rng(2)
    ps = np.array([lr_test(*_sample(rng, 0.0))[2] for _ in range(1000)])
    assert 0.03 < (ps < 0.05).mean() < 0.07      # taxa de falso positivo ~5%
    assert 0.44 < (ps == 0.5).mean() < 0.56      # ~metade na borda (kappa_hat = 0)
