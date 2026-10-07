import numpy as np

from src.inference import fit_model


def test_recovers_injected_kappa(inference_columns, inference_config, dataset_factory):
    df = dataset_factory(kappa=0.08, seed=1, n_events=1500)
    out, _, _ = fit_model(df, inference_columns, inference_config)
    assert abs(out["kappa_hat"] - .08) < .03
    assert abs(out["significance_sigma_asymptotic"] - np.sqrt(out["TS"])) < 1e-12


def test_null_is_not_forced_positive(inference_columns, inference_config, null_dataset):
    out, _, _ = fit_model(null_dataset, inference_columns, inference_config)
    assert out["kappa_hat"] < .04
