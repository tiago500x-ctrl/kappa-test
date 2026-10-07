from src.robustness import bootstrap, calibrate_null


def test_bootstrap_and_null_shapes(inference_columns, inference_config, dataset_factory):
    cfg = {**inference_config, "bootstrap_samples": 5, "null_simulations": 10, "random_seed": 7, "energy_bins_tev": [10, 1000]}
    df = dataset_factory(kappa=0.08, seed=1, n_events=200)
    b = bootstrap(df, inference_columns, cfg)
    assert len(b) == 5
    n, p = calibrate_null(df, inference_columns, cfg, 1.0)
    assert len(n) == 10 and 0 < p <= 1
