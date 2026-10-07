from src.crossmatch import crossmatch


def test_crossmatch_rejects_redshift_mismatch(mock_icecube_catalog, mock_plasma_catalog, crossmatch_columns):
    ci, cp = crossmatch_columns
    plasma_far = mock_plasma_catalog.copy()
    plasma_far["z"] = 1.0
    assert crossmatch(mock_icecube_catalog, plasma_far, ci, cp, 2, .1).empty
