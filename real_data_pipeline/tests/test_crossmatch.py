from src.crossmatch import crossmatch


def test_crossmatch_rejects_redshift_mismatch(mock_icecube_catalog, mismatched_plasma_catalog, crossmatch_columns):
    ci, cp = crossmatch_columns
    assert crossmatch(mock_icecube_catalog, mismatched_plasma_catalog, ci, cp, 2, .1).empty


def test_crossmatch_accepts_matching_catalog(mock_icecube_catalog, mock_plasma_catalog, crossmatch_columns):
    ci, cp = crossmatch_columns
    out = crossmatch(mock_icecube_catalog, mock_plasma_catalog, ci, cp, 2, .1)
    assert len(out) == 1
