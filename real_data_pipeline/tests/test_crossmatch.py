import pandas as pd
from src.crossmatch import crossmatch
CI={"ra":"ra","dec":"dec","redshift":"z","energy":"E","counts_obs":"N","signal_pred":"s","background_pred":"b"}; CP={"ra":"ra","dec":"dec","redshift":"z","dm":"dm"}
def test_crossmatch_rejects_redshift_mismatch():
    ice=pd.DataFrame({"ra":[1.],"dec":[1.],"z":[.1],"E":[10.],"N":[1],"s":[1.],"b":[.1]}); pl=pd.DataFrame({"ra":[1.],"dec":[1.],"z":[1.],"dm":[100.]})
    assert crossmatch(ice,pl,CI,CP,2,.1).empty
