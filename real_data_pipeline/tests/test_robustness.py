from src.robustness import bootstrap,calibrate_null
from test_inference import make,C,CFG
def test_bootstrap_and_null_shapes():
    cfg={**CFG,"bootstrap_samples":5,"null_simulations":10,"random_seed":7,"energy_bins_tev":[10,1000]}; df=make(n=200)
    b=bootstrap(df,C,cfg); assert len(b)==5
    n,p=calibrate_null(df,C,cfg,1.0); assert len(n)==10 and 0<p<=1
