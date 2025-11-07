import pandas as pd
from src.sirs_metrics import compute_section_level_metrics

def test_normalization():
    import yaml
    cfg = {'columns': {'clarity':'clarity','fairness':'fairness','overall':'overall'},
           'scales': {'sirs_min':1.0,'sirs_max':5.0}}
    df = pd.DataFrame({'clarity':[1,3,5],'fairness':[2,3,4],'overall':[1,5,3]})
    out = compute_section_level_metrics(df, cfg)
    assert out['clarity_norm'].iloc[0] == 0.0
    assert round(out['clarity_norm'].iloc[1],2) == 0.5
    assert out['clarity_norm'].iloc[2] == 1.0
