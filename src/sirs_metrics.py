import pandas as pd

def compute_section_level_metrics(sirs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Create normalized (0–1) columns for any SIRS items present in the data/config.
    Produces: clarity_norm, fairness_norm, overall_norm, and teaching_effectiveness_norm (if present).
    """
    df = sirs.copy()
    c = cfg['columns']
    smin, smax = cfg['scales']['sirs_min'], cfg['scales']['sirs_max']

    for key in ['clarity', 'fairness', 'overall', 'teaching_effectiveness']:
        raw = c.get(key)
        if raw and raw in df.columns:
            df[f'{key}_norm'] = (df[raw].astype(float) - smin) / (smax - smin)

    return df


def aggregate_trends(sirs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Unweighted section-level mean by term. Any *_norm present will be averaged.
    """
    ycol, tcol = cfg['columns']['year'], cfg['columns']['term']
    candidates = ['clarity', 'fairness', 'overall', 'teaching_effectiveness']
    metric_cols = [f'{k}_norm' for k in candidates if f'{k}_norm' in sirs.columns]
    trends = sirs.groupby([ycol, tcol], as_index=False)[metric_cols].mean()
    return trends


