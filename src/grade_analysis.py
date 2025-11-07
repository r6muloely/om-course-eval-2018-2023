import pandas as pd

def merge_with_grades(sirs: pd.DataFrame, grades: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    keys = [cfg['columns']['section'], cfg['columns']['year'], cfg['columns']['term']]
    merged = pd.merge(sirs, grades, on=keys, how='left')
    return merged

def summarize_grade_alignment(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = ['mean_gpa','median_gpa','pct_A','pct_B','pct_C','pct_D','pct_F','pct_W']
    agg = df[cols].describe().T
    agg.reset_index(inplace=True)
    agg.rename(columns={'index':'metric'}, inplace=True)
    return agg
