import pandas as pd
import yaml

def load_config(path='config/settings.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_all(cfg):
    sirs = pd.read_csv(cfg['data']['sirs_quant'])
    comments = pd.read_csv(cfg['data']['comments'])
    grades = pd.read_csv(cfg['data']['grades'])
    return sirs, comments, grades
