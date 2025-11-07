import os, random, numpy as np

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
