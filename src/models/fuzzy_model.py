import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd

def build_fis():
    # Inputs
    clarity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'clarity')
    fairness = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'fairness')
    sentiment = ctrl.Antecedent(np.arange(-1, 1.01, 0.01), 'sentiment')
    # Output
    effectiveness = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'effectiveness')

    # Memberships (triangular for simplicity)
    clarity['low'] = fuzz.trimf(clarity.universe, [0.0, 0.0, 0.5])
    clarity['medium'] = fuzz.trimf(clarity.universe, [0.25, 0.5, 0.75])
    clarity['high'] = fuzz.trimf(clarity.universe, [0.5, 1.0, 1.0])

    fairness['low'] = fuzz.trimf(fairness.universe, [0.0, 0.0, 0.5])
    fairness['medium'] = fuzz.trimf(fairness.universe, [0.25, 0.5, 0.75])
    fairness['high'] = fuzz.trimf(fairness.universe, [0.5, 1.0, 1.0])

    sentiment['negative'] = fuzz.trimf(sentiment.universe, [-1.0, -1.0, 0.0])
    sentiment['neutral'] = fuzz.trimf(sentiment.universe, [-0.2, 0.0, 0.2])
    sentiment['positive'] = fuzz.trimf(sentiment.universe, [0.0, 1.0, 1.0])

    effectiveness['low'] = fuzz.trimf(effectiveness.universe, [0.0, 0.0, 0.5])
    effectiveness['medium'] = fuzz.trimf(effectiveness.universe, [0.25, 0.5, 0.75])
    effectiveness['high'] = fuzz.trimf(effectiveness.universe, [0.5, 1.0, 1.0])

    # Rules (example, can be refined):
    rules = [
        ctrl.Rule(clarity['high'] & fairness['high'] & sentiment['positive'], effectiveness['high']),
        ctrl.Rule(clarity['medium'] & fairness['medium'] & sentiment['neutral'], effectiveness['medium']),
        ctrl.Rule(clarity['low'] | fairness['low'] | sentiment['negative'], effectiveness['low']),
        ctrl.Rule(clarity['high'] & fairness['medium'] & sentiment['positive'], effectiveness['high']),
        ctrl.Rule(clarity['medium'] & fairness['high'] & sentiment['positive'], effectiveness['high']),
    ]

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim

def score_effectiveness(df: pd.DataFrame, clarity_col: str, fairness_col: str, sent_col: str) -> pd.DataFrame:
    sim = build_fis()
    out = df.copy()
    eff = []
    for _, row in out.iterrows():
        sim.input['clarity'] = float(row[clarity_col])
        sim.input['fairness'] = float(row[fairness_col])
        sim.input['sentiment'] = float(row[sent_col])
        sim.compute()
        eff.append(sim.output['effectiveness'])
    out['effectiveness_score'] = eff
    return out

# --- Sensitivity: shifted membership functions ------------------------------
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def _clip_tri(a, b, c, lo=0.0, hi=1.0):
    return [float(np.clip(a, lo, hi)), float(np.clip(b, lo, hi)), float(np.clip(c, lo, hi))]

def build_fis_with_delta(delta: float = 0.05):
    """
    Build a Mamdani FIS like the baseline, but with slightly shifted
    membership cutpoints to test robustness. 'delta' nudges the breakpoints.
    """
    # Universes
    clarity   = ctrl.Antecedent(np.arange(0, 1.001, 0.001), 'clarity')
    fairness  = ctrl.Antecedent(np.arange(0, 1.001, 0.001), 'fairness')
    sentiment = ctrl.Antecedent(np.arange(-1, 1.001, 0.001), 'sentiment')
    effectiveness = ctrl.Consequent(np.arange(0, 1.001, 0.001), 'effectiveness')

    # Inputs (triangular) with nudged vertices
    # Clarity/Fairness low/med/high
    # low peaks earlier; high starts later; medium widens slightly
    cl_lo  = _clip_tri(0.0, 0.0, 0.5 - delta)
    cl_md  = _clip_tri(0.25 - delta, 0.50, 0.75 + delta)
    cl_hi  = _clip_tri(0.50 + delta, 1.0, 1.0)

    fa_lo  = cl_lo
    fa_md  = cl_md
    fa_hi  = cl_hi

    # Sentiment negative/neutral/positive with wider neutral band
    se_neg = [-1.0, -1.0, 0.0 - delta]
    se_neu = [-0.2 - delta, 0.0, 0.2 + delta]
    se_pos = [0.0 + delta, 1.0, 1.0]

    clarity['low']    = fuzz.trimf(clarity.universe, cl_lo)
    clarity['medium'] = fuzz.trimf(clarity.universe, cl_md)
    clarity['high']   = fuzz.trimf(clarity.universe, cl_hi)

    fairness['low']    = fuzz.trimf(fairness.universe, fa_lo)
    fairness['medium'] = fuzz.trimf(fairness.universe, fa_md)
    fairness['high']   = fuzz.trimf(fairness.universe, fa_hi)

    sentiment['negative'] = fuzz.trimf(sentiment.universe, se_neg)
    sentiment['neutral']  = fuzz.trimf(sentiment.universe, se_neu)
    sentiment['positive'] = fuzz.trimf(sentiment.universe, se_pos)

    # Output unchanged partition
    effectiveness['low']    = fuzz.trimf(effectiveness.universe, [0.0, 0.0, 0.5])
    effectiveness['medium'] = fuzz.trimf(effectiveness.universe, [0.25, 0.5, 0.75])
    effectiveness['high']   = fuzz.trimf(effectiveness.universe, [0.5, 1.0, 1.0])

    # Monotonic rule set (same as baseline)
    rules = [
        ctrl.Rule(clarity['high'] & fairness['high'] & sentiment['positive'], effectiveness['high']),
        ctrl.Rule(clarity['medium'] & fairness['medium'] & sentiment['neutral'], effectiveness['medium']),
        ctrl.Rule(clarity['low'] | fairness['low'] | sentiment['negative'], effectiveness['low']),
        ctrl.Rule(clarity['high'] & fairness['medium'] & sentiment['positive'], effectiveness['high']),
        ctrl.Rule(clarity['medium'] & fairness['high'] & sentiment['positive'], effectiveness['high']),
    ]
    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)

def score_effectiveness_shifted(df, clarity_col: str, fairness_col: str, sent_col: str, delta: float = 0.05):
    """
    Compute effectiveness with a shifted FIS (cutpoints nudged by 'delta').
    Returns a copy of df with a new column 'effectiveness_score_shifted'.
    """
    sim = build_fis_with_delta(delta=delta)
    out = df.copy()
    scores = []
    for _, row in out.iterrows():
        sim.input['clarity']   = float(row[clarity_col])
        sim.input['fairness']  = float(row[fairness_col])
        sim.input['sentiment'] = float(row[sent_col])
        sim.compute()
        scores.append(sim.output['effectiveness'])
    out['effectiveness_score_shifted'] = scores
    return out
