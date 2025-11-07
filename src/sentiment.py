import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure VADER lexicon is available
def _ensure_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

def score_sentiment(comments: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    _ensure_vader()
    sia = SentimentIntensityAnalyzer()
    text_col = cfg['columns']['comment_text']
    out = comments.copy()
    out['sent_compound'] = out[text_col].fillna('').apply(lambda t: sia.polarity_scores(str(t))['compound'])
    out['sent_label'] = pd.cut(out['sent_compound'], bins=[-1.0, -0.05, 0.05, 1.0], labels=['negative','neutral','positive'])
    return out
