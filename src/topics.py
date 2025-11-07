import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import make_pipeline

def fit_topics(comments: pd.DataFrame, cfg: dict):
    text_col = cfg['columns']['comment_text']
    params = cfg['topics']
    vect = TfidfVectorizer(max_features=params['max_features'], ngram_range=tuple(params['ngram_range']), stop_words='english')
    X = vect.fit_transform(comments[text_col].fillna(''))
    lda = LatentDirichletAllocation(n_components=params['n_topics'], learning_method='batch', random_state=42)
    W = lda.fit_transform(X)
    H = lda.components_
    feature_names = vect.get_feature_names_out()
    return lda, W, H, feature_names

def top_words_per_topic(H, feature_names, n_top=10):
    out = []
    for k, comp in enumerate(H):
        top_idx = comp.argsort()[:-n_top-1:-1]
        out.append({'topic': k, 'top_words': [feature_names[i] for i in top_idx]})
    return pd.DataFrame(out)
