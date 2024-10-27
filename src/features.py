import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer


def one_hot_encoding(x: pd.Series):
    vectorizer = HashingVectorizer(n_features=2**8)
    X = vectorizer.fit_transform(x)
    return X


def make_features(df):
    y = df["is_comic"]

    x = one_hot_encoding(df["video_name"])

    return x, y
