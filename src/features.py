import pandas as pd
from sklearn import preprocessing


def one_hot_encoding(x: pd.Series):
    le = preprocessing.LabelEncoder()
    df_x = x.to_frame()

    le_x = df_x.apply(le.fit_transform)

    enc = preprocessing.OneHotEncoder()
    enc.fit(le_x)

    on_hot_labels = enc.transform(le_x).toarray()

    return on_hot_labels


def make_features(df):
    y = df["is_comic"]

    x = one_hot_encoding(df["video_name"])

    return x, y
