import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_DIR, ENCODED_DATA_DIR
from src.features import make_features


def make_dataset(filename):
    return pd.read_csv(filename)


def split_and_save_dataset(dataset=f"{RAW_DATA_DIR}/names_train.csv", output_dirname=f"{ENCODED_DATA_DIR}",
                           test_size=0.2):
    full_ds = pd.read_csv(dataset)

    train, test = train_test_split(full_ds, test_size=test_size)

    train_path_x = os.path.join(output_dirname, f'train_{test_size}_x.npy')
    train_path_y = os.path.join(output_dirname, f'train_{test_size}_y.npy')
    test_path_x = os.path.join(output_dirname, f'test_{test_size}_x.npy')
    test_path_y = os.path.join(output_dirname, f'test_{test_size}_y.npy')

    train_features_x, train_features_y = make_features(train)
    test_features_x, test_features_y = make_features(test)

    np.save(train_path_x, train_features_x)
    np.save(train_path_y, train_features_y)
    np.save(test_path_x, test_features_x)
    np.save(test_path_y, test_features_y)