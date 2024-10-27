import pandas as pd
import requests
from sklearn.model_selection import train_test_split


def make_dataset(filename):
    return pd.read_csv(filename)


def split_and_save_dataset(dataset, output_dirname,
                           test_size=0.2):
    full_ds = pd.read_csv(dataset)

    train, test = train_test_split(full_ds, test_size=test_size)

    train.to_csv(f'{output_dirname}/train.csv')
    test.to_csv(f'{output_dirname}/test.csv')


def download_dataset_nlp(input_name):
    url = 'https://docs.google.com/spreadsheets/d/1HBs08WE5DLcHEfS6MqTivbyYlRnajfSVnTiKxKVu7Vs/export?format=csv&gid=1482158622'
    response = requests.get(url)
    if response.status_code == 200:
        with open(input_name, 'wb') as f:
            f.write(response.content)
        print(f"Fichier téléchargé avec succès et enregistré à {input_name}")
    else:
        print(f"Erreur lors du téléchargement du fichier: {response.status_code}")
