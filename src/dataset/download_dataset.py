import os

import requests

from src.config import RAW_DATA_DIR


def download_dataset_nlp(input_name="names_train.csv"):
    url = 'https://docs.google.com/spreadsheets/d/1HBs08WE5DLcHEfS6MqTivbyYlRnajfSVnTiKxKVu7Vs/export?format=csv&gid=1482158622'
    path = os.path.join(RAW_DATA_DIR, input_name)
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"Fichier téléchargé avec succès et enregistré à {path}")
    else:
        print(f"Erreur lors du téléchargement du fichier: {response.status_code}")


if __name__ == '__main__':
    download_dataset_nlp()
