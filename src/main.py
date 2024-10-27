import click
import joblib
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_validate

from features import make_features
from models.models import make_model
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, ENCODED_DATA_DIR, MODELS_DIR
from src.dataset.dataset import make_dataset, split_and_save_dataset
from src.dataset.download_dataset import download_dataset_nlp


@click.group()
def cli():
    print("hello world")
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df)

    model = make_model()
    model.fit(X, y)

    return joblib.dump(model, model_dump_filename)


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(input_filename, model_dump_filename, output_filename):
    model = joblib.load(model_dump_filename)
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df)

    # Object with .fit, .predict methods
    model = make_model()

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Run k-fold cross validation. Print results
    pass


@click.command()
@click.option("--input_filename", default=f"{RAW_DATA_DIR}/names_train.csv", help="File where to download data")
def download(input_filename):
    download_dataset_nlp(input_filename)
    pass


@click.command()
@click.option("--input_filename", default=f"{RAW_DATA_DIR}/names_train.csv", help="File where to process data")
@click.option("--output_dirname", default=f"{ENCODED_DATA_DIR}", help="File where to process data")
def process_dataset(input_filename, output_dirname):
    split_and_save_dataset(input_filename, output_dirname)


def evaluate_model(model, X, y):
    k = 5  # Vous pouvez ajuster cette valeur selon vos besoins

    # Créer un objet KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Définir les métriques d'évaluation
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1': make_scorer(f1_score, average='weighted', zero_division=0)
    }

    # Exécuter la validation croisée
    scores = cross_validate(model, X, y, cv=kf, scoring=scoring)

    # Afficher les résultats
    print(f"Résultats de la validation croisée avec {k} plis:")
    for metric in scoring.keys():
        metric_scores = scores[f'test_{metric}']
        print(f"{metric.capitalize()} : {metric_scores.mean():.4f} (+/- {metric_scores.std() * 2:.4f})")

    return scores
    # Run k-fold cross validation. Print results
    pass


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)
cli.add_command(download)
cli.add_command(process_dataset)

if __name__ == "__main__":
    cli()
