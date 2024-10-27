import click
import joblib
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.model_selection import KFold, cross_validate, cross_val_score

from features import make_features
from models.models import make_model
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from src.dataset.dataset import make_dataset, download_dataset_nlp, split_and_save_dataset


@click.group()
def cli():
    pass


@click.command()
@click.option("--input_filename", default=f"{RAW_DATA_DIR}/train.csv", help="File training data")
@click.option("--model_dump_filename", default=f"{MODELS_DIR}/dump.json", help="File to dump model")
def train(input_filename, model_dump_filename):
    print("Training model...")
    df = make_dataset(input_filename)
    X, y = make_features(df)

    model = make_model()
    model.fit(X, y)

    return joblib.dump(model, model_dump_filename)


@click.command()
@click.option("--input_filename", default=f"{RAW_DATA_DIR}/test.csv", help="File training data")
@click.option("--model_dump_filename", default=f"{MODELS_DIR}/dump.json", help="File to dump model")
@click.option("--output_filename", default=f"{PROCESSED_DATA_DIR}/prediction.csv", help="Output file for predictions")
def predict(input_filename, model_dump_filename, output_filename):
    print("Predicting...")
    model = joblib.load(model_dump_filename)
    df = make_dataset(input_filename)
    X, y = make_features(df)
    y_pred = model.predict(X)

    df_final = df["video_name"].copy()
    df_final = pd.concat([df_final, pd.Series(name="prediction", data=y_pred)], axis=1)
    print(df_final.head)
    df_final.to_csv(output_filename, index=False)


@click.command()
@click.option("--input_filename", default=f"{RAW_DATA_DIR}/train.csv", help="File training data")
def evaluate(input_filename):
    print("Evaluating...")
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df)

    # Object with .fit, .predict methods
    model = make_model()

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


@click.command()
@click.option("--input_filename", default=f"{RAW_DATA_DIR}/dataset.csv", help="File where to download data")
def download(input_filename):
    print("Downloading...")
    download_dataset_nlp(input_filename)
    pass


@click.command()
@click.option("--input_filename", default=f"{RAW_DATA_DIR}/dataset.csv", help="File where to download data")
@click.option("--output_dirname", default=f"{RAW_DATA_DIR}", help="File where to download data")
def split_dataset(input_filename, output_dirname):
    print("Splitting dataset...")
    split_and_save_dataset(input_filename, output_dirname=output_dirname)
    pass


def evaluate_model(model, X, y):
    k = 10

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    n_estimators = [50, 100, 150, 200, 250, 300, 350]

    for val in n_estimators:
        score = cross_val_score(model, X, y, cv=kf,
                                scoring="accuracy")
        print(f'Average score({val}): {"{:.3f}".format(score.mean())}')

    cnt = 1
    # split()  method generate indices to split data into training and test set.
    for train_index, test_index in kf.split(X, y):
        print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
        cnt += 1

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    }

    scores = cross_validate(model, X, y, cv=kf, scoring=scoring)

    print(f"Result with {k} split :")
    for metric in scoring.keys():
        metric_scores = scores[f'test_{metric}']
        print(f"{metric.capitalize()} : {metric_scores.mean():.4f} (+/- {metric_scores.std() * 2:.4f})")

    return scores


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)
cli.add_command(download)
cli.add_command(split_dataset)

if __name__ == "__main__":
    cli()
