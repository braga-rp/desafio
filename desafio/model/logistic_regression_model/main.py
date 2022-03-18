import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
import numpy as np

from desafio.model.utils.grid_search import grid_search_estimator
from desafio.model.utils.results import model_metrics_dataframe, plot_roc
from desafio.persistence.microdata import read_dataframe, save_sk_learn_model


@click.command()
@click.option("--features-dataset-path")
@click.option("--test-size", type=click.FLOAT, default=0.3)
@click.option("--class-weight", type=click.STRING, default="balanced")
@click.option("--random-state", type=click.INT, default=42)
@click.option("--target-dataset-path", type=click.STRING)
@click.option("--separator", type=click.STRING, default=",")
@click.option("--encoding", type=click.STRING, default="utf-8")
@click.option("--cross-validation-split", type=click.INT, default=5)
@click.option("--scoring", type=click.STRING, default="roc_auc")
@click.option("--verbose", type=click.INT, default=3)
@click.option("--n-jobs", type=click.INT, default=5)
@click.option("--save-model-path", type=click.STRING)
@click.option("--roc-curve-output-folder", type=click.STRING)
@click.option("--metric-output-folder", type=click.STRING)
def main(features_dataset_path: str, target_dataset_path: str, separator: str, encoding: str,
         class_weight: str, random_state: int, cross_validation_split: int, scoring: str, verbose: int, n_jobs: int,
         save_model_path: str, metric_output_folder: str, roc_curve_output_folder: str, test_size: float):
    features = read_dataframe(features_dataset_path, separator, encoding)
    features_columns = [col for col in features]
    target = pd.read_csv(target_dataset_path)
    target_col = [col for col in target]

    x_health_dataframe = features.join(target)
    x = x_health_dataframe[features_columns].values
    y = x_health_dataframe[target_col].values
    y_vec = y.reshape(y.shape[0])
    x_train, x_test, y_train, y_test = train_test_split(x, y_vec, test_size=test_size, random_state=random_state)

    logistic_regression = Pipeline(
        [
            ("varthres", VarianceThreshold()),
            ("select", SelectKBest(f_classif, k=14)),
            ("clf", LogisticRegression(random_state=42, max_iter=1000, solver="saga", C=0.01, penalty="l1",
                                       class_weight=class_weight))
        ]
    )

    lr_params = {"select__k": range(2, x.shape[1]), 'clf__penalty': ["none", "l2", "l1"],
                 'clf__C': np.logspace(-3, 3, 7)
                 }

    logistic_regression.fit(x_train, y_train)
    save_sk_learn_model(logistic_regression, save_model_path, "logistic_regression_model")

    best_estimator = grid_search_estimator(logistic_regression, lr_params, cross_validation_split, scoring,
                                           verbose, n_jobs, x_train, y_train)

    save_sk_learn_model(best_estimator, save_model_path, "optimized_logistic_model")

    model_metrics_dataframe(logistic_regression, best_estimator, x_test, y_test, metric_output_folder,
                            "logistic_regression_model")

    plot_roc(best_estimator, x_test, y_test, roc_curve_output_folder, "lr_roc_auc")


if __name__ == "__main__":
    main()
