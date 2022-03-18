import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from desafio.model.utils.grid_search import grid_search_estimator
from desafio.model.utils.results import model_metrics_dataframe, plot_roc
from desafio.persistence.microdata import read_dataframe, read_params_file, save_sk_learn_model


@click.command()
@click.option("--features-dataset-path")
@click.option("--params-file-path")
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
@click.option("--roc-curve-output-folder", type=click.STRING)
@click.option("--metric-output-folder", type=click.STRING)
@click.option("--save-model-path", type=click.STRING)
def main(features_dataset_path: str, params_file_path: str, target_dataset_path: str, separator: str, encoding: str,
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

    random_forest = RandomForestClassifier(random_state=random_state, class_weight=class_weight)
    random_forest.fit(x_train, y_train)
    save_sk_learn_model(random_forest, save_model_path, "random_forest_model")

    params = read_params_file(params_file_path)
    best_estimator = grid_search_estimator(random_forest, params, cross_validation_split, scoring,
                                           verbose, n_jobs, x_train, y_train)

    save_sk_learn_model(best_estimator, save_model_path, "optimized_random_forest_model")

    model_metrics_dataframe(random_forest, best_estimator, x_test, y_test, metric_output_folder, "random_forest_model")

    plot_roc(best_estimator, x_test, y_test, roc_curve_output_folder, "rf_roc_auc")


if __name__ == "__main__":
    main()
