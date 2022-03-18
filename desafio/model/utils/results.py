import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, plot_roc_curve, accuracy_score

from desafio.persistence.microdata import save_dataframe


def model_metrics_dataframe(estimator, grid_search_estimator, x_test, y_test, output_folder: str, model_name: str):
    scores_rf = pd.DataFrame({
        "Precision":
            [
                precision_score(y_test, estimator.predict(x_test)),
                precision_score(y_test, grid_search_estimator.predict(x_test))
            ],
        "Recall":
            [
                recall_score(y_test, estimator.predict(x_test)),
                recall_score(y_test, grid_search_estimator.predict(x_test))
            ],
        "F1 score":
            [
                f1_score(y_test, estimator.predict(x_test)),
                f1_score(y_test, grid_search_estimator.predict(x_test))
            ],
        "Accuracy":
            [
                accuracy_score(y_test, estimator.predict(x_test)),
                accuracy_score(y_test, grid_search_estimator.predict(x_test))
            ],
        "ROC AUC":
            [
                roc_auc_score(y_test, estimator.predict_proba(x_test)[:, 1]),
                roc_auc_score(y_test, grid_search_estimator.predict_proba(x_test)[:, 1])
            ]},
        index=["Before", "After"])
    save_dataframe(scores_rf, output_folder, model_name)


def plot_roc(estimator, x_test, y_test, output_folder_path: str, file_name: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_roc_curve(estimator, x_test, y_test, ax=ax, name="Random Forest")
    plt.margins(x=0, y=0)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.tight_layout()
    plt.savefig(f"{output_folder_path}/{file_name}")
