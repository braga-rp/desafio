import matplotlib as plt
import os
import pandas as pd
from smart_open import open
import json
import pickle
from typing import Dict


def read_dataframe(dataset_path, separator: str, encoding: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path, sep=separator, encoding=encoding)


def save_to_html(output_folder, dataframe: pd.DataFrame, file_name: str):
    output_path = os.path.join(output_folder, file_name)
    original_df = dataframe.to_html()
    original_df_file = open(output_path, "w")
    original_df_file.write(original_df)
    original_df_file.close()


def save_plot(output_folder: str, file_name: str):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name, ".jpg")
    plt.savefig(output_path)


def save_dataframe(dataframe: pd.DataFrame, output_folder: str, file_name: str):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{file_name}.csv")
    dataframe.to_csv(output_path, index=False)


def read_params_file(params_file_path: str):
    with open(params_file_path, "r") as source_file:
        params = json.load(source_file)
        params["oob_score"][0] = True
        params["oob_score"][1] = False
    return params


def save_sk_learn_model(estimator, model_path: str, model_name: str):
    os.makedirs(model_path, exist_ok=True)
    path = f"{model_path}/{model_name}.sav"
    with open(path, "wb") as model_file:
        pickle.dump(estimator, model_file)


def load_sklearn_model(model_path: str):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    return model


def read_incoming_data(file_path: str):
    with open(file_path) as incoming_file:
        inc_data = json.load(incoming_file)
    return inc_data


def save_prediction_result(output_path: str, data: Dict):
    with open(output_path, "w") as result_file:
        json.dump(data, result_file)
