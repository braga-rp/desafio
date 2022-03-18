import pandas as pd
from typing import Tuple

from desafio.persistence.microdata import read_incoming_data


def transform_data(dataframe: pd.DataFrame, scaler) -> pd.DataFrame:
    columns_to_transform = [col for col in dataframe.select_dtypes(exclude="object")]
    fields_to_normalize = dataframe.filter(columns_to_transform).to_numpy()

    feature_scaled = scaler.fit_transform(fields_to_normalize)

    return pd.DataFrame(feature_scaled, columns=columns_to_transform)


def split_features_target(target_col_name: str, dataframe: pd.DataFrame) -> Tuple:
    features = dataframe.drop([target_col_name], axis=1)
    target = dataframe[target_col_name]

    return features, target


def prepare_data_for_inference(file_path: str, scaler):
    incoming_data = read_incoming_data(file_path)
    incoming_dataframe = pd.DataFrame([incoming_data])
    scaled_data = transform_data(incoming_dataframe, scaler).values
    return incoming_data, scaled_data
