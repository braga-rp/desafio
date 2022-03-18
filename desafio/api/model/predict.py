import pandas as pd


def prepare_data_for_inference(data, scaler):
    incoming_dataframe = pd.DataFrame.from_records([data])
    scaled_data = transform_data(incoming_dataframe, scaler).values
    return scaled_data


def transform_data(dataframe: pd.DataFrame, scaler) -> pd.DataFrame:
    columns_to_transform = [col for col in dataframe.select_dtypes(exclude="object")]
    fields_to_normalize = dataframe.filter(columns_to_transform).to_numpy()

    feature_scaled = scaler.fit_transform(fields_to_normalize)

    return pd.DataFrame(feature_scaled, columns=columns_to_transform)
