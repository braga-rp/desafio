import click
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, Normalizer

from desafio.persistence.microdata import read_dataframe, save_dataframe
from desafio.exploratory_data_analysis.prepare import split_features_target, transform_data
from desafio.exploratory_data_analysis.graphic import plot


@click.command()
@click.option("--dataset-path", type=click.STRING)
@click.option("--separator", type=click.STRING, default="\t")
@click.option("--encoding", type=click.STRING, default="utf-8")
@click.option("--target-column-name", type=click.STRING, default="default")
@click.option("--columns-to-drop", type=click.STRING)
@click.option("--n-rows", type=click.INT, default=4)
@click.option("--n-cols", type=click.INT, default=4)
@click.option("--graphs-output-folder-path")
@click.option("--dataset-output-folder-path")
def main(dataset_path: str, separator: str, encoding: str, target_column_name: str, columns_to_drop, n_rows: int,
         n_cols: int, dataset_output_folder_path: str, graphs_output_folder_path: str):
    x_health_dataframe = read_dataframe(dataset_path, separator, encoding)
    columns_to_drop = columns_to_drop.split(",")
    x_health_dataframe = x_health_dataframe.drop(columns_to_drop, axis=1)

    features, target = split_features_target(target_column_name, x_health_dataframe)
    save_dataframe(target, dataset_output_folder_path, "target")

    num_features = features.select_dtypes(exclude="object")
    plot(num_features, n_rows, n_cols, graphs_output_folder_path, "unchanged_features", ".jpg")

    normalized_features = transform_data(features, Normalizer())
    plot(normalized_features, n_rows, n_cols, graphs_output_folder_path, "normalizer", ".jpg")
    save_dataframe(normalized_features, dataset_output_folder_path, "normalizer")

    min_max_features = transform_data(features, MinMaxScaler())
    plot(min_max_features, n_rows, n_cols, graphs_output_folder_path, "min_max_scaler", ".jpg")
    save_dataframe(min_max_features, dataset_output_folder_path, "min_max_scaler")

    power_features = transform_data(features, PowerTransformer())
    plot(power_features, n_rows, n_cols, graphs_output_folder_path, "power_transformer", ".jpg")
    save_dataframe(power_features, dataset_output_folder_path, "power_transformer")


if __name__ == "__main__":
    main()
