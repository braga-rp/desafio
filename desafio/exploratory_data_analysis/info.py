import pandas as pd
import click
import numpy as np

from desafio.persistence.microdata import save_to_html, read_dataframe


@click.command()
@click.option("--dataset-path", type=click.STRING, help="The path of the dataset to be analyzed")
@click.option("--separator", type=click.STRING, default="\t", help="The separator of the the file")
@click.option("--encoding", type=click.STRING, default="utf-8", help="The encoding of the file")
@click.option("--replace", type=click.BOOL, default=True, help="Flag for replacing or not some value in the dataset")
@click.option("--replace-ref", type=click.STRING, default="missing", help="The value to replace")
@click.option("--replace-for", default=np.nan, help="The replacement will be performed bi this value")
@click.option("--output-folder", type=click.STRING, help="The folder to save the results")
def main(dataset_path: str, separator: str, encoding: str, replace: bool, replace_ref: str, replace_for,
         output_folder: str):
    dataframe = read_dataframe(dataset_path, separator, encoding)
    if replace:
        dataframe.replace(replace_ref, replace_for, inplace=True)
    columns = []
    uniques = []
    missing = []
    example = []
    _type = []
    for column in dataframe:
        unique = set(dataframe[column])
        columns.append(column)
        uniques.append(len(unique))
        missing.append(dataframe[column].isnull().mean() * 100)
        example.append(list(unique)[:5])
        _type.append(dataframe[column].dtype)
    data = [columns, uniques, missing, _type, example]
    info_df = pd.DataFrame(data)
    info_df = info_df.transpose()
    info_df.columns = ["column", "n_uniques", "missing", "type", "example"]

    save_to_html(output_folder, dataframe, "original_dataframe.html")
    save_to_html(output_folder, info_df, "info_dataframe.html")


if __name__ == "__main__":
    main()
