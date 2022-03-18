import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot(dataframe: pd.DataFrame, n_rows: int, n_cols: int, output_folder_path: str, file_name: str,
         file_extension: str):
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(25, 16))

    r = 0
    c = 0

    for i in dataframe:
        sns.distplot(dataframe[i], bins=15, kde=False, ax=ax[r][c])
        if c == n_cols - 1:
            r += 1
            c = 0
        else:
            c += 1

    output_path = os.path.join(output_folder_path, f"{file_name}{file_extension}")
    plt.savefig(output_path)


def corr_matrix(dataframe):
    correlation = dataframe.corr()
    return correlation.style.background_gradient(cmap='Blues')
