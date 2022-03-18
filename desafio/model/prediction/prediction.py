import click
from sklearn.preprocessing import Normalizer

from desafio.persistence.microdata import load_sklearn_model, save_prediction_result
from desafio.exploratory_data_analysis.prepare import prepare_data_for_inference


@click.command()
@click.option("--model-path", type=click.STRING)
@click.option("--incoming-data-file-path", type=click.STRING)
@click.option("--output-prediction-path", type=click.STRING)
def main(model_path: str, incoming_data_file_path, output_prediction_path: str):
    model = load_sklearn_model(model_path)
    incoming_data, data_for_prediction = prepare_data_for_inference(incoming_data_file_path, Normalizer())
    incoming_data["default"] = int(model.predict(data_for_prediction)[0])
    save_prediction_result(output_prediction_path, incoming_data)


if __name__ == "__main__":
    main()
