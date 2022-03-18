from flask import request
from flask_json import json_response
from sklearn.preprocessing import MinMaxScaler

from desafio.api.blueprints.base import BaseController
from desafio.api.model.predict import prepare_data_for_inference
from desafio.persistence.microdata import load_sklearn_model


def make_prediction_blueprint(model_path):
    controller = BaseController("predict")

    @controller.route("/", methods=["POST"])
    def predict():
        data = request.json
        model = load_sklearn_model(model_path)
        data_for_prediction = prepare_data_for_inference(data, MinMaxScaler())
        data["default"] = int(model.predict(data_for_prediction)[0])
        return json_response(data=data, status_=201)

    return controller
