from flask import Flask
from flask_json import FlaskJSON

from desafio.api.blueprints.prediction_blueprint import make_prediction_blueprint
from desafio.api.blueprints.root_blueprint import root


def create_app(model_path: str) -> Flask:
    app = Flask(__name__)

    FlaskJSON(app)

    predict_blueprint = make_prediction_blueprint(model_path)

    app.register_blueprint(predict_blueprint, url_prefix="/predict")
    app.register_blueprint(root, url_prefix='/')
    app.config['JSON_AS_ASCII'] = False

    return app
