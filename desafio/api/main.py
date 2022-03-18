import os

from desafio.api.blueprints.create_app import create_app


def main():
    host = os.environ.get("HTTP_HOST", "0.0.0.0")
    port = os.environ.get("HTTP_PORT", "8080")
    debug = True if os.environ.get("DEBUG", "false") == "true" else True
    model_path = "resources/random_forest_model.sav"

    app = create_app(model_path)
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
