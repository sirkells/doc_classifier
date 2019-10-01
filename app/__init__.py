from flask import Flask
from config import config
import os
from flask_cors import CORS


def sub_path(path):
    new_path = os.getenv("SUB_PATH") + path
    return new_path


def create_app(config_name):
    """ create app context """
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    @app.after_request
    def inject_cors_header(response):
       response.headers['Access-Control-Allow-Origin'] = '*'
       response.headers['Access-Control-Allow-Methods'] = 'PUT, POST, PATCH, DELETE, GET'
       response.headers['Access-Control-Allow-Headers'] = 'Origin, X-Requested-With, Content-Type, Accept, Authorization'  # noqa
       return response
    return app


app = create_app(os.getenv("FLASK_CONFIG") or "default")
CORS(app, resources={r"/api*": {"origins": "*"}}, headers="Content-Type")
app.jinja_env.globals.update(sub_path=sub_path)

from . import views
