from flask import Flask
from config import config
import os


def create_app(config_name):
    """ create app context """
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    return app


app = create_app(os.getenv("FLASK_CONFIG") or "default")
from . import views
