import os
from pymongo import MongoClient
from datetime import timedelta


class Config:
    # security
    WTF_CSRF_ENABLED = True
    SECRET_KEY = os.environ.get("SECRET_KEY", os.urandom(24).hex())

    @staticmethod
    def init_app(app):
        pass


class DevConfig(Config):
    # properties
    TESTING = False
    DEBUG = True

    MONGODB_SETTINGS = {
        "db": "projectfinder",
        "host": os.getenv("MONGO_HOST"),
        "port": int(os.getenv("MONGO_PORT", "27017")),
    }


class TestConfig(Config):
    # properties
    TESTING = True
    DEBUG = True

    MONGODB_SETTINGS = {
        "db": "projectfinder-test",
        "host": os.getenv("MONGO_HOST", "mongo"),
        "port": int(os.getenv("MONGO_PORT", "27017")),
    }


class WorkingConfig(Config):
    # properties
    TESTING = False
    DEBUG = False

    # persistence layer
    MONGODB_SETTINGS = {
        "db": "projectfinder",
        "host": os.getenv("MONGO_HOST", "10.10.250.0"),
        "port": int(os.getenv("MONGO_PORT", "27017")),
    }


config = {
    "dev": DevConfig,
    "test": TestConfig,
    "work": WorkingConfig,
    "default": DevConfig,
}
