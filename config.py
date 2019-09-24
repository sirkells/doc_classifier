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

 


class TestConfig(Config):
    # properties
    TESTING = True
    DEBUG = True




class WorkingConfig(Config):
    # properties
    TESTING = False
    DEBUG = False




config = {
    "dev": DevConfig,
    "test": TestConfig,
    "work": WorkingConfig,
    "default": DevConfig,
}
