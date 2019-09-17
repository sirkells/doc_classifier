import json, re
import pandas as pd
from flask import Flask, flash, Blueprint
from flask import render_template, request, jsonify, url_for, redirect
import re, pickle
import time, os
from flask_restful import Api, Resource

from models import (
    predict_and_recommend,
    topic_names,
    text_processing,
    PrefixMiddleware,
    prefix_route,
)

PREFIX = "/prod/doc_classifier"
# bp = Blueprint('doc_clasifier', __name__, template_folder='templates')
# app.register_blueprint(bp, url_prefix='/prod/doc_classifier')
subpath = os.environ.get("SUB_PATH")


# app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix='/prod/doc_classifier')
# app.route = prefix_route(app.route, PREFIX)
app = Flask(__name__)
# app.register_blueprint(bp, url_prefix='/prod/doc_classifier')
app.secret_key = "secret"
api = Api(app)
# def get_path(name):
#   subpath = os.environ.get('SUB_PATH')
#  route = url_for(name)
# fullpath = subpath + route
# return fullpath

# app.jinja_env.globals.update(get_path=get_path)


# @app.route("/")
# def index():
# return redirect(url_for("home"))
@app.route("/")
@app.route("/home")
def home():
    # save user input in query
    print(subpath)
    query = request.args.get("query", "")
    labels, recommended_projects = predict_and_recommend(query)
    valid_query = True if len(query) > 30 else False
    if valid_query:
        cleaned_text = text_processing(query)
        print(cleaned_text)
        return render_template(
            "home.html",
            query=query,
            valid_query=valid_query,
            cleaned_text=cleaned_text,
            labels=labels,
            group=topic_names,
            recommended_projects=recommended_projects,
        )

    else:
        query = "Please enter a valid text"
        flash(query)
        return render_template(
            "home.html",
            query=query,
            labels={},
            group={},
            recommended_projects=pd.DataFrame(columns=["Text", "title", "description"]),
        )


@app.route("/about")
def about():
    return render_template("about.html")


class RecommenededProjects(Resource):
    def post(self):
        postedData = request.get_json()
        query = postedData["skills"]
        labels, projects = predict_and_recommend(query)
        projects = projects.to_dict(orient="records")

        bereich = {"Bereich": labels}
        projects.insert(0, bereich)
        return jsonify(projects)


api.add_resource(RecommenededProjects, "/api")


def main():
    app.run(host="0.0.0.0", port=3001, debug=False)


if __name__ == "__main__":
    main()
