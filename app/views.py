import json, re
import pandas as pd
from flask import Flask, flash, Blueprint
from flask import render_template, request, jsonify, url_for, redirect
import re, pickle
import time, os
from flask_restful import Api, Resource
from app import app

from app.extensions import (
    predict_and_recommend,
    topic_names,
    text_processing,
    PrefixMiddleware,
    prefix_route,
)

api = Api(app)
path = os.getenv("SUB_PATH")


#@app.route("/")
#def index():
 #   return redirect(url_for('home'))

@app.route("/")
@app.route("/home")
def home():
    # save user input in query
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


# API


class RecommenededProjects(Resource):
    def post(self):
        postedData = request.get_json()
        query = postedData["skills"]
        labels, projects = predict_and_recommend(query)
        projects = projects.to_dict(orient="records")

        bereich = {"Bereich": labels}
        projects.insert(0, bereich)
        return jsonify(projects)

    def get(self):
        respJson = {
            "message": "This route only receives POST requets. Please send your request in JSON format"
        }
        return jsonify(respJson)


api.add_resource(RecommenededProjects, "/api")
