import json, re
import pandas as pd
from flask import Flask, flash, Blueprint
from flask import render_template, request, jsonify, url_for, redirect, Response
import re, pickle
import time, os
from flask_restful import Api, Resource
from app import app

from app.extensions import predict_and_recommend, topic_names, text_processing, nonIT

api = Api(app)
path = os.getenv("SUB_PATH")


# @app.route("/")
# def index():
#   return redirect(url_for('home'))


@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")


@app.route("/predict")
def predict():
    # save user input in query
    query = request.args.get("query", "")
    labels, recommended_projects, probability_percentage = predict_and_recommend(query)
    valid_query = True if len(query) > 30 else False
    cleaned_text = text_processing(query)
    if valid_query and len(cleaned_text) > 1 and nonIT(probability_percentage):

        print(cleaned_text)
        return render_template(
            "index.html",
            query=query,
            valid_query=valid_query,
            cleaned_text=cleaned_text,
            labels=labels,
            group=topic_names,
            recommended_projects=recommended_projects,
            probability_percentage=probability_percentage,
        )

    else:
        query = "Please enter a valid text"
        flash(query)
        return render_template(
            "index.html",
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
        valid_query = True if len(query) > 30 else False
        cleaned_text = text_processing(query)
        labels, recommended_projects, probability_percentage = predict_and_recommend(
            query
        )
        if valid_query and len(cleaned_text) > 1 and nonIT(probability_percentage):
            # labels, projects = predict_and_recommend(query)
            # projects = projects.to_dict(orient="records")
            # bereich = {"Bereich": labels}
            # projects.insert(0, bereich)
            return jsonify(recommended_projects)
        else:
            respJson = json.dumps({"message": "Please enter a valid text"})

            return Response(respJson, status=409, mimetype="application/json")
            # return jsonify(respJson), 409

    # def get(self):
    #     respJson = {
    #         "message": "This route only receives POST requests. Please send your request in JSON format",
    #         "status": 400
    #     }
    #     return Response(jsonify(respJson), status=401, mimetype='application/json')


api.add_resource(RecommenededProjects, "/api")
