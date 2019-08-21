import json, re
import pandas as pd
from flask import Flask, flash, Blueprint
from flask import render_template, request, jsonify, url_for, redirect
import re, pickle
import time, os


from models import predict_and_recommend, topic_names, text_processing, PrefixMiddleware, prefix_route

PREFIX = '/prod/doc_classifier'
#bp = Blueprint('doc_clasifier', __name__, template_folder='templates')
app = Flask(__name__)
#app.register_blueprint(bp, url_prefix='/prod/doc_classifier')
#app.config["APPLICATION_ROOT"] = os.environ.get('SUB_PATH')
app.secret_key = 'secret'
#app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix='/prod/doc_classifier')
app.route = prefix_route(app.route, PREFIX)



@app.route("/")
def index():
    return redirect(url_for("home"))

@app.route("/test")
def test():
    return redirect(url_for("about"))
@app.route('/home')
def home():
    # save user input in query
    #print(type(stopwords))
    query = request.args.get('query', '')
    labels, recommended_projects = predict_and_recommend(query)
    valid_query = True if len(query) > 30 else False
    if valid_query:
        cleaned_text = text_processing(query)
        print(cleaned_text)
        return render_template(
       	'home2.html',
       	query=query,
        valid_query=valid_query,
        cleaned_text=cleaned_text,
        labels=labels,
        group=topic_names,
        recommended_projects=recommended_projects
    )
        
    else:
        query = 'Please enter a valid text'
        flash(query)
        return render_template(
        'home2.html',
        query=query,
        labels={},
        group={},
        recommended_projects= pd.DataFrame(columns=['Text', 'title', 'description'])
    )
    	

@app.route("/about")
def about():
    return render_template("about.html")

def main():
    app.run(host='0.0.0.0', port=3001, debug=False)


if __name__ == '__main__':
    main()
