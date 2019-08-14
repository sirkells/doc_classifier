import json, re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib

# Save model to disk.
from gensim.test.utils import datapath
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re, pickle
import time
from nltk import FreqDist
from scipy.stats import entropy
import pandas as pd
import numpy as np
from scipy.stats import entropy
from gensim import corpora, models, similarities
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)


#load target data
loaded_target = joblib.load('models/target.sav')
stopwords_ul = joblib.load('models/stopwords.sav')
stopwords_sl = joblib.load('models/stopwords_sl')

pickle_in = open('App-Model/stopwords_all.pickle',"rb")
stopwords_all = pickle.load(pickle_in)


# Get Dictionary of the Trained Data
with open('App-Model/dictionary', 'rb') as data:
    dictionary = pickle.load(data)

# Get Projects data
with open('App-Model/df', 'rb') as data:
    projects = pickle.load(data)

#Get corpus 
with open('App-Model/corpus', 'rb') as data:
    corpus = pickle.load(data)



# later on, load trained model from file
lda_model =  models.LdaModel.load('App-Model/converted_model_skills_title_26_pref')
all_topic_distr_list = lda_model[corpus]

topic_names =  ["IT_Support", "HW_Tech_Embedded","PM_Support","SW_Arch","SW_Dev_Java", "PM_Projectleiter","SAP_Mgt","SW_Dev_Web","PM","SW_Test/Quality_Engr",
               "DevOps", "SW_Arch/Cloud","Consultant_SAP","SW_Dev_Web", "Business_Analyst/BI","PM_Mkt_Vertrieb","IT_Admin", "SW_Dev_Mobile/UI_Design","Infra_Server_Admin",
               "DB_Dev_Admin","IT_Consultant","Infra_Network_Admin","Data_Engr","SW_Dev_Web","SW_Dev_Web_Frontend","IT_Consultant_Operations"]


import string
from string import punctuation
from gensim.utils import SaveLoad
load_bigrams = SaveLoad.load('App-Model/bigram_skills_title')

def text_processing(text):
    """Normalize, tokenize, stem the original text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    cleaned: list of strings. List containing normalized and stemmed word tokens with bigrams
    """

    try:
        text = re.sub(r'(\d)',' ',text.lower())
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        tokens = word_tokenize(text)
        tokens_cleaned = [word for word in tokens if word not in stopwords_all and len(word) > 1]
        bigrams_tokens = load_bigrams[tokens_cleaned]
        
    except IndexError:
        pass

    return bigrams_tokens


def percentage(data):
    total = np.sum(data)
    perc_arr = np.array([(x/total)*100 for x in data])
    return perc_arr

def predict_bereich(text, model=lda_model, topics=topic_names):
    #dictionary.add_documents([text_processing(text)]) 
    clean = text_processing(text)
    bow = dictionary.doc2bow(clean)
    #print(text_processing(text))
    #model.update([bow])
    # get the topic contributions for the document chosen at random above
    topic_dist = model.get_document_topics(bow=bow)
    doc_distribution = np.array([topic[1] for topic in topic_dist])
    
    labels_array_percent = percentage(doc_distribution)
    #print(labels_array_percent)
    labels_array =labels_array_percent.argsort()[-2:][::-1]
    #print(topic_dist)
    
    index = doc_distribution.argmax()
    topic1 = topics[labels_array[0]]
    topic2 = topics[labels_array[1]]
    #print(labels_array[0], labels_array[1])
    #result = f'The project seems to be => {topic1} but could also be => {topic2}'
    #return result, topic1,topic2
    return topic1, topic2
    
def predict_and_recommend(text_data):
    bereich= predict_bereich(text_data)
    rec_projects = projects[projects['category2'] == bereich]
    return bereich, rec_projects


@app.route("/")
@app.route('/home')
def home():
    # save user input in query
    #print(type(stopwords))
    query = request.args.get('query', '')
    labels, recommended_projects = predict_and_recommend(query)

    return render_template(
        'home2.html',
        query=query,
        labels=labels,
        group=topic_names,
        recommended_projects=recommended_projects,
    )

@app.route("/about/")
def about():
    return render_template("about.html")

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()