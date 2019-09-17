import pandas as pd
import numpy as np
import re, pickle, time, json

import nltk
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib

import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
from gensim.utils import SaveLoad

import string
from string import punctuation

# load target data
pickle_in = open("App-Model/stopwords_all.pickle", "rb")
stopwords_all = pickle.load(pickle_in)


# Get Dictionary of the Trained Data
with open("App-Model/dictionary", "rb") as data:
    dictionary = pickle.load(data)

# Get Projects data
with open("App-Model/df", "rb") as data:
    projects = pickle.load(data)

# projects.drop_duplicates(subset='description', inplace=True)
# Get corpus
with open("App-Model/corpus", "rb") as data:
    corpus = pickle.load(data)

# load flatlist of skills
with open("App-Model/flatlist", "rb") as data:
    all_skills = pickle.load(data)


# later on, load trained model from file
lda_model = models.LdaModel.load("App-Model/converted_model_skills_title_26_pref")
all_topic_distr_list = lda_model[corpus]

topic_names = [
    "IT_Support",
    "HW_Tech_Embedded",
    "PM_Support",
    "SW_Arch",
    "SW_Dev_Java",
    "PM_Projectleiter",
    "SAP_Mgt",
    "SW_Dev_Web",
    "PM",
    "SW_Test/Quality_Engr",
    "DevOps",
    "SW_Arch/Cloud",
    "Consultant_SAP",
    "SW_Dev_Web",
    "Business_Analyst/BI",
    "PM_Mkt_Vertrieb",
    "IT_Admin",
    "SW_Dev_Mobile/UI_Design",
    "Infra_Server_Admin",
    "DB_Dev_Admin",
    "IT_Consultant",
    "Infra_Network_Admin",
    "Data_Engr",
    "SW_Dev_Web",
    "SW_Dev_Web_Frontend",
    "IT_Consultant_Operations",
]


load_bigrams = SaveLoad.load("App-Model/bigram_skills_title")


def text_processing(text):
    """Normalize, tokenize, stem the original text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    cleaned: list of strings. List containing normalized and stemmed word tokens with bigrams
    """

    try:
        text = re.sub(r"(\d)", " ", text.lower())
        text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
        tokens = word_tokenize(text)
        tokens_cleaned = [
            word for word in tokens if word not in stopwords_all and len(word) > 1
        ]
        bigrams_tokens = load_bigrams[tokens_cleaned]
        tokens = list({token for token in bigrams_tokens if token in all_skills})
    except IndexError:
        pass

    return tokens


def percentage(data):
    total = np.sum(data)
    perc_arr = np.array([(x / total) * 100 for x in data])
    return perc_arr


def predict_bereich(text, model=lda_model, topics=topic_names):
    # dictionary.add_documents([text_processing(text)])
    # global clean
    clean = text_processing(text)
    bow = dictionary.doc2bow(clean)
    # print(text_processing(text))
    # model.update([bow])
    # get the topic contributions for the document chosen at random above
    topic_dist = model.get_document_topics(bow=bow)
    doc_distribution = np.array([topic[1] for topic in topic_dist])

    labels_array_percent = percentage(doc_distribution)
    # print(labels_array_percent)
    labels_array = labels_array_percent.argsort()[-3:][::-1]
    # print(topic_dist)

    index = doc_distribution.argmax()
    topic1 = topics[labels_array[0]]
    topic2 = topics[labels_array[1]]
    topic3 = topics[labels_array[2]]

    # print(labels_array[0], labels_array[1])
    # result = f'The project seems to be => {topic1} but could also be => {topic2}'
    # return result, topic1,topic2
    return topic1, topic2, topic3


def predict_and_recommend(text_data):
    bereich = predict_bereich(text_data)
    rec_projects1 = projects[projects["category3"] == bereich]
    rec_projects2 = projects[projects["category2"] == bereich[:2]]
    combined_recommendations = pd.concat(
        [rec_projects1, rec_projects2], ignore_index=True
    )
    combined_recommendations.drop_duplicates(subset="title", inplace=True)

    return bereich, combined_recommendations


class PrefixMiddleware(object):
    def __init__(self, app, prefix=""):
        self.app = app
        self.prefix = prefix

    def __call__(self, environ, start_response):

        if environ["PATH_INFO"].startswith(self.prefix):
            environ["PATH_INFO"] = environ["PATH_INFO"][len(self.prefix) :]
            environ["SCRIPT_NAME"] = self.prefix
            return self.app(environ, start_response)
        else:
            start_response("404", [("Content-Type", "text/plain")])
            return ["This url does not belong to the app.".encode()]


def prefix_route(route_function, prefix="", mask="{0}{1}"):
    """
    Defines a new route function with a prefix.
    The mask argument is a `format string` formatted with, in that order:
      prefix, route
  """

    def newroute(route, *args, **kwargs):
        """New function to prefix the route"""
        return route_function(mask.format(prefix, route), *args, **kwargs)

    return newroute
