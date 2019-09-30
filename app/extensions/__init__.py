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

from pymongo import MongoClient


def connect():
    connection = MongoClient("10.10.250.10", 27017)
    handle = connection["projectfinder"]
    return handle


db = connect()


def load_data_from_momgodb():
    exclude_data = {"_id": False}
    raw_data = list(db.itproject_region_bereich.find({}, projection=exclude_data))
    dataset = pd.DataFrame(raw_data)
    # select colunms
    dataset = dataset[
        ["title", "description", "bereich", "rec_bereich", "skill_summary"]
    ]
    dataset = dataset[dataset["description"] != ""]
    dataset = pd.concat(
        [
            dataset.drop(["rec_bereich"], axis=1),
            dataset["rec_bereich"].apply(pd.Series),
        ],
        axis=1,
    )
    return dataset


# load target data
pickle_in = open("models/stopwords_all.pickle", "rb")
stopwords_all = pickle.load(pickle_in)


# Get Dictionary of the Trained Data
with open("models/dictionary", "rb") as data:
    dictionary = pickle.load(data)

# Get Projects data
with open("models/df", "rb") as data:
    projects = pickle.load(data)

# projects.drop_duplicates(subset='description', inplace=True)
# Get corpus
with open("models/corpus", "rb") as data:
    corpus = pickle.load(data)

# load flatlist of skills
with open("models/flatlist", "rb") as data:
    all_skills = pickle.load(data)


# later on, load trained model from file
lda_model = models.LdaModel.load("models/converted_model_skills_title_26_pref")
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


load_bigrams = SaveLoad.load("models/bigram_skills_title")


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

def nonIT(score): 
    #subtracts topic 1 score from topic 3. 
    #if score is greater than 2, its an IT text else its nonIT
    diff = score[0] - score[4]
    #check = diff > 2
    return diff > 2

def predict_bereich(text, model=lda_model, topics=topic_names):
    # dictionary.add_documents([text_processing(text)])
    # global clean
    clean = text_processing(text)
    bow = dictionary.doc2bow(clean)
    # print(text_processing(text))
    # model.update([bow])
    # get the topic contributions for the document chosen at random above
    topic_dist = model.get_document_topics(bow=bow)
    prob_dist = [topic[1] for topic in topic_dist]
    doc_distribution = np.array(prob_dist)

    labels_array_percent = percentage(doc_distribution)
    # print(labels_array_percent)
    labels_array = labels_array_percent.argsort()[-3:][::-1]
    # print(topic_dist)

    index = doc_distribution.argmax()
    
    score = sorted([round(count*100,2) for count in prob_dist], reverse=True)

    topic1 = topics[labels_array[0]]
    topic2 = topics[labels_array[1]]
    topic3 = topics[labels_array[2]]
    
    # predicted_prob_distr = model.predict_proba([text])
    # results = [val[0][1] for val in predicted_prob_distr]
    # predicted_index = np.argmax(results)
    # prediction_percentage = max(results)

    # print(labels_array[0], labels_array[1])
    # result = f'The project seems to be => {topic1} but could also be => {topic2}'
    # return result, topic1,topic2
    return topic1, topic2, topic3, score


def predict_and_recommend(text_data):
    bereich1, bereich2, bereich3, probability_percentage = predict_bereich(text_data)
    bereich = (bereich1, bereich2, bereich3)
    # category_all =  bereich1 +" "+ bereich2 + " " + bereich3
    # category1 = bereich1 +" "+ bereich2
    # category2 = bereich1 +" "+ bereich3
    #bs1 = skills[topic_names.index(bereich1)]
    #bs2 = skills[topic_names.index(bereich2)]
    # rec_project = db.itproject_region_bereich.find({"$or":[{"bereich1": bereich1,"bereich2": bereich2, "bereich3": bereich3}, {"bereich1": bereich1,"bereich2": bereich2}]})
    # new = list({project['title'] for project in rec_project})
    rec_project2 = [
        project
        for project in db.itproject_region_bereich.find(
            {
                "$and": [
                    {"bereich1": bereich1},
                    {"bereich2": bereich2},
                    {"bereich3": bereich3},
                ]
            }, {'_id': False,"tech_summary":False, "summary":False}
        )
    ]
    rec_project3 = [
        project
        for project in db.itproject_region_bereich.find(
            {"$and": [{"bereich1": bereich1}, {"bereich2": bereich2}]}, {'_id': False, "tech_summary":False, "summary":False}
        )
    ]
    rec_project4 = [
    project
    for project in db.itproject_region_bereich.find(
        {"bereich1": bereich1}, {'_id': False, "tech_summary":False, "summary":False}
    )
]
    rec_project = (rec_project3[:11] + rec_project2 + rec_project4)
    # data = load_data_from_momgodb()
    # rec_projects1 = data[data["category_all"] == category_all]
    # rec_projects2 = data[data["category1"] == category1]
    # rec_projects3 = data[data["category2"] == category2]
    # rec_projects1 = projects[projects["category3"] == bereich]
    # rec_projects2 = projects[projects["category2"] == bereich[:2]]
    # combined_recommendations = pd.concat(
    # [rec_projects1, rec_projects2, rec_projects3[:5]], ignore_index=True
    # )
    # combined_recommendations.drop_duplicates(subset="title", inplace=True)

    return bereich, rec_project[:20], probability_percentage


def get_category_prob(text, model=lda_model, labels=topic_names):
    predicted_prob_distr = model.predict_proba([text])
    results = [val[0][1] for val in predicted_prob_distr]
    predicted_index = np.argmax(results)
    prediction_percentage = max(results)
    return labels[predicted_index], prediction_percentage