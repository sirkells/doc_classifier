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
#print(type(stopwords))
# Stem word tokens and remove stop words
stemmer_eng = SnowballStemmer("english", ignore_stopwords=True)
stemmer_germ = SnowballStemmer("german", ignore_stopwords=True) 


# Get Dictionary of the Trained Data
with open('models/dictionary', 'rb') as data:
    dictionary = pickle.load(data)

# Get Projects data
with open('models/APP_DATA.sav', 'rb') as data:
    projects = pickle.load(data)

#Get corpus 
with open('models/corpus', 'rb') as data:
    corpus = pickle.load(data)

# Load a potentially pretrained model from disk.
#lda_model = LdaModel.load('models/model_tm', mmap='r')

# later on, load trained model from file
lda_model =  models.LdaModel.load('models/lda.model')
all_topic_distr_list = lda_model[corpus]

target_labels = ['Data-Engr-Big Data','Data-Sci-BI','Dev-Devops','Dev-Web-Backend','Dev-Web-Frontend','Dev-Web-Fullstack','ERP-SAP','IT-Admin-Others','IT-Mgmt-Consulting','IT-Mgmt-Projectleiter','IT-Technical-Dev','IT/Elektrotechn','Infr-Admin-Database','Infr-Admin-Linux','Infr-Admin-Microsoft','Infr-Admin-Net','Infr-Database-Admin','SW-Dev-Mobile','SW-Dev-Others']


# z = ['nachverfolgung', 'implement', 'organisation', 'lasse', 'ten',  'durchf', 'hinsichtlich', 'suse', 'bersetzung', 'berpr', 'ts', 'auszuf', 'win', 'fung', 'grundlegende', 'sung',  'aufwand',  'vereinbarung', 'liefereinheiten', 'nachbesserung', 'hrende', 'ergebisse', 'aktivit', 'implementierung', 'qualit', 'hrungsverantwortung', 'organisations']
# w = ['august', 'flexibel', 'tzung', 'ngerungsoption', 'ige', 'stammkunden', 'absprache', 'hierf', 'langfristig', 'hauptaufgabe', 'regelm', 'fachabteilung']
# x = ['datenschutzbestimmungen',  'nintex', 'rechte',  'arbeitnehmer', 'entnehmen', 'workflows', 'rechtsform', 'forms', 'verf', 'robotersteuerung',  'workflow', 'gung', 'elisabeth', 'umfang', 'sicherstellen', 'konzipieren', 'aracom', 'personalreferentin', 'bedienbarkeit', 'realisieren', 'besch', 'funktionen', 'insb', 'ftigen', 'ndige', 'personenbezogenen', 'individual', 'oberfl', 'ssig', 'berechtigungsstruktur', 'hemmerle', 'verarbeitung', 'einbezug',  'che', 'benutzerfreundlichen', 'sozialversicherungspflichtige']
# y = ['dealership', 'professsional', 'create', 'motivation', 'invision', 'higkeit', 'ssigkeit', 'gesuchte', 'sale', 'selbstst', 'point', 'teamf', 'ideal', 'figma', 'zuverl', 'desi', 'verhandlungssicher', 'ndiges', 'sprachanforderungen', 'erweiterte', 'sketch', 'first', 'tigkeiten']
# j = ['improve', 'period', 'financial', 'full', 'hearing', 'current', 'live', 'initial', 'term',  'long', 'scratch', 'look',   'minimum', 'build', 'get', 'reviews', 'profile', 'codebase', 'someone',  'help', 'forward', 'soon', 'features', 'offer', 'week', 'available', 'however', 'days', 'legacy',  'instance', 'position', 'superceding']
# update_stopwords = [ 'unterhalten', 'mitarbeitern', 'plattform', 'load', 'verteilten', 'sprintwechsel', 'streamingdienstes', 'gew',  'wochen', 'plattformen', 'owner', 'orientiert', 'betreuen', 'nschenswert', 'rhythmus', 'infrastructure', 'erarbeiten', 'prometheus', 'arbeitsweise', 'tzen', 'aufbau', 'bevorzugt', 'unterst', 'lifecycle', 'mittwoch', 'arbeit', 'donnerstag','freitag', 'montag', 'dienstag', 'reibungslosen', 'nnen', 'spiel', 'gro', 'enth', 'nutzer', 'konzernen', 'umgebungen', 'dot', 'thrivenow', 'startups', 'kollegen', 'passt',  'gbaren', 'nachricht', 'refinement', 'thrive', 'jeweils',  'abdecken', 'termine', 'frameworks', 'logik', 'sprintdauer', 'balancing', 'anbietern', 'festangestellten', 'liebe', 'teamgeist', 'anbietet',  'internen', 'wochentagen', 'kreativit', 'metric', 'aufgabenstellung', 'unterwegs', 'nchen', 'erweitert', 'everything', 'reichen', 'bereit', 'uptimes', 'parallelen', 'haupts', 'projektbezogener', 'daily',  'session', 'testl', 'credo',  'sungen',  'basierend', 'loggregator', 'nscht', 'bestehend', 'hochverf', 'thema', 'chlich', 'hrige', 'dominic', 'performance', 'freelancern', 'mehrj', 'fluentd', 'min']
# #stopwords = stopwords + update_stopwords + w + y + z + x + j

# ab = ['entwick', 'automat', 'scrum']
av = {
    'entwick': 'entwicklung',
    'automat': 'automate',
    'scrum': 'scrum',
    'container': 'containerization',
    'admin': 'administration',
    'analys': 'analyst',
    'softwar': 'software',
    'app': 'application',
    'developer': 'entwicklung',
    'programming': 'entwicklung',
    'mobil': 'mobil',
    'cisco': 'cisco'
}
# def tokenize(text):
#     """Normalize, tokenize and stem text string
    
#     Args:
#     text: string. String containing message for processing
       
#     Returns:
#     stemmed: list of strings. List containing normalized and stemmed word tokens
#     """

#     try:
#         # Convert text to lowercase and remove punctuation
#         text = re.sub("[^a-zA-Z ]", " ", text.lower()) #remove non alphbetic text
#         # Tokenize words
#         tokens = word_tokenize(text)
#         #stemmed = [stemmer_germ.stem(word) for word in tokens if word not in stopwords]
#         #stemmed = [stemmer_eng.stem(word) for word in stemmed if len(word) > 1]
#         stemmed = [word for word in tokens if word not in stopwords and len(word) > 1]
        
#         for index, word in enumerate(stemmed):
#             for key, value in av.items():
#                 if key in word:
#                     stemmed[index] = value
        
#         #duplicates = list(set(stemmed))
#     except IndexError:
#         pass

#     return stemmed

wnl = WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 
    
def lemmatize_sent(text): 
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(word_tokenize(text))]

lemmatize_sent('He is walking to school')

import string
from string import punctuation
def tokenize(text):
    """Normalize, tokenize and stem text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    stemmed: list of strings. List containing normalized and stemmed word tokens
    """
    # Stem word tokens and remove stop words
    stemmer_eng = SnowballStemmer("english", ignore_stopwords=True)
    stemmer_germ = SnowballStemmer("german", ignore_stopwords=True) 
    try:
        # Convert text to lowercase and remove punctuation
        #text = re.sub("[^a-zA-Z ]", " ", text.lower()) #remove non alphbetic text
        text = re.sub(r'(\d)',' ',text.lower())
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        # Tokenize words
        #tokens = word_tokenize(text)
        #print(tokens)
        
        tokens = word_tokenize(text)
        stemmed = [stemmer_germ.stem(word) for word in tokens if word not in stopwords_sl]
        stemmed = [word for word in stemmed if len(word) > 1]
        #stemmed = [word for word in tokens if word not in stopwords and len(word) > 1]
        
        
        #stemmed = [word for word in stemmed if word not in stopwords and len(word) > 2]
    except IndexError:
        pass

    return stemmed
def tokenize_ul(text):
    """Normalize, tokenize and stem text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    stemmed: list of strings. List containing normalized and stemmed word tokens
    """
    # Stem word tokens and remove stop words
    #stemmer_eng = SnowballStemmer("english", ignore_stopwords=True)
    stemmer_germ = SnowballStemmer("german", ignore_stopwords=True) 
    try:
        # Convert text to lowercase and remove punctuation
        text = re.sub("[^a-zA-Z ]", " ", text.lower()) #remove non alphbetic text
        #text = re.sub(r'\b\d+(?:\.\d+)?\s+', ' ', text.lower())
        #text = re.sub(r'(\d)',' ',text.lower())
        #text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        #text = re.sub(r'[^\w\s]',' ',text)
        # Tokenize words
        #tokens = word_tokenize(text)
        #print(tokens)
        
        tokens = word_tokenize(text)
        stemmed = [stemmer_germ.stem(word) for word in tokens if word not in stopwords_ul]
        stemmed = [word for word in stemmed if len(word) > 1]
        #stemmed = [word for word in stemmed if word not in stopwords_ul and len(word) > 1]
        
        
        #stemmed = [word for word in stemmed if word not in stopwords and len(word) > 2]
    except IndexError:
        pass

    return stemmed
def clean_lower_tokenize(text):
    """
    Function to clean, lower and tokenize texts
    Returns a list of cleaned and tokenized text
    """
    #text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)  #remove websites texts like email, https, www
    text = re.sub("[^a-zA-Z ]", "", text) #remove non alphbetic text
    text = text.lower() # lower case the text
    text = nltk.word_tokenize(text)
    return text

def remove_stop_words(text):
    """
    Function that removes all stopwords from text
    """
    return [word for word in text if word not in stopwords_ul]

def stem_eng_german_words(text):
    """
    Function to stem words
    """
    try:
        text = [stemmer_germ.stem(word) for word in text]
        #text = [stemmer_eng.stem(word) for word in text]
        text = [word for word in text if len(word) > 1] 
    except IndexError:
        pass
    return text

def all_processing(text):
    """
    This function applies all the functions above into one
    """
    return stem_eng_german_words(remove_stop_words(clean_lower_tokenize(text)))



# load model
model = joblib.load("models/new_model")


def get_category(text, model, labels):
    predicted = model.predict([text])[0]
    results = dict(zip(labels.columns, predicted))
    return results
def get_category_prob(text, model, labels):
    predicted_prob_distr = model.predict_proba([text])
    results = [val[0][1] for val in predicted_prob_distr]
    predicted_index = np.argmax(results)
    prediction_percentage = max(results)
    return labels[predicted_index], prediction_percentage
    
def percentage(data):
    total = np.sum(data)
    perc_arr = np.array([(x/total)*100 for x in data])
    return perc_arr
# index webpage displays cool visuals and receives user input text for model

def predict_bereich(text, lda_model):
    clean = tokenize_ul(text)
    text_bow = dictionary.doc2bow(clean)
    topic_distr_array = np.array([topic[1] for topic in lda_model.get_document_topics(bow=text_bow)])
    labels_array_percent = percentage(topic_distr_array)
    print(labels_array_percent)
    labels_array =labels_array_percent.argsort()[-2:][::-1]
    print(topic_distr_array)
    return labels_array, labels_array_percent, clean
def js_similarity_score(doc_distr_query, corpus_distr):
    """
    This function finds the similarity score of a given doc accross all docs in the corpus
    It takes two parameters: doc_distr_query and corpus_distr
    (1) doc_distr_query is the input document query which is an LDA topic distr: list of floats (series)
            [1.9573441e-04,...., 2.7876711e-01]
    (2) corpus_dist is the target corpus containing the LDA topic distr of all documents in the corpus: lists of lists of floats (vector)
            [[1.9573441e-04, 2.7876711e-01, 1.9573441e-04]....[1.9573441e-04,...., 2.7876711e-01]]
    It returns an array containing the similarity score of each document in the corpus_dist to the input doc_distr_query
    The output looks like this: [0.3445, 0.35353, 0.5445,.....]
    
    """
    input_doc = doc_distr_query[None,:].T #transpose input
    corpus_doc = corpus_distr.T # transpose corpus
    m = 0.5*(input_doc + corpus_doc)
    sim_score = np.sqrt(0.5*(entropy(input_doc,m) + entropy(corpus_doc,m)))
    return sim_score
def find_top_similar_docs(doc_distr_query, corpus_distr,n=10):
    """
    This function returns the index lists of the top n most similar documents using the js_similarity_score
    n can be changed to any amount desired, default is 10
    """
    sim_score = js_similarity_score(doc_distr_query, corpus_distr)
    similar_docs_index_array = sim_score.argsort()[:n] #argsort sorts from lower to higher
    return similar_docs_index_array

def recommend(text):
    clean = tokenize_ul(text)
    text_bow = dictionary.doc2bow(clean)
    new_doc_distribution = np.array([tup[1] for tup in lda_model.get_document_topics(bow=text_bow)])
    corpus_topic_dist= np.array([[topic[1] for topic in docs] for docs in all_topic_distr_list])
    similar_docs_index = find_top_similar_docs(new_doc_distribution, corpus_topic_dist, 10)
    top_sim_doc = projects[projects.index.isin(similar_docs_index)]
    PROJECT_DICT = top_sim_doc.to_dict() 
    return PROJECT_DICT


@app.route("/")
@app.route('/home')
def home():
    # save user input in query
    #print(type(stopwords))
    query = request.args.get('query', '')
    labels, labels_perc, new_data = predict_bereich(query, lda_model)
    recommended_projects = recommend(query)
    #print(recommended_projects)
    # use model to predict classification for query
    output = get_category_prob(query, model, target_labels)
    #print(labels)
    index_highest = labels_perc.argmax()
    group = ['ERP/SAP','SW_Dev/Web','SW_Dev/Arch','SW_Dev/DevOps','Sys_Admin/Support', 'SW_Dev/Mobile/support','Data/Ops','IT_Process_Mgr/Consultant', 'MS_DEV/Admin','Business_Analyst/Consulting']
    #dictOfWords = { i : other_labels[i] for i in range(0, len(other_labels) ) }
    # This will render the go.html Please see that file. 
    if labels_perc[index_highest] < 20:
        query = 'Please enter a valid text'
        return render_template(
        'home.html',
        query=query,
        output={},
        labels={},
        group={}
    )
    else:
        
        return render_template(
            'home.html',
            query=query,
            output=output,
            labels=labels,
            group=group,
            topic_distr=labels_perc,
            recommended_projects=recommended_projects,
            new_data=new_data
        )

@app.route("/about/")
def about():
    return render_template("about.html")

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
