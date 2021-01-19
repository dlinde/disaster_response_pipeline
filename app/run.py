import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


import numpy as np
#import string
import re
import joblib


from googletrans import Translator
from langdetect import detect
from spellchecker import SpellChecker
from title_transformer import TitleCount


def tokenize(text):
    '''
    Parameters:
            text (string): A document to tokenize

    Returns:
            clean_tokens (dataframe): A list of tokens derived from text and cleaned
    '''

    # initiate stop words
    stop_words = set(stopwords.words('english'))

    # remove urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # intitiate translator
    translator = Translator()

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

     # find words that may be misspelled
    spell = SpellChecker(distance=1)

    # remove white space from ends of text
    text=text.strip()
    # remove urls
    if 'http' in text:
        detected_urls = re.findall(url_regex, text)
        for url in detected_urls:
            text = text.replace(url, "")
    # check if text is english, and if it's not translate it
    try:

        if detect(text)!='en':
            text = translator.translate(text).text
    # if detect yields an error, return empty string
    except:
        return ''

    # tokenize text
    tokens = word_tokenize(text)

    # remove numbers - first character removes e.g. 5,000
    tokens = [item for item in tokens if not item[0].isdigit()]

    # tag part of speech
    tokens = pos_tag([i for i in tokens if i])

    # iterate through each token
    clean_tokens = []

    for tok, pos in tokens:

        # normalize case of token and remove leading/trailing white space
        tok = tok.strip().lower()

        # if a single character or apostrophe and one character, remove
        if len(tok)<3:
            if "'" in tok or len(tok)==1:
                continue

        # remove stop words
        if tok not in stop_words:
             # lemmatize
            if 'NN' in pos:
                tok = lemmatizer.lemmatize(tok, pos='n')
            elif 'V' in pos:
                tok = lemmatizer.lemmatize(tok, pos='v')
            elif 'JJ' in pos:
                tok = lemmatizer.lemmatize(tok, pos='a')
            else:
                pass

            # check spelling, correct to most likely 1 character alternative
            if len(spell.unknown([tok]))>0 and ("'" not in tok):
                tok = spell.correction(tok)

            # append to clean token list
            clean_tokens.append(tok)


    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('labeled_messages', engine)
Y = df.drop(['id','message','original','genre','child_alone'],axis=1).dropna(axis=1,how='all').copy()

# load model
model = joblib.load("../models/logr_multi.pkl")


# index webpage displays cool visuals and receives user input text for model
app = Flask(__name__)
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    '''

    Returns:
            render_template (): contains data required to build visualizations
    '''
    # save user input in query
    query = request.args.get('query', '')
    # convert the predictions to a dataframe then predict true for preds >.35, false otherwise, yielding a series
    classification_labels = pd.DataFrame([model.predict_proba([query])],columns=Y.columns).T[0].map(
        lambda x: 1 if x[0][1]>.35 else 0)
    # sort ascending so positive preds appear first
    classification_labels.sort_values(ascending=False,inplace=True)
    # convert to dictionary and add child alone category, which had no cases in sample
    classification_results = dict(zip(classification_labels.index, classification_labels.values))
    # no instances of child_alone in dataset so prediction always 0
    classification_results['child_alone']=0

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
