import sys
import pandas as pd
import numpy as np
#import string
import re
import joblib

from sqlalchemy import create_engine
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
#from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer#,CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from googletrans import Translator
from langdetect import detect
from spellchecker import SpellChecker
from title_transformer import TitleCount


def load_data(database_filepath,table_name='labeled_messages')):
    '''
    Parameters:
            database_filepath (string): A filepath to a sqlite database in local environment
            table_name (string): A name for tweetdf in db

    Returns:
            X (dataframe): Training data containing tweets
            y (dataframe): Test data containg labels
            y.columns (list): disaster categories to predict
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    tweetdf = pd.read_sql('select * from '+table_name,con=engine)
    # all columns besides related are binary; the 2 value is ambiguous
    tweetdf = tweetdf[df['related']!=2]
    X = tweetdf['message'].values
    # child alone has no positives
    y = tweetdf.drop(['id','message','original','genre','child_alone'],axis=1).dropna(axis=1,how='all').copy()
    return X, y, y.columns


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


def build_model():
    '''

    Returns:
            pipeline (pipeline object): A multioutput classifier to predict disaster categories
    '''

    pipeline= Pipeline([
            ('features',FeatureUnion([
                ("vect", TfidfVectorizer(tokenizer=tokenize,min_df=.005,ngram_range=(1,2))),
                ('title',TitleCount())])),
            ('clf',MultiOutputClassifier(LogisticRegression(solver='liblinear',max_iter=1000,
                                                            C=np.logspace(-4, 4, 4)[2],penalty='l2')))
        ])

    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    '''
    Parameters:
            model (pipeline object): A multioutput classifier fit to predict disaster categories
            X_test (dataframe): holdout data containting documents
            y_test (dataframe): holdout data contains labels for X_test
            category_names (list): disaster response categories
    '''
    y_pred_proba = model.predict_proba(X_test)
    i=0
    for col in category_names:
        print(col)
        if i ==0:
            print(classification_report(y_test[col],pd.DataFrame(
            y_pred_proba[i][:,1:],columns=['proba']).proba.map(lambda x: 1 if x>=.55 else 0)))
        else:
            print(classification_report(y_test[col],pd.DataFrame(
                y_pred_proba[i][:,1:],columns=['proba']).proba.map(lambda x: 1 if x>=.35 else 0)))
        i+=1
    return


def save_model(model, model_filepath):
    """
    Parameters:
        model (pipeline object): A multioutput classifier fit to predict disaster categories
        model_filepath (string): file path to store pickled model locally
    """
    joblib.dump(model, model_filepath)
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
