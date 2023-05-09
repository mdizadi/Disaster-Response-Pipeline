import sys
import pickle

import sqlite3

import re
import numpy as np
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'vader_lexicon', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from nltk import pos_tag
from nltk.corpus import stopwords



from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA

import string

from gensim.models import Word2Vec

import matplotlib.pyplot as plt



def load_data(database_filepath):
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', con=conn)

    X = df['message']
    Y = df.loc[:, 'related':'direct_report']

    category_names = Y.columns

    conn.close()

    return X, Y, category_names


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Define preprocessing functions
def preprocess_text(text):
    # Remove punctuation and lowercase text
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    # Tokenize the text
    #text = tokenize(text)
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    tokens = [word for word, tag in tags if tag.startswith('N') | tag.startswith('V')]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    clean_tokens = [word for word in clean_tokens if word not in stop_words]
    return clean_tokens


class W2vVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, w2v_model):
        self.w2v_model = w2v_model

    def fit(self, X, y):
        return self

    def transform(self, X):
        word_vectors = np.zeros((len(X), self.w2v_model.vector_size))
        for i, sentence in enumerate(X):
            for word in preprocess_text(sentence):
                if word in self.w2v_model.wv:
                    word_vectors[i] += self.w2v_model.wv[word]
        return word_vectors



def build_model(X_preproc):
    w2v_model = Word2Vec(X_preproc, window=5, min_count=5, workers=4)
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
                ])),
            # Define Word2Vec transformer
            ('w2v_transformer', Pipeline([
                ('word2vec', W2vVectorizer(w2v_model)),
                ('pca', PCA(n_components=50))
                ]))
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    i = 0
    report = []
    for column in category_names:
        colreport = classification_report(np.array(Y_test[column]),Y_pred[:,i])
        print(column + '\n' , colreport)
        report.append(colreport)
        i += 1
    
    f1_score = 0
    for i in range(len(report)):
        temp = float(report[i].split()[27])
        f1_score += temp
    
    avg_f1_score = f1_score / len(report)

    return report, avg_f1_score


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        X_preproc = [preprocess_text(text) for text in X]
        
        
        print('Building model...')
        model = build_model(X_preproc)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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