import json
import plotly
import plotly.express as px
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from nltk import pos_tag
from nltk.corpus import stopwords
import string

from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Tokenize text by converting to lowercase, stripping whitespace, and lemmatizing.

    Args:
    text (str): The text to be tokenized.

    Returns:
    clean_tokens (list): A list of cleaned and lemmatized tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def preprocess_text(text):
    """
    Preprocess text by removing punctuation, converting to lowercase, tokenizing, lemmatizing,
    removing non-nouns and non-verbs, and removing stopwords.

    Args:
    text (str): The text to be preprocessed.

    Returns:
    clean_tokens (list): A list of cleaned and preprocessed tokens.
    """
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    tokens = [word for word, tag in tags if tag.startswith('N') | tag.startswith('V')]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    stop_words = set(stopwords.words('english'))
    clean_tokens = [word for word in clean_tokens if word not in stop_words]
    return clean_tokens


class W2vVectorizer(BaseEstimator, TransformerMixin):
    """
    A custom vectorizer that converts text into word vectors using a pre-trained Word2Vec model.

    Args:
    w2v_model (gensim.models.word2vec.Word2Vec): A pre-trained Word2Vec model.

    Methods:
    fit(X, y): Fit the vectorizer to the data.
    transform(X): Transform the data into word vectors.

    Returns:
    word_vectors (numpy.ndarray): An array of word vectors.
    """
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
    

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Define the colors for each category
    colors = ['#1f77b4', '#8c564b', '#d62728']

    # extract data needed for visuals
    Y = df.loc[:, 'related':'direct_report']
    categories = Y.columns 
    values = Y.mean()*100

    # create color scale
    color_scale = np.random.randint(0, 256, size=(len(categories), 3))
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(color=colors)
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
        },
        {
            'data': [
                Bar(
                    x=categories,
                    y=values,
                    marker=dict(color=['rgb({},{},{})'.format(*color) for color in color_scale])
                )
            ],

            'layout': {
                'title': 'Distribution of Assinged Categories',
                'yaxis': {
                    'title': "% of total"
                },
                'xaxis': {
                    'title': {
                    'text': "Categories",
                    'standoff': -200,
                    'autorange': True
                }
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
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()