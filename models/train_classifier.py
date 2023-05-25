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
    """
    Load data from a SQLite database and return the feature matrix X, the target matrix Y,
    and the category names.

    Args:
    database_filepath (str): The filepath of the SQLite database.

    Returns:
    X (pandas.DataFrame): The feature matrix.
    Y (pandas.DataFrame): The target matrix.
    category_names (list): A list of category names.
    """
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', con=conn)

    X = df['message']
    Y = df.loc[:, 'related':'direct_report']

    category_names = Y.columns

    conn.close()

    return X, Y, category_names

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    """
    Tokenize text by removing URLs, converting to lowercase, stripping whitespace, and lemmatizing.

    Args:
    text (str): The text to be tokenized.

    Returns:
    clean_tokens (list): A list of cleaned and lemmatized tokens.
    """
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

def preprocess_text(text):
    """
    Preprocess text by removing URLs, punctuation, converting to lowercase, tokenizing, lemmatizing,
    removing non-nouns and non-verbs, and removing stopwords.

    Args:
    text (str): The text to be preprocessed.

    Returns:
    clean_tokens (list): A list of cleaned and preprocessed tokens.
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

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



def build_model(X_preproc):
    """
    Build a machine learning pipeline for multi-output classification of text data.

    Args:
    X_preproc (array-like): A preprocessed array of text data.

    Returns:
    A scikit-learn pipeline object that uses preprocessed text data to pre-traine a Word2Vec model, then uses a combination of 
    CountVectorizer, TfidfTransformer, and Word2Vec transformers, and trains a multi-output classifier using a Random Forest algorithm.
    """
    w2v_model = Word2Vec(X_preproc, window=5, min_count=5, workers=4)
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
                ])),
            ('w2v_transformer', Pipeline([
                ('word2vec', W2vVectorizer(w2v_model)),
                ('pca', PCA(n_components=100))
                ]))
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline

def grid_search(pipeline, X_train, Y_train):
    """
    Perform a grid search to find the best hyperparameters for the pipeline.

    Args:
    pipeline (sklearn Pipeline): The pipeline to perform the grid search on.
    X_train (pandas Series): The training data.
    Y_train (pandas DataFrame): The training labels.

    Returns:
    best_pipeline (sklearn Pipeline): The pipeline with the best hyperparameters.
    """
    # Define the parameter grid
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
    }

    # Create a GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    # Fit the GridSearchCV object to the data
    cv.fit(X_train, Y_train)

    # Get the best parameters
    best_params = cv.best_params_

    # Print the best parameters and best score
    print("Best Parameters:", cv.best_params_)
    print("Best Score:", cv.best_score_)

    best_pipeline = cv.best_estimator_

    return best_pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of a machine learning model on a test set.

    Args:
    model: A trained machine learning model.
    X_test: A numpy array of shape (n_samples, n_features) containing the test features.
    Y_test: A pandas dataframe containing the test labels.
    category_names: A list of strings containing the names of the categories.

    Returns:
    report: A list of classification reports for each category.
    avg_f1_score: The average F1 score across all categories.
    """
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
    print(avg_f1_score)
    return report, avg_f1_score


def save_model(model, model_filepath):
    """
    Save a trained machine learning model to a file.

    Args:
    model: A trained machine learning model.
    model_filepath: A string containing the file path to save the model to.

    Returns:
    None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """
    Train and evaluate a machine learning model for classifying disaster messages.

    Args:
    None

    Returns:
    None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        X_preproc = [preprocess_text(text) for text in X]
        
        
        print('Building model...')
        model = build_model(X_preproc)
        # model = grid_search(model, X_train, Y_train)
        
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