# Disaster Response Pipeline Project

### Introduction
In this project, a machine learning pipeline has been built to analyze real-time disaster messages from Appen (formerly Figure 8). The objective of the project is to classify these messages and direct them to the appropriate disaster relief agency. 

The ETL pipeline, process_data.py, loads the messages and categories datasets, merges them, cleans the data and stores it in a SQLite database.

The ML pipeline, train_classifier.py, loads data from the SQLite database, splits dataset into training and test sets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set and exports the final model as a pickle file.

The pipeline is used for multi-output classification of text data. It takes in preprocessed text data and builds a machine learning pipeline. The pipeline first pre-trains a Word2Vec model. Then, it uses a combination of CountVectorizer, TfidfTransformer, and Word2Vec transformers to transform the text data. Finally, it trains a multi-output classifier using a Random Forest algorithm.
The pipeline is built using scikit-learn. The pipeline is returned as a scikit-learn pipeline object, which can be used to fit the model on training data and make predictions on new data.

The web app, run.py, runs in the terminal. When a user inputs a message into the app, the app returns classification results for all 36 categories.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves[^1]
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

[^1] The grid search in the main function is disabled by commenting it out, as it requires a significant amount of time to execute. Currently, the pipeline is running with its default parameters. 
