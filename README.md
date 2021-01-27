# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Instruction](#instructions)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Beyond libraries in the Anaconda distribution of Python, running the python files require language detection  (googletrans, langdetect) and spell checking (spellchecker) packages.

## Instructions:

1. Clone the repository
2. Because the repo contains a database as well as a model object, you can skip to step 4 if you prefer. But to run the cleaning script locally to yield a database, in the data directory, run the following command, adding the filepaths of the messages and categories datasets (also in the data directory) as the first and second argument respectively, as well as the filepath of the database to save the cleaned data to as the third argument.
    `python process_data.py`
3. In your local environment, in the models directory,  run the following command, adding the filepath of the disaster messages database as the first argument and the filepath of the pickle file to save the model to as the second argument. 
    `python train_classifier.py`
4. In your local environment, run the following command in the app directory to run the web app.
    `python run.py`
4. Go to http://0.0.0.0:3001/

## Project Motivation<a name="motivation"></a>

For this project, I used Appen's (formerly Figure Eight's) data set containing messages sent during disasters to build a model for an API that classifies likely disasters. Beyond the pro social nature of the model, my motivation was to:

1. Build an ETL pipeline
2. That provided data to a machine learning pipeline
3. That could be improved using an array of NLP techniques

## File Descriptions <a name="files"></a>

1. The data folder contains the training data, a SQLite database, and a python ETL file (process_data.py)
2. The models folder contains a pickled model object, a class for recognizing titles (title_transformer.py), and a machine learning pipeline to fit and evaluate a multioutput classifer (train_classifier.py)
3. The app folder contains a python file to run the web application (run.py), as well as html files used in the web application.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The training data contained in this repository is owned by Appen.  Otherwise, use the code here as you would like! 

