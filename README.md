### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#file)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation <a name="installation"></a>
This code was written using Python3 and listed below are the required packages:

- json
- plotly
- pandas
- flask
- sklearn
- sqlalchemy
- nltk



## Project Motivation<a name="motivation"></a>


A Disaster Response Pipeline that categorizes emergency messages based on the needs communicated by the sender.

This project contains real messages that were sent during disaster events. A machine learning pipeline was created to categorize these events so that the messages are sent to the appropriate disaster relief agency.  This project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 


## File Descriptions<a name="file"></a>

There are three components that are part of this project.

1. ETL Pipeline

data/process_data.py 

This python script:
- Loads the messages - disaster_messages.csv and categories datasets - disaster_categories.csv 
- Merges the two datasets together
- Cleans the data
- Stores this data in a SQLite database - DisasterResponse.db

2. ML Pipeline
    models/train_classifier.py 

This python script:
- Loads data from the SQLite database (DisasterResponse.db) created in the etl pipeline
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App

Created data visualizations using Plotly in the web app to display 

## Instructions <a name="instructions"></a>

Run the following commands in the project's root directory to set up your database and model.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    - `python run.py`

3. Go to http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The dataset used in this analysis was created by [Figure Eight](https://appen.com/) and other components of this project were made available from [Udacity](udacity.com).

