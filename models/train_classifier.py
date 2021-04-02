import sys

import nltk
#import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sqlalchemy import create_engine
import pandas as pd
import re

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


import pickle

nltk.download(['punkt', 'wordnet','stopwords'])

def load_data(database_filepath):
    """Load Database
    
    Args:
        database_filepath: Path to Database file
        
    Returns:
        X: Messages Dataframe
        y: Category Dataframe
        category_names: List of the category names
    """ 
    
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('Disasters', engine)

    # create X dataframe with Message values
    X = df['message'].values
    #create y dataframe with category values
    y = df.iloc[:,4:]
    #get category names
    category_names = y.columns
    
    return X,y,category_names

def tokenize(text):
    """Tokenize Text
    
    Args:
        text: text 
        
    Returns:
        cleaned_tokens: tokens

    """ 
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    stop_words = stopwords.words("english")
    stemmer = PorterStemmer()
    
    #tokenize
    words = word_tokenize(text)
    
    #stemmed
    stemmed = [stemmer.stem(word) for word in words if word not in stop_words]
    
    #lemmatizing
    cleaned_tokens = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
   
    return cleaned_tokens


def build_model():
    
    """ Build model using a Pipeline and GridSearchCV
    
    Returns:
        cv: GridSearchCV 
    """ 
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        
    ])

    parameters =  {'clf__estimator__n_estimators': [10,20,50] ,
              
              'clf__estimator__min_samples_leaf':[2, 5, 10]
              }

    cv =  GridSearchCV(pipeline, parameters)
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """ Evaluate Model
    
    Args:
        model: Model
        X_test: Messages Test data
        Y_test: Category Test data 
        category_names: Category Names
    """ 

    y_pred = model.predict(X_test)
    
    for i in range(y_test.shape[1]):
        report = classification_report(y_test.iloc[:,i].values, y_pred[:,i])
        print("{}: \n".format(category_names[i]))
        print(report)


def save_model(model, model_filepath):

    """ Save Model in Pickle formate
    Args:
        model: model
        model_filepath: Model Filepath
    """ 
    pickle.dump(model, open(model_filepath, 'wb'))
  
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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