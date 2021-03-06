import sys

# import libraries

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load and merge csv files into a dataframe 
    
    Args:
        messages_filepath: Filepath to the messages file 
        
        categories_filepath: Filepath to the categories file  
        
    Returns: 
        A merged dataset
    """ 
    
    #Load Messages file into a dataframe
    messages = pd.read_csv(messages_filepath)
    
    #Load Categories file into a dataframe
    categories = pd.read_csv(categories_filepath)
    
    #Merge files
    df = messages.merge(categories, 'inner', 'id')
    return df


def clean_data(df):
    """Cleans data in the DataFrame
    
    Args:
        df: dataframe with merged messages and categories data
   
    Returns:
        df: cleaned  Dataframe
    """ 
    
    #Split categories into separate category columns.
    categories = df.categories.str.split(';', expand=True)  
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0]) 
    
    # rename the columns of categories
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #Replace categories column in df with new category columns.
    df.drop(columns='categories', axis=1, inplace=True)
    df = pd.concat([df,categories],sort=False,axis=1)
    
    #Remove duplicates
    df.drop_duplicates(inplace=True)
    
    #Remove rows that contain 2 in the Related Category 
    df = df[df['related'] != 2]
    return df


def save_data(df, database_filename):
    """Saves Data into Database
    
    Args:
        df: cleaned dataframe
        database_filename: database file name

    """ 
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Disasters', engine, if_exists='replace',index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()