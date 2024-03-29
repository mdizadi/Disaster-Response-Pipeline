import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge message and category data from CSV files.

    Args:
    messages_filepath (str): Filepath for the messages CSV file.
    categories_filepath (str): Filepath for the categories CSV file.

    Returns:
    df (pandas DataFrame): Merged DataFrame containing messages and categories data.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, how = 'inner', on = 'id')
    
    return df


def clean_data(df):
    """
    Clean the merged DataFrame containing messages and categories data.

    Args:
    df (pandas DataFrame): Merged DataFrame containing messages and categories data.

    Returns:
    df (pandas DataFrame): Cleaned DataFrame containing messages and categories data.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories.
    category_colnames = row.apply(lambda x : x[:-2])

    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.replace(column + '-', '')
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    for column in categories.columns:
        b = categories[column].value_counts().index
        if ((1 & 0) in b) & (len(b) == 2):
            pass
        else:
            print('There is a problem with values of the column "{}", and it has {} distinct value(s)!'.format(column, len(b)))

    categories['related'] = categories['related'].apply(lambda x : 1 if x == 2 else x)

    b = categories['child_alone'].value_counts().index
    if not(((1 & 0) in b) & (len(b) == 2)):
        categories = categories.drop(['child_alone'], axis = 1)

    # drop the original categories column from `df    
    df = df.drop('categories', axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)

    # drop duplicates
    df = df.drop_duplicates()
    
    # duplicate messages
    df = df.drop_duplicates(subset='message')
    
    return df


def save_data(df, database_filepath):
    """
    Save the cleaned DataFrame containing messages and categories data to a SQLite database.

    Args:
    df (pandas DataFrame): Cleaned DataFrame containing messages and categories data.
    database_filepath (str): Filepath for the SQLite database.

    Returns:
    None
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('DisasterResponse', con=engine, if_exists='replace', index=False)  


def main():
    """
    Load, clean, and save message and category data to a SQLite database.

    Args:
    None

    Returns:
    None
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}/DisasterResponse.db'.format(database_filepath))
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