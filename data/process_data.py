import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Parameters:
            messages_filepath (string): A filepath to dataframe with tweets
            categories_filepath (string): A filepath to dataframe with disaster response category labels

    Returns:
            tweetdf (dataframe): A dataframe containing documents and labels
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merged datasets
    tweetdf = messages.merge(categories,on='id')
    return tweetdf


def clean_data(tweetdf):
    '''
    Parameters:
            tweetdf (dataframe): A dataframe containing documents and labels

    Returns:
            tweetdf (dataframe): A datframe containing documents and revised column names for labels
    '''
    # create a dataframe of the 36 individual category columns
    categories = tweetdf.categories.str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.head(1)

    # use this row to extract a list of new column names for categories.
    category_colnames = list(row.apply(lambda x: x.str.split('-')[0][0]))
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    # drop the original categories column from `df`
    tweetdf.drop('categories',axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    tweetdf = pd.concat([tweetdf,categories],axis=1)
    # drop duplicates
    tweetdf.drop_duplicates(inplace=True)
    #converts values to binary
    tweetdf = tweetdf[tweetdf['related']!=2]
    return tweetdf

def save_data(tweetdf, database_filename, table_name='labeled_messages'):
    '''
    Parameters:
            tweetdf (dataframe): A dataframe containing documents and labels
            database_filename (string): A filepath for sqlite database
            table_name (string): A name for tweetdf in db
    '''
    engine = create_engine('sqlite:///'+database_filename)
    tweetdf.to_sql(table_name, engine, index=False)
    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        tweetdf = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        tweetdf = clean_data(tweetdf)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(tweetdf, database_filepath)

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
