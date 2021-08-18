import sys
import pandas as pd

from sqlalchemy import create_engine 


def load_data(messages_filepath, categories_filepath):
    """[summary]

    Args:
        messages_filepath ([python string object]): [path to the message.csv dataset]
        categories_filepath ([python string object]): [path to the categories csv dataset]
        
    output:
        df ([pandas DataFrame from merging the two datasets using "id" as the common column])
    """
    # load the messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(left=messages, right=categories, how='inner', on=['id'])
    
    return df 


def clean_data(df):
    """[summary]

    Args:
        df ([pandas DataFrame]): [the output of the load function above]
    Output:
        df ([cleaned pandas DataFrame])
    """
    # create a DataFrame of the individual category columns
    
    categories = df['categories'].str.split(pat= ";", expand= True)
    
    # select the first row of the categories DataFrame
    
    row = categories[:1]
    """
        use this row to extract a list of column names for categories.
        using a lambda function to take everything up to second to last character of each string with slicing
    """
    extracted = lambda x:x[0][:-2]
    category_colnames = list(row.apply(extracted))
    
    # rename the columns of categories
    
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 to 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x:x[-1]) # sets each value to be the last character of the string
        categories[column] = categories[column].astype(int) # convert column value from string to numeric
        categories['related'] = categories['related'].replace(2, 0) # replacing the 2 in related column with 0
        
        
    """
        dropping the original categories column from the df and
        replace them with the new categories columns generated above
    """
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicate rows and keep the last one.
    df.drop_duplicates(keep = 'last', inplace = True)
    
    return df 
    

def save_data(df, database_filename):
    """[saves the clean dataframe to a new sql db file]

    Args:
        df ([cleaned pandas dataframe])
        database_filename ([sqlite database file])
    """
    #engine = create_engine('sqlite:///DisasterResponse.db')
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql("messages", engine, index=False, if_exists = 'replace')
      


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
