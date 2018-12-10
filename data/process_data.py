import sys
import pandas as pd



def load_data(messages_filepath, categories_filepath):
    """ Loads and merges the disaster messages and the corresponding categories from csv-files
    
    Args:
        messages_filepath(str): path to the csv file with the disaster messages
        categories_filepath(str): path to the csv file withe the categories of the messages
        
    Returns:
        df_combined(obj): a pandas Data Frame which the messages together with the 
                          corresponding categories
    
    """
    
    df_categories = pd.read_csv(categories_filepath)
    df_messages = pd.read_csv(messages_filepath)
    df_combined = df_messages.merge(df_categories,on='id')
    
    return(df_combined)
    


def clean_data(df):
    """Cleans the messages data and the category data
    
    Args:
        df(obj): a pandas Data Frame with a column category whoes values are of type
                 'category_name-category_value', e.g. aid_related-0. For every record 
                 in df all possible categories appear in the category column
                 and they are separated by ';'
    Returns:
        df_clean(obj): a pandas Data Frame with seperate columns for each category with
                       values 0 resp. 1 depending on wether a message belongs to the category or not
    
    """
    
    #create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # create list with different category names
    category_column_names = [cat.split('-')[0] for cat in row]
    # rename the columns of the categories Data Frame
    categories.columns = category_column_names
    # replace the values in the category columns by 0 resp. 1
    for col in categories.columns:
        categories[col] = categories[col].apply(lambda row: row.split('-')[1])
    # convert column from string to numeric   
    for col in categories.columns:
        categories[col] = pd.to_numeric(categories[col])
    # replace the single category column in the original Data Frame df with the
    # cleaned category columns
    df.drop(labels=['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)
    # remove dublicates
    df_clean = df[df.duplicated() == False]
    return df_clean
    

def save_data(df, database_filename):
    pass  


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