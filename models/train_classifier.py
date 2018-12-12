import sys
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV


def load_data(database_filepath, table_name = "disaster_messages_clean"):
    """Loads disaster data from SQLite database
    
    Args:
        database_filepath(str): path to the SQLite database
        table_name(str): name of the table where the data is stored
        
    Returns:
        X(obj): array which contains the text messages
        Y(obj): array which contains the labels to the messages
        col_target(obj): a list with the target column names, i.e. the category names
    
    """
    
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(table_name, engine)
    # define all columns which corresponds to categories
    col_target = [col for col in df.columns if col 
                  not in ['id', 'message', 'original', 'genre', 'index']]
    # seperate the messages (raw features) from the categories (target)
    X = df['message'] 
    Y = df[col_target] 
    
    return X,Y, col_target


def tokenize(text):
    """Tokenizes a text
    
    Args:
        text(str): a raw text 
        
    Returns:
        tokens(obj): list of tokens based on the raw text input
    
    """
    
    text = text.lower()
    #remove punktuation
    text = re.sub("[^a-zA-Z0-9_]", " ", text)
    # apply tokenization
    tokens = word_tokenize(text)
    # apply lemmatisation
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    
    return tokens


def build_model():
    """Build a model including grid search
    
    Args:
        
    Returns:
        gs_cv(obj): an estimator which chains together a nlp-pipeline with
                    a multi-class classifier
    
    """
    
    # initialize transformes and predictor
    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()
    clf = MultiOutputClassifier(RandomForestClassifier())
    
    #define pipeline
    pipeline_params =  [('vect', vect), ('tfidf', tfidf), ('clf', clf)]
    pipeline = Pipeline(pipeline_params)
    
    #define parameter for grid search
    grid_search_params = {'vect__ngram_range': ((1, 1), (1, 2)),
                          'vect__max_df': (0.5, 0.75, 1.0),
                          'vect__max_features': (None, 5000, 10000),
                          'tfidf__use_idf': (True, False)}
    
    gs_cv = GridSearchCV(pipeline, grid_search_params)
    
    return gs_cv

    


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the model performance for all categories on the test set
    
    Args:
        model(obj): a predictor
        X_test(obj): an array with the test features
        Y_test(obj): an array which contains the targets (which correspond to X_test)
        category_names(obj): list with the category names
        
    Returns:
        Nothing. But prints a classification report for every category
    
    """
    
    Y_test_df = pd.DataFrame(data = Y_test, columns=category_names)
    predictions = model.predict(X_test)
    predictions_df = pd.DataFrame(data = predictions, columns=category_names)
    
    for category in category_names:
        print(category)
        print(classification_report(predictions_df[category], Y_test_df[category]))
    

def save_model(model, model_filepath):
    """Takes a model and makes a pickle file out of it
        
    Args:
        model(obj): an predictor
        model_filepath(str): path to the location where the pickle file should be stored
        
    Returns:
        Nothing
        
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """Main function which performs the training of the model and saves it to disk.
    
    Args:
        None
        
    Returns:
        Nothing
    
    """
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