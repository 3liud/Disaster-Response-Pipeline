import sys
#download necessary NLTK daat
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])


# import libraries
import re
import pandas as pd
from nltk.corpus import stopwords
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import gzip 



def load_data(database_filepath):
    """[loads data and splits it into X and y variables]

    Args:
        database_filepath ([Sqlite database file]): [description]
    output: 
        feature and target variables X & Y along with the target column
    """
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql("SELECT * from disaster_dataset", engine)
    X = df['message'] # feature selection
    Y = df[df.columns[4:]] # target values to predict
    
    return X, Y


def tokenize(text):
    # remove url and replace with a placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
        
    # text normalization
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).strip()
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    cleaned_tokens = []
    for token in tokens:
        cleaned_tok = lemmatizer.lemmatize(token).lower()
        cleaned_tokens.append(cleaned_tok)
        
    return cleaned_tokens
    pass


def build_model():
    """[Function to build the model, train it and improve it using SKLEARN]
    args:
        None
    output:
        classifier model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=1)))
    ])
    
    # hyperparameters for Grid search
    
    parameters = {
        'clf__estimator__min_samples_leaf': [8, 10, 12],
        'clf__estimator__min_samples_split': [3, 5, 9]
    }
    
    # model to be used
    model = GridSearchCV(pipeline, param_grid = parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """[function evaluates the model based on teh test data.]

    Args:
        model ([the trained classifier model from above])
        X_test ([the test data]): [data to be used for testing features]
        Y_test ([known values]): [true valeus to be compared with test cases]
        category_names ([column names of the Y_test data])
    Output:
        print the model prediction accuracy on the test data
    """
    y_pred = model.predict(X_test)
    accuracy = (y_pred == Y_test).mean()
    print("Accuracy:", accuracy)
    # print(classification_report(Y_test, y_pred, target_names= df.columns[4:]))
    
    pass


def save_model(model, model_filepath):
    """[function saves the trained model]

    Args:
        model ([trained classifier from above])
        model_filepath ([filepath ]): [provides the location and name to be used in saving the model]
    """
    model_filepath = 'rf_classifier_model.pkl'
    pickle.dump(model, open(model_filepath, 'wb'))
    # with gzip.open(model_filepath, 'wb') as f:
    #     Pickle.dump(model, f) 
        
    pass


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
