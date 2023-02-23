from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
from numpy import ndarray, os



def load_pickle(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

def save_pickle(filename, result):
    with open(filename, "wb") as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


def prepare_data():
        # preparing dataset
        df = pd.read_csv('dataset_with_pages/dataset.csv',  encoding='utf-8')
        df['text'] = df['text'].values.astype('U')

        X = df[['text', 'num_pages']]
        y = df['category']
        return X, y

def train_model():
        pass


# Machine learning NLP Model to recognize the type of the document.
def document_classification(df):
        X, y = prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

        # construct the column transfomer
        column_transformer = ColumnTransformer([('tfidf_transformer', TfidfVectorizer(ngram_range=(1, 2)), 'text'),
                                                ("num_pages", MinMaxScaler(), ["num_pages"])], remainder='drop')
        # build pipeline
        text_clf = Pipeline(steps=[('features', column_transformer), ('classifier', LogisticRegression())])

        # train the model
        # text_clf.fit(X_train, y_train) 

        # Save the model to disk
        filename = r'ocr/data_extraction/saved_model/trained_model.sav'
        # save_pickle(filename, text_clf)

        # load the model from disk
        text_clf = load_pickle(filename)

        # Evaluation of the model
        # tags = ['AH', 'amo', 'audit', 'certificat', 'cofrac', 'devis', 'facture',
        #         'fiche_preconisation', 'geolocalisation', 'geoportail', 'impots',
        #         'justificatif_domicile', 'liste_entreprises', 'synthese_audit']
        # y_pred = text_clf.predict(X_test)

        # print(classification_report(y_test, y_pred, target_names=tags, zero_division=1))
        prediction = text_clf.predict(df.head(1))
        return ndarray.tolist(prediction)[0]
