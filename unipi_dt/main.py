# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:13:50 2021

@author: scott
"""

from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pprint import pprint
from dt.DecisionTreeClassifier import DecisionTreeClassifier
from dt.encoding import encode_attributes
import pandas as pd
import pickle
import numpy as np # linear algebra



def load_iris_data():
    iris = load_iris()
    x = iris.data
    y = iris.target
    cols = iris.feature_names
    return x, y, cols


def load_boston_data():
    boston = load_boston()
    x = boston.data
    y = boston.target
    cols = boston.feature_names
    return x, y, cols


# load csv with last column as target
def load_csv_data_array(file):
    df = pd.read_csv(file)
    x = df.iloc[:,:-1]
    cols = x.columns
    y = df.iloc[:,-1]
    x = x.values
    y = y.values
    return x, y, cols


def load_csv_data(file):
    df = pd.read_csv(file)
    x = df.iloc[:,:-1]
    cols = x.columns
    y = df.iloc[:,-1]
    return x, y, cols


def load_pickle_data(file):
    [df,dec_cat,dec_set,dec_macroset,dec_presmkt,dec_ruolo,dec_target,dec_esposizione] = pickle.load(open(file, 'rb'))
    x =  df.iloc[:, :-1]
    y = df.iloc[:,-1]
    return x, y, x.columns
    
    
def split_data(x, y, size=0.2, rs=123):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=size,random_state=rs)
    return x_train,x_test,y_train,y_test


def main(x, y, cols, x_test = [], y_test = []):
    clf = DecisionTreeClassifier(max_depth=6)
    m = clf.fit(x, y, cols=cols)
    pprint(m)
    predictions = clf.predict(x_test)
    return m, predictions
    # return m


def print_scores(y_test, y_pred):
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')


to_encode = ['n_nucleo', 'negozio_comune', 'negozio_prov',
       'negozio_regione', 'negozio_tipo', 'categoria', 'cooperativa', 'sesso',
       'stato_civile', 'professione', 'titolo_studio', 'cliente_comune',
       'cliente_prov', 'cliente_regione']


# choose load data method, last column is target
x, y, cols = load_csv_data('..\..\..\Data\coop_04_2016_prediction_1000.csv')
# x, y, cols = load_iris_data()
# x, y, cols = load_pickle_data('../../../Data/dataframeclean.pkl')
# split in test and training
x_train,x_test,y_train,y_test = split_data(x, y)
x_train, x_test = encode_attributes(x_train, x_test, y_train, to_encode)
m, predictions = main(x_train.values, y_train.values, cols, x_test=x_test.values)
# m = main(x_train.values, y_train.values, cols, x_test=x_test.values)
print_scores(y_test, predictions)



# create adjusted splitting criterion


# change it all to work with pandas?

# do I want to make a nice vizualization of the splits or something?
#   maybe decide when I have a better idea of how I will analyze
