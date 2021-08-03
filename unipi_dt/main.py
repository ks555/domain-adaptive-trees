# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:13:50 2021

@author: scottfdfdd
"""

from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from pprint import pprint
from dt.DecisionTreeClassifier import DecisionTreeClassifier
from dt.encoding import encode_attributes
import category_encoders as ce
import pandas as pd
import pickle
import numpy as np # linear algebra
import time

start_time = time.time()

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
    
    
def target_encode_ce(x, y, to_encode):
    encoder = ce.TargetEncoder(cols=to_encode)
    encoder.fit(x, y)
    x_cleaned = encoder.transform(x)
    return x_cleaned


def split_data(x, y, size=0.2, rs=123):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=size,random_state=rs)
    return x_train,x_test,y_train,y_test


def homemade_tree(x, y, cols, x_test = [], y_test = []):
    clf = DecisionTreeClassifier(max_depth=6)
    m = clf.fit(x, y, cols=cols)
    pprint(m)
    predictions = clf.predict(x_test)
    print_scores(y_test, predictions)
    return m, predictions


def scikit_tree(x_train, y_train, x_test, y_test):
    # Create Decision Tree classifer object
    clf = tree.DecisionTreeClassifier(max_depth=6)
    # Train Decision Tree Classifer
    clf = clf.fit(x_train,y_train)
    print(tree.plot_tree(clf))
    #Predict the response for test dataset
    predictions = clf.predict(x_test)
    # Model Accuracy, how often is the classifier correct?
    print_scores(y_test, predictions)
    return clf, predictions


def print_scores(y_test, y_pred):
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    return accuracy_score(y_test, y_pred)
    


def __main__(file, data='csv', tree='homemade', encode='ce'):
    ## attributes to encode
    to_encode = ['n_nucleo', 'negozio_comune', 'negozio_prov',
           'negozio_regione', 'negozio_tipo', 'categoria', 'cooperativa', 'sesso',
           'stato_civile', 'professione', 'titolo_studio', 'cliente_comune',
           'cliente_prov', 'cliente_regione']
    ## choose load data method, last column is target
    if data == 'csv':
        x, y, cols = load_csv_data(file)
        print(cols)
    elif data == 'iris':
        x, y, cols = load_iris_data()
    elif data == 'pickle':
        x, y, cols = load_pickle_data('../../../Data/dataframeclean.pkl')
    ## encode attributes listed above
    if encode == 'ce':
        x = target_encode_ce(x, y, to_encode)
        ## split in test and training
        x_train,x_test,y_train,y_test = split_data(x, y)
    elif encode == 'kaggle':
        ## split in test and training
        x_train,x_test,y_train,y_test = split_data(x, y)
        x_train, x_test = encode_attributes(x_train, x_test, y_train, to_encode)
    # call a tree
    if tree == 'scikit':
        m, predictions = scikit_tree(x_train, y_train, x_test, y_test)
    elif tree == 'homemade':
        # works with arrays, not dataframes
        m, predictions = homemade_tree(x_train.values, y_train.values, cols, x_test.values, y_test.values)
        

#__main__('E:\scott\Data\coop_04_2016_prediction_1000.csv')

to_encode = ['n_nucleo', 'negozio_comune', 'negozio_prov',
           'negozio_regione', 'negozio_tipo', 'categoria', 'cooperativa', 'sesso',
           'stato_civile', 'professione', 'titolo_studio', 'cliente_comune',
           'cliente_prov', 'cliente_regione']
## load data from csv
X, y, cols = load_csv_data('E:\scott\Data\coop_04_2016_prediction_1000.csv')
## Load istat data
age_data = pd.read_csv('E:\scott\Data\ISTAT_AGE_GENDER.csv')
marital_data = pd.read_csv('E:\scott\Data\ISTAT_MARITAL.csv')


X_train, X_test, y_train, y_test = split_data(X, y)
clf = DecisionTreeClassifier(max_depth=4, cat=to_encode)
clf.fit(X_train, y_train)
pprint(clf.tree)
# create adjusted splitting criterion
predictions = clf.predict(X_test)
accuracy = print_scores(y_test, predictions['prediction'])



with open('timing.csv','a') as fd:
    fd.write(f"{round(time.time() - start_time, 2)} seconds --- rows {X.shape[0]} cols {X.shape[1]} Acc {accuracy} \n")

