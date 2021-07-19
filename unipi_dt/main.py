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
def load_csv_data(file):
    df = pd.read_csv(file)
    x = df.iloc[:, 4:-1]
    cols = x.columns
    y = df.iloc[:,-1]
    x = x.values
    y = y.values
    return x, y, cols


# def load_pickle_data(file):
#     x = pd.read_csv(file).iloc[:, 1:].values
#     # return x, y, cols
#     file_name =  'dataframeclean.pkl'
#     [df,dec_cat,dec_set,dec_macroset,dec_presmkt,dec_ruolo,dec_target,dec_esposizione] = pickle.load(open(file_name, 'rb'))
#     x =  df.iloc[:, :-1]
#     y = df.iloc[:,-1]
#     return df
    
    
def split_data(x, y, size=0.2, rs=123):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=size,random_state=rs)
    return x_train,x_test,y_train,y_test


def main(x, y, cols, x_test = [], y_test = [] ):
    clf = DecisionTreeClassifier(max_depth=6)
    m = clf.fit(x, y, cols=cols)
    pprint(m)
    predictions = clf.predict(x_test)
    return m, predictions


def print_scores(y_test, y_pred):
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


x, y, cols = load_csv_data('..\..\..\Data\coop_test_100.csv')
x_train,x_test,y_train,y_test = split_data(x, y)

# Target encode
trn, sub = target_encode(x_train[:,134], 
                         x_test[:,134], 
                         target=y_train, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)

# m, predictions = main(x_train, y_train, cols, x_test)
# print_scores(y_test, predictions)



# create adjusted splitting criterion


# change it all to work with pandas?

# do I want to make a nice vizualization of the splits or something?
#   maybe decide when I have a better idea of how I will analyze
