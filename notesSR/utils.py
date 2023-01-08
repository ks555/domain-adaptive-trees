# -*- coding: utf-8 -*-
"""
Utility functions
"""

# global imports
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# external imports
import folktables as ft
from fairlearn.postprocessing import ThresholdOptimizer

# local imports
from decision_tree_classifier import DecisionTreeClassifier

# states
states = sorted(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
          'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
          'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
          'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
          'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR'])
# categorical attributes
cat_atts = ['SCHL', 'MAR', 'SEX',  'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'ESR', 'ST', 'FER', 'RAC1P']

# subset of attributes
def get_subset(subset_string='all'):
    if subset_string=='subset1':
        subset = ['SCHL', 'MAR', 'AGEP', 'SEX', 'CIT', 'RAC1P']
    elif subset_string=='subset2':
        subset = ['AGEP', 'SEX', 'RAC1P']
    elif subset_string=='cat':
        subset = cat_atts
    else:
        subset = ft.ACSPublicCoverage.features
    return subset

# data split into training and test
def split_data(X, y, test_size = 0.25, random_state = 42):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test

# load folktables
def load_folktables_data(states=["CA"], survey_year='2018', horizon='1-Year', survey='person'):
    # add check for data, so it doesn't need to download
    root_dir = "../"
    state_codes = pd.read_csv(os.path.join(root_dir, 'data', 'adult', 'state_codes.csv'))
    acs_data = pd.DataFrame()
    # To avoid downloading each time, check per state if downloaded, if not, download
    # Either way, append the state data to acs_data data frame, updating the region field
    for i in range(0, len(states)):
        # get state code
        code = state_codes.loc[state_codes['USPS'] == states[0]]['numeric'].values[0]
        data_path = os.path.join(root_dir, "data", survey_year, horizon, f"psam_p{code}.csv")
        # This file path works with person, not household survey
        if os.path.exists(data_path):
            # load from csv, update to region == i, append to acs_data
            state_data = pd.DataFrame(data_path)
            state_data.REGION = i = i+1            
            acs_data = acs_data.append(state_data, ignore_index=True)
        else:
            # download that state
            data_source = ft.ACSDataSource(survey_year=survey_year,
                                           horizon=horizon, survey=survey, root_dir=os.path.join(root_dir, 'data', 'adult'))
            state_data = data_source.get_data(states=[states[i]], download=True)
            # update the region field so that data can be identified by state
            state_data.REGION = i = i+1
            # append to acs_data
            acs_data = acs_data.append(state_data, ignore_index=True)
    return acs_data

# takes your loaded data and splits into features, labels, group membership vectors
def load_task(acs_data, task):
    features, labels, group = task.df_to_numpy(acs_data)
    return features, labels, group

# load and split data
def load_ACSPublicCoverage(subset, states=states, source_year="2017", target_year="2017", task_method=ft.ACSPublicCoverage):
    X_train_s, X_test_s, y_train_s, y_test_s = dict(), dict(), dict(), dict()
    for s in states:
        print(s, end=' ')
        source_data = load_folktables_data([s], source_year, '1-Year', 'person')  
        features_s, labels_s, group_s = load_task(source_data, task_method)
        X_s = pd.DataFrame(features_s, columns=task_method.features)
        X_s['y'] = labels_s
        y_s = X_s['y']
        X_s = X_s[subset]
        X_train_s[s], X_test_s[s], y_train_s[s], y_test_s[s] = split_data(X_s, y_s)
        
    if target_year == source_year:
        X_train_t, X_test_t, y_train_t, y_test_t = X_train_s, X_test_s, y_train_s, y_test_s
    else:
        X_train_t, X_test_t, y_train_t, y_test_t = dict(), dict(), dict(), dict()
        for s in states:
            print(s, end=' ')
            target_data = load_folktables_data([s], target_year, '1-Year', 'person')  
            features_t, labels_t, group_t = load_task(target_data, task_method)
            X_t = pd.DataFrame(features_t, columns=task_method.features)
            X_t['y'] = labels_t
            y_t = X_t['y']
            X_t = X_t[subset]
            X_train_t[s], X_test_t[s], y_train_t[s], y_test_t[s] = split_data(X_t, y_t)
    return X_train_s, X_test_s, y_train_s, y_test_s, X_train_t, X_test_t, y_train_t, y_test_t 

# train and test model
def run_test(X_train, y_train, X_test, y_test, X_td, max_depth, min_cases=5, alpha=None, cat=cat_atts, 
             t_o=None, y_td=None, att_td=None):
    clf = DecisionTreeClassifier(max_depth, min_cases=min_cases)
    clf.fit(X_train, y_train, cat_atts=cat, alpha=alpha, X_td=X_td, y_td=y_td, att_td=att_td)    
    if t_o is not None:
        post_clf = ThresholdOptimizer(estimator=clf, constraints=t_o, prefit=True, predict_method='predict')
        post_clf.fit(X_train, y_train, sensitive_features=X_train['SEX'])        
        y_pred = post_clf.predict(X_test, sensitive_features=X_test['SEX'], random_state=42) # fair-corrected predictions 
    else:
        y_pred = clf.predict(X_test)
    # performance confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # fairness confusion matrices
    male = 1
    cm_male = confusion_matrix(y_test[(X_test['SEX'] == male)], y_pred[(X_test.reset_index()['SEX'] == male)]) 
    female = 2
    cm_female = confusion_matrix(y_test[(X_test['SEX'] == female)], y_pred[(X_test.reset_index()['SEX'] == female)])
    return clf, cm, cm_male, cm_female

# extract metrics
def cm_metrics(cm):
    TN, FP, FN, TP = cm.ravel()
    N = TP + FP + FN + TN  # Total population
    ACC = (TP + TN) / N  # Accuracy
    TPR = TP / (TP + FN)  # True positive rate
    FPR = FP / (FP + TN)  # False positive rate
    FNR = FN / (TP + FN)  # False negative rate
    PPP = (TP + FP) / N  # % predicted as positive
    return [ACC, TPR, FPR, FNR, PPP]

# calculate metrics (the higher the better)
def get_metric(r, m):
    if m=='acc':
        return cm_metrics(r['cm'])[0]
    if m=='eqacc':
        return cm_metrics(r['cm_protected'])[0] - cm_metrics(r['cm_unprotected'])[0]
    if m=='eop':
        return cm_metrics(r['cm_protected'])[1] - cm_metrics(r['cm_unprotected'])[1]
    if m=='dp': # the higher the better
        return -abs(cm_metrics(r['cm_protected'])[4] - cm_metrics(r['cm_unprotected'])[4])
    raise "unknown metric"
