"""
Utility functions for loading data and computing (fairness and performance) metrics
"""

# global imports
import os
import pandas as pd
from sklearn.model_selection import train_test_split
# external imports
import folktables as ft

# states
states = sorted(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'])

# categorical attributes
cat_atts = ['SCHL', 'MAR', 'SEX',  'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC',
            'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'ESR', 'ST', 'FER', 'RAC1P']


# subset of attributes: subset1 is the one used in the paper
def get_attributes(subset='all'):
    if subset == 'subset1':
        atts = ['SCHL', 'MAR', 'AGEP', 'SEX', 'CIT', 'RAC1P']
    elif subset == 'subset2':
        atts = ['AGEP', 'SEX', 'RAC1P']
    elif subset == 'cat':
        atts = cat_atts
    else:
        atts = ft.ACSPublicCoverage.features
    return atts


# data split into training and test
def split_data(X, y, test_size=0.25, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


# load folktables state data
def load_folktables_data(state, survey_year='2017', horizon='1-Year', survey='person'):
    # add check for data, so it doesn't need to download
    root_dir = "../"
    state_codes = pd.read_csv(os.path.join(root_dir, 'data', 'adult', 'state_codes.csv'))
    # To avoid downloading each time, check per state if downloaded, if not, download
    # Either way, append the state data to acs_data data frame, updating the region field
    # get state code
    code = state_codes.loc[state_codes['USPS'] == state]['numeric'].values[0]
    data_path = os.path.join(root_dir, "data", survey_year, horizon, f"psam_p{code}.csv")
    # This file path works with person, not household survey
    if os.path.exists(data_path):
        # load from csv, update to region == i, append to acs_data
        state_data = pd.DataFrame(data_path)
    else:
        # download that state (and save in .csv format)
        data_source = ft.ACSDataSource(survey_year=survey_year, horizon=horizon, survey=survey,
                                       root_dir=os.path.join(root_dir, 'data', 'adult'))
        state_data = data_source.get_data(states=[state], download=True)
    return state_data


# load and split data about states
def load_ACSPublicCoverage(subset, states=states, year="2017"):
    # Dictionaries mapping states to train-test data
    X_train_s, X_test_s, y_train_s, y_test_s = dict(), dict(), dict(), dict()
    task_method = ft.ACSPublicCoverage
    for s in states:
        print(s, end=' ')
        source_data = load_folktables_data(s, year, '1-Year', 'person')  
        features_s, labels_s, group_s = task_method.df_to_numpy(source_data)
        X_s = pd.DataFrame(features_s, columns=task_method.features)
        X_s['y'] = labels_s
        y_s = X_s['y']
        X_s = X_s[subset]
        X_train_s[s], X_test_s[s], y_train_s[s], y_test_s[s] = split_data(X_s, y_s)
    # Target is same as source, because in the same year 
    X_train_t, X_test_t, y_train_t, y_test_t = X_train_s, X_test_s, y_train_s, y_test_s
    return X_train_s, X_test_s, y_train_s, y_test_s, X_train_t, X_test_t, y_train_t, y_test_t 


# extract metrics from confusion matrix
def cm_metrics(cm):
    TN, FP, FN, TP = cm.ravel()
    N = TP + FP + FN + TN  # Total population
    ACC = (TP + TN) / N    # Accuracy
    TPR = TP / (TP + FN)   # True positive rate
    FPR = FP / (FP + TN)   # False positive rate
    FNR = FN / (TP + FN)   # False negative rate
    PPP = (TP + FP) / N    # % predicted as positive
    return [ACC, TPR, FPR, FNR, PPP]


# calculate accuracy and fairness metrics
def get_metric(r, m):
    # accuracy - the higher the better
    if m == 'acc':
        return cm_metrics(r['cm'])[0]
    # equal accuracy - the smaller the better
    if m == 'eqacc':
        return abs(cm_metrics(r['cm_protected'])[0] - cm_metrics(r['cm_unprotected'])[0])
    # equality of opportunity - the smaller the better
    if m == 'eop':
        return abs(cm_metrics(r['cm_protected'])[1] - cm_metrics(r['cm_unprotected'])[1])
    # demographic parity - the smaller the better
    if m == 'dp':
        return abs(cm_metrics(r['cm_protected'])[4] - cm_metrics(r['cm_unprotected'])[4])
    raise "unknown metric"

#
# EOF
#
