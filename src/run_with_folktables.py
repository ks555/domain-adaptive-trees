from utils import split_data, print_scores
from src.decision_tree_classifier.decision_tree_classifier import DecisionTreeClassifier
import folktables as ft
import pandas as pd
from pprint import pprint
from utils import load_folktables_data, load_task
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

"""
Created on October 7, 2022

@author: scott
 
"""

source_states = ['AR']
source_year = '2017'
target_states = ['CT']
target_year = '2017'
alpha = 0.1
source_data = load_folktables_data(source_states, source_year, '1-Year', 'person')
target_data = load_folktables_data(target_states, target_year, '1-Year', 'person')


# create dataframes, with column names, of the features and labels, from source
# load task for source - makes numpy arrays of features, labels, protected group category for given folktables task
features_s, labels_s, group_s = load_task(source_data, ft.ACSPublicCoverage)
X_s = pd.DataFrame(features_s, columns=ft.ACSPublicCoverage.features)
X_s['y'] = labels_s
y_s = X_s['y']
X_s = X_s[ft.ACSPublicCoverage.features]
# create train and test set from source
X_train_s, X_test_s, y_train_s, y_test_s = split_data(X_s, y_s)

# create same dataframes from target
features_t, labels_t, group_t = load_task(target_data, ft.ACSPublicCoverage)
X_t = pd.DataFrame(features_t, columns=ft.ACSPublicCoverage.features)
X_t['y'] = labels_t
y_t = X_t['y']
X_t = X_t[ft.ACSPublicCoverage.features]
# create train and test set from target
X_train_t, X_test_t, y_train_t, y_test_t = split_data(X_t, y_t)


def run_tree(X_train, y_train, X_test, y_test, X_td=None, alpha=0.5, max_depth=5, cat=[]):
    clf = DecisionTreeClassifier(max_depth)
    clf.fit(X_train, y_train, cat, alpha, X_td)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions['prediction'])
    return accuracy


def calculate_fairness():
    pass


def create_graph(scores, source_state, target_state):
    x = scores.keys()
    y = scores.values()
    plt.plot(x, y)
    plt.xlabel('Alpha')
    plt.ylabel("Accuracy")
    plt.title(f"Source {source_state}, Target {target_state}")
    plt.show()


def store_results():
    pass


def loop_through_alphas(X_train, y_train, X_test, y_test, X_td=X_test_t, max_depth=5, cat=[]):
    scores = {}
    for i in np.arange(0, 1.1, 0.1):
        scores[i] = run_tree(X_train, y_train, X_test, y_test, X_td=X_td, alpha=i, max_depth=5, cat=cat)
    return scores


scores = loop_through_alphas(X_train_s, y_train_s, X_test_t, y_test_t, X_td=X_train_t, max_depth=5)
create_graph(scores, source_states[0], target_states[0])

# accuracy_1 = run_tree(X_train_s, y_train_s, X_test_s, y_test_s, max_depth=5)
# accuracy_2 = run_tree(X_train_s, y_train_s, X_test_t, y_test_t, max_depth=5)
# accuracy_3 = run_tree(X_train_s, y_train_s, X_test_t, y_test_t, X_td=X_train_t, max_depth=5)

# print(f'Accuracy of standard DT trained on {source_states[0]}tested on {source_states[0]}: {accuracy_1}')
# print(f'Accuracy of standard DT trained on {source_states[0]} tested on {target_states[0]}: {accuracy_2}')
# print(f'Accuracy of DA-DT trained on {source_states[0]} tested on {target_states[0]}, alpha = {alpha}: {accuracy_3}')


"""

-------
ACSPublicCoverage Task: Predict whether a low-income individual, not eligible for Medicare, has coverage from public 
health insurance.

My interest in this one is that the task is one of finding uncovered people, thus already more interesting from a social 
value perspective than a lot of toy examples, so maybe it can take us somewhere interesting.

the thing with this one is that none of the features 'should' lead to lack of coverage, however all of them
are valid for prediction, since you want to find out if there is discrimination, and use knowledge of it 
to identify uncovered people...in this case it seems like all features are of interest...but I will start with
the protected attributes for now...

Demographic Features of Interest:
SEX (Male / Female)
RAC1P (Recoded detailed race code, 9 possible values) - but this is an encoding problem, since it is not binary or ordered
SCHL (educational attainment) - since there is no income...

-------
ACSIncome Task: Predict  whether US working adults’ yearly income is above $50,000 (threshold can be adjusted)

This might be a bit more straight forward.

Demographic Features of Interest:
SEX (Male / Female)
RAC1P (Recoded detailed race code, 9 possible values) [encoding issues? will have to calc every possible split at least]
PINCP (Total person’s income): Range of values
AGEP (Age): Range of values
________

Encoding Discrete Variables

This needs to be dealt with in general and then, whatever is used, calculating proportions of 'treated' variables that
are discrete will require knowledge about the encoding.

Our current proposal:

## proposal - target encode discrete prior to building DT
## send mappings of value / encoded value to DT
## to calculate gain, if c is a 'treated' attribute, map back to values to include stats data adjustment


"""


# Pass location(s) for which you want proportion information, as list to function of relevant demographic
# [I don't remember why we want multiple locations, but Salvatore and I needed it at the time]
# Proportion for the relevant demographic split will be returned as a list of two percentages
# Combining with demographic splits already being made? - may be an issue but possibly:
#   Just need to know what previous demographic splits were made in which order, then use this when building the query
#       For example, branch split on gender, then income, now we need gender proportion
#       Query the Location Demographics table for that by querying in that order

'''
PINCP (Total person’s income): Range of values:
– integers between -19997 and 4209995 to indicate income in US dollars
– loss of $19998 or more is coded as -19998.
– income of $4209995 or more is coded as 4209995.


def get_population(locations, gender_thresh=None, race=None, age_thresh=None, income_thresh=None, income_lower=False):
    if gender_thresh:
        pass


SEX (Sex): Range of values:
– 1: Male
– 2: Female

AGEP (Age): Range of values:
– 0 - 99 (integers) – 0 indicates less than 1 year old.
'''
