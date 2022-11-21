from utils import split_data, print_scores
from src.decision_tree_classifier.decision_tree_classifier import DecisionTreeClassifier
import folktables as ft
import pandas as pd
from pprint import pprint
from utils import load_folktables_data, load_task
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import datetime

"""
Created on October 7, 2022

@author: scott
 
"""

source_states = ['OH']
source_year = '2017'
target_states = ['AL']
target_year = '2017'
alpha = 0.5
task = 'ACSPublicCoverage'


def create_dfs(source_states, source_year, target_states, target_year, task='ACSPublicCoverage'):
    source_data = load_folktables_data(source_states, source_year, '1-Year', 'person')  # todo: change to download list of states at once than parse the DF created
    target_data = load_folktables_data(target_states, target_year, '1-Year', 'person')
    task_method = getattr(ft, task)
    # load task - makes numpy arrays of features, labels, protected group category for given folktables task
    # we use this to create dataframes, with column names, of the features and labels, from source

    features_s, labels_s, group_s = load_task(source_data, task_method)
    X_s = pd.DataFrame(features_s, columns=task_method.features)
    X_s['y'] = labels_s
    y_s = X_s['y']
    X_s = X_s[task_method.features]
    # create train and test set from source
    X_train_s, X_test_s, y_train_s, y_test_s = split_data(X_s, y_s)

    # create same dataframes from target
    features_t, labels_t, group_t = load_task(target_data, task_method)
    X_t = pd.DataFrame(features_t, columns=task_method.features)
    X_t['y'] = labels_t
    y_t = X_t['y']
    X_t = X_t[task_method.features]
    # create train and test set from target
    X_train_t, X_test_t, y_train_t, y_test_t = split_data(X_t, y_t)

    return X_train_s, X_test_s, y_train_s, y_test_s, X_train_t, X_test_t, y_train_t, y_test_t


def run_tree(X_train, y_train, X_test, y_test, X_td=None, alpha=0.5, max_depth=5, cat=[]):
    clf = DecisionTreeClassifier(max_depth)
    clf.fit(X_train, y_train, cat, alpha, X_td)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions['prediction'])
    return accuracy


# returns equalized odds between white and black groups
def run_tree_full_results(X_train, y_train, X_test, y_test, X_td=None, alpha=0.5, max_depth=5, cat=[]):
    clf = DecisionTreeClassifier(max_depth)
    clf.fit(X_train, y_train, cat, alpha, X_td)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions['prediction'])
    confusion_values_black = confusion_matrix_values(y_test[(X_test['RAC1P'] == 2)],
                                                     predictions['prediction'][(X_test.reset_index()['RAC1P'] == 2)])
    confusion_values_white = confusion_matrix_values(y_test[(X_test['RAC1P'] == 1)], predictions['prediction'][(X_test.reset_index()['RAC1P'] == 1)])
    # calculate and return equalized odds for white black
    return accuracy, confusion_values_white[1]-confusion_values_black[1]


def confusion_matrix_values(y_test, predictions):
    """Calculate confusion matrix rates for population or subgroup"""
    cm = confusion_matrix(y_test, predictions)
    TN, FP, FN, TP = cm.ravel()

    N = TP + FP + FN + TN  # Total population
    ACC = (TP + TN) / N  # Accuracy
    TPR = TP / (TP + FN)  # True positive rate
    FPR = FP / (FP + TN)  # False positive rate
    FNR = FN / (TP + FN)  # False negative rate
    PPP = (TP + FP) / N  # % predicted as positive

    return np.array([ACC, TPR, FPR, FNR, PPP])


def create_graph(scores, source_state, target_state, zoom_axis=False):
    x = 1-np.array(list(scores.keys()))
    y = scores.values()

    plt.plot(x, y)
    plt.xlabel('1 - Alpha')
    plt.ylabel("Accuracy")
    plt.title(f"Source {source_state}, Target {target_state}")
    if not zoom_axis:
        plt.ylim(0, 1)
    # after plotting the data, format the labels
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.1%}'.format(x) for x in current_values])
    plt.show()


def create_compiled_graph(scores_dict, sources, targets, fairness_values_dict=None, zoom_axis=True):
    for source in sources:
        # make a graph
        fig, ax = plt.subplots()
        ax.set_xlabel('1 - Alpha')
        ax.set_ylabel("Accuracy", color='red')
        ax.set_title(f"Source {source}")
        ax2 = ax.twinx()
        ax2.set_ylabel('Equal Opportunity', color='blue')
        for target in targets:
            try:
                scores = scores_dict[(source, target)]
                fairness_values = fairness_values_dict[(source, target)]
                # x values are based on alpha
                x = 1 - np.array(list(scores.keys()))
                # y values are accuracy for given lambda
                y = scores.values()
                tpr = fairness_values.values()
                # tnr = [item[2] for item in confusion_values.values()]
                ax.plot(x, y, color='red')
                ax2.plot(x, tpr, color='blue')
                # ax2.plot(x, tnr, color='darkblue')
            except:
                print(f'failed to add {source}, {target} data')
        if not zoom_axis:
            ax.set_ylim(0.0, 1)
        ax2.tick_params(axis='y', labelcolor='blue')
        ax.tick_params(axis='y', labelcolor='red')
        # after plotting the data, format the labels
        left_values = ax.get_yticks()
        ax.set_yticklabels(['{:,.1%}'.format(x) for x in left_values])
        fig.tight_layout()
        # plt.show()
        try:
            plt.savefig(f"../results/Source_{source}.jpg")
        except:
            plt.savefig(f"results/Source_{source}.jpg")



def loop_through_alphas(X_train, y_train, X_test, y_test, X_td=None, max_depth=5, cat=[]):
    scores = {}
    confusion_values = {}
    for i in np.arange(0, 1.1, 0.1):
        scores[i], confusion_values[i] = run_tree_full_results(X_train, y_train, X_test, y_test, X_td=X_td, alpha=i, max_depth=5, cat=cat)
    return scores, confusion_values


def loop_through_sources_targets(sources, targets, source_year='2017', target_year='2017', max_depth=5, task='ACSPublicCoverage'):
    scores_dict = {}
    confusion_values_dict = {}
    # if source and target lists are equal, we get all combos of that list
    # including matching source / target (which is a useful baseline)
    for source in sources:
        print(source)
        for target in targets:
            try:
                X_train_s, X_test_s, y_train_s, y_test_s, X_train_t, X_test_t, y_train_t, y_test_t = create_dfs(
                    [source], source_year, [target], target_year, task=task)
                scores_dict[(source, target)], confusion_values_dict[(source, target)] = loop_through_alphas(
                    X_train_s, y_train_s, X_test_t, y_test_t, X_td=X_train_t, max_depth=max_depth)
            except:
                print(f'failed to run for {source}, {target}')
            try:
                pickle.dump(scores_dict, open(f"../results/scores_{source}.pkl", "wb"))
                pickle.dump(confusion_values_dict, open(f"../results/confusion_{source}.pkl", "wb"))
            except:
                pickle.dump(scores_dict, open(f"results/scores_{source}.pkl", "wb"))
                pickle.dump(confusion_values_dict, open(f"results/confusion_{source}.pkl", "wb"))

    return scores_dict, confusion_values_dict


def run_it_all():
    scores_dict, equalized_odds_dict = loop_through_sources_targets(['AL'], ['MS', 'ID'])
    create_compiled_graph(scores_dict, ['AL'], ['MS', 'ID'], equalized_odds_dict)


states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
          'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
          'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
          'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
          'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']


scores_dict, equalized_odds_dict = loop_through_sources_targets(['AR'], ['AZ'])
# pickle.dump(scores_dict, open(f"../results/scores_dict{datetime.now().strftime('%m%d%Y_%H%M%S')}.pkl", "wb"))
# pickle.dump(equalized_odds_dict, open(f"../results/EO_dict{datetime.now().strftime('%m%d%Y_%H%M%S')}.pkl", "wb"))
create_compiled_graph(scores_dict, ['AR'], ['AZ'], equalized_odds_dict, False)


# Run one tree
# X_train_s, X_test_s, y_train_s, y_test_s, X_train_t, X_test_t, y_train_t, y_test_t = create_dfs(
#                 source_states, source_year, target_states, target_year, task=task)
# accuracy, predictions = run_tree_full_results(X_train_s, y_train_s, X_test_s, y_test_s, X_td=None,
# alpha=1, max_depth=5, cat=[])
# confusion_values = confusion_matrix_values(y_test_s, predictions['prediction'])
# confusion_values_black = confusion_matrix_values(y_test_s[(X_test_s['RAC1P'] == 2)], predictions['prediction'][(X_test_s.reset_index()['RAC1P'] == 2)])
# confusion_values_white = confusion_matrix_values(y_test_s[(X_test_s['RAC1P'] == 1)], predictions['prediction'][(X_test_s.reset_index()['RAC1P'] == 1)])
# Equal Opportunity = ratio of EO between groups
# Equalized odds = EO and also the ratio of false positives
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
