"""
Utility functions for loading data and computing (fairness and performance) metrics
"""

# global imports
from sklearn.metrics import confusion_matrix
from fairlearn.postprocessing import ThresholdOptimizer
# local imports
import utils
from decision_tree_classifier import DecisionTreeClassifier

# train and test decision tree
def run_test(X_train, y_train, X_test, y_test, max_depth, min_cases=5, fairness_metric=None,
             da=False, X_td=None, y_td=None, att_xw=None, maxdepth_td=None):
    cat=utils.cat_atts
    clf = DecisionTreeClassifier(max_depth, min_cases=min_cases)
    clf.fit(X_train, y_train, cat_atts=cat, da=da, X_td=X_td, y_td=y_td, att_xw=att_xw, maxdepth_td=maxdepth_td)    
    if fairness_metric is not None:
        print(fairness_metric)
        post_clf = ThresholdOptimizer(estimator=clf, constraints=fairness_metric, prefit=True, predict_method='predict')
        post_clf.fit(X_train, y_train, sensitive_features=X_train['SEX'])        
        y_pred = post_clf.predict(X_test, sensitive_features=X_test['SEX'], random_state=42) # fair-corrected predictions 
    else:
        y_pred = clf.predict(X_test)
    # performance confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    if False: # debug
        print('TP, Pred=1, True=1', sum( (y_pred==1) & (y_test==1)))
        print('FP, Pred=1, True=0', sum( (y_pred==1) & (y_test==0)))
        print('TN, Pred=0, True=0', sum( (y_pred==0) & (y_test==0)))
        print('FN, Pred=0, True=1', sum( (y_pred==0) & (y_test==1)))
    # fairness confusion matrices, male is 1, female is 2
    cm_male = confusion_matrix(y_test[(X_test['SEX'] == 1)], y_pred[(X_test.reset_index()['SEX'] == 1)]) 
    cm_female = confusion_matrix(y_test[(X_test['SEX'] == 2)], y_pred[(X_test.reset_index()['SEX'] == 2)])
    return clf, cm, cm_male, cm_female

#
# EOF
#