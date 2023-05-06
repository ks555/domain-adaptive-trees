"""
Utility functions for loading data and computing (fairness and performance) metrics
"""

# global imports
import sys
import time
import pickle
import subprocess
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
        #print(fairness_metric)
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

def main():
    if len(sys.argv) == 2 and sys.argv[1]=='all_experiments':
        all_experiments()
        return
    if len(sys.argv) not in {5, 6}:
        # fairness_metric can be "demographic_parity" or "true_positive_rate_parity"
        print('Usage: python all_experiments')
        print('Usage: python experiments.py from_pos to_pos att_string maxdepth_td [fairness_metric]')
        exit(-1)
    # get params
    from_pos = int(sys.argv[1])
    to_pos = int(sys.argv[2])
    att_string = sys.argv[3]
    maxdepth_td = int(sys.argv[4])
    fairness_metric = None if len(sys.argv)==5 else sys.argv[5]
    # set of columns
    attributes = utils.get_attributes(att_string)
    results = run_block(from_pos, to_pos, attributes, maxdepth_td, fairness_metric)
    filename = "results/exp/results_{}_{}_{}_{}".format(from_pos, to_pos, att_string, maxdepth_td) + ( ("_"+fairness_metric) if fairness_metric is not None else "") + ".pkl"
    pickle.dump(results, open( filename, "wb" )) 

def run_block(from_pos, to_pos, attributes, maxdepth_td, fairness_metric):
    # fixed params
    max_depth = 8
    min_pct = 0.05 # 5 percent of cases
    # load data
    X_train_s, X_test_s, y_train_s, y_test_s, X_train_t, X_test_t, y_train_t, y_test_t =\
         utils.load_ACSPublicCoverage(attributes)   
    # load distances among datasets (see calculate_distances.ipynb)
    dists = pickle.load( open("results/distances.pkl", "rb" ) )
    # restrict to subset
    dists = { k:{att:d for att, d in v.items() if att in attributes} for k, v in dists.items()}
    # run tests
    results = []
    for source in utils.states[from_pos:to_pos]:
        size = len(X_train_s[source])
        min_cases = int(size*min_pct)
        for target in utils.states:
            print('source', source, 'size', size, 'min_cases', min_cases, 'target', target, 'size', len(X_train_t[target]))
            _dict = dists[(source, target)]
            att_xw = min(_dict, key=lambda k: _dict[k]['d_y_cond_est']) if source != target else None
            das = [False, True] if source != target else [False]
            for da in das:
                start = time.time()
                clf, cm, cm_unprotected, cm_protected = \
                    run_test(X_train_s[source], y_train_s[source], X_test_t[target], y_test_t[target], 
                                   max_depth=max_depth, min_cases=min_cases, fairness_metric=fairness_metric,
                                   X_td=X_train_t[target] if source != target else None, 
                                   y_td=y_train_t[target] if source != target else None,
                                   da=da, 
                                   att_xw=att_xw if da else None,                                
                                   maxdepth_td=maxdepth_td if da else None)
                end = time.time()
                w_dist = clf.w_dist() if source != target else None
                elapsed = end-start
                results.append({'source':source, 'target':target, 'max_depth':max_depth, 'min_pct':min_pct, 
                                'da':da,
                                'fairness_metric':fairness_metric, 
                                'att_xw':att_xw, 'maxdepth_td':maxdepth_td,
                                'cm':cm, 'cm_unprotected':cm_unprotected, 
                                'cm_protected':cm_protected, 'attributes':attributes, 
                                'elapsed':elapsed, 'w_dist':w_dist})
                print(source, target, da, elapsed)         
    return results

def get_commands(att_string, maxdepth_td, fairness_metric=None):
    step = 5 # smaller -> more processes
    template = 'python experiments.py {} {} {} {}'
    commands = []
    for i in range(0, len(utils.states), step):
        command = template.format(i, i+step, att_string, maxdepth_td)
        if fairness_metric is not None:
          command += ' ' + fairness_metric
        commands.append(command)
    return commands

def all_experiments():
    att_string = 'subset1'
    attributes = utils.get_attributes(att_string)
    natts = len(attributes)
    print('Experimenting', att_string, 'with', natts, 'atts')
    all_cmd = []
    for maxdepth_td in {1,2,natts}:
        all_cmd.extend(get_commands(att_string, maxdepth_td))
        all_cmd.extend(get_commands(att_string, maxdepth_td, 'demographic_parity')) 
        all_cmd.extend(get_commands(att_string, maxdepth_td, 'true_positive_rate_parity')) 
    print('Running', len(all_cmd), 'processes')
    processes = [subprocess.Popen(cmd, shell=True) for cmd in all_cmd]
    # wait all child processes
    _ = [p.wait() for p in processes]

if __name__ == "__main__":
    main()
#
# EOF
#