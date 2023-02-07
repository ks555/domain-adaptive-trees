# -*- coding: utf-8 -*-
"""
Batch test run
"""

# global imports
import sys
import time
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# local imports
import utils

if len(sys.argv) not in {4, 5}:
    # if set, threshold_opt uses the fairness correction of ThresholOptimizer
    # threshold_opt can be "demographic_parity" or "true_positive_rate_parity"
    print('Usage: python batch from_pos to_pos subset [threshold_opt]')
    exit(-1)

# get params
from_pos = int(sys.argv[1])
to_pos = int(sys.argv[2])
subset_string = sys.argv[3]
t_o = None if len(sys.argv)==4 else sys.argv[4] #
# other fixed params
max_depth = 8
min_pct = 0.05 # 1 percent of cases
maxdepth_td = 8

# set of columns
subset = utils.get_subset(subset_string)
# data
X_train_s, X_test_s, y_train_s, y_test_s, X_train_t, X_test_t, y_train_t, y_test_t =\
     utils.load_ACSPublicCoverage(subset)

# load distances among datasets (see calculate_distances.ipynb)
dists = pickle.load( open("results/distances.pkl", "rb" ) )
# restrict to subset
dists = { k:{att:d for att, d in v.items() if att in subset} for k, v in dists.items()}

# run tests
results = []
for source in utils.states[from_pos:to_pos]:
    size = len(X_train_s[source])
    min_cases = int(size*min_pct)
    for target in utils.states:
        print('source', source, 'size', size, 'min_cases', min_cases, 'target', target, 'size', len(X_train_t[target]))
        _dict = dists[(source, target)]
        # target domain knowledge: select attribute with smallest marginal conditional distance
        att_td = min(_dict, key=lambda k: _dict[k]['d_y_cond']) if source != target else None
        alphas = [None, 0] if source != target else [None]
        for alpha in alphas:
            start = time.time()
            clf, cm, cm_unprotected, cm_protected = \
                utils.run_test(X_train_s[source], y_train_s[source], X_test_t[target], y_test_t[target], 
                               X_td=X_train_t[target] if source != target else None, 
                               alpha=alpha, 
                               y_td=y_train_t[target],
                               max_depth=max_depth, 
                               min_cases=min_cases, 
                               att_td=att_td if alpha is not None else None, 
                               t_o=t_o,
                               maxdepth_td=maxdepth_td if alpha is not None else None)
            results.append({'source':source, 'target':target, 'max_depth':max_depth, 'alpha':alpha,
                            'subset':subset, 't_o':t_o, 'cm':cm, 'cm_unprotected':cm_unprotected, 
                            'maxdepth_td':maxdepth_td, 'min_pct':min_pct,'att_td':att_td,
                            'clf':clf, 'cm_protected':cm_protected})
            end = time.time()
            print(end-start, source, target, alpha)
            
filename = "results_{}_{}_{}".format(from_pos, to_pos, subset_string) + ( ("_"+t_o) if t_o is not None else "") + ".pkl"
pickle.dump(results, open( filename, "wb" )) 