# -*- coding: utf-8 -*-
"""
Decision tree classifier
"""

# global imports
import numpy as np
import math
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import wasserstein_distance

# calculate empirical distribution of y wrt domain of values
def epmf(y, values):
    d = y.value_counts(sort=False, normalize=True).to_dict()
    dist = np.array([d[c] if c in d else 0 for c in values])
    return dist

# decision tree classifier
class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth: int, min_cases: int = 5):
        # max tree depth
        self.max_depth = max_depth
        # min number of cases to split a node
        self.min_cases = min_cases
        # array of class values
        self.classes_ = None  
        # current tree depth
        self.depth = 0
        # set of categorical attributes
        self.cat = None
        # actual decision tree
        self.tree = None
        # current path in the decision tree
        self._current_path = []
        # target domain knowledge
        self.X_td = None 
        # target class domain knowledge (used only for W() metrics)
        self.y_td = None 
        # target domain attribute for probability estimation (X_w in Eq. 14)
        self.att_xw = None 

    # fit a decision tree on given training set
    def fit(self, 
            X: DataFrame, y: Series, # training set
            cat_atts = [],  # categorical attributes
            X_td = None, # target domain attributes
            y_td: object = None, # target domain class
            att_xw = None, # target domain attribute for probability estimation (X_w in Eq. 14)
            maxdepth_td = None # max joint distribution target knowledge 
            ):
        # class values
        self.classes_ = np.sort(y.unique())
        # categorical attributes
        self.cat = set(cat_atts)
        # target domain knowledge
        self.X_td = X_td
        # target class knowledge (used only for W() metrics)
        self.y_td = y_td
        # max depth in target domain knowledge
        self.maxdepth_td = len(X_td.columns) if maxdepth_td is None and X_td is not None else maxdepth_td
        # unique values of attributes (stored once for efficiency)
        self.unique = { c:np.sort(X[c].unique()) for c in X.columns.to_list()}
        if X_td is None: # no domain knowledge
            self.att_xw = None # force self.att_xw to None
        elif att_xw is not None: # att_td is given
            self.att_xw = att_xw
        # TBD
        else: # att_td is not given but needed
            c = min(self.unique, key=self.unique.get)
            self.att_xw = c # use att with min number of values
            print("att_td", c)
        # current decision path
        self._current_path = []
        # build tree
        self.tree, self.depth = self.build(X, y, depth=0)       

    # build decision tree
    def build(self, X: DataFrame, y: Series, depth: int):
        leny = len(y)
        assert(leny>0)        
        # get joint target domain knowledge and dynamic alpha (=fraction of splits in current path not in target domain)
        self.target_probs, self.dyn_alpha = self.get_target_probs(self._current_path)
        # distribution of the classes_
        if self.X_td is not None and self.att_xw is not None: 
            # with target domain knowledge Eq. 14
            self.dist = self.distribution_est(y, X[self.att_xw], self.target_probs[self.att_xw])
        else:
            # without
            self.dist = self.distribution(y)
        
        # base case: all y is the same in this group     
        if np.count_nonzero(self.dist)==1:
            #print('a leaf!')
            return {'type': 'leaf', 'tot': leny, 'dist': self.dist}, 0
        # base case: max depth reached
        if depth >= self.max_depth:
            #print('a leaf!')
            return {'type': 'leaf', 'tot': leny, 'dist': self.dist}, 0
        # recursive case:
        col, cutoff, gain = self.find_best_split_of_all(X, y)  
        if col is None:  # no split improves
            return {'type': 'leaf', 'tot': leny, 'dist': self.dist}, 0
        # split data rowss
        cond = (X[col] == cutoff) if col in self.cat else (X[col] <= cutoff)
        #print('split on', col, gain, cutoff, len(X), sum(cond))
        par_node = {'type': 'split', 'gain': gain, 'split_col': col, 'cutoff': cutoff, 
                    'tot': leny, 'dist': self.dist}

        prev_path = self._current_path.copy()
        # generate tree for the left hand side data
        self._current_path = prev_path + [(col, cutoff, 'left')]
        X_left = X[cond]    # < value
        y_left = y[cond]    # left hand side data
        par_node['left'], dleft = self.build(X_left, y_left, depth + 1)

        # generate tree for the right hand side data
        self._current_path = prev_path + [(col, cutoff, 'right')]
        X_right = X[~cond]  # >= value
        y_right = y[~cond]  # right hand side data
        par_node['right'], dright = self.build(X_right, y_right, depth + 1)

        return par_node, max(dleft, dright) + 1

    # return target domain probabilities (\hat{P}_T(X=x|phi') in Sect. 3.1)
    # phi' = given current_path, possibly limited to self.maxdepth_td attributes
    # X belongs to a list of columns (None=all columns) 
    # result as a dictionary 
    def get_target_probs(self, current_path, cols=None):
        target_probs = dict()
        # require domain knowledge
        if self.X_td is None:
            return target_probs, 1
        # self.maxdepth_td is the maximum joint distribution domain knowledge
        curatts=set() # attributes in the domain knowledge currently considered
        allatts=set() # all attributes currently considered
        cond=None # elements in the target domain satisfying all the conditions so far
        # map current_path to the subset of X_td satisfying current_path conditions
        for att, thr, di in current_path:
            # for each condition in the current_path
            if att not in self.X_td.columns:
                continue
            if att not in curatts:
                # attribute not yet in the joint distribution domain knowledge
                if len(curatts)==self.maxdepth_td:
                    allatts.add(att)
                    # maximum size reached, don't add att tot the joint distribution domain knowledge
                    continue
                # add att to the joint distribution domain
                curatts.add(att)
                allatts.add(att)
            # elements in the target domain satisfying the condition
            con = self.X_td[att] == thr if att in self.cat else self.X_td[att] <= thr
            if di == 'right':
                con = ~con
            # elements in the target domain satisfying all the conditions so far
            cond = con if cond is None else cond & con
        # target domain knowledge restricted to conditions in the current_path
        X_td_current = self.X_td[cond] if cond is not None else self.X_td
        # if cols is None, all the columns in the target domain knowledge are considered
        if cols is None:
            cols = X_td_current.columns.to_list()
        # for each column
        for c in cols:
            # pmf of the column in target domain knowledge
            freqs = X_td_current[c].value_counts(normalize=True).to_dict()
            # all the unique values of the column 
            values = self.unique[c]
            # Eq. 8
            if c in self.cat:
                target_probs[c] = {value: (freqs[value] if value in freqs else 0) for value in values}
            else:
                target_probs[c] = {value: sum(freqs[v] for v in freqs if v <= value) for value in values}
        # proportion of attributes not in the joint target domain
        alpha = 0 if len(allatts)==0 else 1-len(curatts)/len(allatts)
        return target_probs, alpha

    def find_best_split_of_all(self, X: DataFrame, y: Series):
        # extract the target weights as a dictionary
        #target_weights = self.get_target_probs(X) if self.running_da_cov_shift else None
        best_gain = 0
        best_col = None
        best_cutoff = None
        self.entropy_total = self.info_props(self.dist)
        for c in X.columns.to_list():
            is_cat = c in self.cat
            # domain-adaptive information gain
            if c in self.target_probs:
                # proceed with da if we have a target weight for att c given the current path
                gain, cur_cutoff = self.da_find_best_split_attribute(c, X[c], y, 
                             is_cat, self.target_probs[c], X[self.att_xw], 
                             self.target_probs[self.att_xw])
            else:
                # standard information gain
                gain, cur_cutoff = self.find_best_split_attribute(X[c], y, is_cat)
            if gain > best_gain:
                best_gain = gain
                best_cutoff = cur_cutoff
                best_col = c
        #print(best_col, best_cutoff)
        return best_col, best_cutoff, best_gain

    def find_best_split_attribute(self, x: Series, y: Series, is_cat: bool):
        best_gain = 0
        best_cutoff = None
        values = np.sort(x.unique())
        # get entropy H(T)
        #entropy_total = self.info(y)
        #print('et', entropy_total)
        n_tot = len(x)
        for value in values:
            cond = x == value if is_cat else x <= value
            n_left = sum(cond)
            if n_left < self.min_cases:
                continue
            n_right = n_tot - n_left
            if n_right < self.min_cases:
                continue
            # get entropy H(T|A=a)
            entropy_left = self.info(y[cond])    # < value
            entropy_right = self.info(y[~cond])  # >= value
            left_prop = n_left / n_tot
            right_prop = 1 - left_prop
            # Information Gain: H(T) - H(T|A=a)
            gain = self.entropy_total - left_prop * entropy_left - right_prop * entropy_right
            #print(value, gain, -gain+entropy_total, left_prop, entropy_left, entropy_right, y[~cond].value_counts() / len(y[~cond]))
            if gain > best_gain:
                best_gain = gain
                best_cutoff = value
        #print('ret', best_gain, best_cutoff)
        return best_gain, best_cutoff

    def da_find_best_split_attribute(self, col:str, x: Series, y: Series,  
            is_cat: bool, t_weight, xt_d, t_weight_td):
        best_gain = 0
        best_cutoff = None
        entropy_total = self.entropy_total
        n_tot = len(x)
        prev = 0
        for value in np.sort(x.unique()):
            weight = t_weight[value]
            if not is_cat:
                weight -= prev
                prev = t_weight[value]
            cond_left = x == value if is_cat else x <= value
            n_left = sum(cond_left)
            if n_left < self.min_cases or weight==0:
                continue
            n_right = n_tot - n_left
            if n_right < self.min_cases or weight==1:
                continue
            y_left = y[cond_left]
            #props_left = self.distribution(y_left)
            target_weights_left, _ = self.get_target_probs(self._current_path+[(col, value, 'left')], [self.att_xw])
            props_left = self.distribution_est(y_left, xt_d[cond_left], target_weights_left[self.att_xw])
            entropy_left = self.info_props(props_left)
            
            cond_right = ~cond_left
            y_right = y[cond_right]
            #props_right = self.distribution(y_right)
            target_weights_right, _ = self.get_target_probs(self._current_path+[(col, value, 'right')], [self.att_xw])
            props_right = self.distribution_est(y_right, xt_d[cond_right], target_weights_right[self.att_xw])
            entropy_right = self.info_props(props_right)

            alpha = self.dyn_alpha # dynamic
            prop_left = alpha * (n_left / n_tot) + (1 - alpha) * t_weight[value]
            #if alpha != 0:
            #    print(alpha, prop_left, t_weight[value])
            prop_right = 1 - prop_left
            gain = entropy_total - prop_left * entropy_left - prop_right * entropy_right
            if gain > best_gain:
                best_gain = gain
                best_cutoff = value
        #print('ret', best_gain, best_cutoff)
        return best_gain, best_cutoff

    # empirical distribution
    def distribution(self, y):
        return epmf(y, self.classes_)
    
    # empirical distribution estimate - Eq. 13 in the paper
    def distribution_est(self, y, x_w_data, x_w_target_probs):
        # estimate of P_T(Y|phi) as in Eq. 13
        props = None
        prev = 0
        values = sorted(x_w_target_probs.keys())
        # for each value in X_w
        for value in values:
            cond = x_w_data == value 
            ycond = y[cond]
            # P_S(Y|X_w=x)
            ydistr = self.distribution(ycond)
            # P_T(X_w=x)
            weight = x_w_target_probs[value]
            # for continuous, we need to subtract t_weight[value_previous]
            #     because t_weight[value] is P_T(X_w <= x)
            if self.att_xw not in self.cat:
                weight -= prev
                prev = x_w_target_probs[value]
            distr = ydistr * weight
            # list of addends in Eq. 14
            props = distr if props is None else props + distr
        tot = np.sum(props)
        # Normalize to ensure that total sum is 1
        return props/tot if tot>0 else self.distribution(y)

    @staticmethod
    def info(y: Series):
        props = y.value_counts() / len(y)
        ent = 0
        for prop in props:
            ent -= prop * math.log2(prop) if prop > 0 else 0
        return ent

    @staticmethod
    def info_props(props: Series):
        ent = 0
        for prop in props:
            ent -= prop * math.log2(prop) if prop > 0 else 0
        return ent

    def predict_proba(self, X):
        res = []
        for _, r in X.iterrows():
            res.append(self._get_prediction(r))
        return np.stack(res, axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def _get_prediction(self, row):
        cur_layer = self.tree
        while cur_layer.get('cutoff'):
            col = cur_layer['split_col']
            cutoff = cur_layer['cutoff']
            left = row[col] == cutoff if col in self.cat else row[col] <= cutoff
            cur_layer = cur_layer['left'] if left else cur_layer['right']
        else:
            return cur_layer.get('dist')

    def w_dist(self):
        tot_td = len(self.y_td)
        return self._w_dist(self.tree, self.X_td, self.y_td)/tot_td

    def _w_dist(self, tree, X, y):
        if tree.get('cutoff'):
            col = tree['split_col']
            cutoff = tree['cutoff']
            if col in self.cat:
                cond = X[col] == cutoff
            else:
                cond = X[col] <= cutoff
            X_left = X[cond]
            y_left = y[cond]
            w1 = self._w_dist(tree['left'], X_left, y_left)
            X_right = X[~cond]
            y_right = y[~cond]
            w2 = self._w_dist(tree['right'], X_right, y_right)           
            return w1 + w2
        tot_td = len(y)
        true_dist = self.distribution(y)
        return tot_td*wasserstein_distance(tree['dist'], true_dist)

#
# EOF
#
