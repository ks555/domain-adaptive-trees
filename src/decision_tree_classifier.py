# -*- coding: utf-8 -*-
"""
Decision tree classifier
"""

import numpy as np
import math
from typing import Dict, List
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
    def __init__(self, max_depth: int, min_cases: int = 5, *args, **kwargs):
        self.classes_ = None  
        self.max_depth = max_depth
        self.depth = 0
        self.min_cases = min_cases
        self.cat = None
        self.tree = None
        self._current_path = []
        self.alpha = 0
        self.X_td = None # target domain knowledge
        self.y_td = None # target class knowledge (used only for W() metrics)
        self.att_td = None # target domain attribute for probability estimation

    def fit(self, X: DataFrame, y: Series, cat_atts: List[str] = None,
            alpha: float = None, X_td: object = None, y_td: object = None, 
            att_td: str = None, maxdepth_td = None):
        # class values
        self.classes_ = np.sort(y.unique())
        # categorical attributes
        self.cat = set() if cat_atts is None else set(cat_atts)
        # correction parameters
        self.alpha = alpha
        # target domain knowledge
        self.X_td = X_td
        # target class knowledge (used only for W() metrics)
        self.y_td = y_td
        # max depth in target domain knowledge
        self.maxdepth_td = len(X_td.columns) if maxdepth_td is None and X_td is not None else maxdepth_td
        # unique values of attributes
        self.unique = { c:np.sort(X[c].unique()) for c in X.columns.to_list()}
        if X_td is None: # no domain knowledge
            self.att_td = None 
        elif att_td is not None: # att_td is given
            self.att_td = att_td
        elif alpha is not None: # att_td is not given but needed
            c = min(self.unique, key=self.unique.get)
            self.att_td = c # use att with min number of values
            print("att_td", c)
        # current decision path
        self._current_path = []
        # build tree
        self.tree, self.depth = self.build(X, y, depth=0)       

    def build(self, X: DataFrame, y: Series, depth: int):
        """
        X: Feature set
        y: target variable
        depth: the depth of the current layer
        """
        leny = len(y)
        assert(leny>0)
        
        # get domain knowledge
        self.target_weights, self.dyn_alpha = self.get_target_weights(self._current_path)
        # distribution given the classes_
        if self.X_td is not None and self.att_td is not None: # domain knowledge
            self.dist = self.distribution_est(y, X[self.att_td], self.target_weights[self.att_td])
        else:
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
        col, cutoff, gain = self.find_best_split_of_all(X, y)  # find one split given an information gain
        if col is None:  # no split improves
            return {'type': 'leaf', 'tot': leny, 'dist': self.dist}, 0
        if col in self.cat:  # split for cont. vars; all-but for cat. vars
            cond = X[col] == cutoff
        else:
            cond = X[col] <= cutoff
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

    def get_target_weights(self, current_path, cols=None) -> Dict:
        # Result is a Dict mapping column names to a Dictionary mapping values to float such that
        # for c in self.cat and value in X[c].unique(), it holds that
        #      target_weights[c][value] is the target P(c=value)
        # for c NOT in self.cat and value in X[c].unique(), it holds that
        #      target_weights[c][value] is the target P(c<value)
        target_weights = dict()
        if self.X_td is None:
            return target_weights, 1

        curatts=set()
        allatts=set()
        cond=None
        # map current_path to the subset of X_td satisfying current_path
        for att, thr, di in current_path:
            if att not in self.X_td.columns:
                continue
            if att not in curatts:
                if len(curatts)==self.maxdepth_td:
                    allatts.add(att)
                    continue
                curatts.add(att)
                allatts.add(att)
            con = self.X_td[att] == thr if att in self.cat else self.X_td[att] <= thr
            if di == 'right':
                con = ~con
            cond = con if cond is None else cond & con
        #
        X_td_current = self.X_td[cond] if cond is not None else self.X_td
        if cols is None:
            cols = X_td_current.columns.to_list()
        for c in cols:
            freqs = X_td_current[c].value_counts(normalize=True).to_dict()
            values = self.unique[c]
            if c in self.cat:
                target_weights[c] = {value: (freqs[value] if value in freqs else 0) for value in values}
            else:
                target_weights[c] = {value: sum(freqs[v] for v in freqs if v <= value) for value in values}

        alpha = 0 if len(allatts)==0 else 1-len(curatts)/len(allatts)
        return target_weights, alpha

    def find_best_split_of_all(self, X: DataFrame, y: Series):
        # extract the target weights as a dictionary
        #target_weights = self.get_target_weights(X) if self.running_da_cov_shift else None
        best_gain = 0
        best_col = None
        best_cutoff = None
        self.entropy_total = self.info_props(self.dist)
        for c in X.columns.to_list():
            is_cat = c in self.cat
            # domain-adaptive information gain
            if self.alpha is not None and c in self.target_weights:
                # proceed with da if we have a target weight for att c given the current path
                gain, cur_cutoff = self.da_find_best_split_attribute(c, X[c], y, 
                             is_cat, self.target_weights[c], X[self.att_td], 
                             self.target_weights[self.att_td])
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
            is_cat: bool, t_weight: Dict, xt_d: Series, t_weight_td: Dict):
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
            target_weights_left, _ = self.get_target_weights(self._current_path+[(col, value, 'left')], [self.att_td])
            props_left = self.distribution_est(y_left, xt_d[cond_left], target_weights_left[self.att_td])
            entropy_left = self.info_props(props_left)
            
            cond_right = ~cond_left
            y_right = y[cond_right]
            #props_right = self.distribution(y_right)
            target_weights_right, _ = self.get_target_weights(self._current_path+[(col, value, 'right')], [self.att_td])
            props_right = self.distribution_est(y_right, xt_d[cond_right], target_weights_right[self.att_td])
            entropy_right = self.info_props(props_right)

            alpha = self.alpha # static
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

    def distribution(self, y: Series):
        return epmf(y, self.classes_)
    
    # should be called as distribution_est(y, X[self.att_td], self.target_weights[self.att_td])
    def distribution_est(self, y, x, t_weight):
        props = None
        prev = 0
        values = sorted(t_weight.keys())
        for value in values:
            cond = x == value #if is_cat else x < value
            ycond = y[cond]
            ydistr = self.distribution(ycond)
            weight = t_weight[value]
            if self.att_td not in self.cat:
                weight -= prev
                prev = t_weight[value]
            distr = ydistr * weight
            props = distr if props is None else props + distr
        tot = np.sum(props)
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
