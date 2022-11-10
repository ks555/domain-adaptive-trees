import numpy as np
import math
from scipy import stats
import pandas as pd
from typing import Dict, List, Tuple
from pandas import DataFrame, Series


class DecisionTreeClassifier(object):
    def __init__(self, max_depth: int, *args, **kwargs):
        self.max_depth = max_depth
        self.depth = 0
        self.min_cases = 5
        self.cat = None
        self.tree = None
        self._current_path = []
        self.running_da_cov_shift = False  # covariate shift
        self.running_da_con_shift = False  # concept shift
        self.alpha = 0.5
        self.X_td = None
        self.y_td = None
        self.y_values = None  # values are based on the source domain todo: might have to delete

    def fit(self, X: DataFrame, y: Series, cat_atts: List[str] = None,
            alpha: float = None, X_td: object = None, y_td: object = None, ):
        # instance-specific params
        cat = [] if cat_atts is None else cat_atts
        self.cat = set(cat)
        self.y_values = y.value_counts().index.to_list()
        # domain adaptation params
        if X_td is not None:
            self.running_da_cov_shift = True
        # if X_td is not None and y_td is not None:
        #     self.running_da_con_shift = True
        self.alpha = alpha if alpha is not None else alpha
        self.X_td = X_td if X_td is not None else X_td
        self.y_td = y_td if y_td is not None else y_td
        # build tree
        self.tree, self.depth = self.build(X, y, depth=0)

    def build(self, X: DataFrame, y: Series, depth: int):
        """
        X: Feature set
        y: target variable
        depth: the depth of the current layer
        """
        leny = len(y)
        print('node: depth=', depth, 'ncases=', leny, 'path:', self._current_path)
        # base case 2: no data in this group
        if leny == 0:
            print('no data')
            return None, 0
        # base case 3: all y is the same in this group
        if self.all_same(y):
            print('a leaf!')
            return {'type': 'leaf',
                    'val': y.iloc[0], 'tot': leny, 'dist': y.value_counts(sort=False) / leny}, 0
        # base case 4: max depth reached
        if depth >= self.max_depth:
            print('a leaf!')
            return {'type': 'leaf',
                    'val': y.mode()[0], 'tot': leny, 'dist': y.value_counts(sort=False) / leny}, 0
        # recursive case:
        col, cutoff, gain = self.find_best_split_of_all(X, y)  # find one split given an information gain
        if col is None:  # no split improves
            return {'type': 'leaf',
                    'val': y.mode()[0], 'tot': leny, 'dist': y.value_counts(sort=False) / leny}, 0
        if col in self.cat:  # split for cont. vars; all-but for cat. vars
            cond = X[col] == cutoff
        else:
            cond = X[col] < cutoff
        print('split on', col, gain, cutoff)
        par_node = {'type': 'split',
                    'gain': gain, 'split_col': col, 'cutoff': cutoff, 'tot': leny, 'dist': y.value_counts(sort=False) / leny}

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

    @staticmethod
    def all_same(items):
        return all(x == items.iloc[0] for x in items)

    def get_target_weights(self, X: DataFrame) -> Dict:
        # Result is a Dict mapping column names to a Dictionary mapping values to float such that
        # for c in self.cat and value in X[c].unique(), it holds that
        #      target_weights[c][value] is the target P(c=value)
        # for c NOT in self.cat and value in X[c].unique(), it holds that
        #      target_weights[c][value] is the target P(c<value)
        target_weights = dict()
        if self.X_td is None:
            target_weights = {c: None for c in X.columns}
            return target_weights

        current_path = self._current_path
        # map current_path to the subset of X_td satisfying current_path
        cond = None
        for att, thr, di in current_path:
            con = self.X_td[att] == thr if att in self.cat else self.X_td[att] < thr
            if di == 'right':
                con = ~con
            cond = con if cond is None else cond & con
        #
        X_td_current = self.X_td[cond] if cond is not None else self.X_td                
        for c in X.columns:
            freqs = X_td_current[c].value_counts(normalize=True).to_dict()
            values = X[c].unique()
            if c in self.cat:
                target_weights[c] = {value: (freqs[value] if value in freqs else 0) for value in values}
            else:
                target_weights[c] = {value: sum(freqs[v] for v in freqs if v < value) for value in values}

        return target_weights

    def find_best_split_of_all(self, X: DataFrame, y: Series):
        # extract the target weights as a dictionary
        target_weights = self.get_target_weights(X) if self.running_da_cov_shift else None
        best_gain = 0
        best_col = None
        best_cutoff = None
        for c in X.columns:
            is_cat = c in self.cat
            # domain-adaptive information gain
            if self.running_da_cov_shift and target_weights[c] is not None:
                # proceed with da if we have a target weight for att c given the current path
                gain, cur_cutoff = self.da_find_best_split_attribute(X[c], y, is_cat, self.alpha, target_weights[c])
            else:
                # standard information gain
                gain, cur_cutoff = self.find_best_split_attribute(X[c], y, is_cat)
            if gain > best_gain:
                best_gain = gain
                best_cutoff = cur_cutoff
                best_col = c
        return best_col, best_cutoff, best_gain

    def find_best_split_attribute(self, x: Series, y: Series, is_cat: bool):
        best_gain = 0
        best_cutoff = None
        values = x.unique() if is_cat else np.sort(x.unique())
        # get entropy H(T)
        entropy_total = self.info(y)
        n_tot = len(x)
        for value in values:
            cond = x == value if is_cat else x < value
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
            gain = entropy_total - left_prop * entropy_left - right_prop * entropy_right
            if gain > best_gain:
                best_gain = gain
                best_cutoff = value
        return best_gain, best_cutoff

    def da_find_best_split_attribute(self, x: Series, y: Series, is_cat: bool, alpha: float, t_weight: Dict):
        best_gain = 0
        best_cutoff = None
        values = x.unique() if is_cat else np.sort(x.unique())
        # get entropy H(T)
        entropy_total = self.info(y)
        n_tot = len(x)
        for value in values:
            cond = x == value if is_cat else x < value
            n_left = sum(cond)
            if n_left < self.min_cases:
                continue
            n_right = n_tot - n_left
            if n_right < self.min_cases:
                continue
            entropy_left = self.info(y[cond])     # < value
            entropy_right = self.info(y[~cond])   # >= value
            # here it goes the mitigation strategy
            # t_weight[value] is the target P(X=value) if is_cat else P(X<value)
            left_prop = alpha*(n_left/n_tot) + (1-alpha)*t_weight[value]
            right_prop = 1 - left_prop
            # Information Gain: H(T) - H(T|A=a)
            gain = entropy_total - left_prop * entropy_left - right_prop * entropy_right
            if gain > best_gain:
                best_gain = gain
                best_cutoff = value
        return best_gain, best_cutoff

    @staticmethod
    def info(y: Series):
        props = y.value_counts()/len(y)
        ent = 0
        for prop in props:
            ent -= prop * math.log2(prop) if prop>0 else 0
        return ent

    def predict(self, X):
        # results = np.array([0]*len(X))
        # results = pd.DataFrame(columns=['pred', 'prob'])
        results = pd.DataFrame({'prediction': pd.Series([], dtype='float'),
                                'probability': pd.Series([], dtype='float')})
        for i, r in enumerate(X.itertuples(index=False)):
            results.loc[i] = self._get_prediction(X.iloc[i])
        return results

    def _get_prediction(self, row):
        cur_layer = self.tree
        while cur_layer.get('cutoff'):
            col = cur_layer['split_col']
            cutoff = cur_layer['cutoff']
            is_cat = col in self.cat
            left = row[col] == cutoff if is_cat else row[col] < cutoff
            cur_layer = cur_layer['left'] if left else cur_layer['right']
        else:
            return [cur_layer.get('val'), cur_layer.get('dist')]

#
# EOF
#
