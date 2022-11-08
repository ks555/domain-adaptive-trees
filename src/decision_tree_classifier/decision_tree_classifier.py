import numpy as np
import math
import yapftests.yapf_test
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

        self._you_are_here = ()
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
        print('.', depth, leny)
        # base case 2: no data in this group
        if leny == 0:
            print('all the same')
            return None, 0
        # base case 3: all y is the same in this group
        if self.all_same(y):
            return {'type': 'leaf',
                    'val': y.iloc[0], 'tot': leny, 'dist': y.value_counts(sort=False) / leny}, 0
        # base case 4: max depth reached
        if depth >= self.max_depth:
            return {'type': 'leaf',
                    'val': stats.mode(y).mode[0], 'tot': leny, 'dist': y.value_counts(sort=False) / leny}, 0
        # recursive case:
        col, cutoff, gain = self.find_best_split_of_all(X, y)  # find one split given an information gain
        if col is None:  # no split improves
            return {'type': 'leaf',
                    'val': stats.mode(y).mode[0], 'tot': leny, 'dist': y.value_counts(sort=False) / leny}, 0
        if col in self.cat:  # split for cont. vars; all-but for cat. vars
            cond = X[col] == cutoff
        else:
            cond = X[col] < cutoff
        X_left = X[cond]    # < value
        X_right = X[~cond]  # >= value
        y_left = y[cond]    # left hand side data
        y_right = y[~cond]  # right hand side data
        print('split', col, gain, cutoff)
        par_node = {'type': 'split',
                    'gain': gain, 'split_col': col, 'cutoff': cutoff, 'tot': leny, 'dist': y.value_counts(sort=False) / leny}

        # the beauty of recursion (?) | delete print statements later
        prev_path = self._current_path.copy()
        print(prev_path)
        # generate tree for the left hand side data
        self._you_are_here = (col, cutoff, 'left')
        print(self._you_are_here)
        self._current_path = prev_path + [self._you_are_here]
        print(self._current_path)
        par_node['left'], dleft = self.build(X_left, y_left, depth + 1)

        # generate tree for the right hand side data
        self._you_are_here = (col, cutoff, 'right')
        print(self._you_are_here)
        self._current_path = prev_path + [self._you_are_here]
        print(self._current_path)
        par_node['right'], dright = self.build(X_right, y_right, depth + 1)

        return par_node, max(dleft, dright) + 1

    @staticmethod
    def all_same(items):
        return all(x == items.iloc[0] for x in items)

    def get_target_weights(self, current_path: List[Tuple[str, object, str]], X: DataFrame) -> Dict:
        target_weights = {}
        if len(current_path) > 0:
            print('provide conditional weights based on current path')  # use current_path TODO
        else:
            # you're at the root node
            print('provide the unconditional weights')  # use current_path TODO
        # use X to build a target_weights dictionary based on the domain source (?)
        # e.g., target_weights[c] = prob[c] where c in X.columns and prob is from the target domain info
        #   can have target_weights[c] = None

        for c in X.columns:
            target_weights[c] = None

        return target_weights

    def find_best_split_of_all(self, X: DataFrame, y: Series):
        # you are here
        if len(self._you_are_here) > 0:  # delete later
            print("you're at the {side} hs of {col}".format(side=self._you_are_here[2], col=self._you_are_here[0]))
        else:
            print("root node")
        # extract the target weights as a dictionary
        if self.running_da_cov_shift:
            print('current path:')
            print(self._current_path)
            target_weights = self.get_target_weights(self._current_path, X)
        else:
            target_weights = None

        best_gain = 0
        best_col = None
        best_cutoff = None
        for c in X.columns:
            is_cat = c in self.cat
            # domain-adaptive information gain
            if self.running_da_cov_shift:
                # proceed with da if we have a target weight for att c given the current path
                if target_weights[c] is not None:
                    gain, cur_cutoff = self.da_find_best_split_attribute(X[c], y, is_cat, self.alpha, target_weights[c])
                # otherwise: alpha=1.0 to use only source domain info; t_weight=0.0 to keep function arch
                else:
                    gain, cur_cutoff = self.da_find_best_split_attribute(X[c], y, is_cat, 1.0, 0.0)
            # standard information gain
            else:
                gain, cur_cutoff = self.find_best_split_attribute(X[c], y, is_cat)
            if gain > best_gain:
                best_gain = gain
                best_cutoff = cur_cutoff
                best_col = c
        return best_col, best_cutoff, best_gain

    def find_best_split_attribute(self, x: Series, y: Series, is_cat: bool):
        best_gain = 0
        best_cutoff = None
        if is_cat:
            values = x.unique()
        else:
            values = np.sort(x.unique())
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

    def da_find_best_split_attribute(self, x: Series, y: Series, is_cat: bool, alpha: float, t_weight: float):
        best_gain = 0
        best_cutoff = None
        if is_cat:
            values = x.unique()
        else:
            values = np.sort(x.unique())
        # get entropy H(T)
        entropy_total = self.da_info(y, t_weight, alpha)
        n_tot = len(x)
        for value in values:
            cond = x == value if is_cat else x < value
            n_left = sum(cond)
            if n_left < self.min_cases:
                continue
            n_right = n_tot - n_left
            if n_right < self.min_cases:
                continue
            # get entropy H(T|A=a) todo: do we update t_weight under A=a?
            entropy_left = self.da_info(y[cond], t_weight, alpha)     # < value
            entropy_right = self.da_info(y[~cond], t_weight, alpha)   # >= value
            left_prop = n_left / n_tot
            right_prop = 1 - left_prop
            # Information Gain: H(T) - H(T|A=a)
            gain = entropy_total - left_prop * entropy_left - right_prop * entropy_right
            if gain > best_gain:
                best_gain = gain
                best_cutoff = value
        return best_gain, best_cutoff

    @staticmethod
    def info(y: Series):
        vc = y.value_counts()
        tot = len(y)
        ent = 0
        for v in vc:
            prop = v / tot
            ent -= prop * math.log2(prop)
        return ent

    @staticmethod
    def da_info(y_source: Series, p_target: float, alpha: float):
        vc = y_source.value_counts()
        tot = len(y_source)
        ent = 0
        for v in vc:
            prop = v / tot
            da_w = (alpha * prop + (1 - alpha) * p_target)
            ent -= da_w * math.log2(prop)  # todo: or da_w goes in I() too?
        return ent

    # def da_find_best_split_attribute_v0(self, x: Series, y: Series, is_cat: bool, x_td: Series, y_td: Series, alpha: float):
    #     best_gain = 0
    #     best_cutoff = None
    #     if is_cat:
    #         values = x.unique()
    #     else:
    #         values = np.sort(x.unique())
    #     # get entropy H(T)
    #     entropy_total = self.da_info_v0(self.y_values, y, y_td, alpha)
    #     n_tot = len(x)
    #     for value in values:
    #         cond = x == value if is_cat else x < value
    #         cond_for_td = x_td == value if is_cat else x_td < value  # index must align for x_td and y_td
    #         n_left = sum(cond)
    #         if n_left < self.min_cases:
    #             continue
    #         n_right = n_tot - n_left
    #         if n_right < self.min_cases:
    #             continue
    #         # get entropy H(T|A=a)
    #         entropy_left = self.da_info_v0(self.y_values, y[cond], y_td[cond_for_td], self.alpha)     # < value
    #         entropy_right = self.da_info_v0(self.y_values, y[~cond], y_td[~cond_for_td], self.alpha)  # >= value
    #         left_prop = n_left / n_tot
    #         right_prop = 1 - left_prop
    #         # Information Gain: H(T) - H(T|A=a)
    #         gain = entropy_total - left_prop * entropy_left - right_prop * entropy_right
    #         if gain > best_gain:
    #             best_gain = gain
    #             best_cutoff = value
    #     return best_gain, best_cutoff
    #
    # @staticmethod
    # def da_info_v0(y_values: List[int], y_source: Series, y_target: Series, alpha: float):
    #     # v0: requires y_s and y_t, expecting both to be pd.Series - the ideal case
    #     # source domain
    #     vc_s = y_source.value_counts()
    #     tot_s = len(y_source)
    #     # target domain
    #     vc_t = y_target.value_counts()
    #     tot_t = len(y_target)
    #     # domain-adaptive entropy
    #     ent = 0
    #     for val in y_values:  # the values are the (potential) index in value_count
    #         if val in vc_s.index:
    #             prop_s = vc_s[val] / tot_s
    #         else:
    #             prop_s = 0.0
    #         if val in vc_t.index:
    #             prop_t = vc_t[val] / tot_t
    #         else:
    #             prop_t = 0.0
    #         da_prop = (alpha * prop_s + (1 - alpha) * prop_t)
    #         if da_prop > 0.0:
    #             ent -= da_prop * math.log2(da_prop)
    #     return ent

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


# if __name__ == "__main__":
#     from sklearn.datasets import load_iris, load_boston
#     import folktables as ft
#
#     iris = load_iris()
#     # boston = load_boston()
#
#     from src.utils import split_data, print_scores
#     from pprint import pprint
#     import src.utils
#
#     # X = iris.data
#     X = pd.DataFrame(iris.data, columns=iris.feature_names)
#     X['y'] = iris.target
#     y = X['y']
#     X = X[iris.feature_names]
#
#     X_train, X_test, y_train, y_test = split_data(X, y)
#
#     clf = DecisionTreeClassifier(max_depth=7, cat=['test', 'me'])  # todo: old version: delete
#     clf.fit(X_train, y_train)
#     pprint(clf.tree)
#     # create adjusted splitting criterion
#     predictions = clf.predict(X_test)
#     accuracy = print_scores(y_test, predictions['prediction'])


    #### uncomment below (and comment above) to run folktables task
    
    # acs_data = src.utils.load_folktables_data(['CA'], '2017', '1-Year', 'person')
    # # load task - just makes numpy arrays of features, labels, protected group category for given task
    # # acs_data = utils.load_data(['AL', 'CA'], '2017', '1-Year', 'person')
    # features, labels, group = src.utils.load_task(acs_data, ft.ACSPublicCoverage)
    #
    # X = pd.DataFrame(features, columns=ft.ACSPublicCoverage.features)
    # X['y'] = labels
    # y = X['y']
    # X = X[ft.ACSPublicCoverage.features]
    #
    # X_train, X_test, y_train, y_test = split_data(X, y)
    #
    # clf = DecisionTreeClassifier(max_depth=7, cat=['test', 'me'])
    # clf.fit(X_train, y_train)
    # # pprint(clf.tree)
    # # create adjusted splitting criterion
    # predictions = clf.predict(X_test)
    # accuracy = print_scores(y_test, predictions['prediction'])