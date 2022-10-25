# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:24:42 2021
implementtion of decision tree adapted from
https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea
@author: scott
"""

import numpy as np
import math
from scipy import stats
import pandas as pd
from typing import List
from pandas import DataFrame, Series
from typing import Dict


class DecisionTreeClassifier(object):
    def __init__(self, max_depth: int, cat: List[str] = None):
        self.max_depth = max_depth
        cat = [] if cat is None else cat
        self.cat = set(cat)
        self.depth = 0
        self.min_cases = 5
        self.tree = None
        self.you_are_here = (0, '')  # the root node: layer 0, side ''
        self.curr_tree = None
        self.prev_tree = None
        self.all_tress = []
        self.alpha = 1.0  # if alpha=1.0, da_prop is s_prop: non-da tree
        self.add_info = None

    def fit(self, X: DataFrame, y: Series, alpha: float = None, add_info: DataFrame = None, ):
        # domain adaptation params
        self.alpha = alpha if alpha is not None else alpha
        self.add_info = add_info if add_info is not None else add_info
        # build tree
        self.tree, self.depth = self.build(X, y, depth=0)

    def build(self, X: DataFrame, y: Series, depth: int):
        """
        x: Feature set
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
            return {'type': 'leaf', 'val': y.iloc[0], 'tot': leny, 'dist': y.value_counts(sort=False) / leny}, 0
        # base case 4: max depth reached
        if depth >= self.max_depth:
            return {'type': 'leaf', 'val': stats.mode(y).mode[0], 'tot': leny, 'dist': y.value_counts(sort=False) / leny}, 0
        # recursive case:
        col, cutoff, gain = self.find_best_split_of_all(X, y)  # find one split given an information gain
        if col is None:  # no split improves
            return {'type': 'leaf', 'val': stats.mode(y).mode[0], 'tot': leny, 'dist': y.value_counts(sort=False) / leny}, 0
        if col in self.cat:  # split for cont. vars; all-but for cat. vars
            cond = X[col] == cutoff
        else:
            cond = X[col] < cutoff
        X_left = X[cond]    # < value
        X_right = X[~cond]  # >= value
        y_left = y[cond]    # left hand side data
        y_right = y[~cond]  # right hand side data
        print('split', col, gain, cutoff)
        par_node = {'type': 'split', 'gain': gain, 'split_col': col, 'cutoff': cutoff, 'tot': leny,
                    'dist': y.value_counts(sort=False) / leny}  # save the information: distribution of y

        # track (sub)trees
        if depth > 0:
            self.prev_tree = self.curr_tree.copy()
            self.all_tress.append(self.prev_tree)  # todo: append as a tuple where you connect to previous tree
        self.curr_tree = self._store_subtree(par_node)

        # generate tree for the left hand side data
        self.you_are_here = (depth + 1, 'lhs')
        par_node['left'], dleft = self.build(X_left, y_left, depth + 1)
        # temp_l_tree, dleft = self.build(X_left, y_left, depth + 1) # todo check bcs order affects the recursive search
        # par_node['left'] = temp_l_tree
        # self.curr_tree['left'] = self._store_subtree(temp_l_tree)
        # generate tree for the right hand side data
        self.you_are_here = (depth + 1, 'rhs')
        par_node['right'], dright = self.build(X_right, y_right, depth + 1)
        # temp_r_tree, dright = self.build(X_right, y_right, depth + 1)
        # par_node['right'] = temp_r_tree
        # self.curr_tree['right'] = self._store_subtree(temp_r_tree)
        # del temp_l_tree, temp_r_tree

        return par_node, max(dleft, dright) + 1

    def _store_subtree(self, par_node: Dict):
        if par_node is None:
            print('no data in this group')
            return None
        else:
            # return `split` info
            if par_node['type'] == 'split':
                if par_node['split_col'] in self.cat:
                    l_rule = '== {cutoff}'.format(cutoff=par_node['cutoff'])
                    r_rule = '!= {cutoff}'.format(cutoff=par_node['cutoff'])
                else:
                    l_rule = '< {cutoff}'.format(cutoff=par_node['cutoff'])
                    r_rule = '>= {cutoff}'.format(cutoff=par_node['cutoff'])
                sub_tree = {'type': par_node['type'],
                            'split_col': par_node['split_col'],
                            'cutoff': par_node['cutoff'],
                            'lhs': l_rule, 'rhs': r_rule}
            # return `leaf` info
            else:
                sub_tree = {'type': par_node['type']}
        return sub_tree

    # todo
    # def _get_current_path(self):
    #     current_path = []
    #     while self.you_are_here[0] > 0:
    #         prev_layer = self.you_are_here[0] - 1

    def all_same(self, items):
        return all(x == items.iloc[0] for x in items)

    def find_best_split_of_all(self, X, y):

        if self.curr_tree is None:
            current_path = ''  # all Xs are candidates
            print('root node')
        else:
            print('in layer {layer}'.format(layer=self.you_are_here[0]))
            print('go back {n} step(s)'.format(n=self.you_are_here[0] - 1))
            current_path = self.curr_tree['split_col'] + ' ' + self.curr_tree[self.you_are_here[1]]
        print(current_path)  # todo: verify | maybe we don't need the full path if we keep track in parallel?
        # todo: maybe grow paths here as a dict?
        best_gain = 0
        best_col = None
        best_cutoff = None
        for c in X.columns:
            ## proposal - target encode discrete prior to building DT
            ## send mappings of value / encoded value to DT
            ## to calculate gain, if c is a 'treated' attribute, map back to values to include stats data adjustment
            is_cat = c in self.cat
            gain, cur_cutoff = self.find_best_split_attribute(X[c], y, is_cat)
            if gain > best_gain:
                best_gain = gain
                best_cutoff = cur_cutoff
                best_col = c
        return best_col, best_cutoff, best_gain

    def find_best_split_attribute(self, x, y, is_cat):
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
            if n_left < self.min_cases:  # todo: might introduce bias by for first min values
                continue
            n_right = n_tot - n_left
            if n_right < self.min_cases:
                continue
            # todo: should here what's the current path in my tree: use self.need to know current path here self.subtree={}
            # get entropy H(T|A=a) given current path
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

    # @staticmethod
    def info(self, y: Series):  # for H(T) part of IG todo: not true.. firts split is uncondtional
        vc = y.value_counts()
        tot = len(y)
        ent = 0
        for v in vc:
            prop = v / tot
            ent -= prop * math.log2(prop)
        return ent

    def da_info(self, y: Series, alpha: float, t_prop: float):  # for H(T|A=a) part of IG todo: t_y
        vc = y.value_counts()
        tot = len(y)
        ent = 0
        for v in vc:
            s_prop = v / tot  # what if we get t_prop like we get s_prop? via partition
            da_prop = (alpha*s_prop + (1 - alpha)*t_prop)  # todo: do we have to loop over t_prop using v? yes!
            ent -= da_prop * math.log2(da_prop)
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


if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_boston
    iris = load_iris()
    # boston = load_boston()

    from tools import *
    from pprint import pprint

    # X = iris.data
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    X['y'] = iris.target
    y = X['y']
    X = X[iris.feature_names]

    X_train, X_test, y_train, y_test = split_data(X, y)

    clf = DecisionTreeClassifier(max_depth=7, cat=['test', 'me'])
    clf.fit(X_train, y_train)
    pprint(clf.tree)
    # create adjusted splitting criterion
    predictions = clf.predict(X_test)
    accuracy = print_scores(y_test, predictions['prediction'])
