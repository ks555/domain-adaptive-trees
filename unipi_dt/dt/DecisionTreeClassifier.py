# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:24:42 2021

implementtion of decision tree adapted from 
https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea

@author: scott
"""
import numpy as np 
import math
from sklearn.datasets import load_iris
from scipy import stats

class DecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None
        self.depth = 0
        self.min_cases = 5


    def fit(self, X, y, cols):
        self.tree, self.depth = self.build(X, y, cols, depth=0)
    
    def build(self, X, y, cols, depth):
        """
        x: Feature set
        y: target variable
        depth: the depth of the current layer
        """
        
        if len(y) == 0:   # base case 2: no data in this group
            print('all the same')
            return None, 0
        
        if self.all_same(y):   # base case 3: all y is the same in this group
            return {'type':'all same leaf', 'val':y.iloc[0], 'dist': y.value_counts()}, 0
        if depth >= self.max_depth:   # base case 4: max depth reached 
            return {'type':'max depth leaf', 'val':stats.mode(y), 'dist': y.value_counts()}, 0
        # Recursively generate trees! 
        # find one split given an information gain 
        col, cutoff, gain = self.find_best_split_of_all(X, y)   
        if col is None: # no split improves
            return {'type':'no improvement leaf', 'val':stats.mode(y), 'dist': y.value_counts()}, 0
        y_left = y[X.iloc[:,col] < cutoff]  # left hand side data
        y_right = y[X.iloc[:,col] >= cutoff]  # right hand side data
        par_node = {'type':'split', 'gain':gain, 
                    'split_col': cols[col], 'split_idx':col,
                    'cutoff':cutoff, 'dist': y.value_counts()}  # save the information: distribution of y
        # generate tree for the left hand side data
        par_node['left'], dleft = self.build(X[X.iloc[:,col] < cutoff], y_left, cols, depth+1)   
        # right hand side trees
        par_node['right'], dright = self.build(X[X.iloc[:,col] >= cutoff], y_right, cols, depth+1)  
        return par_node, max(dleft, dright)+1
    
    def all_same(self, items):
        return all(x == items.iloc[0] for x in items)
    
    def find_best_split_of_all(self, X, y):
        best_gain = 0
        best_col = None
        best_cutoff = None
        for i, c in enumerate(X.columns):
            gain, cur_cutoff = self.find_best_split_continuous(X[c], y)
            if gain > best_gain:
                best_gain = gain
                best_cutoff = cur_cutoff
                best_col = i
        return best_col, best_cutoff, best_gain
    
    def find_best_split_continuous(self, x, y):
        best_gain = 0
        best_cutoff = None
        values = np.sort(x.unique())
        entropy_total = self.info(y)
        n_tot = len(x)
        for value in values:
            cond = x < value
            n_left = len(x[cond]) # check this KS
            if n_left < self.min_cases:
                continue
            n_right = n_tot - n_left
            if n_right < self.min_cases:
                continue
            entropy_left = self.info(y[cond])
            entropy_right = self.info(y[~cond])
            left_prop = n_left/n_tot
            right_prop = 1 - left_prop
            gain = entropy_total - left_prop*entropy_left - right_prop*entropy_right
            if gain > best_gain:
                best_gain = gain
                best_cutoff = value
        return best_gain, best_cutoff

    def find_best_split_discrete(self, x, y):
        best_gain = 0
        best_cutoff = None
        ## get unique values
        values = x.unique()
        ## sort by info?
        ## is this different than doing target encoding first??
        entropy_total = self.info(y)
        n_tot = len(x)
        for value in values:
            ## left node is x = value
            cond = x == value
            n_left = len(x[cond])
            # if n_left < self.min_cases:
            #     continue
            n_right = n_tot - n_left
            # if n_right < self.min_cases:
            #     continue
            entropy_left = self.info(y[cond])
            entropy_right = self.info(y[~cond])
            left_prop = n_left/n_tot
            right_prop = 1 - left_prop
            gain = entropy_total - left_prop*entropy_left - right_prop*entropy_right
            if gain > best_gain:
                best_gain = gain
                best_cutoff = value
        return best_gain, best_cutoff
    
    def info(self, y):
        vc = y.value_counts()
        tot = len(y)
        ent = 0
        for v in vc:
            prop = v/tot 
            ent -= prop*math.log2(prop)
        return ent
                                           
    def predict(self, x):
        results = np.array([0]*len(x))
        for i, c in enumerate(x):
            results[i] = self._get_prediction(c)
        return results
    
    def _get_prediction(self, row):
        cur_layer = self.trees
        print(len(row))
        while cur_layer.get('cutoff'):
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                cur_layer = cur_layer['left']
            else:
                cur_layer = cur_layer['right']
        else:
            return cur_layer.get('val')

    def entropy_func(self, c, n):
        """
        The math formula
        """
        return -(c*1.0/n)*math.log(c*1.0/n, 2)
    
    def entropy_cal(self, c1, c2):
        """
        Returns entropy of a group of data
        c1: count of one class
        c2: count of another class
        """
        if c1== 0 or c2 == 0:  # when there is only one class in the group, entropy is 0
            return 0
        return self.entropy_func(c1, c1+c2) + self.entropy_func(c2, c1+c2)
    
    # get the entropy of one big circle showing above
    def entropy_of_one_division(self, division): 
        """
        Returns entropy of a divided group of data
        Data may have multiple classes
        """
        s = 0
        n = len(division)
        classes = set(division)
        for c in classes:   # for each class, get entropy
            n_c = sum(division==c)
            e = n_c*1.0/n * self.entropy_cal(sum(division==c), sum(division!=c)) # weighted avg
            s += e
        return s, n
    
    # The whole entropy of two big circles combined
    def get_entropy(self, y_predict, y_real):
        """
        Returns entropy of a split
        y_predict is the split decision, True/Fasle, and y_true can be multi class
        """
        if len(y_predict) != len(y_real):
            print('They have to be the same length')
            return None
        n = len(y_real)
        s_true, n_true = self.entropy_of_one_division(y_real[y_predict]) # left hand side entropy
        s_false, n_false = self.entropy_of_one_division(y_real[~y_predict]) # right hand side entropy
        s = n_true*1.0/n * s_true + n_false*1.0/n * s_false # overall entropy, again weighted average
        return s