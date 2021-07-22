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

class DecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.depth = 0
        self.max_depth = max_depth
    
    def fit(self, x, y, cols, par_node={}, depth=0):
        """
        x: Feature set
        y: target variable
        par_node: will be the tree generated for this x and y. 
        depth: the depth of the current layer
        """
        if par_node is None:   # base case 1: tree stops at previous level
            print(f"parnode is none. \n{par_node}")
            return None
        elif len(y) == 0:   # base case 2: no data in this group
            print(f"no data. \n{par_node}")
            return None
        elif self.all_same(y):   # base case 3: all y is the same in this group
            return {'val':y[0]}
        elif depth >= self.max_depth:   # base case 4: max depth reached 
            print(f"max depth.")
            return {'val':np.round(np.mean(y))}
        else:   # Recursively generate trees! 
            # find one split given an information gain 
            col, cutoff, entropy = self.find_best_split_of_all(x, y)   
            y_left = y[x[:, col] < cutoff]  # left hand side data
            y_right = y[x[:, col] >= cutoff]  # right hand side data
            par_node = {'col': cols[col], 'index_col':col,
                        'cutoff':cutoff,
                       'val': np.round(np.mean(y))}  # save the information 
            # generate tree for the left hand side data
            print(f"col: {cols[col]}, depth {depth+1}")
            par_node['left'] = self.fit(x[x[:, col] < cutoff], y_left, cols, {}, depth+1)   
            # right hand side trees
            par_node['right'] = self.fit(x[x[:, col] >= cutoff], y_right, cols, {}, depth+1)  
            self.depth += 1   # increase the depth since we call fit once
            self.trees = par_node  
            return par_node
    
    def all_same(self, items):

        return all(x == items[0] for x in items)
    
    def find_best_split_of_all(self, x, y):
        col = None
        min_entropy = 1
        cutoff = None
        for i, c in enumerate(x.T):
            entropy, cur_cutoff = self.find_best_split(c, y)
            if entropy == 0:    # find the first perfect cutoff. Stop Iterating
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff

        return col, cutoff, min_entropy
    
    def find_best_split(self, col, y):
        min_entropy = 10
        n = len(y)
        for value in set(col):
            y_predict = col < value
            my_entropy = self.get_entropy(y_predict, y)
            if my_entropy <= min_entropy:
                min_entropy = my_entropy
                cutoff = value
        return min_entropy, cutoff
                                           
    def predict(self, x):
        results = np.array([0]*len(x))
        for i, c in enumerate(x):
            print(f'row {i}')
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