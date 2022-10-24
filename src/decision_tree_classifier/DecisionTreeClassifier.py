import pandas as pd
import numpy as np

# local
from _entropy import *


class DecisionTreeClassifier(object):
    def __init__(self, max_depth):  # maybe feed df and then tka eth values from it and store columns | what about diff column types?
        self.depth = 0
        self.max_depth = max_depth
        self.trees = None
        # todo: feed vs store the external information?  maybe at the fit level?

    def fit(self, x: np.ndarray, y: np.ndarray, par_node={}, depth=0):

        # par_node = {} if None else par_node

        if par_node is None:
            return None
        elif len(y) == 0:
            return None
        elif self.all_same(y):
            return {'val': y[0]}
        elif depth >= self.max_depth:
            return None
        else:
            # todo: feed external info
            col, cutoff, entropy = self.find_best_split_of_all(x, y)  # find one split given an information gain
            y_left = y[x[:, col] < cutoff]
            y_right = y[x[:, col] >= cutoff]
            # todo: self.current_tree
            par_node = {'col': iris.feature_names[col],  # todo: this is dumb!!! feed df into class
                        'index_col': col,
                        'cutoff': cutoff,
                        'val': np.round(np.mean(y))
                        } # todo: add intermediate step here that keep track of the growing tree | maybe at counter also? or used depth!
            par_node['left'] = self.fit(x[x[:, col] < cutoff], y_left, {}, depth + 1)  # tricky to follow as the function is recursive within itself...
            par_node['right'] = self.fit(x[x[:, col] >= cutoff], y_right, {}, depth + 1)
            self.depth += 1
            self.trees = par_node  # this is only at the end... we're not tracking the path todo: see sklearn for this
            return par_node

    def find_best_split_of_all(self, x, y):
        col = None
        min_entropy = 1
        cutoff = None
        for i, c in enumerate(x.T):
            entropy, cur_cutoff = self.find_best_split(c, y)
            if entropy == 0:  # find the first perfect cutoff. Stop Iterating
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_entropy

    @staticmethod
    def find_best_split(col: np.ndarray, y: np.ndarray):
        min_entropy = 10
        n = len(y)
        for value in set(col):
            y_predict = col < value
            my_entropy = get_entropy(y_predict, y)
            if my_entropy <= min_entropy:
                min_entropy = my_entropy
                cutoff = value
        return min_entropy, cutoff

    @staticmethod
    def all_same(items):
        return all(x == items[0] for x in items)

    def predict(self, x):
        tree = self.trees  # todo where do we use the tree? this lines can go imo
        results = np.array([0] * len(x))
        for i, c in enumerate(x):
            results[i] = self._get_prediction(c)
        return results

    def _get_prediction(self, row):
        cur_layer = self.trees
        while cur_layer.get('cutoff'):
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                cur_layer = cur_layer['left']
            else:
                cur_layer = cur_layer['right']
        else:
            return cur_layer.get('val')


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()

    X = iris.data
    y = iris.target

    clf = DecisionTreeClassifier(max_depth=7)
    m = clf.fit(X, y)

    print(m)
