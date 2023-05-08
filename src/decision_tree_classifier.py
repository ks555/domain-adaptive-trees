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
        # set of categorical attributes
        self.cat = None
        # array of class values
        self.classes_ = None  
        # actual decision tree
        self.tree = None
        # decision tree depth
        self.depth = None
        # current tree depth
        self.depth = 0
        # current path in the decision tree
        self._current_path = []
        # target domain knowledge
        self.X_td = None 
        # target class domain knowledge (used only for W() metrics)
        self.y_td = None 
        # target domain attribute for probability estimation (X_w in Eq. 13)
        self.att_xw = None 
        # max depth in target domain knowledge
        self.maxdepth_td = None
        # unique values of attributes (stored once for efficiency)
        self.unique = None
        # variables used during tree building
        self.da = False
        self.target_probs = None
        self.dyn_alpha = None
        self.dist = None

    # fit a decision tree on given training set
    def fit(self, 
            X: DataFrame, y: Series,  # training set
            cat_atts=[],              # categorical attributes
            da=False,                 # use domain adaptation
            X_td=None,                # target domain attributes
            y_td: object = None,      # target domain class
            att_xw=None,              # target domain attribute for probability estimation (X_w in Eq. 13)
            maxdepth_td=None          # max joint distribution target knowledge
            ):
        # class values
        self.classes_ = np.sort(y.unique())
        # categorical attributes
        self.cat = set(cat_atts)
        # adopt domain adaptation
        self.da = da
        # target domain knowledge
        self.X_td = X_td
        # target class knowledge (used only for W() metrics)
        self.y_td = y_td
        # target domain attribute for probability estimation (X_w in Eq. 13)
        self.att_xw = att_xw
        # max depth in target domain knowledge
        self.maxdepth_td = len(X_td.columns) if maxdepth_td is None and X_td is not None else maxdepth_td
        # unique values of attributes (stored once for efficiency)
        self.unique = {c: np.sort(X[c].unique()) for c in X.columns.to_list()}
        if da and ((att_xw is None) or (X_td is None)) :  # domain knowledge
            raise RuntimeError('X_td and att_xw are mandatory')
        # current decision path
        self._current_path = []
        # build tree
        self.tree, self.depth = self.build(X, y, depth=0)       

    # build decision tree
    def build(self, X, y, depth):
        leny = len(y)
        assert(leny > 0)
        # get joint target domain knowledge and dynamic alpha (=fraction of splits in current path not in target domain)
        self.target_probs, self.dyn_alpha = self.get_target_probs(self._current_path)
        # distribution of the classes_
        if self.da: 
            # with target domain knowledge Eq. 14
            self.dist = self.distribution_est(y, X[self.att_xw], self.target_probs[self.att_xw])
        else:
            # with source data
            self.dist = self.distribution(y)
        
        # base case: all y is the same in this group     
        if np.count_nonzero(self.dist) == 1:
            return {'type': 'leaf', 'tot': leny, 'dist': self.dist}, 0
        # base case: max depth reached
        if depth >= self.max_depth:
            return {'type': 'leaf', 'tot': leny, 'dist': self.dist}, 0
        # recursive case:
        col, cutoff, gain = self.find_best_split_of_all(X, y)  
        if col is None:  # no split improves
            return {'type': 'leaf', 'tot': leny, 'dist': self.dist}, 0
        # split data rows
        cond = (X[col] == cutoff) if col in self.cat else (X[col] <= cutoff)
        #print('split on', col, gain, cutoff, len(X), sum(cond))
        par_node = {'type': 'split', 'gain': gain, 'split_col': col, 'cutoff': cutoff, 'tot': leny, 'dist': self.dist}

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
    # phi' = given path, possibly limited to self.maxdepth_td attributes
    # X belongs to a list of columns (None=all columns) 
    # result as a dictionary 
    def get_target_probs(self, path, cols=None):
        target_probs = dict()
        # require domain knowledge
        if not self.da:
            return target_probs, 1
        # self.maxdepth_td is the maximum joint distribution domain knowledge
        curatts = set()  # attributes in the domain knowledge currently considered
        allatts = set()  # all attributes currently considered
        cond = None  # elements in the target domain satisfying all the conditions so far
        # map path to the subset of X_td satisfying path conditions
        for att, thr, di in path:
            # for each condition in the path
            if att not in self.X_td.columns:
                continue
            if att not in curatts:
                # attribute not yet in the joint distribution domain knowledge
                if len(curatts) == self.maxdepth_td:
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
        # target domain knowledge restricted to conditions in the path
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
            # Eq. 7
            if c in self.cat:
                target_probs[c] = {value: (freqs[value] if value in freqs else 0) for value in values}
            else:
                target_probs[c] = {value: sum(freqs[v] for v in freqs if v <= value) for value in values}
        # proportion of attributes not in the joint target domain
        alpha = 0 if len(allatts) == 0 else 1-len(curatts)/len(allatts)
        return target_probs, alpha

    # find best split
    def find_best_split_of_all(self, X, y):
        best_gain = 0
        best_col = None
        best_cutoff = None
        entropy_total = self.info_props(self.dist)
        # iterate over columns
        for c in X.columns.to_list():
            is_cat = c in self.cat
            # if we have a target weight for att c 
            if c in self.target_probs:
                # proceed with domain adaptive information gain calculation
                gain, cur_cutoff = self.da_find_best_split_attribute(X[c], y, c, is_cat, entropy_total, self.target_probs[c], X[self.att_xw])
            else:
                # standard information gain calculation
                gain, cur_cutoff = self.find_best_split_attribute(X[c], y, is_cat, entropy_total)
            # test if gain improves
            if gain > best_gain:
                best_gain = gain
                best_cutoff = cur_cutoff
                best_col = c
        return best_col, best_cutoff, best_gain

    # standard information gain calculation
    def find_best_split_attribute(self, x, y, is_cat, entropy_total):
        best_gain = 0
        best_cutoff = None
        values = np.sort(x.unique())
        n_tot = len(x)
        # for each value, consider splitting on that value
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
            gain = entropy_total - left_prop * entropy_left - right_prop * entropy_right
            # test if gain improves
            if gain > best_gain:
                best_gain = gain
                best_cutoff = value
        return best_gain, best_cutoff

    # domain adaptive information gain calculation    
    def da_find_best_split_attribute(self, x, y, x_name, is_cat, entropy_total, target_probs_c, x_xw):
        best_gain = 0
        best_cutoff = None
        n_tot = len(x)
        prev = 0
        # for each value, consider splitting on that value
        for value in np.sort(x.unique()):
            # P_T(X=value)
            weight = target_probs_c[value] 
            if not is_cat:
                # continuous probs are cumulative, compute the pmf
                weight -= prev
                prev = target_probs_c[value]
            # split condition
            cond_left = x == value if is_cat else x <= value
            n_left = sum(cond_left)
            if n_left < self.min_cases or weight == 0:
                continue
            n_right = n_tot - n_left
            if n_right < self.min_cases or weight == 1:
                continue
            # class left
            y_left = y[cond_left]
            # target domain knowledge on att_xs on the left branch
            target_weights_left, _ = self.get_target_probs(self._current_path+[(x_name, value, 'left')], [self.att_xw])
            # P(Y|phi,X=t) used in entropy calculation next
            props_left = self.distribution_est(y_left, x_xw[cond_left], target_weights_left[self.att_xw])
            # H(Y|phi,X=t) in Eq. 4
            entropy_left = self.info_props(props_left)
            
            # right condition
            cond_right = ~cond_left
            # class right
            y_right = y[cond_right]
            # target domain knowledge on att_xs on the right branch
            target_weights_right, _ = self.get_target_probs(self._current_path+[(x_name, value, 'right')], [self.att_xw])
            # P(Y|phi,X\=t) used in entropy calculation next
            props_right = self.distribution_est(y_right, x_xw[cond_right], target_weights_right[self.att_xw])
            # H(Y|phi,X\=t) in Eq. 4
            entropy_right = self.info_props(props_right)
            # dynamic alpha
            alpha = self.dyn_alpha 
            # Eq. 8 and 9
            prop_left = alpha * (n_left / n_tot) + (1 - alpha) * target_probs_c[value]
            prop_right = 1 - prop_left
            # Eq. 4 and 5
            gain = entropy_total - prop_left * entropy_left - prop_right * entropy_right
            # test if gain improves
            if gain > best_gain:
                best_gain = gain
                best_cutoff = value
        return best_gain, best_cutoff

    # empirical distribution
    def distribution(self, y):
        return epmf(y, self.classes_)
    
    # empirical distribution estimate - Eq. 12 in the paper
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
            # for continuous, we need to subtract t_weight[value_previous] because t_weight[value] is P_T(X_w <= x)
            if self.att_xw not in self.cat:
                weight -= prev
                prev = x_w_target_probs[value]
            distr = ydistr * weight
            # list of addends in Eq. 13
            props = distr if props is None else props + distr
        tot = np.sum(props)
        # Normalize to ensure that total sum is 1
        return props/tot if tot > 0 else self.distribution(y)

    # Information given values
    @staticmethod
    def info(y):
        props = y.value_counts() / len(y)
        ent = 0
        for prop in props:
            ent -= prop * math.log2(prop) if prop > 0 else 0
        return ent

    # Information given pmf
    @staticmethod
    def info_props(props):
        ent = 0
        for prop in props:
            ent -= prop * math.log2(prop) if prop > 0 else 0
        return ent

    # Predicting probabilities for a test set
    def predict_proba(self, X):
        res = []
        for _, r in X.iterrows():
            res.append(self._get_prediction(r))
        return np.stack(res, axis=0)

    # Predicting class for a test set
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    # Predicting probabilities for a single row
    def _get_prediction(self, row):
        cur_layer = self.tree
        while cur_layer.get('cutoff'):
            col = cur_layer['split_col']
            cutoff = cur_layer['cutoff']
            left = row[col] == cutoff if col in self.cat else row[col] <= cutoff
            cur_layer = cur_layer['left'] if left else cur_layer['right']
        else:
            return cur_layer.get('dist')

    # Average Wasserstein distance between true target and leaves estimates
    def w_dist(self):
        tot_td = len(self.y_td)
        return self._w_dist(self.tree, self.X_td, self.y_td)/tot_td

    def _w_dist(self, tree, X, y):
        if tree.get('cutoff'):
            # split nodes
            col = tree['split_col']
            cutoff = tree['cutoff']
            if col in self.cat:
                cond = X[col] == cutoff
            else:
                cond = X[col] <= cutoff
            X_left = X[cond]
            y_left = y[cond]
            # left branch
            w1 = self._w_dist(tree['left'], X_left, y_left)
            X_right = X[~cond]
            y_right = y[~cond]
            # right branch
            w2 = self._w_dist(tree['right'], X_right, y_right)           
            return w1 + w2
        # a leaf
        tot_td = len(y)
        true_dist = self.distribution(y)
        return tot_td*wasserstein_distance(tree['dist'], true_dist)

#
# EOF
#
