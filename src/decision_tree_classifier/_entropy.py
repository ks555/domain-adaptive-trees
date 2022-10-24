import math


# entropy functions from Medium post
def entropy_func(c, n):
    """
    The math formula
    """
    return -(c*1.0/n)*math.log(c*1.0/n, 2)


def entropy_cal(c1, c2):
    """
    Returns entropy of a group of data
    c1: count of one class
    c2: count of another class
    """
    if c1 == 0 or c2 == 0:  # when there is only one class in the group, entropy is 0
        return 0
    return entropy_func(c1, c1 + c2) + entropy_func(c2, c1 + c2)


# get the entropy of one big circle showing above
def entropy_of_one_division(division):
    """
    Returns entropy of a divided group of data
    Data may have multiple classes
    """
    s = 0
    n = len(division)
    classes = set(division)
    for c in classes:   # for each class, get entropy
        n_c = sum(division==c)
        e = n_c*1.0/n * entropy_cal(sum(division==c), sum(division!=c))  # weighted avg
        s += e
    return s, n


# The whole entropy of two big circles combined
def get_entropy(y_predict, y_real):
    """
    Returns entropy of a split
    y_predict is the split decision, True/False, and y_true can be multi class
    """
    if len(y_predict) != len(y_real):
        print('They have to be the same length')
        return None
    n = len(y_real)
    s_true, n_true = entropy_of_one_division(y_real[y_predict])  # left hand side entropy
    s_false, n_false = entropy_of_one_division(y_real[~y_predict])  # right hand side entropy
    s = n_true*1.0/n * s_true + n_false*1.0/n * s_false  # overall entropy, again weighted average
    return s
