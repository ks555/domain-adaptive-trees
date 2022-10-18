
#  todo in tree\_classes.py
# CRITERIA_CLF = {"gini": _criterion.Gini,
#                 "entropy": _criterion.Entropy,
#                 "da_entropy": DomainAdaptiveEntropy}

# sklearn runs the criteria via cython: build a basic one and hopefully do the same following the same architecture?
# these classes are actually intended for packaging the inputs and sending them through _criterion.pyx
# we can create ouw own inefficient one for Entropy and DomainAdaptiveEntropy: todo create single _criterion.py
#   that work purely in python

# imports
import builtins as __builtins__
import numpy as np


# functions

def __pyx_unpickle_Enum(*args, **kwargs): # real signature unknown
    pass


# classes

class Criterion(object):
    """
    Interface for impurity criteria.

        This object stores methods on how to calculate how good a split is using
        different metrics.
    """

    def __getstate__(self, *args, **kwargs):  # real signature unknown
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce_cython__(self, *args, **kwargs):  # real signature unknown
        pass

    def __setstate_cython__(self, *args, **kwargs):  # real signature unknown
        pass

    def __setstate__(self, *args, **kwargs):  # real signature unknown
        pass

    __pyx_vtable__ = None  # (!) real value is '<capsule object NULL at 0x000002455E113690>'


class ClassificationCriterion(Criterion):
    """ Abstract criterion for classification. """

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        pass

    __pyx_vtable__ = None  # (!) real value is '<capsule object NULL at 0x000002455E1136C0>'


class Entropy(ClassificationCriterion):
    """
    Cross Entropy impurity criterion.

        This handles cases where the target is a classification taking values
        0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
        then let

            count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k)

        be the proportion of class k observations in node m.

        The cross-entropy is then defined as

            cross-entropy = -\sum_{k=0}^{K-1} count_k log(count_k)
    """

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    __pyx_vtable__ = None  # (!) real value is '<capsule object NULL at 0x000002455E113210>'


class DomainAdaptiveEntropy(ClassificationCriterion):  # TODO?
    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    __pyx_vtable__ = None  # (!) real value is '<capsule object NULL at 0x000002455E113210>'


# todo... add the rest?


# variables with complex values

__loader__ = None # (!) real value is '<_frozen_importlib_external.ExtensionFileLoader object at 0x000002455E113880>'

__spec__ = None # (!) real value is "ModuleSpec(name='sklearn.tree._criterion', loader=<_frozen_importlib_external.ExtensionFileLoader object at 0x000002455E113880>, origin='C:\\\\anaconda\\\\envs\\\\fair_ml\\\\lib\\\\site-packages\\\\sklearn\\\\tree\\\\_criterion.cp38-win_amd64.pyd')"

__test__ = {}
