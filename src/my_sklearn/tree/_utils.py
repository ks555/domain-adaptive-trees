from sklearn.tree import DecisionTreeClassifier
from typing import List, Dict
from pandas import DataFrame


def get_current_path(subtree: DecisionTreeClassifier, ) -> Dict[str, str]:
    # TODO
    pass


def get_target_probability(target_df: DataFrame, parent: str, grandparents: List[str] = None, ) -> Dict[str, float]:

    target_probabilities = {}

    if grandparents:
        print("estimating conditional probabilities for {attribute}".format(attribute=parent))
        # TODO

    else:
        print("estimating probabilities for {attribute}".format(attribute=parent))
        # TODO

    return target_probabilities
