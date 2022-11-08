from abc import ABC, abstractmethod
from typing import Dict

# this is an example for future architecture


class TargetDomainProbabilities(ABC):
    def __init__(self, att, cutoff, path_to_att):
        self.att = att
        self.cutoff = cutoff
        self.path_to_att = path_to_att

    @abstractmethod
    def get_conditional_probability(self) -> float:
        pass

    def get_proportion_groupby(self, pop_data, group_column, threshold=None):
        if threshold is not None:
            pop_data.loc[pop_data[group_column] > threshold, 'above_threshold'] = True
            pop_data.loc[pop_data[group_column] <= threshold, 'above_threshold'] = False
            group_column = 'above_threshold'
        proportions = pop_data.groupby([group_column]).size()
        return proportions.values / len(pop_data)
