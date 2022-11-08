from abc import ABC, abstractmethod
from typing import Dict


class TargetDomainProbabilities(ABC):
    def __init__(self, att, cutoff, path_to_att):
        self.att = att
        self.cutoff = cutoff
        self.path_to_att = path_to_att

    @abstractmethod
    def get_conditional_probability(self) -> float:
        pass
