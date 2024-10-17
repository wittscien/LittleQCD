from abc import ABC, abstractmethod
from typing import List


class geometry(ABC):
    @abstractmethod
    def __update__(self, geometry: List[int]):
        pass


class QCD_geometry(geometry):
    def __init__(self, geometry: List[int]):
        self.T = geometry[0]
        self.X = geometry[1]
        self.Y = geometry[2]
        self.Z = geometry[3]
        self.Ns = 4
        self.Nc = 3
        self.Nl = 4

    def __update__(self, geometry: List[int]):
        self.T = geometry[0]
        self.X = geometry[1]
        self.Y = geometry[2]
        self.Z = geometry[3]
