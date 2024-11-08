from opt_einsum import contract
from lqcd.io.backend import get_backend
from lqcd.core.geometry import QCD_geometry



class DiracOperator:
    def __init__(self, geometry: QCD_geometry):
        self.geometry = geometry
        self.operator = 0


