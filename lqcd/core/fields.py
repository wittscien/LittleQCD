from typing import List
from lqcd.io.backend import get_backend
from lqcd.core.geometry import QCD_geometry



class fermion:
    def __init__(self, geometry: QCD_geometry):
        self.geometry = geometry
        self.field = xp.zeros((geometry.T, geometry.X, geometry.Y, geometry.Z, geometry.Nl, geometry.Nc, geometry.Nc), dtype=xp.complex128)

    def __getitem__(self, pos):
        # The SU(3) matrix at [t, x, y, z, mu]
        return self.field[pos[0], pos[1], pos[2], pos[3], pos[4]]
    
    def __setitem__(self, pos, mat):
        self.field[pos[0], pos[1], pos[2], pos[3], pos[4]] = mat

    def clean(self):
        self.field = 0

    def __eq__(self, other):
        if isinstance(other, LatticeFermion):
            return self.field == other.field
        else:
            return TypeError
    
    def __add__(self, other):
        if isinstance(other, LatticeFermion):
            result = LatticeFermion(self.geometry)
            result.field = self.field + other.field
            return result
        else:
            return TypeError

    def __sub__(self, other):
        if isinstance(other, LatticeFermion):
            result = LatticeFermion(self.geometry)
            result.field = self.field - other.field
            return result
        else:
            return TypeError

xp = get_backend()
