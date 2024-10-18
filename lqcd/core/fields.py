from opt_einsum import contract
from lqcd.io.backend import get_backend
from lqcd.core.geometry import QCD_geometry


class Field:
    def __init__(self, geometry: QCD_geometry):
        self.geometry = geometry
        self.field = 0

    def clean(self):
        self.field = 0

    def __eq__(self, other):
        if isinstance(other, Field):
            return self.field == other.field
        else:
            return TypeError

    def __add__(self, other):
        if isinstance(other, Field):
            result = Field(self.geometry)
            result.field = self.field + other.field
            return result
        else:
            return TypeError

    def __sub__(self, other):
        if isinstance(other, Field):
            result = Field(self.geometry)
            result.field = self.field - other.field
            return result
        else:
            return TypeError

    def __repr__(self):
        return f"{self.field}"


class Gauge(Field):
    def __init__(self, geometry: QCD_geometry):
        xp = get_backend()
        super().__init__(geometry)
        self.field = xp.zeros((geometry.T, geometry.X, geometry.Y, geometry.Z, geometry.Nl, geometry.Nc, geometry.Nc), dtype=xp.complex128)

    def __getitem__(self, pos):
        # The SU(3) matrix at [t, x, y, z], [t, x, y, z, mu] or the point at [t, x, y, z, mu, a, b]
        if len(pos) in [4, 5, 7]:
            return self.field[pos]
        else:
            raise ValueError("Invalid number of indices")

    def __setitem__(self, pos, mat):
        if len(pos) in [4, 5, 7]:
            self.field[pos] = mat
        else:
            raise ValueError("Invalid number of indices")


class Fermion(Field):
    def __init__(self, geometry: QCD_geometry):
        xp = get_backend()
        super().__init__(geometry)
        self.field = xp.zeros((geometry.T, geometry.X, geometry.Y, geometry.Z, geometry.Ns, geometry.Nc), dtype=xp.complex128)

    def __getitem__(self, pos):
        # The spin-color matrix at [t, x, y, z], [t, x, y, z, s] or the point at [t, x, y, z, s, c]
        if len(pos) in [4, 5, 6]:
            return self.field[pos]
        else:
            raise ValueError("Invalid number of indices")

    def __setitem__(self, pos, mat):
        if len(pos) in [4, 5, 6]:
            self.field[pos] = mat
        else:
            raise ValueError("Invalid number of indices")


class Propagator(Field):
    def __init__(self, geometry: QCD_geometry):
        xp = get_backend()
        super().__init__(geometry)
        self.field = xp.zeros((geometry.T, geometry.X, geometry.Y, geometry.Z, geometry.Ns, geometry.Ns, geometry.Nc, geometry.Nc), dtype=xp.complex128)

    def __getitem__(self, pos):
        # The spin-color matrix at [t, x, y, z]
        if len(pos) in [4]:
            return self.field[pos]
        else:
            raise ValueError("Invalid number of indices")

    def __setitem__(self, pos, mat):
        if len(pos) in [4]:
            self.field[pos] = mat
        else:
            raise ValueError("Invalid number of indices")


class Gamma:
    def __init__(self, i):
        self.mat = self.gamma(i)

    def gamma(self, i):
        xp = get_backend()

        g0 = xp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=complex)
        g1 = xp.array([[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]], dtype=complex)
        g2 = xp.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=complex)
        g3 = xp.array([[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]], dtype=complex)
        g4 = xp.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=complex)
        g5 = xp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=complex)

        if i == 0:  # identity
            g = g0
        elif i == 1:  # gamma1
            g = g1
        elif i == 2:  # gamma2
            g = g2
        elif i == 3:  # gamma3
            g = g3
        elif i == 4:  # gamma4
            g = g4
        elif i == 5:  # gamma5
            g = g5
        elif i == 6:  # -gamma1*gamma4*gamma5 (gamma2*gamma3)
            g = xp.matmul(g2, g3)
        elif i == 7:  # -gamma2*gamma4*gamma5 (gamma3*gamma1)
            g = xp.matmul(g3, g1)
        elif i == 8:  # -gamma3*gamma4*gamma5 (gamma1*gamma2)
            g = xp.matmul(g1, g2)
        elif i == 9:  # gamma1*gamma4
            g = xp.matmul(g1, g4)
        elif i == 10:  # gamma2*gamma4
            g = xp.matmul(g2, g4)
        elif i == 11:  # gamma3*gamma4
            g = xp.matmul(g3, g4)
        elif i == 12:  # gamma1*gamma5
            g = xp.matmul(g1, g5)
        elif i == 13:  # gamma2*gamma5
            g = xp.matmul(g2, g5)
        elif i == 14:  # gamma3*gamma5
            g = xp.matmul(g3, g5)
        elif i == 15:  # gamma4*gamma5
            g = xp.matmul(g4, g5)
        else:
            raise ValueError("Invalid gamma matrix index")

        return g

    def __mul__(self, other):
        if isinstance(other, Fermion):
            result = Fermion(other.geometry)
            result.field = contract("ab, txyzbc -> txyzac", self.mat, other.field)
            return result
        else:
            return TypeError
