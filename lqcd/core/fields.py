from opt_einsum import contract
from sympy import N
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
    def __init__(self, i, Nl=4):
        if Nl == 4:
            self.mat = self.gamma_4(i)
        else:
            raise NotImplementedError("Only Nl=4 is supported")

    def gamma_4(self, i):
        xp = get_backend()

        g = xp.zeros((16, 4, 4), dtype=complex)

        # # Chroma convention, my convention.
        # g[0] = xp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=complex)
        # g[1] = xp.array([[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]], dtype=complex)
        # g[2] = xp.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=complex)
        # g[3] = xp.array([[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]], dtype=complex)
        # g[4] = xp.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=complex)
        # g[5] = xp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=complex)
        # g[6] = g[2] @ g[3] # -gamma1*gamma4*gamma5 (gamma2*gamma3)
        # g[7] = g[3] @ g[1] # -gamma2*gamma4*gamma5 (gamma3*gamma1)
        # g[8] = g[1] @ g[2] # -gamma3*gamma4*gamma5 (gamma1*gamma2)
        # g[9] = g[1] @ g[4] # gamma1*gamma4
        # g[10] = g[2] @ g[4] # gamma2*gamma4
        # g[11] = g[3] @ g[4] # gamma3*gamma4
        # g[12] = g[1] @ g[5] # gamma1*gamma5
        # g[13] = g[2] @ g[5] # gamma2*gamma5
        # g[14] = g[3] @ g[5] # gamma3*gamma5
        # g[15] = g[4] @ g[5] # gamma4*gamma5

        # CVC convention.
        g[0] = xp.array([[0, 0, -1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0]], dtype=complex) # gamma_0 = gamma_t
        g[1] = xp.array([[0, 0, 0, -1j], [0, 0, -1j, 0], [0, 1j, 0, 0], [1j, 0, 0, 0]], dtype=complex) # gamma_1 = gamma_x
        g[2] = xp.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=complex) # gamma_2 = gamma_y
        g[3] = xp.array([[0, 0, -1j, 0], [0, 0, 0, 1j], [1j, 0, 0, 0], [0, -1j, 0, 0]], dtype=complex) # gamma_3 = gamma_z
        g[4] = xp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=complex) # gamma_4 = id
        g[5] = xp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=complex)
        g[6] = g[5] @ g[0] # gamma_6 = gamma_5 gamma_t
        g[7] = g[5] @ g[1] # gamma_7 = gamma_5 gamma_x
        g[8] = g[5] @ g[2] # gamma_8 = gamma_5 gamma_y
        g[9] = g[5] @ g[3] # gamma_9 = gamma_5 gamma_z

        return g[i]

    def __mul__(self, other):
        if isinstance(other, Fermion):
            result = Fermion(other.geometry)
            result.field = contract("ab, txyzbc -> txyzac", self.mat, other.field)
            return result
        else:
            return TypeError
