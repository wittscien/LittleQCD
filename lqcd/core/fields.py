import numbers
from opt_einsum import contract
from sympy import N
from lqcd.io.backend import get_backend
from lqcd.core.geometry import QCD_geometry



class Field:
    def __init__(self, geometry: QCD_geometry):
        self.geometry = geometry
        self.T = geometry.T
        self.X = geometry.Z
        self.Y = geometry.Y
        self.Z = geometry.Z
        self.Ns = geometry.Ns
        self.Nc = geometry.Nc
        self.Nl = geometry.Nl

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
            result = type(self)(self.geometry)
            result.field = self.field + other.field
            return result
        else:
            return TypeError

    def __sub__(self, other):
        if isinstance(other, Field):
            result = type(self)(self.geometry)
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
        self.mu_num2st = {0: ['t', '-t'], 1: ['x', '-x'], 2: ['y', '-y'], 3: ['z', '-z']}
        self.mu_neg = {'t': '-t', '-t': 't', 'x': '-x', '-x': 'x', 'y': '-y', '-y': 'y', 'z': '-z', '-z': 'z'}

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

    def init_random(self):
        xp = get_backend()
        xp.random.seed(0)
        self.field = xp.random.rand(self.geometry.T, self.geometry.X, self.geometry.Y, self.geometry.Z, self.geometry.Nl, self.geometry.Nc, self.geometry.Nc)

    def shift(self, m):
        xp = get_backend()
        mu_st2num = {'t': 0, 'x': 1, 'y': 2, 'z': 3}
        result = Gauge(self.geometry)
        # U(x+mu)
        if m in ['t', 'x', 'y', 'z']:
            dir = +1
            mu = mu_st2num[m]
        # U(x-mu)
        elif m in ['-t', '-x', '-y', '-z']:
            dir = -1
            mu = mu_st2num[m[1]]
        result.field = xp.roll(self.field, -dir*mu, axis=mu)
        return result

    def mu(self, m):
        mu_st2num = {'t': 0, 'x': 1, 'y': 2, 'z': 3}
        if m in ['t', 'x', 'y', 'z']:
            result = GaugeMu(self.geometry)
            mu = mu_st2num[m]
            result.field = self.field[:,:,:,:,mu,:,:]
            return result
        elif m in ['-t', '-x', '-y', '-z']:
            xp = get_backend()
            result = GaugeMu(self.geometry)
            mu = mu_st2num[m[1]]
            result.field = xp.conjugate((xp.transpose(self.shift(m).field[:,:,:,:,mu,:,:], axes=(0,1,2,3,5,4))))
            return result

    def plaquette(self, mu, nu):
        mu_neg = self.mu_neg[mu]
        nu_neg = self.mu_neg[nu]
        result = self.mu(mu) * self.shift(mu).mu(nu) * self.shift(mu).shift(nu).mu(mu_neg) * self.shift(nu).mu(nu_neg)
        return result

    def clover(self, mu, nu):
        mu_neg = self.mu_neg[mu]
        nu_neg = self.mu_neg[nu]
        result = self.plaqutte(mu, nu) + self.plaqutte(nu, mu_neg) + self.plaqutte(mu_neg, nu_neg) + self.plaqutte(nu_neg, mu)
        return result


class GaugeMu(Field):
    def __init__(self, geometry: QCD_geometry):
        xp = get_backend()
        super().__init__(geometry)
        self.field = xp.zeros((geometry.T, geometry.X, geometry.Y, geometry.Z, geometry.Nc, geometry.Nc), dtype=xp.complex128)

    def __getitem__(self, pos):
        # The SU(3) matrix at [t, x, y, z], or the point at [t, x, y, z, a, b]
        if len(pos) in [4, 6]:
            return self.field[pos]
        else:
            raise ValueError("Invalid number of indices")

    def __setitem__(self, pos, mat):
        if len(pos) in [4, 6]:
            self.field[pos] = mat
        else:
            raise ValueError("Invalid number of indices")

    def __mul__(self, other):
        xp = get_backend()
        # U_mu * psi
        if isinstance(other, GaugeMu):
            result = GaugeMu(self.geometry)
            result.field = contract("txyzab, txyzbc -> txyzac", self.field, other.field)
            return result
        elif isinstance(other, Fermion):
            result = Fermion(self.geometry)
            result.field = contract("txyzab, txyzsb -> txyzsa", self.field, other.field)
            return result
        else:
            return TypeError


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

    def point_source(self, point):
        xp = get_backend()
        self.field = xp.zeros((self.geometry.T, self.geometry.X, self.geometry.Y, self.geometry.Z, self.geometry.Ns, self.geometry.Nc), dtype=xp.complex128)
        self[point] = 1

    def __mul__(self, other):
        xp = get_backend()
        if isinstance(other, numbers.Number):
            result = Fermion(self.geometry)
            result.field = self.field * other
            return result
        else:
            return TypeError

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return self.__mul__(other)


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

    def __add__(self, other):
        xp = get_backend()
        if isinstance(other, Gamma):
            result = Gamma(0)
            result.mat = self.mat + other.mat
            return result
        elif isinstance(other, int):
            result = Gamma(0)
            result.mat = self.mat + xp.eye(4, dtype=complex) * other
            return result
        else:
            return TypeError

    def __sub__(self, other):
        xp = get_backend()
        if isinstance(other, Gamma):
            result = Gamma(0)
            result.mat = self.mat - other.mat
            return result
        elif isinstance(other, int):
            result = Gamma(0)
            result.mat = self.mat - xp.eye(4, dtype=complex) * other
            return result
        else:
            return TypeError

    def __radd__(self, other):
        xp = get_backend()
        if isinstance(other, int):
            result = Gamma(0)
            result.mat = self.mat + xp.eye(4, dtype=complex) * other
            return result
        else:
            return TypeError

    def __rsub__(self, other):
        xp = get_backend()
        if isinstance(other, int):
            result = Gamma(0)
            result.mat = -self.mat + xp.eye(4, dtype=complex) * other
            return result
        else:
            return TypeError

    def __mul__(self, other):
        if isinstance(other, Fermion):
            result = Fermion(other.geometry)
            result.field = contract("ab, txyzbc -> txyzac", self.mat, other.field)
            return result
        else:
            return TypeError

    def __repr__(self):
        return f"{self.mat}"
