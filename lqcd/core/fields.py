import numbers
import h5py
from opt_einsum import contract
from sympy import N
from lqcd.io.backend import get_backend
from lqcd.core.geometry import QCD_geometry
import lqcd.utils.utils as ut



class Field:
    def __init__(self, geometry: QCD_geometry):
        self.geometry = geometry
        self.T = geometry.T
        self.X = geometry.X
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
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, Field):
            result = type(self)(self.geometry)
            result.field = self.field + other.field
            return result
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Field):
            result = type(self)(self.geometry)
            result.field = self.field - other.field
            return result
        else:
            return NotImplemented

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

    def read(self, filename):
        with h5py.File(filename, 'r') as f:
            self.field = f['field'][()]

    def write(self, filename):
        with h5py.File(filename, 'w') as f:
            f.create_dataset('field', data=self.field)

    def init_trivial(self):
        xp = get_backend()
        for t in range(self.T):
            for x in range(self.X):
                for y in range(self.Y):
                    for z in range(self.Z):
                        for mu in range(self.Nl):
                            self.field[t, x, y, z, mu] = xp.identity(self.Nc)

    def init_random(self):
        xp = get_backend()
        self.field = xp.random.rand(self.geometry.T, self.geometry.X, self.geometry.Y, self.geometry.Z, self.geometry.Nl, self.geometry.Nc, self.geometry.Nc) + 1j * xp.random.rand(self.geometry.T, self.geometry.X, self.geometry.Y, self.geometry.Z, self.geometry.Nl, self.geometry.Nc, self.geometry.Nc)
        self.proj_su3()

    def proj_su3(self):
        for t in range(self.T):
            for x in range(self.X):
                for y in range(self.Y):
                    for z in range(self.Z):
                        for mu in range(self.Nl):
                            self.field[t, x, y, z, mu] = ut.proj_su3(self.field[t, x, y, z, mu])

    def apply_boundary_condition_periodic_quark(self):
        xp = get_backend()
        result = Gauge(self.geometry)
        result.field = xp.copy(self.field)
        phase_factor = xp.exp(1j * xp.pi / self.T)
        # mu=0
        for t in range(self.T):
            result.field[t,:,:,:,0,:,:] *= phase_factor
        return result

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
        result.field = xp.roll(self.field, -dir, axis=mu)
        return result

    def mu(self, m):
        # Return a GaugeMu object.
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
            result.field = self.shift(m).field[:,:,:,:,mu,:,:]
            result = result.dagger()
            return result

    def x_mu(self, t, x, y, z, m):
        # Return a SU(3) matrix at [t, x, y, z, mu].
        mu_st2num = {'t': 0, 'x': 1, 'y': 2, 'z': 3}
        if m in ['t', 'x', 'y', 'z']:
            mu = mu_st2num[m]
            mat = self.field[t,x,y,z,mu,:,:]
            return mat
        elif m in ['-t', '-x', '-y', '-z']:
            xp = get_backend()
            mu = mu_st2num[m[1]]
            mat = self.shift(m).field[t,x,y,z,mu,:,:]
            mat = xp.conjugate(xp.transpose(mat))
            return mat

    def set_mu(self, mu, Umu):
        self.field[:,:,:,:,mu,:,:] = Umu.field

    def plaquette(self, mu, nu):
        mu_neg = self.mu_neg[mu]
        nu_neg = self.mu_neg[nu]
        # Periodic boundary condition is assumed by writing this way.
        result = self.mu(mu) * self.shift(mu).mu(nu) * self.shift(mu).shift(nu).mu(mu_neg) * self.shift(nu).mu(nu_neg)
        return result

    def plaquette_action(self):
        # No beta / N factor.
        xp = get_backend()
        S = 0
        for mu in range(self.Nl - 1):
            for nu in range(mu + 1, self.Nl):
                plaq = self.plaquette(self.mu_num2st[mu][0], self.mu_num2st[nu][0])
                for t in range(self.T):
                    for x in range(self.X):
                        for y in range(self.Y):
                            for z in range(self.Z):
                                S += (xp.identity(self.Nc) - plaq[t,x,y,z]).trace().real
        return S

    def clover(self, mu, nu):
        mu_neg = self.mu_neg[mu]
        nu_neg = self.mu_neg[nu]
        result = self.plaquette(mu, nu) + self.plaquette(nu, mu_neg) + self.plaquette(mu_neg, nu_neg) + self.plaquette(nu_neg, mu)
        return result

    def field_strength(self, mu, nu):
        # 2024.11.11: for plaquette and clover, A_ab = A_ba^dagger. The following is tested to be the same. The plaquette or clover.
        # U.plaquette('t', 'x').field
        # U.plaquette('x', 't').dagger().field
        result = (-1j) * (1/4) * (1/2) * (self.clover(mu, nu) - self.clover(nu, mu))
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
        if isinstance(other, numbers.Number):
            result = GaugeMu(self.geometry)
            result.field = self.field * other
            return result
        elif isinstance(other, GaugeMu):
            result = GaugeMu(self.geometry)
            result.field = contract("txyzab, txyzbc -> txyzac", self.field, other.field)
            return result
        elif isinstance(other, Fermion):
            result = Fermion(self.geometry)
            result.field = contract("txyzab, txyzsb -> txyzsa", self.field, other.field)
            return result
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return self.__mul__(other)
        else:
            return NotImplemented

    def dagger(self):
        xp = get_backend()
        result = GaugeMu(self.geometry)
        result.field = xp.conjugate((xp.transpose(self.field, axes=(0,1,2,3,5,4))))
        return result


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
        self.field[point[0],point[1],point[2],point[3],point[4],point[5]] = 1

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            result = Fermion(self.geometry)
            result.field = self.field * other
            return result
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return self.__mul__(other)

    def dot(self, other):
        xp = get_backend()
        if isinstance(other, Fermion):
            return contract("txyzsc, txyzsc", xp.conjugate(self.field), other.field)
        else:
            return NotImplemented

    def norm(self):
        xp = get_backend()
        return xp.sqrt(self.dot(self).real)


class Propagator(Field):
    def __init__(self, geometry: QCD_geometry):
        xp = get_backend()
        super().__init__(geometry)
        # spin sink, spin source, color sink, color source
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
        g[6] = g[0] @ g[5]
        g[7] = g[1] @ g[5]
        g[8] = g[2] @ g[5]
        g[9] = g[3] @ g[5]
        g[10] = g[0] @ g[1]
        g[11] = g[0] @ g[2]
        g[12] = g[0] @ g[3]
        g[13] = g[1] @ g[2]
        g[14] = g[1] @ g[3]
        g[15] = g[2] @ g[3]

        return g[i]

    def __add__(self, other):
        xp = get_backend()
        if isinstance(other, Gamma):
            result = Gamma(0)
            result.mat = self.mat + other.mat
            return result
        elif isinstance(other, numbers.Number):
            result = Gamma(0)
            result.mat = self.mat + xp.eye(4, dtype=complex) * other
            return result
        else:
            return NotImplemented

    def __sub__(self, other):
        xp = get_backend()
        if isinstance(other, Gamma):
            result = Gamma(0)
            result.mat = self.mat - other.mat
            return result
        elif isinstance(other, numbers.Number):
            result = Gamma(0)
            result.mat = self.mat - xp.eye(4, dtype=complex) * other
            return result
        else:
            return NotImplemented

    def __radd__(self, other):
        xp = get_backend()
        if isinstance(other, numbers.Number):
            result = Gamma(0)
            result.mat = self.mat + xp.eye(4, dtype=complex) * other
            return result
        else:
            return NotImplemented

    def __rsub__(self, other):
        xp = get_backend()
        if isinstance(other, numbers.Number):
            result = Gamma(0)
            result.mat = -self.mat + xp.eye(4, dtype=complex) * other
            return result
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Gamma):
            result = Gamma(0)
            result.mat = self.mat @ other.mat
            return result
        elif isinstance(other, Fermion):
            # capital: spin; small: color
            result = Fermion(other.geometry)
            result.field = contract("CB, txyzBa -> txyzCa", self.mat, other.field)
            return result
        elif isinstance(other, Propagator):
            result = Propagator(other.geometry)
            result.field = contract("CB, txyzBAba -> txyzCAba", self.mat, other.field)
            return result
        elif isinstance(other, numbers.Number):
            result = Gamma(0)
            result.mat = self.mat * other
            return result
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return self.__mul__(other)
        elif isinstance(other, Propagator):
            result = Propagator(other.geometry)
            result.field = contract("txyzBAba, AC -> txyzBCba", other.field, self.mat)
            return result
        else:
            return NotImplemented

    def __repr__(self):
        return f"{self.mat}"


def sigma_munu(mu, nu):
    return (1 / 2j) * (Gamma(mu) * Gamma(nu) - Gamma(nu) * Gamma(mu))



if __name__ == "__main__":
    from lqcd.io.backend import set_backend
    set_backend("numpy")
    xp = get_backend()

    geometry = QCD_geometry([8, 4, 4, 4])
    U = Gauge(geometry)
    U.init_random()

    # Test the boundary condition.
    plaq_GaugeMu = U.plaquette('t','x').field[3,3,1,2]
    plaq_mat = U.field[3,3,1,2,0] @ U.field[4,3,1,2,1] @ xp.conjugate(xp.transpose(U.field[3,0,1,2,0])) @ xp.conjugate(xp.transpose(U.field[3,3,1,2,1]))
    print(plaq_GaugeMu - plaq_mat)

    print(U.shift('x')[3,3,1,2,0] - U.field[3,0,1,2,0])
    print(U.shift('t')[3,3,1,2,0] - U.field[4,3,1,2,0])
