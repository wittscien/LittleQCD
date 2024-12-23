import lqcd.core as cr
from lqcd.io.backend import get_backend



class DiracOperator:
    def __init__(self, U: cr.Gauge, params):
        self.U = U.copy()
        self.geometry = U.geometry
        self.fermion_type = params['fermion_type']
        if 'm' in params:
            assert 'kappa' not in params
            self.m = params["m"]
            self.kappa = 1 / (2 * (params["m"] + 4))
        else:
            assert 'm' not in params
            self.kappa = params["kappa"]
            self.m = 1 / (2 * params["kappa"]) - 4
        self.mu = params["mu"]
        self.csw = params["csw"]
        self.mu_num2st = {0: ['t', '-t'], 1: ['x', '-x'], 2: ['y', '-y'], 3: ['z', '-z']}
        self.mu_neg = {'t': '-t', '-t': 't', 'x': '-x', '-x': 'x', 'y': '-y', '-y': 'y', 'z': '-z', '-z': 'z'}

    def hopping(self, src):
        dst = cr.Fermion(self.geometry)
        for mu in range(self.geometry.Nl):
            fwdmu = self.mu_num2st[mu][0]
            bwdmu = self.mu_num2st[mu][1]
            dst += (-1/2) * ((1 - cr.Gamma(mu)) * (self.U.mu(fwdmu) * src.shift(fwdmu)) + (1 + cr.Gamma(mu)) * (self.U.mu(bwdmu) * src.shift(bwdmu)))
        return dst

    def mass(self, src):
        return (self.m + 4) * src

    def clover(self, src):
        dst = cr.Fermion(self.geometry)
        for mu in range(self.geometry.Nl - 1):
            for nu in range(mu + 1, self.geometry.Nl):
                fwdmu = self.mu_num2st[mu][0]
                fwdnu = self.mu_num2st[nu][0]
                # 2024.11.13: On the flight back to Beijing: it is checked that the order of applying gamma and gauge to the src does not matter.
                dst += self.csw * (1 / 2) * cr.sigma_munu(mu, nu) * (self.U.field_strength(fwdmu, fwdnu) * src)
        return dst

    def twisted_mass(self, src, flavor):
        if flavor == 'u':
            tau3_sign = 1
        elif flavor == 'd':
            tau3_sign = -1
        return 1j * self.mu * (cr.Gamma(5) * (tau3_sign * src))

    def Dirac(self, src, flavor):
        if self.fermion_type == 'twisted_mass_clover':
            return self.hopping(src) + self.mass(src) + self.clover(src) + self.twisted_mass(src, flavor)
