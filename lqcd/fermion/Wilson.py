from opt_einsum import contract
from lqcd.io.backend import get_backend
from lqcd.core.fields import Gauge, GaugeMu, Fermion, Gamma, Propagator



class DiracOperator:
    def __init__(self, U: Gauge, params):
        self.U = U
        self.geometry = U.geometry
        self.m = params["m"]
        self.csw = params["csw"]
        self.mu_num2st = {0: ['t', '-t'], 1: ['x', '-x'], 2: ['y', '-y'], 3: ['z', '-z']}
        self.mu_neg = {'t': '-t', '-t': 't', 'x': '-x', '-x': 'x', 'y': '-y', '-y': 'y', 'z': '-z', '-z': 'z'}

    def hopping(self, src):
        xp = get_backend()
        dst = Fermion(self.geometry)
        for mu in range(self.geometry.Nl):
            src_fwd = Fermion(self.geometry)
            src_bwd = Fermion(self.geometry)
            src_fwd.field = xp.roll(src.field, -1, axis=mu)
            src_bwd.field = xp.roll(src.field, +1, axis=mu)
            fwdmu = self.mu_num2st[mu][0]
            bwdmu = self.mu_num2st[mu][1]
            dst += (-1/2) * ((1 - Gamma(mu)) * (self.U.mu(fwdmu) * src_fwd) + (1 + Gamma(mu)) * (self.U.mu(bwdmu) * src_bwd))
        return dst
