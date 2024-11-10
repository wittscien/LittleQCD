from opt_einsum import contract
from lqcd.io.backend import get_backend
from lqcd.core.fields import Gauge, Fermion, Gamma, Propagator



class DiracOperator:
    def __init__(self, U: Gauge, params):
        self.U = U
        self.geometry = U.geometry
        self.m = params["m"]
        self.csw = params["csw"]

    def hopping(self, src):
        xp = get_backend()
        mu_num2st = {0: ['t', '-t'], 1: ['x', '-x'], 2: ['y', '-y'], 3: ['z', '-z']}
        dst = Fermion(self.geometry)
        for mu in range(self.geometry.Nl):
            src_fwd = Fermion(self.geometry)
            src_bwd = Fermion(self.geometry)
            src_fwd.field = xp.roll(src.field, -1, axis=mu)
            src_bwd.field = xp.roll(src.field, +1, axis=mu)
            fwdmu = mu_num2st[mu][0]
            bwdmu = mu_num2st[mu][1]
            dst += (-1/2) * ((1 - Gamma(mu)) * (self.U.mu(fwdmu) * src_fwd) + (1 + Gamma(mu)) * (self.U.mu(bwdmu) * src_bwd))
        return dst
