from opt_einsum import contract
from lqcd.io import get_backend
from lqcd.core import *
from lqcd.fermion import DiracOperator, tm_rotation
import lqcd.utils as ut



class Inverter:
    def __init__(self, D: DiracOperator, params):
        self.D = D
        self.method = params["method"]
        self.tol = params["tol"]
        self.maxit = params["maxit"]
        self.check_residual = params["check_residual"]
        self.geometry = D.geometry

    def BiCGStab(self, b, x0, flavor):
        r0 = b - self.D.Dirac(x0, flavor)
        r0_prime = Fermion(self.geometry)
        r0_prime.field = r0.field
        rho = alpha = omega = 1
        v = p = Fermion(self.geometry)
        x = Fermion(self.geometry)
        x.field = x0.field

        cnt = 0
        while True:
            cnt += 1
            # print("BiCGStab: ", r0.norm())
            rho_new = r0_prime.dot(r0)
            if rho_new == 0: raise ValueError("Breakdown: rho = 0.")

            beta = (rho_new / rho) * (alpha / omega)
            rho = rho_new
            p = r0 + beta * (p - omega * v)
            v = self.D.Dirac(p, flavor)
            alpha = rho / r0_prime.dot(v)

            s = r0 - alpha * v
            if s.norm() < self.tol:
                x += alpha * p
                break

            t = self.D.Dirac(s, flavor)
            omega = t.dot(s) / t.dot(t)

            x += alpha * p + omega * s
            r0 = s - omega * t

            if r0.norm() < self.tol: break
            if cnt > self.maxit: raise ValueError("BiCGStab: Max iteration reached.")

        print("BiCGStab: Converged in {} iterations.".format(cnt))
        return x

    def invert(self, b, x0, flavor):
        # src rotation
        if self.D.fermion_type == 'twisted_mass_clover':
            src_temp = tm_rotation(b, flavor)
        else:
            src_temp = b
        if self.method == 'BiCGStab':
            prop_temp = self.BiCGStab(src_temp, x0, flavor)
        # Check residual
        if self.check_residual:
            print((self.D.Dirac(prop_temp, 'u') - src_temp).norm())
        # snk rotation
        if self.D.fermion_type == 'twisted_mass_clover':
            prop_pb = tm_rotation(prop_temp, flavor)
        else:
            prop_pb = prop_temp
        return prop_pb


def propagator(Q, inv_params, srcfull, flavor):
    # src_list is 4 x 3
    geometry = Q.geometry
    x0 = Fermion(geometry)
    Inv = Inverter(Q, inv_params)
    prop = Propagator(geometry)
    for s in range(4):
        for c in range(3):
            src = Fermion(geometry)
            src.field = srcfull.field[:,:,:,:,:,s,:,c]
            prop.field[:,:,:,:,:,s,:,c] = Inv.invert(src, x0, 'u').field
    return prop


if __name__ == "__main__":
    from lqcd.io import set_backend
    set_backend("numpy")
    xp = get_backend()

    geometry = QCD_geometry([8, 4, 4, 4])
    U = Gauge(geometry)
    U.init_random()
    src = Fermion(geometry)
    src.point_source([0, 0, 0, 0, 0, 0])

    Q = DiracOperator(U, {'fermion_type':'twisted_mass_clover', 'm': 3, 'mu': 0.1, 'csw': 0.1})
    Inv = Inverter(Q, {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": True})
    x0 = Fermion(geometry)
    prop = Inv.invert(src, x0, 'u')

    # Check residual in the physical basis. But since for tm rotation, the residual is not zero.
    # print((Q.Dirac(prop, 'u') - src).norm())

    # The full propagator
    inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": False}
    srcfull = Propagator(geometry)
    for s in range(4):
        for c in range(3):
            src = Fermion(geometry)
            src.point_source([0, 0, 0, 0, s, c])
            srcfull.field[:,:,:,:,:,s,:,c] = src.field
    prop = propagator(Q, inv_params, srcfull, 'u')
