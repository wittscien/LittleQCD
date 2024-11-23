from opt_einsum import contract
from lqcd.io.backend import get_backend
from lqcd.core.fields import Gauge, GaugeMu, Fermion, Gamma, Propagator
from lqcd.fermion.Wilson import DiracOperator
import lqcd.utils.utils as ut



class Inverter:
    def __init__(self, D: DiracOperator, params):
        self.D = D
        self.tol = params["tol"]
        self.maxit = params["maxit"]
        self.geometry = D.geometry

    def BiCGStab(self, b, x0, flavor):
        r0 = b - self.D.Dirac_twisted_mass_clover(x0, flavor)
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
            v = self.D.Dirac_twisted_mass_clover(p, flavor)
            alpha = rho / r0_prime.dot(v)

            s = r0 - alpha * v
            if s.norm() < self.tol:
                x += alpha * p
                break

            t = self.D.Dirac_twisted_mass_clover(s, flavor)
            omega = t.dot(s) / t.dot(t)

            x += alpha * p + omega * s
            r0 = s - omega * t

            if r0.norm() < self.tol: break
            if cnt > self.maxit: raise ValueError("BiCGStab: Max iteration reached.")

        print("BiCGStab: Converged in {} iterations.".format(cnt))
        return x



if __name__ == "__main__":
    from lqcd.io.backend import set_backend
    from lqcd.core.geometry import QCD_geometry
    set_backend("numpy")
    xp = get_backend()

    geometry = QCD_geometry([8, 4, 4, 4])
    U = Gauge(geometry)
    U.init_random()
    src = Fermion(geometry)
    src.point_source([0, 0, 0, 0, 0, 0])

    Q = DiracOperator(U, {'m': 3, 'mu': 0.1, 'csw': 0.1})
    Inv = Inverter(Q, {"tol": 1e-9, "maxit": 500})
    x0 = Fermion(geometry)
    prop = Inv.BiCGStab(src, x0, 'u')

    print((Q.Dirac_twisted_mass_clover(prop, 'u') - src).norm())