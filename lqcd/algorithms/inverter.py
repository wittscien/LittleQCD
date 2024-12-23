import lqcd.core as cr
from lqcd.io import get_backend
from lqcd.fermion import DiracOperator



class Inverter:
    def __init__(self, D: DiracOperator, params):
        self.D = D
        self.method = params["method"]
        self.tol = params["tol"]
        self.maxit = params["maxit"]
        self.check_residual = params["check_residual"]
        self.geometry = D.geometry
        self.verbose = 0
        if "verbose" in params: self.verbose = params["verbose"]
        if self.D.fermion_type == 'twisted_mass_clover': self.tm_rotation = params["tm_rotation"]

    def BiCGStab(self, b, x0, flavor):
        r0 = b - self.D.Dirac(x0, flavor)
        r0_prime = cr.Fermion(self.geometry)
        r0_prime.field = r0.field
        rho = alpha = omega = 1
        v = p = cr.Fermion(self.geometry)
        x = cr.Fermion(self.geometry)
        x.field = x0.field

        cnt = 0
        while True:
            cnt += 1
            if self.verbose >= 2: print("BiCGStab: ", r0.norm())
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

            if cnt > self.maxit: raise ValueError("BiCGStab: Max iteration reached.")

        if self.verbose >= 1: print("BiCGStab: Converged in {} iterations.".format(cnt))
        return x

    def invert(self, b, x0, flavor):
        # src rotation
        if self.D.fermion_type == 'twisted_mass_clover' and self.tm_rotation:
            src_temp = tm_rotation(b, flavor)
        else:
            src_temp = b
        if self.method == 'BiCGStab':
            prop_temp = self.BiCGStab(src_temp, x0, flavor)
        # Check residual
        if self.check_residual:
            print("%s residual =" % (self.method), (self.D.Dirac(prop_temp, flavor) - src_temp).norm())
        # snk rotation
        if self.D.fermion_type == 'twisted_mass_clover' and self.tm_rotation:
            prop_pb = tm_rotation(prop_temp, flavor)
        else:
            prop_pb = prop_temp
        return prop_pb


# twisted mass rotation
def tm_rotation(src, flavor):
    xp = get_backend()
    if flavor == 'u':
        tau3_sign = 1
    elif flavor == 'd':
        tau3_sign = -1
    return ((1 / xp.sqrt(2)) * (1 + 1j * cr.Gamma(5) * tau3_sign)) * src


if __name__ == "__main__":
    from lqcd.io import set_backend
    set_backend("numpy")
    xp = get_backend()

    geometry =cr.QCD_geometry([8, 4, 4, 4])
    U = cr.Gauge(geometry)
    U.init_random()
    src = cr.Fermion(geometry)
    src.point_source([0, 0, 0, 0, 0, 0])

    Q = DiracOperator(U, {'fermion_type':'twisted_mass_clover', 'm': 3, 'mu': 0.1, 'csw': 0.1})
    Inv = Inverter(Q, {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": True, "tm_rotation": True})
    x0 = cr.Fermion(geometry)
    prop = Inv.invert(src, x0, 'u')

    # Check residual in the physical basis. But since for tm rotation, the residual is not zero.
    # print((Q.Dirac(prop, 'u') - src).norm())

    # The full propagator: commented because propagator function is moved to another file.
    '''
    inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": False, "tm_rotation": True}
    srcfull = cr.Propagator(geometry)
    for s in range(4):
        for c in range(3):
            src = cr.Fermion(geometry)
            src.point_source([0, 0, 0, 0, s, c])
            srcfull.field[:,:,:,:,:,s,:,c] = src.field
    prop = propagator(Q, inv_params, srcfull, 'u')
    '''
