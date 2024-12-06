from opt_einsum import contract
from lqcd.io.backend import get_backend
from lqcd.core import *
import lqcd.utils as ut



class Smear:
    def __init__(self, U: Gauge, psi: Fermion, params):
        self.U = U.copy()
        self.psi = psi.copy()
        self.geometry = U.geometry
        if params['tech'] == 'Jacobi':
            self.niter = params["niter"]
            self.kappa = params["kappa"]
        self.mu_num2st = {0: ['t', '-t'], 1: ['x', '-x'], 2: ['y', '-y'], 3: ['z', '-z']}
        self.mu_neg = {'t': '-t', '-t': 't', 'x': '-x', '-x': 'x', 'y': '-y', '-y': 'y', 'z': '-z', '-z': 'z'}

    def Jacobi_smear(self):
        xp = get_backend()
        result = Fermion(self.geometry)
        psiold = Fermion(self.geometry)
        result = self.psi.copy()
        for _ in range(self.niter):
            psiold = result.copy()
            result = psiold.copy()
            for mu in [1,2,3]:
                fwdmu = self.mu_num2st[mu][0]
                bwdmu = self.mu_num2st[mu][1]
                result += self.kappa * (self.U.mu(fwdmu) * psiold.shift(fwdmu) + self.U.mu(bwdmu) * psiold.shift(bwdmu))
            result.field *= 1 / (1 + 6 * self.kappa)
        return result



if __name__ == "__main__":
    from lqcd.io.backend import set_backend
    set_backend("numpy")
    xp = get_backend()

    geometry = QCD_geometry([8, 4, 4, 4])
    # geometry = QCD_geometry([96, 48, 48, 48])
    U = Gauge(geometry)
    U.init_random()
    src = Fermion(geometry)
    src.point_source([0, 0, 0, 0, 0, 0])

    Smr = Smear(U, src, {"tech": "Jacobi", "kappa": 0.1, "niter": 10})
    src2 = Smr.Jacobi_smear()

    import matplotlib.pyplot as plt
    plt.plot(src.field[0,:,0,0,0,0].real)
    plt.plot(src2.field[0,:,0,0,0,0].real)
