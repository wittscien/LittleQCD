from opt_einsum import contract
from lqcd.io.backend import get_backend
from lqcd.core.fields import Gauge, GaugeMu
import lqcd.utils.utils as ut



class Smear:
    def __init__(self, U: Gauge, params):
        self.U = U
        self.geometry = U.geometry
        if params['tech'] == 'APE':
            self.alpha = params["alpha"]
            self.niter = params["niter"]
        self.mu_num2st = {0: ['t', '-t'], 1: ['x', '-x'], 2: ['y', '-y'], 3: ['z', '-z']}
        self.mu_neg = {'t': '-t', '-t': 't', 'x': '-x', '-x': 'x', 'y': '-y', '-y': 'y', 'z': '-z', '-z': 'z'}

    def APE_smear_space(self):
        result = Gauge(self.geometry)
        Uold = Gauge(self.geometry)
        result.field = self.U.field
        for _ in range(self.niter):
            Uold.field = result.field
            for mu in [1,2,3]:
                for nu in [1,2,3]:
                    if mu == nu: continue
                    fwdmu = self.mu_num2st[mu][0]
                    result.set_mu(mu, Uold.mu(fwdmu) + self.alpha * (self.Cmunu(mu,nu)))
                result.proj_su3()
        return result



if __name__ == "__main__":
    from lqcd.io.backend import set_backend
    from lqcd.core.geometry import QCD_geometry
    set_backend("numpy")
    xp = get_backend()

    geometry = QCD_geometry([8, 4, 4, 4])
    U = Gauge(geometry)
    U.init_random()

    Smr = Smear(U, {"tech": "APE", "alpha": 0.1, "niter": 2})
    U2 = Smr.APE_smear_space()
