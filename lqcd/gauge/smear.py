import lqcd.core as cr
from lqcd.io import get_backend



class Smear:
    def __init__(self, U: cr.Gauge, params):
        self.U = U.copy()
        self.geometry = U.geometry
        self.tech = params['tech']
        if self.tech == 'APE':
            self.alpha = params["alpha"]
            self.niter = params["niter"]
        elif self.tech == 'Stout':
            self.rho = params["rho"]
            self.niter = params["niter"]
        self.mu_num2st = {0: ['t', '-t'], 1: ['x', '-x'], 2: ['y', '-y'], 3: ['z', '-z']}
        self.mu_neg = {'t': '-t', '-t': 't', 'x': '-x', '-x': 'x', 'y': '-y', '-y': 'y', 'z': '-z', '-z': 'z'}

    def smear(self):
        if self.tech == "APE":
            return self.APE_space()
        elif self.tech == "Stout":
            return self.Stout()

    # QUDA convention
    def APE_space(self):
        result = cr.Gauge(self.geometry)
        Uold = cr.Gauge(self.geometry)
        result = self.U.copy()
        for _ in range(self.niter):
            Uold = result.copy()
            for mu in [1,2,3]:
                temp = cr.GaugeMu(self.geometry)
                for nu in [1,2,3]:
                    if mu == nu: continue
                    fwdmu = self.mu_num2st[mu][0]
                    # temp += self.alpha * Uold.Cmunu(mu,nu) # Old CVC convention
                    temp += self.alpha / 4 * Uold.Cmunu(mu,nu)
                # result.set_mu(mu, Uold.mu(fwdmu) + temp) # Old CVC convention
                result.set_mu(mu, (1 - self.alpha)*Uold.mu(fwdmu) + temp)
                result.proj_su3()
        return result

    def Stout(self):
        # The Qmu with rho = 1 is the Z (a Gauge object, has all mu) in the gradient flow.
        # rho can be factored out.
        Uold = cr.Gauge(self.geometry)
        result = self.U.copy()
        for _ in range(self.niter):
            Uold = result.copy()
            for mu in [0,1,2,3]:
                Qmu = self.rho * Uold.Qmu(mu)
                # U' = exp(Q)U
                fwdmu = self.mu_num2st[mu][0]
                result.set_mu(mu, Qmu.to_exp() * Uold.mu(fwdmu))
        return result



if __name__ == "__main__":
    from lqcd.io import set_backend
    set_backend("numpy")
    xp = get_backend()

    geometry = cr.QCD_geometry([8, 4, 4, 4])
    U = cr.Gauge(geometry)
    U.init_random()

    # APE
    Smr = Smear(U, {"tech": "APE", "alpha": 0.1, "niter": 2})
    U2 = Smr.smear()

    # Stout
    Smr = Smear(U, {"tech": "Stout", "rho": 0.1, "niter": 2})
    U2 = Smr.smear()
