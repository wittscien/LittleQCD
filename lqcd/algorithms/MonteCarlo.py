import matplotlib.pyplot as plt
from pathlib import Path
from lqcd.io.backend import get_backend
from lqcd.core.geometry import QCD_geometry
from lqcd.core.fields import Gauge
import lqcd.algorithms.mc_funcs as mf



class MonteCarlo:
    def __init__(self, params):
        self.geometry = QCD_geometry(params["geometry"])
        self.beta = params["beta"]
        self.n_steps = params["n_steps"]
        self.n_hit = params["n_hit"]
        self.n_therm = params["n_therm"]
        self.n_take = params["n_take"]
        self.eps = params["eps"]
        self.mu_num2st = {0: ['t', '-t'], 1: ['x', '-x'], 2: ['y', '-y'], 3: ['z', '-z']}
        Path("confs/beta_%.2f_L%dx%d"%(self.beta, self.geometry.X, self.geometry.T)).mkdir(parents=True, exist_ok=True)

    def Markov(self):
        U0 = Gauge(self.geometry)
        U0.init_trivial()
        U0.init_random()
        self.acceptance_rate = 0
        self.confs = []
        self.confs.append(U0)
        for step in range(self.n_steps):
            Unew = self.metropolis(self.confs[step])
            self.confs.append(Unew)
            if step % 10 == 0:
                print(f"Step {step}/{self.n_steps}: Action = {self.beta / self.geometry.Nc * Unew.plaquette_action()}")
                if step >= self.n_therm and (step - self.n_therm) % self.n_take == 0:
                    Unew.write("confs/beta_%.2f_L%dx%d/beta_%.2f_L%dx%d_conf_%d.h5"%(self.beta, self.geometry.X, self.geometry.T, self.beta, self.geometry.X, self.geometry.T, step))
        self.acceptance_rate /= self.n_steps * self.geometry.T * self.geometry.X * self.geometry.Y * self.geometry.Z * self.geometry.Nl * self.n_hit
        print(f"Acceptance rate = {self.acceptance_rate}")

    def gen_X_list(self):
        X_list = []
        xp.random.seed(0)
        for _ in range(self.n_hit * self.geometry.T * self.geometry.X * self.geometry.Y * self.geometry.Z * self.geometry.Nl // 2):
            r = mf.SU2_eps(self.eps, xp.random.uniform(-0.5, 0.5, 4))
            s = mf.SU2_eps(self.eps, xp.random.uniform(-0.5, 0.5, 4))
            t = mf.SU2_eps(self.eps, xp.random.uniform(-0.5, 0.5, 4))
            X = mf.SU3_SU2(r, s, t)
            X_list.append(X)
            # For detailed balance
            X_list.append(xp.conjugate(xp.transpose(X)))
        X_list = xp.array(X_list)
        xp.random.shuffle(X_list)
        return X_list

    def metropolis(self, U):
        # X_list here has the length to go through the lattice exactly once.
        X_list = self.gen_X_list()
        Unew = Gauge(self.geometry)
        Unew.field = xp.copy(U.field)
        xiter = 0
        for t in range(self.geometry.T):
            for x in range(self.geometry.X):
                for y in range(self.geometry.Y):
                    for z in range(self.geometry.Z):
                        for mu in range(self.geometry.Nl):
                            # staple
                            for nu in range(self.geometry.Nl):
                                if mu == nu: continue
                                # To save the memory, only local SU(3) matrices are computed, without calling GaugeMu.
                                fwdmu = self.mu_num2st[mu][0]
                                fwdnu = self.mu_num2st[nu][0]
                                bwdmu = self.mu_num2st[mu][1]
                                bwdnu = self.mu_num2st[nu][1]
                                A = Unew.shift(fwdmu).x_mu(t,x,y,z,fwdnu) @ Unew.shift(fwdmu).shift(fwdnu).x_mu(t,x,y,z,bwdmu) @ Unew.shift(fwdnu).x_mu(t,x,y,z,bwdnu) \
                                   +Unew.shift(fwdmu).x_mu(t,x,y,z,bwdnu) @ Unew.shift(fwdmu).shift(bwdnu).x_mu(t,x,y,z,bwdmu) @ Unew.shift(bwdnu).x_mu(t,x,y,z,fwdnu)
                            for n in range(self.n_hit):
                                bracket = (X_list[xiter] - xp.identity(3)) @ Unew.field[t,x,y,z,mu,:,:]
                                delta_S = -self.beta / self.geometry.Nc * xp.trace(bracket @ A).real
                                if delta_S < 0 or xp.random.rand() <= xp.exp(-delta_S):
                                    Unew[t,x,y,z,mu] = X_list[xiter] @ Unew[t,x,y,z,mu]
                                    self.acceptance_rate += 1
                                xiter += 1
        return Unew

    def plaquette_plot(self):
        step_list = xp.arange(len(self.confs))
        plaq_list = xp.zeros(len(self.confs))
        for i in range(len(self.confs)):
            plaq_list[i] = self.beta / self.geometry.Nc * self.confs[i].plaquette_action()
        plt.plot(step_list, plaq_list)
        plt.xlim([0, self.n_steps])
        plt.xlabel("Step")
        plt.ylabel("Plaquette action")


if __name__ == "__main__":
    from lqcd.io.backend import set_backend
    set_backend("numpy")
    xp = get_backend()

    geo_vec = [8, 4, 4, 4]
    geometry = QCD_geometry(geo_vec)
    mc_params = {"geometry": geo_vec, "beta": 6.0, "n_steps": 1000, "n_hit": 3, "n_therm": 100, "n_take": 2, "eps": 0.05}
    MC = MonteCarlo(mc_params)
    MC.Markov()
    MC.plaquette_plot()
