import scipy as sp
from opt_einsum import contract
from lqcd.io.backend import get_backend
from lqcd.core.fields import Gauge, GaugeMu, Fermion
import lqcd.utils.utils as ut



class GFlow:
    def __init__(self, U: Gauge, chi: Fermion, params):
        self.U = U
        self.chi = chi
        self.geometry = U.geometry
        self.dt = params["dt"]
        self.niter = params["niter"]
        self.mu_num2st = {0: ['t', '-t'], 1: ['x', '-x'], 2: ['y', '-y'], 3: ['z', '-z']}
        self.mu_neg = {'t': '-t', '-t': 't', 'x': '-x', '-x': 'x', 'y': '-y', '-y': 'y', 'z': '-z', '-z': 'z'}

    def forward(self):
        # In place
        # I did not call Stout smearing because there is procedure like exp(8/9 Z1 - 17/36 Z0) W1, while the Stout smearing is just exp(Q)U.
        self.U_list = []
        for iter in range(self.niter):
            temp_U = Gauge(self.geometry)
            temp_U.field = xp.copy(self.U.field)
            self.U_list.append(temp_U)
            # Step 0: W0 = U; phi3 = chi
            W0 = Gauge(self.geometry)
            W0.field = xp.copy(self.U.field)
            phi3 = Fermion(self.geometry)
            phi3.field = xp.copy(self.chi.field)
            # Step 1: W1 = exp(1/4 Z0) W0
            Z0 = W0.Zgf() * self.dt
            W1 = (1/4 * Z0).to_exp() * W0
            # Step 2: W2 = exp(8/9 Z1 - 17/36 Z0) W1
            Z1 = W1.Zgf() * self.dt
            W2 = (8/9 * Z1 - 17/36 * Z0).to_exp() * W1
            # Step 3: W3 = exp(3/4 Z2 - 8/9 Z1 + 17/36 Z0) W2
            Z2 = W2.Zgf() * self.dt
            W3 = (3/4 * Z2 - 8/9 * Z1 + 17/36 * Z0).to_exp() * W2
            # Set U = W3
            self.U.field = xp.copy(W3.field)

    def plot_action(self):
        n_list = xp.arange(self.niter)
        action_list = xp.zeros(self.niter)
        density_list = xp.zeros(self.niter)
        for iter in range(self.niter):
            action_list[iter] = self.U_list[iter].plaquette_action()
            density_list[iter] = self.U_list[iter].density().real
        fig, ax = plt.subplots(1,2)
        ax[0].plot(n_list, action_list, ls='None', color='k', marker='o', markersize=3)
        ax[0].set_xlim([0,self.niter])
        ax[0].set_xlabel(r'$n_{\mathrm{iter}}}$')
        ax[0].set_ylabel(r'$S_G$')
        ax[1].plot(n_list * self.dt, density_list * (n_list * self.dt) ** 2, ls='None', color='k', marker='o', markersize=3)
        ax[1].set_xlim([0,self.niter * self.dt])
        ax[1].set_xlabel(r'$t$')
        ax[1].set_ylabel(r'$t^2 \langle E \rangle$')
        plt.tight_layout()
        plt.draw()
        plt.savefig('GFlow_test.pdf',transparent=True)

if __name__ == "__main__":
    from lqcd.io.backend import set_backend
    from lqcd.core.geometry import QCD_geometry
    import matplotlib.pyplot as plt
    set_backend("numpy")
    xp = get_backend()

    geometry = QCD_geometry([8, 4, 4, 4])
    U = Gauge(geometry)
    U.read("../algorithms/confs/beta_6.00_L4x8/beta_6.00_L4x8_conf_%d.h5" % 100)
    chi = Fermion(geometry)

    gflow = GFlow(U, chi, {"dt": 0.01, "niter": 10})
    gflow.forward()
    gflow.plot_action()
