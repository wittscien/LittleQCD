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
        self.chi_list = []
        for _ in range(self.niter):
            # Save to the list.
            temp_U = Gauge(self.geometry)
            temp_U.field = xp.copy(self.U.field)
            temp_chi = Fermion(self.geometry)
            temp_chi.field = xp.copy(self.chi.field)
            self.U_list.append(temp_U)
            self.chi_list.append(temp_chi)
            # Step 0: W0 = U; phi0 = chi
            W0 = Gauge(self.geometry)
            W0.field = xp.copy(self.U.field)
            phi0 = Fermion(self.geometry)
            phi0.field = xp.copy(self.chi.field)
            # Step 1: W1 = exp(1/4 Z0) W0; phi1 = phi0 + 1/4 Delta0 phi0
            Z0 = W0.Zgf() * self.dt
            W1 = (1/4 * Z0).to_exp() * W0
            Delta0phi0 = W0.laplacian(phi0) * self.dt
            phi1 = phi0 + 1/4 * Delta0phi0
            # Step 2: W2 = exp(8/9 Z1 - 17/36 Z0) W1; phi2 = phi0 + 8/9 Delta1 phi1 - 2/9 Delta0 phi0
            Z1 = W1.Zgf() * self.dt
            W2 = (8/9 * Z1 - 17/36 * Z0).to_exp() * W1
            Delta1phi1 = W1.laplacian(phi1) * self.dt
            phi2 = phi0 + 8/9 * Delta1phi1 - 2/9 * Delta0phi0
            # Step 3: W3 = exp(3/4 Z2 - 8/9 Z1 + 17/36 Z0) W2; phi3 = phi1 + 3/4 Delta2 phi2
            Z2 = W2.Zgf() * self.dt
            W3 = (3/4 * Z2 - 8/9 * Z1 + 17/36 * Z0).to_exp() * W2
            Delta2phi2 = W2.laplacian(phi2) * self.dt
            phi3 = phi1 + 3/4 * Delta2phi2
            # Set U = W3; chi = phi3
            self.U.field = xp.copy(W3.field)
            self.chi.field = xp.copy(phi3.field)

    def plot(self):
        n_list = xp.arange(self.niter)
        action_list = xp.zeros(self.niter)
        density_list = xp.zeros(self.niter)
        # smear_list = xp.zeros((self.niter,self.geometry.X))
        for iter in range(self.niter):
            action_list[iter] = self.U_list[iter].plaquette_action()
            density_list[iter] = self.U_list[iter].density().real
            # smear_list[iter] = self.chi_list[iter].field[0,:,0,0,0,0].real
        fig, ax = plt.subplots(1,2,figsize=(2*5,4))
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
    chi.point_source([0, 0, 0, 0, 0, 0])

    gflow = GFlow(U, chi, {"dt": 0.01, "niter": 10})
    gflow.forward()
    gflow.plot()
