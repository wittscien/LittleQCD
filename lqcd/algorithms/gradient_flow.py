import lqcd.core as cr
from lqcd.io import get_backend



class GFlow:
    def __init__(self, U: cr.Gauge, chi: cr.Fermion, params):
        self.U = U.copy()
        self.chi = chi.copy()
        self.geometry = U.geometry
        self.dt = params["dt"]
        self.niter = params["niter"]
        self.mu_num2st = {0: ['t', '-t'], 1: ['x', '-x'], 2: ['y', '-y'], 3: ['z', '-z']}
        self.mu_neg = {'t': '-t', '-t': 't', 'x': '-x', '-x': 'x', 'y': '-y', '-y': 'y', 'z': '-z', '-z': 'z'}

    def forward(self):
        # In place
        # I did not call Stout smearing because there is procedure like exp(8/9 Z1 - 17/36 Z0) W1, while the Stout smearing is just exp(Q)U.
        # For small lattices I can save them.
        self.U_list = []
        self.chi_list = []
        self.U_list.append(self.U.copy())
        self.chi_list.append(self.chi.copy())
        for _ in range(self.niter):
            # Step 0: W0 = U; phi0 = chi
            W0 = self.U.copy()
            phi0 = self.chi.copy()
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
            self.U = W3.copy()
            self.chi = phi3.copy()
            # Save to the list.
            self.U_list.append(self.U.copy())
            self.chi_list.append(self.chi.copy())
        return self.U.copy(), self.chi.copy()

    def adjoint(self, xi):
        # In place
        # Not using hierarchial scheme here. For small lattices I don't need to flow the gauge field for each flow time but just save them. I hope the memory is enough.
        # In my way, call forward() first to genrate the list of flowed gauge, and then call adjoint().
        self.xi = xi
        self.xi_list = []
        self.xi_list.append(self.xi.copy())
        for i in range(self.niter):
            U = self.U_list[self.niter - i - 1]
            # Flow the Gauge field, because I didn't save Z's.
            W0 = U.copy()
            Z0 = W0.Zgf() * self.dt
            W1 = (1/4 * Z0).to_exp() * W0
            Z1 = W1.Zgf() * self.dt
            W2 = (8/9 * Z1 - 17/36 * Z0).to_exp() * W1
            # Step 0: lambda3 = xi
            lambda3 = self.xi.copy()
            # Step 1: lambda2 = 3/4 Delta2 lambda3
            Delta2lambda3 = W2.laplacian(lambda3) * self.dt
            lambda2 = 3/4 * Delta2lambda3
            # Step 2: lambda1 = lambda3 + 8/9 Delta1 lambda2
            Delta1lambda2 = W1.laplacian(lambda2) * self.dt
            lambda1 = lambda3 + 8/9 * Delta1lambda2
            # Step 3: lambda0 = lambda1 + lambda2 + 1/4 Delta0 (lambda1 - 8/9 lambda2)
            Delta0lambda1 = W0.laplacian(lambda1) * self.dt
            Delta0lambda2 = W0.laplacian(lambda2) * self.dt
            lambda0 = lambda1 + lambda2 + 1/4 * Delta0lambda1 - 2/9 * Delta0lambda2
            # Set xi = lambda0
            self.xi = lambda0.copy()
            # Save to the list.
            self.xi_list.append(self.xi.copy())
        return self.xi.copy()

    def plot(self):
        xp = get_backend()
        n_list = xp.arange(self.niter + 1)
        action_list = xp.zeros(self.niter + 1)
        density_list = xp.zeros(self.niter + 1)
        # smear_list = xp.zeros((self.niter,self.geometry.X))
        # smearadj_list = xp.zeros((self.niter,self.geometry.X))
        for iter in range(self.niter + 1):
            action_list[iter] = self.U_list[iter].plaquette_action()
            density_list[iter] = self.U_list[iter].density().real
            # smear_list[iter] = self.chi_list[iter].field[0,:,0,0,0,0].real
            # smearadj_list[iter] = self.xi_list[iter].field[0,:,0,0,0,0].real
            # Test the inner product, this passed.
            # print(self.chi_list[iter].dot(self.xi_list[self.niter - iter - 1]))
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

        # Scale setting
        # sqrt(t0) = 0.166 fm
        t0 = n_list[xp.argmin(xp.abs(density_list - 0.3))] * self.dt
        a = 0.166 / xp.sqrt(t0)
        print("The lattice spacing a = %.4f fm" % a)

if __name__ == "__main__":
    from lqcd.io import set_backend
    import matplotlib.pyplot as plt
    set_backend("numpy")
    xp = get_backend()

    geometry = cr.QCD_geometry([8, 4, 4, 4])
    U = cr.Gauge(geometry)
    U.read("../algorithms/confs/beta_6.00_L4x8/beta_6.00_L4x8_conf_%d.h5" % 1000)
    chi = cr.Fermion(geometry)
    chi.point_source([0, 0, 0, 0, 0, 0])

    gflow = GFlow(U, chi, {"dt": 0.01, "niter": 100})
    gflow.forward()
    gflow.adjoint(gflow.chi)
    gflow.plot()
