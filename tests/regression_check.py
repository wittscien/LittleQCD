# Check log
# 2024.12.03: check with cvc on the random 4448 configuration for APE_space, Stout, plaquette_measure, apply_boundary_condition_periodic_quark. This means the underlying functions like U.shift() is working correctly.
# 2024.12.04: the Dirac operator passed the check.
# 2024.12.06: the inverter passed the check; the adjoint gradient flow passed the check.
# 2024.12.07: the inverter with tm rotation passed the check.



from lqcd.io import set_backend, get_backend
from lqcd.core import *
from lqcd.gauge import Smear as gSmear
from lqcd.fermion import DiracOperator, Smear as qSmear
from lqcd.algorithms import Inverter, propagator, GFlow
import lqcd.measurements.contract_funcs as cf
import lqcd.measurements.analysis_funcs as af
from opt_einsum import contract
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import lqcd.utils as ut



def check(msg, a, b):
    if np.isclose(a, b):
        print(f"{msg} {'check'} {ut.bcolors.OKGREEN}{'passed'}{ut.bcolors.ENDC}")
    if not np.isclose(a, b):
        raise ValueError(f"Regression check of {msg} {ut.bcolors.FAIL}{'failed'}{ut.bcolors.ENDC}")

#%%
# Initialization
set_backend("numpy")
xp = get_backend()

# Gauge field
geo_vec = [8, 4, 4, 4]
geometry = QCD_geometry(geo_vec)
confs = xp.arange(100, 1000, 20, dtype=int)
confs = xp.arange(100, 200, 20, dtype=int)
corr = {}
corr['pion'] = xp.zeros((len(confs), geo_vec[0]), dtype=complex)
corr['proton'] = xp.zeros((len(confs), geo_vec[0]), dtype=complex)

first_run = False
U = Gauge(geometry)
src_test = Fermion(geometry)
if first_run:
    xp.random.seed(0)
    U.init_random()
    U.write("U_random.h5")
    # Print out to set explicitly in cvc
    for t in range(geometry.T):
        for x in range(geometry.X):
            for y in range(geometry.Y):
                for z in range(geometry.Z):
                    for mu in range(4):
                        for a in range(3):
                            for b in range(3):
                                loc_real = ((((((t*geometry.X+x)*geometry.Y+y)*geometry.Z+z)*geometry.Nl+mu)*geometry.Nc+a)*geometry.Nc+b)*2
                                print("                if (loc_real == %d){*(g+loc_real) = %.16f; *(g+loc_imag) = %.16f;}" % (loc_real, U.field[t,x,y,z,mu,a,b].real, U.field[t,x,y,z,mu,a,b].imag))

    src_test.init_random()
    src_test.write("s_random.h5")
    for t in range(geometry.T):
        for x in range(geometry.X):
            for y in range(geometry.Y):
                for z in range(geometry.Z):
                    for A in range(4):
                        for a in range(3):
                            loc_real = (((((t*geometry.X+x)*geometry.Y+y)*geometry.Z+z)*geometry.Ns+A)*geometry.Nc+a)*2
                            print("              if (loc_real == %d){*(s+loc_real) = %.16f; *(s+loc_imag) = %.16f;}" % (loc_real, src_test.field[t,x,y,z,A,a].real, src_test.field[t,x,y,z,A,a].imag))
else:
    U.read("U_random.h5")
    src_test.read("s_random.h5")

#%%

# Test the gauge APE smearing
check("Original gauge", U.field[3,0,3,2,1,1,0].real, 0.3767504654460144)
Smr = gSmear(U, {"tech": "APE", "alpha": 0.1, "niter": 10})
U_smeared = Smr.APE_space()
check("APE smearing", U_smeared.field[3,0,3,2,1,1,0].real, -0.06151140865874503)
check("Plaquette", U_smeared.plaquette_measure(), 0.5328135787934447)

# Test the gauge Stout smearing
Smr = gSmear(U, {"tech": "Stout", "rho": 0.1, "niter": 10})
U_smeared_Stout = Smr.Stout()
check("Stout smearing", U_smeared_Stout.field[3,0,3,2,1,1,0].real, -0.22355297609106276)

# Test the boundary condition
U_with_phase = U.apply_boundary_condition_periodic_quark()
check("Boundary condition", U_with_phase.field[3,0,3,2,1,1,0].real, 0.3767504654460144)
check("Plaquette", U_with_phase.plaquette_measure(), 0.11845355681410792)

# Test the Dirac operator
# This point corresponds to ix = 4952, and 2480 for the field with even site. I find this by violently compare the field values.
Q = DiracOperator(U_with_phase, {'fermion_type':'twisted_mass_clover', 'kappa': 0.177, 'mu': 0.1129943503, 'csw': 1.74})
check("Random spinor", src_test.field[3,0,3,2,1,1].real, 0.06540420131142144)
# The hopping term: -2.2875939515965786
# Without the hopping term: -0.40060389427277915.
check("Dirac", Q.Dirac(src_test, 'u')[3,0,3,2,1,1].real, -2.6881978458693574)
check("Dirac", Q.Dirac(src_test, 'd')[3,0,3,2,1,1].real, -2.6737061753850258)

# Test the inverter, without the tm rotation
# Inverter parameters
inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": True, "verbose": False, "tm_rotation": False}
x0 = Fermion(geometry)
Inv = Inverter(Q, inv_params)
src_test_inv = Inv.invert(src_test, x0, 'u')
check("Inverter success", (Q.Dirac(src_test_inv, 'u')-src_test).field[3,0,3,2,1,1].real, 0)
check("Inverter", src_test_inv.field[3,0,3,2,1,1].real, -0.07115880763279339)

# Test the adjoint gradient flow
src = Fermion(geometry)
src.point_source([3,0,3,2,1,1])
gflow = GFlow(U_with_phase, Fermion(geometry), {"dt": 0.01, "niter": 20})
gflow.forward()
src_adj_flowed = gflow.adjoint(src)
check("Adjoint gradient flow", src_adj_flowed.field[3,0,3,2,1,1].real, 0.23570679544838258)

# Test the inverter on a adj flowed source, with the tm rotation also checked
inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": True, "verbose": False, "tm_rotation": True}
Inv = Inverter(Q, inv_params)
prop_gflow_src_adj = Inv.invert(gflow.xi, x0, 'u')
check("Inverter with tm rotation", prop_gflow_src_adj.field[3,0,3,2,1,1].real, 0.021521391753250213)

# Test the forward gradient flow, which must agree since the adjoint gradient flow which uses the forward flow passed the check.
gflow = GFlow(U_with_phase, prop_gflow_src_adj, {"dt": 0.01, "niter": 20})
_, prop_gflow_adj_fwd = gflow.forward()
check("Forward adjoint flow", prop_gflow_adj_fwd.field[3,0,3,2,1,1].real, 0.010707123866497949)

exit()

# Inverter parameters
inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": False, "tm_rotation": True}

# Source: point-to-all propagator
quark_smr_params = {"tech": "Jacobi", "kappa": 0.2, "niter": 20}
srcfull = Propagator(geometry)
for s in range(4):
    for c in range(3):
        src = Fermion(geometry)
        src.point_source([0, 0, 0, 0, s, c])
        Smr = qSmear(U, src, quark_smr_params)
        src = Smr.Jacobi_smear()
        srcfull.field[:,:,:,:,:,s,:,c] = src.field

# Propagator
Su = propagator(Q, inv_params, srcfull, 'u')
Sd = propagator(Q, inv_params, srcfull, 'd')

# Stochastic propagator -> loop, manually construct the loop.
Loop = Propagator(geometry)
for t in range(geometry.T):
    for s in range(4):
        for c in range(3):
            src = Fermion(geometry)
            src.Z2_stochastic_time_spin_color_diluted_source(t,s,c)
            x0 = Fermion(geometry)
            Inv = Inverter(Q, inv_params)
            phi = Inv.invert(src, x0, 'u')
            Loop.field[t,:,:,:,:,s,:,c] = 0 #src.x_dot(phi)

#%%
# Meson Contraction
pion = cf.pion(Su, Sd)
corr['pion'][i] = cf.mom_proj(pion, [0,0,0])

#%%
# Baryon Contraction
cg5 = 1j * Gamma(1) * Gamma(3)
GSdG = cg5 * Sd * cg5
t1 = cf.T1(Su, GSdG, Su)
t2 = cf.T2(Su, GSdG, Su)
proton = cf.mom_proj(t1+t2, [0,0,0])
gammat = Gamma(0).mat
# Parity projection
P = (np.identity(4) + 1 * gammat) / 2
proton = contract('ik,zkj,ji->z', P, proton, P)
T = geo_vec[0]
# BC fix
corr['proton'][i] = np.exp(1j * 3 * np.pi * np.arange(T) / T) * np.roll(proton, -0)
