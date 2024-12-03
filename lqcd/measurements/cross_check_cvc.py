# Check log
# 2024.12.03: check with cvc on the random 4448 configuration for APE_space, Stout, plaquette_measure, apply_boundary_condition_periodic_quark. This means the underlying functions like U.shift() is working correctly.



from lqcd.io import set_backend, get_backend
from lqcd.core import *
from lqcd.gauge import Smear as gSmear
from lqcd.fermion import DiracOperator, Smear as qSmear
from lqcd.algorithms import Inverter, propagator
import lqcd.measurements.contract_funcs as cf
import lqcd.measurements.analysis_funcs as af
from opt_einsum import contract
import numpy as np
import matplotlib.pyplot as plt
import tqdm



def check(msg, a, b):
    print(msg+': ', a / b)
    if not np.isclose(a, b):
        raise ValueError("Regression check failed.")
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

U = Gauge(geometry)
xp.random.seed(0)
U.init_random()
# Print out to set explicitly in cvc
for t in range(geometry.T):
    for x in range(geometry.X):
        for y in range(geometry.Y):
            for z in range(geometry.Z):
                for mu in range(4):
                    for a in range(3):
                        for b in range(3):
                            loc_real = ((((((t*geometry.X+x)*geometry.Y+y)*geometry.Z+z)*geometry.Nl+mu)*geometry.Nc+a)*geometry.Nc+b)*2
                            # print("                if (loc_real == %d){*(g+loc_real) = %.16f; *(g+loc_imag) = %.16f;}" % (loc_real, U.field[t,x,y,z,mu,a,b].real, U.field[t,x,y,z,mu,a,b].imag))

#%%

# Gauge smear
check("Original gauge check", U.field[3,0,3,2,1,1,0].real, 0.3767504654460144)
Smr = gSmear(U, {"tech": "APE", "alpha": 0.1, "niter": 10})
U_smeared = Smr.APE_space()
check("APE smearing check", U_smeared.field[3,0,3,2,1,1,0].real, -0.06151140865874503)
check("Plaquette check", U_smeared.plaquette_measure() / (18 * geometry.T * geometry.X * geometry.Y * geometry.Z), 0.5328135787934447)
if 0:
    Smr = gSmear(U, {"tech": "Stout", "rho": 0.1, "niter": 10})
    U_smeared_Stout = Smr.Stout()
    check("Stout smearing check", U_smeared_Stout.field[3,0,3,2,1,1,0].real, -0.22355297609106276)

# Boundary condition
U_with_phase = U.apply_boundary_condition_periodic_quark()
check("BC check", U_with_phase.field[3,0,3,2,1,1,0].real, 0.3767504654460144)
check("Plaquette check", U_with_phase.plaquette_measure() / (18 * geometry.T * geometry.X * geometry.Y * geometry.Z), 0.11845355681410792)
exit()
# Dirac operator
Q = DiracOperator(U, {'fermion_type':'twisted_mass_clover', 'kappa': 0.177, 'mu': 0.003, 'csw': 1.74})

# Inverter parameters
inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": False}

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