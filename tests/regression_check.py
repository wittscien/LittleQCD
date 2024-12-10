# Check log

# 2024.12.08: The full propagator, the baryon contraction, and the mom projection pass the check.
# 2024.12.07: The inverter with tm rotation passed the check.
# 2024.12.06: The inverter passed the check; the adjoint gradient flow passed the check.
# 2024.12.04: The Dirac operator passed the check.
# 2024.12.03: Check with cvc on the random 4448 configuration for APE_space, Stout, plaquette_measure, apply_boundary_condition_periodic_quark. This means the underlying functions like U.shift() are working correctly.


#%%
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
scalar_test = Scalar(geometry)
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
    src_test.write("psi_random.h5")
    for t in range(geometry.T):
        for x in range(geometry.X):
            for y in range(geometry.Y):
                for z in range(geometry.Z):
                    for A in range(4):
                        for a in range(3):
                            loc_real = (((((t*geometry.X+x)*geometry.Y+y)*geometry.Z+z)*geometry.Ns+A)*geometry.Nc+a)*2
                            print("              if (loc_real == %d){*(s+loc_real) = %.16f; *(s+loc_imag) = %.16f;}" % (loc_real, src_test.field[t,x,y,z,A,a].real, src_test.field[t,x,y,z,A,a].imag))

    scalar_test.init_random()
    scalar_test.write("phi_random.h5")
    for t in range(geometry.T):
        for x in range(geometry.X):
            for y in range(geometry.Y):
                for z in range(geometry.Z):
                    loc_real = ((((t*geometry.X+x)*geometry.Y+y)*geometry.Z+z))*2
                    print("              if (loc_real == %d){*(s+loc_real) = %.16f; *(s+loc_imag) = %.16f;}" % (loc_real, scalar_test.field[t,x,y,z].real, scalar_test.field[t,x,y,z].imag))

else:
    U.read("U_random.h5")
    src_test.read("psi_random.h5")
    scalar_test.read("phi_random.h5")

#%%

# Test the gauge APE smearing
check("Original gauge", U.field[3,0,3,2,1,1,0].real, 0.3767504654460144)
Smr = gSmear(U, {"tech": "APE", "alpha": 0.1, "niter": 10})
U_APE = Smr.smear()
check("APE smearing", U_APE.field[3,0,3,2,1,1,0].real, -0.06151140865874503)
check("Plaquette", U_APE.plaquette_measure(), 0.5328135787934447)


# Test the gauge Stout smearing
Smr = gSmear(U, {"tech": "Stout", "rho": 0.1, "niter": 10})
U_Stout = Smr.smear()
check("Stout smearing", U_Stout.field[3,0,3,2,1,1,0].real, -0.22355297609106276)


# Test the boundary condition
U_with_phase = U.apply_boundary_condition_periodic_quark()
check("Boundary condition", U_with_phase.field[3,0,3,2,1,1,0].real, 0.3767504654460144)
check("Plaquette", U_with_phase.plaquette_measure(), 0.11845355681410792)


# Test the Dirac operator
# This point corresponds to ix = 4952, and 2480 for the field with even site. I find this by violently compare the field values.
Q = DiracOperator(U_with_phase, {'fermion_type':'twisted_mass_clover', 'kappa': 0.05, 'mu': 0.1, 'csw': 1.74})
check("Random spinor", src_test.field[3,0,3,2,1,1].real, 0.06540420131142144)
# The hopping term: -2.2875939515965786
# Without the hopping term: -0.40060389427277915.
check("Dirac", Q.Dirac(src_test, 'u')[3,0,3,2,1,1].real, -2.218080192524525)
check("Dirac", Q.Dirac(src_test, 'd')[3,0,3,2,1,1].real, -2.205255064147879)


# Test the inverter, without the tm rotation
# Inverter parameters
inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": False, "verbose": 0, "tm_rotation": False}
x0 = Fermion(geometry)
Inv = Inverter(Q, inv_params)
src_test_inv = Inv.invert(src_test, x0, 'u')
check("Inverter success", (Q.Dirac(src_test_inv, 'u')-src_test).field[3,0,3,2,1,1].real, 0)
check("Inverter", src_test_inv.field[3,0,3,2,1,1].real, 0.033941647499971334)


# Test the adjoint gradient flow
src = Fermion(geometry)
src.point_source([3,0,3,2,1,1])
gflow = GFlow(U_with_phase, Fermion(geometry), {"dt": 0.01, "niter": 20})
gflow.forward()
src_adj_flowed = gflow.adjoint(src)
check("Adjoint gradient flow", src_adj_flowed.field[3,0,3,2,1,1].real, 0.23570679544838258)


# Test the inverter on a adj flowed source, with the tm rotation also checked
inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": False, "verbose": 0, "tm_rotation": True}
Inv = Inverter(Q, inv_params)
prop_gflow_src_adj = Inv.invert(gflow.xi, x0, 'u')
check("Inverter with tm rotation", prop_gflow_src_adj.field[3,0,3,2,1,1].real, 0.00027630001053216384)


# Test the forward gradient flow, which must agree since the adjoint gradient flow which uses the forward flow passed the check.
gflow = GFlow(U_with_phase, prop_gflow_src_adj, {"dt": 0.01, "niter": 20})
_, prop_gflow_adj_fwd = gflow.forward()
check("Forward gradient flow", prop_gflow_adj_fwd.field[3,0,3,2,1,1].real, 9.596210982957447e-05)


# Test the Jacobian smearing and the full propagator
# Inverter parameters
inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": False, "tm_rotation": True}

# Source: point-to-all propagator
quark_smr_params = {"tech": "Jacobi", "kappa": 0.2, "niter": 20}
Smr = qSmear(U_APE, quark_smr_params)
src = Fermion(geometry)
srcfull = Propagator(geometry)
for s in range(4):
    for c in range(3):
        src.point_source([3, 0, 3, 2, s, c])
        src = Smr.smear(src)
        srcfull.set_Fermion(src, s, c)

# Propagator
Su_ps = propagator(Q, inv_params, srcfull, 'u')
Sd_ps = propagator(Q, inv_params, srcfull, 'd')
# Check the props at sink = t,x,y,z,s,c = 4,0,3,2,3,1 at src = t,x,y,z,s,c = [3,0,3,2,]1,1.
check("Propagator with Jacobi smearing", Su_ps.field[4,0,3,2,3,1,1,1].real * 1e5, -2.477784024521635e-05 * 1e5)
check("Propagator with Jacobi smearing", Sd_ps.field[4,0,3,2,3,1,1,1].real * 1e5, -2.4775045274727597e-05 * 1e5)

# Sink smearing
Su_ss = Smr.prop_smear(Su_ps)
Sd_ss = Smr.prop_smear(Sd_ps)
check("Propagator with sink source smeared", Su_ss.field[4,0,3,2,3,1,1,1].real * 1e8, -5.8107058079283876e-08 * 1e8)
check("Propagator with sink source smeared", Sd_ss.field[4,0,3,2,3,1,1,1].real * 1e8, -5.761446200984382e-08 * 1e8)

# Test the baryon Contraction
# t1 is a minus sign different from cvc
# mom projection is the inverse direction compared to cvc
cg5 = 1j * Gamma(1) * Gamma(3)
GSdG = cg5 * Sd_ss * cg5

check("Cg5 Prop Cg5", GSdG.field[4,0,3,2,3,1,1,1].real * 1e8, -4.289006283323454e-08 * 1e8)

# This is the same as vx of cvc
proton_corr_4x4_space_t1 = - cf.T1(Su_ss, GSdG, Su_ss)
proton_corr_4x4_space_t2 = - cf.T2(Su_ss, GSdG, Su_ss)

check("baryon 2pt T1", proton_corr_4x4_space_t1[4,0,3,2,3,1].real * 1e19, -9.110247193567789e-19 * 1e19)
check("baryon 2pt T2", proton_corr_4x4_space_t2[4,0,3,2,3,1].real * 1e19, -1.822381144456041e-18 * 1e19)

proton_corr_4x4_mom_t1 = cf.mom_proj(proton_corr_4x4_space_t1, [0,0,1])
proton_corr_4x4_mom_t2 = cf.mom_proj(proton_corr_4x4_space_t2, [0,0,1])

check("baryon 2pt mom projection", proton_corr_4x4_mom_t1[4,3,1].real * 1e19, 1.4061654013491913e-17 * 1e19)
