# ut.check log

# 2024.12.08: The full propagator, the baryon contraction, and the mom projection pass the ut.check.
# 2024.12.07: The inverter with tm rotation passed the ut.check.
# 2024.12.06: The inverter passed the ut.check; the adjoint gradient flow passed the ut.check.
# 2024.12.04: The Dirac operator passed the ut.check.
# 2024.12.03: ut.check with cvc on the random 4448 configuration for APE_space, Stout, plaquette_measure, apply_boundary_condition_periodic_quark. This means the underlying functions like U.shift() are working correctly.


#%%
import lqcd.core as cr
from lqcd.io import set_backend, get_backend
from lqcd.gauge import Smear as gSmear
from lqcd.fermion import DiracOperator, Smear as qSmear
from lqcd.algorithms import Inverter, GFlow
import lqcd.measurements.contract_funcs as cf
import lqcd.measurements.analysis_funcs as af
from opt_einsum import contract
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import lqcd.utils as ut



if __name__ == "__main__":
    #%%
    # Initialization
    set_backend("numpy")
    xp = get_backend()

    # Gauge field
    geo_vec = [8, 4, 4, 4]
    geometry = cr.QCD_geometry(geo_vec)
    confs = xp.arange(100, 1000, 20, dtype=int)
    confs = xp.arange(100, 200, 20, dtype=int)
    corr = {}
    corr['pion'] = xp.zeros((len(confs), geo_vec[0]), dtype=complex)
    corr['proton'] = xp.zeros((len(confs), geo_vec[0]), dtype=complex)

    first_run = False
    U = cr.Gauge(geometry)
    src_test = cr.Fermion(geometry)
    scalar_test = cr.Scalar(geometry)
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

        scalar_test.init_random_Z2()
        scalar_test.write("phi_random.h5")
        for t in range(geometry.T):
            for x in range(geometry.X):
                for y in range(geometry.Y):
                    for z in range(geometry.Z):
                        loc_real = ((((t*geometry.X+x)*geometry.Y+y)*geometry.Z+z))*2
                        print("          if (loc_real == %d){*(s+loc_real) = %d; *(s+loc_imag) = %d;}" % (loc_real, scalar_test.field[t,x,y,z].real, scalar_test.field[t,x,y,z].imag))

    else:
        U.read("U_random.h5")
        src_test.read("psi_random.h5")
        scalar_test.read("phi_random.h5")

    #%%

    # Test the gauge APE smearing
    ut.check("Original gauge", U.field[3,0,3,2,1,1,0].real, 0.3767504654460144)
    Smr = gSmear(U, {"tech": "APE", "alpha": 0.1, "niter": 10})
    U_APE = Smr.smear()
    ut.check("APE smearing", U_APE.field[3,0,3,2,1,1,0].real, 0.3268317429489621)
    ut.check("Plaquette", U_APE.plaquette_measure(), 0.38147030356505085)


    # Test the gauge Stout smearing
    Smr = gSmear(U, {"tech": "Stout", "rho": 0.1, "niter": 10})
    U_Stout = Smr.smear()
    ut.check("Stout smearing", U_Stout.field[3,0,3,2,1,1,0].real, -0.22355297609106276)


    # Test the boundary condition
    U_with_phase = U.apply_boundary_condition_periodic_quark()
    ut.check("Boundary condition", U_with_phase.field[3,0,3,2,1,1,0].real, 0.3767504654460144)
    ut.check("Plaquette", U_with_phase.plaquette_measure(), 0.11845355681410792)


    # Test the Dirac operator
    # This point corresponds to ix = 4952, and 2480 for the field with even site. I find this by violently compare the field values.
    Q = DiracOperator(U_with_phase, {'fermion_type':'twisted_mass_clover', 'kappa': 0.05, 'mu': 0.1, 'csw': 1.74})
    ut.check("Random spinor", src_test.field[3,0,3,2,1,1].real, 0.06540420131142144)
    # The hopping term: -2.2875939515965786
    # Without the hopping term: -0.40060389427277915.
    ut.check("Dirac", Q.Dirac(src_test, 'u')[3,0,3,2,1,1].real, -2.218080192524525)
    ut.check("Dirac", Q.Dirac(src_test, 'd')[3,0,3,2,1,1].real, -2.205255064147879)


    # Test the inverter, without the tm rotation
    # Inverter parameters
    inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": False, "verbose": 0, "tm_rotation": False}
    x0 = cr.Fermion(geometry)
    Inv = Inverter(Q, inv_params)
    src_test_inv = Inv.invert(src_test, x0, 'u')
    ut.check("Inverter success", (Q.Dirac(src_test_inv, 'u')-src_test).field[3,0,3,2,1,1].real, 0)
    ut.check("Inverter", src_test_inv.field[3,0,3,2,1,1].real, 0.033941647499971334)


    # Test the adjoint gradient flow
    src = cr.Fermion(geometry)
    src.point_source([3,0,3,2,1,1])
    gflow = GFlow(U_with_phase, cr.Fermion(geometry), {"dt": 0.01, "niter": 20})
    gflow.forward()
    src_adj_flowed = gflow.adjoint(src)
    ut.check("Adjoint gradient flow", src_adj_flowed.field[3,0,3,2,1,1].real, 0.23570679544838258)


    # Test the inverter on a adj flowed source, with the tm rotation also ut.checked
    inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": False, "verbose": 0, "tm_rotation": True}
    Inv = Inverter(Q, inv_params)
    prop_gflow_src_adj = Inv.invert(gflow.xi, x0, 'u')
    ut.check("Inverter with tm rotation", prop_gflow_src_adj.field[3,0,3,2,1,1].real, 0.00027630001053216384)


    # Test the forward gradient flow, which must agree since the adjoint gradient flow which uses the forward flow passed the ut.check.
    gflow = GFlow(U_with_phase, prop_gflow_src_adj, {"dt": 0.01, "niter": 20})
    _, prop_gflow_adj_fwd = gflow.forward()
    ut.check("Forward gradient flow", prop_gflow_adj_fwd.field[3,0,3,2,1,1].real, 9.596210982957447e-05)


    # Test the Jacobian smearing and the full propagator
    # Inverter parameters
    inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": False, "tm_rotation": True}

    # Source: point-to-all propagator
    quark_smr_params = {"tech": "Jacobi", "kappa": 0.2, "niter": 20}
    Smr = qSmear(U_APE, quark_smr_params)
    src = cr.Fermion(geometry)
    srcfull = cr.Propagator(geometry)
    for s in range(4):
        for c in range(3):
            src.point_source([3, 0, 3, 2, s, c])
            src = Smr.smear(src)
            srcfull.set_Fermion(src, s, c)

    # Propagator
    Su_ps = ut.propagator_parallelized(Q, inv_params, srcfull, 'u')
    Sd_ps = ut.propagator_parallelized(Q, inv_params, srcfull, 'd')
    # Test the props at sink = t,x,y,z,s,c = 4,0,3,2,3,1 at src = t,x,y,z,s,c = [3,0,3,2,]1,1.
    ut.check("Propagator with Jacobi smearing", Su_ps.field[4,0,3,2,3,1,1,1].real * 1e5, -1.979621868593944e-05 * 1e5)
    ut.check("Propagator with Jacobi smearing", Sd_ps.field[4,0,3,2,3,1,1,1].real * 1e5, -1.980152358631995e-05 * 1e5)

    # Sink smearing
    Su_ss = ut.prop_smear(Smr, Su_ps)
    Sd_ss = ut.prop_smear(Smr, Sd_ps)
    ut.check("Propagator with sink source smeared", Su_ss.field[4,0,3,2,3,1,1,1].real * 1e8, -4.7245399084364914e-08 * 1e8)
    ut.check("Propagator with sink source smeared", Sd_ss.field[4,0,3,2,3,1,1,1].real * 1e8, -4.705661804453596e-08 * 1e8)

    # Test the baryon Contraction
    # t1 is a minus sign different from cvc
    # mom projection is the inverse direction compared to cvc
    Cg5 = 1j * cr.Gamma(2) * cr.Gamma(0) * cr.Gamma(5)
    GSdG = Cg5 * Sd_ss * Cg5

    ut.check("Cg5 Prop Cg5", GSdG.field[4,0,3,2,3,1,1,1].real * 1e8, -7.303735155437977e-08 * 1e8)

    # This is the same as vx of cvc
    proton_corr_4x4_space_t1 = - cf.T1(Su_ss, GSdG, Su_ss)
    proton_corr_4x4_space_t2 = - cf.T2(Su_ss, GSdG, Su_ss)

    ut.check("Baryon 2pt T1", proton_corr_4x4_space_t1[4,0,3,2,3,1].real * 1e19, 2.0441615571400817e-20 * 1e19)
    ut.check("Baryon 2pt T2", proton_corr_4x4_space_t2[4,0,3,2,3,1].real * 1e19, 4.3570256300920155e-20 * 1e19)

    proton_corr_4x4_mom_t1 = cf.mom_proj(proton_corr_4x4_space_t1, [0,0,1])
    proton_corr_4x4_mom_t2 = cf.mom_proj(proton_corr_4x4_space_t2, [0,0,1])

    ut.check("Baryon 2pt mom projection", proton_corr_4x4_mom_t1[4,3,1].real * 1e19, 6.710782747388034e-19 * 1e19)

    # Test the random scalar, for loops
    ut.check("Random scalar", scalar_test.field[3,0,3,2], 1)
