#%%
import lqcd.core as cr
from lqcd.io import set_backend, get_backend
from lqcd.gauge import Smear as gSmear
from lqcd.fermion import DiracOperator, Smear as qSmear
from lqcd.algorithms import Inverter
import lqcd.measurements.contract_funcs as cf
import lqcd.measurements.analysis_funcs as af
import lqcd.utils as ut
from opt_einsum import contract
import numpy as np
import matplotlib.pyplot as plt
import tqdm



if __name__ == "__main__":
    #%%
    # Initialization
    set_backend("numpy")
    xp = get_backend()

    # Gauge field
    geo_vec = [8, 4, 4, 4]
    geometry = cr.QCD_geometry(geo_vec)
    confs = xp.arange(400, 2020, 20, dtype=int)
    confs = xp.arange(600, 1020, 20, dtype=int)
    corr = {}
    corr['pion'] = xp.zeros((len(confs), geo_vec[0]), dtype=complex)
    corr['proton'] = xp.zeros((len(confs), geo_vec[0]), dtype=complex)
    for i in tqdm.tqdm(range(len(confs))):
        U = cr.Gauge(geometry)
        U.read("../algorithms/confs/beta_6.00_L4x8/beta_6.00_L4x8_conf_%d.h5" % confs[i])

        # Gauge smear
        Smr = gSmear(U, {"tech": "APE", "alpha": 0.1, "niter": 10})
        U_APE = Smr.smear()

        # Boundary condition
        U_with_phase = U.apply_boundary_condition_periodic_quark()

        # Dirac operator
        Q = DiracOperator(U_with_phase, {'fermion_type':'twisted_mass_clover', 'm': 0.5, 'mu': 0.1129943503, 'csw': 1.74})

        # Inverter parameters
        inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": False, "verbose": 0, "tm_rotation": True}

        # Source: point-to-all propagator
        quark_smr_params = {"tech": "Jacobi", "kappa": 0.2, "niter": 20}
        Smr = qSmear(U_APE, quark_smr_params)
        src = cr.Fermion(geometry)
        srcfull = cr.Propagator(geometry)
        for s in range(4):
            for c in range(3):
                src.point_source([0, 0, 0, 0, s, c])
                src = Smr.smear(src)
                srcfull.set_Fermion(src, s, c)

        # Propagator
        Su_ps = ut.propagator_parallelized(Q, inv_params, srcfull, 'u')
        Sd_ps = ut.propagator_parallelized(Q, inv_params, srcfull, 'd')

        # Sink smearing
        Su_ss = ut.prop_smear(Smr, Su_ps)
        Sd_ss = ut.prop_smear(Smr, Sd_ps)

        #%%
        # Meson Contraction
        pion = cf.pion(Su_ss, Su_ss)
        corr['pion'][i] = cf.mom_proj(pion, [0,0,0])

        #%%
        # Baryon Contraction
        Cg5 = 1j * cr.Gamma(2) * cr.Gamma(0) * cr.Gamma(5)
        GSdG = Cg5 * Sd_ss * Cg5
        proton_corr_4x4_space_t1 = - cf.T1(Su_ss, GSdG, Su_ss)
        proton_corr_4x4_space_t2 = - cf.T2(Su_ss, GSdG, Su_ss)
        proton_corr_4x4_mom = cf.mom_proj(proton_corr_4x4_space_t1 + proton_corr_4x4_space_t2, [0,0,0])
        gammat = cr.Gamma(0).mat
        # Parity projection
        P = (np.identity(4) + 1 * gammat) / 2
        proton = contract('ik,zkj,ji->z', P, proton_corr_4x4_mom, P)
        T = geo_vec[0]
        # BC fix
        corr['proton'][i] = np.exp(1j * 3 * np.pi * np.arange(T) / T) * np.roll(proton, -0)

    #%%
    # Plotting
    params = {'tech': 'jackknife', 'T': geo_vec[0]}
    relist = af.resamplelist(len(confs), params)
    bindata2 = {}
    bindata2 = corr
    relen = relist.shape[0]
    redata2 = {}
    for k in bindata2.keys():
        redata2[k] = np.zeros([relen,params['T']],dtype=complex)
    for k in bindata2.keys():
        if params['tech'] == 'bootstrap':
            for ls in range(relen):
                redata2[k][ls] = np.mean(bindata2[k][relist[ls]],axis=0)
        elif params['tech'] == 'jackknife':
            redata2[k][0] = np.mean(bindata2[k][relist[0]],axis=0)
            for ls in range(1, relen):
                redata2[k][ls] = np.mean(bindata2[k][relist[ls][:-1]],axis=0)
    data = redata2

    meff = {}
    meff_mean = {}
    meff_err = {}
    fig, ax = plt.subplots(1,1)
    for i,k in enumerate(data.keys()):
        T = data[k].shape[1]
        x = np.arange(data[k].shape[1])
        meff[k] = np.zeros_like(data[k])
        for j in range(data[k].shape[0]):
            meff[k][j] = af.cal_mass(data[k][j],mtype='cosh',tau=1)
        meff_mean[k] = af.cal_mean(meff[k])
        meff_err[k] = af.cal_err(meff[k],tech=params['tech'])
        ax.errorbar(x=x+0.05*i,y=meff_mean[k],yerr=meff_err[k],ls='None',marker='o',capsize=2,fillstyle='none',label=k)
    ax.axis([-0.2,T,1,10])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$m_{\mathrm{eff}}$')
    ax.legend()
    plt.draw()
    plt.savefig('meff.pdf',transparent=True)
