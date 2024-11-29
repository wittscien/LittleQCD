from lqcd.io.backend import set_backend, get_backend
from lqcd.core.geometry import QCD_geometry
from lqcd.core.fields import Gauge, Fermion, Gamma, Propagator
from lqcd.fermion.Wilson import DiracOperator
from lqcd.gauge.smear import Smear as gSmear
from lqcd.fermion.smear import Smear as qSmear
from lqcd.algorithms.inverter import propagator
import lqcd.measurements.contract_funcs as cf
import lqcd.measurements.analysis_funcs as af
from opt_einsum import contract
import numpy as np
import matplotlib.pyplot as plt
import tqdm



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
for i in tqdm.tqdm(range(len(confs))):
    U = Gauge(geometry)
    U.read("../algorithms/confs/beta_6.00_L4x8/beta_6.00_L4x8_conf_%d.h5" % confs[i])

    # Gauge smear
    Smr = gSmear(U, {"tech": "APE", "alpha": 0.1, "niter": 10})
    U = Smr.APE_space()

    # Boundary condition
    U = U.apply_boundary_condition_periodic_quark()

    # Dirac operator
    Q = DiracOperator(U, {'fermion_type':'twisted_mass_clover', 'm': 3, 'mu': 0.1, 'csw': 0.1})

    # Inverter parameters
    inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": False}

    # Source
    quark_smr_params = {"tech": "Jacobi", "kappa": 0.1, "niter": 10}
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

fig, ax = plt.subplots(1,1)
meff = {}
meff_mean = {}
meff_err = {}
for i,k in enumerate(data.keys()):
    T = data[k].shape[1]
    x = np.arange(data[k].shape[1])
    meff[k] = np.zeros_like(data[k])
    for j in range(data[k].shape[0]):
        meff[k][j] = af.cal_mass(data[k][j],mtype='cosh',tau=1)
    meff_mean[k] = af.cal_mean(meff[k])
    meff_err[k] = af.cal_err(meff[k],tech=params['tech'])
    ax.errorbar(x=x+0.05*i,y=meff_mean[k],yerr=meff_err[k],ls='None',marker='o',capsize=2,fillstyle='none')
ax.axis([-0.2,T,1,18])
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$m_{\mathrm{eff}}$')
ax.legend()
plt.draw()
plt.savefig('meff_%s.pdf'%(k),transparent=True)
