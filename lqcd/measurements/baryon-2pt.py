from lqcd.io.backend import set_backend, get_backend
from lqcd.core.geometry import QCD_geometry
from lqcd.core.fields import Gauge, Fermion, Gamma, Propagator
from lqcd.fermion.Wilson import DiracOperator
from lqcd.fermion.smear import Smear
from lqcd.algorithms.inverter import Inverter, propagator
import lqcd.utils.utils as ut
import lqcd.measurements.contract_funcs as cf



#%%
# Initialization
set_backend("numpy")
xp = get_backend()

# Gauge field
geometry = QCD_geometry([8, 4, 4, 4])
U = Gauge(geometry)
U.init_random()

# Dirac operator
Q = DiracOperator(U, {'fermion_type':'twisted_mass_clover', 'm': 3, 'mu': 0.1, 'csw': 0.1})

# Inverter parameters
inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": False}

# Source
smr_params = {"tech": "Jacobi", "kappa": 0.1, "niter": 10}
srcfull = Propagator(geometry)
for s in range(4):
    for c in range(3):
        src = Fermion(geometry)
        src.point_source([0, 0, 0, 0, s, c])
        Smr = Smear(U, src, smr_params)
        src = Smr.Jacobi_smear()
        srcfull.field[:,:,:,:,:,s,:,c] = src.field

# Propagator
Su = propagator(Q, inv_params, srcfull, 'u')
Sd = propagator(Q, inv_params, srcfull, 'd')

#%%
# Contraction
cg5 = 1j * Gamma(1) * Gamma(3)
GSdG = cg5 * Sd * cg5
t1 = cf.T1(Su, GSdG, Su)
t2 = cf.T2(Su, GSdG, Su)
corr = cf.mom_proj(t1+t2, [0,0,0])
