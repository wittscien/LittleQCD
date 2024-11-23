from lqcd.io.backend import set_backend, get_backend
from lqcd.core.geometry import QCD_geometry
from lqcd.core.fields import Gauge, Fermion, Gamma, Propagator
from lqcd.fermion.Wilson import DiracOperator
import lqcd.utils.utils as ut



set_backend("numpy")
xp = get_backend()

geometry = QCD_geometry([8, 4, 4, 4])
U = Gauge(geometry)
U.init_random()
src = Fermion(geometry)
src.point_source([0, 0, 0, 0, 0, 0])
g5 = Gamma(5)


params = {'m': 0.1,
          'mu': 0.1,
          'csw': 0.1}
Q = DiracOperator(U, params)
a = Q.hopping(src)
b = Q.mass(src)
c = Q.clover(src)
d = Q.twisted_mass(src, 'u')
e = Q.Dirac_twisted_mass_clover(src, 'u')
