from lqcd.io.backend import set_backend, get_backend
from lqcd.core.geometry import QCD_geometry
from lqcd.core.fields import Gauge, Fermion, Gamma, Propagator
from lqcd.fermion.Wilson import DiracOperator


set_backend("numpy")
xp = get_backend()

geometry = QCD_geometry([4, 4, 4, 8])
U = Gauge(geometry)
U.init_random()
src = Fermion(geometry)
src.point_source([0, 0, 0, 0, 0, 0])
g5 = Gamma(5)


params = {'m': 0.1,
          'csw': 0.1}
Q = DiracOperator(U, params)
a = Q.hopping(src)
b = U.plaquette('t', 'x')
c = U.clover('t', 'x')
