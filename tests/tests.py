import lqcd.core as cr
from lqcd.io import set_backend, get_backend
from lqcd.fermion import DiracOperator
import lqcd.utils as ut



set_backend("numpy")
xp = get_backend()

geometry = cr.QCD_geometry([8, 4, 4, 4])
U = cr.Gauge(geometry)
U.init_random()
src = cr.Fermion(geometry)
src.point_source([0, 0, 0, 0, 0, 0])
g5 = cr.Gamma(5)


params = {'fermion_type': 'twisted_mass_clover',
          'm': 0.1,
          'mu': 0.1,
          'csw': 0.1}
Q = DiracOperator(U, params)
a = Q.hopping(src)
b = Q.mass(src)
c = Q.clover(src)
d = Q.twisted_mass(src, 'u')
e = Q.Dirac(src, 'u')
