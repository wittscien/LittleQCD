from lqcd.io.backend import get_backend
from lqcd.core.geometry import QCD_geometry
from lqcd.core.fields import Fermion
from lqcd.core.fields import Gamma



geometry = QCD_geometry([4, 4, 4, 8])
source = Fermion(geometry)
g5 = Gamma(5)
