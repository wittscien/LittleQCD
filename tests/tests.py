from lqcd.io.backend import get_backend
from lqcd.core.geometry import QCD_geometry
from lqcd.core.fields import fermion



geometry = QCD_geometry([4, 4, 4, 8])
source = fermion(geometry)
