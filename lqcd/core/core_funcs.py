import scipy as sp
from lqcd.io.backend import get_backend



def proj_su3(A):
    # A: 3 x 3 complex matrix
    xp = get_backend()
    Ap = A @ xp.linalg.inv(xp.array(sp.linalg.sqrtm(xp.conjugate(xp.transpose(A)) @ A),dtype=complex))
    Ap = Ap / xp.linalg.det(Ap) ** (1/3)
    return Ap
