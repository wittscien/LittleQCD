import scipy as sp
from lqcd.io.backend import get_backend



class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def proj_su3(A):
    # A: 3 x 3 complex matrix
    xp = get_backend()
    Ap = A @ xp.linalg.inv(xp.array(sp.linalg.sqrtm(xp.conjugate(xp.transpose(A)) @ A),dtype=complex))
    Ap = Ap / xp.linalg.det(Ap) ** (1/3)
    return Ap
