from lqcd.io import set_backend, get_backend
from opt_einsum import contract



xp = get_backend()
sigma0 = xp.array([[1,0],[0,1]])
sigma1 = xp.array([[0,1],[1,0]])
sigma2 = xp.array([[0,-1j],[1j,0]])
sigma3 = xp.array([[1,0],[0,-1]])

def SU2_xvec(xvec):
    return xvec[0] * sigma0 + 1j * xvec[1] * sigma1 + 1j * xvec[2] * sigma2 + 1j * xvec[3] * sigma3

def SU2_eps(eps, rvec):
    xp = get_backend()
    xvec = xp.zeros(4, dtype=complex)
    xvec[0] = xp.sign(rvec[0]) * xp.sqrt(1 - eps ** 2)
    xvec[1:] = eps * xp.array(rvec[1:]) / xp.sqrt(xp.sum(xp.array(rvec[1:]) ** 2))
    return SU2_xvec(xvec)

def SU3_SU2(r, s, t):
    xp = get_backend()
    R = xp.array([
        [r[0, 0], r[0, 1], 0],
        [r[1, 0], r[1, 1], 0],
        [0, 0, 1]
    ])
    S = xp.array([
        [s[0, 0], 0, s[0, 1]],
        [0, 1, 0],
        [s[1, 0], 0, s[1, 1]]
    ])
    T = xp.array([
        [1, 0, 0],
        [0, t[0, 0], t[0, 1]],
        [0, t[1, 0], t[1, 1]]
    ])
    return R @ S @ T
