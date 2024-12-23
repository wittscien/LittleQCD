from lqcd.io.backend import get_backend
from opt_einsum import contract



def epsilon_3d():
    xp = get_backend()
    epsilon = xp.zeros((3, 3, 3), dtype=int)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
    return epsilon

# Meson
def pion(Su, Sd):
    xp = get_backend()
    return contract('txyzBAba,txyzBAba->txyz', Su.field, xp.conjugate(Sd.field))

# Baryon cross
# A, B, C: Propagator
def T1(A, B, C):
    eps = epsilon_3d()
    return - contract('ijk,abc,txyzACia,txyzDCjb,txyzDBkc->txyzAB', eps, eps, A.field, B.field, C.field)

# Baryon direct
def T2(A, B, C):
    eps = epsilon_3d()
    return - contract('ijk,abc,txyzABia,txyzDCjb,txyzDCkc->txyzAB', eps, eps, A.field, B.field, C.field)

def phase_grid(momvec, Lx, Ly, Lz):
    xp = get_backend()
    px, py, pz = momvec
    x = xp.arange(Lx)
    y = xp.arange(Ly)
    z = xp.arange(Lz)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    phase = xp.exp(-1j * 2 * xp.pi * px * X / Lx) * xp.exp(-1j * 2 * xp.pi * py * Y / Ly) * xp.exp(-1j * 2 * xp.pi * pz * Z / Lz)
    return phase

# Momentum projection
def mom_proj(corr, momvec):
    xp = get_backend()
    if corr.ndim == 6:
        T, Lx, Ly, Lz, dim1, dim2 = corr.shape
        corr_type = 'baryon'
    elif corr.ndim == 4:
        T, Lx, Ly, Lz = corr.shape
        corr_type = 'meson'
    phase = phase_grid(momvec, Lx, Ly, Lz)
    if corr_type == 'baryon':
        projected_corr = xp.zeros((T, dim1, dim2), dtype=complex)
        for t in range(T):
            for mu in range(dim1):
                for nu in range(dim2):
                    projected_corr[t, mu, nu] = xp.sum(corr[t, :, :, :, mu, nu] * phase)
    elif corr_type == 'meson':
        projected_corr = xp.zeros(T, dtype=complex)
        for t in range(T):
            projected_corr[t] = xp.sum(corr[t] * phase)
    return projected_corr


def hpv_contract_project(contype, A, B, C, momvec):
    if contype == 'T1':
        corr_space = - T1(A, B, C)
    elif contype == 'T2':
        corr_space = - T2(A, B, C)

    corr_mom = mom_proj(corr_space, momvec)
    return corr_mom
