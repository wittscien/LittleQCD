from multiprocessing import Pool
import lqcd.core as cr
from lqcd.io import set_backend, get_backend
from lqcd.gauge import Smear as gSmear
from lqcd.fermion import DiracOperator, Smear as qSmear
from lqcd.algorithms import Inverter, GFlow
import lqcd.measurements.contract_funcs as cf
import lqcd.measurements.analysis_funcs as af
from opt_einsum import contract
import numpy as np
import matplotlib.pyplot as plt
import tqdm



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


def check(msg, a, b):
    if np.isclose(a, b):
        print(f"{msg} {'check'} {bcolors.OKGREEN}{'passed'}{bcolors.ENDC}")
    if not np.isclose(a, b):
        raise ValueError(f"Regression check of {msg} {bcolors.FAIL}{'failed'}{bcolors.ENDC}, new = {a}")


# full propagator
def propagator(Q, inv_params, srcfull, flavor):
    geometry = Q.geometry
    x0 = cr.Fermion(geometry)
    Inv = Inverter(Q, inv_params)
    prop = cr.Propagator(geometry)
    for s in range(4):
        for c in range(3):
            src = srcfull.to_Fermion(s,c)
            prop.set_Fermion(Inv.invert(src, x0, flavor), s, c)
    return prop


# full propagator parallelized
def propagator_parallelized_core(s, c, srcfull, x0, Inv, flavor):
    src = srcfull.to_Fermion(s,c)
    return Inv.invert(src, x0, flavor)


# full propagator parallelized
# If the code is really used for production, I think I don't use the parallelization here but brute-force parallel the configurations.
# Only around 2 times faster, though processes = 12.
def propagator_parallelized(Q, inv_params, srcfull, flavor):
    geometry = Q.geometry
    x0 = cr.Fermion(geometry)
    Inv = Inverter(Q, inv_params)
    prop = cr.Propagator(geometry)
    with Pool(processes = 12) as pool:
        pool_result = pool.starmap(propagator_parallelized_core, [(s, c, srcfull, x0, Inv, flavor) for s in range(4) for c in range(3)])
    for i in range(len(pool_result)):
        prop.set_Fermion(pool_result[i], i // 3, i % 3)
    return prop


# propagator sink smearing
def prop_smear(smear: qSmear, prop: cr.Propagator):
    result = cr.Propagator(prop.geometry)
    for s in range(prop.geometry.Ns):
        for c in range(prop.geometry.Nc):
            result.set_Fermion(smear.smear(prop.to_Fermion(s,c)), s, c)
    return result


# Forward flowing the propagator-like objects, including the srcfull
def prop_fwd_flow(U, gflow_params, prop: cr.Propagator):
    result = cr.Propagator(prop.geometry)
    for s in range(prop.geometry.Ns):
        for c in range(prop.geometry.Nc):
            flow = GFlow(U, prop.to_Fermion(s,c), gflow_params)
            fermion_fwd_flowed = flow.forward()[1]
            result.set_Fermion(fermion_fwd_flowed, s, c)
    return result


# Adjoint flowing the propagator-like objects, including the srcfull
def prop_adj_flow(U, gflow_params, prop: cr.Propagator):
    result = cr.Propagator(prop.geometry)
    for s in range(prop.geometry.Ns):
        for c in range(prop.geometry.Nc):
            flow = GFlow(U, cr.Fermion(prop.geometry), gflow_params)
            flow.forward()
            fermion_adj_flowed = flow.adjoint(prop.to_Fermion(s,c))
            result.set_Fermion(fermion_adj_flowed, s, c)
    return result
