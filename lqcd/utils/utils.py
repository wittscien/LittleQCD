from lqcd.core.fields import Fermion, Gauge, Gamma



def sigma_munu(mu, nu):
    return (1 / 2j) * (Gamma(mu) * Gamma(nu) - Gamma(nu) * Gamma(mu))
