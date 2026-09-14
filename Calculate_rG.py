import numpy as np
from scipy.integrate import quad


def gaussian_membership(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2)


def r_gaussian(mu1, sigma1, mu2, sigma2, epsabs=1e-12, epsrel=1e-12):


    if mu1 > mu2:
        mu1, sigma1, mu2, sigma2 = mu2, sigma2, mu1, sigma1


    G1 = lambda x: gaussian_membership(x, mu1, sigma1)
    G2 = lambda x: gaussian_membership(x, mu2, sigma2)

    I1, _ = quad(lambda x: G1(x) - G2(x), -np.inf, mu1, epsabs=epsabs, epsrel=epsrel)

    I2, _ = quad(lambda x: abs(G1(x) - G2(x)), mu1, mu2, epsabs=epsabs, epsrel=epsrel)

    I3, _ = quad(lambda x: G2(x) - G1(x), mu2, np.inf, epsabs=epsabs, epsrel=epsrel)

    return I1 + I2 + I3

