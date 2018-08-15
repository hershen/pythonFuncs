#!/usr/bin/python3
import numpy as np
import math
from scipy.special import erf

_sln4 = math.sqrt(math.log(4))


def vectorDot(v1, v2):
    """Return dot product of vectors v1 and v2

    Different from np.dot, np.inner in that if the input is an array of arrays, returns the right array."""

    return np.sum(np.multiply(v1, v2)) if isinstance(v1, list) or v1.ndim < 2 else np.sum(np.multiply(v1, v2), axis=1)


def radial_xyz(x, y, z):
    """ Return radial from x,y,z coordinates """
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def phi_xyz(x, y, z):
    """Return spherical phi coordinate from x,y,z cartezian coordinates"""
    return np.arctan2(y, x)


def cosTheta(x, y, z):
    """ Returns the cos(theta) of the cartezian vector (x,y,z) in radians"""
    return z / radial_xyz(x, y, z)


def angleBetween(v1, v2):
    """ Returns the angle in radians between vectors v1 and v2 """

    v1norms = np.linalg.norm(v1) if isinstance(v1, list) or v1.ndim < 2 else np.linalg.norm(v1, axis=1)
    v2norms = np.linalg.norm(v2) if isinstance(v2, list) or v2.ndim < 2 else np.linalg.norm(v2, axis=1)

    return np.arccos(np.true_divide(np.true_divide(vectorDot(v1, v2), v1norms), v2norms))


def lorentzDot(v1, v2):
    """ Perform 4 vector dot product
    Assume time like value at end"""

    multiplied = np.multiply(v1, v2)

    if isinstance(v1, list) or v1.ndim < 2:
        return multiplied[3] - np.sum(multiplied[:3])

    return multiplied[:, 3] - multiplied[:, 0] - multiplied[:, 1] - multiplied[:, 2]


def mass2(lorentzVector):
    """Get mass of lorentz vector squared"""
    return lorentzDot(lorentzVector, lorentzVector)


def mass(lorentzVector):
    """Get mass of lorentz vecot"""
    return np.sqrt(mass2(lorentzVector))


def cosHelicity1(grandParent, parent, daughter):
    """ Calculate cosine helicity of the daughter.
    grandParent, parent, daughter are the 4 vectors of the particles in any frame
    Taken from BAD522 v6, page 120, eq. 141
    grandparent, parent and daughter are P, Q, D in that equation
    """
    return (lorentzDot(grandParent, daughter) * mass2(parent) - lorentzDot(grandParent, parent) * lorentzDot(parent,
                                                                                                             daughter)) / np.sqrt(
        (lorentzDot(grandParent, parent) ** 2 - mass2(parent) * mass2(grandParent)) * (
                lorentzDot(parent, daughter) ** 2 - mass2(parent) * mass2(daughter)))


def effError(nom, denom):
    """Calculate efficiency error using binomial distribution
    eff = nom / denom

    Does not represent errors close to 0 or 1!!!
    Can be extended to include edge cases with CDF note 5894"""

    nom, denom = np.asarray(nom, dtype=float), np.asarray(denom)
    eff = nom / denom

    return np.sqrt(eff * (1 - eff) / denom)


def _sigmaZero(eta):
    etaSln4 = eta * _sln4

    return math.log(etaSln4 + math.sqrt(1.0 + etaSln4 ** 2)) / _sln4


def novosibirsk(x, norm, peak, width, eta):
    """Novosibirsk function
    See H. Ikeda et al. / Nuclear Instruments and Methods in Physics Research A 441 (2000) 401-426
    """

    x = np.asarray(x, dtype=float)

    # If the tail variable is small enough, this is just a Gaussian.
    if abs(eta) < 1e-7:
        return norm * np.exp(-0.5 * (x - peak) * (x - peak) / width / width)

    lnArg = 1.0 - (x - peak) * eta / width

    # Argument of logarithm negative. Real continuation -> function equals zero
    # calculate values only for lnArg non zero.
    lnArgNonZero = lnArg > 1e-7

    lnArg = lnArg[lnArgNonZero]

    log = np.log(lnArg)

    sigmaZero2 = _sigmaZero(eta) ** 2

    exponent = -0.5 / sigmaZero2 * log ** 2 - 0.5 * sigmaZero2

    result = np.zeros_like(x)

    result[lnArgNonZero] = norm * np.exp(exponent)

    return result


def novosibirsk_cdf(x, norm, peak, width, eta):
    """Novosibirsk function
    See H. Ikeda et al. / Nuclear Instruments and Methods in Physics Research A 441 (2000) 401-426
    """

    x = np.asarray(x, dtype=float)

    sigmaZero = _sigmaZero(eta)

    return np.sqrt(np.pi / 2.) * norm * width * sigmaZero / eta * erf(
        (sigmaZero ** 2 - np.log(1 - (x - peak) / width)) / math.sqrt(2) / sigmaZero)


def novosibirsk_norm(eta, width):
    sigmaZero = _sigmaZero(eta)
    return math.sqrt(2. / np.pi) * eta / sigmaZero / width / (erf(sigmaZero / math.sqrt(2)) - erf())


def listCenters(myList):
    myList = np.asarray(myList)

    return 0.5 * (myList[:-1] + myList[1:])
