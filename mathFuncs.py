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


def cosHelicity(grandParent, parent, daughter):
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


def _sigmaZero(tail):
    tailSln4 = tail * _sln4

    return math.log(tailSln4 + math.sqrt(1.0 + tailSln4 ** 2)) / _sln4


def novosibirsk(x, norm, peak, width, tail):
    """Novosibirsk function
    See H. Ikeda et al. / Nuclear Instruments and Methods in Physics Research A 441 (2000) 401-426
    """

    x = np.asarray(x, dtype=float)

    lnArg = 1.0 - (x - peak) * tail / width

    log = np.ma.log(lnArg)

    sigmaZero2 = _sigmaZero(tail) ** 2

    exponent = -0.5 / sigmaZero2 * log ** 2 - 0.5 * sigmaZero2

    return np.where(lnArg > 1e-7, norm * np.exp(exponent), 0.0)


def novosibirskForTf1(x, params):
    """
    params[0]: norm
    params[1]: peak
    params[2]: width
    params[3]: tail
    """
    return novosibirsk(x[0], params[0], params[1], params[2], params[3])


# def novosibirsk_cdf(x, norm, peak, width, tail):
#     """Novosibirsk function
#     See H. Ikeda et al. / Nuclear Instruments and Methods in Physics Research A 441 (2000) 401-426
#     """
#
#     x = np.asarray(x, dtype=float)
#
#     sigmaZero = _sigmaZero(tail)
#
#     return np.sqrt(np.pi / 2.) * norm * width * sigmaZero / tail * erf(
#         (sigmaZero ** 2 - np.log(1 - (x - peak) / width)) / math.sqrt(2) / sigmaZero)
#
#
# def novosibirsk_norm(tail, width):
#     sigmaZero = _sigmaZero(tail)
#     return math.sqrt(2. / np.pi) * tail / sigmaZero / width / (erf(sigmaZero / math.sqrt(2)) - erf())


def listCenters(myList):
    myList = np.asarray(myList)

    return 0.5 * (myList[:-1] + myList[1:])


def gaussExp(x, norm, peak, sigma, tail):
    """
    Based on https://arxiv.org/pdf/1603.08591.pdf
    :param norm: Normalization
    :param peak: Gaussian peak location
    :param sigma: Gaussian sigma
    :param tail: Tail parameter
    """
    x = np.asarray(x, dtype=float)
    gausArg = (x - peak) / sigma

    return np.where(gausArg >= -tail, norm * np.exp(-0.5 * gausArg ** 2),
                    norm * np.exp(0.5 * tail ** 2 + tail * gausArg))


def gaussExpForTf1(x, params):
    """
    Works only for a single element array!
    params[0]: norm
    params[1]: peak
    params[2]: sigma
    params[3]: tail
    """

    return gaussExp(x[0], params[0], params[1], params[2], params[3])


def indicesPercentageOfMax(x, percentage):
    """
    Find last index greater than maxElement*percentage% from the left and right of maxElement - the maximum element in x
    :param x: list of numbers
    :param percentage: percentage out of 100
    :return: lowIdx, highIdx
    """
    x = np.asarray(x)
    maxElement = x.argmax()
    if np.abs(maxElement) < 1e-7:
        raise ValueError("Maximum element is {}. This won't work.".format(np.abs(maxElement)))

    threshold = percentage / 100. * x[maxElement]

    # start searching from maximum element

    # Index of first element that's smaller than threshold
    firstSmallerHighSide = np.argmax(x[maxElement:] < threshold)
    # argmax returns 0 if it doesn't find anything - in that case return maxBin
    # If it finds something, return the element before it
    highIdx = maxElement if firstSmallerHighSide == 0 else maxElement + firstSmallerHighSide - 1

    # Search in reverse order
    firstSmallerLowSide = np.argmax(np.flipud(x[:maxElement + 1]) < threshold)
    lowIdx = maxElement if firstSmallerLowSide == 0 else maxElement - firstSmallerLowSide + 1

    return lowIdx, highIdx


def calcPulls(measuredValues, stds, expectedValues):
    """
    Calculate pulls
    :param measuredValues:
    :param stds:
    :param expectedValues:
    :return: (measuredValues - expectedValues)/stds
    """
    measuredValues = np.atleast_1d(measuredValues)
    stds = np.atleast_1d(np.asarray(stds, dtype=float))
    expectedValues = np.atleast_1d(expectedValues)

    # sanity
    assert len(measuredValues) == len(stds), "measuredValues size {} != stds size {}".format(len(measuredValues),
                                                                                             len(stds))
    assert len(measuredValues) == len(expectedValues), "measuredValues size {} != expectedValues size {}".format(
        len(measuredValues), len(expectedValues))

    return (measuredValues - expectedValues) / stds
