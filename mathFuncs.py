#!/usr/bin/python3
import math

from numba import njit, vectorize
import numpy as np
from scipy import stats
from scipy.special import erf

import ROOT

_sln4 = np.sqrt(np.log(4))


def vectorDot(v1, v2):
    """Return dot product of vectors v1 and v2

    Different from np.dot, np.inner in that if the input is an array of arrays, returns the right array."""

    return np.sum(np.multiply(v1, v2)) if isinstance(v1, list) or v1.ndim < 2 else np.sum(np.multiply(v1, v2), axis=1)


def radial_xyz(x, y, z):
    """ Return radial from x,y,z coordinates """
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def phi_xyz(x, y, _):
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
    return (lorentzDot(grandParent, daughter) * mass2(parent) - lorentzDot(grandParent, parent) *
            lorentzDot(parent, daughter)) / np.sqrt(
        (lorentzDot(grandParent, parent) ** 2 - mass2(parent) * mass2(grandParent)) * (
                lorentzDot(parent, daughter) ** 2 - mass2(parent) * mass2(daughter)))


def effError(nom, denom):
    """Calculate efficiency error using binomial distribution
    eff = nom / denom

    Does not represent errors close to 0 or 1!!!
    Can be extended to include edge cases with CDF note 5894"""

    nom, denom = np.asarray(nom, dtype=float), np.asarray(denom)
    eff = nom / denom

    return np.sqrt(eff * (1 - eff) / (denom-1))


def _sigmaZero(tail):
    tailSln4 = tail * _sln4

    return np.log(tailSln4 + np.sqrt(1.0 + tailSln4 ** 2)) / _sln4


def novosibirsk(x, peak, width, tail):
    """Novosibirsk function
    See H. Ikeda et al. / Nuclear Instruments and Methods in Physics Research A 441 (2000) 401-426
    """

    x = np.asarray(x, dtype=float)

    lnArg = 1.0 - (x - peak) * tail / width

    log = np.ma.log(lnArg)

    sigmaZero2 = _sigmaZero(tail) ** 2

    exponent = -0.5 / sigmaZero2 * log ** 2 - 0.5 * sigmaZero2

    return np.where(lnArg > 1e-7, np.exp(exponent), 0.0)


def novosibirskForTf1(x, params):
    """
    params[0]: norm
    params[1]: peak
    params[2]: width
    params[3]: tail
    """

    return params[0] * novosibirsk(x[0], params[1], params[2], params[3])


def listCenters(myList):
    myList = np.asarray(myList)

    return 0.5 * (myList[:-1] + myList[1:])


def gaussExp(x, peak, sigma, tail):
    """
    Modified version of https://arxiv.org/pdf/1603.08591.pdf, that allows both high and low tails.
    Inspired by https://github.com/souvik1982/GaussExp/blob/master/RooFitImplementation/RooGaussExp.cxx
    :param x:
    :param peak: Gaussian peak location
    :param sigma: Gaussian sigma
    :param tail: Tail parameter. Can be any value.

    Note - There is an assymetry when tail = 0.
    """
    x = np.asarray(x, dtype=float)

    gausArg = (x - peak) / sigma if tail <= 0 else (peak - x) / sigma

    absTail = abs(tail)
    return np.where(gausArg >= -absTail, np.exp(-0.5 * gausArg ** 2),
                    np.exp(0.5 * tail ** 2 + absTail * gausArg))


def gaussExpForTf1(x, params):
    """
    Works only for a single element array!
    params[0]: norm
    params[1]: peak
    params[2]: sigma
    params[3]: tail
    """

    return params[0] * gaussExp(x[0], params[1], params[2], params[3])


def gaussExpNormalizationWholeRange(mean, sigma, tail):
    #First term is regular gaussian. Second term is of the exponantial decaying.
    return sigma*np.sqrt(np.pi/2)*(1+erf(np.abs(tail)/np.sqrt(2))) + sigma/np.abs(tail)*np.exp(-tail**2/2)

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


def valuesPercentageOfMax(x, percentage):
    """
    Return last values greater than maxElement*percentage% from the left and right of maxElement - the maximum element in x
    :param x:
    :param percentage:
    :return:
    """
    lowIdx, highIdx = indicesPercentageOfMax(x, percentage)
    return x[lowIdx], x[highIdx]


def calcPulls(measuredValues, stds, expectedValues):
    """
    Calculate pulls (measuredValues - expectedValues) / stds
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


@vectorize(nopython=True)
def expGaussExp(x, peak, sigma, tailLow, tailHigh):
    """
    From https://arxiv.org/pdf/1603.08591.pdf.
    Inspired by https://github.com/souvik1982/GaussExp/blob/master/RooFitImplementation/RooGaussDoubleSidedExp.cxx
    :param x:
    :param peak:
    :param sigma:
    :param tailLow:
    :param tailHigh:
    :return:
    """
    gausArg = (x - peak) / sigma

    if gausArg < -tailLow:
        return np.exp(0.5 * tailLow ** 2 + tailLow * gausArg)
    elif gausArg > tailHigh:
        return np.exp(0.5 * tailHigh ** 2 - tailHigh * gausArg)
    else:
        return np.exp(-0.5 * gausArg ** 2)


# def expGaussExp(x, peak, sigma, tailLow, tailHigh):
#     """
#     From https://arxiv.org/pdf/1603.08591.pdf.
#     Inspired by https://github.com/souvik1982/GaussExp/blob/master/RooFitImplementation/RooGaussDoubleSidedExp.cxx
#     :param x:
#     :param peak:
#     :param sigma:
#     :param tailLow:
#     :param tailHigh:
#     :return:
#     """
#     return expGaussExp_numba(np.asarray(x, dtype=float), peak, sigma, tailLow, tailHigh)

@vectorize
def expGaussExp_FWHM_xHigh(peak, sigma, tailHigh):
    """
    Return the x value for which expGaussExp(x) = 0.5*expGaussExp(peak), on the high side tail
    :param peak:
    :param sigma:
    :param tailHigh:
    :return:
    """
    return peak + sigma * _sln4 if tailHigh >= _sln4 else peak + sigma * (np.log(2) / tailHigh + 0.5 * tailHigh)


@vectorize
def expGaussExp_FWHM_xLow(peak, sigma, tailLow):
    """
    Return the x value for which expGaussExp(x) = 0.5*expGaussExp(peak), on the low side tail
    :param peak:
    :param sigma:
    :param tailLow:
    :return:
    """
    return peak - sigma * _sln4 if tailLow >= _sln4 else peak - sigma * (np.log(2) / tailLow + 0.5 * tailLow)


@vectorize
def expGaussExp_FWHM(peak, sigma, tailLow, tailHigh):
    """
    Return FWHM of expGaussExp
    :param peak:
    :param sigma:
    :param tailLow:
    :param tailHigh:
    :return:
    """
    return expGaussExp_FWHM_xHigh(peak, sigma, tailHigh) - expGaussExp_FWHM_xLow(peak, sigma, tailLow)


@vectorize
def expGaussExp_gausEqeuivalentSigma(peak, sigma, tailLow, tailHigh):
    """
    Return FWHM of expGaussExp / (2 * sqrt(ln(4)) : (Gaussian equivalent of sigma)
    :param peak:
    :param sigma:
    :param tailLow:
    :param tailHigh:
    :return:
    """
    return expGaussExp_FWHM(peak, sigma, tailLow, tailHigh) / 2 / _sln4


def expGaussExpForTf1(x, params):
    """
    Works only for a single element array!
    params[0]: norm
    params[1]: peak
    params[2]: sigma
    params[3]: tailLow - low side tail parameter. Can't be negative.
    params[4]: tailHigh - high side tail parameter. Can't be negative.
    """

    return params[0] * expGaussExp(x[0], params[1], params[2], params[3], params[4])


def expGaussExp_integral(norm, peak, sigma, tailLow, tailHigh, xMin, xMax):
    tMin = (xMin - peak) / sigma
    tMax = (xMax - peak) / sigma

    result = 0.0
    if tMin < -tailLow:
        result += np.exp(0.5 * tailLow ** 2) * sigma / tailLow * (
                np.exp(min(tMax, -tailLow) * tailLow) - np.exp(tMin * tailLow))

    if tMin < tailHigh and tMax > -tailLow:
        result += np.sqrt(2 * np.pi) * sigma * (
                stats.norm.cdf(min(tMax, tailHigh)) - stats.norm.cdf(max(tMin, -tailLow)))

    if tMax > tailHigh:
        result += np.exp(0.5 * tailHigh ** 2) * sigma / (-tailHigh) * (
                np.exp(tMax * (-tailHigh)) - np.exp(max(tMin, tailHigh) * (-tailHigh)))

    return norm * result


def crystalBall(x, peak, sigma, alpha, n):
    """
    From https: // arxiv.org / pdf / 1603.08591.pdf.
    Inspired by http: // roofit.sourceforge.net / docs / classref // src / RooCBShape.cxx.html  # RooCBShape:evaluate
    :param x:
    :param peak:
    :param sigma:
    :param alpha:
    :param n:
    :return:
    """

    x = np.atleast_1d(x)

    gausArg = (x - peak) / sigma if alpha >= 0 else (peak - x) / sigma

    absAlpha = abs(alpha)

    # using np.ma to prevent warning of invalid values

    return np.where(gausArg >= -absAlpha,
                    np.exp(-0.5 * gausArg ** 2),
                    (n / absAlpha) ** n * np.exp(-0.5 * absAlpha ** 2) / np.ma.power(n / absAlpha - absAlpha - gausArg,
                                                                                     n))


def crystalBallForTf1(x, params):
    """
    Works only for a single element array!
    params[0]: norm
    params[1]: peak
    params[2]: sigma
    params[3]: alpha
    params[4]: n
    """

    return params[0] * crystalBall(x[0], params[1], params[2], params[3], params[4])


def doubleSidedCrystalBall(x, peak, sigma, alphaLow, alphaHigh, nLow, nHigh):
    """
    From https://arxiv.org/pdf/1505.01609.pdf, p.5, margin.
    That had a mistake - the sign of the gausArg in the denominatro.

    :param x:
    :param peak:
    :param sigma:
    :param alphaLow:
    :param alphaHigh:
    :param nLow:
    :param nHigh:
    :return:
    """

    x = np.ma.asarray(x, dtype=float)
    gausArg = (x - peak) / sigma

    conditions = [gausArg <= -alphaLow, gausArg >= alphaHigh, True]

    absAlphaLow = abs(alphaLow)
    absAlphaHigh = abs(alphaHigh)

    return np.select(conditions,
                     [np.ma.asarray((nLow / absAlphaLow) ** nLow * np.exp(-0.5 * alphaLow ** 2)) / np.ma.power(
                         nLow / absAlphaLow - absAlphaLow - gausArg, nLow),
                      np.ma.asarray((nHigh / absAlphaHigh) ** nHigh * np.exp(-0.5 * alphaHigh ** 2)) / np.ma.power(
                          nHigh / absAlphaHigh - absAlphaHigh + gausArg, nHigh),
                      np.exp(-0.5 * gausArg ** 2)])


def doubleSidedCrystalBallForTf1(x, params):
    """
    Works only for a single element array!
    param[0] norm
    param[1] peak
    param[2] sigma
    param[3] alphaLow
    param[4] alphaHigh
    param[5] nLow
    param[6] nHigh
    """

    return params[0] * doubleSidedCrystalBall(x[0], params[1], params[2], params[3], params[4], params[5],
                                              params[6])


def doubleSidedCrystalBall_FWHM_xHigh(peak, sigma, tailHigh, nHigh):
    """
    Return the x value for which doubleSidedCrystalBall(x) = 0.5*doubleSidedCrystalBall(peak), on the high side tail
    :param peak:
    :param sigma:
    :param tailHigh:
    :return:
    """
    absTailHigh = abs(tailHigh)
    return peak + sigma * _sln4 if tailHigh >= _sln4 else peak + sigma * (
            nHigh / absTailHigh * (2 * math.exp(-tailHigh ** 2 / 2)) ** (
            1. / nHigh) + absTailHigh - nHigh / absTailHigh)


def doubleSidedCrystalBall_FWHM_xLow(peak, sigma, tailLow, nLow):
    """
    Return the x value for which doubleSidedCrystalBall(x) = 0.5*doubleSidedCrystalBall(peak), on the low side tail
    :param peak:
    :param sigma:
    :param tailLow:
    :return:
    """
    absTailLow = abs(tailLow)
    return peak - sigma * _sln4 if tailLow >= _sln4 else peak - sigma * (
            nLow / absTailLow * (2 * math.exp(-tailLow ** 2 / 2)) ** (1. / nLow) + absTailLow - nLow / absTailLow)


def doubleSidedCrystalBall_FWHM(peak, sigma, tailLow, tailHigh, nLow, nHigh):
    """
    Return FWHM of doubleSidedCrystalBall
    :param peak:
    :param sigma:
    :param tailLow:
    :param tailHigh:
    :return:
    """
    return doubleSidedCrystalBall_FWHM_xHigh(peak, sigma, tailHigh, nHigh) - doubleSidedCrystalBall_FWHM_xLow(peak,
                                                                                                              sigma,
                                                                                                              tailLow,
                                                                                                              nLow)


def doubleSidedCrystalBall_gausEqeuivalentSigma(peak, sigma, tailLow, tailHigh, nLow, nHigh):
    """
    Return FWHM of doubleSidedCrystalBall / (2 * sqrt(ln(4)) : (Gaussian equivalent of sigma)
    :param peak:
    :param sigma:
    :param tailLow:
    :param tailHigh:
    :return:
    """
    return doubleSidedCrystalBall_FWHM(peak, sigma, tailLow, tailHigh, nLow, nHigh) / 2 / _sln4


def idxFirst(data, comp, threshold, startingIdx=0):
    """
    Return index of first element that has relation comp to threshold. If none exists, return None.
    :param data:
    :param comp: Mathematical comparator. Meant to be used with operator module.
    :param threshold:
    :param startingLocation:
    :return:
    """
    try:
        return startingIdx + next(idx for idx, value in enumerate(data[startingIdx:]) if comp(value, threshold))
    except StopIteration:
        return None


def idxFirstToLeft(data, comp, threshold, startingIdx=0):
    """
    Return index of first element that has relation comp to threshold, when searching to the left. If none exists, return None.
    :param data:
    :param comp: Mathematical comparator. Meant to be used with operator module.
    :param threshold:
    :param startingIdx:
    :return:
    """
    reversedPosition = idxFirst(data[::-1], comp, threshold, startingIdx=len(data) - startingIdx - 1)
    if reversedPosition == None:
        return None
    return len(data) - reversedPosition - 1


@njit()
def poly_numba(x, params):
    sum = 0
    for n, p in enumerate(params):
        sum += p * (x ** n)

    return sum


def expGausExp_poln(x, params):
    polyParams = [params[i] for i in range(5, len(params))]
    return params[0] * expGaussExp(x[0], params[1], params[2], params[3], params[4]) + poly_numba(x[0], polyParams)

def gaussExp_poln(x, params):
    polyParams = [params[i] for i in range(4, len(params))]
    return params[0] * gaussExp(x[0], params[1], params[2], params[3]) + poly_numba(x[0], np.array(polyParams))

def myround(x, base=5, func=round):
    """
    Inspired by
    https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
    :param x:
    :param base:
    :param func: function to use for rounding (i.e. round, math.floor, etc.)
    :return:
    """
    return base * func(float(x) / base)


def rebin(tops, bins, n):
    """
    Group every n bins together
    :param tops:
    :param bins:
    :param n:
    :return:
    """

    groupedTops = tops.reshape(-1, n).sum(axis=1)
    groupedBins = bins[::n]
    return groupedTops, groupedBins

class PolySpline(object):
    def __init__(self, polys, ranges):
        self.polys = np.array([np.poly1d(poly) for poly in polys])
        self.ranges = ranges
    def __call__(self, x):
        if np.ndim(x) == 0:
            return self._eval(x)
        return [self._eval(xi) for xi in x]
    def _eval(self, x):
        index = np.searchsorted(self.ranges, x, side='right') - 1 #-1 because index 0 is for searchsorted=1, for example
        return self.polys[index](x)


class InterpolateHist:
    """
    Used to create TF1.

    params[0] Scales histogram.
    
    """
    def __init__(self, hist):
        self.hist = hist

    def normalize(self, value=1):
        """
        Normalize number of entries in the xrange of the x axis to 1 (NOT
        AREA!!!). Changing the x axis range will change the normalization.

        Note - The integral is done between GetXaxis().GetFirst(), and
        GetXaxis().GetLast(). This range doesn't neccessarily coincide with the
        range when the histogram is drawn - if the SetRangeUser is chosen to
        coincide with a bin edge, the integral is up to the lower edge of the
        previous bin, whereas the drawn range is up to the bin that includes the
        max range.

        Note - Due to the previous note, use only GetFirst, GetLast to determine
        the relevant range.
        """
        try:
            self.hist.Scale(value/self.hist.Integral())
        except ZeroDivisionError:
            pass
        
    def __call__(self, x, params):        
        return params[0]*self.hist.Interpolate(x[0])

class Hist_chebyshev:
    """
    Hist+Chebyshev as background

    params[0] Scales histogram.
    params[1:] Are the Chebyshev coefficients. The length determines how many polynomials are used.

    Note - Domain of Chebyshev polynomials is scaled to hist effective range.
    This means they're orthogonal in that range.
    """

    def __init__(self, hist):
        self.interpolateHist = InterpolateHist(hist)
        self.interpolateHist.normalize(1)

        self.domain = [hist.GetXaxis().GetBinLowEdge(hist.GetXaxis().GetFirst()),
                       hist.GetXaxis().GetBinLowEdge(hist.GetXaxis().GetLast())]                       
        self.tf1Hist = ROOT.TF1("tf1Hist", self.interpolateHist, *self.domain, 1)
        self.tf1Hist.SetParameter(0,1)
        
        self.scaleA = sum(self.domain)/(self.domain[0] - self.domain[1])
        self.scaleB = 2/(self.domain[1] - self.domain[0])

        #Fix N_s to 1 and I'll normalize by myself.

    def __call__(self, x, params):
        return params[0]*self.tf1Hist.Eval(x[0]) + np.polynomial.chebyshev.chebval(self.scaleA + self.scaleB*x[0], list(params)[1:])

class Hist_omega_chebyshev:
    """
    Hist+Chebyshev as background

    params[0] Scales histogram.
    paramsp[1] scale of omega component
    params[2:] Are the Chebyshev coefficients. The length determines how many polynomials are used.

    Note - Domain of Chebyshev polynomials is scaled to hist effective range.
    This means they're orthogonal in that range.
    """

    def __init__(self, signalHist, omegaHist):
        self.interpolateSignalHist = InterpolateHist(signalHist)
        self.interpolateSignalHist.normalize(1)

        self.interpolateOmegaHist = InterpolateHist(omegaHist)
        #Assume omega hist already normalized

        self.domain = [signalHist.GetXaxis().GetBinLowEdge(signalHist.GetXaxis().GetFirst()),
                       signalHist.GetXaxis().GetBinLowEdge(signalHist.GetXaxis().GetLast())]                       
        self.tf1signalHist = ROOT.TF1("tf1signalHist", self.interpolateSignalHist, *self.domain, 1)
        self.tf1signalHist.SetParameter(0,1)

        self.tf1omegaHist = ROOT.TF1("tf1omegaHist", self.interpolateOmegaHist, *self.domain, 1)
        self.tf1omegaHist.SetParameter(0,1)
        
        self.scaleA = sum(self.domain)/(self.domain[0] - self.domain[1])
        self.scaleB = 2/(self.domain[1] - self.domain[0])

        #Fix N_s to 1 and I'll normalize by myself.

    def __call__(self, x, params):
        return params[0]*self.tf1signalHist.Eval(x[0]) + params[1]*self.tf1omegaHist.Eval(x[0]) + np.polynomial.chebyshev.chebval(self.scaleA + self.scaleB*x[0], list(params)[2:])

class Hist_omega_eta_phi_chebyshev:
    """
    Hist+Chebyshev as background

    params[0] Scales histogram.
    params[1] scale of omega component
    params[2] area under eta gaussian
    params[3] area under phi gaussian
    params[4:] Are the Chebyshev coefficients. The length determines how many polynomials are used.

    Note - Domain of Chebyshev polynomials is scaled to hist effective range.
    This means they're orthogonal in that range.
    """

    def __init__(self, signalHist, omegaHist, etaMean, etaStd, phiMean, phiStd):
        self.etaMean = etaMean
        self.etaStd = etaStd
        self.phiMean = phiMean
        self.phiStd = phiStd

        self.interpolateSignalHist = InterpolateHist(signalHist)
        self.interpolateSignalHist.normalize(1)

        self.interpolateOmegaHist = InterpolateHist(omegaHist)
        #Assume omega hist already normalized

        self.domain = [signalHist.GetXaxis().GetBinLowEdge(signalHist.GetXaxis().GetFirst()),
                       signalHist.GetXaxis().GetBinLowEdge(signalHist.GetXaxis().GetLast())]                       
        self.tf1signalHist = ROOT.TF1("tf1signalHist", self.interpolateSignalHist, *self.domain, 1)
        self.tf1signalHist.SetParameter(0,1)

        self.tf1omegaHist = ROOT.TF1("tf1omegaHist", self.interpolateOmegaHist, *self.domain, 1)
        self.tf1omegaHist.SetParameter(0,1)
        
        self.scaleA = sum(self.domain)/(self.domain[0] - self.domain[1])
        self.scaleB = 2/(self.domain[1] - self.domain[0])

        #Fix N_s to 1 and I'll normalize by myself.

    def __call__(self, x, params):
        return (params[0]*self.tf1signalHist.Eval(x[0]) +
               params[1]*self.tf1omegaHist.Eval(x[0]) + 
               params[2]*ROOT.TMath.Gaus(x[0], self.etaMean, self.etaStd, True) + 
               params[3]*ROOT.TMath.Gaus(x[0], self.phiMean, self.phiStd, True) + 
               np.polynomial.chebyshev.chebval(self.scaleA + self.scaleB*x[0], list(params)[4:]) )

class Hist_omega_pi0_eta_phi_chebyshev:
    """
    Hist+Chebyshev as background

    params[0] Scales histogram.
    params[1] scale of omega component
    params[2] area under pi0 gaussian
    params[3] area under eta gaussian
    params[4] area under phi gaussian
    params[5:] Are the Chebyshev coefficients. The length determines how many polynomials are used.

    Note - Domain of Chebyshev polynomials is scaled to hist effective range.
    This means they're orthogonal in that range.
    """

    def __init__(self, signalHist, omegaHist, pi0Mean, pi0Std, etaMean, etaStd, phiMean, phiStd):
        self.pi0Mean = pi0Mean
        self.pi0Std = pi0Std
        self.etaMean = etaMean
        self.etaStd = etaStd
        self.phiMean = phiMean
        self.phiStd = phiStd

        self.interpolateSignalHist = InterpolateHist(signalHist)
        self.interpolateSignalHist.normalize(1)

        self.interpolateOmegaHist = InterpolateHist(omegaHist)
        #Assume omega hist already normalized

        self.domain = [signalHist.GetXaxis().GetBinLowEdge(signalHist.GetXaxis().GetFirst()),
                       signalHist.GetXaxis().GetBinLowEdge(signalHist.GetXaxis().GetLast())]                       
        self.tf1signalHist = ROOT.TF1("tf1signalHist", self.interpolateSignalHist, *self.domain, 1)
        self.tf1signalHist.SetParameter(0,1)

        self.tf1omegaHist = ROOT.TF1("tf1omegaHist", self.interpolateOmegaHist, *self.domain, 1)
        self.tf1omegaHist.SetParameter(0,1)
        
        self.scaleA = sum(self.domain)/(self.domain[0] - self.domain[1])
        self.scaleB = 2/(self.domain[1] - self.domain[0])

        #Fix N_s to 1 and I'll normalize by myself.

    def __call__(self, x, params):
        return (params[0]*self.tf1signalHist.Eval(x[0]) +
               params[1]*self.tf1omegaHist.Eval(x[0]) + 
               params[2]*ROOT.TMath.Gaus(x[0], self.pi0Mean, self.pi0Std, True) +
               params[3]*ROOT.TMath.Gaus(x[0], self.etaMean, self.etaStd, True) + 
               params[4]*ROOT.TMath.Gaus(x[0], self.phiMean, self.phiStd, True) + 
               np.polynomial.chebyshev.chebval(self.scaleA + self.scaleB*x[0], list(params)[5:]) )


def chi2Prob(chi2, ndof):
    return stats.chi2.sf(chi2, ndof)
