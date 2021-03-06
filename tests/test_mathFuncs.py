#!/usr/bin/python3
import mathFuncs
import math
import numpy as np
import pytest
import ROOT
import operator


def test_vectorDot():
    # All zeros
    assert mathFuncs.vectorDot([0, 0, 0], [0, 0, 0]) == 0

    # nans
    assert np.isnan(mathFuncs.vectorDot([np.nan, 0, 0], [0, 0, 0]))
    assert np.isnan(mathFuncs.vectorDot([0, np.nan, 0], [0, 0, 0]))
    assert np.isnan(mathFuncs.vectorDot([0, 0, np.nan], [0, 0, 0]))
    assert np.isnan(mathFuncs.vectorDot([0, 0, 0], [np.nan, 0, 0]))
    assert np.isnan(mathFuncs.vectorDot([0, 0, 0], [0, np.nan, 0]))
    assert np.isnan(mathFuncs.vectorDot([0, 0, 0], [0, 0, np.nan]))

    assert mathFuncs.vectorDot([1, 2, 3], [4, 5, 6]) == 32

    v1 = np.array([[1, 2, 3], [1, 2, 3]])
    v2 = np.array([[1, 2, 3], [4, 5, 6]])

    assert (mathFuncs.vectorDot(v1, v2) == np.array([14, 32])).all()


def test_cosTheta():
    # All non zeroes
    assert mathFuncs.cosTheta(1, 2, 3) == 3 / math.sqrt(1 + 4 + 9)

    # x zero
    assert mathFuncs.cosTheta(0, 2, 3) == 3 / math.sqrt(0 + 4 + 9)
    # y zero
    assert mathFuncs.cosTheta(1, 0, 3) == 3 / math.sqrt(1 + 0 + 9)
    # z zero
    assert mathFuncs.cosTheta(1, 2, 0) == 0 / math.sqrt(1 + 4 + 9)

    # x nan
    assert np.isnan(mathFuncs.cosTheta(np.nan, 2, 3))
    # y zero
    assert np.isnan(mathFuncs.cosTheta(1, np.nan, 3))
    # z zero
    assert np.isnan(mathFuncs.cosTheta(1, 2, np.nan))

    # all zero - divistion by 0 warning
    with pytest.warns(RuntimeWarning, match='invalid value encountered in double_scalars'):
        assert np.isnan(mathFuncs.cosTheta(0, 0, 0))


def test_angleBetween():
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 0, 0])

    assert mathFuncs.angleBetween(v1, v2) == 0
    assert mathFuncs.angleBetween(v1, -v2) == np.pi
    assert mathFuncs.angleBetween(-v1, -v2) == 0

    assert mathFuncs.angleBetween([1, 2, 3], [4, 5, 6]) == math.acos(
        (4 + 10 + 18) / math.sqrt(1 + 4 + 9) / math.sqrt(16 + 25 + 36))

    with pytest.warns(RuntimeWarning, match='invalid value encountered in true_divide'):
        assert np.isnan(mathFuncs.angleBetween([0, 0, 0], [0, 0, 0]))

    assert np.isnan(mathFuncs.angleBetween([np.nan, 0, 0], [0, 0, 0]))
    assert np.isnan(mathFuncs.angleBetween([0, np.nan, 0], [0, 0, 0]))
    assert np.isnan(mathFuncs.angleBetween([0, 0, np.nan], [0, 0, 0]))
    assert np.isnan(mathFuncs.angleBetween([0, 0, 0], [np.nan, 0, 0]))
    assert np.isnan(mathFuncs.angleBetween([0, 0, 0], [0, np.nan, 0]))
    assert np.isnan(mathFuncs.angleBetween([0, 0, 0], [0, 0, np.nan]))

    v1 = np.array([[1, 2, 3], [1, 2, 3]])
    v2 = np.array([[1, 2, 3], [4, 5, 6]])

    assert (mathFuncs.angleBetween(v1, v2) == np.array(
        [0, math.acos((4 + 10 + 18) / math.sqrt(1 + 4 + 9) / math.sqrt(16 + 25 + 36))])).all()


def test_lorentzDot():
    v1 = [1, 2, 3, 4]
    v2 = [5, 6, 7, 8]

    assert mathFuncs.lorentzDot(v1, v2) == -6
    assert mathFuncs.lorentzDot(-np.array(v1), v2) == 6

    assert mathFuncs.lorentzDot([1, 2, 3, 4], [0, 0, 0, 0]) == 0

    assert np.isnan(mathFuncs.lorentzDot([np.nan, 1, 2, 3], [5, 6, 7, 8]))

    v1 = np.array([[1, 2, 3, 4], [1, 2, 3, 5]])
    v2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    assert (mathFuncs.lorentzDot(v1, v2) == np.array([2, 2])).all()


def test_mass():
    assert mathFuncs.mass([1, 2, 3, 10]) == np.sqrt(86)
    assert mathFuncs.mass([0, 0, 0, 0]) == 0

    v = np.array([[0, 0, 0, 0], [1, 2, 3, 10], [1, 2, 3, np.sqrt(14)]])
    assert (mathFuncs.mass(v) == np.array([0, np.sqrt(86), 0])).all()


def test_effError():
    assert mathFuncs.effError(0.5, 1) == 0.5
    assert np.allclose(mathFuncs.effError(np.array([8, 5]), np.array([10, 10])),
                       np.array([math.sqrt(0.8 * 0.2 / 10), math.sqrt(0.5 * 0.5 / 10)]))

    # These 2 are not strictly correct. They're correct for the current formula
    assert mathFuncs.effError(0, 1) == 0
    assert mathFuncs.effError(1, 1) == 0


def test_radial_xyz():
    # All non zeroes
    assert mathFuncs.radial_xyz(1, 2, 3) == math.sqrt(14)

    # x zero
    assert mathFuncs.radial_xyz(0, 2, 3) == math.sqrt(13)

    assert mathFuncs.radial_xyz(0, 0, 0) == 0

    # x nan
    assert np.isnan(mathFuncs.radial_xyz(np.nan, 2, 3))


def test_phi_xyz():
    assert mathFuncs.phi_xyz(1, 2, 3) == 1.107148717794090408972

    assert mathFuncs.phi_xyz(0, 2, 3) == 1.570796326794896557999

    assert mathFuncs.phi_xyz(1, 0, 3) == 0

    assert (mathFuncs.phi_xyz([1, 0, 1], [2, 2, 0], [3, 3, 3]) == [1.107148717794090408972, 1.570796326794896557999,
                                                                   0]).all()

    # x nan
    assert np.isnan(mathFuncs.phi_xyz(np.nan, 2, 3))


def test_novosibirsk():
    # values taken from using my c++ implementation
    assert 0.5 * mathFuncs.novosibirsk(0.5, 0.5, 1, 1) == 0.3482433775621212591

    x = [0.5, 1, 2, 3]

    assert np.allclose(0.5 * mathFuncs.novosibirsk(x, 0.5, 1, 1),
                       np.array([0.3482433775621212591, 0.2498417691242378613, 0, 0]))
    assert np.allclose(0.3 * mathFuncs.novosibirsk(x, 0.7, 0.1, 0.5),
                       np.array([0.09236319105347574887, 0, 0, 0]))
    assert np.allclose(0.1 * mathFuncs.novosibirsk(x, 0.7, 0.9, -1),
                       np.array([0.0666736776538124909, 0.06577645431068489257, 0.04009622918521545815,
                                 0.02290351914617567639]))


def test_listCenters():
    assert mathFuncs.listCenters([]).size == 0
    assert mathFuncs.listCenters([1]).size == 0
    assert mathFuncs.listCenters([1, 2]) == np.array([1.5])
    assert np.allclose(mathFuncs.listCenters(np.array([1, 2, 3, 4, 5, 6])), np.array([1.5, 2.5, 3.5, 4.5, 5.5]))


def test_novosibirskForTf1():
    x = [0.5]
    params = [10, 2, 0.1, 0.5]
    np.testing.assert_array_almost_equal(params[0] * mathFuncs.novosibirsk(x, params[1], params[2], params[3]),
                                         mathFuncs.novosibirskForTf1(x, params))

    params = [10, 2, 1, 0.5]
    np.testing.assert_array_almost_equal(params[0] * mathFuncs.novosibirsk(x, params[1], params[2], params[3]),
                                         mathFuncs.novosibirskForTf1(x, params))
    params = [10, 2, 4, -0.5]
    np.testing.assert_array_almost_equal(params[0] * mathFuncs.novosibirsk(x, params[1], params[2], params[3]),
                                         mathFuncs.novosibirskForTf1(x, params))
    params = [10, 0.1, 0.5, -0.1]
    np.testing.assert_array_almost_equal(params[0] * mathFuncs.novosibirsk(x, params[1], params[2], params[3]),
                                         mathFuncs.novosibirskForTf1(x, params))


def test_gaussExp():
    assert np.isclose(10 * mathFuncs.gaussExp(1, 0, 1, 0), 6.065306597126334236)
    assert np.isclose(10 * mathFuncs.gaussExp(1, 0, 1, 0.5), 6.065306597126334236)
    assert np.isclose(10 * mathFuncs.gaussExp(1, 0, 1, 1.01), 6.065306597126334236)
    assert np.isclose(10 * mathFuncs.gaussExp(1, 0, 1, -0), 6.06530659712633423604)
    assert np.isclose(10 * mathFuncs.gaussExp(1, 0, 1, -0.1), 9.09372934468231420493)
    assert np.isclose(10 * mathFuncs.gaussExp(1, 0, 1, -0.5), 6.8728927879097219855)
    assert np.isclose(10 * mathFuncs.gaussExp(1, 0, 1, -0.7), 6.344479679482281821)
    assert np.isclose(10 * mathFuncs.gaussExp(1, 0, 1, -2), 6.06530659712633423604)


def test_gaussExpForTf1():
    x = [0.5]
    params = [10, 2, 0.1, 0.5]
    np.testing.assert_array_almost_equal(params[0] * mathFuncs.gaussExp(x, params[1], params[2], params[3]),
                                         mathFuncs.gaussExpForTf1(x, params))
    params = [10, 2, 1, 0.5]
    np.testing.assert_array_almost_equal(params[0] * mathFuncs.gaussExp(x, params[1], params[2], params[3]),
                                         mathFuncs.gaussExpForTf1(x, params))
    params = [10, 2, 4, -0.5]
    np.testing.assert_array_almost_equal(params[0] * mathFuncs.gaussExp(x, params[1], params[2], params[3]),
                                         mathFuncs.gaussExpForTf1(x, params))
    params = [10, 0.1, 0.5, -0.1]
    np.testing.assert_array_almost_equal(params[0] * mathFuncs.gaussExp(x, params[1], params[2], params[3]),
                                         mathFuncs.gaussExpForTf1(x, params))


def test_idicesPercentageOfMax():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10., 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert mathFuncs.indicesPercentageOfMax(x, 51) == (5, 13)

    assert mathFuncs.indicesPercentageOfMax(x, 41) == (4, 14)
    assert mathFuncs.indicesPercentageOfMax(x, 40) == (3, 15)
    assert mathFuncs.indicesPercentageOfMax(x, 39) == (3, 15)

    assert mathFuncs.indicesPercentageOfMax(x, 0) == (9, 9)
    assert mathFuncs.indicesPercentageOfMax(x, 100) == (9, 9)

    # maximum is zero - expect fail
    with pytest.raises(ValueError):
        mathFuncs.indicesPercentageOfMax([0, 0, 0], 40)

def test_valuesPercentageOfMax():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10., 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert mathFuncs.valuesPercentageOfMax(x, 51) == (6, 6)

    assert mathFuncs.valuesPercentageOfMax(x, 41) == (5, 5)
    assert mathFuncs.valuesPercentageOfMax(x, 40) == (4, 4)
    assert mathFuncs.valuesPercentageOfMax(x, 39) == (4, 4)

    assert mathFuncs.valuesPercentageOfMax(x, 0) == (10, 10)
    assert mathFuncs.valuesPercentageOfMax(x, 100) == (10, 10)

    # maximum is zero - expect fail
    with pytest.raises(ValueError):
        mathFuncs.valuesPercentageOfMax([0, 0, 0], 40)

def test_calcPulls():
    with pytest.raises(AssertionError):
        mathFuncs.calcPulls([1], [2, 2], [3])

    with pytest.raises(AssertionError):
        mathFuncs.calcPulls([1, 1], [2, 2], [3])

    assert mathFuncs.calcPulls(1, 3, 2) == -1. / 3

    with pytest.warns(RuntimeWarning):
        assert np.isneginf(mathFuncs.calcPulls(1, 0, 2))
        assert np.isposinf(mathFuncs.calcPulls(2, 0, 1))
        assert np.isnan(mathFuncs.calcPulls(1, 0, 1))

    np.testing.assert_allclose(mathFuncs.calcPulls([1, 2, 3], [0.1, 1, 10], [1, 4, 7]), np.array([0, -2, -0.4]))


def test_cosHelicity():
    """
    cosHelicity values calculated with c function from
    https://www.slac.stanford.edu/BFROOT/www/doc/workbook/analysis/analysis.html#helicity
    These give the negative value of my function.
    The numbers for the momentum vector components were drawn from a Gaus(0,5), and the mass from Uniform(0,5)
    """
    assert np.isclose(mathFuncs.cosHelicity(np.array([1.483607321, -1.057359938, 4.64859099, 5.125154677]),
                                            np.array([-2.872029834, -2.696355793, -10.83872179, 11.63641534]),
                                            np.array([-9.360268377, 5.374778891, -7.668973941, 13.77497287])),
                      -0.9621238708)

    assert np.isclose(mathFuncs.cosHelicity(np.array([-5.053961598, 0.2945919288, 7.497966283, 10.0386344]),
                                            np.array([3.004968973, 2.161233848, 4.120989493, 6.214798517]),
                                            np.array([13.40814548, -2.011223834, -2.179347456, 13.93400376])),
                      -0.3659081757)
    assert np.isclose(mathFuncs.cosHelicity(np.array([2.816980283, 6.081124253, -9.421044698, 11.56563981]),
                                            np.array([0.642004394, -1.252326062, 3.879911562, 4.638644664]),
                                            np.array([-2.06614717, -5.495463191, -3.781723353, 7.882918615])),
                      -0.9164004326)
    assert np.isclose(mathFuncs.cosHelicity(np.array([0.3033471748, -2.666406462, 0.1447857893, 4.698073577]),
                                            np.array([-3.794295829, -4.722211459, -0.658236281, 6.480472775]),
                                            np.array([-1.476877853, 3.277558051, -12.70315193, 13.21825152])),
                      -0.878118217)


# def test_expGaussExp():
#     assert mathFuncs.gaussExp(1, 10, 0, 1, 0, 0) == 6.065306597126334236
#     assert mathFuncs.gaussExp(1, 10, 0, 1, 0.5) == 6.065306597126334236
#     assert mathFuncs.gaussExp(1, 10, 0, 1, 1.01) == 6.065306597126334236
#     assert mathFuncs.gaussExp(1, 10, 0, 1, -0.5) == 6.065306597126334236
#     assert mathFuncs.gaussExp(1, 10, 0, 1, -2) == 10
#     assert mathFuncs.gaussExp(1, 10, 0, 1, -5) == 18080.42414456063206903801
#     assert mathFuncs.gaussExp(1, 10, 0, 1, -10) == 2353852668370199854.07899910749034804509
#
# def test_expGaussExpForTf1():
#     x = [0.5]
#     params = [10, 2, 0.1, 0.5]
#     np.testing.assert_array_almost_equal(mathFuncs.expGaussExp(x, params[0], params[1], params[2], params[3]),
#                                          mathFuncs.gaussExpForTf1(x, params))
#     params = [10, 2, 1, 0.5]
#     np.testing.assert_array_almost_equal(mathFuncs.expGaussExp(x, params[0], params[1], params[2], params[3]),
#                                          mathFuncs.gaussExpForTf1(x, params))
#     params = [10, 2, 4, -0.5]
#     np.testing.assert_array_almost_equal(mathFuncs.expGaussExp(x, params[0], params[1], params[2], params[3]),
#                                          mathFuncs.gaussExpForTf1(x, params))
#     params = [10, 0.1, 0.5, -0.1]
#     np.testing.assert_array_almost_equal(mathFuncs.expGaussExp(x, params[0], params[1], params[2], params[3]),
#                                          mathFuncs.gaussExpForTf1(x, params))

def test_expGaussExp_FWHM_low_high():
    np.random.seed(10)

    for i in range(100):
        peak = np.random.uniform(-5, 5)
        sigma = np.random.uniform(1e-6, 3)
        tailLow = np.random.uniform(0.1, 5)
        tailHigh = np.random.uniform(0.1, 5)

        # test FWHM low point
        xLow = mathFuncs.expGaussExp_FWHM_xLow(peak, sigma, tailLow)
        fwhm_y_low = mathFuncs.expGaussExp(xLow, peak, sigma, tailLow, tailHigh)
        np.testing.assert_almost_equal(0.5 * mathFuncs.expGaussExp(peak, peak, sigma, tailLow, tailHigh), fwhm_y_low)

        # test FWHM high point
        xHigh = mathFuncs.expGaussExp_FWHM_xHigh(peak, sigma, tailHigh)
        fwhm_y_high = mathFuncs.expGaussExp(xHigh, peak, sigma, tailLow, tailHigh)
        np.testing.assert_almost_equal(0.5 * mathFuncs.expGaussExp(peak, peak, sigma, tailLow, tailHigh), fwhm_y_high)

        # Test FWHM
        fwhm = xHigh - xLow
        np.testing.assert_almost_equal(fwhm, mathFuncs.expGaussExp_FWHM(peak, sigma, tailLow, tailHigh))

        # Test FWHM/2.355
        np.testing.assert_almost_equal(fwhm / 2 / mathFuncs._sln4,
                                       mathFuncs.expGaussExp_gausEqeuivalentSigma(peak, sigma, tailLow, tailHigh))


def test_expGaussExp_integral():
    np.random.seed(10)
    funcRange = [-50, 50]

    for i in range(20):
        while True:
            norm = np.random.uniform(0.01, 100)
            peak = np.random.uniform(-5, 5)
            sigma = np.random.uniform(1e-6, 3)
            tailLow = np.random.uniform(0.1, 5)
            tailHigh = np.random.uniform(0.1, 5)
            params = [norm, peak, sigma, tailLow, tailHigh]

            intRange = [-5 * sigma, 5 * sigma]

            # make sure integration range doesn't make exp blow up
            if abs((intRange[0] - peak) / sigma) < 10 and abs((intRange[1] - peak) / sigma) < 10:
                break
        # # Gaussian only
        # params[0] = 1
        # params[1] = 0
        # params[2] = 1.5
        # params[3] = 2
        # params[4] = 10

        # Integrate using TF1.Integrate
        tf1 = ROOT.TF1("tf1_{}".format(i), mathFuncs.expGaussExpForTf1, funcRange[0], funcRange[1], 5)
        tf1.SetParameters(*params)
        tf1.SetNpx(10000)
        integral = tf1.Integral(intRange[0], intRange[1])

        np.testing.assert_almost_equal(mathFuncs.expGaussExp_integral(*params, intRange[0], intRange[1]), integral)


def test_crystalBall():
    np.random.seed(10)

    for i in range(20):
        alpha, n = np.random.uniform(-3, 3), np.random.uniform(0, 10)

        np.testing.assert_almost_equal(mathFuncs.crystalBall(1, 0, 1, alpha, n),
                                       ROOT.Math.crystalball_function(1, alpha, n, 1, 0))


def test_crystalBallForTf1():
    np.random.seed(100)
    x = [0.5]

    for i in range(20):
        params = [10, np.random.normal(-3, 3), np.random.uniform(0, 3), np.random.uniform(-3, 3),
                  np.random.uniform(0, 10)]
        np.testing.assert_array_almost_equal(
            params[0] * mathFuncs.crystalBall(x, params[1], params[2], params[3], params[4]),
            mathFuncs.crystalBallForTf1(x, params))


def test_doubleSidedCrystalBall_FWHM_low_high():
    np.random.seed(10)

    for i in range(100):
        peak = np.random.uniform(-5, 5)
        sigma = np.random.uniform(1e-6, 3)
        tailLow = np.random.uniform(0.1, 5)
        tailHigh = np.random.uniform(0.1, 5)
        nLow = np.random.uniform(0.1, 20)
        nHigh = np.random.uniform(0.1, 20)

        # test FWHM low point
        xLow = mathFuncs.doubleSidedCrystalBall_FWHM_xLow(peak, sigma, tailLow, nLow)
        fwhm_y_low = mathFuncs.doubleSidedCrystalBall(xLow, peak, sigma, tailLow, tailHigh, nLow, nHigh)
        np.testing.assert_almost_equal(
            0.5 * mathFuncs.doubleSidedCrystalBall(peak, peak, sigma, tailLow, tailHigh, nLow, nHigh), fwhm_y_low)

        # test FWHM high point
        xHigh = mathFuncs.doubleSidedCrystalBall_FWHM_xHigh(peak, sigma, tailHigh, nHigh)
        fwhm_y_high = mathFuncs.doubleSidedCrystalBall(xHigh, peak, sigma, tailLow, tailHigh, nLow, nHigh)
        np.testing.assert_almost_equal(
            0.5 * mathFuncs.doubleSidedCrystalBall(peak, peak, sigma, tailLow, tailHigh, nLow, nHigh), fwhm_y_high)

        # Test FWHM
        fwhm = xHigh - xLow
        np.testing.assert_almost_equal(fwhm,
                                       mathFuncs.doubleSidedCrystalBall_FWHM(peak, sigma, tailLow, tailHigh, nLow,
                                                                             nHigh))

        # Test FWHM/2.355
        np.testing.assert_almost_equal(fwhm / 2 / mathFuncs._sln4,
                                       mathFuncs.doubleSidedCrystalBall_gausEqeuivalentSigma(peak, sigma, tailLow,
                                                                                             tailHigh, nLow, nHigh))


_dataForIdxSearchingFunctions = [1, 2, 3, 4, 5, 6, 5, 4, 5, 4, 5, 3, 2, 1]


def test_idxFirst():
    # LT
    assert mathFuncs.idxFirst([], operator.lt, 3) == None
    assert mathFuncs.idxFirst([10], operator.lt, 3) == None
    assert mathFuncs.idxFirst(_dataForIdxSearchingFunctions, operator.lt, 3) == 0
    assert mathFuncs.idxFirst(_dataForIdxSearchingFunctions, operator.lt, 3, startingIdx=5) == 12

    # GT
    assert mathFuncs.idxFirst([], operator.gt, 3) == None
    assert mathFuncs.idxFirst([1], operator.gt, 3) == None
    assert mathFuncs.idxFirst(_dataForIdxSearchingFunctions, operator.gt, 3) == 3
    assert mathFuncs.idxFirst(_dataForIdxSearchingFunctions, operator.gt, 3, startingIdx=5) == 5


def test_idxFirstToLeft():
    # LT
    assert mathFuncs.idxFirstToLeft([], operator.lt, 3) == None
    assert mathFuncs.idxFirstToLeft([10], operator.lt, 3) == None
    assert mathFuncs.idxFirstToLeft(_dataForIdxSearchingFunctions, operator.lt, 3, startingIdx=12) == 12
    assert mathFuncs.idxFirstToLeft(_dataForIdxSearchingFunctions, operator.lt, 3, startingIdx=5) == 1

    # GT
    assert mathFuncs.idxFirstToLeft([], operator.gt, 3) == None
    assert mathFuncs.idxFirstToLeft([1], operator.gt, 3) == None
    assert mathFuncs.idxFirstToLeft(_dataForIdxSearchingFunctions, operator.gt, 3) == None
    assert mathFuncs.idxFirstToLeft(_dataForIdxSearchingFunctions, operator.gt, 6, startingIdx=5) == None
    assert mathFuncs.idxFirstToLeft(_dataForIdxSearchingFunctions, operator.gt, 4, startingIdx=12) == 10

def test_InterpolateHist():
    hist = ROOT.TH1D("hist", "", 100, -5, 5)
    hist.FillRandom('gaus', 1000)
    hist.GetXaxis().SetRangeUser(-1, 1)
    tf1Object = mathFuncs.InterpolateHist(hist)

    assert np.isclose(hist.Integral(), 1)

    del hist
