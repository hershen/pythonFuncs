#!/usr/bin/python3
import mathFuncs
import math
import numpy as np
import pytest


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
    assert mathFuncs.novosibirsk(0.5, 0.5, 0.5, 1, 1) == 0.3482433775621212591

    x = [0.5, 1, 2, 3]

    assert np.allclose(mathFuncs.novosibirsk(x, 0.5, 0.5, 1, 1),
                       np.array([0.3482433775621212591, 0.2498417691242378613, 0, 0]))
    assert np.allclose(mathFuncs.novosibirsk(x, 0.3, 0.7, 0.1, 0.5),
                       np.array([0.09236319105347574887, 0, 0, 0]))
    assert np.allclose(mathFuncs.novosibirsk(x, 0.1, 0.7, 0.9, -1),
                       np.array([0.0666736776538124909, 0.06577645431068489257, 0.04009622918521545815,
                                 0.02290351914617567639]))


def test_listCenters():
    assert mathFuncs.listCenters([]).size == 0
    assert mathFuncs.listCenters([1]).size == 0
    assert mathFuncs.listCenters([1, 2]) == np.array([1.5])
    assert (mathFuncs.listCenters(np.array([1, 2, 3, 4, 5, 6])) == np.array([1.5, 2.5, 3.5, 4.5, 5.5])).all()
