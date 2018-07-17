#!/usr/bin/python3
import mathFuncs
import math
import numpy as np
import pytest

def test_vectorDot():

    #All zeros
    assert mathFuncs.vectorDot([0,0,0], [0,0,0]) == 0

    #nans
    assert np.isnan(mathFuncs.vectorDot([np.nan,0,0], [0,0,0]))
    assert np.isnan(mathFuncs.vectorDot([0, np.nan ,0], [0,0,0]))
    assert np.isnan(mathFuncs.vectorDot([0,0, np.nan], [0,0,0]))
    assert np.isnan(mathFuncs.vectorDot([0,0,0], [np.nan,0,0]))
    assert np.isnan(mathFuncs.vectorDot([0,0,0], [0,np.nan, 0]))
    assert np.isnan(mathFuncs.vectorDot([0,0,0], [0, 0, np.nan]))


    assert mathFuncs.vectorDot([1,2,3], [4,5,6]) == 32

    v1 = np.array([[1,2,3],[1,2,3]])
    v2 = np.array([[1,2,3],[4,5,6]])

    assert (mathFuncs.vectorDot(v1,v2) == np.array([14, 32])).all()

def test_cosTheta():
    #All non zeroes
    assert mathFuncs.cosTheta(1,2,3) == 3 / math.sqrt(1+4+9)

    #x zero
    assert mathFuncs.cosTheta(0,2,3) == 3 / math.sqrt(0+4+9)
    #y zero
    assert mathFuncs.cosTheta(1,0,3) == 3 / math.sqrt(1+0+9)
    #z zero
    assert mathFuncs.cosTheta(1,2,0) == 0 / math.sqrt(1+4+9)

    #x nan
    assert np.isnan(mathFuncs.cosTheta(np.nan,2,3))
    #y zero
    assert np.isnan(mathFuncs.cosTheta(1, np.nan,3))
    #z zero
    assert np.isnan(mathFuncs.cosTheta(1, 2, np.nan))

    #all zero - divistion by 0 warning
    with pytest.warns(RuntimeWarning, match='invalid value encountered in double_scalars'):
        assert np.isnan(mathFuncs.cosTheta(0,0,0))


def test_angleBetween():
    v1 = np.array([1,0, 0])
    v2 = np.array([1,0, 0])

    assert mathFuncs.angleBetween(v1, v2) == 0
    assert mathFuncs.angleBetween(v1, -v2) == np.pi
    assert mathFuncs.angleBetween(-v1, -v2) == 0

    assert mathFuncs.angleBetween([1,2,3], [4,5,6]) == math.acos((4+10+18) / math.sqrt(1 + 4 +9) / math.sqrt(16 + 25 + 36))

    with pytest.warns(RuntimeWarning, match='invalid value encountered in divide'):
        assert np.isnan(mathFuncs.angleBetween([0,0,0], [0,0,0]))

    assert np.isnan(mathFuncs.angleBetween([np.nan,0,0], [0,0,0]))
    assert np.isnan(mathFuncs.angleBetween([0, np.nan ,0], [0,0,0]))
    assert np.isnan(mathFuncs.angleBetween([0,0, np.nan], [0,0,0]))
    assert np.isnan(mathFuncs.angleBetween([0,0,0], [np.nan,0,0]))
    assert np.isnan(mathFuncs.angleBetween([0,0,0], [0,np.nan, 0]))
    assert np.isnan(mathFuncs.angleBetween([0,0,0], [0, 0, np.nan]))

    v1 = np.array([[1,2,3],[1,2,3]])
    v2 = np.array([[1,2,3],[4,5,6]])

    assert (mathFuncs.angleBetween(v1, v2) == np.array([0,math.acos((4+10+18) / math.sqrt(1 + 4 +9) / math.sqrt(16 + 25 + 36))])).all()

def test_lorentzDot():
    v1 = [1, 2, 3, 4]
    v2 = [5, 6, 7, 8]

    assert mathFuncs.lorentzDot(v1, v2) == -6
    assert mathFuncs.lorentzDot(-np.array(v1), v2) == 6

    assert mathFuncs.lorentzDot([1,2,3,4], [0,0,0,0]) == 0

    assert np.isnan(mathFuncs.lorentzDot([np.nan,1, 2, 3], [5, 6, 7, 8]))

    v1 = np.array([[1,2,3, 4], [1, 2, 3, 5]])
    v2 = np.array([[1,2,3, 4], [5, 6, 7, 8]])

    assert (mathFuncs.lorentzDot(v1, v2) == np.array([2, 2])).all()

def test_mass():

    assert mathFuncs.mass([1,2,3,10]) == np.sqrt(86)
    assert mathFuncs.mass([0,0,0,0]) == 0

    v = np.array([[0,0,0,0], [1,2,3,10], [1,2,3,np.sqrt(14)]])
    assert (mathFuncs.mass(v) == np.array([0, np.sqrt(86), 0])).all()

def test_effError():
    assert mathFuncs.effError(0.5,1) == 0.5
    assert np.allclose(mathFuncs.effError(np.array([8,5]), np.array([10,10])) , np.array([math.sqrt(0.8*0.2/10), math.sqrt(0.5*0.5/10)]) )

    #These 2 are not strictly correct. They're correct for the current formula
    assert mathFuncs.effError(0,1) == 0
    assert mathFuncs.effError(1,1) == 0
