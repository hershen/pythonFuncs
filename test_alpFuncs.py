#!/usr/bin/python3
import alpFuncs
# import pytest

filename = 'flat_mass7.00e+00_coup1.0000e-03_ISR_numEvents50000_1-Run3.01.root'

def test_getMass():
    assert alpFuncs.getMass(filename) == '7.00e+00'

def test_getRun():
    assert alpFuncs.getRun(filename) == '3'
