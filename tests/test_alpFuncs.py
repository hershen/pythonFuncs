#!/usr/bin/python3
import alpFuncs

# import pytest

filename_Run3 = 'flat_mass7.00e+00_coup1.0000e-03_ISR_numEvents50000_1-Run3.01.root'
filename_Y2S = 'flat_mass8.50e+00_coup1.0000e-03_Y2S_ISR_numEvents2500_3-Ups2S.01.root'
filename_Y3S = 'flat_mass4.00e+00_coup1.0000e-03_Y3S_ISR_numEvents2500_2-Ups3S.01.root'


def test_getMass():
    assert alpFuncs.getMass(filename_Run3) == '7.00e+00'
    assert alpFuncs.getMass(filename_Y2S) == '8.50e+00'
    assert alpFuncs.getMass(filename_Y3S) == '4.00e+00'


def test_getRun():
    assert alpFuncs.getRun(filename_Run3) == '3'
    assert alpFuncs.getRun(filename_Y2S) == '2S'
    assert alpFuncs.getRun(filename_Y3S) == '3S'
