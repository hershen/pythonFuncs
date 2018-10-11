#!/usr/bin/python3
import alpFuncs
import pandas as pd
import root_numpy
import numpy as np
import pytest

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


def test_loadDF(tmpdir):
    # create dataframe
    numpyArray = np.array([(1, 2.5, 3.4), (4, 5, 6.8)], dtype=[('a', np.float), ('b', np.float32), ('c', np.float64)])
    df = pd.DataFrame(numpyArray)

    # write dataframe
    tempFile = tmpdir.join("pandas.root")

    # Write array to file
    root_numpy.array2root(numpyArray, str(tempFile), treename='ntp1', mode='recreate')

    # Check arrays equal
    with pytest.warns(FutureWarning):
        pd.testing.assert_frame_equal(alpFuncs.loadDF(str(tempFile)), df)

def test_getSignalFilenames():
    Runs = ['1', '2', '3', '4', '5', '6', '1-6', '2S', '3S', '7' , 'all']
    for alpMass in range(1, 2, 10):
        for triggered in [True, False]:
            for Run in Runs:
                assert alpFuncs.getSignalFilenames(alpMass, Run, triggered)