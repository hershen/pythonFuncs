#!/usr/bin/python3
import itertools
import os

import pandas as pd
import ROOT
import root_numpy
import numpy as np
import pytest

import alpFuncs

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
    Runs = ['1', '2', '3', '4', '5', '6', '1-6', '2S', '3S', '7', '1-7']
    for Run in Runs:
        for alpMass in alpFuncs.getAlpMasses(Run):
            filenames = alpFuncs.getSignalFilenames(alpMass, Run)
            assert len(filenames) > 0
            if alpMass <= 1:
                assert '_minE12cmCut' in filenames[0]
            else:
                assert '_minE12cmCut' not in filenames[0]


def test_getDatasets():
    assert alpFuncs.getDatasets('1-6') == ['Y4S_OffPeak', 'Y4S_OnPeak']
    assert alpFuncs.getDatasets('7') == ['Y2S_OffPeak', 'Y2S_OnPeak', 'Y3S_OffPeak', 'Y3S_OnPeak']
    assert alpFuncs.getDatasets('1-7') == ['Y2S_OffPeak', 'Y2S_OnPeak', 'Y3S_OffPeak', 'Y3S_OnPeak', 'Y4S_OffPeak',
                                           'Y4S_OnPeak']


_signalFileFolder = '/home/hershen/PhD/ALPs/analysis/ntuples/MC/sig'
_signalLooseMinE12cmFolder = 'looseMinE12cmCut'


@pytest.mark.parametrize("Run, alpMass",
                         [vals for Run in ['1-6', '7', '2S', '3S'] for vals in
                          itertools.product([Run], alpFuncs.getAlpMasses(Run))])
def test_getNumberOfGeneratedSignal(Run, alpMass):
    signalFolder = os.path.join(_signalFileFolder, _signalLooseMinE12cmFolder if alpMass <= 1.0 else '')
    if Run == '1-6':
        RunIdentifier = 'Run'
    elif Run == '7':
        RunIdentifier = 'Ups'
    elif Run == '2S':
        RunIdentifier = 'Ups2'
    elif Run == '3S':
        RunIdentifier = 'Ups3'
    filenameTemplate = os.path.join(signalFolder, f'mass{alpMass:.2e}*{RunIdentifier}*.root')
    filenameTemplate = filenameTemplate.replace('+', '\\+')

    chain = ROOT.TChain('ntp1')
    chain.Add(filenameTemplate)
    assert alpFuncs.getNumberOfGeneratedSignal(Run, alpMass) == chain.GetEntries()
