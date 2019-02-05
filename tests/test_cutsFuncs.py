#!/usr/bin/python3
import itertools
from unittest import mock

import numpy as np

import cutsFuncs

signalHdfGroups = [f'/Run{Run}_alpMass{alpMass}' for Run, alpMass in
                   itertools.product(list(range(1, 7)) + ['2S', '3S'], np.arange(0.5, 10.5, 0.5))]
data5perHdfGroups = [f'/Run{Run}_{OnOff}Peak_alpMass{alpMass}' for Run, OnOff, alpMass in
                     itertools.product(list(range(1, 7)) + ['2S', '3S'], ['On', 'Off'], np.arange(0.5, 10.5, 0.5))]
SP1074HdfGroups = [f'/Run{Run}_{OnOff}Peak_alpMass{alpMass}' for Run, OnOff, alpMass in
                   itertools.product(['2S', '3S', '4S'], ['On', 'Off'], np.arange(0.5, 10.5, 0.5))]

filename2groups = {'massInCountingwindow_data5perc.h5': data5perHdfGroups,
                   'massInCountingwindow_signal.h5': signalHdfGroups,
                   'massInCountingwindow_SP1074.h5': SP1074HdfGroups}


def getGroup(arg):
    return filename2groups[arg]


expectedGroups = {'massInCountingwindow_signal.h5':
    {'1-6': {
        0.5: ['/Run1_alpMass0.5', '/Run2_alpMass0.5', '/Run3_alpMass0.5', '/Run4_alpMass0.5', '/Run5_alpMass0.5',
              '/Run6_alpMass0.5'],
        5.0: ['/Run1_alpMass5.0', '/Run2_alpMass5.0', '/Run3_alpMass5.0', '/Run4_alpMass5.0', '/Run5_alpMass5.0',
              '/Run6_alpMass5.0'],
        9.5: ['/Run1_alpMass9.5', '/Run2_alpMass9.5', '/Run3_alpMass9.5', '/Run4_alpMass9.5', '/Run5_alpMass9.5',
              '/Run6_alpMass9.5']
    }, '7': {
        0.5: ['/Run2S_alpMass0.5', '/Run3S_alpMass0.5'],
        5.0: ['/Run2S_alpMass5.0', '/Run3S_alpMass5.0'],
        9.5: ['/Run2S_alpMass9.5', '/Run3S_alpMass9.5'],
    }},
    'massInCountingwindow_SP1074.h5':
        {'1-6': {
            0.5: ['/Run4S_OnPeak_alpMass0.5', '/Run4S_OffPeak_alpMass0.5'],
            5.0: ['/Run4S_OnPeak_alpMass5.0', '/Run4S_OffPeak_alpMass5.0'],
            9.5: ['/Run4S_OnPeak_alpMass9.5', '/Run4S_OffPeak_alpMass9.5']
        }, '7': {
            0.5: ['/Run2S_OnPeak_alpMass0.5', '/Run2S_OffPeak_alpMass0.5', '/Run3S_OnPeak_alpMass0.5',
                  '/Run3S_OffPeak_alpMass0.5'],
            5.0: ['/Run2S_OnPeak_alpMass5.0', '/Run2S_OffPeak_alpMass5.0', '/Run3S_OnPeak_alpMass5.0',
                  '/Run3S_OffPeak_alpMass5.0'],
            9.5: ['/Run2S_OnPeak_alpMass9.5', '/Run2S_OffPeak_alpMass9.5', '/Run3S_OnPeak_alpMass9.5',
                  '/Run3S_OffPeak_alpMass9.5'],
        }},
    'massInCountingwindow_data5perc.h5':
        {'1-6': {
            0.5: ['/Run1_OnPeak_alpMass0.5', '/Run1_OffPeak_alpMass0.5',
                  '/Run2_OnPeak_alpMass0.5', '/Run2_OffPeak_alpMass0.5',
                  '/Run3_OnPeak_alpMass0.5', '/Run3_OffPeak_alpMass0.5',
                  '/Run4_OnPeak_alpMass0.5', '/Run4_OffPeak_alpMass0.5',
                  '/Run5_OnPeak_alpMass0.5', '/Run5_OffPeak_alpMass0.5',
                  '/Run6_OnPeak_alpMass0.5', '/Run6_OffPeak_alpMass0.5'],
            5.0: ['/Run1_OnPeak_alpMass5.0', '/Run1_OffPeak_alpMass5.0',
                  '/Run2_OnPeak_alpMass5.0', '/Run2_OffPeak_alpMass5.0',
                  '/Run3_OnPeak_alpMass5.0', '/Run3_OffPeak_alpMass5.0',
                  '/Run4_OnPeak_alpMass5.0', '/Run4_OffPeak_alpMass5.0',
                  '/Run5_OnPeak_alpMass5.0', '/Run5_OffPeak_alpMass5.0',
                  '/Run6_OnPeak_alpMass5.0', '/Run6_OffPeak_alpMass5.0'],
            9.5: ['/Run1_OnPeak_alpMass9.5', '/Run1_OffPeak_alpMass9.5',
                  '/Run2_OnPeak_alpMass9.5', '/Run2_OffPeak_alpMass9.5',
                  '/Run3_OnPeak_alpMass9.5', '/Run3_OffPeak_alpMass9.5',
                  '/Run4_OnPeak_alpMass9.5', '/Run4_OffPeak_alpMass9.5',
                  '/Run5_OnPeak_alpMass9.5', '/Run5_OffPeak_alpMass9.5',
                  '/Run6_OnPeak_alpMass9.5', '/Run6_OffPeak_alpMass9.5'],
        }, '7': {
            0.5: ['/Run2S_OnPeak_alpMass0.5', '/Run2S_OffPeak_alpMass0.5', '/Run3S_OnPeak_alpMass0.5',
                  '/Run3S_OffPeak_alpMass0.5'],
            5.0: ['/Run2S_OnPeak_alpMass5.0', '/Run2S_OffPeak_alpMass5.0', '/Run3S_OnPeak_alpMass5.0',
                  '/Run3S_OffPeak_alpMass5.0'],
            9.5: ['/Run2S_OnPeak_alpMass9.5', '/Run2S_OffPeak_alpMass9.5', '/Run3S_OnPeak_alpMass9.5',
                  '/Run3S_OffPeak_alpMass9.5'],
        }}
}


def test_getGroupsForRun_Mass():
    for filename in filename2groups.keys():
        for Run in ['1-6', '7']:
            for alpMass in [0.5, 5.0, 9.5]:
                print(filename, Run, alpMass)
                with mock.patch('generalFuncs.getHdfGroups', side_effect=getGroup):
                    assert cutsFuncs.getGroupsForRun_Mass(Run, alpMass, filename) == expectedGroups[filename][Run][
                        alpMass]