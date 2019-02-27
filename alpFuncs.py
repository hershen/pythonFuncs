#!/usr/bin/python3
import re
import uproot
import pandas as pd
import glob
import os

import numpy as np

SCALE_MC_TO_DATA = {'Y2S_OffPeak': 0.436278194778745,
                    'Y2S_OnPeak': 0.426093591118402,
                    'Y3S_OffPeak': 0.29208690815739,
                    'Y3S_OnPeak': 0.374345675403956,
                    'Y4S_OffPeak': 5.85653901412377,
                    'Y4S_OnPeak': 5.61997987743639}


def getMass(filename):
    """Return mass from the filename"""

    # remove up to 'mass'
    filename = filename[filename.find('mass') + 4:]

    return filename[:filename.find('_')]


def getRun(filename):
    """Return Run from the filename"""

    # string after run
    postRunLoc = filename.find('.01.')

    # remove up to 'mass'
    preRunLoc = re.search('_.-', filename).end() + 3  # +3 for either 'Run' or 'Ups'

    return filename[preRunLoc:postRunLoc]


def loadDF(filenames, columns=None, tree="ntp1", preselection=None):
    """

    :param filenames:
    :param columns:
    :param tree:
    :param preselection: Function that Determines which rows to keep.
                         The function receives the loaded dataframe and
                         returns a list of True or False.
                         Example: def cutEtaMass(df):
                                    return (df.eta_Mass > 1.5) & (df.eta_Mass < 3)
    :return:
    """
    if not filenames:
        raise ValueError("filenames is empty")

    dfs = []

    for tmpDf in uproot.iterate(filenames, tree, columns, outputtype=pd.DataFrame, namedecode="utf-8"):

        # preselect
        if preselection:
            tmpDf = tmpDf[preselection(tmpDf)]

        dfs.append(tmpDf)

    # Concat at end
    df = pd.concat(dfs)

    # change column names to strings
    # df.columns = df.columns.astype(str)

    return df


def getSignalFilenames(alpMass, Run):
    """
    Return list of SIGNAL filenames that match parameters
    :param alpMass: float
    :param Run: Can be 1,2,3,4,5,6, 1-6, 2S, 3S, 7, 1-7
    :return: list of filenames
    """

    baseFolder = '/home/hershen/PhD/ALPs/analysis/ntuples/MC/sig/flatNtuples'
    baseFilename = 'flat'

    baseFullFilename = os.path.join(baseFolder, baseFilename)

    if Run == '1' or Run == '2' or Run == '3' or Run == '4' or Run == '5' or Run == '6':
        fileTemplate = f'{baseFullFilename}_mass{alpMass:.2e}*Run{Run}.01.root'
        expectedFiles = 1
    elif Run == '1-6':
        fileTemplate = f'{baseFullFilename}_mass{alpMass:.2e}*Run?.01.root'
        expectedFiles = 6
    elif Run == '2S':
        fileTemplate = f'{baseFullFilename}_mass{alpMass:.2e}*Y2S*-Ups2S*.root'
        expectedFiles = 1
    elif Run == '3S':
        fileTemplate = f'{baseFullFilename}_mass{alpMass:.2e}*Y3S*-Ups3S*.root'
        expectedFiles = 1
    elif Run == '7':
        fileTemplate = f'{baseFullFilename}_mass{alpMass:.2e}*Y?S*-Ups?S*.root'
        expectedFiles = 2
    elif Run == '1-7':
        fileTemplate = f'{baseFullFilename}_mass{alpMass:.2e}*.root'
        expectedFiles = 8
    else:
        raise ValueError(f'Run {Run} not recognized')

    filenames = glob.glob(fileTemplate)

    if len(filenames) != expectedFiles:
        raise RuntimeError(f'found {len(filenames)} files in {fileTemplate}, expecting {expectedFiles}')

    return filenames


def getSignalData(alpMass, Run, columns, mcMatched, preselection=None):
    """
    Return dataframe of SIGNAL events matching parameters

    :param alpMass: float
    :param Run: string. Can be 1,2,3,4,5,6, 1-6, 2S, 3S, 7, all
    :param columns:
    :param mcMatched: bool
    :return: dataframe
    """

    filenames = getSignalFilenames(alpMass, Run)

    try:
        columnsToLoad = columns.copy()
    except AttributeError:  # If columns is none
        columns = uproot.open(filenames[0])['ntp1'].keys()
        columnsToLoad = columns

    _extendColumns = ['mcMatched', 'entryNum', 'e1Mag', 'eta_Mass']
    if mcMatched:
        # load more values, used for mcMatching
        columnsToLoad.extend(_extendColumns)

    # load files
    df = loadDF(filenames, columns=columnsToLoad, preselection=preselection)

    if mcMatched:
        df = df[df.mcMatched == 1]

        # ----------------------------------
        # Remove duplicate mcMatched entries - probably VERY inefficient
        # ----------------------------------

        # Get 1 entry for each event with duplicates
        duplicateRepresentatives = df.loc[df.entryNum.shift() == df.entryNum]

        # create tmp DF, so we can keep track of mass difference to alp mass
        tmpDf = df[(df.entryNum.isin(duplicateRepresentatives.entryNum)) & (
            df.e1Mag.isin(duplicateRepresentatives.e1Mag))].copy()
        tmpDf['massDiff'] = (tmpDf.eta_Mass - alpMass).abs()

        # Find which rows should be kept (this is done instead of finding which rows to drop because some events might
        # have more than 2 duplicates
        rowsToKeep = tmpDf.groupby('entryNum', sort=False).massDiff.min()

        # transform to rows, because groupby loses this info
        idxsToKeep = tmpDf[(tmpDf.entryNum.isin(rowsToKeep.index)) & (tmpDf.massDiff.isin(rowsToKeep.values))].index

        # Find indices to drop
        idxsToDrop = tmpDf.drop(idxsToKeep).index

        # drop from main DF
        df.drop(idxsToDrop, inplace=True)

        # remove columns added but not requested
        if 'e1Mag' not in columns:
            df.drop('e1Mag', axis=1, inplace=True)
        if 'eta_Mass' not in columns:
            df.drop('eta_Mass', axis=1, inplace=True)
        if 'mcMatched' not in columns:
            df.drop('mcMatched', axis=1, inplace=True)
        if 'entryNum' not in columns:
            df.drop('entryNum', axis=1, inplace=True)

    return df


def getDatasets(Runs):
    if Runs == '1-6':
        return ['Y4S_OffPeak', 'Y4S_OnPeak']
    if Runs == '7':
        return ['Y2S_OffPeak', 'Y2S_OnPeak', 'Y3S_OffPeak', 'Y3S_OnPeak']
    if Runs == '1-7':
        return ['Y2S_OffPeak', 'Y2S_OnPeak', 'Y3S_OffPeak', 'Y3S_OnPeak', 'Y4S_OffPeak', 'Y4S_OnPeak']
    raise ValueError(f'Unkown Run {Runs}')


def getTriggered(df):
    return (df.DigiFGammaGamma == 1) | (df.DigiFSingleGamma == 1) | \
           (((df.L3OutDch == 1) | (df.L3OutEmc == 1)) & (df.BGFSingleGammaInvisible == 1))

_nominalAlpMasses = list(np.arange(0.5, 10.5, 0.5))
_highAlpMasses = [10.3, 10.4, 10.5]
_lowAlpMasses = [0.1, 0.135, 0.15, 0.2, 0.25, 0.3, 0.4, 0.548, 0.6, 0.7, 0.8, 0.9, 0.958]

def getAlpMasses(Run='all'):
    res = _lowAlpMasses.copy() + _nominalAlpMasses.copy()
    if Run == '1-6' or Run == 'all':
        res = res + _highAlpMasses.copy()
    res.sort()
    return res