#!/usr/bin/python3
import re
import pickle as pl
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

SCALE_SP1074_TO_DATA = SCALE_MC_TO_DATA

SCALE_SP7957_TO_DATA = {'Y2S_OffPeak': 0.212249235772358,
                        'Y2S_OnPeak': 0.211250957755102,
                        'Y3S_OffPeak': 0.199735021929825,
                        'Y3S_OnPeak': 0.236864720234604,
                        'Y4S_OffPeak': 0.678979604596849,
                        'Y4S_OnPeak': 0.656229936094144}

SCALE_SP2400_TO_DATA = {'Y2S_OffPeak':4.881377136,
                        'Y2S_OnPeak': 4.92927027672283,
                        'Y3S_OffPeak': 3.52962600010526,
                        'Y3S_OnPeak': 4.20954351890712,
                        'Y4S_OffPeak': 28.9508817394704,
                        'Y4S_OnPeak': 26.3976194064715}


LUMI_FULL_DATASET_FB = {'Y2S_OffPeak': 1.41884,
                        'Y2S_OnPeak': 13.560651,
                        'Y3S_OffPeak': 2.602262,
                        'Y3S_OnPeak': 27.852024,
                        'Y4S_OffPeak': 43.922002,
                        'Y4S_OnPeak': 424.290696,
                        'Run7_OffPeak': 4.021102,
                        'Run7_OnPeak': 41.412675}

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
    return pd.concat(dfs)


def getSignalFilenames(alpMass, Run, minE12cm=None):
    """
    Return list of SIGNAL filenames that match parameters
    :param alpMass: float
    :param Run: Can be 1,2,3,4,5,6, 1-6, 2S, 3S, 7, 1-7
    :param minE12cm: minE12cm cut. Default is 0.7 GeV
    :return: list of filenames
    """

    if minE12cm:
        subFolder = os.path.join('looseMinE12cmCut', str(minE12cm))
    else:
        subFolder = ''
    baseFolder = f'/home/hershen/PhD/ALPs/analysis/ntuples/MC/sig/{subFolder}/flatNtuples'
    baseFilename = 'flat'

    baseFullFilename = os.path.join(baseFolder, baseFilename)

    if Run == '1' or Run == '2' or Run == '3' or Run == '4' or Run == '5' or Run == '6':
        fileTemplate = f'{baseFullFilename}_mass{alpMass:.2e}*Run{Run}.01*.root'
        expectedFiles = 1
    elif Run == '1-6':
        fileTemplate = f'{baseFullFilename}_mass{alpMass:.2e}*Run?.01*.root'
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


def getSignalData(alpMass, Run, columns, mcMatched, preselection=None, minE12cm=None):
    """
    Return dataframe of SIGNAL events matching parameters

    :param alpMass: float
    :param Run: string. Can be 1,2,3,4,5,6, 1-6, 2S, 3S, 7, all
    :param columns:
    :param mcMatched: bool
    :return: dataframe
    """

    filenames = getSignalFilenames(alpMass, Run, minE12cm)

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
_lowAlpMasses = [0.05, 0.1, 0.135, 0.15, 0.2, 0.25, 0.3, 0.4, 0.548, 0.6, 0.7, 0.8, 0.9, 0.958]


def getAlpMasses(Run='all'):
    res = _lowAlpMasses.copy() + _nominalAlpMasses.copy()
    if Run == '1-6' or Run == 'all':
        res = res + _highAlpMasses.copy()
    if Run == '7':
        res.remove(0.05)
    res.sort()
    return res


_massesWith10ksignalEvents = [0.02, 0.05, 0.1, 0.135, 0.4]

def getNumberOfGeneratedSignal(Run, alpMass):
    if Run == '1-6':
        return 49965
    elif Run == '7':
        if alpMass in _massesWith10ksignalEvents:
            return 19996
        else:
            return 4997
    elif Run == '2S':
        if alpMass in _massesWith10ksignalEvents:
            return 10000
        else:
            return 2500
    elif Run == '3S':
        if alpMass in _massesWith10ksignalEvents:
            return 9996
        else:
            return 2497
    else:
        raise ValueError(f'Run {Run} not recognized')

_nonNominalGeneratedEvents_Y4S = {1.98: 49947, 6.71: 49410, 4.72: 48299, 4.83: 49448, 7.60: 49105, 6.17: 48210, 7.59: 49417, 9.79: 48870, 5.84: 49030, 8.59: 48234, 4.37: 49755,
                                  5.99: 49843, 0.27: 49796, 8.90: 49539, 4.80: 49961, 9.09: 49097, 7.91: 49780, 2.38: 49612, 4.22: 49178, 0.58: 49696, 7.42: 48781, 9.94: 49843,
                                  5.82: 49891, 6.67: 49311, 8.47: 49883, 5.07: 49496, 3.99: 49168, 8.68: 48801, 8.39: 49585, 8.61: 48716, 3.63: 48829, 9.24: 49828, 4.20: 48924,
                                  9.16: 48671, 0.46: 49696, 9.59: 49483, 10.15: 49105, 10.14: 49243}
_nonNominalGeneratedMasses = np.array(list(_nonNominalGeneratedEvents_Y4S.keys()))
_nonNominalGeneratedEvents= np.array(list(_nonNominalGeneratedEvents_Y4S.values()))
def getNumberOf10MeVresuotionGeneratedSignal(YnS, alpMass):
    if not isinstance(alpMass, float):
        raise ValueError(f'alpMass is expected to be float')

    if YnS == 'Y2S':
        if np.isclose(alpMass, 4.29):
            return 9136
        elif np.isclose(alpMass, 4.81):
            return 3353
        elif np.isclose(alpMass, 6.97):
            return 2143
        else:
            return 10000
    elif YnS == 'Y3S':
        if np.isclose(alpMass, 2.11):
            return 9969
        elif np.isclose(alpMass, 2.92):
            return 8802
        elif np.isclose(alpMass, 6.65):
            return 8478
        else:
            return 9996
    elif YnS == 'Y4S':
        closeToNonNominal = np.isclose(_nonNominalGeneratedMasses, alpMass) 
        if np.any(closeToNonNominal):
            index = np.argmax(closeToNonNominal)
            return _nonNominalGeneratedEvents[index]
        else:
            return 49965
    else:
        raise ValueError(f'Run {YnS} not recognized')

def getRelevantRangeByRunType(runType):
    if 'Y2S' in runType:
        return [0,10.02326*0.975]
    elif 'Y3S' in runType:
        return [0, 10.3552*0.975]
    elif 'Y4S' in runType:
        return [0,12]
    else:
        raise ValueError

def chebyshevDegree(mass, df):
    idx = np.argmax(mass<df.edges.values)
    if df.edges.values[idx-1] <= mass < df.edges.values[idx]:
        return int(df.chebyshevDegree.values[idx-1])
    return -1


def getWindowLimitPolynomials():
    return pl.load(open('/home/hershen/PhD/ALPs/analysis/cuts/calcCountingWindow/signalWidthPercantageOfPeak/windowPolynomials.pl', 'rb'))

def getIdealFitRange(branchName, rebin, numOnSides=2):
    YnS = branchName.split('_')[1]
    mass = float(getMass(branchName))
    windowLimitPolynomials = getWindowLimitPolynomials()
    signalUpperLimit = windowLimitPolynomials[f'{YnS} upper limit'](mass)
    signalLowerLimit = windowLimitPolynomials[f'{YnS} lower limit'](mass)
    signalWindow =  signalUpperLimit - signalLowerLimit

    upperLimit = np.around((mass + signalUpperLimit + numOnSides*signalWindow)/rebin, decimals=3)*rebin
    lowerLimit = max(0, np.around((mass + signalLowerLimit - numOnSides*signalWindow)/rebin, decimals=3)*rebin)
    return [lowerLimit, upperLimit]
