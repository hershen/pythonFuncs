#!/usr/bin/python3
import re
import uproot
import pandas as pd
import glob


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


def loadDF(filenames, columns=None, tree="ntp1"):
    if not filenames:
        raise ValueError("filenames is empty")

    dfs = []

    for i, tmpDF in enumerate(uproot.iterate(filenames, tree, columns, outputtype=pd.DataFrame)):
        dfs.append(tmpDF)

    # Concat at end
    df = pd.concat(dfs)

    # change column names to strings
    df.columns = df.columns.astype(str)

    return df


def getSignalFilenames(alpMass, Run):
    """
    Return list of SIGNAL filenames that match parameters
    :param alpMass:
    :param Run: Can be 1,2,3,4,5,6, 1-6, 2S, 3S, 7, all
    :return: list of filenames
    """
    if Run == '1' or Run == '2' or Run == '3' or Run == '4' or Run == '5' or Run == '6':
        fileTemplate = f'/home/hershen/PhD/ALPs/analysis/ntuples/MC/sig/flatNtuples/flat_mass{alpMass:.2e}*Run{Run}.01.root'
        expectedFiles = 1
    elif Run == '1-6':
        fileTemplate = f'/home/hershen/PhD/ALPs/analysis/ntuples/MC/sig/flatNtuples/flat_mass{alpMass:.2e}*Run?.01.root'
        expectedFiles = 6
    elif Run == '2S':
        fileTemplate = f'/home/hershen/PhD/ALPs/analysis/ntuples/MC/sig/flatNtuples/flat_mass{alpMass:.2e}*Y2S*-Ups2S*.root'
        expectedFiles = 1
    elif Run == '3S':
        fileTemplate = f'/home/hershen/PhD/ALPs/analysis/ntuples/MC/sig/flatNtuples/flat_mass{alpMass:.2e}*Y3S*-Ups3S*.root'
        expectedFiles = 1
    elif Run == '7':
        fileTemplate = f'/home/hershen/PhD/ALPs/analysis/ntuples/MC/sig/flatNtuples/flat_mass{alpMass:.2e}*Y?S*-Ups?S*.root'
        expectedFiles = 2
    elif Run == 'all':
        fileTemplate = f'/home/hershen/PhD/ALPs/analysis/ntuples/MC/sig/flatNtuples/flat_mass{alpMass:.2e}*.root'
        expectedFiles = 8

    filenames = glob.glob(fileTemplate)

    if len(filenames) != expectedFiles:
        raise RuntimeError('found {} files'.format(len(filenames)))

    return filenames


def getSignalData(alpMass, Run, columns, triggered, mcMatched):
    """
    Return dataframe of SIGNAL events matching parameters

    :param alpMass: string
    :param Run: string. Can be 1,2,3,4,5,6, 1-6, 2S, 3S, 7, all
    :param columns:
    :param triggered: bool
    :param mcMatched: bool
    :return: dataframe
    """

    filenames = getSignalFilenames(alpMass, Run)

    # load files
    df = loadDF(filenames, columns=columns)

    if triggered:
        # Find triggered events
        L3 = (df.L3OutDch == 1) | (df.L3OutEmc == 1)
        triggered = (df.DigiFGammaGamma == 1) | (df.DigiFSingleGamma == 1) | (
                (L3 == 1) & ((df.BGFIsr == 1) | (df.BGFSingleGammaInvisible == 1) | (df.BGFNeutralHadron == 1)))
        df = df[triggered]

    if mcMatched:
        df = df[df.mcMatched == 1]

    return df
