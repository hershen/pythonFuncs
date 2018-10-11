#!/usr/bin/python3
import re
import uproot
import pandas as pd
import glob
import os


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

    for tmpDF in uproot.iterate(filenames, tree, columns, outputtype=pd.DataFrame):
        dfs.append(tmpDF)

    # Concat at end
    df = pd.concat(dfs)

    # change column names to strings
    df.columns = df.columns.astype(str)

    return df


def getSignalFilenames(alpMass, Run, triggered):
    """
    Return list of SIGNAL filenames that match parameters
    :param alpMass: float
    :param Run: Can be 1,2,3,4,5,6, 1-6, 2S, 3S, 7, all
    :param triggered: give triggered filenames
    :return: list of filenames
    """

    baseFolder = '/home/hershen/PhD/ALPs/analysis/ntuples/MC/sig/flatNtuples'
    baseFilename = 'flat'
    if triggered:
        baseFolder = os.path.join(baseFolder, 'triggered')
        baseFilename += '_triggered'

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
    elif Run == 'all':
        fileTemplate = f'{baseFullFilename}_mass{alpMass:.2e}*.root'
        expectedFiles = 8

    filenames = glob.glob(fileTemplate)

    if len(filenames) != expectedFiles:
        raise RuntimeError(f'found {len(filenames)} files in {fileTemplate}, expecting {expectedFiles}')

    return filenames


def getSignalData(alpMass, Run, columns, triggered, mcMatched):
    """
    Return dataframe of SIGNAL events matching parameters

    :param alpMass: float
    :param Run: string. Can be 1,2,3,4,5,6, 1-6, 2S, 3S, 7, all
    :param columns:
    :param triggered: bool
    :param mcMatched: bool
    :return: dataframe
    """

    filenames = getSignalFilenames(alpMass, Run, triggered)

    if mcMatched:
        columns.append('mcMatched')

    # load files
    df = loadDF(filenames, columns=columns)

    if mcMatched:
        df = df[df.mcMatched == 1]

    return df
