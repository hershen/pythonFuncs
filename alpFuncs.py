#!/usr/bin/python3
import re
import uproot
import pandas as pd


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
