#!/usr/bin/python3
import errno
import os

import pandas as pd


def getCsv(filename):
    with open(filename, 'r') as SP1074csv:
        return pd.read_csv(SP1074csv)


def getHdfGroups(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    with pd.HDFStore(filename) as file:
        return list(file.keys())
