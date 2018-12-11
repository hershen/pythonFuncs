#!/usr/bin/python3
import pandas as pd

def getCsv(filename):
    with open(filename, 'r') as SP1074csv:
        return pd.read_csv(SP1074csv)


def getHdfGroups(filename):
    with pd.HDFStore(filename) as file:
        return list(file.keys())