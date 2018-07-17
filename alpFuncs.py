#!/usr/bin/python3

def getMass(filename):
    """Return mass from the filename"""

    #remove up to 'mass'
    filename = filename[filename.find('mass') + 4:]

    return filename[:filename.find('_')]

def getRun(filename):
    """Return Run from the filename"""

    #remove up to 'mass'
    filename = filename[filename.find('Run') + 3:]

    return filename[:filename.find('.')]
