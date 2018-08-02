#!/usr/bin/python3
import pickle as pl
import os

def getHistBinEdges_topsFromAxes(axes):
    #list of (x,y) coordinates of each vertex of histograms.
    xy = axes.patches[0].get_xy()

    #Take only vertical lines (contain all information)
    data = xy[::2]

    #bin edges are first column.
    binEdges = data[:,0]

    #tops can disregard first entry (probably at 0)
    tops = data[1:,1]

    return binEdges, tops

def _saveFigFlat(fig, filename):
    """ filename should contain extension"""
    fig.savefig('figureDump/{}'.format(filename), dpi=300)

def _saveFigPickled(fig, filename):
    """ filename should contain extension"""
    pl.dump(fig, open('figureDump/{}'.format(filename), 'wb'))

def saveFig(fig, filename):
    """ Save a pyplot figure """
    filename, ext = os.path.splitext(filename)

    #Create figureDump dir
    if not os.path.isdir('figureDump'):
        os.mkdir('figureDump')

    #If extension included, save only thatfiletype
    if ext:
        if ext == '.pl':
            _saveFigPickled(fig, filename + ext)
        else:
            _saveFigFlat(fig, filename + ext)
        return

    #otherwise, save multiple filetypes
    _saveFigFlat(fig, filename + '.png')
    _saveFigFlat(fig, filename + '.pdf')
    # _saveFigPickled(fig, filename + '.pl')
    pl.dump(fig, open('figureDump/{}'.format(filename), 'wb'))
