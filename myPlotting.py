#!/usr/bin/python3
import math
import pickle as pl
import os
import warnings

import matplotlib
import numpy as np

import mathFuncs

#Need to delete all non-root things!!!


def getHistTops_BinEdges_FromAxes(axes):
    patch = axes.patches[0]

    assert isinstance(patch, matplotlib.patches.Polygon), "Currently {} not supported. ..." \
                                                          ". Only histograms with histtype=\'step\'".format(type(patch))

    # list of (x,y) coordinates of each vertex of histograms.
    xy = patch.get_xy()

    # Take only vertical lines (contain all information)
    data = xy[::2]

    # bin edges are first column.
    binEdges = data[:, 0]

    # tops can disregard first entry (probably at 0)
    tops = data[1:, 1]

    return tops, binEdges


def _saveFigFlat(fig, fullFilename, **kwargs):
    """ filename should contain extension"""
    fig.savefig('{}'.format(fullFilename), dpi=300, **kwargs)


def _saveFigPickled(fig, fullFilename):
    """ filename should contain extension"""
    pl.dump(fig, open('{}'.format(fullFilename), 'wb'))


def saveFig(fig, filename, folder='.', subFolder='', **kwargs):
    """ Save a pyplot figure """

    if 'subfolder' in kwargs:
        warnings.warn('Are you sure you didn\'t mean \'subFolder\'?')

    fullDir = os.path.join(folder, 'figureDump', subFolder)

    # Create dir
    if not os.path.isdir(fullDir):
        os.makedirs(fullDir)

    # If extension included, save only thatfiletype
    filename, ext = os.path.splitext(filename)
    fullFilename_noExt = os.path.join(fullDir, filename)
    if ext and (len(ext) == 3 or len(ext) == 4):
        if ext == '.pl':
            _saveFigPickled(fig, fullFilename_noExt + ext)
        else:
            _saveFigFlat(fig, fullFilename_noExt + ext, **kwargs)
        return
    else:
        # the filename has a '.' which does not seperate the extension
        fullFilename_noExt = fullFilename_noExt + ext
        # otherwise, save multiple filetypes
        _saveFigFlat(fig, fullFilename_noExt + '.png', **kwargs)
        _saveFigFlat(fig, fullFilename_noExt + '.pdf', **kwargs)
        _saveFigPickled(fig, fullFilename_noExt + '.pl')


def saveCanvas(canvas, filename, ext='', folder='.', subFolder='', **kwargs):
    """
    Save a Root canvas
    :param canvas:
    :param filename:
    :param ext:
    :param folder: folder inside figureDump to save the figures.
    :return:
    """

    if 'subfolder' in kwargs:
        warnings.warn('Are you sure you didn\'t mean \'subFolder\'?')

    fullDir = os.path.join(folder, 'figureDump', subFolder)

    # Create dir
    if not os.path.isdir(fullDir):
        os.makedirs(fullDir)

    # If extension included, save only thatfiletype
    fullFilename_noExt = os.path.join(fullDir, filename)
    if ext:
        canvas.SaveAs(fullFilename_noExt + ext)
    else:
        # otherwise, save multiple filetypes
        canvas.SaveAs(fullFilename_noExt + '.png')
        canvas.SaveAs(fullFilename_noExt + '.pdf')
        canvas.SaveAs(fullFilename_noExt + '.root')



def _getXvalues_TH1(hist):
    return np.array([hist.GetBinCenter(iBin) for iBin in range(1, hist.GetNbinsX() + 1)])


def _getXvalues_TGraph(graph):
    return np.array([x for x in graph.GetX()])


def setHistNominalYtitle(hist, units=''):
    """
    Set
    :param hist:
    :param units:
    :return:
    """
    binWidth = hist.GetBinWidth(1)
    # Round binWidth to 6 + numSignificanDigits to account for floating point errors
    binWidth = round(binWidth, -int(math.floor(math.log10(abs(binWidth)))) + 6)
    if binWidth.is_integer():
        binWidth = int(binWidth)

    title = "Entries / {}".format(binWidth)
    if units:
        title = title + " " + units

    hist.GetYaxis().SetTitle(title)


def getFitParamaeters(function):
    return {function.GetParName(i): function.GetParameter(i) for i in range(function.GetNpar())}


def getFitParUncertainty(function):
    return {function.GetParName(i) + "_uncertainty": function.GetParError(i) for i in range(function.GetNpar())}


# Addopted from https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python
def latex_float(x, precision=3):
    x = float(x)
    string = f'{x:.{precision}}' if precision else str(x)
    if 'e' in string:
        base, exponent = string.split('e')
        exponent = exponent.lstrip('+0')
        if exponent[0] == '-':
            exponent = '-' + exponent[1:].lstrip('0')
        return f'{base} $\\times 10^{{{exponent}}}$'
    else:
        return string


def weightsForArea(targetArea, otherBinWidth, thisArray, range=None):
    """

    :param targetArea: Target area for histogram
    :param otherBinWidth: Bin width of other histogram
    :param thisArray: The array used to create this histogram
    # :param range: Optional range of the histogram - we don't want entries oustside this range to change our weights
    # weights to use for histograming
    :return: weights array so that hist has targetArea
    """
    #
    # if range:
    #     numEntriesInRange = ((array >= range[0]) & (array <= range[1])).sum()
    # else:
    #     numEntriesInRange = len(array)
    if range:
        raise ValueError("Need to implement range support")

    numEntries = len(thisArray)
    return [targetArea / numEntries / otherBinWidth] * numEntries


def getOptimalRange(lowEdge, highEdge, nBins, binWidths):
    oldRange = highEdge - lowEdge

    if oldRange <= 0:
        raise ValueError(f'range of [{lowEdge}, {highEdge}] is negative')
    oldBinWidth = oldRange / nBins

    try:
        newBinWidth = binWidths[np.searchsorted(binWidths, oldBinWidth)]
    except IndexError:
        raise ValueError(
            f'Provided bin width {oldBinWidth} outside bin width range provided [{binWidths[0]}, {binWidths[-1]}]')

    newRange = newBinWidth * nBins
    deltaRange = newRange - oldRange
    return [lowEdge - deltaRange / 2, highEdge + deltaRange / 2]

def hist_binned(tops, edges, **args):
    binCenters = mathFuncs.listCenters(edges)
    return matplotlib.pyplot.hist(binCenters, bins=edges, weights=tops, **args)
