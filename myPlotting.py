#!/usr/bin/python3
import pickle as pl
import os
import ROOT
import numpy as np
import mathFuncs
import root_numpy
import matplotlib
import math

colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan, ROOT.kYellow]


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


def _saveFigFlat(fig, fullFilename):
    """ filename should contain extension"""
    fig.savefig('{}'.format(fullFilename), dpi=300)


def _saveFigPickled(fig, fullFilename):
    """ filename should contain extension"""
    pl.dump(fig, open('{}'.format(fullFilename), 'wb'))


def saveFig(fig, filename, folder='.'):
    """ Save a pyplot figure """

    fullDir = os.path.join(folder, 'figureDump')

    # Create dir
    if not os.path.isdir(fullDir):
        os.mkdir(fullDir)

    # If extension included, save only thatfiletype
    filename, ext = os.path.splitext(filename)
    fullFilename_noExt = os.path.join(fullDir, filename)
    if ext and (len(ext) == 3 or len(ext) == 4):
        if ext == '.pl':
            _saveFigPickled(fig, fullFilename_noExt + ext)
        else:
            _saveFigFlat(fig, fullFilename_noExt + ext)
        return
    else:
        # the filename has a '.' which does not seperate the extension
        fullFilename_noExt = fullFilename_noExt + ext
        # otherwise, save multiple filetypes
        _saveFigFlat(fig, fullFilename_noExt + '.png')
        _saveFigFlat(fig, fullFilename_noExt + '.pdf')
        _saveFigPickled(fig, fullFilename_noExt + '.pl')


def saveCanvas(canvas, filename, ext='', folder='.'):
    """ Save a pyplot figure """

    fullDir = os.path.join(folder, 'figureDump')

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


def getPullGraph(xValues, residuals):
    """

    :param xValues:
    :param residuals:
    :return:
    """

    xValues = np.asarray(np.atleast_1d(xValues), dtype=float)
    residuals = np.asarray(np.atleast_1d(residuals), dtype=float)

    # sanity
    assert len(xValues) == len(residuals), "xValues size {} != residuals size {}".format(len(xValues), len(residuals))

    if len(xValues) == 0:
        return ROOT.TGraph()

    residualsGraph = ROOT.TGraph(len(xValues), xValues, residuals)
    # fill_graph(residualsGraph, re)
    maxResidual = np.ma.masked_invalid(np.abs(residuals)).max()
    maxResidualInteger = np.ceil(maxResidual)
    residualsGraph.SetMaximum(maxResidualInteger)
    residualsGraph.SetMinimum(-maxResidualInteger)
    residualsGraph.GetYaxis().SetNdivisions(int(maxResidualInteger),
                                            ROOT.kFALSE)  # false - no optimization - forces current value

    return residualsGraph


def _getXvalues_TH1(hist):
    return np.array([hist.GetBinCenter(iBin) for iBin in range(1, hist.GetNbinsX() + 1)])


def _getXvalues_TGraph(graph):
    return np.array([x for x in graph.GetX()])


def getXvalues(rootObj):
    if isinstance(rootObj, ROOT.TH1):
        return _getXvalues_TH1(rootObj)
    elif isinstance(rootObj, ROOT.TGraph):
        return _getXvalues_TGraph(rootObj)

    raise ValueError("{} is not supported".format(type(rootObj)))


def _calcPulls_graphErrors(graphErrors, modelFunc):
    """
    Calculate pulls of model function at each (x,y) value of graph
    :param graphErrors:
    :param modelFunc:
    :return:
    """
    yValues = [y for y in graphErrors.GetY()]
    stds = [ey for ey in graphErrors.GetEY()]

    xValues = getXvalues(graphErrors)
    expectedValues = root_numpy.evaluate(modelFunc, xValues)
    return mathFuncs.calcPulls(yValues, stds, expectedValues)


def _calcPulls_TH1(hist, modelFunc):
    """
    Calculate pulls of model function at hist bin centers
    :param hist:
    :param modelFunc:
    :return:
    """
    yValues = root_numpy.hist2array(hist)
    stds = [hist.GetBinError(iBin) for iBin in range(1, hist.GetNbinsX() + 1)]
    xValues = getXvalues(hist)
    expectedValues = root_numpy.evaluate(modelFunc, xValues)
    return mathFuncs.calcPulls(yValues, stds, expectedValues)


def calcPulls_fromRootObj(rootObj, modelFunc):
    if isinstance(rootObj, ROOT.TH1):
        return _calcPulls_TH1(rootObj, modelFunc)

    elif isinstance(rootObj, ROOT.TGraphErrors):
        return _calcPulls_graphErrors(rootObj, modelFunc)

    raise ValueError("{} is not supported".format(type(rootObj)))


class PullCanvas:
    """

    """

    def __init__(self, canvas, topObject, func, maxPull=5):
        """

        :param canvas:
        :param topObject:
        :param func:
        :param maxPull:
        """
        assert func.IsValid(), "PullCanvas::PullCanvas: function isn't valid."
        self.canvas = canvas
        self.topObject = topObject
        self.function = func
        self.maxPull = maxPull

        self._prepareCanvas()

        # Calculate pulls
        pulls = calcPulls_fromRootObj(self.topObject, self.function)

        # Get x values of topObj
        xValues = getXvalues(self.topObject)

        # Mask of values |val| < maxPull
        belowMaxPullMask = np.abs(pulls) < self.maxPull

        # Mask xValues outside function range
        rangeMin = ROOT.Double(0)
        rangeMax = ROOT.Double(0)
        self.function.GetRange(rangeMin, rangeMax)
        xInRangeMask = np.logical_and(xValues >= rangeMin, xValues <= rangeMax)

        # mask for pullGraph
        pullGraphMask = np.logical_and(xInRangeMask, belowMaxPullMask)

        # Create pull graph
        self._pullGraph = getPullGraph(xValues[(pullGraphMask)], pulls[pullGraphMask])

        # mask for pullOverflowGraph
        # bins with 0 entries have 0 std and therefore a pull of +-inf
        pullOverflowGraphMask = np.logical_and.reduce((xInRangeMask, ~belowMaxPullMask, ~np.isinf(pulls)))

        # Create pull graph of values |val| >= maxPull
        self._pullOverflowGraph = getPullGraph(xValues[pullOverflowGraphMask],
                                               np.sign(pulls[pullOverflowGraphMask]) * self.maxPull)
        self._prepareObjects()

    def draw(self, options=""):
        self.canvas.GetPad(1).cd()
        if isinstance(self.topObject, ROOT.TH1):
            if options:
                self.topObject.Draw(options)
            else:
                self.topObject.Draw()
        elif isinstance(self.topObject, ROOT.TGraph):
            if options:
                self.topObject.Draw(options)
            else:
                self.topObject.Draw("AP")
        else:
            raise RuntimeError("Don't know how to draw object of type {}".format(type(self.topObject)))

        self.function.Draw("Same")

        self.canvas.GetPad(2).cd()
        self.pullMultiGraph.Draw("AP")

    def _prepareCanvas(self, bottomPadYpercentage=0.22, bottomTopSeperation=0.025):
        """
        Prepare the top and botom canvases
        :param bottomPadYpercentage:
        :param bottomTopSeperation:
        :return:
        """
        self.canvas.cd()
        self.canvas.Clear()
        self.canvas.Divide(1, 2)
        self.canvas.GetPad(1).SetPad(0.0, bottomPadYpercentage, 1, 1)
        self.canvas.GetPad(1).SetBottomMargin(bottomTopSeperation)
        self.canvas.GetPad(1).SetRightMargin(0.05)
        self.canvas.GetPad(2).SetPad(0.0, 0.0, 1, bottomPadYpercentage)
        self.canvas.GetPad(2).SetBottomMargin(0.37)
        self.canvas.GetPad(2).SetTopMargin(0.0)
        self.canvas.GetPad(2).SetRightMargin(0.05)
        self.canvas.GetPad(2)
        self.canvas.GetPad(2).SetGridy()

        self.canvas.cd()

    def _prepareObjects(self):
        validOverflow = bool(self._pullOverflowGraph.GetN())

        # Don't draw x labels and title on top object
        self.topObject.GetXaxis().SetLabelSize(0.0)
        self.topObject.GetXaxis().SetTitleSize(0.0)

        # Create multiGraph
        self.pullMultiGraph = ROOT.TMultiGraph()
        self.pullMultiGraph.Add(self._pullGraph)
        self.pullMultiGraph.SetMinimum(self._pullGraph.GetMinimum())
        self.pullMultiGraph.SetMaximum(self._pullGraph.GetMaximum())

        if validOverflow:
            self.pullMultiGraph.Add(self._pullOverflowGraph)
            self.pullMultiGraph.SetMinimum(self._pullOverflowGraph.GetMinimum())
            self.pullMultiGraph.SetMaximum(self._pullOverflowGraph.GetMaximum())

        # draw multigraph so we can GetXaxis
        self.canvas.GetPad(2).cd()
        self.pullMultiGraph.Draw("A")

        # Set y axis divisions
        if validOverflow:
            self.pullMultiGraph.GetYaxis().SetNdivisions(self._pullOverflowGraph.GetYaxis().GetNdivisions(),
                                                         ROOT.kFALSE)
        else:
            self.pullMultiGraph.GetYaxis().SetNdivisions(self._pullGraph.GetYaxis().GetNdivisions(), ROOT.kFALSE)

        # sync top and botom xAxes ranges
        self.pullMultiGraph.GetXaxis().SetLimits(self.topObject.GetXaxis().GetXmin(),
                                                 self.topObject.GetXaxis().GetXmax())

        # Set axis label and Title size to absolute
        self.topObject.GetYaxis().SetLabelFont(43);
        self.pullMultiGraph.GetXaxis().SetLabelFont(43)
        self.pullMultiGraph.GetYaxis().SetLabelFont(43)
        # Title
        self.topObject.GetYaxis().SetTitleFont(43)
        self.pullMultiGraph.GetXaxis().SetTitleFont(43)
        self.pullMultiGraph.GetYaxis().SetTitleFont(43)

        # Delete graph title
        self.pullMultiGraph.SetTitle("")

        # Set x + y axis label size
        textSize = min(0.04 * self.canvas.cd(1).GetWh(), 0.04 * self.canvas.cd(1).GetWw())
        self.topObject.GetYaxis().SetLabelSize(textSize)

        self.pullMultiGraph.GetYaxis().SetLabelSize(textSize)
        # x axis
        self.pullMultiGraph.GetXaxis().SetLabelSize(textSize)

        # Set axis title sizes
        self.pullMultiGraph.GetXaxis().SetTitleSize(textSize)
        self.pullMultiGraph.GetYaxis().SetTitleSize(textSize)
        self.topObject.GetYaxis().SetTitleSize(textSize)

        # Set title offsets
        self.pullMultiGraph.GetXaxis().SetTitleOffset(3.75)

        # Set bottom x title
        self.pullMultiGraph.GetXaxis().SetTitle(self.topObject.GetXaxis().GetTitle())
        # Set y title
        self.pullMultiGraph.GetYaxis().SetTitle("Pull (#sigma)")

        # set marker style
        self._pullGraph.SetMarkerStyle(ROOT.kFullCircle)
        self._pullOverflowGraph.SetMarkerStyle(ROOT.kFullCircle)

        # set marker color (like RooFit)
        self._pullGraph.SetMarkerColor(ROOT.kBlue)
        self._pullOverflowGraph.SetMarkerColor(ROOT.kRed)

        # Set marker size
        markerSize = np.interp(self._pullGraph.GetN(), [1, 17500], [0.8, 0.1])
        self._pullGraph.SetMarkerSize(markerSize)
        self._pullOverflowGraph.SetMarkerSize(markerSize)

    def getTopPad(self):
        return self.canvas.GetPad(1)

    def getBottomPad(self):
        return self.canvas.GetPad(2)


class Legend(ROOT.TLegend):
    def __init__(self, x1, y1, x2, y2):
        ROOT.TLegend.__init__(self, x1, y1, x2, y2)
        self.SetTextFont(22)
        self.SetFillStyle(0)  # Transpartent fill color
        self.SetBorderSize(1)  # No shadow


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
    title = "Entries / {}".format(binWidth)
    if units:
        title = title + " " + units

    hist.GetYaxis().SetTitle(title)


class PaveText(ROOT.TPaveText):
    def __init__(self, x1, y1, x2, y2, options="NDCNB"):
        ROOT.TPaveText.__init__(self, x1, y1, x2, y2, options)
        self.SetTextFont(22)
        self.SetFillStyle(0)  # Make fill color transparent
        self.SetBorderSize(0)  # No border

    # Constructor which puts text at the top of a histogram. Just need to set x2
    @classmethod
    def atTop(cls, x2, options="NDCNB"):
        if ROOT.gPad:
            ROOT.gPad.Update()
            bottom = 1 - ROOT.gPad.GetTopMargin()
            left = ROOT.gPad.GetLeftMargin()
        else:
            bottom = 1 - ROOT.gStyle.GetPadTopMargin()
            left = ROOT.gStyle.GetPadLeftMargin()
        pt = cls(left, bottom, x2, 1, options)
        pt.SetTextAlign(12)
        return pt


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
