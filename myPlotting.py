#!/usr/bin/python3
import pickle as pl
import os
import ROOT
import numpy as np
import mathFuncs
import root_numpy
import matplotlib


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
    if ext:
        if ext == '.pl':
            _saveFigPickled(fig, fullFilename_noExt + ext)
        else:
            _saveFigFlat(fig, fullFilename_noExt + ext)
        return

    # otherwise, save multiple filetypes
    _saveFigFlat(fig, fullFilename_noExt + '.png')
    _saveFigFlat(fig, fullFilename_noExt + '.pdf')
    _saveFigPickled(fig, fullFilename_noExt + '.pl')


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

    residualsGraph = ROOT.TGraph(len(xValues), xValues, residuals)
    # fill_graph(residualsGraph, re)
    maxResidual = np.abs(residuals).max()
    maxResidualInteger = np.ceil(maxResidual)
    residualsGraph.SetMaximum(maxResidualInteger)
    residualsGraph.SetMinimum(-maxResidualInteger)
    residualsGraph.GetYaxis().SetNdivisions(int(maxResidualInteger),
                                            ROOT.kFALSE)  # false - no optimization - forces current value

    return residualsGraph


def _getXvalues_TH1(hist):
    return [hist.GetBinCenter(iBin) for iBin in range(1, hist.GetNbinsX() + 1)]


def _getXvalues_TGraph(graph):
    return [x for x in graph.GetX()]


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


class ResidualCanvas:
    def __init__(self, canvas, topObject, func):
        assert func.IsValid(), "ResidualCanvas::ResidualCanvas: function isn't valid."

        self.canvas = canvas
        self.topObject = topObject
        self.function = func

        self._prepareCanvas()

        residuals = calcPulls_fromRootObj(self.topObject, self.function)

        self.residualGraph = ROOT.TGraph(getPullGraph(getXvalues(self.topObject), residuals))

        # self.prepareObjects()

    #
    #
    def draw(self):
        self.canvas.GetPad(1).cd()
        self.topObject.Draw("AP")
        self.function.Draw("Same")

        self.canvas.GetPad(2).cd()
        self.residualGraph.Draw("AP")

    def _prepareCanvas(self, bottomPadYpercentage=0.22, bottomTopSeperation=0.05):
        """
        Prepare the top and botom canvases
        :param bottomPadYpercentage:
        :param bottomTopSeperation:
        :return:
        """
        self.canvas.cd()
        self.canvas.Divide(1, 2)
        self.canvas.GetPad(1).SetPad(0.0, bottomPadYpercentage, 1, 1)
        self.canvas.GetPad(1).SetBottomMargin(bottomTopSeperation)
        self.canvas.GetPad(1).SetRightMargin(0.05)
        self.canvas.GetPad(2).SetPad(0.0, 0.0, 1, bottomPadYpercentage)
        self.canvas.GetPad(2).SetBottomMargin(0.32)
        self.canvas.GetPad(2).SetTopMargin(0.0)
        self.canvas.GetPad(2).SetRightMargin(0.05)
        self.canvas.GetPad(2)
        self.canvas.GetPad(2).SetGridy()

        self.canvas.cd()
# #
# #   // Prepare objects for drawing in a prettyResidualGraph
# #   void prepareObjects() {
# #     mtopObject.GetXaxis()->SetLabelSize(0.0); // Don't draw x labels on top object
# #     // mtopObject.GetXaxis()->SetTitleSize(0.0); // Don't draw x title on top object
# #
# #     // Set axis label and Title size to absolute
# #     mtopObject.GetYaxis()->SetLabelFont(43);     // Absolute font size in pixel (precision 3)
# #     mresidualGraph.GetXaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)
# #     mresidualGraph.GetYaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)
# #     // Title
# #     mtopObject.GetYaxis()->SetTitleFont(43);     // Absolute font size in pixel (precision 3)
# #     mresidualGraph.GetXaxis()->SetTitleFont(43); // Absolute font size in pixel (precision 3)
# #     mresidualGraph.GetYaxis()->SetTitleFont(43); // Absolute font size in pixel (precision 3)
# #
# #     // Set x + y axis label size
# #     const double labelSize = std::min(0.03 * mcanvas.cd(1)->GetWh(), 0.03 * mcanvas.cd(1)->GetWw());
# #     mtopObject.GetYaxis()->SetLabelSize(labelSize);
# #
# #     mresidualGraph.GetYaxis()->SetLabelSize(labelSize);
# #     // x axis
# #     mresidualGraph.GetXaxis()->SetLabelSize(labelSize);
# #
# #     // Set axis title sizes
# #     const double titleSize = std::min(0.03 * mcanvas.cd(1)->GetWh(), 0.03 * mcanvas.cd(1)->GetWw());
# #     mresidualGraph.GetXaxis()->SetTitleSize(titleSize);
# #     mresidualGraph.GetYaxis()->SetTitleSize(titleSize);
# #     mtopObject.GetYaxis()->SetTitleSize(titleSize);
# #
# #     // Set title offsets
# #     mresidualGraph.GetXaxis()->SetTitleOffset(3.75);
# #
# #     // Set bottom x title
# #     mresidualGraph.GetXaxis()->SetTitle(mtopObject.GetXaxis()->GetTitle());
# #     // Set y title
# #     mresidualGraph.GetYaxis()->SetTitle("Pull (#sigma)");
# #
# #     // Set residual y axis divisions
# #     const auto maxResidual =
# #         std::abs(*std::max_element(mresidualGraph.GetY(), mresidualGraph.GetY() + mresidualGraph.GetN() - 1,
# #                                    [](const double residual1, const double residual2) { // find max absolute value residual
# #                                      return std::abs(residual1) < std::abs(residual2);
# #                                    }));
# #     mresidualGraph.SetMaximum(std::ceil(maxResidual));
# #     mresidualGraph.SetMinimum(-std::ceil(maxResidual));
# #     const int maxDivisions = std::min(5., std::ceil(maxResidual));
# #     mresidualGraph.GetYaxis()->SetNdivisions(maxDivisions, false); // false - no optimization - forces current value
# #
# #     // Set marker size
# #     double markerSize = myFuncs::linearInterpolate(100, 17500, 1.2, 0.1, mresidualGraph.GetN());
# #     // markerSize = std::min(static_cast<double>(gStyle->GetMarkerSize()), markerSize);
# #     mresidualGraph.SetMarkerSize(markerSize);
# #   }
# # };
