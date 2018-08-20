#!/usr/bin/python3
import pickle as pl
import os
from ROOT import TCanvas, TGraph, kFALSE, TH1
from root_numpy import fill_graph
from numpy import abs, ceil, atleast_1d, asarray
from mathFuncs import calcPulls


def getHistBinEdges_topsFromAxes(axes):
    # list of (x,y) coordinates of each vertex of histograms.
    xy = axes.patches[0].get_xy()

    # Take only vertical lines (contain all information)
    data = xy[::2]

    # bin edges are first column.
    binEdges = data[:, 0]

    # tops can disregard first entry (probably at 0)
    tops = data[1:, 1]

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

    # Create figureDump dir
    if not os.path.isdir('figureDump'):
        os.mkdir('figureDump')

    # If extension included, save only thatfiletype
    if ext:
        if ext == '.pl':
            _saveFigPickled(fig, filename + ext)
        else:
            _saveFigFlat(fig, filename + ext)
        return

    # otherwise, save multiple filetypes
    _saveFigFlat(fig, filename + '.png')
    _saveFigFlat(fig, filename + '.pdf')
    _saveFigPickled(fig, filename + '.pl')
    # pl.dump(fig, open('figureDump/{}'.format(filename), 'wb'))


def getPullGraph(xValues, residuals):
    """

    :param xValues:
    :param residuals:
    :return:
    """

    xValues = asarray(atleast_1d(xValues), dtype=float)
    residuals = asarray(atleast_1d(residuals), dtype=float)

    # sanity
    assert len(xValues) == len(residuals), "xValues size {} != residuals size {}".format(len(xValues), len(residuals))

    residualsGraph = TGraph(len(xValues), xValues, residuals)
    # fill_graph(residualsGraph, re)
    maxResidual = abs(residuals).max()
    maxResidualInteger = ceil(maxResidual)
    residualsGraph.SetMaximum(maxResidualInteger)
    residualsGraph.SetMinimum(-maxResidualInteger)
    residualsGraph.GetYaxis().SetNdivisions(int(maxResidualInteger),
                                            kFALSE)  # false - no optimization - forces current value

    return residualsGraph


def _getXvalues_TH1(hist):
    return [hist.GetBinCenter(iBin) for iBin in range(1, hist.GetNbinsX() + 1)]


def _getXvalues_TGraph(graph):
    return [x for x in graph.GetX()]


def getXvalues(rootObj):
    if isinstance(rootObj, TH1):
        return _getXvalues_TH1(rootObj)
    elif isinstance(rootObj, TGraph):
        return _getXvalues_TGraph(rootObj)

    raise ValueError("{} is not supported".format(type(rootObj)))

class ResidualCanvas:
    _residualGraph
    __init__(canvas, topObject, function):
        assert function.IsValid(), "ResidualCanvas::ResidualCanvas: function isn't valid."

        self.prepareCanvas()

        const auto residuals(myFuncs::calcResiduals(m_topObject, m_function));

        _residualGraph = TGraph(getResidualsGraph(myFuncs::getXvalues(topObject), residuals));

        self.prepareObjects()
#
#   void draw() {
#     m_topPad->cd();
#     m_topObject.Draw("AP");
#     m_function.Draw("Same");
#
#     m_bottomPad->cd();
#     m_residualGraph.Draw("AP");
#   }
#
#   TPad* getTopPad() const { return m_topPad; }
#   TPad* getBottomPad() const { return m_bottomPad; }
#   TGraph getResidualGraph() const { return m_residualGraph; }
#
# private:
#   TCanvas& m_canvas;
#   topType& m_topObject;
#   TF1& m_function;
#   TGraph m_residualGraph;
#   TPad* m_topPad;
#   TPad* m_bottomPad;
#
#   void prepareCanvas() {
#
#     const double bottomPadYpercentage = 0.22;
#     const double bottomTopSeperation = 0.05;
#
#     m_canvas.cd();
#     m_topPad = new TPad((m_canvas.GetName() + std::string("_topPad")).data(), "", 0, bottomPadYpercentage, 1, 1);
#     m_topPad->SetNumber(1);
#     m_topPad->Draw();
#
#     // can't set margins with m_topPad->Set__Margin() for some reason. Have to go through m_canvas.cd(x)...
#     m_canvas.cd(1)->SetBottomMargin(0.01);
#     // Change to canvas before creating second pad
#     m_canvas.cd();
#
#     m_bottomPad = new TPad((m_canvas.GetName() + std::string("_bottomPad")).data(), "", 0, 0, 1, bottomPadYpercentage);
#     m_bottomPad->SetNumber(2);
#
#     m_bottomPad->Draw();
#     m_canvas.cd(2)->SetTopMargin(bottomTopSeperation);
#     m_canvas.cd(2)->SetBottomMargin(gStyle->GetPadBottomMargin() * 1.5);
#     m_bottomPad->SetGridy();
#
#     m_canvas.cd();
#   }
#
#   // Prepare objects for drawing in a prettyResidualGraph
#   void prepareObjects() {
#     m_topObject.GetXaxis()->SetLabelSize(0.0); // Don't draw x labels on top object
#     // m_topObject.GetXaxis()->SetTitleSize(0.0); // Don't draw x title on top object
#
#     // Set axis label and Title size to absolute
#     m_topObject.GetYaxis()->SetLabelFont(43);     // Absolute font size in pixel (precision 3)
#     m_residualGraph.GetXaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)
#     m_residualGraph.GetYaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)
#     // Title
#     m_topObject.GetYaxis()->SetTitleFont(43);     // Absolute font size in pixel (precision 3)
#     m_residualGraph.GetXaxis()->SetTitleFont(43); // Absolute font size in pixel (precision 3)
#     m_residualGraph.GetYaxis()->SetTitleFont(43); // Absolute font size in pixel (precision 3)
#
#     // Set x + y axis label size
#     const double labelSize = std::min(0.03 * m_canvas.cd(1)->GetWh(), 0.03 * m_canvas.cd(1)->GetWw());
#     m_topObject.GetYaxis()->SetLabelSize(labelSize);
#
#     m_residualGraph.GetYaxis()->SetLabelSize(labelSize);
#     // x axis
#     m_residualGraph.GetXaxis()->SetLabelSize(labelSize);
#
#     // Set axis title sizes
#     const double titleSize = std::min(0.03 * m_canvas.cd(1)->GetWh(), 0.03 * m_canvas.cd(1)->GetWw());
#     m_residualGraph.GetXaxis()->SetTitleSize(titleSize);
#     m_residualGraph.GetYaxis()->SetTitleSize(titleSize);
#     m_topObject.GetYaxis()->SetTitleSize(titleSize);
#
#     // Set title offsets
#     m_residualGraph.GetXaxis()->SetTitleOffset(3.75);
#
#     // Set bottom x title
#     m_residualGraph.GetXaxis()->SetTitle(m_topObject.GetXaxis()->GetTitle());
#     // Set y title
#     m_residualGraph.GetYaxis()->SetTitle("Pull (#sigma)");
#
#     // Set residual y axis divisions
#     const auto maxResidual =
#         std::abs(*std::max_element(m_residualGraph.GetY(), m_residualGraph.GetY() + m_residualGraph.GetN() - 1,
#                                    [](const double residual1, const double residual2) { // find max absolute value residual
#                                      return std::abs(residual1) < std::abs(residual2);
#                                    }));
#     m_residualGraph.SetMaximum(std::ceil(maxResidual));
#     m_residualGraph.SetMinimum(-std::ceil(maxResidual));
#     const int maxDivisions = std::min(5., std::ceil(maxResidual));
#     m_residualGraph.GetYaxis()->SetNdivisions(maxDivisions, false); // false - no optimization - forces current value
#
#     // Set marker size
#     double markerSize = myFuncs::linearInterpolate(100, 17500, 1.2, 0.1, m_residualGraph.GetN());
#     // markerSize = std::min(static_cast<double>(gStyle->GetMarkerSize()), markerSize);
#     m_residualGraph.SetMarkerSize(markerSize);
#   }
# };
