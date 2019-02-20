#!/usr/bin/python3
import myPlotting
import numpy as np
import ROOT
import matplotlib.pyplot as plt
import os
import pickle as pl
import pytest


def test_getHistTops_BinEdges_FromAxes():
    # Test that hist without histtype='step' raises error
    with pytest.raises(AssertionError):
        _, ax = plt.subplots()
        plt.hist([1, 2, 3])
        myPlotting.getHistTops_BinEdges_FromAxes(ax)

    _, ax = plt.subplots()

    # intiger bins
    plt.hist([1, 1, 1, 2, 3, 4], bins=3, histtype='step')
    tops, bins = myPlotting.getHistTops_BinEdges_FromAxes(ax)
    np.testing.assert_array_equal(tops, [3, 1, 2])
    np.testing.assert_array_equal(bins, [1, 2, 3, 4])

    ax.clear()

    # float bins
    plt.hist([1, 1, 1, 2, 3, 4], bins=4, range=[0.5, 4.5], histtype='step')
    tops, bins = myPlotting.getHistTops_BinEdges_FromAxes(ax)
    np.testing.assert_array_equal(tops, [3, 1, 1, 1])
    np.testing.assert_array_almost_equal(bins, [0.5, 1.5, 2.5, 3.5, 4.5])


def test_getXvalues():
    # test unsupported argument
    with pytest.raises(ValueError):
        myPlotting.getXvalues(int(3))

    # hist
    hist = ROOT.TH1D("h", "", 10, 0, 10)
    np.testing.assert_array_almost_equal(myPlotting.getXvalues(hist),
                                         [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])

    hist = ROOT.TH1D("h1", "", 6, -0.5, 5.5)
    np.testing.assert_array_almost_equal(myPlotting.getXvalues(hist), [0, 1, 2, 3, 4, 5])

    # TGraph
    graph = ROOT.TGraph()
    graph.SetPoint(0, 0, 1)
    np.testing.assert_array_almost_equal(myPlotting.getXvalues(graph), [0.0])
    graph.SetPoint(1, 0, 1)
    np.testing.assert_array_almost_equal(myPlotting.getXvalues(graph), [0.0, 0.0])
    graph.SetPoint(2, 10, 1)
    np.testing.assert_array_almost_equal(myPlotting.getXvalues(graph), [0.0, 0.0, 10])


def test_saveFig(tmpdir):
    # create figure
    fig, _ = plt.subplots()
    plt.plot([0, 1, 2], [2, 4, 6])

    # save png
    myPlotting.saveFig(fig, 'pngFile.png', str(tmpdir))
    # make sure png exists
    assert os.path.isfile(str(tmpdir.join('figureDump/pngFile.png')))

    # save pdf
    myPlotting.saveFig(fig, 'pdfFile.pdf', str(tmpdir))
    # make sure pdf exists
    assert os.path.isfile(str(tmpdir.join('figureDump/pdfFile.pdf')))

    # save pl
    myPlotting.saveFig(fig, 'plFile.pl', str(tmpdir))
    # make sure fig is ok
    fig1 = pl.load(open(tmpdir.join('figureDump/plFile.pl'), 'rb'))
    assert fig1.dpi == 300

    # Save all file types
    tmpdir = tmpdir.mkdir('allTypes')
    myPlotting.saveFig(fig, 'allTypesFile', str(tmpdir))
    assert os.path.isfile(str(tmpdir.join('figureDump/allTypesFile.png')))
    assert os.path.isfile(str(tmpdir.join('figureDump/allTypesFile.pdf')))
    fig2 = pl.load(open(tmpdir.join('figureDump/allTypesFile.pl'), 'rb'))
    assert fig2.dpi == 300


def test_getPullGraph():
    # Empty input
    assert myPlotting.getPullGraph([], []).GetN() == 0

    x = [1, 2, 3]
    y = [4.5, 5.6, 6.5]
    graph = myPlotting.getPullGraph(x, y)

    # Check minimum/maximum
    assert graph.GetMaximum() == 7
    assert graph.GetMinimum() == -7

    # check num divisions
    assert graph.GetYaxis().GetNdivisions() == -7

    # check graph x values are as expected
    np.testing.assert_array_almost_equal(myPlotting.getXvalues(graph), x)

    # check graph y values are as expected
    graphY = [y for y in graph.GetY()]
    np.testing.assert_array_almost_equal(graphY, y)


def test_calcPulls_fromRootObj():
    # model function
    func = ROOT.TF1("func", "x*x", -2, 2)

    # check unsuported
    with pytest.raises(ValueError):
        myPlotting.calcPulls_fromRootObj(3, func)

    xs = [-2, -1., 0, 1, 2]
    ys = [5, 1, 1, 4, 3]
    errs = [0, 1, 0.5, 0.5, 5]
    expectedPulls = [np.inf, 0, 2, 6, -0.2]

    ##################################
    # TGraph
    ##################################

    # Create TGraph
    graph = ROOT.TGraphErrors()
    for (i, x), y, err in zip(enumerate(xs), ys, errs):
        graph.SetPoint(i, x, y)
        graph.SetPointError(i, 0, err)

    # calculate pulls
    # warning due to division by zero
    with pytest.warns(RuntimeWarning):
        pullsGraph = myPlotting.calcPulls_fromRootObj(graph, func)

    # test pulls
    np.testing.assert_array_almost_equal(pullsGraph, expectedPulls)

    ##################################
    # TH1D
    ##################################
    # Create TH1D
    hist = ROOT.TH1D("hist", "", 5, -2.5, 2.5)
    for (i, x), y, err in zip(enumerate(xs), ys, errs):
        hist.SetBinContent(i + 1, y)
        hist.SetBinError(i + 1, err)

    # calculate pulls
    # warning due to division by zero
    with pytest.warns(RuntimeWarning):
        pullsHist = myPlotting.calcPulls_fromRootObj(hist, func)

    # test pulls
    np.testing.assert_array_almost_equal(pullsHist, expectedPulls)


def test_PullCanvas():
    # Set batch mode so canvas aren't really drawn
    ROOT.gROOT.SetBatch(ROOT.kTRUE)

    # Can't create PullCanvas with nov valid function
    with pytest.raises(AssertionError):
        pullCanvas = myPlotting.PullCanvas(ROOT.TCanvas(), ROOT.TGraph(), ROOT.TF1())

    # Create objects to create PullCanvas
    func = ROOT.TF1("func", "x*x", -2, 2)
    xs = [-2, -1., 0, 1, 2]
    ys = [5, 1, 1, 4, 3]
    errs = [0, 1, 0.5, 0.5, 5]
    expectedPulls = [np.inf, 0, 2, 6, -0.2]

    # Create TGraphErrors
    graph = ROOT.TGraphErrors()
    for (i, x), y, err in zip(enumerate(xs), ys, errs):
        graph.SetPoint(i, x, y)
        graph.SetPointError(i, 0, err)

    canvas = ROOT.TCanvas()
    with pytest.warns(RuntimeWarning):
        pullCanvas = myPlotting.PullCanvas(canvas, graph, func)

    pullCanvas.draw()

    # restore batch mode
    ROOT.gROOT.SetBatch(ROOT.kFALSE)


def test_getFitParameters():
    dictionary = {'alon': 1, 'david': 1.5, 'Ewan': -0.5}
    func = ROOT.TF1("func", "[0] + [1] + [2] + x", -1, 1, 3)
    for i, (name, val) in enumerate(dictionary.items()):
        func.SetParName(i, name)
        func.SetParameter(i, val)

    assert dictionary == myPlotting.getFitParamaeters(func)


def test_latex_float():
    assert myPlotting.latex_float(1) == '1.0'
    assert myPlotting.latex_float(10) == '10.0'
    assert myPlotting.latex_float(100) == '1 $\\times 10^{2}$'
    assert myPlotting.latex_float(100.058) == '1 $\\times 10^{2}$'

    assert myPlotting.latex_float(0.1) == '0.1'
    assert myPlotting.latex_float(0.01) == '0.01'
    assert myPlotting.latex_float(0.001) == '0.001'
    assert myPlotting.latex_float(0.0001) == '0.0001'

    assert myPlotting.latex_float(0.00001) == '1 $\\times 10^{-5}$'
    assert myPlotting.latex_float(0.00001058) == '1.06 $\\times 10^{-5}$'


def test_weightsForArea():
    for i in range(1, 5):
        tops = np.random.poisson(10, size=100)
        bins = np.arange(0, len(tops))
        binwidth = bins[1] - bins[0]

        array = np.random.normal(loc=len(tops) / 2, scale=0.1, size=400)
        area = len(array) * binwidth
        expectedWeights = [area / binwidth / len(array)] * len(array)

        assert np.allclose(expectedWeights, myPlotting.weightsForArea(area, binwidth, array))


def test_getOptimalRange():
    # negative range
    with pytest.raises(ValueError):
        myPlotting.getOptimalRange(lowEdge=123, highEdge=0, nBins=1000, binWidths=[0.0625, 0.125, 0.25, 0.5])

    # bin width above bin width range
    with pytest.raises(ValueError):
        myPlotting.getOptimalRange(lowEdge=0, highEdge=123, nBins=100, binWidths=[0.0625, 0.125, 0.25, 0.5])

    assert np.allclose(
        myPlotting.getOptimalRange(lowEdge=0, highEdge=123, nBins=1000, binWidths=[0.0625, 0.125, 0.25, 0.5]),
        [-1, 124])
