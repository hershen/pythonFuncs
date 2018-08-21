#!/usr/bin/python3
import myPlotting
import numpy as np
import ROOT
import matplotlib.pyplot as plt
import os
import pickle as pl


def test_getXvalues():
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
    plt.plot([0,1,2], [2,4,6])

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

    #Save all file types
    tmpdir = tmpdir.mkdir('allTypes')
    myPlotting.saveFig(fig, 'allTypesFile', str(tmpdir))
    assert os.path.isfile(str(tmpdir.join('figureDump/allTypesFile.png')))
    assert os.path.isfile(str(tmpdir.join('figureDump/allTypesFile.pdf')))
    fig2 = pl.load(open(tmpdir.join('figureDump/allTypesFile.pl'), 'rb'))
    assert fig2.dpi == 300


