#!/usr/bin/python3

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
