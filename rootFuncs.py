#!/usr/bin/python3

import ROOT

import mathFuncs


def getExpGaussExpTf1(rangeMin, rangeMax, name=""):
    func = ROOT.TF1(name, mathFuncs.expGaussExpForTf1, rangeMin, rangeMax, 5)
    func.SetParName(0, "Norm")
    func.SetParName(1, "Peak")
    func.SetParName(2, "Sigma")
    func.SetParName(3, "Tail_low")
    func.SetParName(4, "Tail_high")

    return func


def expGausExp_pol0(x, params):
    return params[5] + params[0] * mathFuncs.expGaussExp(x[0], params[1], params[2], params[3], params[4])


def getExpGaussExp_pol0_Tf1(rangeMin, rangeMax, name=""):
    func = ROOT.TF1(name, expGausExp_pol0, rangeMin, rangeMax, 6)
    func.SetParName(0, "Norm")
    func.SetParName(1, "Peak")
    func.SetParName(2, "Sigma")
    func.SetParName(3, "Tail_low")
    func.SetParName(4, "Tail_high")
    func.SetParName(5, "p0")

    return func


def expGausExp_pol1(x, params):
    return params[5] + params[6] * x[0] + params[0] * mathFuncs.expGaussExp(x[0], params[1], params[2], params[3],
                                                                            params[4])


def getExpGaussExp_pol1_Tf1(rangeMin, rangeMax, name=""):
    func = ROOT.TF1(name, expGausExp_pol1, rangeMin, rangeMax, 7)
    func.SetParName(0, "Norm")
    func.SetParName(1, "Peak")
    func.SetParName(2, "Sigma")
    func.SetParName(3, "Tail_low")
    func.SetParName(4, "Tail_high")
    func.SetParName(5, "p0")
    func.SetParName(6, "p1")

    return func


def getExpGaussExp_pol2_Tf1(rangeMin, rangeMax, name=""):
    func = ROOT.TF1(name, mathFuncs.expGausExp_pol2, rangeMin, rangeMax, 8)
    func.SetParName(0, "Norm")
    func.SetParName(1, "Peak")
    func.SetParName(2, "Sigma")
    func.SetParName(3, "Tail_low")
    func.SetParName(4, "Tail_high")
    func.SetParName(5, "p0")
    func.SetParName(6, "p1")
    func.SetParName(7, "p2")

    return func
