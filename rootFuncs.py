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

def getExpGaussExp_poln_Tf1(n, rangeMin, rangeMax, name=""):
    func = ROOT.TF1(name, mathFuncs.expGausExp_poln, rangeMin, rangeMax, 5 + n + 1)

    # ExpGaussExp params
    func.SetParName(0, "Norm")
    func.SetParName(1, "Peak")
    func.SetParName(2, "Sigma")
    func.SetParName(3, "Tail_low")
    func.SetParName(4, "Tail_high")

    # set poln parameters
    for i in range(n + 1):
        func.SetParName(5 + i, f'p{i}')

    return func

def isFitValid(tf1, fitResult):
    if not fitResult.IsValid():
        print(f'FitResult IsValid == {fitResult.IsValid()}')
        print(fitResult.Status())
        return False

    if ROOT.gMinuit.fStatus != 0:
        print(f'ROOT.gMinuit.fStatus = {ROOT.gMinuit.fStatus}')
        return False

#    if ROOT.gMinuit.fCstatu.rstrip() != 'CONVERGED':
#        print(f'ROOT.gMinuit.fCstatu = {ROOT.gMinuit.fCstatu}')
#        return False

#    for iPar in range( fitResult.NPar() ):
    
    return True

