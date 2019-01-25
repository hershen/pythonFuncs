import numpy as np
import pandas as pd
import uncertainties
from uncertainties import umath, unumpy

import generalFuncs
import mathFuncs
import alpFuncs


def addMinEcm(df):
    df['minEcm'] = df.loc[:,
                   ('gamma1_energyCM', 'gamma2_energyCM', 'gammaRecoil_energyCM')].min(axis=1)


def addMinE12cm(df):
    df['minE12cm'] = df.loc[:,
                     ('gamma1_energyCM', 'gamma2_energyCM')].min(axis=1)


def theta(x, y, z):
    return np.arctan2(np.sqrt(x ** 2 + y ** 2), z)


def phi(x, y, z):
    return np.arctan2(y, z);


def addPhiCM_deg(df):
    df['gamma1_phiCM_deg'] = phi(df.gamma1_pxCM, df.gamma1_pyCM, df.gamma1_pzCM) * 180 / np.pi
    df['gamma2_phiCM_deg'] = phi(df.gamma2_pxCM, df.gamma2_pyCM, df.gamma2_pzCM) * 180 / np.pi
    df['gammaRecoil_phiCM_deg'] = phi(df.gammaRecoil_pxCM, df.gammaRecoil_pyCM, df.gammaRecoil_pzCM) * 180 / np.pi

def addThetaCM_deg(df):
    df['gamma1_thetaCM_deg'] = theta(df.gamma1_pxCM, df.gamma1_pyCM, df.gamma1_pzCM)
    df['gamma2_thetaCM_deg'] = theta(df.gamma2_pxCM, df.gamma2_pyCM, df.gamma2_pzCM)
    df['gammaRecoil_thetaCM_deg'] = theta(
        df.gammaRecoil_pxCM, df.gammaRecoil_pyCM, df.gammaRecoil_pzCM)

def addThetaLab(df):
    df['gamma1_theta'] = theta(df.gamma1_px, df.gamma1_py, df.gamma1_pz)
    df['gamma2_theta'] = theta(df.gamma2_px, df.gamma2_py, df.gamma2_pz)
    df['gammaRecoil_theta'] = theta(
        df.gammaRecoil_px, df.gammaRecoil_py, df.gammaRecoil_pz)


def addMinTheta(df):
    df['minTheta_deg'] = df.loc[:,
                         ('gamma1_theta', 'gamma2_theta', 'gammaRecoil_theta')].min(axis=1) * 180. / np.pi


def addMaxTheta(df):
    df['maxTheta_deg'] = df.loc[:,
                         ('gamma1_theta', 'gamma2_theta', 'gammaRecoil_theta')].max(axis=1) * 180. / np.pi


def angleBetweenVectors_deg(v1_x, v1_y, v1_z, v2_x, v2_y, v2_z):
    return mathFuncs.angleBetween(np.transpose([v1_x, v1_y, v1_z]), np.transpose([v2_x, v2_y, v2_z])) * 180 / np.pi


def addAngleCM(df):
    df['angle12CM_deg'] = angleBetweenVectors_deg(
        df.gamma1_pxCM, df.gamma1_pyCM, df.gamma1_pzCM, df.gamma2_pxCM, df.gamma2_pyCM, df.gamma2_pzCM)
    df['angle1RecoilCM_deg'] = angleBetweenVectors_deg(
        df.gamma1_pxCM, df.gamma1_pyCM, df.gamma1_pzCM, df.gammaRecoil_pxCM, df.gammaRecoil_pyCM, df.gammaRecoil_pzCM)
    df['angle2RecoilCM_deg'] = angleBetweenVectors_deg(
        df.gamma2_pxCM, df.gamma2_pyCM, df.gamma2_pzCM, df.gammaRecoil_pxCM, df.gammaRecoil_pyCM, df.gammaRecoil_pzCM)


def addMaxAngleCM(df):
    df['maxAngleCM_deg'] = df.loc[:,
                           ('angle12CM_deg', 'angle1RecoilCM_deg', 'angle2RecoilCM_deg')].max(axis=1)


def addExtraColumns(df):
    functionsThatAdd = [addMinEcm, addMinE12cm, addThetaLab,
                        addMinTheta, addMaxTheta, addAngleCM, addMaxAngleCM]
    for func in functionsThatAdd:
        func(df)


def getScaleFactorToTotalLumi(sample, scaleFor):
    if isinstance(scaleFor, int) or isinstance(scaleFor, float):
        return scaleFor
    elif scaleFor == 'data':
        return 20  # 1/0.05
    elif scaleFor == 'SP1074':
        keyForScaleFactor = sample.replace('Run', 'Y').lstrip('/')
        keyForScaleFactor = keyForScaleFactor[:keyForScaleFactor.find('_alpMass')]
        return alpFuncs.SCALE_MC_TO_DATA[keyForScaleFactor]
    else:
        raise ValueError(f'{scaleFor} not supported yet')


def loadDfsFromHdf(filename, scaleFor):
    dfs = {}
    allGroups = generalFuncs.getHdfGroups(filename)

    groups1_6 = [group for group in allGroups if
                 int(group[4]) in np.arange(1, 7) and ('S' not in group or '4S' in group)]
    dfs1_6 = []
    for group in groups1_6:
        df = pd.read_hdf(filename, group)
        addExtraColumns(df)
        df['scaleforTotalLumi'] = getScaleFactorToTotalLumi(group, scaleFor)
        dfs1_6.append(df)
    dfs['1-6'] = pd.concat(dfs1_6)

    groups7 = [group for group in allGroups if group[4:6] in ['2S', '3S']]

    dfs7 = []
    for group in groups7:
        df = pd.read_hdf(filename, group)
        addExtraColumns(df)
        df['scaleforTotalLumi'] = getScaleFactorToTotalLumi(group, scaleFor)
        dfs7.append(df)
    dfs['7'] = pd.concat(dfs7)
    return dfs


def getGroupsForRun_Mass(Run, alpMass, filename):
    groupsForMass = [group for group in generalFuncs.getHdfGroups(filename) if str(alpMass) in group]

    if Run == '1-6':
        return [group for group in groupsForMass if 'S' not in group or '4S' in group]
    elif Run == '7':
        return [group for group in groupsForMass if 'S' in group and 'Run4' not in group]

    raise ValueError(f'Run {Run} not supported')


def getDf(Run, alpMass, filename, scaleFor):
    groups = getGroupsForRun_Mass(Run, alpMass, filename)

    dfs = [pd.read_hdf(filename, group) for group in groups]

    # Add scale to total lumi factor
    for i, group in enumerate(groups):
        try:
            dfs[i].loc[:, 'scaleforTotalLumi'] = getScaleFactorToTotalLumi(group, scaleFor)
        except ValueError as e:
            if str(e) != 'cannot set a frame with no defined index and a scalar':
                raise
            if group != '/Run2S_alpMass10.0':
                raise

    df = pd.concat(dfs, sort=False)

    addExtraColumns(df)
    return df


def splitFieldMinMax(string):
    return string[:-3], string[-3:]


def dictToFilter(df, **args):
    filt = np.ones(len(df), dtype=bool)
    for key, value in args.items():
        variable, minMax = splitFieldMinMax(key)
        if minMax == 'Max':
            filt = np.logical_and(filt, df[variable] < value)
        elif minMax == 'Min':
            filt = np.logical_and(filt, df[variable] > value)
        else:
            raise ValueError(f'Field {key} should end with Min or Max')
    return filt


def count(df, **args):
    return df[dictToFilter(df, **args)].scaleforTotalLumi.sum()


def calcSensitivity(S, B, punziFactor):
    if B == 0:
        return -1
    return uncertainties.ufloat(S, umath.sqrt(S)) / (punziFactor + umath.sqrt(uncertainties.ufloat(B, np.sqrt(B))))


def calcSensitivityFromDfs(signalDf, backgroundDf, cutDict, punziFactor):
    S = count(signalDf, **cutDict)
    B = count(backgroundDf, **cutDict)

    return calcSensitivity(S, B, punziFactor)


def filterDf(df, cutDict):
    return df[dictToFilter(df, **cutDict)]
