import itertools
import os
import glob
import warnings

import numpy as np
import pandas as pd
import uncertainties
from uncertainties import umath, unumpy

import generalFuncs
import mathFuncs
import alpFuncs


def addMinEcm(df):
    df.loc[:, 'minEcm'] = df.loc[:,
                   ('gamma1_energyCM', 'gamma2_energyCM', 'gammaRecoil_energyCM')].min(axis=1)


def addMinE12cm(df):
    df.loc[:, 'minE12cm'] = df.loc[:,
                     ('gamma1_energyCM', 'gamma2_energyCM')].min(axis=1)


def theta(x, y, z):
    return np.arctan2(np.sqrt(x ** 2 + y ** 2), z)


def addPhiCM_deg(df):
    df.loc[:, 'gamma1_phiCM_deg'] = mathFuncs.phi_xyz(df.gamma1_pxCM, df.gamma1_pyCM, df.gamma1_pzCM) * 180 / np.pi
    df.loc[:, 'gamma2_phiCM_deg'] = mathFuncs.phi_xyz(df.gamma2_pxCM, df.gamma2_pyCM, df.gamma2_pzCM) * 180 / np.pi
    df.loc[:, 'gammaRecoil_phiCM_deg'] = mathFuncs.phi_xyz(df.gammaRecoil_pxCM, df.gammaRecoil_pyCM,
                                                    df.gammaRecoil_pzCM) * 180 / np.pi


def addPhiLab_deg(df):
    df.loc[:, 'gamma1_phiLab_deg'] = mathFuncs.phi_xyz(df.gamma1_px, df.gamma1_py, df.gamma1_pz) * 180 / np.pi
    df.loc[:, 'gamma2_phiLab_deg'] = mathFuncs.phi_xyz(df.gamma2_px, df.gamma2_py, df.gamma2_pz) * 180 / np.pi
    df.loc[:, 'gammaRecoil_phiLab_deg'] = mathFuncs.phi_xyz(df.gammaRecoil_px, df.gammaRecoil_py,
                                                     df.gammaRecoil_pz) * 180 / np.pi


def addAcolPhiCM_deg(df):
    df.loc[:, 'acolPhi12CM_deg'] = df.gamma1_phiCM_deg - df.gamma2_phiCM_deg - 180
    df.loc[:, 'acolPhi1recoilCM_deg'] = df.gamma1_phiCM_deg - df.gammaRecoil_phiCM_deg - 180
    df.loc[:, 'acolPhi2recoilCM_deg'] = df.gamma2_phiCM_deg - df.gammaRecoil_phiCM_deg - 180

    # Make sure delta phi is in range [-180, 180]
    for field in ['acolPhi12CM_deg', 'acolPhi1recoilCM_deg', 'acolPhi2recoilCM_deg']:
        df.loc[df[field] < 180, field] += 360
        df.loc[df[field] > 180, field] -= 360


def addMinAbsAcolPhiCM_deg(df):
    df.loc[:, 'minAbsAcolPhiCM_deg'] = df.loc[:,
                                ('acolPhi12CM_deg', 'acolPhi1recoilCM_deg', 'acolPhi2recoilCM_deg')].abs().min(axis=1)


def addThetaCM_deg(df):
    df.loc[:, 'gamma1_thetaCM_deg'] = theta(df.gamma1_pxCM, df.gamma1_pyCM, df.gamma1_pzCM) * 180 / np.pi
    df.loc[:, 'gamma2_thetaCM_deg'] = theta(df.gamma2_pxCM, df.gamma2_pyCM, df.gamma2_pzCM) * 180 / np.pi
    df.loc[:, 'gammaRecoil_thetaCM_deg'] = theta(
        df.gammaRecoil_pxCM, df.gammaRecoil_pyCM, df.gammaRecoil_pzCM) * 180 / np.pi


def addThetaLab_deg(df):
    df.loc[:, 'gamma1_theta_deg'] = theta(df.gamma1_px, df.gamma1_py, df.gamma1_pz) * 180 / np.pi
    df.loc[:, 'gamma2_theta_deg'] = theta(df.gamma2_px, df.gamma2_py, df.gamma2_pz) * 180 / np.pi
    df.loc[:, 'gammaRecoil_theta_deg'] = theta(
        df.gammaRecoil_px, df.gammaRecoil_py, df.gammaRecoil_pz) * 180 / np.pi


def addAbsDeltaThetaLab_deg(df):
    df.loc[:, 'absDeltaThetaLab12_deg'] = np.abs(df.gamma1_theta_deg - df.gamma2_theta_deg)


def addMinTheta_deg(df):
    try:
        df.loc[:, 'minTheta_deg'] = df.loc[:,
                             ('gamma1_theta', 'gamma2_theta', 'gammaRecoil_theta')].min(axis=1) * 180. / np.pi
    except KeyError:
        df.loc[:, 'minTheta_deg'] = df.loc[:,
                             ('gamma1_theta_deg', 'gamma2_theta_deg', 'gammaRecoil_theta_deg')].min(axis=1)


def addMaxTheta(df):
    try:
        df.loc[:, 'maxTheta_deg'] = df.loc[:,
                             ('gamma1_theta', 'gamma2_theta', 'gammaRecoil_theta')].max(axis=1) * 180. / np.pi
    except KeyError:
        df.loc[:, 'maxTheta_deg'] = df.loc[:,
                             ('gamma1_theta_deg', 'gamma2_theta_deg', 'gammaRecoil_theta_deg')].max(axis=1)


def angleBetweenVectors_deg(v1_x, v1_y, v1_z, v2_x, v2_y, v2_z):
    return mathFuncs.angleBetween(np.transpose([v1_x, v1_y, v1_z]), np.transpose([v2_x, v2_y, v2_z])) * 180 / np.pi


def addAngleCM(df):
    df.loc[:, 'angle12CM_deg'] = angleBetweenVectors_deg(
        df.gamma1_pxCM, df.gamma1_pyCM, df.gamma1_pzCM, df.gamma2_pxCM, df.gamma2_pyCM, df.gamma2_pzCM)
    df.loc[:, 'angle1RecoilCM_deg'] = angleBetweenVectors_deg(
        df.gamma1_pxCM, df.gamma1_pyCM, df.gamma1_pzCM, df.gammaRecoil_pxCM, df.gammaRecoil_pyCM, df.gammaRecoil_pzCM)
    df.loc[:, 'angle2RecoilCM_deg'] = angleBetweenVectors_deg(
        df.gamma2_pxCM, df.gamma2_pyCM, df.gamma2_pzCM, df.gammaRecoil_pxCM, df.gammaRecoil_pyCM, df.gammaRecoil_pzCM)


def addMaxAngleCM(df):
    df.loc[:, 'maxAngleCM_deg'] = df.loc[:,
                           ('angle12CM_deg', 'angle1RecoilCM_deg', 'angle2RecoilCM_deg')].max(axis=1)


def addExtraColumns(df):
    functionsThatAdd = [addMinE12cm, addThetaLab_deg,
                        addMinTheta_deg, addMaxTheta, addAngleCM, addMaxAngleCM, addPhiCM_deg, addAcolPhiCM_deg,
                        addMinAbsAcolPhiCM_deg, addAbsDeltaThetaLab_deg]
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
    groupsForMass = [group for group in generalFuncs.getHdfGroups(filename) if group.endswith('Mass' + str(alpMass))]

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
        return uncertainties.ufloat(-np.inf, np.inf)
    return uncertainties.ufloat(S, umath.sqrt(S)) / (punziFactor + umath.sqrt(uncertainties.ufloat(B, np.sqrt(B))))


def calcSensitivityFromDfs(signalDf, backgroundDf, cutDict, punziFactor):
    S = count(signalDf, **cutDict)
    B = count(backgroundDf, **cutDict)

    return calcSensitivity(S, B, punziFactor)


def filterDf(df, cutDict):
    return df[dictToFilter(df, **cutDict)]


def getSP1074MassesForCuts(Run):
    if Run == '1-6':
        return [0.02, 0.2, 0.4, 6.5, 7.0]
    elif Run == '7':
        return [0.02, 0.05, 0.1, 0.135, 0.2, 0.3, 0.4, 0.5, 0.548, 0.6, 0.9, 0.958, 1.0]
    else:
        raise ValueError(f'Run {Run} not supported')


def getBkgFilename(Run, alpMass):
    """
    If 5% of the data had less than 100 background events
    that passed all cuts, use SP1074, otherwise use data.
    """
    if np.isclose(alpMass, getSP1074MassesForCuts(Run)).any():
        return 'massInCountingwindow_SP1074.h5'
    else:
        return 'massInCountingwindow_data5perc.h5'


def getRun(filename):
    filename = os.path.basename(filename)
    return filename[filename.find('Run') + 3: filename.rfind('_')]


def getAlpMass(filename):
    filename = os.path.basename(filename)
    return filename[filename.find('alpMass') + 7: filename.rfind('.')]


def loadDfsFromFolder(folder):
    files = glob.glob(os.path.join(folder, 'csv', '*.csv'))
    result = {}
    for file in files:
        df = pd.read_csv(file)
        try:
            result[getRun(file)][float(getAlpMass(file))] = df
        except KeyError:
            result[getRun(file)] = {}
            result[getRun(file)][float(getAlpMass(file))] = df
    return result


def getDfs(folders):
    dfs = {}
    for folder in folders:
        df = loadDfsFromFolder(folder)
        if len(df) == 0:
            raise ValueError(f'Cannot load from {folder}')
        dfs[os.path.basename(folder)] = df
    return dfs


def getRowsColumn(df, rowVal, column=None):
    result = df[df['Unnamed: 0'] == rowVal]
    if isinstance(column, int):
        columns = result.columns
        result = result[columns[column]]
    elif column:
        result = result[column]
    return result


def getLabel(sample):
    label = sample.replace('data5perc', '5% Data')
    label = label[:label.find('_')]
    if label == 'hybrid':
        if 'uncMass' in sample:
            label = 'Upsilon'
        else:
            label = 'pi0'

    return label

    # Add punzi factor info


#     try:
#         punziFactor = float(sample[sample.find('punzi')+5:])
#         sample = sample.replace(
#             '_s_sqrtB_punzi', f' $S/({punziFactor} + \sqrt{{B}})$')
#         sample = sample[:-3]
#     except:
#         pass

#     sample = sample.replace('_s_sqrtB', r' $S/\sqrt{B}$')
#     return sample

def getMultiDf(dfs):
    arrays = []
    for Run in generalFuncs.getArbitraryDictItem(dfs).keys():
        arrays = arrays + list(itertools.product(dfs.keys(), [Run], sorted(
            generalFuncs.getArbitraryDictItem(dfs)[Run].keys())))

    index = pd.MultiIndex.from_tuples(
        arrays, names=['sample', 'Run', 'alpMass'])

    return pd.DataFrame(index=index).sort_index()


def fillMultiDf(dfs, multiDf, columnsToExtract):
    for sample in dfs:
        for Run in sorted(list(dfs[sample].keys())):
            alpMasses = sorted(dfs[sample][Run].keys())

            for column in columnsToExtract:
                if 'Final' in column:
                    csvColumn = -1
                else:
                    raise ValueError(f'Columns {column} not supported')

                multiDf.loc[(sample, Run, alpMasses), column] = np.array(
                    [getRowsColumn(dfs[sample][Run][alpMass], column.split(' ')[1], csvColumn).tail(1).values
                     for alpMass in alpMasses])

            # Add optimal values
            for variable in generalFuncs.getArbitraryDictItem(
                    generalFuncs.getArbitraryDictItem(generalFuncs.getArbitraryDictItem(dfs))).columns:
                if variable in ['Unnamed: 0', 'Run', 'alpMass', 'beforeCuts']:
                    continue
                try:
                    multiDf.loc[(sample, Run, alpMasses), f'Optimal {variable}'] = np.array(
                        [getRowsColumn(dfs[sample][Run][alpMass], 'optimalValues', variable).tail(1).values
                         for alpMass in alpMasses])
                except KeyError as e:
                    warnings.warn(f'Couldn\'t find key {e}')
                    pass


_field2cppVariables = {'Optimal minE12cmMin': 'const std::vector<double> minE12cm_Run',
                       'Optimal minTheta_degMin': 'const std::vector<double> minTheta_deg_Run',
                       'Optimal maxTheta_degMax': 'const std::vector<double> maxTheta_deg_Run',
                       'Optimal chi2Max': 'const std::vector<double> chi2_Run',
                       'Optimal minAbsAcolPhiCM_degMin': 'const std::vector<double> minAbsAcolPhiCM_degMin_Run',
                       }
_Run2cppRun = {'1-6': '16',
               '7': '7'}


def printOptimalCuts(multiDf):
    for field in [column for column in multiDf.columns if 'Optimal' in column]:
        print('')
        for sample in multiDf.index.levels[0]:
            for Run in multiDf.index.levels[1]:
                values = multiDf.loc[(sample, Run), field].values
                values = [str(x) for x in values]
                print(_field2cppVariables[field] + _Run2cppRun[Run] + ' = {' + ', '.join(values) + '};')


def getOptimalValues(multiDfLT10, multiDfGT10):
    arrays = []
    for Run in ['1-6', '7']:
        arrays = arrays + list(itertools.product([Run], sorted(
            alpFuncs.getAlpMasses(Run))))

    index = pd.MultiIndex.from_tuples(arrays, names=['Run', 'alpMass'])

    df = pd.DataFrame(index=index).sort_index()

    columns = set()
    for tmpDf in [multiDfLT10, multiDfGT10]:
        for column in tmpDf.columns:
            columns.add(column)

    for variable in columns:
        try:
            # Mass <= 10
            sample = multiDfLT10.index.levels[0][0]
            for Run in multiDfLT10.index.levels[1]:
                alpMasses = multiDfLT10.loc[(sample, Run)].index
                alpMasses = alpMasses[alpMasses <= 10]
                df.loc[(Run, alpMasses), variable] = multiDfLT10.loc[(
                                                                         sample, Run, alpMasses), variable].values
        except KeyError:
            pass

        # Mass > 10

        #Don't do delta theta for m>10
        if variable == 'Optimal absDeltaThetaLab12_degMin':
            continue

        sample = multiDfGT10.index.levels[0][0]
        for Run in multiDfGT10.index.levels[1]:
            alpMasses = multiDfGT10.loc[(sample, Run)].index
            alpMasses = alpMasses[alpMasses > 10]
            df.loc[(Run, alpMasses), variable] = multiDfGT10.loc[(
                                                                     sample, Run, alpMasses), variable].values

    return df


def getSignalFilename(alpMass, bkgFilename):
    if alpMass <= 1.0 and 'data' in bkgFilename:
        return 'massInCountingwindow_signal_minE12cm0.05.h5'
    else:
        return 'massInCountingwindow_signal.h5'


def getBkgScaleFor(bkgFilename):
    if 'SP1074' in bkgFilename:
        return 'SP1074'
    elif 'data' in bkgFilename:
        return 20
    else:
        raise ValueError(f'Background filename {bkgFilename} not recognized')


def getNumGeneratedEvents(Run, alpMasses):
    return [alpFuncs.getNumberOfGeneratedSignal(Run, mass) for mass in alpMasses]


def getSmoothedCutDict(Run, alpMass, smoothCutsSplineDf):
    cutDict = {}
    for field in smoothCutsSplineDf.columns.levels[0]:
        if alpMass <= 10 and field == 'minAbsAcolPhiCM_degMin':
            continue
        xVals = smoothCutsSplineDf.loc[Run, (field, 'xVals')]
        yVals = smoothCutsSplineDf.loc[Run, (field, 'yVals')]
        cutDict[field] = np.interp(alpMass, xVals, yVals)
    return cutDict

def getSimple_smoothedMinE12cmCutDict(Run, alpMass, smoothCutsSplineDf):
    cutDict = getSimpleCutDict(Run, alpMass)
    xVals = smoothCutsSplineDf.loc[Run, ('minE12cmMin', 'xVals')]
    yVals = smoothCutsSplineDf.loc[Run, ('minE12cmMin', 'yVals')]
    cutDict['minE12cmMin'] = np.interp(alpMass, xVals, yVals)
    return cutDict

def getSimpleCutDict(Run, alpMass):
    cutDict = {'chi2Max': 100.0,
               'minE12cmMin': 1, 'minTheta_degMin': 22.5}
    if alpMass < 0.6:
        cutDict['absDeltaThetaLab12_degMin'] = 1
    if alpMass > 10:
        cutDict['minAbsAcolPhiCM_degMin'] = 0.5
    return cutDict

def getSmoothCutsSpline(filename = '/home/hershen/PhD/ALPs/analysis/cuts/optimization/smoothCuts/smoothedCuts.hdf'):
    return pd.read_hdf(filename, 'smooth')

def addSimpleCuts(df, Run):
    cutDict = getSimpleCutDict(Run, 3.0)

    #Remove fields which will be handled later
    try:
        for field in ['simple_minAbsAcolPhiCM_degMin', 'absDeltaThetaLab12_degMin']:
            cutDict.pop(field)
    except KeyError:
        pass

    for field in cutDict.keys():
        df['simple_' + field] = cutDict[field]

    df['simple_minAbsAcolPhiCM_degMin'] = -1
    df.loc[df.eta_Mass > 10, 'simple_minAbsAcolPhiCM_degMin'] = getSimpleCutDict(
        Run, 10.3)['minAbsAcolPhiCM_degMin']

    df['simple_absDeltaThetaLab12_degMin'] = -1
    df.loc[df.eta_Mass <= 0.6, 'simple_absDeltaThetaLab12_degMin'] = getSimpleCutDict(
        Run, 0.2)['absDeltaThetaLab12_degMin']

def addSmoothedCuts(df, Run, smoothCutsSplineDf):
    # ignore minAbsAcolPhiCM_degMin if Run 7
    fields = [field for field in smoothCutsSplineDf.columns.levels[0]
              if not np.isnan(smoothCutsSplineDf.loc[Run, (field, 'xVals')]).all()]

    for field in fields:
        xVals = smoothCutsSplineDf.loc[Run, (field, 'xVals')]
        yVals = smoothCutsSplineDf.loc[Run, (field, 'yVals')]
        df['smooth_' + field] = np.interp(df.eta_Mass, xVals, yVals)
    if 'minAbsAcolPhiCM_degMin' in fields:
        df.loc[df.eta_Mass <= 10, 'smooth_minAbsAcolPhiCM_degMin'] = -1

def getFilterOfCuts(df, columns):
    filt = np.ones(len(df), dtype=bool)
    for field in columns:
        variable, minMax = splitFieldMinMax(field)
        variable = variable[variable.find('_')+1:]
        if 'absDeltaThetaLab12_degMin' in field:
            df['failedDeltaThetaCut'] = df[variable] < df[field]
            filt = np.logical_and(filt,
                    df.groupby([df.entryNum.diff().ne(0).cumsum()]).failedDeltaThetaCut.transform('sum').eq(0))
            del df['failedDeltaThetaCut']
        elif minMax == 'Max':
            filt = np.logical_and(filt, df[variable] < df[field])
        elif minMax == 'Min':
            filt = np.logical_and(filt, df[variable] > df[field])
        else:
            raise ValueError(f'Field {field} ends with {minMax}. It should end with Min or Max')
    return filt

def makeCuts(df, columns):
    """
    Take columns in the form of XXX_fieldnameMin/Max and return df that is filtered with them
    :param df:
    :param columns:
    :return:
    """
    return df[getFilterOfCuts(df, columns)]

def makeSmoothCuts(df, Run, smoothCutsSplineDf, thetaMin_deg=22.5):
    df = df[alpFuncs.getTriggered(df)]
    df = df[df.nTracks <=1]
    addExtraColumns(df)
    df = df[df.minE12cm >= 0.7]
    addSmoothedCuts(df, Run, smoothCutsSplineDf)
    filtered = makeCuts(df, ['smooth_minE12cmMin', 'smooth_chi2Max',
                                           'smooth_minAbsAcolPhiCM_degMin', 
                                           'smooth_absDeltaThetaLab12_degMin'])
    filtered = filtered[filtered.minTheta_deg > thetaMin_deg]
    return filtered

def getColumnsForCuts():
    columns = ['entryNum', 'L3OutGammaGamma', 'DigiFGammaGamma',
            'DigiFSingleGamma', ' BGFSingleGammaPair', 'L3OutDch', 'L3OutEmc',
            'nTrakcs', 'chi2', 'eta_Mass', 'gamma1_px', 'gamma1_py',
            'gamma1_pz', 'gamma1_energy', 'gamma2_px', 'gamma2_py',
            'gamma2_pz', 'gamma2_energy', 'gammaRecoil_px', 'gammaRecoil_py',
            'gammaRecoild_pz', 'gammmaRecoil_energy']
