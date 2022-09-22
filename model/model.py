import pandas as pd
import numpy as np
import sys
import itertools
sys.path.insert(0, "..")

from utils.utils import *
from dataProcessing.TeamPerformance import *
from dataProcessing.PCA import *
from kelly import *

import matplotlib.pyplot as plt
import ray 
import multiprocessing
import random
import statistics

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


# Assign # of cpus to work on process based on each computers total cpu count
cpuCount = multiprocessing.cpu_count()
if (cpuCount == 4):
    ray.init(num_cpus=2)
elif (cpuCount > 4 and cpuCount < 8):
    ray.init(num_cpus=4)
else:
    ray.init(num_cpus=6)

def selectColOdds(select_x):
    bettingOdds = pd.read_csv('../data/bettingOddsData/adj_prob_home_win_ALL.csv', index_col = 0)
    return bettingOdds[select_x]

# SELECTED HIGH PERFORMING BETTING BOOKMAKERS
bettingOddsAll = selectColOdds(['1xBet (%)', 'Marathonbet (%)', 'Pinnacle (%)', 'Unibet (%)', 'William Hill (%)', 'bet-at-home (%)', 'bet365 (%)', 'bwin (%)'])
bettingOdds = selectColOdds(['Pinnacle (%)'])

bettingOddsPCA, coeff = performPCA(bettingOddsAll, 2)

def selectColElo(select_x):
    eloData = pd.read_csv('../data/eloData/nba_elo_all.csv', index_col = 0)
    return eloData[select_x]
# columns: season, neutral, team1, team2, elo1_pre, elo2_pre, elo_prob1, elo_prob2, elo1_post, elo2_post, carm-elo1_pre, carm-elo2_pre, carm-elo_prob1, carm-elo_prob2, carm-elo1_post, carm-elo2_post, raptor1_pre, raptor2_pre, raptor_prob1, raptor_prob2
#elo = selectColElo(['elo_prob1', 'raptor_prob1', 'elo1_pre', 'elo2_pre', 'raptor1_pre', 'raptor2_pre'])
elo = selectColElo(['elo_prob1', 'raptor_prob1'])

def selectColPerMetric(select_x):
    perMetric = pd.read_csv('../data/perMetric/performance_metric_all.csv', index_col = 0)
    return perMetric[select_x]

#perMetric = selectColPerMetric(['perMetricAway', 'perMetricHome', 'perMetricEloAway','perMetricEloHome', 'perMetricEloNAway', 'perMetricEloNHome','perMetricNAway', 'perMetricNHome', 'perMetricRaptorAway','perMetricRaptorHome', 'perMetricRaptorNAway', 'perMetricRaptorNHome'])

perMetric = selectColPerMetric(['perMetricHome', 'perMetricAway', 'perMetricEloAway','perMetricEloHome', 'perMetricEloNAway', 'perMetricEloNHome', 'perMetricRaptorAway','perMetricRaptorHome', 'perMetricRaptorNAway', 'perMetricRaptorNHome'])

#perMetric = selectColPerMetric(['perMetricHome', 'perMetricAway'])

def getDFAll(dfList, years, dropNA = True):
    df_all = pd.concat(dfList, axis = 1, join = 'inner')
    df_all.reset_index(inplace = True)
    df_all['year'] = df_all.apply(lambda d: getYearFromId(d['index']), axis = 1)
    df_all.set_index('index', inplace = True)
    df_all = df_all[df_all['year'].isin(years)]
    df_all.drop('year', axis = 1, inplace = True)
    if dropNA == True:
        df_all.dropna(axis = 0, inplace = True)
    return df_all

def splitTrainTestYear(X, Y, year):
    '''
    splits data into training data and testing data (data that is tested is last year of input data
    
    '''

    X.reset_index(inplace = True)
    X['year'] = X.apply(lambda d: getYearFromId(d['index']), axis = 1)
    X.set_index('index', inplace = True)
    X_train = X[X['year'] != year]
    X_test = X[X['year'] == year]
    X_train.drop('year', axis = 1, inplace = True)
    X_test.drop('year', axis = 1, inplace = True)
    

    Y_train = Y[Y.index.isin(X_train.index)].reindex(X_train.index)
    Y_test = Y[Y.index.isin(X_test.index)].reindex(X_test.index)
    return X_train, X_test, Y_train, Y_test

def splitTrainTest(X, Y, p, state, shuffle = True):
    '''
    splits data into training data and testing data (data that is tested is the last p (where p is expressed as a decimal) of input data
    
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 1-p, test_size = p, random_state = state, shuffle = shuffle)
    
    return X_train, X_test, Y_train, Y_test 

    
years = list(np.arange(2015, 2023))
df_all = getDFAll([bettingOddsPCA, elo, perMetric], years, True)

X = df_all
Y = getSignal().reindex(X.index)

X_train, X_test, Y_train, Y_test = splitTrainTestYear(X, Y, 2022)
#X_train, X_test, Y_train, Y_test = splitTrainTest(X, Y, 0.2, 2022, True)

# PARAMATER TUNING
def findParamsXGB(X_train, Y_train):
    param_grid = {
        "n_estimators" : [50, 100, 150],
        "max_depth" : [1, 3, 5, 7],
        "learning_rate" : [0.005, 0.01, 0.02],
        "min_child_weight" : [4, 5, 6]
    }

    grid = GridSearchCV(XGBClassifier(), param_grid, refit = True, verbose = 3)
    grid.fit(X_train, Y_train)
    print(grid.best_params_)
    print(grid.best_estimator_)
    return grid.best_estimator_

#clf = findParamsXGB(X_train, Y_train)

clf = XGBClassifier(learning_rate = 0.02, max_depth = 6, n_estimators = 150, min_child_weight = 6)
#clf = XGBClassifier(learning_rate = 0.02, max_depth = 1, min_child_weight = 4, n_estimators = 150)

def xgboost(clf, X_train, Y_train, X_test, Y_test):
    model = clf.fit(X_train, Y_train)
    name = 'XGBOOST'
    calibrated_clf = CalibratedClassifierCV(clf, cv = 5)
    calibrated_clf.fit(X_train, Y_train)

    #calibrated_clf.predict_proba(X_test)
    #Y_pred = clf.predict(X_test)
    #Y_train_pred = clf.predict(X_train)

    Y_pred_prob = calibrated_clf.predict_proba(X_test)[:, 1]
    Y_train_pred = calibrated_clf.predict_proba(X_train)[:, 1]

    Y_pred = [1 if p > 0.5 else 0 for p in Y_pred_prob]
    Y_train_pred = [1 if p > 0.5 else 0 for p in Y_train_pred]

    acc = accuracy_score(Y_test, Y_pred)
    print("\nAccuracy of %s is %s"%(name, acc))
    #print(clf.feature_importances_)
    print(pd.DataFrame(data = list(model.feature_importances_), index = list(X_train.columns), columns = ["score"]).sort_values(by = "score", ascending = False).head(30))
    
    #cm = confusion_matrix(Y_test, Y_pred)/len(Y_pred) 

    print("Test  Accuracy : %.3f" %accuracy_score(Y_test, Y_pred))
    print("Train Accuracy : %.3f" %accuracy_score(Y_train, Y_train_pred))
    return Y_pred_prob

Y_pred_prob = xgboost(clf, X_train, Y_train, X_test, Y_test)
gameStateData = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', header = [0,1], index_col = 0)
gameStateData = gameStateData[gameStateData.index.isin(Y_test.index)]['home']

def getOddBreakdown(Y_pred_prob, Y_test):
    testOdds = bettingOddsAll[bettingOddsAll.index.isin(Y_test.index)]
    testOdds = testOdds.reindex(Y_test.index)
    for col in testOdds.columns:
        odd_preds = [1 if odd > 0.5 else 0 for odd in list(testOdds[col])]
        print("Odd Accuracy of {}".format(col) + " : %.3f"%accuracy_score(Y_test, odd_preds))
    #print("Confusion Matrix of %s is %s"%(name, cm))

    Y_pred_prob = pd.Series(Y_pred_prob, name = 'predProb', index = Y_test.index)
   
    df = pd.concat([Y_test, Y_pred_prob, testOdds['Pinnacle (%)'], gameStateData['numberOfGamesPlayed']],join = 'inner', axis = 1)
    df['num_game_bkt'] = pd.qcut(df['numberOfGamesPlayed'], 10, duplicates = 'drop')
    df['pred_bkt'] = pd.qcut(df['predProb'], 10 , duplicates = 'drop')
    df['odd_bkt'] = pd.qcut(df['Pinnacle (%)'], 10)
    df['stat_pred'] = df.apply(lambda d: 1 if d['predProb'] > 0.5 else 0, axis = 1)
    df['stat_odd'] = df.apply(lambda d: 1 if d['Pinnacle (%)'] > 0.5 else 0 ,axis = 1)
    
    print(df.groupby('num_game_bkt').signal.sum()/df.groupby('num_game_bkt').size(),df.groupby('num_game_bkt').size())
    print(df.groupby('num_game_bkt').stat_pred.sum()/df.groupby('num_game_bkt').size(),df.groupby('num_game_bkt').size())
    print(df.groupby('pred_bkt').signal.sum()/df.groupby('pred_bkt').size(),df.groupby('pred_bkt').size())
    print(df.groupby('odd_bkt').signal.sum()/df.groupby('odd_bkt').size(),df.groupby('pred_bkt').size())
    df = df[df.columns[:2]]
    return df

df = getOddBreakdown(Y_pred_prob, Y_test)

def findReturns(df, x_columns):
    retHome, retAway = findProportionGained(x_columns)
    retHome = retHome[retHome.index.isin(df.index)].rename('retHome', inplace = True)
    retAway = retAway[retAway.index.isin(df.index)].rename('retAway', inplace = True) 
    df = pd.concat([df, retHome, retAway], axis =1)
    df = df.reindex(sortDate(df.index))
    return df

def getKellyBreakdown(df, alpha, x_columns, max_bet, n):
    df = findReturns(df, x_columns)
    df['per_bet'] = df.apply(lambda d: kellyBet(d['predProb'], alpha, d['retHome'], d['retAway'], n)[0], axis = 1)
    df['home'] = df.apply(lambda d: kellyBet(d['predProb'], alpha, d['retHome'], d['retAway'], n)[1], axis = 1)
    df['per_bet'] = df['per_bet'].where(df['per_bet'] <= max_bet, max_bet)
    df['return'] = df.apply(lambda d: 1 + returnBet(d['per_bet'], d['signal'], d['retHome'], d['retAway'], d['home']), axis = 1)

    
    df['adj_return'] = df.apply(lambda d: 1 if d['return'] < 1 else d['return']/(1-d['per_bet']), axis = 1)
    print(df)
    return df 

def Kelly(df, alpha, x_columns, max_bet, n):
    df = getKellyBreakdown(df, alpha, x_columns, max_bet, n)
    index = sortAllDates(df.index)
    
    per_bet = convertReturns(df['per_bet'], index)
    returns = convertReturns(df['return'] - 1, index)
    dictReturns = pd.concat([per_bet, returns], axis = 1).T.to_dict()

    dfAll = pd.DataFrame(findTotal(dictReturns)).T
    
    print(dfAll['total'])
    return dfAll, dfAll['total']

def convertReturns(series, index):
    df1 = pd.DataFrame(series)
    df1['start'] = 1
    df2 = pd.DataFrame(series)
    df2['start'] = 0

    df = pd.concat([df1, df2], axis = 0)
    df.reset_index(inplace = True)
    df.set_index(['index', 'start'], inplace = True)
    df = df.reindex(index)
    return df

def findTotal(dictReturns):
    dictReturns[list(dictReturns)[0]]['pre_total'] = 1
    dictReturns[list(dictReturns)[0]]['total'] = 1 - dictReturns[list(dictReturns)[0]]['per_bet']

    keys = list(dictReturns.keys())
    for k in keys[1:] :
        dictReturns[k]['pre_total'] = dictReturns[keys[keys.index(k) - 1]]['total']
        if k[1] == 1:
            dictReturns[k]['total'] = dictReturns[k]['pre_total'] * (1 - dictReturns[k]['per_bet'])
        if k[1] == 0:
            if dictReturns[k]['return'] == 0:
                dictReturns[k]['total'] = dictReturns[k]['pre_total']
            else:
                dictReturns[k]['total'] = dictReturns[k]['pre_total'] + dictReturns[(k[0], 1)]['pre_total'] * (dictReturns[k]['return'] + dictReturns[k]['per_bet'])
            
    return dictReturns
        
    
x_columns = ['bet365_return', 'William Hill_return', 'Pinnacle_return', 'Coolbet_return', 'Unibet_return', 'Marathonbet_return']

dfAll, returns = Kelly(df, 0.2, x_columns, 1, 0)
#print(cum_returns)

x = np.arange(1, len(returns) + 1)
y = list(returns.array)
plt.plot(x, y, label = 'PERCENTAGE RETURN')
plt.show()

clf = XGBClassifier(learning_rate = 0.02, max_depth = 6, n_estimators = 150, min_child_weight = 6)

def testSeeds(clf, n):
    returnList = []
    maxReturns = []
    rand = random.sample(range(2**32), n)
    for i in range(len(rand)):
        X_train, X_test, Y_train, Y_test = splitTrainTest(X, Y, 0.2, rand[i], True)
        with HiddenPrints():
            Y_pred_prob = xgboost(clf, X_train, Y_train, X_test, Y_test)

            df = getOddBreakdown(Y_pred_prob, Y_test)
            dfAll, returnsAll = Kelly(df, 0.15, x_columns, 0.05, 0.5)
        returnList.append(returnsAll[-1])
        maxReturns.append(max(list(returnsAll)))

    return returnList, maxReturns
    
returnList, maxReturns = testSeeds(clf, 2)

print('median returns: {}'.format(statistics.median(returnList)))
print('median max returns: {}'.format(statistics.median(maxReturns)))
print('average returns: {}'.format(statistics.mean(returnList)))
print('average max returns: {}'.format(statistics.mean(maxReturns)))

print('successful returns: {}'.format((len([1 for i in returnList if i > 1]))/len(returnList)))
wonReturns = [item for item in returnList if item > 1]
lostReturns = [item for item in returnList if item < 1]
print('successful return average: {}'.format(statistics.mean(wonReturns)))
print('unsuccessful return average: {}'.format(statistics.mean(lostReturns)))

def findParamsXGBPost():
    param_grid = {
        "n_estimators" : [50, 100, 150],
        "max_depth" : [1, 3, 5, 7],
        "learning_rate" : [0.005, 0.01, 0.02],
        "min_child_weight" : [4, 5, 6]
    }

    keys, values = zip(*param_grid.items())
    permutations_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_grid

X_train, X_test, Y_train, Y_test = splitTrainTestYear(X, Y, 2022)

def findOptimalParams(): 
    returnParams = []
    permutations_grid = findParamsXGBPost()
    for i in range (0, len(permutations_grid) - 1):
        clf = XGBClassifier(n_estimators = permutations_grid[i]['n_estimators'], max_depth = permutations_grid[i]['max_depth'], learning_rate = permutations_grid[i]['learning_rate'], min_child_weight = permutations_grid[i]['min_child_weight'])
        with HiddenPrints():
            Y_pred_prob = xgboost(clf, X_train, Y_train, X_test, Y_test)
            df = getOddBreakdown(Y_pred_prob, Y_test)
            dfAll, returns = Kelly(df, 0.15, x_columns, 0.05, 0)
            returnParams.append(returns[-1])
    return permutations_grid, returnParams
permutations_grid, returnParams = findOptimalParams()
print(max(returnParams))
print(permutations_grid[returnParams.index(max(returnParams))])

j = returnParams.index(max(returnParams))
clfOpt = XGBClassifier(n_estimators = permutations_grid[j]['n_estimators'], max_depth = permutations_grid[j]['max_depth'], learning_rate = permutations_grid[j]['learning_rate'], min_child_weight = permutations_grid[j]['min_child_weight'])

Y_pred_prob = xgboost(clfOpt)
df = getOddBreakdown(Y_pred_prob, Y_test)

dfAll, returns = Kelly(df, 0.15, x_columns, 0.05, 0)
x = np.arange(1, len(returns) + 1)
y = list(returns.array)
plt.plot(x, y, label = 'PERCENTAGE RETURN')
plt.show()

ray.shutdown()
