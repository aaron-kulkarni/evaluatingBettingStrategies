import pandas as pd
import numpy as np
import sys
import itertools
sys.path.insert(0, "..")

from utils.utils import *
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


def getSignal():
    df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', index_col=0, header=[0,1])
    signal = pd.DataFrame(df['gameState'])
    signal['signal'] = signal.apply(lambda d: return_signal(d['winner'], d['teamHome'], d['teamAway']), axis=1)
    signal = signal.dropna(axis=0)
    signal['signal'] = signal['signal'].apply(int)
    return signal['signal']

def return_signal(winner,home_team,away_team):
    if home_team == winner:
        return int(1)
    if away_team == winner:
        return int(0)
    else:
        return winner
    
def splitTrainTestYear(index, train_years, test_year):
    '''
    splits data into training data and testing data (data that is tested is last year of input data
    
    '''
    df = pd.DataFrame(index = index)
    df['index'] = df.index.get_level_values(0)
    df['year'] = df.apply(lambda d: getYearFromId(d['index']), axis = 1)
    df.drop('index', axis = 1, inplace = True)
    X_train = df[df['year'].isin(train_years)]
    X_test = df[df['year'] == test_year]
    return X_train.index, X_test.index

def iteratedPCA(df, n, train_index, test_index):
    df_train = df[df.index.isin(train_index)].reindex(train_index)
    df_train_PCA, coeff = performPCA(df_train, n)

    df_test = df[df.index.isin(test_index)].reindex(test_index)
    df_test_PCA = pd.DataFrame()  
    with HiddenPrints():
        for i in range(1, len(df_test.index.get_level_values(0).unique())+1):
            df_test_all = pd.concat([df_train, df_test.iloc[:2*i]], axis = 0)
            df_test_i, coeff = performPCA(df_test_all, n)
            df_test_PCA = pd.concat([df_test_PCA, pd.DataFrame(df_test_i.iloc[-2:])], axis = 0)
    print(df_train_PCA)
    print(df_test_PCA)
    dfPCA = pd.concat([df_train_PCA, df_test_PCA], axis = 0)
    return dfPCA

def selectColOdds(select_x):
    select_x.append('home')
    bettingOdds = pd.read_csv('../data/bettingOddsData/adj_prob_win_ALL.csv', header = [0,1], index_col = 0)
    bettingOdds['homeProbAdj', 'home'], bettingOdds['awayProbAdj', 'home'] = 1, 0
    df = pd.concat([bettingOdds['homeProbAdj'][select_x], bettingOdds['awayProbAdj'][select_x]], axis=0)
    df.reset_index(inplace = True)
    df.set_index(['game_id', 'home'], inplace = True)
    df = df.reindex(sortDateMulti(bettingOdds.index))
    return df

def selectMLVal(select_x):
    mlval = pd.read_csv('../data/eloData/adj_elo_ml.csv', index_col = 0)
    mlval.reset_index(inplace = True)
    mlval['home'] = mlval.apply(lambda d: 1 if d['team'] == d['gameid'][-3:] else 0, axis=1)
    mlval.set_index(['gameid', 'home'], inplace = True)
    mlval = mlval.reindex(sortDateMulti(mlval.index.get_level_values(0)))
    return mlval[select_x]

def selectColElo(select_x):
    elo_data = pd.read_csv('../data/eloData/nba_elo_all.csv', index_col = 0)
    home_elo = elo_data[['elo_prob1', 'raptor_prob1']].rename(columns = lambda x : str(x)[:-1])
    away_elo = elo_data[['elo_prob2', 'raptor_prob2']].rename(columns = lambda x : str(x)[:-1])
    home_elo['home'], away_elo['home'] = 1, 0
    df = pd.concat([home_elo, away_elo], axis=0)
    df.reset_index(inplace = True)
    df.set_index(['index', 'home'], inplace = True)
    df = df.reindex(sortDateMulti(elo_data.index))
    return df[select_x]

def selectColPerMetric(select_x):
    perMetric = pd.read_csv('../data/perMetric/performance_metric_ALL.csv', index_col = 0, header = [0,1])
    perMetric['home', 'home'], perMetric['away', 'home'] = 1, 0
    df = pd.concat([perMetric['home'], perMetric['away']], axis=0)
    df.reset_index(inplace = True)
    df.set_index(['game_id', 'home'], inplace = True)
    df = df.reindex(sortDateMulti(perMetric.index))
    return df[select_x]

def selectColGameData(select_x):
    game_data = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', index_col = 0, header = [0,1])
    game_data['home', 'home'], game_data['away', 'home'] = 1, 0
    df = pd.concat([game_data['home'], game_data['away']], axis=0)
    df.reset_index(inplace = True)
    df.set_index(['game_id', 'home'], inplace = True)
    df = df.reindex(sortDateMulti(game_data.index))
    return df[select_x]

def selectColTeamData(select_x, n):
    team_avg = pd.read_csv('../data/averageTeamData/average_team_stats_per_{}.csv'.format(n), index_col = [0,1])
    return team_avg[select_x]

def testData(dfList, train_years, test_year, drop_na = True):
    X = pd.concat(dfList, axis = 1)
    X['home'] = X.index.get_level_values(1)
    if drop_na == True:
        ret_all = X.dropna(axis = 0).index.get_level_values(0)
        ret = [i for i in ret_all if list(ret_all).count(i) == 2]
        X = X[X.index.isin(sortDateMulti(ret))]
    gameIdList = X.index.get_level_values(0).unique()
    X_train_index, X_test_index = splitTrainTestYear(X.index, train_years, test_year)
    X_train = X[X.index.isin(X_train_index)]
    X_test = X[X.index.isin(X_test_index)]
    Y = get_signal()
    Y_train = Y[Y.index.isin(X_train.index)].reindex(X_train.index)
    Y_test = Y[Y.index.isin(X_test.index)].reindex(X_test.index)
    return X_train, X_test, Y_train

def get_signal():
    signal_home = pd.DataFrame(getSignal())
    signal_away = pd.DataFrame(1-getSignal())
    signal_home['home'], signal_away['home'] = 1, 0
    signal = pd.concat([signal_home, signal_away], axis=0)
    signal.reset_index(inplace = True)
    signal.set_index(['game_id', 'home'], inplace = True)
    signal = signal.reindex(sortDateMulti(getSignal().index))
    return signal


perMetric = selectColPerMetric(['pm_elo_prob1','pm_odd_prob','pm_raptor_prob1','pm_6_elo_prob1','pm_6_odd_prob','pm_6_raptor_prob1'])    
mlval = selectMLVal(['team.elo.booker.lm', 'opp.elo.booker.lm', 'team.elo.booker.combined', 'opp.elo.booker.combined', 'elo.prob', 'predict.prob.booker', 'predict.prob.combined', 'elo.court30.prob', 'raptor.court30.prob', 'booker_odds.Pinnacle'])
bettingOddsAll = selectColOdds(['1xBet (%)', 'Marathonbet (%)', 'Pinnacle (%)', 'Unibet (%)', 'William Hill (%)'])
elo = selectColElo(['elo_prob', 'raptor_prob'])
gameData = selectColGameData(['streak', 'numberOfGamesPlayed', 'daysSinceLastGame'])
#tr_in, te_in = splitTrainTestYear(bettingOddsAll.index, np.arange(2015,2022), 2022)
#bettingOddsPCA = iteratedPCA(bettingOddsAll, 2, tr_in, te_in)
#bettingOddsPCA, coeff = performPCA(bettingOddsAll, 2)
teamData = selectColTeamData(['3P%', 'Drtg', 'PTS', 'TOV%', 'eFG%'], 5)
X_train, X_test, Y_train = testData([bettingOddsAll, elo, perMetric, mlval, gameData, teamData], [2022], 2023, True)
clf = XGBClassifier(learning_rate = 0.02, max_depth = 6, min_child_weight = 6, n_estimators = 150)

def xgboost(clf, X_train, Y_train, X_test):
    model = clf.fit(X_train, Y_train)
    name = 'XGBOOST'
    calibrated_clf = CalibratedClassifierCV(clf, cv = 5)
    calibrated_clf.fit(X_train, Y_train)

    Y_pred_prob_adj = calibrated_clf.predict_proba(X_test)[:, 1]
    Y_train_pred_adj = calibrated_clf.predict_proba(X_train)[:, 1]
    Y_train_pred = Y_train_pred_adj/(np.repeat(Y_train_pred_adj[0::2] + Y_train_pred_adj[1::2], 2))
    Y_pred_prob = Y_pred_prob_adj/(np.repeat(Y_pred_prob_adj[0::2] + Y_pred_prob_adj[1::2], 2))
    
    Y_pred = [1 if p > 0.5 else 0 for p in Y_pred_prob]
    Y_train_pred = [1 if p > 0.5 else 0 for p in Y_train_pred]

    return Y_pred_prob


def findReturns(df, x_columns):
    retHome, retAway = findProportionGained(x_columns)
    retHome = retHome[retHome.index.isin(df.index)].rename('retHome', inplace = True)
    retAway = retAway[retAway.index.isin(df.index)].rename('retAway', inplace = True) 
    df = pd.concat([df, retHome, retAway], axis =1)
    df = df.reindex(sortDate(df.index))
    return df

df = pd.DataFrame(index = X_test.index, columns = ['Y_prob'], data = Y_pred_prob)
df.reset_index(inplace = True)
df = df[::2]
df.drop(['level_1'], axis=1, inplace=True)
df.set_index('level_0', inplace = True)
df.index.name = 'game_id'

x_columns = ['bet365_return', 'William Hill_return', 'Pinnacle_return', 'Coolbet_return', 'Unibet_return', 'Marathonbet_return', 'Unibet_return']

df = findReturns(df, x_columns)

df['per_bet'] = df.apply(lambda d: kellyBet(d['Y_prob'], 0.2, d['retHome'], d['retAway'], [0,100])[0], axis=1)
df['home'] = df.apply(lambda d: kellyBet(d['Y_prob'], 0.2, d['retHome'], d['retAway'], [0,100])[1], axis=1)
