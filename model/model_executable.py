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

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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
    #mlval = mlval[mlval.index.isin(getYearIds(2022))]
    #mlval_yr = pd.read_csv('../data/eloData/adj_elo_ml_year.csv', index_col=0)
    #df = pd.concat([mlval, mlval_yr], axis=0).drop_duplicates()
    #df.to_csv('../data/eloData/adj_elo_ml.csv')
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


def testDataNow(dfList, start_train_year, test_games, drop_na = True):
    X = pd.concat(dfList, axis = 1)
    X['home'] = X.index.get_level_values(1)
    if drop_na == True:
        ret_all = X.dropna(axis = 0).index.get_level_values(0)
        ret = [i for i in ret_all if list(ret_all).count(i) == 2]
        X = X[X.index.isin(sortDateMulti(ret))]
    gameIdList = X.index.get_level_values(0).unique()
    X_train = X[X.index.isin(sortAllDates(getPreviousGames()))]
    X_train['game_id'] = X_train.index.get_level_values(0)
    X_train['year'] = X_train.apply(lambda d: getYearFromId(d['game_id']), axis=1)
    X_train = X_train[~X_train['year'].isin(range(2015, start_train_year))]
    X_train.drop(['game_id', 'year'], inplace = True, axis=1)
    try:
        X_test = X[X.index.isin(sortDateMulti(test_games))]
    except:
        X_test = None
        print('Data has not been updated or error in compilation')
    Y = get_signal()
    Y_train = Y[Y.index.isin(X_train.index)].reindex(X_train.index)
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


def check_dataframe_NaN(dfList, gameIdList):
    i = 0
    for df in dfList:
        df = df[df.index.isin(sortDateMulti(gameIdList))]
        if df.empty:
            print('ERROR - {} columns are empty'.format(df.columns))
        if df.isnull().values.any():
            i = i + 1
            col = df.columns[df.isna().any()].tolist()
            print('ERROR - {} columns are NaN'.format(col))
    if i == 0:
        print('ALL ENTRIES ARE FILLED IN')
    return 


perMetric = selectColPerMetric(['pm_elo_prob1','pm_odd_prob','pm_raptor_prob1','pm_6_elo_prob1','pm_6_odd_prob','pm_6_raptor_prob1']) 
mlval = selectMLVal(['team.elo.booker.lm', 'opp.elo.booker.lm', 'team.elo.booker.combined', 'opp.elo.booker.combined', 'elo.prob', 'predict.prob.booker', 'predict.prob.combined', 'elo.court30.prob', 'raptor.court30.prob', 'booker_odds.Pinnacle'])
bettingOddsAll = selectColOdds(['1xBet (%)', 'Marathonbet (%)', 'Pinnacle (%)', 'Unibet (%)', 'William Hill (%)'])
elo = selectColElo(['elo_prob', 'raptor_prob'])
gameData = selectColGameData(['streak', 'numberOfGamesPlayed', 'daysSinceLastGame', 'matchupWins'])
teamData = selectColTeamData(['3P%', 'Drtg', 'Ortg', 'TOV%', 'eFG%'], 5)
check_dataframe_NaN([bettingOddsAll, elo, perMetric, mlval, gameData, teamData], getNextGames())

X_train, X_test, Y_train = testData([bettingOddsAll, elo, perMetric, mlval, gameData, teamData], [2021, 2022], 2023, True)
X_train_, X_test_, Y_train_ = testDataNow([bettingOddsAll, elo, perMetric, mlval, gameData, teamData], 2021, getNextGames(), True)
clf = XGBClassifier(learning_rate = 0.02, max_depth = 6, min_child_weight = 6, n_estimators = 150)
save_training_data(X_test_)

def save_training_data(X_test):
    df = pd.read_csv('../data/testingData/test_data.csv', index_col = [0,1])
    df = pd.concat([df, X_test], axis=0)
    df.to_csv('../data/testingData/test_data.csv')
    return 

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
    print("Train Accuracy : %.3f" %accuracy_score(Y_train, Y_train_pred))

    print(pd.DataFrame(data = list(model.feature_importances_), index = list(X_train.columns), columns = ["score"]).sort_values(by = "score", ascending = False).head(30))

    return Y_pred_prob

def findReturns(df, x_columns):
    retHome, retAway = findProportionGained(x_columns)
    retHome = retHome[retHome.index.isin(df.index)].rename('retHome', inplace = True)
    retAway = retAway[retAway.index.isin(df.index)].rename('retAway', inplace = True) 
    df = pd.concat([df, retHome, retAway], axis =1)
    df = df.reindex(sortDate(df.index))
    return df

def get_firm(home, firm_home, firm_away):
    if type(home) != bool:
        return None
    if home == True:
        return firm_home
    if home == False:
        return firm_away


def get_ret(home, retHome, retAway):
    if type(home) != bool:
        return 0
    if home == True:
        return retHome
    if home == False:
        return retAway

def get_team(home, teamHome, teamAway):
    if type(home) != bool:
        return None
    if home == True:
        return teamHome
    if home == False:
        return teamAway
        

def getDataFrame(Y_pred_prob, x_columns, test_index):
    #teamDict = {v: k for k, v in getTeamDict().items()}
    df = pd.DataFrame(index = test_index, columns = ['Y_prob'], data = Y_pred_prob)[::2]
    df = df.set_index(df.index.get_level_values(0))
    df.index.name = 'game_id'
    odds = pd.read_csv('../data/bettingOddsData/adj_prob_win_ALL.csv', header = [0,1], index_col = 0)
    odds = odds[odds.index.isin(df.index)]
    df['odds_mean'] = odds['homeProbAdj'].mean(axis=1)
    df = findReturns(df, x_columns)
    df['home_bet'], df['away_bet'] = returnBettingFirm(x_columns, df.index)
    df['per_bet'] = df.apply(lambda d: kellyBet(d['Y_prob'], 0.2, d['retHome'], d['retAway'])[0], axis=1)
    df['home'] = df.apply(lambda d: kellyBet(d['Y_prob'], 0.2, d['retHome'], d['retAway'])[1], axis=1)
    df['firm'] = df.apply(lambda d: get_firm(d['home'], d['home_bet'], d['away_bet']), axis=1)
    df['p_return'] = df.apply(lambda d: get_ret(d['home'], d['retHome'], d['retAway']), axis=1)
    df = pd.concat([df, getTeamsAllYears()[getTeamsAllYears().index.isin(df.index)]], axis=1)
    df['team_abbr'] = df.apply(lambda d: get_team(d['home'], d['teamHome'], d['teamAway']), axis=1)
    #df['team'] = df.apply(lambda d: None if type(d['home']) != bool else teamDict[d['team_abbr']], axis=1)
    acc = get_accuracy(df)
    print(acc)
    df.drop(['teamHome', 'teamAway', 'home_bet', 'away_bet', 'retHome', 'retAway', 'home', 'acc'], axis=1, inplace=True)
    return df

def get_acc(home, signal):
    if type(home) != bool:
        return None
    if home == True and signal == 1:
        return 1
    if home == True and signal == 0:
        return 0
    if home == False and signal == 1:
        return 0
    if home == False and signal == 0:
        return 1

def get_accuracy(df):
    df['signal'] = getSignal()[getSignal().index.isin(df.index)]
    df['acc'] = df.apply(lambda d: get_acc(d['home'], d['signal']), axis=1)
    df = df[df['acc'].notna()]
    return df['acc'].sum()/(len(df.index))
    
             
Y_pred_prob = xgboost(clf, X_train, Y_train, X_test)
x_columns = ['bet365_return', 'Unibet_return']
Y_pred_prob_ = xgboost(clf, X_train_, Y_train_, X_test_)
df = getDataFrame(Y_pred_prob, x_columns, X_test.index)
df_ = getDataFrame(Y_pred_prob_, x_columns, X_test_.index)
