import sys
sys.path.insert(0, "..")

from utils.utils import *
from dataProcessing.PCA import *
from kelly import *

import itertools
from collections import Counter
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


def init_ray():
    cpuCount = multiprocessing.cpu_count()
    if (cpuCount == 4):
        ray.init(num_cpus=2)
        print('initialized 2 cpus')
    elif (cpuCount > 4 and cpuCount < 8):
        ray.init(num_cpus=4)
        print('initialized 4 cpus')
    else:
        ray.init(num_cpus=6)
        print('initialized 6 cpus')
    return

def get_signal():
    signal_home = pd.DataFrame(getSignal())
    signal_away = pd.DataFrame(1-getSignal())
    signal_home['home'], signal_away['home'] = 1, 0
    signal = pd.concat([signal_home, signal_away], axis=0)
    signal.reset_index(inplace = True)
    signal.set_index(['game_id', 'home'], inplace = True)
    signal = signal.reindex(sortDateMulti(getSignal().index))
    return signal

class select_attributes:
    def __init__(self, n):
        self.game_df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', index_col = 0, header = [0,1])
        self.odds_df = pd.read_csv('../data/bettingOddsData/adj_prob_win_ALL.csv', header = [0,1], index_col = 0)
        self.per_metric_df = pd.read_csv('../data/perMetric/performance_metric_ALL.csv', index_col = 0, header = [0,1])
        self.team_avg_df = pd.read_csv('../data/averageTeamData/average_team_stats_per_{}.csv'.format(n), index_col = [0,1])
        self.mlval_df = pd.read_csv('../data/eloData/adj_elo_ml.csv', index_col = 0)
        self.elo_df = pd.read_csv('../data/eloData/nba_elo_all.csv', index_col = 0)

    def select_col_odds(self, select_x):
         select_x.append('home')
         odds_df = self.odds_df 
         odds_df['homeProbAdj', 'home'], odds_df['awayProbAdj', 'home'] = 1, 0
         df = pd.concat([odds_df['homeProbAdj'][select_x], odds_df['awayProbAdj'][select_x]], axis=0)
         df.reset_index(inplace = True)
         df.set_index(['game_id', 'home'], inplace = True)
         df = df.reindex(sortDateMulti(odds_df.index))
         return df

    def select_mlval(self, select_x):
        mlval_df = self.mlval_df
        mlval_df.reset_index(inplace = True)
        mlval_df['home'] = mlval_df.apply(lambda d: 1 if d['team'] == d['gameid'][-3:] else 0, axis=1)
        mlval_df.set_index(['gameid', 'home'], inplace = True)
        mlval_df = mlval_df.reindex(sortDateMulti(mlval_df.index.get_level_values(0)))
        return mlval_df[select_x]

    def select_elo(self, select_x):
        elo_df = self.elo_df 
        home_elo = elo_df[['elo_prob1', 'raptor_prob1']].rename(columns = lambda x : str(x)[:-1])
        away_elo = elo_df[['elo_prob2', 'raptor_prob2']].rename(columns = lambda x : str(x)[:-1])
        home_elo['home'], away_elo['home'] = 1, 0
        df = pd.concat([home_elo, away_elo], axis=0)
        df.reset_index(inplace = True)
        df.set_index(['index', 'home'], inplace = True)
        df = df.reindex(sortDateMulti(elo_df.index))
        return df[select_x]

    def select_col_per(self, select_x):
        per_metric = self.per_metric_df
        per_metric['home', 'home'], per_metric['away', 'home'] = 1, 0
        df = pd.concat([per_metric['home'], per_metric['away']], axis=0)
        df.reset_index(inplace = True)
        df.set_index(['game_id', 'home'], inplace = True)
        df = df.reindex(sortDateMulti(per_metric.index))
        return df[select_x]

    def select_col_game(self, select_x):
        game_df = self.game_df
        game_df['home', 'home'], game_df['away', 'home'] = 1, 0
        df = pd.concat([game_df['home'], game_df['away']], axis=0)
        df.reset_index(inplace = True)
        df.set_index(['game_id', 'home'], inplace = True)
        df = df.reindex(sortDateMulti(game_df.index))
        return df[select_x]

    def select_col_team(self, select_x):
        team_avg_df = self.team_avg_df
        return team_avg_df[select_x]

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

def split_train_test_year(index, train_years, test_year):
    df = pd.DataFrame(index = index)
    df['index'] = df.index.get_level_values(0)
    df['year'] = df.apply(lambda d: getYearFromId(d['index']), axis = 1)
    df.drop('index', axis = 1, inplace = True)
    X_train = df[df['year'].isin(train_years)]
    X_test = df[df['year'] == test_year]
    return X_train.index, X_test.index

def split_data(data_list, train_years, test_year, drop_na = True):
    Y = get_signal()
    X = pd.concat(data_list, axis = 1)
    X['home'] = X.index.get_level_values(1)
    if drop_na == True:
        ret_all = X.dropna(axis = 0).index.get_level_values(0)
        counter = Counter(list(ret_all))
        ret = [i for i in ret_all if counter[i] == 2]
        X = X[X.index.isin(sortDateMulti(ret))]
    X_train_index, X_test_index = split_train_test_year(X.index, train_years, test_year)
    X_train, X_test = X[X.index.isin(X_train_index)], X[X.index.isin(X_test_index)]
    Y_train, Y_test = Y[Y.index.isin(X_train.index)].reindex(X_train.index), Y[Y.index.isin(X_test.index)].reindex(X_test.index)
    return X_train, X_test, Y_train, Y_test

def split_data_test_games(data_list, train_window, test_games, size_cons = True, drop_na = True):
    X = pd.concat(data_list, axis = 1)
    X['home'] = X.index.get_level_values(1)
    if drop_na == True:
        ret_all = X.dropna(axis = 0).index.get_level_values(0)
        counter = Counter(list(ret_all))
        ret = [i for i in ret_all if counter[i] == 2]
        X = X[X.index.isin(sortDateMulti(ret))]
    if len(set([getYearFromId(i) for i in test_games])) == 1:
        test_year = getYearFromId(test_games[0])
    else:
        return print('test games are not in same year')
    train_years = list(range(test_year - train_window, test_year + 1))
    X_test = X[X.index.isin(sortDateMulti(test_games))]
    X_train_index = split_train_test_year(X.index, train_years, test_year)[0]
    X_train = X[X.index.isin(X_train_index)].sort_index(level=0)[:X_test.index[0]].drop(index = X_test.index)
    if size_cons == True:
        size = len(split_train_test_year(X_train.index, list(range(test_year - train_window, test_year)), test_year)[0])
        X_train = X_train.tail(size)
    Y_train, Y_test = Y[Y.index.isin(X_train.index)].reindex(X_train.index), Y[Y.index.isin(X_test.index)].reindex(X_test.index)
    return X_train, X_test, Y_train, Y_test 

def check_dataframe_NaN(df_list, gameIdList):
    i = 0
    for df in df_list:
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

'''
EXECUTION 
---------------------------------------------
'''

init_ray()

train_years = [2021, 2022]
test_year = 2023
train_window = 2

odds_df = select_attributes(5).select_col_odds(['1xBet (%)', 'Pinnacle (%)', 'Unibet (%)', 'William Hill (%)', 'bet365 (%)'])
mlval_df = select_attributes(5).select_mlval(['team.elo.booker.lm', 'opp.elo.booker.lm', 'team.elo.booker.combined', 'opp.elo.booker.combined', 'elo.prob', 'predict.prob.booker', 'predict.prob.combined', 'elo.court30.prob', 'raptor.court30.prob', 'booker_odds.Pinnacle'])
per_metric_df = select_attributes(5).select_col_per(['pm_elo_prob1','pm_odd_prob','pm_raptor_prob1','pm_6_elo_prob1','pm_6_odd_prob','pm_6_raptor_prob1'])
elo_df = select_attributes(5).select_elo(['elo_prob', 'raptor_prob'])
game_df = select_attributes(5).select_col_game(['streak', 'numberOfGamesPlayed', 'daysSinceLastGame', 'matchupWins', 'win_per'])
team_stat_df = select_attributes(5).select_col_team(['3P%', 'Ortg', 'Drtg', 'PTS', 'TOV%', 'eFG%'])
data_list = [odds_df, mlval_df, per_metric_df, elo_df, game_df, team_stat_df]

X_train, X_test, Y_train, Y_test = split_data(data_list, train_years, test_year, True)
X_train_, X_test_, Y_train_, Y_test_ = split_data_test_games(data_list, train_window, test_games, True, True)












def findParamsXGB(X_train, Y_train):
    param_grid = {
        "n_estimators" : [50, 100, 150],
        "max_depth" : [1, 3, 5, 7],
        "learning_rate" : [0.01, 0.02, 0.03, 0.04],
        "min_child_weight" : [4, 5, 6]
    }

    grid = GridSearchCV(XGBClassifier(), param_grid, refit = True, verbose = 3)
    grid.fit(X_train, Y_train)
    print(grid.best_params_)
    print(grid.best_estimator_)
    return grid.best_estimator_

#clf = findParamsXGB(X_train, Y_train)
#clf = XGBClassifier(learning_rate = 0.01, max_depth = 6, n_estimators = 150, min_child_weight = 4) 

def xgboost(clf, X_train, Y_train, X_test, Y_test):
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
    
    acc = accuracy_score(Y_test, Y_pred)
    
    print("\nAccuracy of %s is %s"%(name, acc))
    print(pd.DataFrame(data = list(model.feature_importances_), index = list(X_train.columns), columns = ["score"]).sort_values(by = "score", ascending = False).head(30))
    print("Test  Accuracy : %.3f" %accuracy_score(Y_test, Y_pred))
    print("Train Accuracy : %.3f" %accuracy_score(Y_train, Y_train_pred))
    return Y_pred_prob, Y_pred_prob_adj

def getOddBreakdown(Y_pred_prob, Y_test, bettingOddsAll):
    odds_all = bettingOddsAll[bettingOddsAll.index.isin(Y_test.index)]
    for col in odds_all.columns:
        odd_preds = [1 if odd > 0.5 else 0 for odd in list(odds_all[col])]
        print("Odd Accuracy of {}".format(col) + " : %.3f"%accuracy_score(Y_test, odd_preds))
    df = pd.DataFrame(index = Y_test.index.get_level_values(0).unique(), data = Y_pred_prob[::2], columns = ['predProb'])
    odds_all_adj = odds_all[::2].set_index(df.index)
    df['signal'] = Y_test[::2].set_index(df.index)
    return df, odds_all_adj

def findReturns(df, x_columns):
    retHome, retAway = findProportionGained(x_columns)
    retHome = retHome[retHome.index.isin(df.index)].rename('retHome', inplace = True)
    retAway = retAway[retAway.index.isin(df.index)].rename('retAway', inplace = True) 
    df = pd.concat([df, retHome, retAway], axis =1)
    df = df.reindex(sortDate(df.index))
    return df

def getKellyBreakdown(df, odds_all, alpha, x_columns, max_bet, bet_diff):
    df = findReturns(df, x_columns)
    df['mean'] = odds_all.mean(axis = 1)[odds_all.index.isin(df.index)]
    df['mean_diff'] = abs(df['predProb'] - df['mean'])
    df['per_bet'] = df.apply(lambda d: kellyBet(d['predProb'], alpha, d['retHome'], d['retAway'])[0], axis = 1)
    df['per_bet'] = df.apply(lambda d: d['per_bet'] if bet_diff[0] < d['mean_diff'] < bet_diff[1] else 0, axis = 1)  
    df['home'] = df.apply(lambda d: kellyBet(d['predProb'], alpha, d['retHome'], d['retAway'])[1], axis = 1)
    df['per_bet'] = df['per_bet'].where(df['per_bet'] <= max_bet, max_bet)
    df['return'] = df.apply(lambda d: 1 + returnBet(d['per_bet'], d['signal'], d['retHome'], d['retAway'], d['home']), axis = 1)
    
    df['adj_return'] = df.apply(lambda d: 1 if d['return'] < 1 else d['return'], axis = 1)
    print(df)
    return df 

def Kelly(df, odds_all, alpha, x_columns, max_bet, bet_diff):
    df = getKellyBreakdown(df, odds_all, alpha, x_columns, max_bet, bet_diff)
    df = df.reindex(sortDate(df.index))
    index = sortAllDates(df.index)
    
    per_bet = convertReturns(df['per_bet'], index)
    returns = convertReturns(df['adj_return'] - 1, index)
    returns.rename(columns = {'adj_return' : 'return'}, inplace = True)
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

x_columns = ['bet365_return', 'Unibet_return']
clf = XGBClassifier(learning_rate = 0.02, max_depth = 6, min_child_weight = 6, n_estimators = 150)
Y_pred_prob, Y_pred_prob_adj = xgboost(clf, X_train, Y_train, X_test, Y_test)
df, odds_all = getOddBreakdown(Y_pred_prob, Y_test, bettingOddsAll)
dfAll, returns = Kelly(df, odds_all, 0.2, x_columns, 1, [0, 100])

x = np.arange(1, len(returns) + 1)
y = list(returns.array)
plt.plot(x, y, label = 'PERCENTAGE RETURN')
plt.show()

def findOptimalKellyParameters():
    param_grid = {
        'kelly_factor' : [0.15, 0.2, 0.25],
        'max_bet' : [0.05, 0.075, 0.10, 0.15, 0.2],
        'bet_diff' : [[0, 0.2], [0, 0.1], [0.03, 0.2], [0.03, 0.1]]
                  }
    keys, values = zip(*param_grid.items())
    permutations_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_grid

def findMaxParams():
    permutations_grid = findOptimalKellyParameters()
    returnList = []
    for i in range (0, len(permutations_grid) - 1):
        with HiddenPrints(): 
            dfAll, returns = Kelly(df, odds_all, permutations_grid[i]['kelly_factor'], x_columns, permutations_grid[i]['max_bet'], bet_diff = permutations_grid[i]['bet_diff'])
            returnList.append(returns[-1])
    print(max(returnList), returnList.index(max(returnList)))
    return returnList, permutations_grid[returnList.index(max(returnList))]
'''
EXECUTION
-------------------------------------------
'''
returns_params, opt_params = findMaxParams()

dfAll, returns = Kelly(df, opt_params['kelly_factor'], x_columns, opt_params['max_bet'], opt_params['avoid_odds'], opt_params['odd_difference'])

x = np.arange(1, len(returns) + 1)
y = list(returns.array)
plt.plot(x, y, label = 'PERCENTAGE RETURN')
plt.show()

#END 

clf = XGBClassifier(learning_rate = 0.02, max_depth = 6, n_estimators = 150, min_child_weight = 6)


def findParamsXGBPost():
    param_grid = {
        "n_estimators" : [50, 100, 150],
        "max_depth" : [3, 5, 7],
        "learning_rate" : [0.005, 0.01, 0.02],
        "min_child_weight" : [4, 5, 6]
    }

    keys, values = zip(*param_grid.items())
    permutations_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_grid

def findOptimalParams(): 
    returnParams = []
    permutations_grid = findParamsXGBPost()
    for i in range(0, len(permutations_grid) - 1):
        clf = XGBClassifier(learning_rate = permutations_grid[i]['learning_rate'], max_depth = permutations_grid[1]['max_depth'], n_estimators = permutations_grid[1]['n_estimators'], min_child_weight = permutations_grid[1]['min_child_weight'])
        with HiddenPrints():
            Y_pred_prob, Y_pred_prob_adj = xgboost(clf, X_train, Y_train, X_test, Y_test)
            df, odds_all = getOddBreakdown(Y_pred_prob, Y_test, bettingOddsAll)
            dfAll, returns = Kelly(df, odds_all, 0.2, x_columns, 1, [0, 100])
            returnParams.append(returns[-1])
    return permutations_grid, returnParams

permutations_grid, returnParams = findOptimalParams()
print(max(returnParams))
print(permutations_grid[returnParams.index(max(returnParams))])

j = returnParams.index(max(returnParams))
clfOpt = XGBClassifier(n_estimators = permutations_grid[j]['n_estimators'], max_depth = permutations_grid[j]['max_depth'], learning_rate = permutations_grid[j]['learning_rate'], min_child_weight = permutations_grid[j]['min_child_weight'])

Y_pred_prob = xgboost(clfOpt)
df = getOddBreakdown(Y_pred_prob, Y_test)

dfAll, returns = Kelly(df, 0.15, x_columns, 0.05, [0, 100], [0.1, 1])
x = np.arange(1, len(returns) + 1)
y = list(returns.array)
plt.plot(x, y, label = 'PERCENTAGE RETURN')
plt.show()

ray.shutdown()
