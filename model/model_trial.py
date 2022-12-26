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
import warnings

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category = DeprecationWarning) 

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
    X_train = X_train.reindex(sortDateMulti(X_train.index.get_level_values(0).unique()))
    Y = get_signal()
    Y_train = Y[Y.index.isin(X_train.index)].reindex(X_train.index)
    Y_train = Y_train.dropna()
    X_train = X_train[X_train.index.isin(Y_train.index)]
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

def find_params_XGB(X_train, Y_train):
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

'''
EXECUTION 
---------------------------------------------
'''

init_ray()

train_years = [2021, 2022]
test_year = 2023
train_window = 2

odds_df = select_attributes(5).select_col_odds(['Marathonbet (%)', '1xBet (%)', 'Pinnacle (%)', 'Unibet (%)', 'William Hill (%)'])
mlval_df = select_attributes(5).select_mlval(['team.elo.booker.lm', 'opp.elo.booker.lm', 'team.elo.booker.combined', 'opp.elo.booker.combined', 'elo.prob', 'predict.prob.booker', 'predict.prob.combined', 'elo.court30.prob', 'raptor.court30.prob', 'booker_odds.Pinnacle'])
per_metric_df = select_attributes(5).select_col_per(['pm_elo_prob1','pm_odd_prob','pm_raptor_prob1','pm_6_elo_prob1','pm_6_odd_prob','pm_6_raptor_prob1'])
elo_df = select_attributes(5).select_elo(['elo_prob', 'raptor_prob'])
game_df = select_attributes(5).select_col_game(['streak', 'numberOfGamesPlayed', 'daysSinceLastGame', 'matchupWins', 'win_per'])
team_stat_df = select_attributes(5).select_col_team(['3P%', 'Ortg', 'Drtg', 'TOV%', 'eFG%', 'PTS'])
data_list = [odds_df, mlval_df, per_metric_df, elo_df, game_df, team_stat_df]
check_dataframe_NaN(data_list, getNextGames())

X_train, X_test, Y_train, Y_test = split_data(data_list, train_years, test_year, True)
X_train_, X_test_, Y_train_, Y_test_ = split_data_test_games(data_list, train_window, getNextGames(), True, True)
clf = XGBClassifier(learning_rate = 0.02, max_depth = 4, min_child_weight = 6, n_estimators = 150)

Y_pred_prob = xgboost(clf, X_train, Y_train, X_test, Y_test, 10)
Y_pred_prob_ = xgboost(clf, X_train_, Y_train_, X_test_, Y_test_, 10)
x_columns = ['bet365_return', 'Unibet_return']

def xgboost(clf, X_train, Y_train, X_test, Y_test, cv_value):
    model = clf.fit(X_train, Y_train)
    calibrated_clf = CalibratedClassifierCV(clf, cv = cv_value)
    calibrated_clf.fit(X_train, Y_train)

    Y_pred_prob_adj = calibrated_clf.predict_proba(X_test)[:, 1]
    Y_train_pred_adj = calibrated_clf.predict_proba(X_train)[:, 1]
    Y_train_pred = Y_train_pred_adj/(np.repeat(Y_train_pred_adj[0::2] + Y_train_pred_adj[1::2], 2))
    Y_pred_prob = Y_pred_prob_adj/(np.repeat(Y_pred_prob_adj[0::2] + Y_pred_prob_adj[1::2], 2))
    
    Y_pred = [1 if p > 0.5 else 0 for p in Y_pred_prob]
    Y_train_pred = [1 if p > 0.5 else 0 for p in Y_train_pred]
    print(pd.DataFrame(data = list(model.feature_importances_), index = list(X_train.columns), columns = ["score"]).sort_values(by = "score", ascending = False).head(30))
    Y_test = Y_test.dropna()
    if Y_test.size == 0: 
        print("\nTest Accuracy is not available")
    elif len(Y_test) == len(Y_pred):
        acc = accuracy_score(Y_test, Y_pred)
        print("\nTest Accuracy is %s"%(acc))
    else:
        acc = accuracy_score(Y_test, Y_pred[:len(Y_test)])
        print("\nTest Accuracy is %s"%(acc))
    print("Train Accuracy is %.3f" %accuracy_score(Y_train, Y_train_pred))
    return pd.DataFrame(index = X_test.index, data = Y_pred_prob, columns = ['Y_prob'])

def get_odd_acc(Y_pred_prob, Y_test, odds_all):
    Y_test = Y_test.dropna()
    Y_pred_prob = Y_pred_prob[Y_pred_prob.index.isin(Y_test.index)]
    odds_all = odds_all[odds_all.index.isin(Y_test.index)]
    print('{} home wins this season'.format(int(Y_test[::2].sum())))
    print('{} predicted wins this season'.format(int(Y_pred_prob[::2].sum())))
    for col in odds_all.columns:
        odd_preds = [1 if odd > 0.5 else 0 for odd in list(odds_all[col])]
        print("Odd Accuracy of {}".format(col) + ": %.3f"%accuracy_score(Y_test, odd_preds))
        print('Predicted home wins of {}: {}'.format(col, odds_all[col][::2].sum()))
    return

def init_home(home, plc_home, plc_away):
    if type(home) != bool:
        return home
    if home == True:
        return plc_home
    if home == False:
        return plc_away

def init_signal(home, signal):
    if type(home) != bool:
        return None
    if home == True:
        return signal
    if home == False:
        return 1 - signal
    
def find_returns(df, x_columns):
    ret_home, ret_away = findProportionGained(x_columns)
    ret_home = ret_home[ret_home.index.isin(df.index)].rename('ret_home', inplace = True)
    ret_away = ret_away[ret_away.index.isin(df.index)].rename('ret_away', inplace = True) 
    df = pd.concat([df, ret_home, ret_away], axis =1)
    df = df.reindex(sortDate(df.index))
    return df

def perform_bet(Y_pred_prob, x_columns, alpha, odds_df):
    df = Y_pred_prob[::2].droplevel(1)
    df['odds_mean'] = odds_df[::2].droplevel(1).mean(axis=1)
    df = find_returns(df, x_columns)
    df = pd.concat([df, getTeamsAllYears()[getTeamsAllYears().index.isin(df.index)]], axis=1)
    df['firm_home'], df['firm_away'] = returnBettingFirm(x_columns, df.index)
    df['per_bet'] = df.apply(lambda d: kellyBet(d['Y_prob'], alpha, d['ret_home'], d['ret_away'])[0], axis=1)
    df['home'] = df.apply(lambda d: kellyBet(d['Y_prob'], alpha, d['ret_home'], d['ret_away'])[1], axis=1)
    df['firm'] = df.apply(lambda d: init_home(d['home'], d['firm_home'], d['firm_away']), axis=1)
    df['p_return'] = df.apply(lambda d: init_home(d['home'], d['ret_home'], d['ret_away']), axis=1)
    df['team_abbr'] = df.apply(lambda d: init_home(d['home'], d['teamHome'], d['teamAway']), axis=1)
    df.drop(df.columns[2:8], axis=1, inplace=True)
    return df

def convert_returns(series, index):
    df1, df2 = pd.DataFrame(series),pd.DataFrame(series)
    df1['start'], df2['start'] = 1, 0

    df = pd.concat([df1, df2], axis = 0).reset_index()
    df.set_index(['index', 'start'], inplace = True)
    df = df.reindex(index)
    return df

def find_total(dictReturns):
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

def backtesting_returns(df):
    df['signal_adj'] = getSignal()[getSignal().index.isin(df.index)]
    df = df[df['signal_adj'].notna()]
    df['signal'] = df.apply(lambda d: init_signal(d['home'], d['signal_adj']), axis=1)
    df['return'] = df.apply(lambda d: d['p_return'] if d['signal'] == 1 else 0, axis=1)
    df['adj_return'] = df.apply(lambda d: d['return'] * d['per_bet'], axis=1)
    index = sortAllDates(df.index)
    per_bet, returns = convert_returns(df['per_bet'], index), convert_returns(df['adj_return'], index)
    returns.rename(columns = {'adj_return' : 'return'}, inplace = True)
    dict_returns = pd.concat([per_bet, returns], axis = 1).T.to_dict()
    df_all = pd.DataFrame(find_total(dict_returns)).T
    return df_all

def plot_day_increments(df):
    df['date'] = [gameIdToDateTime(i) for i in df.index.get_level_values(0)]
    df_grouped = df.groupby('date').last()
    return df_grouped.index, df_grouped['total'].array

def write_day_trade(init_amount, df):
    team_dict = {v:k for k,v in getTeamDict().items()}
    bet_dict = df.T.to_dict()
    bet_df = pd.DataFrame(index = df.index, columns = ['amount_bet', 'team', 'potential_return', 'firm'])
    for key in bet_dict.keys():
        amount_bet = round(init_amount * bet_dict[key]['per_bet'], 2)
        init_amount = init_amount - amount_bet
        if amount_bet != 0:
            team = team_dict[bet_dict[key]['team_abbr']]
            potential_return = round(bet_dict[key]['p_return'] * amount_bet, 2)
            firm = bet_dict[key]['firm']
            print('{} - Bet {} on {} with betting firm {}. Potential return {}'.format(key, amount_bet, team, firm, potential_return))
        else:
            print('{} - Empty bet'.format(key))
        bet_df.loc[key] = [amount_bet, team, potential_return, firm]
    return bet_df
        
df = perform_bet(Y_pred_prob, x_columns, 0.15, odds_df)
df_ = perform_bet(Y_pred_prob_, x_columns, 0.15, odds_df)

write_day_trade(23550, df[df.index.isin(getGamesToday())])
write_day_trade(23550, df_)
df_all = backtesting_returns(df)
returns = df_all['total']
get_odd_acc(Y_pred_prob, Y_test, odds_df)

x = np.arange(1, len(returns) + 1)
y = list(returns.array)
plt.plot(x, y, label = 'PERCENTAGE RETURN')
plt.show()

x, y = plot_day_increments(df_all)
plt.plot(x, y, label = 'PERCENTAGE RETURN')
plt.show()

ray.shutdown()
