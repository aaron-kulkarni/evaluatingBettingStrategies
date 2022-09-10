import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "..")

from utils.utils import *
from dataProcessing.TeamPerformance import *
from dataProcessing.PCA import *
from kelly import *

import matplotlib.pyplot as plt
import ray
import multiprocessing

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
bettingOdds = selectColOdds(['1xBet (%)', 'Marathonbet (%)', 'Pinnacle (%)', 'Unibet (%)', 'William Hill (%)', 'bet-at-home (%)', 'bet365 (%)', 'bwin (%)'])

bettingOddsPCA, coeff = performPCA(bettingOdds, 2)

def selectColElo(select_x):
    eloData = pd.read_csv('../data/eloData/nba_elo_all.csv', index_col = 0)
    return eloData[select_x]
# columns: season, neutral, team1, team2, elo1_pre, elo2_pre, elo_prob1, elo_prob2, elo1_post, elo2_post, carm-elo1_pre, carm-elo2_pre, carm-elo_prob1, carm-elo_prob2, carm-elo1_post, carm-elo2_post, raptor1_pre, raptor2_pre, raptor_prob1, raptor_prob2
elo = selectColElo(['elo_prob1', 'raptor_prob1'])

def selectColPerMetric(select_x):
    perMetric = pd.read_csv('../data/perMetric/performance_metric_all.csv', index_col = 0)
    return perMetric[select_x]

perMetric = selectColPerMetric(['perMetricAway', 'perMetricHome', 'perMetricNAway', 'perMetricNHome'])

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

years = list(np.arange(2015, 2023))
df_all = getDFAll([elo, perMetric], years, True)

X = df_all
Y = getSignal().reindex(X.index)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, random_state = 126, shuffle = True)

# PARAMATER TUNING
def findParamsXGB(X_train, Y_train):
    param_grid = {
        "n_estimators" : [50, 100],
        "max_depth" : [1, 3, 5, 7],
        "learning_rate" : [0.01, 0.1],
        "min_child_weight" : [4, 5, 6]
    }

    grid = GridSearchCV(XGBClassifier(), param_grid, refit = True, verbose = 3)
    grid.fit(X_train, Y_train)
    print(grid.best_params_)
    print(grid.best_estimator_)
    return grid.best_estimator_

#params = findParamsXGB(X_train, Y_train)

clf = XGBClassifier(learning_rate = 0.1, max_depth = 1, n_estimators = 50, min_child_weight = 4)

def xgboost(clf):
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
    #X_test0=pd.concat([X_test,df_game_sub[['Pinnacle (%)']]],axis=1,join='inner')

    print("Test  Accuracy : %.3f" %accuracy_score(Y_test, Y_pred))
    print("Train Accuracy : %.3f" %accuracy_score(Y_train, Y_train_pred))
    return Y_pred_prob

Y_pred_prob = xgboost(clf)

def getOddBreakdown(Y_pred_prob):
    testOdds = bettingOdds[bettingOdds.index.isin(X_test.index)]
    testOdds = testOdds.reindex(X_test.index)
    for col in testOdds.columns:
        odd_preds = [1 if odd > 0.5 else 0 for odd in list(testOdds[col])]
        print("Odd Accuracy of {}".format(col) + " : %.3f"%accuracy_score(Y_test, odd_preds))
    #print("Confusion Matrix of %s is %s"%(name, cm))

    Y_pred_prob = pd.Series(Y_pred_prob, name = 'predProb', index = Y_test.index)
    df = pd.concat([Y_test, Y_pred_prob, testOdds['Pinnacle (%)']], join = 'inner', axis = 1)

    df['pred_bkt'] = pd.qcut(df['predProb'], 10 , duplicates = 'drop')
    df['odd_bkt'] = pd.qcut(df['Pinnacle (%)'], 10)
    df['stat_pred'] = df.apply(lambda d: 1 if d['predProb'] > 0.5 else 0, axis = 1)
    df['stat_odd'] = df.apply(lambda d: 1 if d['Pinnacle (%)'] > 0.5 else 0 ,axis = 1)

    print(df.groupby('pred_bkt').signal.sum()/df.groupby('pred_bkt').size(),df.groupby('pred_bkt').size())
    print(df.groupby('odd_bkt').signal.sum()/df.groupby('odd_bkt').size(),df.groupby('pred_bkt').size())
    return df

df = getOddBreakdown(Y_pred_prob)

def Kelly(df, alpha, predProb, x_columns):
    retHome, retAway = findProportionGained(x_columns)
    retHome = retHome[retHome.index.isin(df.index)].rename('retHome', inplace = True)
    retAway = retAway[retAway.index.isin(df.index)].rename('retAway', inplace = True)
    predProb.rename('prob_Y', inplace = True)
    df_ = pd.concat([df, retHome, retAway, predProb], axis = 1)
    df_['per_bet'] = df_.apply(lambda d: kellyBet(d['prob_Y'], alpha, d['retHome'], d['retAway'])[0], axis = 1)
    df_['home'] = df_.apply(lambda d: kellyBet(d['prob_Y'], alpha, d['retHome'], d['retAway'])[1], axis = 1)
    df_['return'] = df_.apply(lambda d: 1 + returnBet(d['per_bet'], d['signal'], d['retHome'], d['retAway'], d['home']), axis = 1)
    df_['cum_return'] = df_['return'].cumprod()
    #df_['reg_return'] = df_.apply(lambda d: 1 - d['return'], axis = 1).cumsum()

    return df_, df_['cum_return']


x_columns = ['bet365_return', 'William Hill_return', 'Pinnacle_return', 'Coolbet_return', 'Unibet_return', 'Marathonbet_return']

df, returns = Kelly(df, 0.3, df['predProb'], x_columns)
print(returns)
#print(cum_returns)


x = np.arange(1, len(returns) + 1)
y = list(returns.array)
plt.plot(x, y, label = 'PERCENTAGE RETURN')
plt.show()

maxReturns = []
returnAll = []

for i in range (1, 1000):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, random_state = i, shuffle = True)
    clf = XGBClassifier(learning_rate = 0.1, max_depth = 1, n_estimators = 50, min_child_weight = 4)
    Y_pred_prob = xgboost(clf)
    df = getOddBreakdown(Y_pred_prob)
    x_columns = ['bet365_return', 'William Hill_return', 'Pinnacle_return', 'Coolbet_return', 'Unibet_return', 'Marathonbet_return']

    df_, returns = Kelly(df, 0.3, df['predProb'], x_columns)
    x = np.arange(1, len(returns) + 1)
    y = list(returns.array)
    maxReturns.append(max(y))
    returnAll.append(y[-1])

results = [maxReturns, returnAll]
results_df = pd.DataFrame(data = np.array(results).T, columns = ['max', 'end'])
results_df['max'].groupby([0, 5])

ray.shutdown()