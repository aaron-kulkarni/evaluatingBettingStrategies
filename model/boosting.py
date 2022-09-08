import pandas as pd
import numpy as np
import os
import sys
import time
import datetime

sys.path.insert(0, "..")
from utils.utils import *
from dataProcessing.TeamPerformance import *
from dataProcessing.PCA import *

import matplotlib.pyplot as plt
import ray

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import graphviz
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor

#from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import LGBMClassifier

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# My computer has 10 cpu cores, 8 for performance and 2 for efficiency
ray.init(num_cpus = 6)

def selectColOdds(select_x):
    bettingOdds = pd.read_csv('../data/bettingOddsData/adj_prob_home_win_ALL.csv', index_col = 0)
    return bettingOdds[select_x]

# SELECTED HIGH PERFORMING BETTING BOOKMAKERS
bettingOdds = selectColOdds(['1xBet (%)', 'Marathonbet (%)', 'Pinnacle (%)', 'Unibet (%)', 'William Hill (%)', 'bet-at-home (%)', 'bet365 (%)', 'bwin (%)'])

bettingOddsPCA, coeff = performPCA(bettingOdds, 2)

#df['odds'] = bettingOdds
#df['signal', 'status'] = getSignal()

X = bettingOddsPCA
Y = getSignal().reindex(X.index)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, random_state = 15)

# PARAMATER TUNING

param_grid = {
    "n_estimators" : [50, 100],
    "max_depth" : [1, 3, 5],
    "learning_rate" : [0.01, 0.1],
    "min_child_weight" : [4, 5, 6]
}

grid = GridSearchCV(XGBClassifier(), param_grid, refit = True, verbose = 3)
grid.fit(X_train, Y_train)
print(grid.best_params_)
print(grid.best_estimator_)

clf = XGBClassifier(learning_rate = 0.1,max_depth = 1, n_estimators = 50, min_child_weight = 4)

# MODEL

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import graphviz
from sklearn.metrics import accuracy_score

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

testOdds = bettingOdds[bettingOdds.index.isin(X_test.index)]
testOdds = testOdds.reindex(X_test.index)
odd_preds = [1 if odd > 0.5 else 0 for odd in list(testOdds['Pinnacle (%)'])]

#print("Confusion Matrix of %s is %s"%(name, cm))
print("Test  Accuracy : %.3f" %accuracy_score(Y_test, Y_pred))
print("Train Accuracy : %.3f" %accuracy_score(Y_train, Y_train_pred))
print("Odd Accuracy : %.3f" %accuracy_score(Y_test, odd_preds))

Y_pred_prob = pd.Series(Y_pred_prob, name = 'predProb', index = Y_test.index)
df = pd.concat([Y_test, Y_pred_prob, testOdds['Pinnacle (%)']], join = 'inner', axis = 1)

df['pred_bkt'] = pd.qcut(df['predProb'], 10 , duplicates = 'drop')
df['odd_bkt'] = pd.qcut(df['Pinnacle (%)'], 10)
df['stat_pred'] = df.apply(lambda d: 1 if d['predProb'] > 0.5 else 0, axis = 1)
df['stat_odd'] = df.apply(lambda d: 1 if d['Pinnacle (%)'] > 0.5 else 0 ,axis = 1)

print(df.groupby('pred_bkt').signal.sum()/df.groupby('pred_bkt').size(),df.groupby('pred_bkt').size())
print(df.groupby('odd_bkt').signal.sum()/df.groupby('odd_bkt').size(),df.groupby('pred_bkt').size())

ray.shutdown()
