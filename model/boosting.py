import pandas as pd
import numpy as np
import os
import sys
import time
import datetime

sys.path.insert(0, "..")
from utils.utils import *
from dataProcessing.TeamPerformance import *

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

# My computer has 10 cpu cores, 8 for performance and 2 for efficiency
ray.init(num_cpus = 6)

def getDFAll():
    df = pd.read_csv('../data/all.csv', index_col = 0, header = [0,1])
    return df



booster = xgb.train({'max_depth':4, 'eta': 0.01,'objective':'binary:logistic','max_cat_to_onehot': 5,'min_child_weight': 4},dmatrix,num_boost_round=100)

feature_important=booster.get_score(importance_type='gain')
pd.DataFrame(data=list(feature_important.values()), index=list(feature_important.keys()), columns=["score"]).sort_values(by = "score", ascending=False)

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

df = pd.read_csv('../data/eloData/nba_elo_all_filled.csv', index_col = 0)
df['signal'] = getSignal()

X = df[['homeElo', 'awayElo']]
Y = df['signal']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=10)

#paramater tuning

param_grid = {
    "n_estimators":[50,100],
    "max_depth":[1,3,5],
    "learning_rate":[0.01,0.1],
    "min_child_weight":[4,5,6]
}
grid = GridSearchCV(XGBClassifier(), param_grid, refit = True, verbose = 3)
grid.fit(X_train, Y_train)
print(grid.best_params_)
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

clf = XGBClassifier(learning_rate=0.1,max_depth=5,n_estimators=100)


#model

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import graphviz
from sklearn.metrics import accuracy_score

model = clf.fit(X_train, Y_train)
name = 'xgboost'
calibrated_clf = CalibratedClassifierCV(clf, cv=5)
calibrated_clf.fit(X_train, Y_train)
    #pdb.set_trace()
#calibrated_clf.predict_proba(X_test)
    #Y_pred= clf.predict(X_test)
    #Y_train_pred= clf.predict(X_train)
Y_pred_prob= calibrated_clf.predict_proba(X_test)[:,1]
Y_train_pred= calibrated_clf.predict_proba(X_train)[:,1]

Y_pred=[1 if p>0.5 else 0 for p in Y_pred_prob]
Y_train_pred=[1 if p>0.5 else 0 for p in Y_train_pred]
    #pdb.set_trace()
acc = accuracy_score(Y_test, Y_pred)
print("\nAccuracy of %s is %s"%(name, acc))
    #print(clf.feature_importances_)

df_game_sub = getDFAll()['booker_odds']

print(pd.DataFrame(data=list(model.feature_importances_), index=list(X_train.columns), columns=["score"]).sort_values(by = "score", ascending=False).head(30))

#cm = confusion_matrix(Y_test, Y_pred)/len(Y_pred) !dont have
    #pdb.set_trace()
X_test0=pd.concat([X_test,df_game_sub[['Pinnacle (%)']]],axis=1,join='inner')
odd_preds = [1 if odd>0.5 else 0 for odd in list(X_test0['Pinnacle (%)'])]
    #print("Confusion Matrix of %s is %s"%(name, cm))
print("Test  Accuracy : %.3f"%accuracy_score(Y_test, Y_pred))
print("Train Accuracy : %.3f"%accuracy_score(Y_train, Y_train_pred))
print("Odd Accuracy : %.3f"%accuracy_score(Y_test, odd_preds))
    
dd=pd.concat([X_test,Y_test,df_game_sub['Pinnacle (%)']],join='inner',axis=1)
dd['pred'] = Y_pred_prob
dd['pred_bkt']=pd.qcut(dd['pred'],10,duplicates='drop')
dd['odd_bkt']=pd.qcut(dd['Pinnacle (%)'],10)
dd['stat_pred']=dd.apply(lambda d: 1 if d['pred']>0.5 else 0 ,axis=1)
dd['stat_odd']=dd.apply(lambda d: 1 if d['Pinnacle (%)']>0.5 else 0 ,axis=1)
print(dd.groupby('pred_bkt').signal.sum()/dd.groupby('pred_bkt').size(),dd.groupby('pred_bkt').size())
print(dd.groupby('odd_bkt').signal.sum()/dd.groupby('odd_bkt').size(),dd.groupby('pred_bkt').size())




