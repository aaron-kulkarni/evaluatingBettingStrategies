import pandas as pd
import os
import sys
sys.path.append("/Users/shaolin/iCloud/Project/py_lib/")
import read_excel as myread
import strat 
import glob
import ray
import pdb
import time
import numpy_financial as npf
import datetime
import numpy as np
import ray
import matplotlib.pyplot as plt
import warnings
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 150)
#https://www.anyscale.com/blog/how-to-distribute-hyperparameter-tuning-using-ray-tune
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import graphviz
from sklearn.metrics import accuracy_score

@ray.remote
def read_all(fl_):
    import sys
    import pandas as pd
    sys.path.append("/Users/shaolin/iCloud/Project/py_lib/")
    import read_excel as myread
    
    df_=pd.read_csv(fl_,header=[0,1],index_col=0)
    f_=fl_.split('/')[::-1][0]
    print(f_)
    #df_['_file']=f_
    #myread.clean_df_col(df_)
    print(df_.columns)
    return(df_)

@ray.remote
def read_all_elo(fl_):
    import sys
    import pandas as pd
    sys.path.append("/Users/shaolin/iCloud/Project/py_lib/")
    import read_excel as myread
    
    df_=pd.read_csv(fl_,index_col=0)
    f_=fl_.split('/')[::-1][0]
    print(f_)
    #df_['_file']=f_
    #myread.clean_df_col(df_)
    print(df_.columns)
    return(df_)


ray.init(num_cpus=3)

# betting odds
fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/bettingOddsData/'
fl_list=glob.glob(fl_dir+'*')

read_list=[i for i in fl_list if 'adj_prob_2' in i]
start_time = time.time()
rslt= ray.get([read_all.remote(f) for  f in read_list])
duration = time.time() - start_time
ray.shutdown() 
df_odds=pd.concat(rslt)

df_odds['homeProbAdj'].info()
odds_var =[i for i in df_odds['homeProbAdj'].columns if 1-df_odds['homeProbAdj'][i].isna().sum()/df_odds.shape[0]>.8]

#multi columns index slice
idx = pd.IndexSlice
df_odds.loc[:,idx[:,'Parimatch (%)']]
df_odds[[('homeProbAdjMiss',i ) for i in df_odds['homeProbAdj'].columns]]=df_odds['homeProbAdj'].isna()
#fill in missing value by columns mean
df_odds_keep= df_odds['homeProbAdj'][odds_var]
df_odds_keep['odds_mean'] = df_odds_keep.mean(axis=1)

for var in odds_var:
    df_odds_keep[var] = df_odds_keep[var].fillna(df_odds_keep['odds_mean'])
#
pca = PCA(n_components=2, svd_solver='full')
pca_df = pca.fit_transform(df_odds_keep[odds_var])
odds_pca = pd.DataFrame(data = pca_df,index=df_odds_keep.index, columns = ['PCA{}'.format(i) for i in range(1, 2 + 1)])
print('Total variance explained by PCA is {}'.format(sum(pca.explained_variance_ratio_)))
loading = pd.DataFrame(data = pca.components_, index = odds_pca.columns, columns = df_odds_keep.columns[:-1])

# choose booker with less missing, then fill missing by avg non-miss booker
# for that game ; doing PCA for choosing booker
df_odds_cl = pd.concat([df_odds_keep,odds_pca],keys=['homeProbAdj','PCA'],axis=1)
#
#pca=pd.read_csv(fl_dir+'PCA_2_betting_odds_all.csv',index_col=0)

#ELO
fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/eloData/'
fl_list=glob.glob(fl_dir+'*')

read_list=[i for i in fl_list if 'team_elo' in i]

start_time = time.time()
rslt= ray.get([read_all_elo.remote(f) for  f in read_list])
duration = time.time() - start_time
ray.shutdown() 
df_elo=pd.concat(rslt)
# use ELO all files + raptor elo 
elo_all=pd.read_csv(fl_dir+'nba_elo_all.csv',index_col=0)

elo_all=elo_all[[i for i in elo_all.columns if 'pre' in i or  'neu' in i or 'sea' in i]]

elo1_var = [ i  for i in elo_all.columns if '2' not in i]
elo2_var = [ i  for i in elo_all.columns if '1' not in i]

y_var = 'raptor1_pre'
Y = elo_all[~pd.isna(elo_all[y_var])][elo1_var][y_var]
X = elo_all[~pd.isna(elo_all[y_var])][elo1_var].drop(columns=y_var)

dmatrix = xgb.DMatrix(data=X, label=Y)

params = { 'max_depth': [3,6,10],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [100, 500, 1000],
           'colsample_bytree': [0.3, 0.7]}

xgbr = xgb.XGBRegressor(seed = 20)
clf = GridSearchCV(estimator=xgbr, 
                   param_grid=params,
                   scoring='neg_mean_squared_error', 
                   verbose=1)
clf.fit(X, Y)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
params =  clf.best_params_
booster = xgb.XGBRegressor(**params)
booster.fit(X, Y)
elo_all[y_var+'_pred'] = booster.predict(elo_all[elo1_var].drop(columns=y_var))

y_var = 'raptor2_pre'
Y = elo_all[~pd.isna(elo_all[y_var])][elo2_var][y_var]
X = elo_all[~pd.isna(elo_all[y_var])][elo2_var].drop(columns=y_var)

dmatrix = xgb.DMatrix(data=X, label=Y)

params = { 'max_depth': [3,6,10],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [100, 500, 1000],
           'colsample_bytree': [0.3, 0.7]}

xgbr = xgb.XGBRegressor(seed = 20)
clf = GridSearchCV(estimator=xgbr, 
                   param_grid=params,
                   scoring='neg_mean_squared_error', 
                   verbose=1)
clf.fit(X, Y)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
params =  clf.best_params_
booster = xgb.XGBRegressor(**params)
booster.fit(X, Y)
elo_all[y_var+'_pred'] = booster.predict(elo_all[elo2_var].drop(columns=y_var))

# randomized serach can be used
params = { 'max_depth': [3, 5, 6, 10, 15, 20],
           'learning_rate': [0.01, 0.1, 0.2, 0.3],
           'subsample': np.arange(0.5, 1.0, 0.1),
           'colsample_bytree': np.arange(0.4, 1.0, 0.1),
           'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
           'n_estimators': [100, 500, 1000]}
xgbr = xgb.XGBRegressor(seed = 20)
clf = RandomizedSearchCV(estimator=xgbr,
                         param_distributions=params,
                         scoring='neg_mean_squared_error',
                         n_iter=25,
                         verbose=1)
clf.fit(X, Y)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
params =  clf.best_params_
booster = xgb.XGBRegressor(**params)
booster.fit(X, Y)
y_pred = booster.predict(X)

# plot importance 
with plt.style.context("ggplot"):
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    xgb.plot_importance(booster)
plt.show()

with plt.style.context("ggplot"):
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)
    xgb.plotting.plot_tree(booster, ax=ax, num_trees=1)

plt.show()

# elo clean up elo + elo_raptor 
# fill missing raptor
elo_all['raptor1_pre'] = elo_all.apply(lambda d: d['raptor1_pre_pred'] if pd.isna(d['raptor1_pre']) else d['raptor1_pre'],axis=1)
elo_all['raptor2_pre'] = elo_all.apply(lambda d: d['raptor2_pre_pred'] if pd.isna(d['raptor2_pre']) else d['raptor2_pre'],axis=1)

df_elo_cl = elo_all[['season','neutral','elo1_pre','elo2_pre','raptor1_pre','raptor2_pre']].copy()
df_elo_cl['elo_rt'] = df_elo_cl['elo1_pre']/df_elo_cl['elo2_pre']
df_elo_cl['raptor_rt'] = df_elo_cl['raptor1_pre']/df_elo_cl['raptor2_pre']
#

# gameState

fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/gameStats/'
fl_list=glob.glob(fl_dir+'*')

read_list=[i for i in fl_list if 'game_state_data' in i]
 
start_time = time.time()
rslt= ray.get([read_all.remote(f) for  f in read_list])
duration = time.time() - start_time
ray.shutdown() 
df_game =pd.concat(rslt)
df_game[('gameState','status')] = df_game['gameState'].apply(lambda d: 1 if d['winner'] ==d['teamHome'] else 0,axis=1)
df_game[('gameState','spread')] = df_game['home']['points'] - df_game['away']['points'] 
df_game[('gameState','streak_df')] = df_game['home']['streak'] - df_game['away']['streak']
df_game[('gameState','machupWin_df')] = df_game['home']['matchupWins'] - df_game['away']['matchupWins']
df_game[('gameState','avgSalary_rt')] = df_game['home']['avgSalary'] /df_game['away']['avgSalary'] 

game_state_var =[
    ('gameState',            'teamHome'),
    ('gameState',            'teamAway'),
    ('gameState',           'timeOfDay'),
    ('gameState',            'location'),
    ('gameState',              'status'),
    ('gameState',              'spread'),
    ('gameState',           'streak_df'),
    ('gameState',           'avgSalary_rt'),
    ('gameState',        'machupWin_df'),
    ('gameState',             'rivalry'),
    (     'home',              'streak'),
    (     'home',   'daysSinceLastGame'),
    (     'home',         'matchupWins'),
    #(     'home',           'avgSalary'),
    (     'home', 'numberOfGamesPlayed'),
    (     'away',   'daysSinceLastGame'),
    (     'away', 'numberOfGamesPlayed')
     ]
df_game_cl = df_game[game_state_var].copy()

df_game_cl[('home','daysSinceLastGame')] = df_game_cl[('home','daysSinceLastGame')].fillna(0)
df_game_cl[('away','daysSinceLastGame')] = df_game_cl[('away','daysSinceLastGame')].fillna(0)

#df_game clean up

#perf matrix

fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/perMetric/'
fl_list=glob.glob(fl_dir+'*')

read_list=[i for i in fl_list if 'metric' in i]

start_time = time.time()
rslt= ray.get([read_all.remote(f) for  f in read_list])
duration = time.time() - start_time
ray.shutdown() 
df_perf=pd.concat(rslt)
df_perf.drop(columns=df_perf.columns[4],inplace=True)
df_orig_col=df_perf.columns
new_level0=df_perf.columns.get_level_values(1)
new_level1=df_perf.columns.get_level_values(0)
array_d=[list(new_level0),list(new_level1)]
df_perf[list(zip(*array_d))] = df_perf
df_perf.drop(columns=df_orig_col,inplace=True)

# teamstat
fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/teamStats/'
fl_list=glob.glob(fl_dir+'*')

read_list=[i for i in fl_list if 'team_total_stats_2' in i]

start_time = time.time()
rslt= ray.get([read_all.remote(f) for  f in read_list])
duration = time.time() - strt_time
ray.shutdown() 
df_team=pd.concat(rslt)

df_team_home2away = df_team['home'][df_team['home'].columns[1:]]/df_team['away'][df_team['away'].columns[1:]]
df_md=pd.concat([df_game_cl['gameState']['status'],df_team['home'],df_team_home2away],keys=['stat','game','rt'],axis=1)
df_md.columns = df_md.columns.get_level_values(0) + '_' +  df_md.columns.get_level_values(1)
df_md=df_md.drop(columns=[i for i in df_md.columns if 'PTS'  in i or 'Ortg' in i or 'Drtg' in i or 'pos' in i])
df_md.replace([np.inf, -np.inf], np.nan,inplace=True)
X=df_md[list(df_md.columns[2:])]
Y=df_md[['stat_status']]
dmatrix = xgb.DMatrix(data=X, label=Y)

booster = xgb.train({'max_depth':4, 'eta': 0.01,'objective':'binary:logistic','max_cat_to_onehot': 5,'min_child_weight': 4},dmatrix,num_boost_round=100)

feature_important=booster.get_score(importance_type='gain')
pd.DataFrame(data=list(feature_important.values()), index=list(feature_important.keys()), columns=["score"]).sort_values(by = "score", ascending=False)

team_stat_var =['Ortg','Drtg',
'PTS','3P%','poss_per_poss','ass_per_poss',
'TRB%','TOV%','FG','FGA','3P','FT','FTA',
'ORB','TRB','TS%','eFG%','ORB%','DRB%','TOV%']

with plt.style.context("ggplot"):
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    xgb.plot_importance(booster)
plt.show()

with plt.style.context("ggplot"):
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)
    xgb.plotting.plot_tree(booster, ax=ax, num_trees=1)

plt.show()

# df_team stat include current game; cannot be used , however
# try to see whuch team stat is important.
# team_stat_var


# team stat before game
fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/averageTeamData/'
fl_list=glob.glob(fl_dir+'*')

read_list=[i for i in fl_list if 'team_per_5' in i]

start_time = time.time()
rslt= ray.get([read_all.remote(f) for  f in read_list])
duration = time.time() - start_time
ray.shutdown() 
df_avg_home=pd.concat(rslt)
df_avg_home = df_avg_team.select_dtypes(include=np.number)

df_avg_home_d = df_avg_home['team'] / df_avg_home['opp']


fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/averageTeamData/'
fl_list=glob.glob(fl_dir+'*')

read_list=[i for i in fl_list if 'away_per_5' in i]

start_time = time.time()
rslt= ray.get([read_all.remote(f) for  f in read_list])
duration = time.time() - start_time
ray.shutdown() 
df_avg_away=pd.concat(rslt)
df_avg_away = df_avg_away.select_dtypes(include=np.number)

level_col = dict(zip(df_avg_away.columns.levels[0], ["away_opps","away_team"]))
df_avg_away  = df_avg_away.rename(columns=level_col, level=0)

df_avg_away_d = df_avg_away['away_team'] / df_avg_away['away_opps']
df_avg_home_away_std = pd.concat([df_avg_home_d,df_avg_away_d],keys=['home','away'],axis=1)

df_avg_home_away_rt=df_avg_home_away_std['home']/df_avg_home_away_std['away']

df_avg_home_away_rt=df_avg_home_away_rt.dropna(axis=0,how='all')
#df_avg_team_away_std.columns = df_avg_team_away_std.columns.get_level_values(0) + '_' +  df_avg_team_away_std.columns.get_level_values(1)

#team stat PCA
fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/averageTeamData/'
team_pca_home=pd.read_csv(fl_dir+'avg_5_PCA_home_all.csv',index_col=0)
#team_pca_away=pd.read_csv(fl_dir+'avg_5_PCA_away_all.csv',index_col=0)
#pca_df=pd.concat([team_pca_home,team_pca_away],keys=['home','away'],axis=1)
#pca_df_d=pca_df.copy()
#pca_df_d.columns = pca_df_d.columns.get_level_values(0) + '_' + pca_df_d.columns.get_level_values(1)

team_pca_home_win=pd.read_csv(fl_dir+'avg_5_PCA_home_win_all.csv',index_col=0)
team_pca_home_loss=pd.read_csv(fl_dir+'avg_5_PCA_home_loss_all.csv',index_col=0)
#team_pca_away_win=pd.read_csv(fl_dir+'avg_5_PCA_away_win_all.csv',index_col=0)
#team_pca_away_loss=pd.read_csv(fl_dir+'avg_5_PCA_away_loss_all.csv',index_col=0)

var_pca=team_pca_home.columns[:24]
df_team_pca=pd.concat([team_pca_home,team_pca_home_win,team_pca_home_loss],keys=['team_pca','home_win','home_loss'],axis=1)

# team shoot quality
fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/shotData/'
fl_list=glob.glob(fl_dir+'*')

read_list=[i for i in fl_list if '_home' in i]

start_time = time.time()
rslt= ray.get([read_all_elo.remote(f) for  f in read_list])
duration = time.time() - start_time
ray.shutdown() 
home_shot=pd.concat(rslt)

read_list=[i for i in fl_list if '_away' in i]

start_time = time.time()
rslt= ray.get([read_all_elo.remote(f) for  f in read_list])
duration = time.time() - start_time
ray.shutdown() 
away_shot=pd.concat(rslt)

home_shot['shot_quality'] = home_shot['home_avg_shot_quality']/home_shot['away_avg_shot_quality']
away_shot['shot_quality'] = away_shot['home_avg_shot_quality']/away_shot['away_avg_shot_quality']

shot_quality=pd.concat([home_shot[['shot_quality','home_avg_shot_quality']],away_shot[['shot_quality']]],keys=['home','away'],axis=1)

#comb all data source
#df_game_cl,df_odds_cl,df_perf
#df_elo_cl
#df_team_pca
game_df_var = [
    #('gameState',            'teamHome'),
     #       ('gameState',            'teamAway'),
      #      ('gameState',           'timeOfDay'),
       #     ('gameState',            'location'),
            ('gameState',              'status'),
            ('gameState',              'spread'),
            ('gameState',           'streak_df'),
            ('gameState',        'avgSalary_rt'),
            ('gameState',        'machupWin_df'),
            ('gameState',             'rivalry'),
            (     'home',              'streak'),
            (     'home',   'daysSinceLastGame'),
            (     'home',         'matchupWins'),
            (     'home', 'numberOfGamesPlayed'),
            (     'away',   'daysSinceLastGame'),
            (     'away', 'numberOfGamesPlayed')]

df_game_all = pd.concat(
    [df_game_cl['gameState'][['status','spread']],
     df_game_cl['gameState'][['streak_df','machupWin_df','avgSalary_rt', 'rivalry']],
     df_elo_cl[['season','neutral']],
df_elo_cl[['elo1_pre','elo2_pre','raptor1_pre','raptor2_pre','elo_rt','raptor_rt']],
     df_game_cl['home'],
     df_game_cl['away'],
     df_odds_cl['homeProbAdj'],
     df_odds_cl['PCA'],
     df_perf['home'],
     df_perf['away'],
     df_avg_home_away_rt,
     df_team_pca['team_pca'],
     df_team_pca['home_win'],
     df_team_pca['home_loss'],
     shot_quality['home'],
     shot_quality['away']
     ],
    keys=['y','game','game1','elo','game_home','game_away','booker_odds','booker_pca','homeOdds_perf','awayOdds_perf','home2away','homeStat_pca','homeStat_pca_w','homeStat_pca_l','home_shot','away_shot'],axis=1)

df_game_all = pd.concat(
    [df_game_cl['gameState'][['status','spread']],
     df_game_cl['gameState'][['streak_df','machupWin_df','avgSalary_rt', 'rivalry']],
     df_elo_cl[['season','neutral']],
df_elo_cl[['elo1_pre','elo2_pre','raptor1_pre','raptor2_pre','elo_rt','raptor_rt']],
     df_game_cl['home'],
     df_game_cl['away'],
     df_odds_cl['homeProbAdj'],
     df_odds_cl['PCA'],
     df_perf['home'],
     df_perf['away'],
     df_avg_home_away_rt,
     df_team_pca['team_pca'],
     df_team_pca['home_win'],
     df_team_pca['home_loss']
     ],
    keys=['y','game','game1','elo','game_home','game_away','booker_odds','booker_pca','homeOdds_perf','awayOdds_perf','home2away','homeStat_pca','homeStat_pca_w','homeStat_pca_l'],axis=1)

df_game_sub=df_game_all.dropna()
df_game_sub=df_game_sub[(df_game_sub['booker_odds']['Pinnacle (%)']<=.7)&(df_game_sub['booker_odds']['Pinnacle (%)']>=0.3)]

Y=df_game_sub['y']['status']
X=df_game_sub[['game','game1','elo','game_home','game_away','booker_odds','booker_pca','homeOdds_perf','awayOdds_perf','home2away','homeStat_pca','homeStat_pca_w','homeStat_pca_l']]

X.columns = X.columns.get_level_values(0) + '_' +  X.columns.get_level_values(1)
#categorical_features to numeric
X[['conference','division','none']]=pd.get_dummies(X['game_rivalry'])
X=X.drop(columns=['game_rivalry'])

#X=X[[i for i in X.columns if 'booker' not in i]]
X=X[[i for i in X.columns if 'elo'  in i]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=10)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.80, stratify=Y, random_state=15)
dmat_train = xgb.DMatrix(X_train, Y_train, feature_names=X.columns, enable_categorical=True)
dmat_test = xgb.DMatrix(X_test, Y_test, feature_names=X.columns, enable_categorical=True)

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from  lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import LGBMClassifier

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV


names = [
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "GradientBoosting",
    "LGBMClassifier",
    "XGBoost"
]
classifiers = [
    SVC(kernel="rbf", C=1,gamma='scale'),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=7, n_estimators=200,criterion="log_loss",n_jobs=-1),
    AdaBoostClassifier(learning_rate=0.01, n_estimators=30),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=1,  n_estimators=90),
    #XGBClassifier(learning_rate=0.1,max_depth=1,n_estimators=90,colsample_bytree=.9
    LGBMClassifier(learning_rate=0.2,max_depth=1,n_estimators=100),
    XGBClassifier(learning_rate=0.1,max_depth=1,n_estimators=100)
]

classifiers4sub = [
    SVC(kernel="rbf", C=1.5,gamma='scale'),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=600,criterion="log_loss",n_jobs=-1),
    AdaBoostClassifier(learning_rate=0.01, n_estimators=30),
    GradientBoostingClassifier(learning_rate=0.01, max_depth=3,  n_estimators=100),
    #XGBClassifier(learning_rate=0.1,max_depth=1,n_estimators=90,colsample_bytree=.9
    LGBMClassifier(learning_rate=0.05,max_depth=1,n_estimators=100),
    XGBClassifier(learning_rate=0.01,max_depth=1,n_estimators=100,min_child_weight=4)
]

for name,clf  in zip(names, classifiers):
    model=clf.fit(X_train, Y_train)
    print(name)
    calibrated_clf = CalibratedClassifierCV(clf, cv=5)
    calibrated_clf.fit(X_train, Y_train)
    #pdb.set_trace()
    calibrated_clf.predict_proba(X_test)
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
    if 'SVM' in name:
        print("SVM")
    else:
        print(pd.DataFrame(data=list(model.feature_importances_), index=list(X_train.columns), columns=["score"]).sort_values(by = "score", ascending=False).head(30))
    cm = confusion_matrix(Y_test, Y_pred)/len(Y_pred)
    #pdb.set_trace()
    X_test0=pd.concat([X_test,df_game_sub['booker_odds'][['Pinnacle (%)']]],axis=1,join='inner')
    odd_preds = [1 if odd>0.5 else 0 for odd in list(X_test0['Pinnacle (%)'])]
    #print("Confusion Matrix of %s is %s"%(name, cm))
    print("Test  Accuracy : %.3f"%accuracy_score(Y_test, Y_pred))
    print("Train Accuracy : %.3f"%accuracy_score(Y_train, Y_train_pred))
    print("Odd Accuracy : %.3f"%accuracy_score(Y_test, odd_preds))
    
    dd=pd.concat([X_test,Y_test,df_game_sub['booker_odds']['Pinnacle (%)']],join='inner',axis=1)
    dd['pred'] = Y_pred_prob
    dd['pred_bkt']=pd.qcut(dd['pred'],10,duplicates='drop')
    dd['odd_bkt']=pd.qcut(dd['Pinnacle (%)'],10)
    dd['stat_pred']=dd.apply(lambda d: 1 if d['pred']>0.5 else 0 ,axis=1)
    dd['stat_odd']=dd.apply(lambda d: 1 if d['Pinnacle (%)']>0.5 else 0 ,axis=1)
    print(dd.groupby('pred_bkt').status.sum()/dd.groupby('pred_bkt').size(),dd.groupby('pred_bkt').size())
    print(dd.groupby('odd_bkt').status.sum()/dd.groupby('odd_bkt').size(),dd.groupby('pred_bkt').size())



for name,clf  in zip(names, classifiers4sub):
    clf.fit(X_train, Y_train)
    Y_pred= clf.predict(X_test)
    Y_pred_prob= clf.predict_proba(X_test)
    Y_train_pred= clf.predict(X_train)
    #pdb.set_trace()
    acc = accuracy_score(Y_test, Y_pred)
    print("\nAccuracy of %s is %s"%(name, acc))
    #print(clf.feature_importances_)
    if 'SVM' in name:
        print("SVM")
    else:
        print(pd.DataFrame(data=list(clf.feature_importances_), index=list(X_train.columns), columns=["score"]).sort_values(by = "score", ascending=False).head(30))
    cm = confusion_matrix(Y_test, Y_pred)/len(Y_pred)
    #pdb.set_trace()
    X_test0=pd.concat([X_test,df_game_sub['booker_odds'][['Pinnacle (%)']]],axis=1,join='inner')
    odd_preds = [1 if odd>0.5 else 0 for odd in list(X_test0['Pinnacle (%)'])]
    #print("Confusion Matrix of %s is %s"%(name, cm))
    print("Test  Accuracy : %.3f"%accuracy_score(Y_test, Y_pred))
    print("Train Accuracy : %.3f"%accuracy_score(Y_train, Y_train_pred))
    print("Odd Accuracy : %.3f"%accuracy_score(Y_test, odd_preds))
    dd=pd.concat([X_test,Y_test,df_game_sub['booker_odds']['Pinnacle (%)']],join='inner',axis=1)
    dd['pred'] = Y_pred_prob
    dd['pred_bkt']=pd.qcut(dd['pred'],10,duplicates='drop')
    dd['odd_bkt']=pd.qcut(dd['Pinnacle (%)'],10)
    dd['stat_pred']=dd.apply(lambda d: 1 if d['pred']>0.5 else 0 ,axis=1)
    dd['stat_odd']=dd.apply(lambda d: 1 if d['Pinnacle (%)']>0.5 else 0 ,axis=1)
    print(dd.groupby('pred_bkt').status.sum()/dd.groupby('pred_bkt').size(),dd.groupby('pred_bkt').size())
    print(dd.groupby('odd_bkt').status.sum()/dd.groupby('odd_bkt').size(),dd.groupby('pred_bkt').size())



#para tunexs  

param_grid = {'C': [0.1, 1, 1.5], 
              'gamma': ["auto","scale"],
              'kernel': ['rbf']} 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(X_train, Y_train)
print(grid.best_params_)
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)


param_grid = {
    "n_estimators":[50,100,250],
    "max_depth":[1,3,5,7],
    "learning_rate":[0.01,0.1,1]
}
grid = GridSearchCV(GradientBoostingClassifier(), param_grid, refit = True, verbose = 3)
grid.fit(X_train, Y_train)
print(grid.best_params_)
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

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

param_grid = {
    "n_estimators":[30,50,100],
    #"max_depth":[1,3,5,7],
    "learning_rate":[0.005,0.01,0.05]
}
grid = GridSearchCV(AdaBoostClassifier(base_estimator=DecisionTreeClassifier()), param_grid, refit = True, verbose = 3)
grid.fit(X_train, Y_train)
print(grid.best_params_)
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

param_grid = {
    "n_estimators":[100,200,250,300],
    "max_depth":[1,5,7,10],
    #"learning_rate":[0.01,0.05,0.1]
    "n_jobs":[-1],
    "criterion":["log_loss"]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose = 3)
grid.fit(X_train, Y_train)
print(grid.best_params_)
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

param_grid = {
    "n_estimators":[100,200,250,300],
    "max_depth":[1,5,7,10],
    "learning_rate":[0.01,0.05,0.1]
}
grid = GridSearchCV(LGBMClassifier(), param_grid, refit = True, verbose = 3)
grid.fit(X_train, Y_train)
print(grid.best_params_)
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)




