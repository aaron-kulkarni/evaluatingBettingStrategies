import pandas as pd
import numpy as np
import sys
import itertools
sys.path.insert(0, "..")
from datetime import datetime, date, timedelta
from dateutil import rrule
from utils.utils import *
from dataProcessing.TeamPerformance import *
from dataProcessing.PCA import *
from kelly import *
from xgboost import plot_tree
import ray
import glob

import matplotlib.pyplot as plt
import ray 
import multiprocessing
import random
import statistics
from ast import literal_eval
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 150)

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
bettingOddsAll = selectColOdds(['Marathonbet (%)', 'Pinnacle (%)', 'Unibet (%)', 'William Hill (%)', 'bet-at-home (%)', 'bet365 (%)', 'bwin (%)'])

bettingOdds = selectColOdds(['Pinnacle (%)'])

bettingOddsPCA, coeff = performPCA(bettingOddsAll, 2)
bettingOddsAll['odds_mean'] = bettingOddsAll.median(axis=1)
bettingOddsPCA = pd.concat([bettingOddsPCA,bettingOddsAll[['odds_mean']]],axis=1)


def selectColElo(select_x):
    eloData = pd.read_csv('../data/eloData/nba_elo_all.csv', index_col = 0)
    return eloData[select_x]
# columns: season, neutral, team1, team2, elo1_pre, elo2_pre, elo_prob1, elo_prob2, elo1_post, elo2_post, carm-elo1_pre, carm-elo2_pre, carm-elo_prob1, carm-elo_prob2, carm-elo1_post, carm-elo2_post, raptor1_pre, raptor2_pre, raptor_prob1, raptor_prob2
#elo = selectColElo(['elo_prob1', 'raptor_prob1', 'elo1_pre', 'elo2_pre', 'raptor1_pre', 'raptor2_pre'])
elo = selectColElo(['elo_prob1', 'raptor_prob1','team1','team2'])

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

def splitTrainTestYear(X, Y, train_yr,test_yr):
    '''
    splits data into training data and testing data (data that is tested is last year of input data
    
    '''

    X.reset_index(inplace = True)
    X['year'] = X.apply(lambda d: getYearFromId(d['index']), axis = 1)
    X.set_index('index', inplace = True)
    #pdb.set_trace()
    X_train = X[X['year'].isin(train_yr)]
    X_test = X[X['year'].isin(test_yr)]
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

ray.init(num_cpus=3)

# teamstat
fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/averageTeamData/'
fl_list=glob.glob(fl_dir+'*')
read_list=[i for i in fl_list if 'team_per_5' in i]
rslt= ray.get([read_all.remote(f) for  f in read_list])
ray.shutdown() 
df_avg_team=pd.concat(rslt)

fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/averageTeamData/'
fl_list=glob.glob(fl_dir+'*')
read_list=[i for i in fl_list if 'away_per_5' in i]
rslt= ray.get([read_all.remote(f) for  f in read_list])
ray.shutdown() 
df_avg_away=pd.concat(rslt)

team_stat_var =['teamAbbr','Ortg','Drtg','PTS','3P%','poss_per_poss','ass_per_poss',
                   'TRB%','FG','FGA','3P','FT','FTA','ORB','TRB','TS%','eFG%','ORB%','DRB%','TOV%']

df_team_comb=pd.concat([df_avg_team['team'][team_stat_var].rename(columns={'teamAbbr':'team'}),
                        df_avg_away['home'][team_stat_var].rename(columns={'teamAbbr':'team'})],axis=0).dropna().reset_index().set_index(['index','team'])



gameStateData = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', header = [0,1], index_col = 0)
elo = selectColElo(['elo_prob1', 'raptor_prob1','elo_prob2', 'raptor_prob2','team1','team2','elo1_pre','elo2_pre','raptor1_pre','raptor2_pre' ])
bettingOddsAll['odds_mean']=bettingOddsAll['odds_mean']

df_game_home = pd.concat([gameStateData['gameState'][['teamHome','datetime','endtime','rivalry','winner']].rename(columns={'teamHome':'team'}),gameStateData['home'][['numberOfGamesPlayed','playerRoster','streak','matchupWins']],bettingOddsAll[['odds_mean']],elo[['elo_prob1', 'raptor_prob1','elo1_pre','raptor1_pre']].rename(columns={'elo_prob1':'elo_prob','raptor_prob1':'raptor_prob','elo1_pre':'elo','raptor1_pre':'raptor_elo'})],axis=1)
df_game_home['home']=1

df_game_away = pd.concat([gameStateData['gameState'][['teamAway','datetime','endtime','rivalry','winner']].rename(columns={'teamAway':'team'}),gameStateData['away'][['numberOfGamesPlayed','playerRoster','streak','matchupWins']],1-bettingOddsAll[['odds_mean']],elo[['elo_prob2', 'raptor_prob2','elo2_pre','raptor2_pre' ]].rename(columns={'elo_prob2':'elo_prob','raptor_prob2':'raptor_prob','elo2_pre':'elo','raptor2_pre':'raptor_elo'})],axis=1)
df_game_away['home']=0

df_game=pd.concat([df_game_home,df_game_away],axis=0).sort_values(['team','datetime'])

df_game=df_game.reset_index()

df_game['year'] = df_game.apply(lambda d: getYearFromId(d['index']), axis = 1)                   
df_game['stat'] = df_game.apply(lambda d: 1 if d['team']==d['winner'] else 0 ,axis=1)
df_game['playerRoster_1'] = df_game.sort_values('datetime').groupby(['team'])['playerRoster'].shift(1)
df_game['player_change'] = df_game.apply(lambda d: len(set(literal_eval(d['playerRoster']))- set(literal_eval(d['playerRoster_1']))) if 1-pd.isna(d['playerRoster_1']) else 0,axis=1)
df_game['team01'] = df_game['team'].apply(lambda d: 1 if d in ['UAT'] else 0)
df_game['datetime'] = pd.to_datetime(df_game['datetime'])
df_game['weeks'] = df_game.groupby('year')['datetime'].transform(lambda x: (x -  x.min())/np.timedelta64(1, 'D') //7).astype(int)
#df_game.groupby('year')['datetime'].agg([min,max])

fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/perMetric/'
df_perf0=pd.read_csv(fl_dir+'performance_metric_all_elo.csv',header=[0,1],index_col=0)
df_perf0a=pd.concat([gameStateData['gameState'][['teamHome']].rename(columns={'teamHome':'team'}),df_perf0.loc[:,pd.IndexSlice[:,'home']].droplevel(level=1, axis=1)],axis=1)
df_perf0b=pd.concat([gameStateData['gameState'][['teamAway']].rename(columns={'teamAway':'team'}),df_perf0.loc[:,pd.IndexSlice[:,'away']].droplevel(level=1, axis=1)],axis=1)
df_perf=pd.concat([df_perf0a,df_perf0b],axis=0).reset_index().set_index(['game_id','team'])
                   
#df_elo_jin = pd.read_csv('/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/new.team.dat.jin.elo.csv')
df_elo_jin = pd.read_csv('/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/new.team.dat.jin.elo.all.csv')
df_elo_outlier = pd.read_csv('/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/gameid.outlier.alltype.csv')

df_elo_jin = pd.merge(df_elo_jin,df_elo_outlier[df_elo_outlier['type']=='elo.booker'].drop_duplicates(),how='outer',on='gameid')
df_elo_jin = pd.merge(df_elo_jin,df_elo_outlier[df_elo_outlier['type']=='raptor'].drop_duplicates(),how='outer',on='gameid')
df_elo_jin = pd.merge(df_elo_jin,df_elo_outlier[df_elo_outlier['type']=='raptor.booker'].drop_duplicates(),how='outer',on='gameid')

df_elo_jin['elo_booker_outlier'] = df_elo_jin['type_x'].apply(lambda d: 1 if pd.isna(d) else 0)
df_elo_jin['elo_raptor_outlier'] = df_elo_jin['type_y'].apply(lambda d: 1 if pd.isna(d) else 0)
df_elo_jin['booker_raptor_outlier'] = df_elo_jin['type_y'].apply(lambda d: 1 if pd.isna(d) else 0)

df_elo_jin = df_elo_jin.set_index(['gameid','team'])

df_game = df_game.set_index(['index','team'])

df_game_adj = pd.concat([df_game,df_team_comb,df_perf,df_elo_jin[['team.elo.booker.lm', 'team.elo.booker.combined', 
       'predict.prob.booker', 'predict.prob.combined','elo.court30.prob',
'raptor.court30.prob','elo_booker_outlier', 'elo_raptor_outlier', 'booker_raptor_outlier']]],axis=1).reset_index().rename(columns={'level_0':'index'})

#df_game_bak = df_game.copy()
gameStateData = gameStateData['home']

#train_yr = [2015,2016,2017,2018]
#test_yr = [2019]

#X_train= df_game_adj[(df_game_adj['year'].isin(train_yr))][x_var].set_index(['index','team'])
#X_test = df_game_adj[(df_game_adj['year'].isin(test_yr))][x_var].set_index(['index','team'])
#Y_train= df_game_adj[(df_game_adj['year'].isin(train_yr))][y_var].set_index(['index','team'])
#Y_test = df_game_adj[(df_game_adj['year'].isin(test_yr))][y_var].set_index(['index','team'])

####

x_var = ['index','team','odds_mean','streak', 'matchupWins','elo_prob', 'raptor_prob','raptor_elo', 'elo','numberOfGamesPlayed','player_change','home','team01','team.elo.booker.lm', 'team.elo.booker.combined', 'predict.prob.booker', 'predict.prob.combined','elo.court30.prob', 'raptor.court30.prob']

x_var = ['index','team','streak','matchupWins','elo_prob', 'raptor_prob','raptor_elo', 'elo','numberOfGamesPlayed','player_change','home','team01','team.elo.booker.lm', 'team.elo.booker.combined', 'predict.prob.booker', 'predict.prob.combined','elo.court30.prob', 'raptor.court30.prob','elo_booker_outlier', 'elo_raptor_outlier', 'booker_raptor_outlier','Ortg', 'Drtg', 'PTS', '3P%', 'poss_per_poss',
       'ass_per_poss', 'TRB%', 'TOV%', 'FG', 'FGA', '3P', 'FT', 'FTA', 'ORB',
         'TRB', 'TS%', 'eFG%', 'ORB%', 'DRB%']
#,'perMetric', 'perMetricElo', 'perMetricEloN','perMetricN', 'perMetricRaptor', 'perMetricRaptorN']

x_var = ['index','team','odds_mean','streak','matchupWins','elo_prob', 'raptor_prob','raptor_elo', 'elo','numberOfGamesPlayed','player_change','home','team01','team.elo.booker.lm', 'team.elo.booker.combined', 'predict.prob.booker', 'predict.prob.combined','elo.court30.prob', 'raptor.court30.prob','perMetric', 'perMetricElo', 'perMetricEloN',
       'perMetricN', 'perMetricRaptor', 'perMetricRaptorN','Ortg', 'Drtg', 'PTS', '3P%', 'poss_per_poss',
       'ass_per_poss', 'TRB%', 'TOV%', 'FG', 'FGA', '3P', 'FT', 'FTA', 'ORB',
         'TRB', 'TS%', 'eFG%', 'ORB%', 'DRB%']


x_var = ['index','team','streak', 'matchupWins','elo_prob', 'raptor_prob','raptor_elo','numberOfGamesPlayed','player_change','home','team01',
         #'team.elo.booker.lm', 'team.elo.booker.combined','elo',
         'predict.prob.booker', 'predict.prob.combined','elo.court30.prob', 'raptor.court30.prob']

x_var = ['index','team','streak','matchupWins','elo_prob', 'raptor_prob','raptor_elo', 'elo','numberOfGamesPlayed','player_change','home','team.elo.booker.lm', 'team.elo.booker.combined', 'predict.prob.booker', 'predict.prob.combined','elo.court30.prob', 'raptor.court30.prob', 'TOV%', 'eFG%', 'ORB%','TS%', '3P%', 'TRB%' , 'DRB%']

x_var = ['index','team','streak','matchupWins','elo_prob', 'raptor_prob','raptor_elo', 'elo','numberOfGamesPlayed','player_change','home','perMetric', 'perMetricElo', 'perMetricEloN','perMetricN', 'perMetricRaptor', 'perMetricRaptorN','Ortg', 'Drtg', 'PTS', '3P%', 'poss_per_poss',
       'ass_per_poss', 'TRB%', 'TOV%', 'FG', 'FGA', '3P', 'FT', 'FTA', 'ORB',
         'TRB', 'TS%', 'eFG%', 'ORB%', 'DRB%']


#x_var = ['index','team','streak', 'matchupWins','elo_prob', 'raptor_prob','raptor_elo', 'elo','numberOfGamesPlayed','player_change','home','team01']

#x_var = ['index','team','odds_mean','streak', 'matchupWins','elo_prob', 'raptor_prob','raptor_elo', 'elo','numberOfGamesPlayed','player_change','home','team01']

y_var = ['index','team','stat']

train_yr = [2021]
test_yr = [2022]
train_yr = [2015,2016,2017,2018]
test_yr = [2019]
#df_game_adj= df_game_adj[(pd.isna(df_game_adj['team.elo.booker.lm'])==False) & (df_game_adj['numberOfGamesPlayed']>=8)]

def predict(train_yr,test_yr, _df):
    _pred_df = pd.DataFrame()
    for w in _df[_df['year'].isin(test_yr)]['weeks'].unique():
        print(w)
        #pdb.set_trace()
        X_train= df_game_adj[(df_game_adj['year'].isin(train_yr)) | ((df_game_adj['year'].isin(test_yr)) & (df_game_adj['weeks']< w))][x_var].set_index(['index','team'])
        X_test = df_game_adj[(df_game_adj['year'].isin(test_yr)) & (df_game_adj['weeks']==w)][x_var].set_index(['index','team'])
        Y_train= df_game_adj[(df_game_adj['year'].isin(train_yr)) | ((df_game_adj['year'].isin(test_yr)) & (df_game_adj['weeks']< w))][y_var].set_index(['index','team'])
        Y_test = df_game_adj[(df_game_adj['year'].isin(test_yr)) & (df_game_adj['weeks']==w)][y_var].set_index(['index','team'])

        clf = XGBClassifier(learning_rate = 0.01, max_depth = 3, n_estimators =150, min_child_weight = 4)

        #clf = XGBClassifier(learning_rate = 0.02, max_depth = 1, min_child_weight = 4, n_estimators = 150)
        #model = clf.fit(X_train, Y_train)
        #y_p = model.predict_proba(X_test)[:,1]
        #clf = RandomForestClassifier(max_depth = 6, n_estimators =500,n_jobs=-1,  criterion =  "log_loss")                       
        model = clf.fit(X_train, np.ravel(Y_train))
        #plot_tree(model, num_trees=0, rankdir='LR')
        #plt.show()
        name = 'XGBOOST'
        calibrated_clf = CalibratedClassifierCV(clf,cv = 5)
        calibrated_clf.fit(X_train, np.ravel(Y_train))
        Y_pred_prob = calibrated_clf.predict_proba(X_test)[:, 1]
        Y_train_pred_prob = calibrated_clf.predict_proba(X_train)[:, 1]
        Y_pred = [1 if p > 0.5 else 0 for p in Y_pred_prob]
        Y_train_pred = [1 if p > 0.5 else 0 for p in Y_train_pred_prob]

        acc = accuracy_score(Y_test, Y_pred)
        print("\nAccuracy of %s is %s"%(name, acc))
        Y_pred_prob = pd.Series(Y_pred_prob, name = 'predProb', index = Y_test.index)
        df = pd.concat([Y_pred_prob,df_game_adj.set_index(['index','team'])],join = 'inner', axis = 1)
        df_train_ = pd.concat([pd.Series(Y_train_pred_prob, name = 'predProb', index = Y_train.index),df_game_adj.set_index(['index','team'])],join = 'inner', axis = 1)

        #df.groupby('home')['predProb'].mean()
        df_prob_sum=pd.DataFrame(df.reset_index().groupby('index')['predProb'].sum()).rename(columns={'predProb':'prob_scale'})
        df=df[df['home']==1].copy()
        #df['predProb'] = df['predProb']/df_prob_sum['prob_scale']

        df['num_game_bkt'] = pd.qcut(df['numberOfGamesPlayed'], 10, duplicates = 'drop')
        df['pred_bkt'] = pd.qcut(df['predProb'], 10 , duplicates = 'drop')
        df['odd_bkt'] = pd.qcut(df['odds_mean'], 10, duplicates = 'drop')
        df['stat_pred'] = df.apply(lambda d: 1 if d['predProb'] > 0.50 else 0, axis = 1)
        df['stat_odd'] = df.apply(lambda d: 1 if d['odds_mean'] > 0.50 else 0 ,axis = 1)
        df['signal'] = df['stat']
        acc = accuracy_score(df['stat'],df['stat_odd'] )
        print("\nAccuracy of %s is %s"%(name, acc))
        acc = accuracy_score(df['stat'],df['stat_pred'] )
        print("\nAccuracy of %s is %s"%(name, acc))
        home_win=pd.DataFrame(df_game.groupby(['home','year']).stat.sum()/df_game.groupby(['home','year']).size()).reset_index().set_index('home').loc[1]
        df_game['stat_odd'] = df_game['stat'].apply(lambda d: 1 if d>.50 else 0)
        #home_win_odd=pd.DataFrame(df_game.groupby(['home','year'])['stat','stat_odd'].mean()).reset_index().set_index('home').loc[1]
        print("\nHomeWin Act vs. Proj",(pd.DataFrame(df[['stat','stat_pred','stat_odd','predProb','odds_mean']].mean())).round(3))

        #home_win2=pd.DataFrame(df_game_adj.groupby(['home','year']).stat.sum()/df_game_adj.groupby(['home','year']).size()).reset_index().set_index('home').loc[1]

        print("\nHomeWin by YR",pd.concat([home_win,pd.DataFrame(df_game.groupby(['home','year']).odds_mean.mean()).reset_index().set_index('home').loc[1]],axis=1))
        print("\nTrain",df_train_.groupby('home')[['stat','predProb','odds_mean']].mean())

        print(df.groupby('num_game_bkt').stat.sum()/df.groupby('num_game_bkt').size(),df.groupby('num_game_bkt').size())
        print(df.groupby('num_game_bkt').stat_pred.sum()/df.groupby('num_game_bkt').size(),df.groupby('num_game_bkt').size())

        print(df.groupby('pred_bkt').stat.sum()/df.groupby('pred_bkt').size(),df.groupby('pred_bkt').size())
        print(df.groupby('odd_bkt').stat.sum()/df.groupby('odd_bkt').size(),df.groupby('pred_bkt').size())

        print(pd.DataFrame(data = list(model.feature_importances_), index = list(X_train.columns), columns = ["score"]).sort_values(by = "score", ascending = False).head(30))
        _pred_df = pd.concat([_pred_df,df],axis=0)

    return(_pred_df)

dd =  predict(train_yr,test_yr, df_game_adj)
df=dd

df = getOddBreakdown(df.reset_index().set_index('index')['predProb'], df.reset_index().set_index('index')[['signal']])
x_columns = ['bet365_return', 'William Hill_return', 'Pinnacle_return', 'Coolbet_return', 'Unibet_return', 'Marathonbet_return']

dfAll, returns = Kelly(df, 0.1, x_columns, 1, 0)
#print(cum_returns)

x = np.arange(1, len(returns) + 1)
y = list(returns.array)
plt.plot(x, y, label = 'PERCENTAGE RETURN')
plt.show()
#####

Df_Game=gameStateData[[('home','daysSinceLastGame'),('away','daysSinceLastGame')]].fillna(0)
df_game.columns = df_game.columns.get_level_values(0) + '_' +  df_game.columns.get_level_values(1)

df_all = getDFAll([bettingOddsPCA, elo, perMetric,df_game], years, True)

df_all = getDFAll([bettingOddsPCA, elo,df_game], years, False)
df_all['stat']=

df_all['team1'] = df_all['team1'].apply(lambda d: 1 if d in ['UAT','ATL'] else 0)
df_all['team2'] = df_all['team2'].apply(lambda d: 1 if d in ['UAT','ATL'] else 0)

#df_all = getDFAll([bettingOddsPCA, elo, perMetric], years, True)

X = df_all
Y = getSignal().reindex(X.index)

X_train, X_test, Y_train, Y_test = splitTrainTestYear(X, Y, [2019,2021],[2022])
X_train, X_test, Y_train, Y_test = splitTrainTestYear(X, Y, [2018,2019],[2020])

#X_train, X_test, Y_train, Y_test = splitTrainTestYear(X, Y, 2019)
#X_train, X_test, Y_train, Y_test = splitTrainTest(X, Y, 0.2, 2022, True)

# PARAMATER TUNING
def findParamsXGB(X_train, Y_train):
    param_grid = {
        "n_estimators" : [50, 100, 150,200],
        "max_depth" : [1, 3, 5, 6],
        "learning_rate" : [0.005, 0.01, 0.02],
        #"min_child_weight" : [4, 5, 6]
        "min_child_weight" : [4,6]
    }

    grid = GridSearchCV(XGBClassifier(), param_grid, refit = True, verbose = 3)
    grid.fit(X_train, Y_train)
    print(grid.best_params_)
    print(grid.best_estimator_)
    return grid.best_estimator_

clf = findParamsXGB(X_train, Y_train)

clf = XGBClassifier(learning_rate = 0.01, max_depth = 6, n_estimators =150,min_child_weight = 4)

clf = XGBClassifier(learning_rate = 0.005, max_depth = 3, n_estimators =50,min_child_weight = 5)

clf = XGBClassifier(learning_rate = 0.01, max_depth = 3, min_child_weight = 4, n_estimators = 150)
#model = clf.fit(X_train, Y_train)
#y_p = model.predict_proba(X_test)[:,1]
clf = RandomForestClassifier(max_depth = 6, n_estimators =500,n_jobs=-1,  criterion =  "log_loss")                      

def xgboost(clf, X_train, Y_train, X_test, Y_test):
    model = clf.fit(X_train, Y_train)
    name = 'XGBOOST'
    calibrated_clf = CalibratedClassifierCV(clf,cv = 5)
    calibrated_clf.fit(X_train, Y_train)

    #calibrated_clf.predict_proba(X_test)
    #Y_pred_prob = clf.predict_proba(X_test)[:,1]
    #Y_train_pred_prob = clf.predict_proba(X_train)[:,1]

    Y_pred_prob = calibrated_clf.predict_proba(X_test)[:, 1]
    Y_train_pred_prob = calibrated_clf.predict_proba(X_train)[:, 1]

    Y_pred = [1 if p > 0.5 else 0 for p in Y_pred_prob]
    Y_train_pred = [1 if p > 0.5 else 0 for p in Y_train_pred_prob]

    acc = accuracy_score(Y_test, Y_pred)
    print("\nAccuracy of %s is %s"%(name, acc))
    #print(clf.feature_importances_)
    print(pd.DataFrame(data = list(model.feature_importances_), index = list(X_train.columns), columns = ["score"]).sort_values(by = "score", ascending = False).head(30))
    
    #cm = confusion_matrix(Y_test, Y_pred)/len(Y_pred) 

    print("Test  Accuracy : %.3f" %accuracy_score(Y_test, Y_pred))
    print("Train Accuracy : %.3f" %accuracy_score(Y_train, Y_train_pred))
    return Y_pred_prob,Y_train_pred_prob

Y_pred_prob ,Y_train_prob= xgboost(clf, X_train, Y_train, X_test, Y_test)
gameStateData = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', header = [0,1], index_col = 0)
#gameStateData = gameStateData[gameStateData.index.isin(Y_test.index)]['home']
gameStateData = gameStateData['home']



def getOddBreakdown(Y_pred_prob, Y_test):
    testOdds = bettingOddsAll[bettingOddsAll.index.isin(Y_test.index)]
    testOdds = testOdds.reindex(Y_test.index)
    for col in testOdds.columns:
        odd_preds = [1 if odd > 0.5 else 0 for odd in list(testOdds[col])]
        print("Odd Accuracy of {}".format(col) + " : %.3f"%accuracy_score(Y_test, odd_preds))
    #print("Confusion Matrix of %s is %s"%(name, cm))
    #pdb.set_trace()
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
df = getOddBreakdown(df.reset_index().set_index('index')['predProb'], df.reset_index().set_index('index')[['signal']])
df_train=getOddBreakdown(Y_train_prob, Y_train)

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

    
    df['adj_return'] = df.apply(lambda d: 1 if d['return'] < 1 else d['return'], axis = 1)
    print(df)
    return df 

def Kelly(df, alpha, x_columns, max_bet, n):
    df = getKellyBreakdown(df, alpha, x_columns, max_bet, n)
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
        

x_columns = ['bet365_return', 'William Hill_return', 'Pinnacle_return', 'Coolbet_return', 'Unibet_return', 'Marathonbet_return']

dfAll, returns = Kelly(df, 0.15, x_columns, 1, 0)
#print(cum_returns)

x = np.arange(1, len(returns) + 1)
y = list(returns.array)
plt.plot(x, y, label = 'PERCENTAGE RETURN')
plt.show()

#clf = XGBClassifier(learning_rate = 0.02, max_depth = 6, n_estimators = 150, min_child_weight = 6)

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
    
returnList, maxReturns = testSeeds(clf, 10)

print('median returns: {}'.format(statistics.median(returnList)))
print('median max returns: {}'.format(statistics.median(maxReturns)))
print('average returns: {}'.format(statistics.mean(returnList)))
print('average max returns: {}'.format(statistics.mean(maxReturns)))

print('successful returns: {}'.format((len([1 for i in returnList if i > 1]))/len(returnList)))
wonReturns = [item for item in returnList if item > 1]
lostReturns = [item for item in returnList if item < 1]
print('successful return average: {}'.format(statistics.mean(wonReturns)))
if statistics.mean(wonReturns) < 1:
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
