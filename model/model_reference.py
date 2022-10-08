


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

df_elo_jin = pd.read_csv('/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/new.team.dat.jin.elo.csv')
df_elo_jin = pd.read_csv('/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/new.team.dat.jin.elo.all.csv')

df_elo_jin = df_elo_jin.set_index(['gameid','team'])
df_game = df_game.set_index(['index','team'])

df_game_adj = pd.concat([df_game,df_elo_jin[['team.elo.booker.lm', 'team.elo.booker.combined',
       'predict.prob.booker', 'predict.prob.combined','elo.court30.prob',
                                         'raptor.court30.prob']]],axis=1).reset_index().rename(columns={'level_0':'index'})
#df_game_bak = df_game.copy()
gameStateData = gameStateData['home']

####

x_var = ['index','team','odds_mean','streak', 'matchupWins','elo_prob', 'raptor_prob','raptor_elo', 'elo','numberOfGamesPlayed','player_change','home','team01','team.elo.booker.lm', 'team.elo.booker.combined', 'predict.prob.booker', 'predict.prob.combined','elo.court30.prob', 'raptor.court30.prob']
x_var = ['index','team','streak', 'matchupWins','elo_prob', 'raptor_prob','raptor_elo', 'elo','numberOfGamesPlayed','player_change','home','team01','team.elo.booker.lm', 'team.elo.booker.combined', 'predict.prob.booker', 'predict.prob.combined','elo.court30.prob', 'raptor.court30.prob']

x_var = ['index','team','streak', 'matchupWins','elo_prob', 'raptor_prob','raptor_elo','numberOfGamesPlayed','player_change','home','team01',
         #'team.elo.booker.lm', 'team.elo.booker.combined','elo',
         'predict.prob.booker', 'predict.prob.combined','elo.court30.prob', 'raptor.court30.prob']

#x_var = ['index','team','streak', 'matchupWins','elo_prob', 'raptor_prob','raptor_elo', 'elo','numberOfGamesPlayed','player_change','home','team01']

#x_var = ['index','team','odds_mean','streak', 'matchupWins','elo_prob', 'raptor_prob','raptor_elo', 'elo','numberOfGamesPlayed','player_change','home','team01']

y_var = ['index','team','stat']

train_yr = [2018,2019]
test_yr = [2020]
df_game_adj= df_game_adj[pd.isna(df_game_adj['team.elo.booker.lm'])==False]

def predict(train_yr,test_yr, _df):
    _pred_df = pd.DataFrame()
    for w in _df[_df['year'].isin(test_yr)]['weeks'].unique():
        X_train= df_game_adj[(df_game_adj['year'].isin(train_yr)) | ((df_game_adj['year'].isin(test_yr)) & (df_game_adj['weeks']< w))][x_var].set_index(['index','team'])
        X_test = df_game_adj[(df_game_adj['year'].isin(test_yr)) & (df_game_adj['weeks']==w)][x_var].set_index(['index','team'])
        Y_train= df_game_adj[(df_game_adj['year'].isin(train_yr)) | ((df_game_adj['year'].isin(test_yr)) & (df_game_adj['weeks']< w))][y_var].set_index(['index','team'])
        Y_test = df_game_adj[(df_game_adj['year'].isin(test_yr)) & (df_game_adj['weeks']==w)][y_var].set_index(['index','team'])

        clf = XGBClassifier(learning_rate = 0.01, max_depth = 6, n_estimators =300, min_child_weight = 4)

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
        df['predProb'] = df['predProb']/df_prob_sum['prob_scale']

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

dfAll, returns = Kelly(df, 0.3, x_columns, 1, 0)
#print(cum_returns)

x = np.arange(1, len(returns) + 1)
y = list(returns.array)
plt.plot(x, y, label = 'PERCENTAGE RETURN')
plt.show()
