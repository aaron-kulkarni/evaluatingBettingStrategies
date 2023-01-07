import imp
import os

file = imp.load_source("model.py", os.path.abspath('model.py'))

for name in dir(file):
    if not name.startswith("_"):
        globals()[name] = getattr(file, name)

init_ray()

train_years = [2021, 2022]
test_year = 2023
train_window = 2
test_games = getNextGames()

odds_df = select_attributes(5).select_col_odds(['Marathonbet (%)', '1xBet (%)', 'Pinnacle (%)', 'Unibet (%)', 'William Hill (%)'])
mlval_df = select_attributes(5).select_mlval(['team.elo.booker.lm', 'opp.elo.booker.lm', 'team.elo.booker.combined', 'opp.elo.booker.combined', 'elo.prob', 'predict.prob.booker', 'predict.prob.combined', 'elo.court30.prob', 'raptor.court30.prob', 'booker_odds.Pinnacle'])
per_metric_df = select_attributes(5).select_col_per(['pm_elo_prob1','pm_odd_prob','pm_raptor_prob1','pm_6_elo_prob1','pm_6_odd_prob','pm_6_raptor_prob1'])
elo_df = select_attributes(5).select_elo(['elo_prob', 'raptor_prob'])
game_df = select_attributes(5).select_col_game(['streak', 'numberOfGamesPlayed', 'daysSinceLastGame', 'matchupWins', 'win_per'])
team_stat_df = select_attributes(5).select_col_team(['3P%', 'Ortg', 'Drtg', 'TOV%', 'eFG%', 'PTS'])
data_list = [odds_df, mlval_df, per_metric_df, elo_df, game_df, team_stat_df]
check_dataframe_NaN(data_list, test_games)

X_train, X_test, Y_train, Y_test = split_data(data_list, train_years, test_year, True)
X_train_, X_test_, Y_train_, Y_test_ = split_data_test_games(data_list, train_window, test_games, True, True)
clf = XGBClassifier(learning_rate = 0.02, max_depth = 4, min_child_weight = 6, n_estimators = 150)

Y_pred_prob = xgboost(clf, X_train, Y_train, X_test, Y_test, 10)
Y_pred_prob_ = xgboost(clf, X_train_, Y_train_, X_test_, Y_test_, 10)
x_columns = ['bet365_return', 'Unibet_return']

#data_params = [data_list, train_window, X_test.index.get_level_values(0).unique(), True, True]
#Y_pred_prob_all = iterative_training(data_params, clf, 10)

df = perform_bet(Y_pred_prob, x_columns, 0.15, odds_df)
df_test = perform_bet(Y_pred_prob_, x_columns, 0.15, odds_df)
#df_ = perform_bet(Y_pred_prob_all, x_columns, 0.15, odds_df)

pool = 31614.45

df_bet = write_day_trade(pool, df[df.index.isin(getGamesToday())])
print(df_bet)
print(df[df.index.isin(getGamesToday())])
write_day_trade(pool, df_test)

def kelly_simulation(df):
    rand_floats = np.random.rand(len(df))
    df['signal'] = np.where(rand_floats <= df['Y_prob'], 1, 0)
    df['return'] = df.apply(lambda d: d['p_return'] if d['signal'] == 1 else 0, axis=1)
    df['adj_return'] = df.apply(lambda d: d['return'] * d['per_bet'], axis=1)
    index = sortAllDates(df.index)
    per_bet, returns = convert_returns(df['per_bet'], index), convert_returns(df['adj_return'], index)
    returns.rename(columns = {'adj_return' : 'return'}, inplace = True)
    dict_returns = pd.concat([per_bet, returns], axis = 1).T.to_dict()
    df_all = pd.DataFrame(find_total(dict_returns)).T
    df.drop(['signal'], axis=1, inplace=True)
    return df_all

def run_simulation(df, n):
    return_array = []
    for i in range(0, n):
        return_val = kelly_simulation(df)['total'][-1]
        return_array.append(return_val)
    return return_array
    
if __name__ == "__main__":
    values = run_simulation(df, 1000)
    plt.hist(values, bins = 100)
    plt.show()
    
    df_all = backtesting_returns(df)
    df_all_ = backtesting_returns(df_)

    returns = df_all['total']
    returns = df_all_['total']
    get_odd_acc(Y_pred_prob, Y_test, odds_df)

    x = np.arange(1, len(returns) + 1)
    y = list(returns.array)
    plt.plot(x, y, label = 'PERCENTAGE RETURN')
    plt.show()

    x, y = plot_day_increments(df_all)
    plt.plot(x, y, label = 'PERCENTAGE RETURN')
    plt.show()

    ray.shutdown()
