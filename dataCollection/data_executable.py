import sys
sys.path.insert(0, '..')
from collectPlayerData import *

#updateGameStateData()

update_team_stats(np.arange(2015, 2024))



def updateGameStateDataAll(years):
    df = pd.DataFrame()
    for year in years:
        df_current = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0)
        df = pd.concat([df, df_current], axis = 0)
    return df

years=np.arange(2015,2024)
for year in years:
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header=[0,1], index_col=0)
    df['gameState', 'datetime'] = pd.to_datetime(df['gameState', 'datetime'])
    df['gameState', 'endtime'] = pd.to_datetime(df['gameState', 'endtime'])
    df.to_csv('../data/gameStats/game_state_data_{}.csv'.format(year))

updateGameStateDataAllf(years)

# df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', header=[0,1], index_col=0)
# df['gameState', 'datetime'] = pd.to_datetime(df['gameState', 'datetime'])
# df['gameState', 'endtime'] = pd.to_datetime(df['gameState', 'endtime'])
# df.to_csv('../data/gameStats/game_state_data_ALL.csv')
