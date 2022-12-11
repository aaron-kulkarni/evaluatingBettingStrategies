import sys
import pandas as pd
import numpy as np
sys.path.insert(0, '..')

from dataCollection.collectGameIds import *
from dataCollection.collectPlayerData import *

current_year = getYearHelper(dt.datetime.now().year, dt.datetime.now().month)

updateGameStateData()
update_team_stats(np.arange(2015, current_year + 1))

years = np.arange(2015, current_year + 1)

def updateGameStateDataAllf(years):
    df = pd.DataFrame()
    for year in years:
        df_current = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0)
        df = pd.concat([df, df_current], axis = 0)
    df.to_csv('../data/gameStats/game_state_data_ALL.csv')
    return df

#updateGameStateDataAllf(years)

def convert_to_datetime(years):
    for year in years:
        df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header=[0,1], index_col=0)
        df['gameState', 'datetime'] = pd.to_datetime(df['gameState', 'datetime'])
        df['gameState', 'endtime'] = pd.to_datetime(df['gameState', 'endtime'])
        df.to_csv('../data/gameStats/game_state_data_{}.csv'.format(year))
    df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', header=[0,1], index_col=0)
    df['gameState', 'datetime'] = pd.to_datetime(df['gameState', 'datetime'])
    df['gameState', 'endtime'] = pd.to_datetime(df['gameState', 'endtime'])
    df.to_csv('../data/gameStats/game_state_data_ALL.csv')
    return 
