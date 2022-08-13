import re
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
import re
import sys

def fixGameStateData(filename):
    year = re.findall('[0-9]+', filename)[0]
    gameState = pd.read_csv(filename, index_col = 0)
    df = pd.read_csv('../data/bettingOddsData/adj_prob_{}.csv'.format(year), index_col = 0, header = [0,1])
    for col in gameState.columns[0:5]:
        df['gameState', '{}'.format(col)] = gameState[col]
    for col in gameState.columns[5:16]:
        colN = col.replace('Home', '')
        colN = colN.replace('home', '')
        colN = colN[0].lower() + colN[1:]
        df['home',  '{}'.format(colN)] = gameState[col]
    for col in gameState.columns[16:27]:
        colN = col.replace('away', '')
        colN = colN.replace('Away', '')
        colN = colN[0].lower() + colN[1:]
        df['away',  '{}'.format(colN)] = gameState[col]
    df['gameState', 'rivalry'] = gameState['rivalry']
    df.drop(['homeProb', 'awayProb', 'homeProbAdj', 'awayProbAdj'], inplace = True, axis = 1)
    return df

#years = np.arange(2015, 2023)
#for year in years:
    #fixGameStateData('../data/gameStats/game_state_data_{}.csv'.format(year)).to_csv('game_state_data_new_{}.csv'.format(year))
        
def getTeamSchedule(team, year):
    gameState = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col = 0, header = [0,1])
    dfHome = gameState[gameState['gameState']['teamHome'] == team]
    dfAway = gameState[gameState['gameState']['teamAway'] == team]
    return dfHome, dfAway
    

def getTeamPerformance(team, year):
    dfHome = getTeamSchedule(team, year)[0]
    dfAway = getTeamSchedule(team, year)[1]
    
    adjProb = pd.read_csv('../data/bettingOddsData/adj_prob_{}.csv'.format(year), index_col = 0, header = [0,1])

    
    
