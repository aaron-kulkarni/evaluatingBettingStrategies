import os
import re
import requests
import numpy as np
import pandas as pd
from datetime import date
import datetime as dt 
from sportsipy.nba.teams import Teams
from sportsipy.nba.boxscore import Boxscores

#filename = '/Users/jasonli/Projects/evaluatingBettingStrategies/data/bettingOddsData/closing_betting_odds_2022.csv'

def extract_lines(filename):
    startGameId = pd.read_csv(filename).head(1)['gameid'].iloc[0]
    endGameId = pd.read_csv(filename).tail(1)['gameid'].iloc[0]
    
    startDate = dt.datetime.strptime(startGameId[0:4]+', '+startGameId[4:6]+', '+startGameId[6:8], '%Y, %m, %d')
    endDate = dt.datetime.strptime(endGameId[0:4]+', '+endGameId[4:6]+', '+endGameId[6:8], '%Y, %m, %d')
    
    return startDate, endDate

def convertBettingOdds(filename):
    '''
    function does following:
    1. Removes games that are not in regular season
    2. Adds GameID
    3. Checks if all games in betting odds file are in regular season and vice versa
    '''

    year = re.findall('[0-9]+', filename)[0]
    
    teamDict = {}
    for team in Teams():
        teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
        teamDict[team.name] = teamAbbr
        
    df = pd.read_csv(filename, sep = ';')
    df = df[df.Home_id.isin(list(teamDict.keys()))]
    df.drop('Season', inplace = True, axis = 1)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by = 'Date', ascending = True, inplace = True)

    # IMPORTANT: MUST CHANGE FOR SPECIFIC FILE WHERE CLEAN DATA FOR CORRESPONDING YEAR IS LOCATED 
    fileLocation = '/Users/jasonli/Projects/evaluatingBettingStrategies/data/gameStats/game_data_player_stats_{}_clean.csv'.format(year)

    startDate = str(extract_lines(fileLocation)[0])[0:10]
    endDate = str(extract_lines(fileLocation)[1])[0:10]
    
    df = df[(df['Date'] >= startDate) & (df['Date'] <= endDate)]
    df['game_id'] = df.apply(lambda d: str(d['Date'])[0:10].replace('-','') + '0' + teamDict[d['Home_id']], axis = 1)
    
   
    gameIdList = [] 
    allGames = Boxscores(dt.datetime.strptime(startDate, '%Y-%m-%d'), dt.datetime.strptime(endDate, '%Y-%m-%d')).games
    for key in allGames.keys():
        for i in range(len(allGames[key])):
            gameIdList.append(allGames[key][i]['boxscore'])
    
    if set(gameIdList) != set(df['game_id'].unique().tolist()):
        print('Issue with GameID')
        
    return df 

def cleanBettingOdds(filename):


years = np.arrange(2015, 2022)
for year in years:
    convertBettingOdds('--')
    clearnBettingOdds('--')
    

