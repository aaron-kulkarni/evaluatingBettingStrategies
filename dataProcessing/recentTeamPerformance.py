import re
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
import re

from sportsipy.nba.teams import Teams
from sportsreference.nba.roster import Roster
from sportsreference.nba.roster import Player
from sportsreference.nba.schedule import Schedule
from sportsipy.nba.boxscore import Boxscore
from sportsipy.nba.boxscore import Boxscores


# filename = '../data/bettingOddsData/closing_betting_odds_2022_FIXED.csv'

def extract_lines(filename):
    startGameId = pd.read_csv(filename).head(1)['gameid'].iloc[0]
    endGameId = pd.read_csv(filename).tail(1)['gameid'].iloc[0]

    startDate = dt.datetime.strptime(startGameId[0:4] + ', ' + startGameId[4:6] + ', ' + startGameId[6:8], '%Y, %m, %d')
    endDate = dt.datetime.strptime(endGameId[0:4] + ', ' + endGameId[4:6] + ', ' + endGameId[6:8], '%Y, %m, %d')

    return startDate, endDate


def getRecentNGames(gameId, n, team):
    '''
    Obtains ids of the past n games (non inclusive) given the gameId of current game and team abbreviation
    
    '''
    if bool(re.match("^[\d]{9}[A-Z]{3}$", gameId)) == False:
        
        raise Exception('Issue with Game ID')

    
    if int(gameId[0:4]) == 2020: 
        if int(gameId[4:6].lstrip("0")) < 11: 
            teamSchedule = Schedule(team, int(gameId[0:4])).dataframe
        else:
            teamSchedule = Schedule(team, int(gameId[0:4]) + 1).dataframe
    else:
        if int(gameId[4:6].lstrip("0")) > 7: 
            teamSchedule = Schedule(team, int(gameId[0:4]) + 1).dataframe
        else:
            teamSchedule = Schedule(team, int(gameId[0:4])).dataframe

    teamSchedule.sort_values(by = 'datetime')
    currentDate = teamSchedule['datetime'].loc[gameId]
    temp = teamSchedule[(teamSchedule['datetime'] < currentDate)]

    if n > len(temp['datetime']):
        raise Exception('{} is too large'.format(n))

    gameIdList = list(temp['datetime'].tail(n).index)

    return gameIdList

def getTeamAveragePerformance(gameId, n, team):
    '''
    Returns a dataframe of average team performances in last n games
   
    '''

    return print('Not implemented yet')

def getPlayerAveragePerformance(gameId, n, playerId):
    '''
    Returns a dataframe of average player performances in last n games
   
    '''

    return print('Not implemented yet')
