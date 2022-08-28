import re
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
import sys

#sys.path.insert(1, '')

from sportsipy.nba.teams import Teams
from sportsreference.nba.roster import Roster
from sportsreference.nba.roster import Player
from sportsreference.nba.schedule import Schedule
from sportsipy.nba.boxscore import Boxscore
from sportsipy.nba.boxscore import Boxscores

from teamPerformance import teamAverageHelper, playerAverageHelper, opponentAverageHelper, getTeamSchedule, getYearFromId, getTeamGameIds
 
def getRecentNGames(gameId, n, team):
    '''
    Obtains ids of the past n games (non inclusive) given the gameId of current game and team abbreviation
    
    '''
    if n <= 0:
        raise Exception('N parameter must be greater than 0')
    
    if bool(re.match("^[\d]{9}[A-Z]{3}$", gameId)) == False:
        
        raise Exception('Issue with Game ID')
    
    year = getYearFromId(gameId)
    gameIdList = getTeamGameIds(team, year)
    index = gameIdList.index(gameId)
    gameIdList = gameIdList[index-n:index]

    return gameIdList 


    teamScheduleTotal.sort_index(inplace = True)
    return list(teamScheduleTotal.tail(n).index)

def getTeamAveragePerformance(gameId, n, team):
    '''
    Returns a row of data of average team performances in last n games
   
    '''
    try:
         gameIdList = getRecentNGames(gameId, n, team)
    except:
        s = pd.Series('NaN', index=['teamAbbr','MP','FG','FGA','FG%','3P','3PA','3P%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','TS%','eFG%','3pAr','FTr','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%','USG%','Ortg','Drtg','poss','pace','poss_per_poss','ass_per_poss'])
        s.name = gameId
        return s

    #trying to only return the team stats of the team that we are asking for, rather than the team plus their opponents
    year = getYearFromId(gameId)

    gameIdList = getRecentNGames(gameId, n, team)

    df1, df2 = teamAverageHelper(team, year)
    
    df1 = df1[df1.index.isin(gameIdList)]
    df2 = df2[df2.index.isin(gameIdList)]

    df = pd.concat([df1['home'], df2['home']], axis = 0)

    df.loc[gameId] = df.mean()

    df['teamAbbr'] = team
    
    return df.loc[gameId]

def getPlayerAveragePerformance(gameId, n, team, playerId):
    '''
    Returns a row of data of average player performances in last n games
    
    '''
    try:
         gameIdList = getRecentNGames(gameId, n, team)
    except:
        s = pd.Series('NaN', index=['teamAbbr','MP','FG','FGA','FG%','3P','3PA','3P%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','TS%','eFG%','3pAr','FTr','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%','USG%','Ortg','Drtg','poss','pace','poss_per_poss','ass_per_poss'])
        s.name = gameId
        return s

    year = getYearFromId(gameId)

    gameIdList = getRecentNGames(gameId, n, team)

    dfPlayer = playerAverageHelper(playerId, year)

    dfPlayer = dfPlayer[dfPlayer['gameid'].isin(gameIdList)]

    dfPlayer['MP'] = dfPlayer.apply(lambda d: round(float(d['MP'][0:2]) + (float(d['MP'][3:5])/60), 2), axis = 1)

    storedName = dfPlayer.iloc[1]['Name']
    
    dfPlayer.loc['mean'] = dfPlayer.mean()
    dfPlayer['playerid'] = playerId
    dfPlayer['Name'] = storedName
    dfPlayer.at['mean','gameid'] = gameId

    return dfPlayer.loc['mean']

def getOpponentAveragePerformance(gameId, n, team):

    '''
    Returns a row of data of average opposing team performances in last n games
   
    '''
    try:
         gameIdList = getRecentNGames(gameId, n, team)
    except:
        s = pd.Series('NaN', index=['teamAbbr','MP','FG','FGA','FG%','3P','3PA','3P%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','TS%','eFG%','3pAr','FTr','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%','USG%','Ortg','Drtg','poss','pace','poss_per_poss','ass_per_poss'])
        s.name = gameId
        return s

    year = getYearFromId(gameId)

    df1, df2 = opponentAverageHelper(team, year)

    df1 = df1[df1.index.isin(gameIdList)]
    df2 = df2[df2.index.isin(gameIdList)]

    df = pd.concat([df1['home'], df2['home']], axis = 0)

    df.loc[gameId] = df.mean()

    df['teamAbbr'] = team
    
    return df.loc[gameId]

def getTeamPerformanceDF(year, n, home = True):
    teamDF = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0)
    homeTeam = teamDF['gameState']['teamHome']
    awayTeam = teamDF['gameState']['teamAway']
    if home == True:
        teamDF = homeTeam
    else:
        teamDF = awayTeam
    df = pd.DataFrame()
    gameIdList = homeTeam.index
    for gameId in gameIdList:
        team = teamDF.loc[gameId]
        teamTotalStats = pd.concat([getTeamAveragePerformance(gameId, n, team), getOpponentAveragePerformance(gameId, n, team)], axis = 0, keys = ['home', 'opp'], join = 'inner')
        teamTotalStats = teamTotalStats.to_frame().T
        df = pd.concat([df, teamTotalStats], axis = 0)
    return df



years = np.arange(2015, 2019)
for year in years:
    getTeamPerformanceDF(year, 10, True).to_csv('average_team_per_10_{}.csv'.format(year))
    getTeamPerformanceDF(year, 10, False).to_csv('average_away_per_10_{}.csv'.format(year))

