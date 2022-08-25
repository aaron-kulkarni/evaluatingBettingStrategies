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

from teamPerformance import teamAverageHelper, playerAverageHelper, opponentAverageHelper, getTeamSchedule

def getFirstGame(team, year):
    teamSchedule = Schedule(team, year).dataframe
    teamSchedule.sort_values(by = 'datetime')
    return teamSchedule.index[0] 
    
    
def getRecentNGames(gameId, n, team):
    '''
    Obtains ids of the past n games (non inclusive) given the gameId of current game and team abbreviation
    
    '''
    if n <= 0:
        raise Exception('N parameter must be greater than 0')
    
    if bool(re.match("^[\d]{9}[A-Z]{3}$", gameId)) == False:
        
        raise Exception('Issue with Game ID')
    
    if int(gameId[0:4]) == 2020: 
        if int(gameId[4:6].lstrip("0")) < 11: 
            year = 2020
        else:
            year = 2021
    else:
        if int(gameId[4:6].lstrip("0")) > 7: 
            year = int(gameId[0:4]) + 1
        else:
            year = int(gameId[0:4])

    teamScheduleHome, teamScheduleAway = getTeamSchedule(team, year)
    teamScheduleHome = teamScheduleHome.loc[:gameId]
    teamScheduleAway = teamScheduleAway.loc[:gameId]
    teamScheduleTotal = teamScheduleHome.append(teamScheduleAway)
    teamScheduleTotal.sort_index(inplace=True)
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
    if int(gameId[0:4]) == 2020: 
        if int(gameId[4:6].lstrip("0")) < 11: 
            year = int(gameId[0:4])
        else:
            year = int(gameId[0:4]) + 1
    else:
        if int(gameId[4:6].lstrip("0")) > 7: 
            year = int(gameId[0:4]) + 1
        else:
            year = int(gameId[0:4])

    gameIdList = getRecentNGames(gameId, n, team)

    df1, df2 = teamAverageHelper(team, year)
    
    df1 = df1[df1.index.isin(gameIdList)]
    df2 = df2[df2.index.isin(gameIdList)]

    df = df1['home'].append(df2['away'])

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

    if int(gameId[0:4]) == 2020: 
        if int(gameId[4:6].lstrip("0")) < 11: 
            year = int(gameId[0:4])
        else:
            year = int(gameId[0:4]) + 1
    else:
        if int(gameId[4:6].lstrip("0")) > 7: 
            year = int(gameId[0:4]) + 1
        else:
            year = int(gameId[0:4])

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

    if int(gameId[0:4]) == 2020: 
        if int(gameId[4:6].lstrip("0")) < 11: 
            year = int(gameId[0:4])
        else:
            year = int(gameId[0:4]) + 1
    else:
        if int(gameId[4:6].lstrip("0")) > 7: 
            year = int(gameId[0:4]) + 1
        else:
            year = int(gameId[0:4])

    df1, df2 = opponentAverageHelper(team, year)

    df1 = df1[df1.index.isin(gameIdList)]
    df2 = df2[df2.index.isin(gameIdList)]

    df = df1['home'].append(df2['away'])

    df.loc[gameId] = df.mean()

    df['teamAbbr'][n] = team
    
    return df.loc[gameId]

def getTeamPerformanceDF(year, n):
    teamDF = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0)
    homeTeam = teamDF['gameState']['teamHome']
    df = pd.DataFrame()
    gameIdList = homeTeam.index
    for gameId in gameIdList:
        team = homeTeam.loc[gameId]
        teamTotalStats = pd.concat([getTeamAveragePerformance(gameId, n, team), getOpponentAveragePerformance(gameId, n, team)], axis = 0, keys = ['home', 'opp'], join = 'inner')
        teamTotalStats = teamTotalStats.to_frame().T
        df = pd.concat([df, teamTotalStats], axis = 0)
    return df

def cleanTeamPerformanceDF(year):
    df = pd.read_csv('../data/averageTeamData/average_team_per_5_{}'.format(year), header = [0,1], index_col = 0)
    df.set_index[('', 'gameId')]
    df.drop('gameId', axis = 1, level = 1)
    return df

#year = np.arrange(2015, 2023)
#for year in years:
#    cleanTeamPerformanceDF(year).to_csv('average_team_per_5_clean_{}'.format(year))


        
        
        
        
    

    
