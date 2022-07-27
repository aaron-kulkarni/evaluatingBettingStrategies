import numpy as np
import pandas as pd
from datetime import date
import datetime as dt
import matplotlib.pyplot as plt
import bs4 as bs
from urllib.request import urlopen
import requests
from lxml import html
from dateutil.relativedelta import relativedelta
import pdb
from sportsipy.nba.teams import Teams
from sportsreference.nba.roster import Roster
from sportsreference.nba.roster import Player

from sportsreference.nba.schedule import Schedule
from sportsipy.nba.boxscore import Boxscore
from sportsipy.nba.boxscore import Boxscores
import re


def getGameData(gameId):

    '''
    Returns a 1 by x array of relevant game data given a game Id.

    '''
    if bool(re.match("^[\d]{9}[A-Z]{3}$", gameId)):
        gameYear = gameId[0:4]
        gameMonth = gameId[4:6]
        gameDay = gameId[6:8]
        gameDate = gameYear + ', ' + gameMonth + ', ' + gameDay
        gamesToday = list(Boxscores(dt.datetime.strptime(gameDate, '%Y, %m, %d')).games.values())[0]
        a = next(item for item in gamesToday if item["boxscore"] == gameId)
        teamHome = a['home_abbr']
        teamAway = a['away_abbr']
        teams = [teamHome, teamAway]
    else:
        raise Exception('Issue with Game ID')

        
    # wins against team is computed with past 5 years data
    gameData = Boxscore(gameId)
    q1ScoreHome = gameData.summary['home'][0]
    q1ScoreAway = gameData.summary['away'][0]
    q2ScoreHome = gameData.summary['home'][1]
    q2ScoreAway = gameData.summary['away'][1]
    q3ScoreHome = gameData.summary['home'][2]
    q3ScoreAway = gameData.summary['away'][2]
    q4ScoreHome = gameData.summary['home'][3]
    q4ScoreAway = gameData.summary['away'][3]
    
    pointsHome = gameData.home_points
    pointsAway = gameData.away_points 
        
    if (int(gameYear) == 2020): #2020 was exception because covid messed up schedule
        if int(gameMonth.lstrip("0")) < 11: #converted gameMonth to int without leading 0. check month to find correct season
            teamHomeSchedule = Schedule(teamHome, int(gameYear)).dataframe
            teamAwaySchedule = Schedule(teamAway, int(gameYear)).dataframe
        else:
            teamHomeSchedule = Schedule(teamHome, int(gameYear) + 1).dataframe
            teamAwaySchedule = Schedule(teamAway, int(gameYear) + 1).dataframe
    else:
        if int(gameMonth.lstrip("0")) > 7: #games played after july are part of next season
            teamHomeSchedule = Schedule(teamHome, int(gameYear) + 1).dataframe
            teamAwaySchedule = Schedule(teamAway, int(gameYear) + 1).dataframe
        else:
            teamHomeSchedule = Schedule(teamHome, int(gameYear)).dataframe
            teamAwaySchedule = Schedule(teamAway, int(gameYear)).dataframe

    
    timeOfDay = teamHomeSchedule.loc[gameId][13]

    #since streak counts current game, look at streak based on last game
    streakHome = teamHomeSchedule.shift().loc[gameId][12]
    streakAway = teamAwaySchedule.shift().loc[gameId][12]

    #takes care of first game of season problem
    #also changed format from 'L 5' to -5
    streakHome = 0 if pd.isna(streakHome) else int(streakHome[-1:]) if streakHome.startswith('W') else -int(streakHome[-1:])
    streakAway = 0 if pd.isna(streakAway) else int(streakAway[-1:]) if streakAway.startswith('W') else -int(streakAway[-1:])

    teamHomeSchedule.sort_values(by='datetime')
    teamAwaySchedule.sort_values(by='datetime')

    prevHomeDate = teamHomeSchedule['datetime'].shift().loc[gameId]
    prevAwayDate = teamAwaySchedule['datetime'].shift().loc[gameId]
    currentdate = teamHomeSchedule.loc[gameId]['datetime']
    
    daysSinceLastGameHome = (currentdate - prevHomeDate).total_seconds() / 86400
    daysSinceLastGameAway = (currentdate - prevAwayDate).total_seconds() / 86400
    homePlayerRoster = [player.player_id for player in gameData.home_players]
    awayPlayerRoster = [player.player_id for player in gameData.away_players]

    '''
    Gets the name of coaches based on year of game day from https://www.basketball-reference.com/teams/.
    '''
    urlHome = f"https://www.basketball-reference.com/teams/{teamHome}/{gameDate[:4].lower()}.html"
    try:
        page = requests.get(urlHome)
        doc = html.fromstring(page.content)
        homeCoach = doc.xpath('//*[@id="meta"]/div[2]/p[2]/a/text()')
    except:
        raise Exception('Coach not found on basketball-reference.com for ' + Teams()(teamHome).name)

    urlAway = f"https://www.basketball-reference.com/teams/{teamAway}/{gameDate[:4].lower()}.html"

    try:
        page = requests.get(urlAway)
        doc2 = html.fromstring(page.content)
        awayCoach = doc2.xpath('//*[@id="meta"]/div[2]/p[2]/a/text()')
    except:
        raise Exception('Coach not found on basketball-reference.com for ' + Teams()(teamAway).name)


    location = gameData.location

    '''
    Find records of home and away teams by searching schedule arrays of home and 
    away team respectively
    '''
    
    # Calculating Home Record
    homeResults = teamHomeSchedule.result #only show results column
    homeResults = homeResults.loc[homeResults.index[0]:gameId] #only show up to current game row
    homeRecord = homeResults.value_counts(ascending = True) #sorts strings 'Win' and 'Loss' alphabetically which makes homeRecord[1] Wins and homeRecord[0] Losses
    try:
        homeWins = homeRecord[1]
    except:
        homeWins = 0 #sets value to 0 if no 'Win' are found in array
    try:
        homeLosses = homeRecord[0]
    except:
        homeLosses = 0 #sets value to 0 if no 'Loss' are found in array

    # Calculating Away Record (Same as Home Record)

    awayResults = teamAwaySchedule.result
    awayResults = awayResults.loc[awayResults.index[0]:gameId]
    awayRecord = awayResults.value_counts(ascending = True)
    try:
        awayWins = awayRecord[1]
    except:
        awayWins = 0
    try:
        awayLosses = awayRecord[0]
    except:
        awayLosses = 0


    #switches variables for away team. this looks wrong but for some reason makes code work :)
    tempVar = awayWins
    awayWins = awayLosses
    awayLosses = tempVar
    

    if pointsHome > pointsAway:
        homeWins-=1
        awayLosses-=1
    else:
        homeLosses-=1
        awayWins-=1    
    
    homeRecord = [homeWins, homeLosses]
    awayRecord = [awayWins, awayLosses] 


    tempDf = teamHomeSchedule.loc[teamHomeSchedule['opponent_abbr'] == teamHome]
    tempDf = tempDf.loc[teamHomeSchedule['datetime'] < currentdate]
    
    matchupWinsHome = tempDf.loc[teamHomeSchedule['result'] == 'Win'].shape[0]
    matchupWinsAway = tempDf.loc[teamHomeSchedule['result'] == 'Loss'].shape[0]
            
    gameData = [gameId, teamHome, teamAway, timeOfDay, location, q1ScoreHome, q2ScoreHome, q3ScoreHome, q4ScoreHome, pointsHome, streakHome, daysSinceLastGameHome, homePlayerRoster, homeRecord, matchupWinsHome, q1ScoreAway, q2ScoreAway, q3ScoreAway, q4ScoreAway, pointsAway, streakAway, daysSinceLastGameAway, awayPlayerRoster, awayRecord, matchupWinsAway] 
    
    return gameData

def getGameDataframe(startTime, endTime):
    '''
    startTime and endTime must be in format '%Y, %m, %d'
    
    '''

    allGames = Boxscores(dt.datetime.strptime(startTime, '%Y, %m, %d'), dt.datetime.strptime(endTime, '%Y, %m, %d')).games
    gameIdList = [] 
    for key in allGames.keys():
        for i in range(len(allGames[key])):
             gameIdList.append(allGames[key][i]['boxscore'])
    gameDataList = []
    for id in gameIdList:
        gameDataList.append(getGameData(id))

    df = pd.DataFrame(gameDataList, columns = ['gameId', 'teamHome', 'teamAway', 'timeOfDay', 'location', 'q1ScoreHome', 'q2ScoreHome', 'q3ScoreHome', 'q4ScoreHome', 'pointsHome', 'streakHome', 'daysSinceLastGameHome', 'homePlayerRoster', 'homeRecord', 'matchupWinsHome', 'q1ScoreAway', 'q2ScoreAway', 'q3ScoreAway', 'q4ScoreAway', 'pointsAway', 'streakAway', 'daysSinceLastGameAway', 'awayPlayerRoster', 'awayRecord', 'matchupWinsAway'])
    
    df.set_index('gameId', inplace = True)
    
    return df 