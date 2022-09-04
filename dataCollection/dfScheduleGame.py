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

# Check using the following regex:
# 20[12][\d][01][\d][0-3][\d]0[A-Z]{3},([A-Z]{3},){3}[0-1]?[\d]:[0-5][\d](a|p),".+",(([\d]+,){4}"?\[([\d]+,? ?)*\]"?,[\d]+,-?[\d]+.[\d.]*,"?\[('[a-z\d]+',? ?)*\]"?,"\[[\d]+, [\d]+\]",[\d]+,){2}(none|conference|division)\n

teamsDict = {
    "TOR": ["Eastern", "Atlantic"],
    "BOS": ["Eastern", "Atlantic"],
    "BRK": ["Eastern", "Atlantic"],
    "PHI": ["Eastern", "Atlantic"],
    "NYK": ["Eastern", "Atlantic"],
    "CLE": ["Eastern", "Central"],
    "CHI": ["Eastern", "Central"],
    "MIL": ["Eastern", "Central"],
    "IND": ["Eastern", "Central"],
    "DET": ["Eastern", "Central"],
    "ATL": ["Eastern", "Southeast"],
    "WAS": ["Eastern", "Southeast"],
    "MIA": ["Eastern", "Southeast"],
    "CHO": ["Eastern", "Southeast"],
    "ORL": ["Eastern", "Southeast"],
    "POR": ["Western", "Northwest"],
    "OKC": ["Western", "Northwest"],
    "UTA": ["Western", "Northwest"],
    "DEN": ["Western", "Northwest"],
    "MIN": ["Western", "Northwest"],
    "GSW": ["Western", "Pacific"],
    "LAC": ["Western", "Pacific"],
    "PHO": ["Western", "Pacific"],
    "SAC": ["Western", "Pacific"],
    "LAL": ["Western", "Pacific"],
    "HOU": ["Western", "Southwest"],
    "MEM": ["Western", "Southwest"],
    "SAS": ["Western", "Southwest"],
    "DAL": ["Western", "Southwest"],
    "NOP": ["Western", "Southwest"]
}


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

        
    gameData = Boxscore(gameId)
    summary = gameData.summary
    q1ScoreHome = summary['home'][0]
    q1ScoreAway = summary['away'][0]
    q2ScoreHome = summary['home'][1]
    q2ScoreAway = summary['away'][1]
    q3ScoreHome = summary['home'][2]
    q3ScoreAway = summary['away'][2]
    q4ScoreHome = summary['home'][3]
    q4ScoreAway = summary['away'][3]
    overtimeScoresHome = []
    overtimeScoresAway = []

    overtimePeriods = len(summary['home']) - 4
    for x in range(4, 4+overtimePeriods-1):
        overtimeScoresHome.append(summary['home'][x])
        overtimeScoresAway.append(summary['away'][x])
    
    pointsHome = gameData.home_points
    pointsAway = gameData.away_points

    if pointsHome > pointsAway:
        winner = teamHome
    else:
        winner = teamAway 
        
    if int(gameYear) == 2020: #2020 was exception because covid messed up schedule
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
    homeResults = teamHomeSchedule.result.shift()
    homeResults = homeResults.loc[homeResults.index[0]:gameId] 
    homeRecord = homeResults.value_counts(ascending = True)
    try:
        homeWins = homeRecord['Win']
    except:
        homeWins = 0 
    try:
        homeLosses = homeRecord['Loss']
    except:
        homeLosses = 0 


    awayResults = teamAwaySchedule.result.shift()
    awayResults = awayResults.loc[awayResults.index[0]:gameId]
    awayRecord = awayResults.value_counts(ascending = True)
    try:
        awayWins = awayRecord['Win']
    except:
        awayWins = 0
    try:
        awayLosses = awayRecord['Loss']
    except:
        awayLosses = 0

    homeRecord = [homeWins, homeLosses]
    awayRecord = [awayWins, awayLosses] 

    tempDf = teamHomeSchedule.loc[teamHomeSchedule['opponent_abbr'] == teamAway]
    tempDf = tempDf.loc[teamHomeSchedule['datetime'] < currentdate]
    
    matchupWinsHome = tempDf.loc[teamHomeSchedule['result'] == 'Win'].shape[0]
    matchupWinsAway = tempDf.loc[teamHomeSchedule['result'] == 'Loss'].shape[0]

    if teamsDict[teamHome] == teamsDict[teamAway]:
        rivalry = 'division'
    elif teamsDict[teamHome][0] == teamsDict[teamAway][0]:
        rivalry = 'conference'
    else:
        rivalry = 'none'
            
    gameData = [gameId, winner, teamHome, teamAway, timeOfDay, location, q1ScoreHome, q2ScoreHome, q3ScoreHome, q4ScoreHome, overtimeScoresHome, pointsHome, streakHome, daysSinceLastGameHome, homePlayerRoster, homeRecord, matchupWinsHome, q1ScoreAway, q2ScoreAway, q3ScoreAway, q4ScoreAway, overtimeScoresAway, pointsAway, streakAway, daysSinceLastGameAway, awayPlayerRoster, awayRecord, matchupWinsAway, rivalry] 
    
    return gameData

def getGameDataframe(startTime, endTime):
    '''
    startTime and endTime must be in format '%m-%d-%Y'
    
    '''

    allGames = Boxscores(dt.datetime.strptime(startTime, '%m-%d-%Y'), dt.datetime.strptime(endTime, '%m-%d-%Y')).games
    gameIdList = [] 
    for key in allGames.keys():
        for i in range(len(allGames[key])):
             gameIdList.append(allGames[key][i]['boxscore'])
    gameDataList = []
    for id in gameIdList:
        gameDataList.append(getGameData(id))

    df = pd.DataFrame(gameDataList, columns = ['gameId', 'winner', 'teamHome', 'teamAway', 'timeOfDay', 'location', 'q1ScoreHome', 'q2ScoreHome', 'q3ScoreHome', 'q4ScoreHome', 'overtimeScoresHome', 
    'pointsHome', 'streakHome', 'daysSinceLastGameHome', 'homePlayerRoster', 'homeRecord', 'matchupWinsHome', 'q1ScoreAway', 'q2ScoreAway', 'q3ScoreAway', 'q4ScoreAway', 
    'overtimeScoresAway', 'pointsAway', 'streakAway', 'daysSinceLastGameAway', 'awayPlayerRoster', 'awayRecord', 'matchupWinsAway', 'rivalry'])
    
    df.set_index('gameId', inplace = True)
    
    return df 

def getGameStatYear(year):
    fileLocation = '/Users/jasonli/Projects/evaluatingBettingStrategies/data/gameStats/game_data_player_stats_{}_clean.csv'.format(year)

    startDate = str(extract_lines(fileLocation)[0])[0:10]
    endDate = str(extract_lines(fileLocation)[1])[0:10]

    df = getGameDataframe(startDate, endDate)
    return df
    
def getNumberGamesPlayed(team, year, game_id):
    index = getTeamGameIds(team, year).index(game_id)
    return index

def getTeamGameIds(team, year):
    homeTeamSchedule, awayTeamSchedule = getTeamSchedule(team, year)
    teamSchedule = pd.concat([homeTeamSchedule, awayTeamSchedule], axis=0)
    teamSchedule = teamSchedule.sort_index(ascending=True)
    return list(teamSchedule.index)
def getTeamSchedule(team, year):
    df = pd.DataFrame(pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1]))

    dfHome = df[df['gameState']['teamHome'] == team]
    dfAway = df[df['gameState']['teamAway'] == team]
    return dfHome, dfAway


def addNumberOfGamesPlayed(year):
    df = pd.read_csvdef getGameStatYear(year):
    fileLocation = '/Users/jasonli/Projects/evaluatingBettingStrategies/data/gameStats/game_data_player_stats_{}_clean.csv'.format(year)

    startDate = str(extract_lines(fileLocation)[0])[0:10]
    endDate = str(extract_lines(fileLocation)[1])[0:10]

    df = getGameDataframe(startDate, endDate)
    return df
    
def getNumberGamesPlayed(team, year, game_id):
    index = getTeamGameIds(team, year).index(game_id)
    return index

def getTeamGameIds(team, year):
    homeTeamSchedule, awayTeamSchedule = getTeamSchedule(team, year)
    teamSchedule = pd.concat([homeTeamSchedule, awayTeamSchedule], axis=0)
    teamSchedule = teamSchedule.sort_index(ascending=True)
    return list(teamSchedule.index)

def getNumberGamesPlayedDF(year):
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1])
    df['gameState', 'index'] = df.index.to_series()
    df['home', 'numberOfGamesPlayed'] = df.apply(lambda d: getNumberGamesPlayed(d['gameState', 'teamHome'], year, d['gameState', 'index']), axis = 1)
    df['away', 'numberOfGamesPlayed'] = df.apply(lambda d: getNumberGamesPlayed(d['gameState', 'teamAway'], year, d['gameState', 'index']), axis = 1)
    df.drop('index', level = 1, inplace = True, axis = 1)

    return df

years = np.arange(2015, 2023)
for year in years:
    getNumberGamesPlayedDF(year).to_csv('../data/gameStats/1game_state_data_{}.csv'.format(year))
