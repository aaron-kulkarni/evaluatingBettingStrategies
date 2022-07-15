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

#listGames = 0
#listGames = Boxscores(date(2021, 10, 19), date(2021, 10, 19)).games

#i = 0
#gameDay = '10-19-2021'
#game = listGames[gameDay][i]

#gameIdList = [game['boxscore'], game['boxscore']]
#gameId = game['boxscore']


def getGameData(gameId):
    df = pd.DataFrame()

    teams, gameIdList, q1Score, q2Score, q3Score, q4Score, points, location, daysSinceLastGame, gamesInPastWeek, timeOfDay, roster, coach, record, winsAgainstTeam, streak = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    if bool(re.match("^[\d]{9}[A-Z]{3}$", gameId)):
        gameDate = gameId[0:4] + ', ' + gameId[4:6] + ', ' + gameId[6:8]
        gamesToday = list(Boxscores(dt.datetime.strptime(gameDate, '%Y, %m, %d')).games.values())[0]
        a = next(item for item in gamesToday if item["boxscore"] == gameId)
        home_abbr = a['home_abbr']
        away_abbr = a['away_abbr']
        teams = [home_abbr, away_abbr]
    else:
        raise Exception('Issue with Game ID')
    # wins against team is computed with past 5 years data
    gameData = Boxscore(gameId)
    
    gameIdList = [gameId, gameId] 
    q1Score = [gameData.summary['home'][0], gameData.summary['away'][0]]
    q2Score = [gameData.summary['home'][1], gameData.summary['away'][1]]
    q3Score = [gameData.summary['home'][2], gameData.summary['away'][2]]
    q4Score = [gameData.summary['home'][3], gameData.summary['away'][3]]
    points = [game['home_score'], game['away_score']]

    teamHomeSchedule = Schedule(teams[0], year=2022).dataframe
    teamAwaySchedule = Schedule(teams[1], year=2022).dataframe
    timeOfDay = [teamHomeSchedule.loc[gameId][13], teamAwaySchedule.loc[gameId][13]]
    streak = [teamHomeSchedule.loc[gameId][12], teamAwaySchedule.loc[gameId][12]]
    # caution: streak might be included with the current loss/win

    df['teams'] = teams
    df['gameId'] = gameIdList

    teamHomeSchedule.sort_values(by='datetime')
    teamAwaySchedule.sort_values(by='datetime')

    daysSinceLastGame = []
    gamesInPastWeek = []

    prevdate = teamHomeSchedule['datetime'].shift().loc[gameId]
    currentdate = teamHomeSchedule.loc[gameId]['datetime']
    daysSinceLastGame.append((currentdate - prevdate).total_seconds() / 86400)

    temp = teamHomeSchedule[(teamHomeSchedule['datetime'] - currentdate).dt.total_seconds() < 86400 * 7]
    temp = temp[temp['datetime'] < currentdate]
    gamesInPastWeek.append(temp.shape[0])

    prevdate = teamAwaySchedule['datetime'].shift().loc[gameIdList[0]]
    daysSinceLastGame.append((currentdate - prevdate).total_seconds() / 86400)

    temp = teamAwaySchedule[(teamAwaySchedule['datetime'] - currentdate).dt.total_seconds() < 86400 * 7]
    temp = temp[temp['datetime'] < currentdate]
    gamesInPastWeek.append(temp.shape[0])

    roster = [gameData.home_players, gameData.away_players]
    
    homePlayerRoster = [player.player_id for player in gameData.home_players]
    awayPlayerRoster = [player.player_id for player in gameData.away_players]
    roster = [homePlayerRoster, awayPlayerRoster]

    '''
    Gets the name of coaches based on year of game day from https://www.basketball-reference.com/teams/.
    '''
    urlHome = f"https://www.basketball-reference.com/teams/{home_abbr}/{gameDate[:4].lower()}.html"
    try:
        #print(urlHome)
        page = requests.get(urlHome)
        doc = html.fromstring(page.content)
        homeCoach = doc.xpath('//*[@id="meta"]/div[2]/p[2]/a/text()')
    except:
        raise Exception('Coach not found on basketball-reference.com for ' + Teams()(home_abbr).name)

    urlAway = f"https://www.basketball-reference.com/teams/{away_abbr}/{gameDate[:4].lower()}.html"

    try:
        #print(urlAway)
        page = requests.get(urlAway)
        doc2 = html.fromstring(page.content)
        awayCoach = doc2.xpath('//*[@id="meta"]/div[2]/p[2]/a/text()')
    except:
        raise Exception('Coach not found on basketball-reference.com for ' + Teams()(away_abbr).name)

    coach = [homeCoach[0], awayCoach[0]]

    location = [gameData.location, gameData.location]

    record = [str(gameData.home_wins) + " - " + str(gameData.home_losses), str(gameData.away_wins) + " - " + str(gameData.away_losses)]

    tempDf = teamHomeSchedule.loc[teamHomeSchedule['opponent_abbr'] == game['away_abbr']]
    tempDf = tempDf.loc[teamHomeSchedule['datetime'] < currentdate]
    homeTeamMatchupWins = tempDf.loc[teamHomeSchedule['result'] == 'Win'].shape[0]
    homeTeamMatchupLosses = tempDf.loc[teamHomeSchedule['result'] == 'Loss'].shape[0]

    winsAgainstTeam = [homeTeamMatchupWins, homeTeamMatchupLosses]

    df['teams'] = teams
    df['gameId'] = gameIdList
    df['q1Score'] = q1Score
    df['q2Score'] = q2Score
    df['q3Score'] = q3Score
    df['q4Score'] = q4Score
    df['points'] = points
    df['location'] = location
    df['daysSinceLastGame'] = daysSinceLastGame
    df['gamesInPastWeek'] = gamesInPastWeek
    df['timeOfDay'] = timeOfDay
    df['roster'] = roster
    # df['coach'] = coach
    df['record'] = record
    df['winsAgainstTeam'] = winsAgainstTeam

    df.set_index(['teams'])
    return df
