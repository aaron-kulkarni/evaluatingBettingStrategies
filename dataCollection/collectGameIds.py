import pandas as pd
import numpy as np
from dataCollection.collectGameData import *
import bs4 as bs
from urllib.request import urlopen
import re
from dateutil import parser
import requests
import time

import sys
sys.path.insert(0, '..')
from utils.utils import *
from dataProcessing.progress.rosterDict import scrapeRoster

monthDict = {
    "01": "january",
    "02": "february",
    "03": "march",
    "04": "april",
    "05": "may",
    "06": "june",
    "07": "july",
    "08": "august",
    "09": "september",
    "10": "october",
    "11": "november",
    "12": "december"
}

def getStaticMonthData(month, year):
    url = 'https://www.basketball-reference.com/leagues/NBA_{}_games-{}.html'.format(year, month)
    soup = bs.BeautifulSoup(urlopen(url), features='lxml')
    rows = [p for p in soup.find('div', {'id': 'div_schedule'}).findAll('tr')]
    rowList = []
    for row in rows:
        rowList.append([td for td in row.findAll(['td', 'th'])])
    gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList, neutralList = [], [], [], [], [], []
    for i in range(1, len(rowList)):
        dateTime = parser.parse(rowList[i][0].getText())
        aChildrenH = str(rowList[i][4].findChildren('a'))
        homeTeam = aChildrenH.partition('teams/')[2][:3]
        aChildrenA = str(rowList[i][2].findChildren('a'))
        awayTeam = aChildrenA.partition('teams/')[2][:3]

        gameId = '{}0{}'.format(dateTime.strftime("%Y%m%d"), homeTeam)
        dateTime = convDateTime(gameId, rowList[i][1].getText())
        location = rowList[i][9].getText()
        if rowList[i][10].getText() == '':
            neutral = 0
        else:
            neutral = 1
        gameIdList.append(gameId)
        dateTimeList.append(dateTime)
        homeTeamList.append(homeTeam)
        awayTeamList.append(awayTeam)
        locationList.append(location)
        neutralList.append(neutral)

    return gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList, neutralList

def getStaticGameData(game_id):
    month = monthDict[game_id[4:6]]
    year = getYearFromId(game_id)
    url = 'https://www.basketball-reference.com/leagues/NBA_{}_games-{}.html'.format(year, month)
    soup = bs.BeautifulSoup(urlopen(url), features='lxml')
    rows = [p for p in soup.find('div', {'id': 'div_schedule'}).findAll('tr')]
    rowList = []
    for row in rows:
        rowList.append([td for td in row.findAll(['td', 'th'])])
    gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList, neutralList = [], [], [], [], [], []
    for i in range(1, len(rowList)):
        dateTime = parser.parse(rowList[i][0].getText())
        aChildrenH = str(rowList[i][4].findChildren('a'))
        homeTeam = aChildrenH.partition('teams/')[2][:3]
        aChildrenA = str(rowList[i][2].findChildren('a'))
        awayTeam = aChildrenA.partition('teams/')[2][:3]

        gameId = '{}0{}'.format(dateTime.strftime("%Y%m%d"), homeTeam)
        dateTime = convDateTime(gameId, rowList[i][1].getText())
        location = rowList[i][9].getText()
        if rowList[i][10].getText() == '':
            neutral = 0
        else:
            neutral = 1
        gameIdList.append(gameId)
        dateTimeList.append(dateTime)
        homeTeamList.append(homeTeam)
        awayTeamList.append(awayTeam)
        locationList.append(location)
        neutralList.append(neutral)
        if gameId == game_id:
            break

    return dateTimeList[-1], homeTeamList[-1], awayTeamList[-1], locationList[-1], neutralList[-1]

def getTeamCurrentRoster(team_abbr):
    """
        Scrapes a teams current roster. Players who are injured and guaranteed not to play
        are removed from the roster

        Parameters
        ----------
        playerid : the player id to scrape
        team_abbr: the team that the player is playing on

        Returns
        -------
        a numerical salary value
    """

    year = 2023;
    url = f"https://www.basketball-reference.com/teams/{str(team_abbr).upper()}/{str(year)}.html"

    try:
        teamRoster = scrapeRoster(team_abbr, year)[0]
        soup = bs.BeautifulSoup(urlopen(url), features='lxml')
        injuriesTable = re.split(r"<th scope=\"row\"[^<>]*class=\"left \"", str(soup.find('div', {'id': 'all_injuries'})))
        regex = r"data-append-csv=\"([a-z0-9]+)\"[^<>]*data-stat=\"player\"[^<>]*>[\W\w]*<td[^<>]*data-stat=\"note\"[^<>]*>([\S ]*)</td></tr>"
        for row in injuriesTable:
            matches = re.findall(regex, row)
            try:
                if len(matches) != 1:
                    raise Exception()
                else:
                    matches = matches[0]

                if len(matches) != 2:
                    raise Exception()
                else:
                    if str(matches[1]).lower().startswith('out'):
                        teamRoster.remove(matches[0])
            except Exception:
                formatString = matches[0] if len(matches) else 'UNKNOWN'
                #print('Player {0} status not found on basketball-reference.com'.format(formatString))

        return teamRoster

    except Exception as e:
        print(e)
        raise Exception('UNABLE TO NAVIGATE TO: {}'.format(url))

def getStaticYearData(year):

    gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList, neutralList = [], [], [], [], [], []
    months = ['october', 'november', 'december', 'january', 'february', 'march', 'april']

    for month in months:
        gameIdMonth, dateTimeMonth, homeTeamMonth, awayTeamMonth, locationMonth, neutralMonth = getStaticMonthData(month, year)
        gameIdList.extend(gameIdMonth)
        dateTimeList.extend(dateTimeMonth)
        homeTeamList.extend(homeTeamMonth)
        awayTeamList.extend(awayTeamMonth)
        locationList.extend(locationMonth)
        neutralList.extend(neutralMonth)

    return gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList, neutralList


def initGameStateData(year):

    col = [['gameState','gameState','gameState','gameState','gameState','gameState','gameState','gameState', 'gameState', 'gameState', 'home','home','home','home','home','home','home','home','home','home','home','home','home','home','away','away','away','away','away','away','away','away','away','away','away','away','away','away'],['winner','teamHome','teamAway','location','rivalry','datetime','neutral', 'endtime', 'attendance', 'referees', 'q1Score','q2Score','q3Score','q4Score','overtimeScores','points','streak','daysSinceLastGame','playerRoster','record','matchupWins','salary','avgSalary','numberOfGamesPlayed','q1Score','q2Score','q3Score','q4Score','overtimeScores','points','streak','daysSinceLastGame','playerRoster','record','matchupWins','salary','avgSalary','numberOfGamesPlayed']]

    col = pd.MultiIndex.from_arrays(col, names = ['','teamData'])
    df = pd.DataFrame(index = getYearIds(year), columns = col)
    df.index.name = 'game_id'
    return df


def gameFinished(gameId):
    try:
        r = requests.get("https://www.basketball-reference.com/boxscores/{}.html".format(gameId))
        if r.status_code == 200:
            return 1
        else:
            return 0
    except Exception as e:
        print("Game id({}) does not exist".format(gameId))
        return 0

def fillStaticValues(year):
    df = initGameStateData(year)
    gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList, neutralList = getStaticYearData(year)
    df['gameState', 'datetime'] = dateTimeList
    df['gameState', 'teamHome'] = homeTeamList
    df['gameState', 'teamAway'] = awayTeamList
    df['gameState', 'location'] = locationList

    df['gameState', 'neutral'] = neutralList
    df['gameState', 'index'] = df.index

    df['home', 'numberOfGamesPlayed'] = df.apply(lambda d: getNumberGamesPlayed(d['gameState', 'teamHome'], year, d['gameState', 'index']), axis=1)
    df['away', 'numberOfGamesPlayed'] = df.apply(lambda d: getNumberGamesPlayed(d['gameState', 'teamAway'], year, d['gameState', 'index']), axis=1)
    df['home', 'daysSinceLastGame'] = df.apply(lambda d: getDaysSinceLastGame(getTeamScheduleAPI(d['gameState', 'teamHome'], d['gameState', 'index']), d['gameState', 'index']), axis=1)
    df['away', 'daysSinceLastGame'] = df.apply(lambda d: getDaysSinceLastGame(getTeamScheduleAPI(d['gameState', 'teamAway'], d['gameState', 'index']), d['gameState', 'index']), axis=1)
    df['gameState', 'rivalry'] = df.apply(lambda d: getRivalry(d['gameState', 'teamHome'], d['gameState', 'teamAway']), axis=1)
    df.drop('index', inplace=True, level=1, axis=1)
    return df

#fillStaticValues(2023).to_csv('../data/gameStats/game_state_data_2023.csv')

def fillDependentStaticValues(year):
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0)
    df['gameState', 'index'] = df.index

    df['home', 'numberOfGamesPlayed'] = df.apply(lambda d: getNumberGamesPlayed(d['gameState', 'teamHome'], year, d['gameState', 'index']), axis = 1)
    df['away', 'numberOfGamesPlayed'] = df.apply(lambda d: getNumberGamesPlayed(d['gameState', 'teamAway'], year, d['gameState', 'index']), axis = 1)
    df['gameState', 'rivalry'] = df.apply(lambda d: getRivalry(d['gameState', 'teamHome'], d['gameState', 'teamAway']), axis = 1)
    df.drop('index', inplace = True, level = 1, axis = 1)
    return df

#fillDependentStaticValues(2023).to_csv('../data/gameStats/game_state_data_2023.csv')

def updateGameStateDataAll(years):
    df = pd.DataFrame()
    for year in years:
        df_current = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0)
        df = pd.concat([df, df_current], axis = 0)
    return df

def updateGameStateData():
    """
    Checks if current year's CSV is up to date with game data information. If not, update it.

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    """

    df = pd.read_csv('../data/gameStats/game_state_data_2023.csv', header = [0,1], index_col = 0, dtype = object)
    df.dropna()
    df2 = df[df['gameState']['winner'].isnull()]
    df_dict = df2.to_dict('index')
    previousGameList = []
    for key, value in df_dict.items():
        if gameFinished(key):
            previousGameList.append(key)
        else:
            break

    #previousGameList holds all gameids that have been played but do not have data in the files
    for curId in previousGameList:
        # fills in values for game that has already happened
        tempList = getGameData(curId, int(df.loc[curId]['gameState']['neutral']))
        tempList = tempList[1:]
        df.loc[curId] = tempList

        #edit the next game data for both teams

        teamHome = tempList[1]
        teamAway = tempList[2]
        indexHome = getTeamsNextGame(teamHome, curId)
        indexAway = getTeamsNextGame(teamAway, curId)

        homeGameData = getGameStateFutureData(indexHome)
        homeData = getTeamFutureData(indexHome, teamHome, teamAway, 1)

        awayGameData = getGameStateFutureData(indexAway)
        awayData = getTeamFutureData(indexAway, teamAway, teamHome, 0)

        df.loc[indexHome, 'gameState'] = homeGameData
        if homeGameData[1] == teamHome:
            df.loc[indexHome, 'home'] = homeData
        else:
            df.loc[indexHome, 'away'] = homeData

        df.loc[indexAway, 'gameState'] = awayGameData
        if awayGameData[1] == teamAway:
            df.loc[indexAway, 'home'] = awayData
        else:
            df.loc[indexAway, 'away'] = awayData

        print(curId)

    df.to_csv('../data/gameStats/game_state_data_2023.csv')
    updateGameStateDataAll(np.arange(2015,2024)).to_csv('../data/gameStats/game_state_data_ALL.csv')
    return

def getTeamFutureData(game_id, team_abbr, opp_team, home):
    teamSchedule = getTeamScheduleAPI(team_abbr, gameIdToDateTime(game_id).strftime('%Y%m%d'))
    streak = getTeamStreak(teamSchedule, game_id)
    days = getDaysSinceLastGame(teamSchedule, game_id)
    roster = getTeamCurrentRoster(team_abbr)
    record = getTeamRecord(teamSchedule, game_id)
    matchupRecord = getPastMatchUpWinLoss(teamSchedule, game_id, opp_team)
    if home == 1:
        matchupWins = matchupRecord[0]
    else:
        matchupWins = matchupRecord[1]
    salary, avgSalary = getTeamSalaryData(team_abbr, game_id, roster)
    gamesPlayed = getNumberGamesPlayed(team_abbr, 2023, game_id)
    return [None,None,None,None,None,None,streak, days, roster, record, matchupWins, salary, avgSalary, gamesPlayed]

def getGameStateFutureData(game_id):
    datetime, teamHome, teamAway, location, neutral = getStaticGameData(game_id)
    rivalry = getRivalry(teamHome, teamAway)
    return [None, teamHome, teamAway, location, rivalry, datetime, datetime, None, None, neutral]


updateGameStateData()
#df = pd.read_csv('../data/gameStats/game_state_data_2023.csv', index_col=0, header=[0, 1])

#updateGameStateDataAll(np.arange(2015,2024)).to_csv('../data/gameStats/game_state_data_ALL.csv')

