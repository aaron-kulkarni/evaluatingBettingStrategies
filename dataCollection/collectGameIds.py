import pandas as pd
import numpy as np
from collectGameData import *
import bs4 as bs
from urllib.request import urlopen
import re
from dateutil import parser
import requests

import sys
sys.path.insert(0, '..')
from utils.utils import *
from dataProcessing.progress.rosterDict import scrapeRoster


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

#initGameStateData(2023).to_csv('../data/gameStats/game_state_data_2023.csv')

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
    df2 = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', header = [0,1], index_col = 0, dtype = object)
    df.dropna()
    df2.dropna()
    df_dict = df.to_dict('index')
    lastGameRecorded = 0 #most recent game that was played and we have data for
    previousGameList = []
    for key, value in df_dict.items():
        if str(value[('gameState', 'datetime')]) != 'nan' and gameFinished(key):
            previousGameList.append(key)
            if str(value[('gameState', 'winner')]) != 'nan':
                lastGameRecorded = key
        else:
            break
    mostRecentGame = previousGameList[-1] #most recent game played
    if lastGameRecorded == mostRecentGame:
        # list is already fully updated
        return

    if lastGameRecorded != 0:
        idx = previousGameList.index(lastGameRecorded)
        previousGameList = previousGameList[idx + 1:]  # don't care about previous games that already have data

    for curId in previousGameList:  # i honestly didn't know how to do it better than a for loop. should be relatively short list though
        tempList = getGameData(curId, int(df.loc[curId]['gameState']['neutral']))
        tempList = tempList[1:]
        df.loc[curId] = tempList
        df2.loc[curId] = tempList
        #edit the next game data for both teams

        teamHome = tempList[1]
        teamAway = tempList[2]
        indexHome = getTeamsNextGame(teamHome, curId)
        indexAway = getTeamsNextGame(teamAway, curId)

        homeTeamSchedule = getTeamScheduleAPI(teamHome, gameIdToDateTime(indexHome).strftime('%Y%m%d'))
        awayTeamSchedule = getTeamScheduleAPI(teamAway, gameIdToDateTime(indexHome).strftime('%Y%m%d'))
        homeTeamRoster = getTeamCurrentRoster(teamHome)
        awayTeamRoster = getTeamCurrentRoster(teamAway)
        homeTotalSalary, homeAverageSalary = getTeamSalaryData(teamHome, indexHome, homeTeamRoster)
        awayTotalSalary, awayAverageSalary = getTeamSalaryData(teamAway, indexAway, awayTeamRoster)
        homeStreak = getTeamStreak(homeTeamSchedule, indexHome)
        awayStreak = getTeamStreak(awayTeamSchedule, indexAway)
        homeRecord = getTeamRecord(homeTeamSchedule, indexHome)
        awayRecord = getTeamRecord(awayTeamSchedule, indexAway)
        matchupWinsHome, matchupWinsAway = getPastMatchUpWinLoss(homeTeamSchedule, indexHome, teamAway)

        df.loc[indexHome, ('home', 'playerRoster')] = str(homeTeamRoster)
        df.loc[indexAway, ('away', 'playerRoster')] = str(awayTeamRoster)
        df.loc[indexHome, ('home', 'salary')] = homeTotalSalary
        df.loc[indexAway, ('away', 'salary')] = awayTotalSalary
        df.loc[indexHome, ('home', 'avgSalary')] = homeAverageSalary
        df.loc[indexAway, ('away', 'avgSalary')] = awayAverageSalary
        df.loc[indexHome, ('home', 'record')] = str(homeRecord)
        df.loc[indexAway, ('away', 'record')] = str(awayRecord)
        df.loc[indexHome, ('home', 'streak')] = homeStreak
        df.loc[indexAway, ('away', 'streak')] = awayStreak
        df.loc[indexHome, ('home', 'matchupWins')] = int(matchupWinsHome)
        df.loc[indexAway, ('away', 'matchupWins')] = int(matchupWinsAway)

        df2.loc[indexHome, ('home', 'playerRoster')] = str(homeTeamRoster)
        df2.loc[indexAway, ('away', 'playerRoster')] = str(awayTeamRoster)
        df2.loc[indexHome, ('home', 'salary')] = homeTotalSalary
        df2.loc[indexAway, ('away', 'salary')] = awayTotalSalary
        df2.loc[indexHome, ('home', 'avgSalary')] = homeAverageSalary
        df2.loc[indexAway, ('away', 'avgSalary')] = awayAverageSalary
        df2.loc[indexHome, ('home', 'record')] = str(homeRecord)
        df2.loc[indexAway, ('away', 'record')] = str(awayRecord)
        df2.loc[indexHome, ('home', 'streak')] = homeStreak
        df2.loc[indexAway, ('away', 'streak')] = awayStreak
        df2.loc[indexHome, ('home', 'matchupWins')] = int(matchupWinsHome)
        df2.loc[indexAway, ('away', 'matchupWins')] = int(matchupWinsAway)
        print(curId)

    df.to_csv('../data/gameStats/game_state_data_2023.csv')
    df2.to_csv('../data/gameStats/game_state_data_ALL.csv')
    return

updateGameStateData()
#df = pd.read_csv('../data/gameStats/game_state_data_2023.csv', index_col=0, header=[0, 1])

#getTeamCurrentRoster('DET')


def updateGameStateDataAll(years):
    df = pd.DataFrame()
    for year in years:
        df_current = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0)
        df = pd.concat([df, df_current], axis = 0)
    return df

#updateGameStateDataAll(np.arange(2015,2024)).to_csv('../data/gameStats/game_state_data_ALL.csv')

