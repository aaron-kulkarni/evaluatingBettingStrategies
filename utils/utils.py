import datetime as dt
import pandas as pd
import re
import sys
import os
import numpy as np
from urllib.request import urlopen
import requests

from sportsipy.nba.boxscore import Boxscores, Boxscore
from sportsipy.nba.teams import Teams
from sportsipy.nba.schedule import Schedule

"""
The following functions collect data from the Sportsipy API
"""

teamDict = {
    "Toronto Raptors": "TOR",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Philadelphia 76ers": "PHI",
    "New York Knicks": "NYK",
    "Cleveland Cavaliers": "CLE",
    "Chicago Bulls": "CHI",
    "Milwaukee Bucks": "MIL",
    "Indiana Pacers": "IND",
    "Detroit Pistons": "DET",
    "Atlanta Hawks": "ATL",
    "Washington Wizards": "WAS",
    "Miami Heat": "MIA",
    "Charlotte Hornets": "CHO",
    "Orlando Magic": "ORL",
    "Portland Trail Blazers": "POR",
    "Oklahoma City Thunder": "OKC",
    "Utah Jazz": "UTA",
    "Denver Nuggets": "DEN",
    "Minnesota Timberwolves": "MIN",
    "Golden State Warriors": "GSW",
    "Los Angeles Clippers": "LAC",
    "Phoenix Suns": "PHO",
    "Sacramento Kings": "SAC",
    "Los Angeles Lakers": "LAL",
    "Houston Rockets": "HOU",
    "Memphis Grizzlies": "MEM",
    "San Antonio Spurs": "SAS",
    "Dallas Mavericks": "DAL",
    "New Orleans Pelicans": "NOP"
}


def getGamesOnDate(date):
    return list(Boxscores(dateToDateTime(date)).games.values())[0]


def getGamesBetween(start_date, end_date):
    return Boxscores(dateToDateTime(start_date), dateToDateTime(end_date)).games


def getBoxscoreData(game_id):
    return Boxscore(game_id)



def getTeamScheduleAPI(team, game_date):
    game_date = str(game_date)
    gameYear = game_date[0:4]
    gameMonth = game_date[4:6]

    if int(gameYear) == 2020:  # 2020 was exception because covid messed up schedule
        if int(gameMonth.lstrip(
                "0")) < 11:  # converted gameMonth to int without leading 0. check month to find correct season
            teamSchedule = Schedule(team, int(gameYear)).dataframe
        else:
            teamSchedule = Schedule(team, int(gameYear) + 1).dataframe
    else:
        if int(gameMonth.lstrip("0")) > 7:  # games played after july are part of next season
            teamSchedule = Schedule(team, int(gameYear) + 1).dataframe
        else:
            teamSchedule = Schedule(team, int(gameYear)).dataframe

    return teamSchedule


def getTeamsNextGame(team, game_id):

    year = getYearFromId(game_id)
    gameIdList = getTeamGameIds(team, year)
    index = gameIdList.index(game_id)
    return gameIdList[index+1]


def getGameIdList(year):
    df = pd.read_csv('../data/gameStats/all_game_ids.csv')
    gameIdList = list(df[str(year)])
    gameIdList = [i for i in gameIdList if str(i) != 'nan']
    return gameIdList

"""
The following functions convert and validate items
"""


def gameIdToDateTime(game_id):
    return dt.datetime.strptime(game_id[0:8], '%Y%m%d')


def dateToDateTime(date, format='%Y%m%d'):
    return dt.datetime.strptime(date[0:8], format)


def dateDateTime(date, format='%Y-%m-%d %H:%M:%S'):
    return dt.datetime.strptime(date, format)

def getYearFromId(game_id):
    return getYearHelper(int(game_id[0:4]), int(game_id[4:6].lstrip("0")))

def getYearFromDate(date):
    year = date.strftime("%Y")
    month = date.strftime("%m")
    return getYearHelper(year, month)


def getYearHelper(gameYear, month):
    gameYear = int(gameYear)
    month = int(month)
    seasonYear = gameYear
    if gameYear == 2020:
        if month >= 11:
            seasonYear += 1
    else:
        if month > 7:
            seasonYear += 1

    return seasonYear


def gameIdIsValid(game_id):
    return bool(re.match(r"^[\d]{9}[A-Z]{3}$", game_id))


"""
The following functions manage reading from and writing to files
"""


def readCSV(filepath, **kwargs):
    if filepath.startswith(r"data/"):
        filepath = filepath[5:]
    else:
        filepath = filepath.split(r"/data/")[-1]
    return pd.read_csv("../data/" + filepath, **kwargs)


def writeCSV(df, filepath, **kwargs):
    if filepath.startswith(r"data/"):
        filepath = filepath[5:]
    else:
        filepath = filepath.split(r"/data/")[-1]
    return df.to_csv("../data/" + filepath, **kwargs)


def getTeamScheduleCSVSplit(team, year):
    df = pd.DataFrame(pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1]))

    dfHome = df[df['gameState']['teamHome'] == team]
    dfAway = df[df['gameState']['teamAway'] == team]
    return dfHome, dfAway

def getTeamScheduleCSV(team, year):
    df = pd.DataFrame(pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(2023), index_col=0, header=[0, 1]))

    dfHome = df[df['gameState']['teamHome'] == team]
    dfAway = df[df['gameState']['teamAway'] == team]

    df = pd.concat([dfHome['home'], dfAway['away']], axis=0)
    df = df.sort_index()

    dfState = pd.concat([dfHome['gameState'], dfAway['gameState']], axis=0)
    dfState = dfState.sort_index()

    dfTotal = pd.concat([dfState, df], axis=1)

    return dfTotal

def getTeamsCSV(game_id):
    year = getYearFromId(game_id)
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1])
    teamHome = df.loc[game_id]['gameState']['teamHome']
    teamAway = df.loc[game_id]['gameState']['teamAway']
    return teamHome, teamAway


def getTeamsAllYears():
    df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', index_col=0, header=[0, 1])['gameState']
    df = df[['teamHome', 'teamAway']]

    return df


def getTeamsDF(year):
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1])['gameState']
    df = df[['teamHome', 'teamAway']]
    return df


"""
Other functions
"""

def getYearIds(year):
    df = pd.read_csv('../data/gameStats/all_game_ids.csv', index_col=False)
    gameIdList = df[str(year)].to_list()
    gameIdList = [i for i in gameIdList if pd.notnull(i)]
    return gameIdList


def getNumberGamesPlayed(team, year, game_id):
    index = getTeamGameIds(team, year).index(game_id)
    return index


def getTeamGameIds(team, year):
    homeTeamSchedule, awayTeamSchedule = getTeamScheduleCSVSplit(team, year)
    teamSchedule = pd.concat([homeTeamSchedule, awayTeamSchedule], axis=0)
    teamSchedule = teamSchedule.sort_index(ascending=True)
    return list(teamSchedule.index)


def getAllTeams():
    return list(teamDict.values())

def getTeamDict():
    return teamDict

def getSeasonGames(gameId, team):
    if not gameIdIsValid(gameId):
        raise Exception('Issue with Game ID')

    year = getYearFromId(gameId)
    gameIdList = getTeamGameIds(team, year)
    index = gameIdList.index(gameId)
    gameIdList = gameIdList[:index]

    return gameIdList


def getRecentNGames(gameId, n, team):
    '''
    Obtains ids of the past n games (non inclusive) given the game_id of current game and team abbreviation

    '''
    if n <= 0:
        raise Exception('N parameter must be greater than 0')

    if not gameIdIsValid(gameId):
        raise Exception('Issue with Game ID')

    year = getYearFromId(gameId)
    gameIdList = getTeamGameIds(team, year)
    index = gameIdList.index(gameId)
    gameIdList = gameIdList[index - n:index]

    return gameIdList


def convDateTime(gameId, timeOfDay):
    if timeOfDay[-1:] == 'p':
        timeOfDay = timeOfDay[:-1] + 'PM'
        return dt.datetime.strptime(gameId[0:8] + timeOfDay, '%Y%m%d%I:%M%p')
    if timeOfDay[-1:] == 'a':
        timeOfDay = timeOfDay[:-1] + 'AM'
        return dt.datetime.strptime(gameId[0:8] + timeOfDay, '%Y%m%d%I:%M%p')

    else:
        return print('Error')


def getGamesToday():
    df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', header=[0,1], index_col=0)['gameState']
    df['id'] = df.apply(lambda d: str(d['datetime'])[0:10].replace('-', ''), axis=1)
    index = df[df['id'] == dt.datetime.today().strftime('%Y%m%d')].index
    return index

def getPreviousGames():
    df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', header=[0,1], index_col=0)['gameState']
    df['id'] = df.apply(lambda d: str(d['datetime'])[0:10].replace('-', ''), axis=1)
    index = df[df['id'] < dt.datetime.today().strftime('%Y%m%d')].index
    return index


def sortDate(gameIdList):
    '''
    Sorts a gameIdList by time order and outputs sorted gameId list

    '''

    df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', index_col=0, header=[0, 1])
    dateCol = pd.DataFrame()
    dateCol['datetime'] = pd.to_datetime(df['gameState']['datetime'])
    dateCol = dateCol[dateCol.index.isin(gameIdList)]
    dateCol.sort_values(by='datetime', ascending=True, inplace=True)
    return list(dateCol.index)

def sortDateMulti(gameIdList):
    gameIdList = sortDate(gameIdList)
    ids = [i for pair in zip(gameIdList,gameIdList) for i in pair]
    alt = list(np.resize([1,0],len(ids)))
    index = pd.MultiIndex.from_arrays([ids, alt])
    return index
    
def returnDate(gameId, start):
    df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', index_col=0, header=[0, 1])
    if start == 1:
        return df['gameState', 'datetime'].loc[gameId]
    if start == 0:
        return df['gameState', 'endtime'].loc[gameId]
    else:
        return 'Error'


def orderAllDates(df):
    df.reset_index(inplace=True)
    df['date'] = df.apply(lambda d: returnDate(d['index'], d['start']), axis=1)
    df.sort_values(by='date', ascending=True, inplace=True)
    df.set_index(['index'], inplace=True)
    return df



def sortAllDates(gameIdList):
    df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', index_col=0, header=[0, 1])
    df1, df2 = pd.DataFrame(), pd.DataFrame()
    df1['date'] = pd.to_datetime(df['gameState']['datetime'])
    df1['start'] = 1
    df2['date'] = pd.to_datetime(df['gameState']['endtime'])
    df2['start'] = 0
    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    df = pd.concat([df1, df2], axis=0)
    df.sort_values(by='date', ascending=True, inplace=True)
    df.set_index(['game_id'], inplace=True)
    df = df[df.index.isin(gameIdList)]
    df.reset_index(inplace=True)
    df.set_index(['game_id', 'start'], inplace=True)

    return df.index


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def cumsum_reset_on_null(srs):
    cumulative = srs.cumsum().fillna(method='ffill')
    restart = ((cumulative * srs.isnull()).replace(0.0, np.nan).fillna(method='ffill').fillna(0))
    result = (cumulative - restart)

    return result.replace(0, np.nan)

def returnDatetime(gameId):
    df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', header=[0,1], index_col=0)['gameState']
    return df.loc[gameId]['datetime']
    
    
def getGameInfo():
    for i in getGamesToday():
        print('{} - {}'.format(i, returnDatetime(i)))
        
def getNextGames():
    datetime = dt.datetime.now()
    df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', header=[0,1], index_col=0)['gameState']
    df['datetime'] = pd.to_datetime(df['datetime'])
    datetimelist = pd.to_datetime(df['datetime']).to_list()
    next_date = min(item for item in datetimelist if item > datetime)
    index = df[df['datetime'] == next_date].index
    print(next_date)
    return index