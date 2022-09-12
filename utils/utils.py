import datetime as dt
import pandas as pd
import re

from sportsipy.nba.boxscore import Boxscores, Boxscore
from sportsipy.nba.teams import Teams
from sportsipy.nba.schedule import Schedule

"""
The following functions collect data from the Sportsipy API
"""


def getGamesOnDate(date):
    return list(Boxscores(dateToDateTime(date)).games.values())[0]


def getGamesBetween(start_date, end_date):
    return Boxscores(dateToDateTime(start_date), dateToDateTime(end_date)).games


def getGameData(game_id):
    return Boxscore(game_id)


def getTeamScheduleAPI(team, game_date):
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


"""
The following functions convert and validate items
"""


def gameIdToDateTime(game_id):
    return dt.datetime.strptime(game_id[0:8], '%Y%m%d')


def dateToDateTime(date, format='%Y%m%d'):
    return dt.datetime.strptime(date[0:8], format)

def dateDateTime(date, format = '%Y-%m-%d %H:%M:%S'):
    return dt.datetime.strptime(date, format)


def getYearFromId(game_id):
    if int(game_id[0:4]) == 2020:
        if int(game_id[4:6].lstrip("0")) < 11:
            year = int(game_id[0:4])
        else:
            year = int(game_id[0:4]) + 1
    else:
        if int(game_id[4:6].lstrip("0")) > 7:
            year = int(game_id[0:4]) + 1
        else:
            year = int(game_id[0:4])
    return year


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


def getTeamScheduleCSV(team, year):
    df = pd.DataFrame(pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1]))

    dfHome = df[df['gameState']['teamHome'] == team]
    dfAway = df[df['gameState']['teamAway'] == team]
    return dfHome, dfAway


def getTeamsCSV(game_id):
    year = getYearFromId(game_id)
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1])
    teamHome = df.loc[game_id]['gameState']['teamHome']
    teamAway = df.loc[game_id]['gameState']['teamAway']
    return teamHome, teamAway


def getTeamsAllYearsCSV(years):
    df = pd.DataFrame()
    for year in years:
        teamDF = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header=[0, 1], index_col=0)
        teams = pd.concat([teamDF['gameState']['teamHome'], teamDF['gameState']['teamAway']], axis=1)
        df = pd.concat([df, teams], axis=0)

    return df


def getTeamsDF(year):
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1])['gameState']
    df = df[['teamHome', 'teamAway']]
    return df


"""
Other functions
"""


def getNumberGamesPlayed(team, year, game_id):
    index = getTeamGameIds(team, year).index(game_id)
    return index


def getTeamGameIds(team, year):
    homeTeamSchedule, awayTeamSchedule = getTeamScheduleCSV(team, year)
    teamSchedule = pd.concat([homeTeamSchedule, awayTeamSchedule], axis=0)
    teamSchedule = teamSchedule.sort_index(ascending=True)
    return list(teamSchedule.index)


def getAllTeams():
    teamList = []
    for team in Teams():
        teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
        teamList.append(teamAbbr)

    return teamList


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
        timeOfDat = timeOfDay[:-1] + 'AM'
        return dt.datetime.strptime(gameId[0:8] + timeOfDay, '%Y%m%d%I:%M%p')

    else:
        return print('Error')
    
def sortDate(gameIdList):
    '''
    Sorts a gameIdList by time order and outputs sorted gameId list
    
    '''

    df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', index_col = 0, header = [0, 1])
    dateCol = pd.DataFrame()
    dateCol['datetime'] = pd.to_datetime(df['gameState']['datetime'])
    dateCol = dateCol[dateCol.index.isin(gameIdList)]
    dateCol.sort_values(by = 'datetime', ascending = True, inplace = True)
    return list(dateCol.index)
    
    
    
