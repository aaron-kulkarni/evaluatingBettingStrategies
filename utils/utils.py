import datetime as dt
import pandas as pd
import re
from sportsipy.nba.teams import Teams
from sportsipy.nba.schedule import Schedule


def gameIdToDateTime(game_id):
    return dt.datetime.strptime(game_id[0:8], '%Y%m%d')


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


def getNumberGamesPlayed(team, year, game_id):
    index = getTeamGameIds(team, year).index(game_id)
    return index


def getTeams(game_id):
    year = getYearFromId(game_id)
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1])
    teamHome = df.loc[game_id]['gameState']['teamHome']
    teamAway = df.loc[game_id]['gameState']['teamAway']
    return teamHome, teamAway


def getTeamScheduleCSV(team, year):
    df = pd.DataFrame(pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1]))

    dfHome = df[df['gameState']['teamHome'] == team]
    dfAway = df[df['gameState']['teamAway'] == team]
    return dfHome, dfAway

def getTeamScheduleAPI(gameYear, gameMonth, team):
    if int(gameYear) == 2020: #2020 was exception because covid messed up schedule
        if int(gameMonth.lstrip("0")) < 11: #converted gameMonth to int without leading 0. check month to find correct season
            teamSchedule = Schedule(team, int(gameYear)).dataframe
        else:
            teamSchedule = Schedule(team, int(gameYear) + 1).dataframe
    else:
        if int(gameMonth.lstrip("0")) > 7: #games played after july are part of next season
            teamSchedule = Schedule(team, int(gameYear) + 1).dataframe
        else:
            teamSchedule = Schedule(team, int(gameYear)).dataframe

    return teamSchedule


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

def getTeamsDF(year):
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col = 0, header = [0,1])['gameState']
    df = df[['teamHome', 'teamAway']]
    return df 
    
def getSeasonGames(gameId, team):
    if bool(re.match("^[\d]{9}[A-Z]{3}$", gameId)) == False:
        raise Exception('Issue with Game ID')

    year = getYearFromId(gameId)
    gameIdList = getTeamGameIds(team, year)
    index = gameIdList.index(gameId)
    gameIdList = gameIdList[:index]

    return gameIdList 

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

