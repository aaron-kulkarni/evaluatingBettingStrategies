import datetime as dt
import pandas as pd
import numpy as np


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


def getTeamSchedule(team, year):
    df = pd.DataFrame(pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1]))

    dfHome = df[df['gameState']['teamHome'] == team]
    dfAway = df[df['gameState']['teamAway'] == team]
    return dfHome, dfAway


def getTeamGameIds(team, year):
    homeTeamSchedule, awayTeamSchedule = getTeamSchedule(team, year)
    teamSchedule = pd.concat([homeTeamSchedule, awayTeamSchedule], axis=0)
    teamSchedule = teamSchedule.sort_index(ascending=True)
    return list(teamSchedule.index)

