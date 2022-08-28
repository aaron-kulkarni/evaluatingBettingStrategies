import pandas as pd
from teamPerformance import teamAverageHelper, playerAverageHelper, opponentAverageHelper, getTeamSchedule, getYearFromId, getTeamGameIds
import datetime as dt

def getPreviousGamePlayerStats(gameId):
    """
    Gets all numerical stats in game_data_player_stats for games
    that took place before (and not including) the given game, for each
    player on the roster in the given game. Does not include player
    statistics from when a player was benched the entire game.

    Parameters
    ----------
    game_id : the basketball-reference.com id of the game

    Returns
    -------
    a pandas dataframe with all the players' statistics for each
    previous game.
    """
    
    year = getYearFromId(gameId);
    df = pd.DataFrame(pd.read_csv('data/gameStats/game_data_player_stats_{}.csv'.format(year), index_col = 0, header = 0))

    playerIdList = getPlayerIdsFromGame(gameId)

    df = df[df['playerid'].isin(playerIdList)]
    df = df[pd.to_datetime(df['gameid'].str.slice(0,8), format="%Y%m%d") < gameIdToDateTime(gameId)]
    mask = pd.isna(df['MP'])
    df = df[~mask]
    df = df.drop(columns=['MP', 'Name'])
    return df

def getPreviousGameSinglePlayerStats(gameId, playerId):
    df = getPreviousGamePlayerStats(gameId)
    return df[df['playerid'] == playerId]

def getPreviousGameTeamStats(gameId):
    """
    Gets all numerical stats in team_total_stats for games that
    took place before (and not including) the given game, for each
    team playing in the given game.

    Parameters
    ----------
    game_id : the basketball-reference.com id of the game

    Returns
    -------
    a pandas dataframe with all the players' statistics for each
    previous game.
    """

    year = getYearFromId(gameId);
    df = pd.DataFrame(pd.read_csv('data/teamStats/team_total_stats_{}.csv'.format(year), index_col = 0, header = [0,1]))
    dfTemp = df.loc[gameId]
    teamNameHome = dfTemp['home']['teamAbbr']
    teamNameAway = dfTemp['away']['teamAbbr']

    dfHome = getPreviousGameSingleTeamStats(gameId, teamNameHome, year)
    dfAway = getPreviousGameSingleTeamStats(gameId, teamNameAway, year)

    df = pd.concat([dfHome, dfAway], axis = 0)
    df = df.drop(index=gameId)

    return df

def getPreviousGameSingleTeamStats(gameId, team, year): 
    df = pd.DataFrame(pd.read_csv('data/teamStats/team_total_stats_{}.csv'.format(year), index_col = 0, header = [0,1]))

    dfHome = df[df['home']['teamAbbr'] == team]
    dfAway = df[df['away']['teamAbbr'] == team]

    df = pd.concat([dfHome['home'], dfAway['away']], axis = 0)

    df.sort_index(inplace=True)

    df = df[:gameId]

    return df

def getPlayerIdsFromGame(gameId):
    year = getYearFromId(gameId)
    df = pd.read_csv('data/gameStats/game_state_data_{}.csv'.format(year), index_col = 0, header = [0, 1])
    df = df.loc[gameId]

    homeList = df['home']['playerRoster']
    awayList = df['away']['playerRoster']
    homeList = homeList.replace("'", "").replace("]", "").replace("[", "").replace(" ", "").split(",")
    awayList = awayList.replace("'", "").replace("]", "").replace("[", "").replace(" ", "").split(",")
    homeList.extend(awayList)
    return homeList


def gameIdToDateTime(gameId):
    return dt.datetime.strptime(gameId[0:8], '%Y%m%d')

print(getPreviousGameSinglePlayerStats('201601090LAC', 'linje01'))
