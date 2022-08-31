import re
import numpy as np
import pandas as pd
import datetime as dt
import sys

def concatTeamStats(years):

    df = pd.DataFrame()
    for year in years:
        teamStats = pd.read_csv('../data/teamStats/team_total_stats_{}.csv'.format(year), header = [0,1], index_col = 0)
        df = pd.concat([df, teamStats], axis = 0)

    return df

years = np.arange(2015, 2023)
concatTeamStats(years).to_csv('team_total_stats_all.csv')


from TeamPerformance import TeamPerformance

tp = TeamPerformance(2015)
tp.getSignal()


def getSignal():
    df = pd.DataFrame()
    years = np.arange(2015, 2023)
    for year in years:
        dfCurrent = pd.DataFrame(
            pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1]))
        df = pd.concat([df, dfCurrent], axis=0)
    df = df['gameState']
    df['signal'] = df.apply(lambda d: 1 if d['winner'] == d['teamHome'] else 0, axis=1)
    return df['signal']


def performPCA(n, include = True): 
    df = pd.read_csv('../data/teamStats/team_total_stats_all.csv', index_col = 0, header = [0,1])
    df.drop('teamAbbr', axis = 1, level = 1, inplace = True)

    if include == True: 
        colHome = ['home_{}'.format(col) for col in df['home'].columns]
        colAway = ['away_{}'.format(col) for col in df['away'].columns]
        df.columns = colHome + colAway
    else:
        df = df['home']
        
    from sklearn.preprocessing import StandardScaler
    
    features = list(df.columns)
    x_select = df.loc[:, features].values
    y_select = getSignal().values
    x_select = StandardScaler().fit_transform(x_select)

    from sklearn.decomposition import PCA
    
    pca = PCA(n_components = n)
    principalComponents = pca.fit_transform(x_select)
    principalDF = pd.DataFrame(data = principalComponents, columns = ['PCA{}'.format(i) for i in range(1, n + 1)])
    print(pca.explained_variance_ratio_)
    print('Total variance explained by PCA is {}'.format(sum(pca.explained_variance_ratio_)))
    
    principalDF = principalDF.set_index(df.index)
    
    return principalDF

performPCA(25, True).to_csv('pca_team_stats_all.csv')


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

def getRollingAverage(gameId, n, home = True):
    if home == True:
        games = getRecentNGames(gameId, n, getTeams(np.arange(2015, 2023)).loc[gameId]['teamHome'])
    else:
        games = getRecentNGames(gameId, n, getTeams(np.arange(2015, 2023)).loc[gameId]['teamAway'])
    df = pd.read_csv('../data/teamStats/pca_team_stats_all.csv', index_col = 0)
    df = df[df.index.isin(games)]

    return df.mean()
    
    
def getTeams(years):
    df = pd.DataFrame()
    for year in years:
        teamDF = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0)
        teams = pd.concat([teamDF['gameState']['teamHome'],teamDF['gameState']['teamAway']], axis = 1)
        df = pd.concat([df, teams], axis = 0)

    return df
    
def getRollingAverageDF(n):
    df = pd.read_csv('../data/teamStats/pca_team_stats_all.csv', index_col = 0)
    avgDF = pd.DataFrame(index = df.index, columns = df.columns)
    for gameId in df.index:
        avgDF.loc[gameId] = getRollingAverage(gameId, n, True)
        
    return avgDF
    

getRollingAverageDF(8).to_csv('avg_8_PCA.csv')
