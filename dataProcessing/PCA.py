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

#years = np.arange(2015, 2023)
#concatTeamStats(years).to_csv('team_total_stats_all.csv')

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


def getDataFrame(): 
    df = pd.read_csv('../data/teamStats/team_total_stats_all.csv', index_col = 0, header = [0,1])
    df.drop('teamAbbr', axis = 1, level = 1, inplace = True)
 
    colHome = ['home_{}'.format(col) for col in df['home'].columns]
    colAway = ['away_{}'.format(col) for col in df['away'].columns]
    df.columns = colHome + colAway
    df['signal'] = getSignal()

    return df

def getGameIdBySignal():
    df = getDataFrame()
    
    gameIdWin = df[df['signal'] == 1].index
    gameIdLoss = df[df['signal'] == 0].index
 
    return gameIdWin, gameIdLoss

def getStandardDF(df):
    from sklearn.preprocessing import StandardScaler
    
    features = list(df.columns[:-1])
    x_select = df.loc[:, features].values
    y_select = getSignal().values
    x_select = StandardScaler().fit_transform(x_select)
    dfSTD = pd.DataFrame(x_select, index = df.index, columns = df.columns[:-1])
    return dfSTD

def getStandardDFBySignal(df):
    df = getStandardDF(df)
    gameIdWin, gameIdLoss = getGameIdBySignal()
    dfWin = df[df.index.isin(gameIdWin)]
    dfLoss = df[df.index.isin(gameIdLoss)]

    return dfWin, dfLoss
    
def performPCA(df, n):
    '''
    input requires a standardized dataframe 
    
    '''
    
    x_select = df.to_numpy()

    from sklearn.decomposition import PCA
    
    pca = PCA(n_components = n)
    principalComponents = pca.fit_transform(x_select)
    principalDF = pd.DataFrame(data = principalComponents, columns = ['PCA{}'.format(i) for i in range(1, n + 1)])
    print(pca.explained_variance_ratio_)
    print('Total variance explained by PCA is {}'.format(sum(pca.explained_variance_ratio_)))

    weight = pd.DataFrame(data = pca.components_, index = principalDF.columns, columns = df.columns)

    principalDF = principalDF.set_index(df.index)
    
    return principalDF, weight

'''
EXECUTION 
----------------------------------------
'''

#stdDF = getStandardDF(getDataFrame())
#principalDF, weight = performPCA(stdDF, 25)
#print(weight)

#dfWin, dfLoss = getStandardDFBySignal(getDataFrame())
#principalDF_win, weight_win = performPCA(dfWin, 25)
#principalDF_loss, weight_loss = performPCA(dfLoss, 25)
#print(weight_win)
#print(weight_loss)
#principalDF_win.to_csv('pca_team_stats_all_win.csv')
#principalDF_loss.to_csv('pca_team_stats_all_loss.csv')

#weight_win.to_csv('coefficient_matrix_win.csv')
#weight_loss.to_csv('coefficient_matrix_loss.csv')
#weight.to_csv('coefficient_matrix.csv')

'''
----------------------------------------
Filler functions since import * does not work

'''

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

def getTeams(years):
    df = pd.DataFrame()
    for year in years:
        teamDF = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0)
        teams = pd.concat([teamDF['gameState']['teamHome'],teamDF['gameState']['teamAway']], axis = 1)
        df = pd.concat([df, teams], axis = 0)

    return df

'''
END OF FILLER FUNCTIONS
----------------------------------------
'''

    
def getRollingAverage(filename, gameId, n, home = True):
    if home == True:
        games = getRecentNGames(gameId, n, getTeams(np.arange(2015, 2023)).loc[gameId]['teamHome'])
    else:
        games = getRecentNGames(gameId, n, getTeams(np.arange(2015, 2023)).loc[gameId]['teamAway'])
    df = pd.read_csv(filename, index_col = 0)
    df = df[df.index.isin(games)]

    return df.mean()


def getRollingAverageDF(filename, n, home = True):
    df = pd.read_csv(filename, index_col = 0)
    avgDF = pd.DataFrame(index = df.index, columns = df.columns)
    for gameId in df.index:
        avgDF.loc[gameId] = getRollingAverage(filename, gameId, n, home)
    return avgDF


def convertLossDF(win = True):
    df = getStandardDF(getDataFrame())
    if win == True:
        coeff = pd.read_csv('../data/teamStats/PCA_breakdown/coefficient_matrix_win.csv', index_col = 0)
    else:
        coeff = pd.read_csv('../data/teamStats/PCA_breakdown/coefficient_matrix_loss.csv', index_col = 0)

    conv = pd.DataFrame(index = df.index)
    
    for i in range(0, len(coeff.index)):
        arr = coeff.loc[coeff.index[i]].values
        df['{}*'.format(coeff.index[i])] = df.apply(lambda d: d.dot(arr), axis = 1)
        conv['{}*'.format(coeff.index[i])] = df['{}*'.format(coeff.index[i])]
        df.drop('{}*'.format(coeff.index[i]), inplace = True, axis =1)

    return conv


'''
EXECUTION 
----------------------------------------
'''

#getStandardDF(getDataFrame().drop('signal', axis = 1)).to_csv('standard_team_stats_all.csv')

#convertLossDF(True).to_csv('pca_by_win_all.csv')
#convertLossDF(False).to_csv('pca_by_loss_all.csv')

#getRollingAverageDF('../data/teamStats/pca_by_win_all.csv', 5, True).to_csv('avg_5_PCA_home_win_all.csv')
#getRollingAverageDF('../data/teamStats/pca_by_win_all.csv', 5, False).to_csv('avg_5_PCA_away_win_all.csv') 
#getRollingAverageDF('../data/teamStats/pca_by_loss_all.csv', 5, True).to_csv('avg_5_PCA_home_loss_all.csv') 
#getRollingAverageDF('../data/teamStats/pca_by_loss_all.csv', 5, False).to_csv('avg_5_PCA_away_loss_all.csv') 

