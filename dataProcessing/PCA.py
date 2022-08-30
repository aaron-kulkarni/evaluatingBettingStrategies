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


