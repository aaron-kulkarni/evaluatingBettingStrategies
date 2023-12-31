import re
import numpy as np
import pandas as pd
import datetime as dt

import pandas as pd
import sys

sys.path.insert(0, "..")
from utils.utils import *

import requests
import bs4 as bs
from urllib.request import urlopen

def concatOdds(file_1, file_2):
    df_1, df_2 = convertBettingOdds(file_1), convertBettingOdds(file_2)
    df = pd.concat([df_1, df_2], axis=0).drop_duplicates()
    df.dropna(axis=0, inplace=True)
    
    return df

def addReturns(filename = '../data/bettingOddsData/closing_betting_odds_all.csv'):
    df = pd.read_csv(filename, index_col=0, header=[0,1])
    col_drop = list(df.columns.get_level_values(1).unique())
    for col in df['OddHome'].columns:
        df['OddHome', '{}_return'.format(col)] = df.apply(lambda d: 1+convOdds(d['OddHome', col]), axis=1)
    for col in df['OddAway'].columns:
        df['OddAway', '{}_return'.format(col)] = df.apply(lambda d: 1+convOdds(d['OddAway', col]), axis=1)
    df.drop(col_drop, level=1,  axis=1, inplace=True)
    df.to_csv('../data/bettingOddsData/closing_betting_odds_returns.csv')

    return df

def convertBettingOdds(filename):
    """
    function does following:
    1. Removes games that are not in regular season
    2. Adds GameID
    3. Checks if all games in betting odds file are in regular season and vice versa
    4. Pivots table
    """

    year = re.findall('[0-9]+', filename)[0]
    teamDict = getTeamDict()

    df = pd.read_csv(filename, sep=';')
    df = df[df['Home_id'].isin(list(teamDict.keys()))]
    df = df[df['Away_id'].isin(list(teamDict.keys()))]

    df['game_id'] = df.apply(lambda d: '{}0{}'.format(pd.to_datetime(d['Date']).strftime('%Y%m%d'), teamDict[d['Home_id']]), axis = 1)
    gameIdList = getGameIdList(year)
    df = df[df['game_id'].isin(gameIdList)]

    df = df.pivot(index = 'game_id', columns = 'Bookmaker', values = ['OddHome', 'OddAway'])

    return df

def findArb(odd_a, odd_b):
    oddA =  convOdds(odd_a) + 1
    oddB = convOdds(odd_b) + 1

    return (oddB)/(oddA + oddB), (oddA)/(oddA + oddB)


def computeImpliedProb(x):
    if pd.isna(x) == True:
        return x
    if x > 0:
        return 100/(x+100) * 100
    if x < 0:
        return (-x)/((-x)+100) * 100

def adjProb(x, y):
    if pd.isna(x) == True:
        return x
    if pd.isna(y) == True:
        return y
    return x/(x+y)

def convOdds(odd):
    if pd.isna(odd) == True:
        return odd

    if odd > 0:
        return odd/100 
    elif odd < 0:
        return - 100/odd 

def addImpliedProb(filename):
    odds_df = pd.read_csv(filename, index_col=0, header=[0,1])
        
    oddsHome = odds_df['OddHome'] 
    for col in oddsHome.columns:
        oddsHome['{} (%)'.format(col)] = oddsHome[col].apply(computeImpliedProb)
        oddsHome.drop(col, inplace = True, axis = 1)
        
    oddsAway = odds_df['OddAway']    
    for col in oddsAway.columns:
        oddsAway['{} (%)'.format(col)] = oddsAway[col].apply(computeImpliedProb)
        oddsAway.drop(col, inplace = True, axis = 1)

    df = pd.concat([oddsHome, oddsAway], axis = 1, keys = ['homeProb','awayProb'], join = 'inner')
        
    return df

def addAdjProb(filename):
    prob_df = addImpliedProb(filename)

    for col in prob_df['homeProb'].columns:
        prob_df['homeProbAdj', '{}'.format(col)] = prob_df.apply(lambda d: adjProb(d['homeProb'][col], d['awayProb'][col]), axis = 1)
    for col in prob_df['awayProb'].columns:
        prob_df['awayProbAdj', '{}'.format(col)] = prob_df.apply(lambda d: adjProb(d['awayProb'][col], d['homeProb'][col]), axis = 1)
        
    return prob_df 
        

#filename = '../data/bettingOddsData/closing_betting_odds_all.csv'

def convAmericanOdds(filename):
    df = pd.read_csv(filename, index_col = 0, header = [0,1])
    for col in df['OddHome'].columns:
        df['OddHome', '{}_return'.format(col)] = df.apply(lambda d: convOdds(d['OddHome', col]), axis = 1)
    for col in df['OddAway'].columns:
        df['OddAway', '{}_return'.format(col)] = df.apply(lambda d: convOdds(d['OddAway', col]), axis = 1)
        df.drop(col, axis = 1, inplace = True, level = 1)
    return df


#convertBettingOdds('../data/bettingOddsData/closing_betting_odds_2023_FIXED.csv').to_csv('../data/bettingOddsData/closing_betting_odds_2023_clean.csv')
#convAmericanOdds('../data/bettingOddsData/closing_betting_odds_all.csv').to_csv('../data/bettingOddsData/closing_betting_odds_returns.csv')
#addAdjProb('../data/bettingOddsData/closing_betting_odds_2023_clean.csv').to_csv('../data/bettingOddsData/adj_prob_2023.csv')



'''
---------------------
EXECUTABLE 
---------------------
'''

def fillBettingOdds(years):
    '''

    INITIALIZES DATAFRAME FOR PCA => CONCATS ALL SEPERATE ADJPROB CSVS AND REPLACES ALL NAN VALUES WITH THE ROW AVERAGE

    '''
    
    df = pd.DataFrame()
    for year in years:
        adjProb = pd.read_csv('../data/bettingOddsData/adj_prob_{}.csv'.format(year), header = [0,1], index_col = 0)
        df = pd.concat([df, adjProb], axis = 0)
    df.drop('homeProb', axis = 1, inplace = True, level = 0)
    df.drop('awayProb', axis = 1, inplace = True, level = 0)
    df_home, df_away = df['homeProbAdj'], df['awayProbAdj']
    df_home = df_home.apply(lambda row: row.fillna(row.mean()), axis = 1)
    df_away = df_away.apply(lambda row: row.fillna(row.mean()), axis = 1)
    df = pd.concat([df_home, df_away], axis=1, keys=['homeProbAdj', 'awayProbAdj'])
    
    return df

#fillBettingOdds(np.arange(2015,2024)).to_csv('../data/bettingOddsData/adj_prob_win_ALL.csv')

from TeamPerformance import * 

#df = pd.read_csv('../data/bettingOddsData/adj_prob_home_win_ALL.csv', index_col = 0)

#df['signal'] = getSignal()

'STANDARIZE DATA FEATURES TO UNIT SCALE (mean = 0 and variance = 1)'

from sklearn.preprocessing import StandardScaler

#features = list(df.columns[:-1])
#x = df.loc[:, features].values
#y = df.loc[:, ['signal']].values

#x = StandardScaler().fit_transform(x)

'BEGIN PERFORMING PCA'

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

def performPCA(n):
    pca = PCA(n_components = n)
    principalComponents = pca.fit_transform(x)
    principalDF = pd.DataFrame(data = principalComponents, columns = ['PCA{}'.format(i) for i in range(1, n + 1)])
    print(pca.explained_variance_ratio_)
    return principalDF

#dfFinal = performPCA(2).set_index(df.index)
#dfFinal.to_csv('PCA_2_betting_odds_all.csv')


def concatBettingOdds(years):
    df_all = pd.DataFrame()
    for year in years:
        df = pd.read_csv('../data/bettingOddsData/closing_betting_odds_{}_clean.csv'.format(year), header = [0,1], index_col = 0)
        df_all = pd.concat([df, df_all], axis = 0)
    df_all = df_all.reindex(sortDate(df_all.index))

    return df_all

#concatBettingOdds(np.arange(2015, 2024)).to_csv('../data/bettingOddsData/closing_betting_odds_all.csv')
