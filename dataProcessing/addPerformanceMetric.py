import re
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
from sportsipy.nba.teams import Teams
import sys
import math

from teamPerformance import getTeamSchedule, getTeamGameIds, getNumberGamesPlayed, getYearFromId, getTeams


def returnBettingOddsAverage(team, year):
    adjProb = pd.read_csv('../data/bettingOddsData/adj_prob_{}.csv'.format(year), index_col = 0, header = [0,1])
    homeTeamSchedule, awayTeamSchedule = getTeamSchedule(team, year)
    
    adjProbHome = adjProb[adjProb.index.isin(homeTeamSchedule.index)]
    adjProbAway = adjProb[adjProb.index.isin(awayTeamSchedule.index)]

    adjProbHome['homeProbAdj', 'mean'] = adjProbHome['homeProbAdj'].mean(skipna = True, axis = 1)
    adjProbAway['awayProbAdj', 'mean'] = adjProbAway['awayProbAdj'].mean(skipna = True, axis = 1)
    mean = pd.concat([adjProbHome['homeProbAdj', 'mean'], adjProbAway['awayProbAdj', 'mean']], axis = 0)
    mean = mean.sort_index(ascending = True)

    return mean

def getSignal(team, year):
    homeTeamSchedule, awayTeamSchedule = getTeamSchedule(team, year)

    df = pd.concat([homeTeamSchedule['gameState', 'winner'], awayTeamSchedule['gameState', 'winner']], axis = 0).to_frame()['gameState']
    df = df.sort_index(ascending = True)
    df['signal'] = df.apply(lambda d: 1 if d['winner'] == team else 0, axis = 1)

    return df['signal']


def getPerformanceMetric(team, year, cumSum = True):
    '''
    Assumption - Probability of win in each game is independent(this is obviously not a true statement), each variable is a Bernoulli random variable and thus the expectation of the number of wins they have is the sum of the expected values of each individual game)

***CUMULATIVE***
    '''

    if cumSum == True:
        actWins = getSignal(team, year).cumsum()
        expWins = returnBettingOddsAverage(team, year).cumsum()
    else:
        actWins = getSignal(team, year)
        expWins = returnBettingOddsAverage(team, year)
    perMetric = actWins - expWins
    
    return perMetric


def getPerformanceMetricN(team, year, n):
    
    actWins = getSignal(team, year)
    expWins = returnBettingOddsAverage(team, year)
    perMetric = (actWins - expWins).rolling(window = n).mean() * n

    return perMetric

def getPerformanceMetricDataFrame(year):
    df = pd.DataFrame()
    for team in Teams():
        teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
        perMetric = getPerformanceMetric(teamAbbr, year, cumSum = True).shift()
        perMetric = perMetric.to_frame(name = 'perMetric')
        perMetricN = getPerformanceMetricN(teamAbbr, year, 6).shift()
        perMetricN = perMetricN.to_frame(name = 'perMetricN')
        perTotal = pd.concat([perMetric, perMetricN], axis = 1)
        perTotal['team'] = teamAbbr
        df = pd.concat([df, perTotal], axis = 0)
    return df

def convertDataFrame(year):
    df = getPerformanceMetricDataFrame(year)
    df.reset_index(inplace = True)
    df['homeTeam'] = df.apply(lambda d: 'home' if d['game_id'][-3:] == d['team'] else 'away', axis = 1)
    dfPivot = df.pivot_table(index = 'game_id', columns = 'homeTeam', values = ['perMetric', 'perMetricN'])
    dfPivot['game', 'year'] = year 

    return dfPivot

years = np.arange(2015, 2023)
for year in years:
    convertDataFrame(year).to_csv('performance_metric_{}.csv'.format(year))

                 

year = 2018
n = 10
for team in Teams():
    teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
    x = np.arange(1, len(list(getPerformanceMetricN(teamAbbr, year, n))) + 1)
    y = list(getPerformanceMetricN(teamAbbr, year, n))
    plt.plot(x, y, label = teamAbbr)
plt.xlabel('Games')
plt.ylabel('Performance Metric')
plt.legend()
plt.show() 


year = 2015
for team in Teams():
    teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
    x = np.arange(1, len(list(getPerformanceMetric(teamAbbr, year, True))) + 1)
    y = list(getPerformanceMetric(teamAbbr, year))
    plt.plot(x, y, label = teamAbbr)
plt.xlabel('Games')
plt.ylabel('Performance Metric')
plt.legend()
plt.show()
