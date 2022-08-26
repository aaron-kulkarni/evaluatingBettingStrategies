import re
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
from sportsipy.nba.teams import Teams
import sys
import math

from teamPerformance import getTeamSchedule, getTeamGameIds


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


def getPerformanceMetric(team, year):
    '''
    Assumption - Probability of win in each game is independent(this is obviously not a true statement), each variable is a Bernoulli random variable and thus the expectation of the number of wins they have is the sum of the expected values of each individual game)

***CUMULATIVE***
    '''
    actWins = getSignal(team, year).cumsum()
    expWins = returnBettingOddsAverage(team, year).cumsum()
    perMetric = actWins - expWins
    
    return perMetric


def getPerformanceMetricN(team, year, n):
    
    actWins = getSignal(team, year)
    expWins = returnBettingOddsAverage(team, year)
    perMetric = (actWins - expWins).rolling(window = n).mean().iloc[n-1:].values

    return perMetric


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
    x = np.arange(1, len(list(getPerformanceMetric(teamAbbr, year))) + 1)
    y = list(getPerformanceMetric(teamAbbr, year))
    plt.plot(x, y, label = teamAbbr)
plt.xlabel('Games')
plt.ylabel('Performance Metric')
plt.legend()
plt.show()
