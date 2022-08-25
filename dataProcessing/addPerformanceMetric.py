import re
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
from sportsipy.nba.teams import Teams
import sys
import math

from teamPerformance import getTeamSchedule

def getTeamGameIds(team, year):
    homeTeamSchedule, awayTeamSchedule = getTeamSchedule(team, year)
    teamSchedule = pd.concat([homeTeamSchedule, awayTeamSchedule], axis = 0)
    teamSchedule = teamSchedule.sort_index(ascending = True)
    return list(teamSchedule.index)

def getTeamPerformance(team, year):
    dfHome = getTeamSchedule(team, year)[0]
    dfAway = getTeamSchedule(team, year)[1]
    
    adjProb = pd.read_csv('../data/bettingOddsData/adj_prob_{}.csv'.format(year), index_col = 0, header = [0,1])

    adjProbHome = adjProb.loc[adjProb.index.isin(dfHome.index)]
    adjProbAway = adjProb.loc[adjProb.index.isin(dfAway.index)]
    dfHome = pd.concat([dfHome, adjProbHome], join = 'inner', axis = 1)
    dfAway = pd.concat([dfAway, adjProbAway], join = 'inner', axis = 1)
    dfHome['homeProbAdj', 'mean'] = dfHome['homeProbAdj'].mean(skipna = True, axis = 1)
    dfAway['awayProbAdj', 'mean'] = dfAway['awayProbAdj'].mean(skipna = True, axis = 1)
    
    dfHome['per', 'val'] = returnX(dfHome['home']['points'], dfHome['away']['points'], dfHome['homeProbAdj']['mean'], True)
    dfAway['per', 'val'] = returnX(dfAway['home']['points'], dfAway['away']['points'], dfAway['awayProbAdj']['mean'], False)

    df = pd.concat([dfHome, dfAway], axis = 0)
    df.sort_index(ascending = True)
    
    return df['per']['val']

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
    '''
    actWins = getSignal(team, year).cumsum()
    expWins = returnBettingOddsAverage(team, year).cumsum()
    perMetric = actWins - expWins
    
    return perMetric

    

def plotValues(team, year, cumulative = True):
    df = getTeamPerformance(team, year)
    if cumulative == True:
        df = df.expanding().mean()
    return df.array


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

# rate = []
# for team in Teams():
#     teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
#     rate.extend(list(plotValues(teamAbbr, 2018, False)))

# plt.hist(np.log10(rate), density=True, bins=30) 

# #x = list(range(0, len(rate)))

# #plt.plot(x, rate)
# plt.show()

