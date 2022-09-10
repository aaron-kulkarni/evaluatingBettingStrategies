import re
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
from sportsipy.nba.teams import Teams
import math

from TeamPerformance import *

import sys
sys.path.insert(0, "..")
from utils.utils import *
#from evaluatingBettingStrategies.utils.utils import *

class PerformanceMetric:

    def __init__(self, year):
        self.year = year
        self.adj_prob_df = pd.read_csv('../data/bettingOddsData/adj_prob_{}.csv'.format(self.year),
                                       index_col=0, header=[0, 1])

    def returnBettingOddsAverage(self, team):
        adjProb = self.adj_prob_df
        homeTeamSchedule, awayTeamSchedule = getTeamSchedule(team, self.year)

        adjProbHome = adjProb[adjProb.index.isin(homeTeamSchedule.index)]
        adjProbAway = adjProb[adjProb.index.isin(awayTeamSchedule.index)]

        adjProbHome['homeProbAdj', 'mean'] = adjProbHome['homeProbAdj'].mean(skipna=True, axis=1)
        adjProbAway['awayProbAdj', 'mean'] = adjProbAway['awayProbAdj'].mean(skipna=True, axis=1)
        mean = pd.concat([adjProbHome['homeProbAdj', 'mean'], adjProbAway['awayProbAdj', 'mean']], axis=0)
        mean = mean.sort_index(ascending=True)

        return mean

    def getSignal(self, team):
        homeTeamSchedule, awayTeamSchedule = getTeamSchedule(team, self.year)

        df = pd.concat([homeTeamSchedule['gameState', 'winner'], awayTeamSchedule['gameState', 'winner']],
                       axis=0).to_frame()['gameState']
        df = df.sort_index(ascending=True)
        df['signal'] = df.apply(lambda d: 1 if d['winner'] == team else 0, axis=1)

        return df['signal']

    def getPerformanceMetric(self, team, cum_sum=True):
        """
        Assumption - Probability of win in each game is independent (this
        is obviously not a true statement). Each variable is a Bernoulli
        random variable and thus the expectation of the number of
        wins they have is the sum of the expected values of each individual game

        ***CUMULATIVE***
        """

        if cum_sum:
            actWins = self.getSignal(team).cumsum()
            expWins = self.returnBettingOddsAverage(team).cumsum()
        else:
            actWins = self.getSignal(team)
            expWins = self.returnBettingOddsAverage(team)
        perMetric = actWins - expWins

        return perMetric

    def getPerformanceMetricN(self, team, n):

        actWins = self.getSignal(team)
        expWins = self.returnBettingOddsAverage(team)
        perMetric = (actWins - expWins).rolling(window=n).mean() * n

        return perMetric

    def getPerformanceMetricDataFrame(self):
        df = pd.DataFrame()
        for team in Teams():
            teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
            perMetric = self.getPerformanceMetric(teamAbbr, cum_sum=True).shift()
            perMetric = perMetric.to_frame(name='perMetric')
            perMetricN = self.getPerformanceMetricN(teamAbbr, 6).shift()
            perMetricN = perMetricN.to_frame(name='perMetricN')
            perTotal = pd.concat([perMetric, perMetricN], axis=1)
            perTotal['team'] = teamAbbr
            df = pd.concat([df, perTotal], axis=0)
        return df

    def convertDataFrame(self):
        df = self.getPerformanceMetricDataFrame()
        df.reset_index(inplace=True)
        df['homeTeam'] = df.apply(lambda d: 'home' if d['game_id'][-3:] == d['team'] else 'away', axis=1)
        dfPivot = df.pivot_table(index='game_id', columns='homeTeam', values=['perMetric', 'perMetricN'])
        dfPivot['game', 'year'] = self.year

        return dfPivot

def concatPerMetric(years):
    df_all = pd.DataFrame()
    for year in years:
        df = pd.read_csv('../data/perMetric/performance_metric_{}.csv'.format(year), index_col = 0, header = [0,1])
        df_all = pd.concat([df_all, df], axis = 0)
    df_all.drop('year', inplace = True, level = 1, axis = 1)
    return df_all
    
years = np.arange(2015, 2023)
for year in years:
    PerformanceMetric(year).convertDataFrame().to_csv('performance_metric_{}.csv'.format(year))

concatPerMetric(years).to_csv('../data/perMetric/performance_metric_all.csv')
    
year = 2018
n = 10
pm = PerformanceMetric(year)
for team in Teams():
    teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
    x = np.arange(1, len(list(pm.getPerformanceMetricN(teamAbbr, n))) + 1)
    y = list(pm.getPerformanceMetricN(teamAbbr, n))
    plt.plot(x, y, label=teamAbbr)
plt.xlabel('Games')
plt.ylabel('Performance Metric')
plt.legend()
plt.show()

year = 2015
pm = PerformanceMetric(year)
for team in Teams():
    teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
    x = np.arange(1, len(list(pm.getPerformanceMetric(teamAbbr, True))) + 1)
    y = list(pm.getPerformanceMetric(teamAbbr))
    plt.plot(x, y, label=teamAbbr)
plt.xlabel('Games')
plt.ylabel('Performance Metric')
plt.legend()
plt.show()
