import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sportsipy.nba.teams import Teams

from TeamPerformance import *

import sys
sys.path.insert(0, "..")
from utils.utils import *

class PerformanceMetric:

    def __init__(self, year):
        self.year = year
        self.adj_prob_df = pd.read_csv('../data/bettingOddsData/adj_prob_{}.csv'.format(self.year), index_col=0, header=[0, 1])
        self.elo = pd.read_csv('../data/eloData/nba_elo_all.csv', index_col = 0)
        self.elo_df = self.elo[self.elo['season'] == self.year]

    def returnBettingOdds(self): 
        adj_series = self.adj_prob_df['homeProbAdj'].mean(skipna=True, axis=1), self.adj_prob_df['awayProbAdj'].mean(skipna=True, axis=1), 
        return adj_series

    def returnTeamSeries(self, series, team):
        home_schedule, away_schedule = getTeamScheduleCSVSplit(team, self.year)
        series_home = series[series.index.isin(home_schedule.index)]
        series = 1 - series
        series_away = series[series.index.isin(away_schedule.index)]
        series_total = pd.concat([series_home, series_away], axis=0)
        series_total = series_total.sort_index(ascending=True)
        return series_total
    
    def getSignalTeam(self, team):
        return self.returnTeamSeries(getSignal(), team)

    def returnPerMetricTeam(self, series, team):
        exp_wins = series.cumsum()
        act_wins = self.getSignalTeam(team).cumsum()
        return (act_wins - exp_wins).shift()
        
    def returnPerMetricNTeam(self, series, team, n):
        return self.returnPerMetricTeam(series, team).rolling(window = n).mean()

    def returnPerMetric(self, series):
        
        '''
        Assumption - Probability of win in each game is independent (this is obviously not a true statement). Each variable is a Bernoulli random variable and thus the expectation of the number of wins they have is the sum of the expected values of each individual game
        
        '''
        
        df_all = getTeamsDF(self.year)
        for team in getAllTeams():
            perMetricTeam = self.returnPerMetricTeam(self.returnTeamSeries(series, team), team).to_frame(name=team)
            df_all = pd.concat([df_all, perMetricTeam], axis = 1)
        df_all['home'] =  df_all.apply(lambda d: d[d['teamHome']], axis=1)
        df_all['away'] =  df_all.apply(lambda d: d[d['teamAway']], axis=1)
        col = [['home','away'],['pm_{}'.format(series.name), 'pm_{}'.format(series.name)]]
        df = pd.DataFrame(columns = pd.MultiIndex.from_arrays(col))
        df['home', 'pm_{}'.format(series.name)] = df_all['home']
        df['away', 'pm_{}'.format(series.name)] = df_all['away']
        return df

    def returnPerMetricN(self, series, n):
        '''
        Rolling average of performance metrics
        
        '''
        df_all = getTeamsDF(self.year)
        for team in getAllTeams():
            perMetricNTeam = self.returnPerMetricNTeam(self.returnTeamSeries(series, team), team, n).to_frame(name=team)
            df_all = pd.concat([df_all, perMetricNTeam], axis = 1)
        df_all['home'] =  df_all.apply(lambda d: d[d['teamHome']], axis=1)
        df_all['away'] =  df_all.apply(lambda d: d[d['teamAway']], axis=1)

        col = [['home','away'],['pm_{}_{}'.format(n, series.name), 'pm_{}_{}'.format(n, series.name)]]
        df = pd.DataFrame(columns = pd.MultiIndex.from_arrays(col))
        df['home', 'pm_{}_{}'.format(n, series.name)] = df_all['home']
        df['away', 'pm_{}_{}'.format(n, series.name)] = df_all['away']
        return df

    def concatPerMetric(self, n):
        eloProb = self.elo_df['elo_prob1']
        raptorProb = self.elo_df['raptor_prob1']
        bettingOdds = self.returnBettingOdds()[0].rename('odd_prob')
        perMetricElo, perMetricOdds, perMetricRaptor = self.returnPerMetric(eloProb), self.returnPerMetric(bettingOdds), self.returnPerMetric(raptorProb)
        perMetricNElo, perMetricNOdds, perMetricNRatpor = self.returnPerMetricN(eloProb, n), self.returnPerMetricN(bettingOdds, n), self.returnPerMetricN(raptorProb, n)
        df = pd.concat([perMetricElo, perMetricOdds, perMetricRaptor, perMetricNElo, perMetricNOdds, perMetricNRatpor], axis = 1)
        df.index.name = 'game_id'
        return df

#years = np.arange(2015, 2023)
#dfAll = pd.DataFrame()
#for year in years:
#    df = pd.read_csv('../data/perMetric/performance_metric_{}.csv'.format(year), index_col=0, header=[0,1])
#    dfAll = pd.concat([df, dfAll], axis = 0)

def updatePerMetric(year):
    df = PerformanceMetric(year).concatPerMetric(6)
    df.to_csv('../data/perMetric/performance_metric_{}.csv'.format(year))
    #dfAll = concatAll(range(2015,int(year + 1)))
    #dfAll.to_csv('../data/perMetric/performance_metric_ALL.csv')
    return 

def concatAll(years):
    df = pd.DataFrame()
    for year in years:
        df_current = pd.read_csv('../data/perMetric/performance_metric_{}.csv'.format(year), header=[0,1], index_col=0)
        df = pd.concat([df, df_current], axis=0)
    
    df.to_csv('../data/perMetric/performance_metric_ALL.csv')
    return 




#year = 2018
#n = 10
#pm = PerformanceMetric(year)
#for team in Teams():
#    teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
#    x = np.arange(1, len(list(pm.getPerformanceMetricN(teamAbbr, n))) + 1)
#    y = list(pm.getPerformanceMetricN(teamAbbr, n))
#    plt.plot(x, y, label=teamAbbr)
#plt.xlabel('Games')
#plt.ylabel('Performance Metric')
#plt.legend()
#plt.show()

#year = 2015
#pm = PerformanceMetric(year)
#for team in Teams():
#    teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
#    x = np.arange(1, len(list(pm.getPerformanceMetric(teamAbbr, True))) + 1)
#    y = list(pm.getPerformanceMetric(teamAbbr))
#    plt.plot(x, y, label=teamAbbr)
#plt.xlabel('Games')
#plt.ylabel('Performance Metric')
#plt.legend()
#plt.show()
