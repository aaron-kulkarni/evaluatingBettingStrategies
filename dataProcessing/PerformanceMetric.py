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
        home_schedule, away_schedule = getTeamScheduleCSV(team, self.year)
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
        
        df = getTeamsDF(self.year)
        for team in getAllTeams():
            perMetricTeam = self.returnPerMetricTeam(self.returnTeamSeries(series, team), team).to_frame(name=team)
            df = pd.concat([df, perMetricTeam], axis = 1)
        df['perMetricHome'] =  df.apply(lambda d: d[d['teamHome']], axis=1)
        df['perMetricAway'] =  df.apply(lambda d: d[d['teamAway']], axis=1)
        return df[['perMetricHome', 'perMetricAway']]

    def returnPerMetricN(self, series, n):
        '''
        Rolling average of performance metrics
        
        '''
        df = getTeamsDF(self.year)
        for team in getAllTeams():
            perMetricNTeam = self.returnPerMetricNTeam(self.returnTeamSeries(series, team), team, n).to_frame(name=team)
            df = pd.concat([df, perMetricNTeam], axis = 1)
        df['perMetric{}Home'.format(n)] =  df.apply(lambda d: d[d['teamHome']], axis=1)
        df['perMetric{}Away'.format(n)] =  df.apply(lambda d: d[d['teamAway']], axis=1)
        return df[['perMetric{}Home'.format(n), 'perMetric{}Away'.format(n)]]



    '''
    -----------------------------------------------
    REDUNDANT
    
    '''
    
    def getSignal(self, team):
        homeTeamSchedule, awayTeamSchedule = getTeamScheduleCSV(team, self.year)
        df = pd.concat([homeTeamSchedule['gameState', 'winner'], awayTeamSchedule['gameState', 'winner']], axis=0).to_frame()['gameState']
        df = df.sort_index(ascending=True)
        df['signal'] = df.apply(lambda d: 1 if d['winner'] == team else 0, axis=1)
        return df['signal']
    
    def returnBettingOddsAverage(self, team):
        adjProb = self.adj_prob_df
        homeTeamSchedule, awayTeamSchedule = getTeamScheduleCSV(team, self.year)

        adjProbHome = adjProb[adjProb.index.isin(homeTeamSchedule.index)]
        adjProbAway = adjProb[adjProb.index.isin(awayTeamSchedule.index)]

        adjProbHome['homeProbAdj', 'mean'] = adjProbHome['homeProbAdj'].mean(skipna=True, axis=1)
        adjProbAway['awayProbAdj', 'mean'] = adjProbAway['awayProbAdj'].mean(skipna=True, axis=1)
        mean = pd.concat([adjProbHome['homeProbAdj', 'mean'], adjProbAway['awayProbAdj', 'mean']], axis=0)
        mean = mean.sort_index(ascending=True)

        return mean


    def returnBettingOddProbs(self, team, oddName):
        adjProb = self.adj_prob_df
        homeTeamSchedule, awayTeamSchedule = getTeamScheduleCSV(team, self.year)

        adjProbHome = adjProb[adjProb.index.isin(homeTeamSchedule.index)]
        adjProbAway = adjProb[adjProb.index.isin(awayTeamSchedule.index)]
        res = pd.concat([adjProbHome['homeProbAdj', oddName], adjProbAway['awayProbAdj', oddName]], axis = 0)
        res = res.sort_index(ascending=True)
        return res
        

    def returnEloData(self, team):
        elo_df = self.elo_df
        homeTeamSchedule, awayTeamSchedule = getTeamScheduleCSV(team, self.year) 

        elo_dfHome = elo_df[elo_df.index.isin(homeTeamSchedule.index)]['elo_prob1']
        elo_dfAway = elo_df[elo_df.index.isin(awayTeamSchedule.index)]['elo_prob2']
        raptor_elo_dfHome = elo_df[elo_df.index.isin(homeTeamSchedule.index)]['raptor_prob1']
        raptor_elo_dfAway = elo_df[elo_df.index.isin(awayTeamSchedule.index)]['raptor_prob2']
        elo_df = pd.concat([elo_dfHome, elo_dfAway], axis = 0)
        raptor_elo_df = pd.concat([raptor_elo_dfHome, raptor_elo_dfAway], axis = 0)
        elo_df = elo_df.sort_index(ascending = True)
        raptor_elo_df = raptor_elo_df.sort_index(ascending = True)
        return elo_df, raptor_elo_df


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

    def getPerformanceMetricElo(self, team, cum_sum=True, eloProb=True):
        elo, raptor = self.returnEloData(team)
        if cum_sum:
            actWins = self.getSignal(team).cumsum()
            if eloProb:
                expWins = elo.cumsum()
            else:
                expWins = raptor.cumsum()
        else:
            actWins = self.getSignal(team)
            if eloProb:
                expWins = elo
            else:
                expWins = raptor
        perMetric = actWins - expWins
        return perMetric

    def getPerformanceMetricN(self, team, n):

        perMetric = self.getPerformanceMetric(team, True).rolling(window=n).mean()
        return perMetric

    def getPerformanceMetricEloN(self, team, n, eloProb=True):
        perMetric = self.getPerformanceMetricElo(team, True, eloProb).rolling(window=n).mean() 
        return perMetric


    def getPerformanceMetricDataFrame(self, n):
        df = pd.DataFrame()
        for team in Teams():
            teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
            perMetric = self.getPerformanceMetric(teamAbbr, True).shift()
            perMetric = perMetric.to_frame(name='perMetric')
            perMetricN = self.getPerformanceMetricEloN(teamAbbr, n).shift()
            perMetricN = perMetricN.to_frame(name='perMetricN')
            perTotal = pd.concat([perMetric, perMetricN], axis=1)
            perTotal['team'] = teamAbbr
            df = pd.concat([df, perTotal], axis=0)
        return df

    def getPerformanceMetricEloDataFrame(self, n):
        df = pd.DataFrame()
        for team in Teams():
            teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
            perMetricElo = self.getPerformanceMetricElo(teamAbbr, True, True).shift().rename('perMetricElo')

            perMetricRaptor = self.getPerformanceMetricElo(teamAbbr, True, False).shift().rename('perMetricRaptor')
            
            perMetricEloN = self.getPerformanceMetricEloN(teamAbbr, n, True).shift().rename('perMetricEloN')

            perMetricRaptorN = self.getPerformanceMetricEloN(teamAbbr, n, False).shift().rename('perMetricRaptorN')
            
            perTotal = pd.concat([perMetricElo, perMetricRaptor, perMetricEloN, perMetricRaptorN], axis=1)
            perTotal['team'] = teamAbbr
            df = pd.concat([df, perTotal], axis=0)
        return df


    def convertDataFrame(self, df):
        df.reset_index(inplace=True)
        df['homeTeam'] = df.apply(lambda d: 'home' if d['game_id'][-3:] == d['team'] else 'away', axis=1)
        dfPivot = df.pivot_table(index='game_id', columns='homeTeam', values=['perMetric', 'perMetricN', 'perMetricElo', 'perMetricEloN', 'perMetricRaptor', 'perMetricRaptorN'])
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
    pm = PerformanceMetric(year)
    perMetric = pm.getPerformanceMetricDataFrame(6)
    eloMetric = pm.getPerformanceMetricEloDataFrame(6).drop('team', axis =1)
    df = pd.concat([eloMetric, perMetric], axis = 1)
    pm.convertDataFrame(df).to_csv('../data/perMetric/performance_metric_{}.csv'.format(year))

concatPerMetric(years).to_csv('../data/perMetric/performance_metric_all.csv')

df = pd.read_csv('../data/eloData/nba_elo_all.csv', index_col=0)
df = df[df['season'] == year]

    
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
