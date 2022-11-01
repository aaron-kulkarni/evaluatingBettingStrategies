import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "..")
from utils.utils import *


class TeamPerformance:

    def __init__(self, year):
        self.year = year
        self.team_df = pd.read_csv('../data/teamStats/team_total_stats_{}.csv'.format(self.year), index_col=0, header=[0, 1])
#        self.player_df = pd.read_csv('../data/gameStats/game_data_player_stats_{}.csv'.format(self.year), index_col=0, header=[0])
        self.adj_prob_df = pd.read_csv('../data/bettingOddsData/adj_prob_{}.csv'.format(self.year), index_col=0, header=[0, 1])
        self.game_state_df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(self.year),index_col=0, header=[0, 1])

    def teamAverageHelper(self, team):
        df = self.team_df

        dfHome = df[df['home']['teamAbbr'] == team]
        dfAway = df[df['away']['teamAbbr'] == team]
        return dfHome, dfAway

    def opponentAverageHelper(self, team):
        df = self.team_df

        dfHome = df[df['home']['teamAbbr'] != team]
        dfAway = df[df['away']['teamAbbr'] != team]
        return dfHome, dfAway

    def playerAverageHelper(self, player_id):
        df = self.player_df

        dfPlayer = df[df['playerid'] == player_id]
        # dfPlayer = df['playerid'] == playerId
        return dfPlayer
    
    def getTeamAveragePerformance(self, gameId, n, team):
        '''
        Returns a row of data of average team performances in last n games
   
        '''
        try:
            gameIdList = getRecentNGames(gameId, n, team)
        except:
            s = pd.Series('NaN', index=['teamAbbr','MP','FG','FGA','FG%','3P','3PA','3P%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','TS%','eFG%','3pAr','FTr','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%','USG%','Ortg','Drtg','poss','pace','poss_per_poss','ass_per_poss'])
            s.name = gameId
            return s

        #trying to only return the team stats of the team that we are asking for, rather than the team plus their opponents
        df1, df2 = self.teamAverageHelper(team)
        
        df1 = df1[df1.index.isin(gameIdList)]
        df2 = df2[df2.index.isin(gameIdList)]

        df = pd.concat([df1['home'], df2['away']], axis = 0)

        df.loc[gameId] = df.mean()

        df['teamAbbr'] = team
        
        return df.loc[gameId]

    def getPlayerAveragePerformance(self, gameId, n, team, playerId):
        '''
        Returns a row of data of average player performances in last n games
        
        '''
        try:
            gameIdList = getRecentNGames(gameId, n, team)
        except:
            s = pd.Series('NaN', index=['teamAbbr','MP','FG','FGA','FG%','3P','3PA','3P%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','TS%','eFG%','3pAr','FTr','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%','USG%','Ortg','Drtg','poss','pace','poss_per_poss','ass_per_poss'])
            s.name = gameId
            return s

        dfPlayer = self.playerAverageHelper(playerId)

        dfPlayer = dfPlayer[dfPlayer['gameid'].isin(gameIdList)]

        dfPlayer['MP'] = dfPlayer.apply(lambda d: round(float(d['MP'][0:2]) + (float(d['MP'][3:5])/60), 2), axis = 1)

        storedName = dfPlayer.iloc[1]['Name']
        
        dfPlayer.loc['mean'] = dfPlayer.mean()
        dfPlayer['playerid'] = playerId
        dfPlayer['Name'] = storedName
        dfPlayer.at['mean','gameid'] = gameId

        return dfPlayer.loc['mean']

    def getOpponentAveragePerformance(self, gameId, n, team):

        '''
        Returns a row of data of average opposing team performances in last n games
    
        '''
        try:
            gameIdList = getRecentNGames(gameId, n, team)
        except:
            s = pd.Series('NaN', index=['teamAbbr','MP','FG','FGA','FG%','3P','3PA','3P%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','TS%','eFG%','3pAr','FTr','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%','USG%','Ortg','Drtg','poss','pace','poss_per_poss','ass_per_poss'])
            s.name = gameId
            return s

        df1, df2 = self.opponentAverageHelper(team)

        df1 = df1[df1.index.isin(gameIdList)]
        df2 = df2[df2.index.isin(gameIdList)]

        df = pd.concat([df1['home'], df2['away']], axis = 0)

        df.loc[gameId] = df.mean()

        df['teamAbbr'] = team
        
        return df.loc[gameId]

    def getTeamPerformanceDF(self, n, home):
        if home == True:
            teams = getTeamsDF(self.year)['teamHome']
        if home == False:
            teams = getTeamsDF(self.year)['teamAway']
        df = pd.DataFrame()
        for gameId in getYearIds(self.year):
            team = teams.loc[gameId]
            team_df = pd.concat([self.getTeamAveragePerformance(gameId, n, team), self.getOpponentAveragePerformance(gameId, n, team)], axis = 0, keys = ['team', 'opp'], join = 'inner')
            df = pd.concat([df, pd.DataFrame(team_df).T], axis=0)
        df.index.name = 'gameId' 
        if home == True:
            df.to_csv('../data/averageTeamData/per_{}/average_team_per_{}_{}.csv'.format(n, n, self.year))
        if home == False:
            df.to_csv('../data/averageTeamData/per_{}/average_away_per_{}_{}.csv'.format(n, n, self.year))
        return df

#TeamPerformance(2023).getTeamPerformanceDF(5, True)
#TeamPerformance(2023).getTeamPerformanceDF(5, False)

def concat(n, years):
    df_all = pd.DataFrame()
    for year in years:
        df_home = pd.read_csv('../data/averageTeamData/per_{}/average_team_per_{}_{}.csv'.format(n, n, year), index_col=0, header=[0,1])['team']
        df_away = pd.read_csv('../data/averageTeamData/per_{}/average_away_per_{}_{}.csv'.format(n, n, year), index_col=0, header=[0,1])['team']
        df_home['home'], df_away['home'] = 1, 0
        df_home.reset_index(inplace = True)
        df_away.reset_index(inplace = True)
        df_home.set_index(['gameId', 'home'], inplace=True)
        df_away.set_index(['gameId', 'home'], inplace=True)
        df_year = pd.concat([df_home, df_away], axis=0)
        df_year = df_year.reindex(sortDateMulti(df_year.index.get_level_values(0).unique()))
        df_all = pd.concat([df_all, df_year], axis=0)
    df_all.to_csv('../data/averageTeamData/average_team_stats_per_{}.csv'.format(n))
    return df_all

#concat(5, np.arange(2015,2024))

def getSignal():
    df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', index_col=0, header=[0,1])
    signal = pd.DataFrame(df['gameState'])
    signal['signal'] = signal.apply(lambda d: return_signal(d['winner'], d['teamHome'], d['teamAway']), axis=1)
    signal = signal.dropna(axis=0)
    signal['signal'] = signal['signal'].apply(int)
    return signal['signal']

def return_signal(winner,home_team,away_team):
    if home_team == winner:
        return int(1)
    if away_team == winner:
        return int(0)
    else:
        return winner

    
    
