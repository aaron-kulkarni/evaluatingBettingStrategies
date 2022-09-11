import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "..")
from utils.utils import *


class TeamPerformance:

    def __init__(self, year):
        self.year = year
        self.team_df = pd.read_csv('../data/teamStats/team_total_stats_{}.csv'.format(self.year),
                                   index_col=0, header=[0, 1])
        self.player_df = pd.read_csv('../data/gameStats/game_data_player_stats_{}.csv'.format(self.year),
                                     index_col=0, header=[0])
        self.adj_prob_df = pd.read_csv('../data/bettingOddsData/adj_prob_{}.csv'.format(self.year),
                                       index_col=0, header=[0, 1])
        self.game_state_df = pd.read_csv('../data/gameStats/game_state_data{}.csv'.format(self.year),
                                       index_col=0, header=[0, 1])

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

        df = pd.concat([df1['home'], df2['home']], axis = 0)

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

        df = pd.concat([df1['home'], df2['home']], axis = 0)

        df.loc[gameId] = df.mean()

        df['teamAbbr'] = team
        
        return df.loc[gameId]

    def getTeamPerformanceDF(self, n, home):

        teamDF = self.game_state_df
        if home:
            teamDF = teamDF['gameState']['teamHome']
        else:
            teamDF = teamDF['gameState']['teamAway']
            
        df = pd.DataFrame()
        gameIdList = teamDF.index
        for gameId in gameIdList:
            team = teamDF.loc[gameId]
            teamTotalStats = pd.concat([self.getTeamAveragePerformance(gameId, n, team), self.getOpponentAveragePerformance(gameId, n, team)], axis = 0, keys = ['home', 'opp'], join = 'inner')
            teamTotalStats = teamTotalStats.to_frame()
            df = pd.concat([df, teamTotalStats], axis = 0)

        return df

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