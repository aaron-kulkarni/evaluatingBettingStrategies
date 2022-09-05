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

    def getSignal(self):
        df = pd.DataFrame()
        years = np.arange(2015, 2023)
        for year in years:
            dfCurrent = pd.DataFrame(
                pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1]))
            df = pd.concat([df, dfCurrent], axis=0)
        df = df['gameState']
        df['signal'] = df.apply(lambda d: 1 if d['winner'] == d['teamHome'] else 0, axis=1)
        return df['signal']

