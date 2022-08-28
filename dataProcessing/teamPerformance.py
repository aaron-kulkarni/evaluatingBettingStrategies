import numpy as np
import pandas as pd
from evaluatingBettingStrategies.utils.utils import getTeamSchedule


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

    def returnX(self, points_home, points_away, prob, home=True):
        if home:
            return (points_home / points_away) / prob
        if not home:
            return (points_away / points_home) / prob

    def getTeamPerformance(self, team):
        dfHome = getTeamSchedule(team, self.year)[0]
        dfAway = getTeamSchedule(team, self.year)[1]

        adjProb = self.adj_prob_df

        adjProbHome = adjProb.loc[adjProb.index.isin(dfHome.index)]
        adjProbAway = adjProb.loc[adjProb.index.isin(dfAway.index)]
        dfHome = pd.concat([dfHome, adjProbHome], join='inner', axis=1)
        dfAway = pd.concat([dfAway, adjProbAway], join='inner', axis=1)
        dfHome['homeProbAdj', 'mean'] = dfHome['homeProbAdj'].mean(skipna=True, axis=1)
        dfAway['awayProbAdj', 'mean'] = dfAway['awayProbAdj'].mean(skipna=True, axis=1)

        dfHome['per', 'val'] = self.returnX(dfHome['home']['points'], dfHome['away']['points'],
                                            dfHome['homeProbAdj']['mean'], True)
        dfAway['per', 'val'] = self.returnX(dfAway['home']['points'], dfAway['away']['points'],
                                            dfAway['awayProbAdj']['mean'], False)

        df = pd.concat([dfHome, dfAway], axis=0)
        df.sort_index(ascending=True)

        return df['per']['val']

# def fixRecentStats(year):
#     df = pd.read_csv('data/averageTeamData/average_team_per_5_{}.csv'.format(year), header = [1])
#     #dfPlayer['MP'] = dfPlayer.apply(lambda d: round(float(d['MP'][0:2]) + (float(d['MP'][3:5])/60), 2), axis = 1)
#     #df['NanN'] = df.apply(lambda d: )
#     #df['<= 53'] = df['mynumbers'].apply(lambda x: 'True' if x <= 53 else 'False')
#     #df['Unnamed: 0'] = df['Unnamed: 0'].apply(lambda d: '' if d == 0 else d)
#     #df['gameId'] = df['gameId'].apply(lambda x: '' if x == 'NaN' else x)
#     df['firstCol'] = df['Unnamed: 0'].apply(lambda d: '' if d == '0' else d)
#     df['secondCol'] = df['gameId'].apply(lambda x: '' if type(x) == float else x)
#     df['fixed'] = df['firstCol'] + df['secondCol']
#     del df['Unnamed: 0']
#     del df['gameId']
#     del df['firstCol']
#     del df['secondCol']
#     df.set_index('fixed', inplace = True)
#     df.index.names = ['gameId']

#     return df

# rate = []
# for team in Teams():
#     teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
#     rate.extend(list(plotValues(teamAbbr, 2018, False)))

# plt.hist(np.log10(rate), density=True, bins=30) 

# #x = list(range(0, len(rate)))

# #plt.plot(x, rate)
# plt.show()

# fixRecentStats(2022).to_csv('data/averageTeamData/average_team_per_5_2022.csv')
# years = np.arange(2015, 2022)
# for year in years:
#     df = pd.read_csv('data/averageTeamData/average_team_per_5_{}.csv'.format(year), index_col = 0, header = [0,1])
#     print(df)
#     fixRecentStats(year).to_csv('average_team_per_5_{}.csv'.format(year))
