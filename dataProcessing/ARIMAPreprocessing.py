import pandas as pd

from evaluatingBettingStrategies.utils.utils import gameIdToDateTime, getYearFromId
from TeamPerformance import TeamPerformance
import datetime as dt

class ARIMAPreprocessing:

    def __init__(self, year):
        self.year = year
        self.player_df = pd.read_csv('../data/gameStats/game_data_player_stats_{}.csv'.format(self.year),
                                     index_col=0, header=0)
        self.team_df = pd.read_csv('../data/teamStats/team_total_stats_{}.csv'.format(self.year),
                                   index_col=0, header=[0, 1])
        self.game_df = pd.read_csv('data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1])

    def getPreviousGamePlayerStats(self, game_id):
        """
        Gets all numerical stats in game_data_player_stats for games
        that took place before (and not including) the given game, for each
        player on the roster in the given game. Does not include player
        statistics from when a player was benched the entire game. Does not
        include the name or minutes played columns.

        Parameters
        ----------
        game_id : the basketball-reference.com id of the game

        Returns
        -------
        a pandas dataframe with all the players' statistics for each
        previous game.
        """

        df = self.player_df

        playerIdList = self.getPlayerIdsFromGame(game_id)

        df = df[df['playerid'].isin(playerIdList)]
        df = df[pd.to_datetime(df['gameid'].str.slice(0, 8), format="%Y%m%d") < gameIdToDateTime(game_id)]
        df = df[~pd.isna(df['MP'])]
        df = df.drop(columns=['MP', 'Name'])
        return df

    def getPreviousGameSinglePlayerStats(self, game_id, player_id):
        df = self.getPreviousGamePlayerStats(game_id)
        return df[df['playerid'] == player_id]

    def getPreviousGameTeamStats(self, game_id):
        """
        Gets all numerical stats in team_total_stats for games that
        took place before (and not including) the given game, for each
        team playing in the given game.

        Parameters
        ----------
        game_id : the basketball-reference.com id of the game

        Returns
        -------
        a pandas dataframe with all the players' statistics for each
        previous game.
        """

        df = self.team_df
        dfTemp = df.loc[game_id]
        teamNameHome = dfTemp['home']['teamAbbr']
        teamNameAway = dfTemp['away']['teamAbbr']

        dfHome = self.getPreviousGameSingleTeamStats(game_id, teamNameHome)
        dfAway = self.getPreviousGameSingleTeamStats(game_id, teamNameAway)

        df = pd.concat([dfHome, dfAway], axis=0)
        df = df.drop(index=game_id)

        return df

    def getPreviousGameSingleTeamStats(self, game_id, team):
        df = self.team_df

        dfHome = df[df['home']['teamAbbr'] == team]
        dfAway = df[df['away']['teamAbbr'] == team]

        df = pd.concat([dfHome['home'], dfAway['away']], axis=0)
        df.sort_index(inplace=True)
        df = df[:game_id]

        return df

    def getPlayerIdsFromGame(self, game_id):
        df = self.game_df
        df = df.loc[game_id]

        homeList = df['home']['playerRoster']
        awayList = df['away']['playerRoster']
        homeList = homeList.replace("'", "").replace("]", "").replace("[", "").replace(" ", "").split(",")
        awayList = awayList.replace("'", "").replace("]", "").replace("[", "").replace(" ", "").split(",")
        homeList.extend(awayList)
        return homeList


print(ARIMAPreprocessing(2015).getPreviousGameSinglePlayerStats('201601090LAC', 'linje01'))
