import pandas as pd
import numpy as np
from utils.utils import *


class TeamPerformance:

    def __init__(self, year):
        self.year = year
        self.team_shot_df = pd.read_csv('../data/shotData/team_shot_data_{}.csv'.format(self.year),
                                   index_col=0, header=0)
        self.shot_df = pd.read_csv('../data/shotData/shot_data_{}.csv'.format(self.year),
                                   index_col=0, header=0)

        self.angleWeight = 0.1
        self.distanceWeight = 0.9


    def getShotQuality(self, distance, angle, three):

        distanceQuality = 0
        angleQuality = 0

        if not three:
            if distance <= 2:
                distanceQuality = 1
            else:
                distanceQuality = 1 - ((distance - 24) * 0.05)
        else:
            distanceQuality = 1 - ((distance - 276) * 0.05)

        
        if angle >= 90:
            angleQuality = (-0.0455 * angle) + 4.4279
            #y = -0.0455x + 4.4279. well thought out and highly scientific formula
        else:
            angleQuality = (-1/135 * angle) + 1 
            #y = -1/135x + 1. anotha one

        return ((distanceQuality * self.distanceWeight) + (angleQuality * self.angleWeight))

    def getShotQualityList(self, distanceList, angleList, threeList):
        allLists = [distanceList, angleList, threeList]
        if len(set(map(len, allLists))) != 1:
            raise Exception("Issue with length of lists")
        
        res = list(map(self.getShotQuality, distanceList, angleList, threeList))
        return res
        
    def getAverageShotQuality(self, distanceList, angleList, threeList):
        res = self.getShotQualityList(distanceList, angleList, threeList)
        return sum(res)

    def getShotQualityDF(self):
        df = self.team_shot_df
        df['home_avg_shot_quality'] = df.apply(lambda d: self.getAverageShotQuality(eval(d['homeDistances']), eval(d['homeAngles']), eval(d['homeThrees'])), axis = 1)
        df['away_avg_shot_quality'] = df.apply(lambda d: self.getAverageShotQuality(eval(d['awayDistances']), eval(d['awayAngles']), eval(d['awayThrees'])), axis = 1)
        return df

    

    def getRollingAverageDF(self, df, n, home = True):
        avgDF = pd.DataFrame(index = df.index, columns = df.columns)
        for gameId in df.index:
            avgDF.loc[gameId] = self.getRollingAverage(df, gameId, n, home)
        return avgDF

    def getRollingAverage(self, df, gameId, n, home = True):
        if home == True:
            games = getRecentNGames(gameId, n, getTeams(np.arange(2015, 2023)).loc[gameId]['teamHome'])
        else:
            games = getRecentNGames(gameId, n, getTeams(np.arange(2015, 2023)).loc[gameId]['teamAway'])
        df = df[df.index.isin(games)]

        return df.mean()

    # years = np.arange(2015, 2023)
    # for year in years:
    #     df = pd.read_csv('../data/shotData/shot_data_{}.csv'.format(year), index_col = 0)
    #     df = df[['home_avg_shot_quality', 'away_avg_shot_quality']]
    #     getRollingAverageDF(df, 5, True).to_csv('../data/shotData/avg_5_shot_quality_home_{}.csv'.format(year))
    #     getRollingAverageDF(df, 5, False).to_csv('../data/shotData/avg_5_shot_quality_away_{}.csv'.format(year))


