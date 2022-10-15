import numpy as np
import pandas as pd
import re
import math
import sys
sys.path.insert(0, "..")

from utils.utils import *

class EloCalculator:

    def __init__(self):
        self._ = None

    @staticmethod
    def win_probs(home_elo, away_elo, home_court_advantage):
        h = math.pow(10, home_elo / 400)
        r = math.pow(10, away_elo / 400)

        a = math.pow(10, home_court_advantage / 400)
        denom = r + a * h
        home_prob = a * h / denom
        away_prob = r / denom

        return home_prob, away_prob

    @staticmethod
    def home_odds_on(home_elo, away_elo, home_court_advantage):
        h = math.pow(10, home_elo / 400)
        r = math.pow(10, away_elo / 400)
        a = math.pow(10, home_court_advantage / 400)
        return a * h / r

    @staticmethod
    def elo_k(mov, elo_diff):
        k = 20
        if mov > 0:
            multiplier = (mov + 3) ** (0.8) / (7.5 + 0.006 * (elo_diff))
        else:
            multiplier = (-mov + 3) ** (0.8) / (7.5 + 0.006 * (-elo_diff))

        return k * multiplier

    @staticmethod
    def update_elo(home_score, away_score, home_elo, away_elo, home_court_advantage):
        home_prob, away_prob = EloCalculator.win_probs(home_elo, away_elo, home_court_advantage)

        if home_score - away_score > 0:
            home_win = 1
            away_win = 0
        else:
            home_win = 0
            away_win = 1

        k = EloCalculator.elo_k(home_score - away_score, home_elo - away_elo)

        updated_home_elo = home_elo + k * (home_win - home_prob)
        updated_away_elo = away_elo + k * (away_win - away_prob)

        return updated_home_elo, updated_away_elo

    def ret_ct_adv(neutral):
        if neutral == 1:
            return 0
        if neutral == 0:
            return 100

    @staticmethod
    def getElo(year):
        eloDict = {}
        if year == 2015:
            eloDict = {
                'ATL': 1490,
                'BOS': 1379,
                'BRK': 1518,
                'CHO': 1511,
                'CHI': 1547,
                'CLE': 1464,
                'DAL': 1592,
                'DEN': 1473,
                'DET': 1389,
                'GSW': 1591,
                'HOU': 1596,
                'IND': 1533,
                'LAC': 1631,
                'LAL': 1422,
                'MEM': 1578,
                'MIA': 1579,
                'MIL': 1318,
                'MIN': 1513,
                'NOP': 1457,
                'NYK': 1534,
                'OKC': 1620,
                'ORL': 1359,
                'PHI': 1316,
                'PHO': 1560,
                'POR': 1568,
                'SAC': 1431,
                'SAS': 1700,
                'TOR': 1551,
                'UTA': 1373,
                'WAS': 1541
            }

        elif year == 2016:
            eloDict = {
                'ATL': 1562,
                'BOS': 1520,
                'BRK': 1470,
                'CHO': 1427,
                'CHI': 1570,
                'CLE': 1645,
                'DAL': 1544,
                'DEN': 1443,
                'DET': 1472,
                'GSW': 1743,
                'HOU': 1617,
                'IND': 1505,
                'LAC': 1647,
                'LAL': 1339,
                'MEM': 1583,
                'MIA': 1468,
                'MIL': 1459,
                'MIN': 1324,
                'NOP': 1521,
                'NYK': 1318,
                'OKC': 1564,
                'ORL': 1360,
                'PHI': 1333,
                'PHO': 1476,
                'POR': 1544,
                'SAC': 1440,
                'SAS': 1667,
                'TOR': 1502,
                'UTA': 1543,
                'WAS': 1536
            }
        elif year == 2017:
            eloDict = {
                'ATL': 1571,
                'BOS': 1540,
                'BRK': 1343,
                'CHO': 1546,
                'CHI': 1467,
                'CLE': 1696,
                'DAL': 1503,
                'DEN': 1446,
                'DET': 1497,
                'GSW': 1693,
                'HOU': 1528,
                'IND': 1533,
                'LAC': 1584,
                'LAL': 1333,
                'MEM': 1455,
                'MIA': 1574,
                'MIL': 1420,
                'MIN': 1434,
                'NOP': 1407,
                'NYK': 1415,
                'OKC': 1685,
                'ORL': 1454,
                'PHI': 1278,
                'PHO': 1393,
                'POR': 1585,
                'SAC': 1445,
                'SAS': 1695,
                'TOR': 1569,
                'UTA': 1530,
                'WAS': 1524
            }
        elif year == 2018:
            eloDict = {
                'ATL': 1486,
                'BOS': 1532,
                'BRK': 1405,
                'CHO': 1473,
                'CHI': 1497,
                'CLE': 1648,
                'DAL': 1441,
                'DEN': 1540,
                'DET': 1457,
                'GSW': 1761,
                'HOU': 1574,
                'IND': 1503,
                'LAC': 1591,
                'LAL': 1401,
                'MEM': 1489,
                'MIA': 1553,
                'MIL': 1508,
                'MIN': 1474,
                'NOP': 1488,
                'NYK': 1407,
                'OKC': 1518,
                'ORL': 1390,
                'PHI': 1380,
                'PHO': 1381,
                'POR': 1531,
                'SAC': 1421,
                'SAS': 1617,
                'TOR': 1532,
                'UTA': 1580,
                'WAS': 1566
            }
        elif year == 2019:
            eloDict = {
                'ATL': 1388,
                'BOS': 1562,
                'BRK': 1432,
                'CHO': 1502,
                'CHI': 1364,
                'CLE': 1559,
                'DAL': 1394,
                'DEN': 1567,
                'DET': 1492,
                'GSW': 1685,
                'HOU': 1654,
                'IND': 1555,
                'LAC': 1506,
                'LAL': 1491,
                'MEM': 1367,
                'MIA': 1499,
                'MIL': 1518,
                'MIN': 1537,
                'NOP': 1565,
                'NYK': 1410,
                'OKC': 1584,
                'ORL': 1378,
                'PHI': 1607,
                'PHO': 1334,
                'POR': 1560,
                'SAC': 1396,
                'SAS': 1540,
                'TOR': 1577,
                'UTA': 1623,
                'WAS': 1500
            }
        elif year == 2020:
            eloDict = {
                'ATL': 1423,
                'BOS': 1578,
                'BRK': 1495,
                'CHO': 1497,
                'CHI': 1350,
                'CLE': 1350,
                'DAL': 1462,
                'DEN': 1586,
                'DET': 1476,
                'GSW': 1635,
                'HOU': 1653,
                'IND': 1510,
                'LAC': 1517,
                'LAL': 1473,
                'MEM': 1459,
                'MIA': 1499,
                'MIL': 1643,
                'MIN': 1465,
                'NOP': 1415,
                'NYK': 1319,
                'OKC': 1552,
                'ORL': 1543,
                'PHI': 1582,
                'PHO': 1338,
                'POR': 1602,
                'SAC': 1468,
                'SAS': 1554,
                'TOR': 1673,
                'UTA': 1596,
                'WAS': 1435
            }
        elif year == 2021:
            eloDict = {
                'ATL': 1385,
                'BOS': 1646,
                'BRK': 1489,
                'CHO': 1424,
                'CHI': 1383,
                'CLE': 1363,
                'DAL': 1528,
                'DEN': 1557,
                'DET': 1382,
                'GSW': 1395,
                'HOU': 1541,
                'IND': 1527,
                'LAC': 1597,
                'LAL': 1646,
                'MEM': 1538,
                'MIA': 1603,
                'MIL': 1605,
                'MIN': 1393,
                'NOP': 1500,
                'NYK': 1384,
                'OKC': 1534,
                'ORL': 1495,
                'PHI': 1535,
                'PHO': 1555,
                'POR': 1511,
                'SAC': 1504,
                'SAS': 1529,
                'TOR': 1648,
                'UTA': 1542,
                'WAS': 1407
            }
        elif year == 2022:
            eloDict = {
                'ATL': 1570,
                'BOS': 1500,
                'BRK': 1605,
                'CHO': 1438,
                'CHI': 1495,
                'CLE': 1353,
                'DAL': 1541,
                'DEN': 1577,
                'DET': 1381,
                'GSW': 1529,
                'HOU': 1346,
                'IND': 1487,
                'LAC': 1624,
                'LAL': 1549,
                'MEM': 1541,
                'MIA': 1513,
                'MIL': 1658,
                'MIN': 1439,
                'NOP': 1482,
                'NYK': 1548,
                'OKC': 1309,
                'ORL': 1330,
                'PHI': 1609,
                'PHO': 1650,
                'POR': 1581,
                'SAC': 1454,
                'SAS': 1478,
                'TOR': 1449,
                'UTA': 1615,
                'WAS': 1495
            }
        elif year == 2023:
            eloDict = {
                'ATL': 1546,
                'BOS': 1709,
                'BRK': 1536,
                'CHO': 1535,
                'CHI': 1428,
                'CLE': 1483,
                'DAL': 1632,
                'DEN': 1532,
                'DET': 1356,
                'GSW': 1712,
                'HOU': 1300,
                'IND': 1364,
                'LAC': 1524,
                'LAL': 1421,
                'MEM': 1638,
                'MIA': 1656,
                'MIL': 1611,
                'MIN': 1555,
                'NOP': 1545,
                'NYK': 1526,
                'OKC': 1320,
                'ORL': 1320,
                'PHI': 1608,
                'PHO': 1616,
                'POR': 1253,
                'SAC': 1383,
                'SAS': 1485,
                'TOR': 1570,
                'UTA': 1567,
                'WAS': 1418
            }
            eloDict = {i: 3/4*eloDict[i]+1504/4 for i in eloDict}
        else:
            raise Exception("Unsupported year {0}".format(year))

        gameIdList = getYearIds(year)
        df = pd.DataFrame(columns=['game_id', 'elo1_pre', 'elo2_pre', 'elo1_post', 'elo_2post', 'neutral'])
        for gameId in list(gameIdList):
            teamHome, teamAway, pointsAway, pointsHome, neutral = EloCalculator.getEloInputs(gameId)
            eloHome = eloDict[teamHome]
            eloAway = eloDict[teamAway]
            eloDict = EloCalculator.getEloDict(eloDict, gameId, neutral)

            newRow = {'game_id': gameId, 'elo1_pre': eloHome, 'elo2_pre': eloAway,
                      'elo1_post': eloDict[teamHome], 'elo_2post': eloDict[teamAway], 'neutral': neutral}
            df = df.append(newRow, ignore_index=True)
        df.set_index('game_id', inplace=True)
        return df


    @staticmethod
    def getEloDict(eloDict, gameId, neutral):
        teamHome, teamAway, pointsAway, pointsHome, neutral = EloCalculator.getEloInputs(gameId)
        eloHome = eloDict[teamHome]
        eloAway = eloDict[teamAway]

        eloHome, eloAway = EloCalculator.update_elo(pointsHome, pointsAway, eloHome, eloAway, EloCalculator.ret_ct_adv(neutral))
        upDict = {teamHome: eloHome, teamAway: eloAway
                  }
        eloDict.update(upDict)
        return eloDict


    @staticmethod
    def getEloInputs(gameId):
        year = getYearFromId(gameId)
        df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header=[0, 1], index_col=0)
        df = df.loc[:, [('home', 'points'), ('away', 'points'), ('gameState', 'teamHome'), ('gameState', 'teamAway'), ('gameState', 'neutral')]]
        pointsHome = df.loc[gameId]['home']['points']
        pointsAway = df.loc[gameId]['away']['points']

        teamHome = df.loc[gameId]['gameState']['teamHome']
        teamAway = df.loc[gameId]['gameState']['teamAway']
        neutral = df.loc[gameId]['gameState']['neutral']

        return teamHome, teamAway, pointsAway, pointsHome, neutral

    @staticmethod
    def getEloProb(year):
        df = EloCalculator.getElo(year)
        df['elo_prob1'] = df.apply(lambda d: EloCalculator.win_probs(d['elo1_pre'], d['elo2_pre'], EloCalculator.ret_ct_adv(d['neutral']))[0], axis=1)
        df['elo_prob2'] = df.apply(lambda d: EloCalculator.win_probs(d['elo1_pre'], d['elo2_pre'], EloCalculator.ret_ct_adv(d['neutral']))[1], axis=1)

        return df

EloCalculator.getEloProb(2023).to_csv('../data/eloData/elo_2023.csv')

