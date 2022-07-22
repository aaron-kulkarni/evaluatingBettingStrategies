import numpy as np
import pandas as pd
from datetime import date
import datetime as dt
import matplotlib.pyplot as plt
import bs4 as bs
from urllib.request import urlopen
import requests
from dateutil.relativedelta import relativedelta
import pdb
from sportsipy.nba.teams import Teams
from sportsreference.nba.roster import Roster
from sportsreference.nba.roster import Player

from sportsreference.nba.schedule import Schedule
from sportsipy.nba.boxscore import Boxscore
from sportsipy.nba.boxscore import Boxscores
import re


def getPlayerStats(playerId):
    '''
    Gets the static data about a player by scraping
    it from https://www.basketball-reference.com.

    Parameters
    ----------
    The playerID to look for in basketball-reference.com

    Returns
    -------
    Dataframe indexed by year on player performance statistics
    '''

    url = f"https://www.basketball-reference.com/players/{playerId[0:1].lower()}/{playerId}.html"
    stats = None
    try:
        soup = bs.BeautifulSoup(urlopen(url), features='lxml')
        rows = [p for p in soup.find('div', {'id': 'switcher_per_game-playoffs_per_game'}).findAll('tr')]
        rowList = []
        for row in rows:
            rowList.append([td for td in row.findAll(['td', 'th'])])
        rowList2 = []
            
        for row in rowList:
            rowList2.append([td.getText() for td in row])

        rowList = rowList2
    except Exception as e:
        print(e)
        raise Exception('Player ID not found on basketball-reference.com'.format(gameId))
    statsDict = {}

    for i in range(len(rowList[1])):
        if rowList[1][i] not in statsDict:
            statsDict[rowList[1][i]] = []
        for j in range(len(rowList)):
            if len(rowList[j]) != len(rowList[1]) and i != 0:
                statsDict[rowList[1][i]].append(None)
            else:
                statsDict[rowList[1][i]].append(rowList[j][i])
    
    df = pd.DataFrame()
    for k, v in statsDict.items():
        df[k] = v

    return df

