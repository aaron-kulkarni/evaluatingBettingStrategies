import pandas as pd
import numpy as np
import bs4 as bs
from urllib.request import urlopen
import re
from dateutil import parser

import sys
sys.path.insert(0, '..')
from utils.utils import *


def getYearIds(year):
    df = pd.read_csv('../data/gameStats/allGameIds.csv', index_col=False)
    return df[str(year)].to_list()




def getCurrentMonthIds(month, year):
    url = 'https://www.basketball-reference.com/leagues/NBA_{}_games-{}.html'.format(year, month)
    soup = bs.BeautifulSoup(urlopen(url), features='lxml')
    rows = [p for p in soup.find('div', {'id': 'div_schedule'}).findAll('tr')]
    rowList = []
    for row in rows:
        rowList.append([td for td in row.findAll(['td', 'th'])])
    gameIdList = []
    dateTimeList = []
    for i in range(1, len(rowList)):
        dateTime = parser.parse(rowList[i][0].getText())
        aChildren = str(rowList[i][4].findChildren('a'))
        homeTeam = aChildren.partition('teams/')[2][:3]
        
        gameId = '{}0{}'.format(dateTime.strftime("%Y%m%d"), homeTeam)
        dateTime = convDateTime(gameId, rowList[i][1].getText())
        gameIdList.append(gameId)
        dateTimeList.append(dateTime)
    return gameIdList, dateTimeList

getYearIds(2017)

# def getCurrentYearIds(year):
#
#     gameIdList = []
#     dateTimeList = []
#
#     months = ['october', 'november', 'december', 'january', 'february', 'march', 'april']
#     for month in months:
#         gameIdMonth, dateTimeMonth = getCurrentMonthIds(month, year)
#         gameIdList.extend(gameIdMonth)
#         dateTimeList.extend(dateTimeMonth)
#
#     return gameIdList, dateTimeList



