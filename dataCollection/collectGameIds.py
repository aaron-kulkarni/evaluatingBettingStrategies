import pandas as pd
import numpy as np
import bs4 as bs
from urllib.request import urlopen
import re
from dateutil import parser

import sys
sys.path.insert(0, '..')
from utils.utils import *
from collectGameData import * 


def getYearIds(year):
    df = pd.read_csv('../data/gameStats/all_game_ids.csv', index_col=False)
    return df[str(year)].to_list()


def getStaticMonthData(month, year):
    url = 'https://www.basketball-reference.com/leagues/NBA_{}_games-{}.html'.format(year, month)
    soup = bs.BeautifulSoup(urlopen(url), features='lxml')
    rows = [p for p in soup.find('div', {'id': 'div_schedule'}).findAll('tr')]
    rowList = []
    for row in rows:
        rowList.append([td for td in row.findAll(['td', 'th'])])
    gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList = [], [], [], [], []
    for i in range(1, len(rowList)):
        dateTime = parser.parse(rowList[i][0].getText())
        aChildrenH = str(rowList[i][4].findChildren('a'))
        homeTeam = aChildrenH.partition('teams/')[2][:3]
        aChildrenA = str(rowList[i][2].findChildren('a'))
        awayTeam = aChildrenA.partition('teams/')[2][:3]
        
        gameId = '{}0{}'.format(dateTime.strftime("%Y%m%d"), homeTeam)
        dateTime = convDateTime(gameId, rowList[i][1].getText())
        location = rowList[i][9].getText()
        gameIdList.append(gameId)
        dateTimeList.append(dateTime)
        homeTeamList.append(homeTeam)
        awayTeamList.append(awayTeam)
        locationList.append(location)
        
    return gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList

def getStaticYearData(year):

    gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList = [], [], [], [], []
    months = ['october', 'november', 'december', 'january', 'february', 'march', 'april']
    
    for month in months:
        gameIdMonth, dateTimeMonth, homeTeamMonth, awayTeamMonth, locationMonth = getStaticMonthData(month, year)
        gameIdList.extend(gameIdMonth)
        dateTimeList.extend(dateTimeMonth)
        homeTeamList.extend(homeTeamMonth)
        awayTeamList.extend(awayTeamMonth)
        locationList.extend(locationMonth)

    return gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList


def initGameStateData(year):

    col = [['gameState','gameState','gameState','gameState','gameState','gameState','gameState','home','home','home','home','home','home','home','home','home','home','home','home','home','home','away','away','away','away','away','away','away','away','away','away','away','away','away','away'],['winner','teamHome','teamAway','location','rivalry','datetime','endtime','q1Score','q2Score','q3Score','q4Score','overtimeScores','points','streak','daysSinceLastGame','playerRoster','record','matchupWins','salary','avgSalary','numberOfGamesPlayed','q1Score','q2Score','q3Score','q4Score','overtimeScores','points','streak','daysSinceLastGame','playerRoster','record','matchupWins','salary','avgSalary','numberOfGamesPlayed']]

    col = pd.MultiIndex.from_arrays(col, names = ['','teamData'])
    df = pd.DataFrame(index = getYearIds(year), columns = col)
    df.index.name = 'game_id'
    return df 

#initGameStateData(2023).to_csv('../data/gameStats/game_state_data_2023.csv')

def fillStaticValues(year):
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0)
    gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList = getStaticYearData(year)
    df['gameState', 'datetime'] = dateTimeList
    df['gameState', 'teamHome'] = homeTeamList
    df['gameState', 'teamAway'] = awayTeamList
    df['gameState', 'location'] = locationList

    return df 

#fillStaticValues(2023).to_csv('../data/gameStats/game_state_data_2023.csv')

def fillDependentStaticValues(year):
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0)
    df['gameState', 'index'] = df.index
    
    df['home', 'numberOfGamesPlayed'] = df.apply(lambda d: getNumberGamesPlayed(d['gameState', 'teamHome'], year, d['gameState', 'index']), axis = 1)
    df['away', 'numberOfGamesPlayed'] = df.apply(lambda d: getNumberGamesPlayed(d['gameState', 'teamAway'], year, d['gameState', 'index']), axis = 1)
    df['gameState', 'rivalry'] = df.apply(lambda d: getRivalry(d['gameState', 'teamHome'], d['gameState', 'teamAway']), axis = 1)
    df.drop('index', inplace = True, level = 1, axis = 1)
    return df 

#fillDependentStaticValues(2023).to_csv('../data/gameStats/game_state_data_2023.csv')

def updateGameStateDataAll(year):
    df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', header = [0,1], index_col = 0)
    df_current = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0)
    df = pd.concat([df, df_current], axis = 0)
    return df

updateGameStateDataAll(2023).to_csv('../data/gameStats/game_state_data_ALL.csv')
                                                 
