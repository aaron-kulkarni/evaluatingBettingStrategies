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

def getStaticMonthData(month, year):
    url = 'https://www.basketball-reference.com/leagues/NBA_{}_games-{}.html'.format(year, month)
    soup = bs.BeautifulSoup(urlopen(url), features='lxml')
    rows = [p for p in soup.find('div', {'id': 'div_schedule'}).findAll('tr')]
    rowList = []
    for row in rows:
        rowList.append([td for td in row.findAll(['td', 'th'])])
    gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList, neutralList = [], [], [], [], [], []
    for i in range(1, len(rowList)):
        dateTime = parser.parse(rowList[i][0].getText())
        aChildrenH = str(rowList[i][4].findChildren('a'))
        homeTeam = aChildrenH.partition('teams/')[2][:3]
        aChildrenA = str(rowList[i][2].findChildren('a'))
        awayTeam = aChildrenA.partition('teams/')[2][:3]
        
        gameId = '{}0{}'.format(dateTime.strftime("%Y%m%d"), homeTeam)
        dateTime = convDateTime(gameId, rowList[i][1].getText())
        location = rowList[i][9].getText()
        if rowList[i][10].getText() == '': 
            neutral = 0
        else:
            neutral = 1
        gameIdList.append(gameId)
        dateTimeList.append(dateTime)
        homeTeamList.append(homeTeam)
        awayTeamList.append(awayTeam)
        locationList.append(location)
        neutralList.append(neutral)
        
    return gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList, neutralList

def getStaticYearData(year):

    gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList, neutralList = [], [], [], [], [], []
    months = ['october', 'november', 'december', 'january', 'february', 'march', 'april']
    
    for month in months:
        gameIdMonth, dateTimeMonth, homeTeamMonth, awayTeamMonth, locationMonth, neutralMonth = getStaticMonthData(month, year)
        gameIdList.extend(gameIdMonth)
        dateTimeList.extend(dateTimeMonth)
        homeTeamList.extend(homeTeamMonth)
        awayTeamList.extend(awayTeamMonth)
        locationList.extend(locationMonth)
        neutralList.extend(neutralMonth)

    return gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList, neutralList


def initGameStateData(year):

    col = [['gameState', 'gameState','gameState','gameState','gameState','gameState','gameState','gameState','home','home','home','home','home','home','home','home','home','home','home','home','home','home','away','away','away','away','away','away','away','away','away','away','away','away','away','away'],['winner','teamHome','teamAway','location','rivalry','datetime','endtime', 'neutral', 'q1Score','q2Score','q3Score','q4Score','overtimeScores','points','streak','daysSinceLastGame','playerRoster','record','matchupWins','salary','avgSalary','numberOfGamesPlayed','q1Score','q2Score','q3Score','q4Score','overtimeScores','points','streak','daysSinceLastGame','playerRoster','record','matchupWins','salary','avgSalary','numberOfGamesPlayed']]

    col = pd.MultiIndex.from_arrays(col, names = ['','teamData'])
    df = pd.DataFrame(index = getYearIds(year), columns = col)
    df.index.name = 'game_id'
    return df 

#initGameStateData(2023).to_csv('../data/gameStats/game_state_data_2023.csv')

def fillStaticValues(year):
    df = initGameStateData(year)
    gameIdList, dateTimeList, homeTeamList, awayTeamList, locationList, neutralList = getStaticYearData(year)
    df['gameState', 'datetime'] = dateTimeList
    df['gameState', 'teamHome'] = homeTeamList
    df['gameState', 'teamAway'] = awayTeamList
    df['gameState', 'location'] = locationList
    df['gameState', 'neutral'] = neutralList
    df['gameState', 'index'] = df.index

    df['home', 'numberOfGamesPlayed'] = df.apply(lambda d: getNumberGamesPlayed(d['gameState', 'teamHome'], year, d['gameState', 'index']), axis=1)
    df['away', 'numberOfGamesPlayed'] = df.apply(lambda d: getNumberGamesPlayed(d['gameState', 'teamAway'], year, d['gameState', 'index']), axis=1)
    df['home', 'daysSinceLastGame'] = df.apply(lambda d: getDaysSinceLastGame(getTeamScheduleAPI(d['gameState', 'teamHome'], d['gameState', 'index']), d['gameState', 'index']), axis=1)
    df['away', 'daysSinceLastGame'] = df.apply(lambda d: getDaysSinceLastGame(getTeamScheduleAPI(d['gameState', 'teamAway'], d['gameState', 'index']), d['gameState', 'index']), axis=1)
    df['gameState', 'rivalry'] = df.apply(lambda d: getRivalry(d['gameState', 'teamHome'], d['gameState', 'teamAway']), axis=1)
    df.drop('index', inplace=True, level=1, axis=1)

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


def updatePastGameData():
    """
    Checks if current year's CSV is up to date with game data information. If not, update it.
    Parameters
    ----------
    None
    Returns
    -------
    Nothing
    """

    today = dt.datetime.today()
    year = getYearFromDate(today)
    gameIdList = getYearIds(year)
    previousGameList = [gameId for gameId in gameIdList if gameIdToDateTime(gameId) < today]
    mostRecentGame = previousGameList[len(previousGameList) - 1]
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year))
    df2 = df.loc[:mostRecentGame]
    lastGameRecorded = df2.last_valid_index()
    if lastGameRecorded == mostRecentGame:
        # list is already fully updated
        return
    else:
        idx = previousGameList.index(lastGameRecorded)
        previousGameList = previousGameList[idx:] # don't care about previous games that already have data
        for curId in previousGameList: # i honestly didn't know how to do it better than a for loop. should be relatively short though
            df.loc[curId] = getGameData(curId)
        df.to_csv('../data/gameStats/game_state_data_{}.csv'.format(year))
        return

def updateFutureGameData():
    today = dt.datetime.today()
    tomorrow = today + dt.datetime.timedelta(days=1)
    year = getYearFromDate(today)
    gameIdList = getYearIds(year)
    futureGameList = [gameId for gameId in gameIdList if gameIdToDateTime(gameId) >= today]
    nextGame = futureGameList[0]
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year))
    df2 = df.loc[nextGame:]
    tooFarFuture = df2[df2['gameState', 'datetime'] > tomorrow]
    df2.drop(tooFarFuture, inplace=True)
    #df2 should now only contain gameIds that occur within the next 24 hours
    if (df2.empty): #no games happening in next 24 hours
        return
    df2['home', 'streak'] = df2.apply(lambda d: getTeamStreak(getTeamScheduleAPI(d['gameState', 'teamHome'], d['gameState', 'index']), d['gameState', 'index']), axis=1)
    df2['away', 'streak'] = df2.apply(lambda d: getTeamStreak(getTeamScheduleAPI(d['gameState', 'teamAway'], d['gameState', 'index']), d['gameState', 'index']), axis=1)
    df2['home', 'record'] = df2.apply(lambda d: getTeamRecord(getTeamScheduleAPI(d['gameState', 'teamHome'], d['gameState', 'index']), d['gameState', 'index']), axis=1)
    df2['home', 'record'] = df2.apply(lambda d: getTeamRecord(getTeamScheduleAPI(d['gameState', 'teamHome'], d['gameState', 'index']), d['gameState', 'index']), axis=1)
    df2['home', 'matchupWins'] = df2.apply(lambda d: getPastMatchUpWinLoss(getTeamScheduleAPI(d['gameState', 'teamHome'], d['gameState', 'index']), d['gameState', 'index'], d['gameState', 'teamAway']), axis=1)
    df2['home', 'matchupWins'] = df2.apply(lambda d: getPastMatchUpWinLoss(getTeamScheduleAPI(d['gameState', 'teamHome'], d['gameState', 'index']),d['gameState', 'index'], d['gameState', 'teamAway']), axis=1)

    for gameid in df2.index.values.tolist():
        df.loc[gameid] = df2.loc[gameid]
    df.to_csv('../data/gameStats/game_state_data_{}.csv'.format(year))
    return


def updateGameStateDataAll(year):
    df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', header = [0,1], index_col = 0)
    df_current = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0)
    df = pd.concat([df, df_current], axis = 0)
    return df

updateGameStateDataAll(2023).to_csv('../data/gameStats/game_state_data_ALL.csv')

    
