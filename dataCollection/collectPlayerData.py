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


def getPlayerGameStatDataFrame(gameId):
    '''
    Gets the static data about a player by scraping
    it from https://www.basketball-reference.com.

    Parameters
    ----------
    The gameID to look for in basketball-reference.com

    Returns
    -------
    Dataframe indexed by playerID on player performance statistics
    '''

    url = f"https://www.basketball-reference.com/boxscores/{gameId}.html"
    stats = None
    if bool(re.match("^[\d]{9}[A-Z]{3}$", gameId)):
        gameDate = gameId[0:4] + ', ' + gameId[4:6] + ', ' + gameId[6:8]
        gamesToday = list(Boxscores(dt.datetime.strptime(gameDate, '%Y, %m, %d')).games.values())[0]
        temp = next(item for item in gamesToday if item["boxscore"] == gameId)
        away_abbr = temp['away_abbr']
        home_abbr = temp['home_abbr']
    else:
        raise Exception('Issue with Game ID') 

    statsDict = {}
    statsDict = getPlayerGameStats(home_abbr, statsDict, url, True)
    statsDict = getPlayerGameStats(away_abbr, statsDict, url, False)
    
    df = pd.DataFrame()
    for k, v in statsDict.items():
        df[k] = v

    print(df)

    return df


def getPlayerGameStats(teamAbbr, statsDict, url, home):
    """
    Scrapes the data for every player on a team in a given game. 

    Parameters
    ----------
    teamAbbr : A string representation of team abbreviation
    statsDict : The statistics dictionary to append the statistics to
    url : The boxscore URL to scrape from
    home : Boolean, true if team is home, false if away

    Returns
    -------
    The updated statistics dictionary
    """
    
    try:
        
        #gets html of page
        soup = bs.BeautifulSoup(urlopen(url), features='lxml')
        rows = [p for p in soup.find('div', {'id': 'div_box-' + teamAbbr + '-game-basic'}).findAll('tr')]
        
        #getting all of the elements in each row
        rowList = []
        for row in rows:
            rowList.append([td for td in row.findAll(['td', 'th'])])

        #
        playerids = []
        for row in rowList:
            achildren = row[0].findChildren('a')
            if len(achildren) == 1 and achildren[0].has_attr('href'):
                playerids.append(achildren[0]['href'].split("/")[3].split(".")[0])
            else:
                playerids.append(None)

        rowList2 = []
        for row in rowList:
            rowList2.append([td.getText() for td in row])

        rowList = rowList2
              
            
            
    except Exception as e:
        print(e)
        raise Exception('Game {0} not found on basketball-reference.com'.format(gameId))

    if not statsDict:
        statsDict = {}
    
    for i in range(len(rowList[1])):
        if rowList[1][i] not in statsDict:
            statsDict[rowList[1][i]] = []
        for j in range(len(rowList)):
            if rowList[j][0] == 'Reserves' or rowList[j][0] == 'Team Totals' or rowList[j][0] == '' or rowList[j][0] == 'Starters':
                continue
            if len(rowList[j]) != len(rowList[1]) and i != 0:
                statsDict[rowList[1][i]].append(None)
            else:
                statsDict[rowList[1][i]].append(rowList[j][i])


    if 'started' not in statsDict:
        statsDict['started'] = []
    if 'home' not in statsDict:
        statsDict['home'] = []
    if 'playerid' not in statsDict:
        statsDict['playerid'] = []
    # if 'effectivefg' not in statsDict:
    #     statsDict['effectivefg'] = []
    # if 'trueshooting' not in statsDict:
    #     statsDict['trueshooting'] = []
    # if 'ftrate' not in statsDict:
    #     statsDict['ftrate'] = []
    # if '3rate' not in statsDict:
    #     statsDict['3rate'] = []
    # if 'tov%' not in statsDict:
    #     statsDict['tov%'] = []
    # if 'dreb%' not in statsDict:
    #     statsDict['dreb%'] = []
    # if 'oreb%' not in statsDict:
    #     statsDict['oreb%'] = []
    # if 'efficiency' not in statsDict:
    #     statsDict['efficiency'] = []
    # if 'ast/tov' not in statsDict:
    #     statsDict['ast/tov'] = []
    
        
    isStarted = True
    for j in range(2, len(rowList)):            
        if rowList[j][0] == 'Reserves':
            isStarted = False
            continue
        elif rowList[j][0] == 'Starters':
            isStarted = True
            continue
        elif rowList[j][0] == 'Team Totals' or rowList[j][0] == '':
            continue
        statsDict['started'].append(1 if isStarted else 0)
        statsDict['home'].append(1 if home else 0)
        statsDict['playerid'].append(playerids[j])

    
    try:
        
        #gets html of page
        soup = bs.BeautifulSoup(urlopen(url), features='lxml')
        rows = [p for p in soup.find('div', {'id': 'div_box-' + teamAbbr + '-game-advanced'}).findAll('tr')]
        
        #getting all of the elements in each row
        rowList = []
        for row in rows:
            rowList.append([td.getText() for td in row.findAll(['td', 'th'])])

              
            
            
    except Exception as e:
        print(e)
        raise Exception('Game {0} not found on basketball-reference.com'.format(gameId))

    if not statsDict:
        statsDict = {}
    
    for i in range(len(rowList[1])):
        if rowList[1][i] not in statsDict:
            statsDict[rowList[1][i]] = []
        for j in range(len(rowList)):
            if rowList[j][0] == 'Reserves' or rowList[j][0] == 'Team Totals' or rowList[j][0] == '' or rowList[j][0] == 'Starters':
                continue
            if len(rowList[j]) != len(rowList[1]) and i != 0:
                statsDict[rowList[1][i]].append(None)
            else:
                statsDict[rowList[1][i]].append(rowList[j][i])


    return statsDict




print(getPlayerGameStatDataFrame('202012230PHI'))                

# print(getPlayerData('labissk01'))
# print(getPlayersDf(2021))
# getPlayersDf(2021).to_csv('static_player_stats_2021.csv')
