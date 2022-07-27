import os
import pandas as pd
import datetime as dt
import bs4 as bs
from urllib.request import urlopen
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

    # Checks that the game id is a valid match id and gets it
    if bool(re.match("^[\d]{9}[A-Z]{3}$", gameId)):
        gameDate = gameId[0:4] + ', ' + gameId[4:6] + ', ' + gameId[6:8]
        gamesToday = list(Boxscores(dt.datetime.strptime(gameDate, '%Y, %m, %d')).games.values())[0]
        temp = next(item for item in gamesToday if item["boxscore"] == gameId)
        away_abbr = temp['away_abbr']
        home_abbr = temp['home_abbr']
    else:
        raise Exception('Issue with Game ID') 

    # Gets the player stats on the home and away sides
    statsDict = {}
    statsDict = getPlayerGameStats(home_abbr, statsDict, url, True)
    statsDict = getPlayerGameStats(away_abbr, statsDict, url, False)

    # Makes dictionary of lists into dataframe
    df = pd.DataFrame()
    for k, v in statsDict.items():
        df[k] = v

    df.rename(columns={'Starters':'Name'}, inplace=True)
    # print(df)

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
    The updated statistics dictionary, see method for content details
    """
    
    try:
        
        # Gets the html of the page and each row
        soup = bs.BeautifulSoup(urlopen(url), features='lxml')
        rows = [p for p in soup.find('div', {'id': 'div_box-' + teamAbbr + '-game-basic'}).findAll('tr')]
        
        # Gets all of the elements in each row s.t. rowList is a 2-d table
        rowList = []
        for row in rows:
            rowList.append([td for td in row.findAll(['td', 'th'])])

        # Gets the player ids by accessing the links in the player name column
        playerids = []
        for row in rowList:
            achildren = row[0].findChildren('a')
            if len(achildren) == 1 and achildren[0].has_attr('href'):
                playerids.append(achildren[0]['href'].split("/")[3].split(".")[0])
            else:
                playerids.append(None)

        # Changes rowList from html elements to text
        rowList2 = []
        for row in rowList:
            rowList2.append([td.getText() for td in row])

        rowList = rowList2
              
            
            
    except Exception as e:
        print(e)
        raise Exception('Game {0} not found on basketball-reference.com'.format(url))

    # If statsDict does not exist, initialize it
    if not statsDict:
        statsDict = {}

    # Go through each column in the table
    for i in range(len(rowList[1])):

        # If the stat does not exist in statsDict, initialize it as empty list
        if rowList[1][i] not in statsDict:
            statsDict[rowList[1][i]] = []

        # Go through each player for a given stat
        for j in range(len(rowList)):

            # Check for special cases like headers or where player did not play in game, otherwise append stat
            if rowList[j][0] == 'Reserves' or rowList[j][0] == 'Team Totals' or rowList[j][0] == '' or rowList[j][0] == 'Starters':
                continue
            elif len(rowList[j]) != len(rowList[1]) and i != 0:
                statsDict[rowList[1][i]].append(None)
            else:
                statsDict[rowList[1][i]].append(rowList[j][i])

    # Adds the started, home, and playerid columns
    if 'started' not in statsDict:
        statsDict['started'] = []
    if 'home' not in statsDict:
        statsDict['home'] = []
    if 'playerid' not in statsDict:
        statsDict['playerid'] = []

    # Loops through the rows to append started, home, and playerid for each player
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
        
        # Same as above, but with the advanced stats table
        soup = bs.BeautifulSoup(urlopen(url), features='lxml')
        rows = [p for p in soup.find('div', {'id': 'div_box-' + teamAbbr + '-game-advanced'}).findAll('tr')]

        rowList = []
        for row in rows:
            rowList.append([td.getText() for td in row.findAll(['td', 'th'])])

    except Exception as e:
        print(e)
        raise Exception('Game {0} not found on basketball-reference.com'.format(url))

    # Same as above, but with the advanced stats table
    for i in range(len(rowList[1])):
        if rowList[1][i] not in statsDict:
            statsDict[rowList[1][i]] = []
        elif rowList[1][i] == "Starters" or rowList[1][i] == "MP":
            continue;
        for j in range(len(rowList)):
            if rowList[j][0] == 'Reserves' or rowList[j][0] == 'Team Totals' or rowList[j][0] == '' or rowList[j][0] == 'Starters':
                continue
            if len(rowList[j]) != len(rowList[1]) and i != 0:
                statsDict[rowList[1][i]].append(None)
            else:
                statsDict[rowList[1][i]].append(rowList[j][i])


    return statsDict

def getGameDataframe(startTime, endTime):
    
    
