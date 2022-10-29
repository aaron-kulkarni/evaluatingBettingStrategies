import pandas as pd
import datetime as dt
import bs4 as bs
from urllib.request import urlopen
from sportsipy.nba.boxscore import Boxscores
import re

import sys
sys.path.insert(0, '..')
from utils.utils import *

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
    statsDict = getPlayerGameStats(home_abbr, statsDict, url, True, gameId)
    statsDict = getPlayerGameStats(away_abbr, statsDict, url, False, gameId)

    # Makes dictionary of lists into dataframe
    df = pd.DataFrame()
    for k, v in statsDict.items():
        df[k] = v

    df.rename(columns={'Starters':'Name'}, inplace=True)

    return df

def getPlayerGameStats(teamAbbr, statsDict, url, home, gameId):
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
    if 'gameid' not in statsDict:
        statsDict['gameid'] = []

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
        statsDict['gameid'].append(gameId)
    
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
      
def getTeamGameStat(gameId):
    '''
    Gets the static data about a team by scraping
    it from https://www.basketball-reference.com.
    Parameters
    ----------
    The gameID to look for in basketball-reference.com
    Returns
    -------
    Array indexed by playerID on player performance statistics
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

    # homeTeamArray = getTeamGameStats(home_abbr, url, True, game_id)
    # awayTeamArray = getTeamGameStats(away_abbr, url, False, game_id)
    result = []
    result = getTeamGameStatHelper(home_abbr, result, url)
    result = getTeamGameStatHelper(away_abbr, result, url)

    columns = ['teamAbbr', 'Placeholder', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'TS%', 'eFG%', '3pAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'Drtg', 'poss', 'pace', 'poss_per_poss', 'ass_per_poss']
    col = [['home'] * len(columns) + ['away'] * len(columns), columns * 2]
    col = pd.MultiIndex.from_arrays(col, names = ['','gameId'])
    df = pd.DataFrame(columns = col)
    df.loc[gameId] = result
    df.drop(['Placeholder'], axis = 1, level = 1, inplace = True)
    return df

def getTeamGameStatHelper(teamAbbr, result, url):
    """
    Scrapes the data for an entire team in a given game. 
    Parameters
    ----------
    teamAbbr : A string representation of team abbreviation
    result : The array to return
    url : The boxscore URL to scrape from
    Returns
    -------
    An array, with format: team1, attr1, attr2, ... team2, attr1, attr2,....
    """

    try:
        
        # Gets the html of the page and each row
        soup = bs.BeautifulSoup(urlopen(url), features='lxml')
        rows = [p for p in soup.find('div', {'id': 'div_box-' + teamAbbr + '-game-basic'}).findAll('tr')]
        
        # Gets all of the elements in each row s.t. rowList is a 2-d table
        rowList = []
        for row in rows:
            rowList.append([td for td in row.findAll(['td', 'th'])])
        # Changes rowList from html elements to text
        rowList2 = []
        for row in rowList:
            rowList2.append([td.getText() for td in row])

        rowList = rowList2
                      
    except Exception as e:
        print(e)
        raise Exception('Game {0} not found on basketball-reference.com'.format(url))


    if not result:
        result = []

    result.append(teamAbbr)    

    # Go through each column in the table
    for i in range(len(rowList[1])):

        # Go through each player for a given stat
        for j in range(len(rowList)):

            if rowList[j][0] != 'Team Totals':
                continue
            #!!!!!!!!!!!!!!!!! Maybe a logic error
            else:
                if len(rowList[j]) != len(rowList[1]) and i != 0:
                    result.append(None)
                elif rowList[j][i] != '':
                    result.append(rowList[j][i])
    
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

    for i in range(len(rowList[1])):
        if rowList[1][i] == 'Starters' or rowList[1][i] == 'MP':
            continue
        for j in range(len(rowList)):
            if rowList[j][0] != 'Team Totals':
                continue
            #!!!!!!!!!!!!!!!!! Maybe a logic error
            else:
                if len(rowList[j]) != len(rowList[1]) and i != 0:
                    result.append(None)
                elif rowList[j][i] != '':
                    result.append(rowList[j][i])
    length = len(result)
    #possessions = .5 * (FGA + .475 * FTA - ORB + TOV)
    possessions = .5 * (int(result[length-31]) + .475 * int(result[length-25]) - int(result[length-23]) + int(result[length-17]))

    #pace = possessions/48 minutes. our pace is kinda different from other websites. ask aaron personally for clarification
    pace = (int(result[2])/240) * possessions

    #points per possession
    ppposs = round(int(result[length-15])/possessions, 3)

    #assists per possession
    apposs = round(int(result[length-20])/possessions, 3)

    result.append(possessions)
    result.append(pace)
    result.append(ppposs)
    result.append(apposs)

    return result


def getGameStatsDataFrame(startTime, endTime):
    '''
    startTime and endTime must be in format '%Y, %m, %d'
    
    '''

    allGames = Boxscores(dt.datetime.strptime(startTime, '%Y, %m, %d'), dt.datetime.strptime(endTime, '%Y, %m, %d')).games
    gameIdList = [] 
    for key in allGames.keys():
        for i in range(len(allGames[key])):
             gameIdList.append(allGames[key][i]['boxscore'])
    df = pd.DataFrame()
    for id in gameIdList:
        gameData = getPlayerGameStatDataFrame(id) 
        df = df.append(gameData, ignore_index = True)
    return df


# Check format of the output with the following regex:
# [\d]{8}0[A-Z]{3},([A-Z]{3},([\d]+,){3}((1\.000)|(\.[\d]{3})),([\d]+,){2}((1\.000)|(\.[\d]{3})),([\d]+,){2}((1\.000)|(\.[\d]{3})),([\d]+,){9}(((1\.000)|(\.[\d]{3})),){4}([\d\.]*,){13}([\d\.]*,?)){2}\n
def getTeamGameStatDataFrame(gameIdList):
    df = pd.DataFrame()
    for gameId in gameIdList:
        df = pd.concat([df, getTeamGameStat(gameId)], axis=0)
    return df

#getTeamGameStatDataFrame().to_csv('../data/teamStats/team_total_stats_2023.csv')
     
def update_stats(year):
    df = pd.read_csv('../data/teamStats/team_total_stats_{}.csv'.format(year), header=[0,1], index_col = 0)
    index_fin = df[df.index.isin(getPreviousGames())].index
    index_upd = df[df.index.isin(index_fin)].dropna().index
    print('updating:')
    df_update = pd.DataFrame()
    for gameId in list((set(index_fin) - set(index_upd))):
        print(gameId)
        if gameFinished(gameId) == 1:
            df_update = pd.concat([df_update, getTeamGameStat(gameId)], axis=0)
    df = pd.concat([df_update, df.drop(df_update.index, axis=0)], axis=0)
    df = df.reindex(sortDate(df.index))
    df.to_csv('../data/teamStats/team_total_stats_{}.csv'.format(year))
    return df 
    
def update_team_stats(years):
    df = pd.DataFrame()
    for year in years:
        df_current = update_stats(year)
        df = pd.concat([df, df_current], axis=0)
    df = df.reindex(sortDate(df.index))
    df.to_csv('../data/teamStats/team_total_stats_ALL.csv')
    return 
    
def init_team_stat_dataframe(year):
    columns = ['teamAbbr', 'Placeholder', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'TS%', 'eFG%', '3pAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'Drtg', 'poss', 'pace', 'poss_per_poss', 'ass_per_poss']
    col = [['home'] * len(columns) + ['away'] * len(columns), columns * 2]
    col = pd.MultiIndex.from_arrays(col, names = ['', 'gameId'])
    df = pd.DataFrame(columns = col, index = getYearIds(year))
    df.drop(['Placeholder'], axis = 1, level = 1, inplace = True)
    df = pd.concat([getTeamGameStat(df.index[0]), df.drop(df.index[0], axis=0)], axis=0)
    return df

#init_team_stat_dataframe(2023).to_csv('../data/teamStats/team_total_stats_2023.csv')

update_team_stats(np.arange(2015,2024))
