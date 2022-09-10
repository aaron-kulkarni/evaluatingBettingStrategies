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

    return result

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

def extract_lines(filename):
    startGameId = pd.read_csv(filename).head(1)['gameid'].iloc[0]
    endGameId = pd.read_csv(filename).tail(1)['gameid'].iloc[0]
    
    startDate = dt.datetime.strptime(startGameId[0:4]+', '+startGameId[4:6]+', '+startGameId[6:8], '%Y, %m, %d')
    endDate = dt.datetime.strptime(endGameId[0:4]+', '+endGameId[4:6]+', '+endGameId[6:8], '%Y, %m, %d')
    
    return startDate, endDate

# Check format of the output with the following regex:
# [\d]{8}0[A-Z]{3},([A-Z]{3},([\d]+,){3}((1\.000)|(\.[\d]{3})),([\d]+,){2}((1\.000)|(\.[\d]{3})),([\d]+,){2}((1\.000)|(\.[\d]{3})),([\d]+,){9}(((1\.000)|(\.[\d]{3})),){4}([\d\.]*,){13}([\d\.]*,?)){2}\n
def getTeamGameStatDataFrame(year):
    
    fileLocation = 'data\gameStats\game_data_player_stats_{}.csv'.format(year)

    startDate = str(extract_lines(fileLocation)[0])[0:10]
    endDate = str(extract_lines(fileLocation)[1])[0:10]
    gameIdList = []
    allGames = Boxscores(dt.datetime.strptime(startDate, '%Y-%m-%d'), dt.datetime.strptime(endDate, '%Y-%m-%d')).games
    for key in allGames.keys():
        for i in range(len(allGames[key])):
             gameIdList.append(allGames[key][i]['boxscore'])
    gameDataList = []
    for id in gameIdList:
        gameDataList.append(getTeamGameStat(id))
    df = pd.DataFrame(gameDataList, columns = ['homeTeamAbbr', 'PlaceholderH', 'MP_H', 'FG_H', 'FGA_H', 'FG%_H', '3P_H', '3PA_H', '3P%_H', 'FT_H', 'FTA_H', 'FT%_H', 'ORB_H', 'DRB_H', 'TRB_H', 'AST_H', 'STL_H', 'BLK_H', 'TOV_H', 'PF_H', 'PTS_H', 'TS%_H', 'eFG%_H', '3pAr_H', 'FTr_H', 'ORB%_H', 'DRB%_H', 'TRB%_H', 'AST%_H', 'STL%_H', 'BLK%_H', 'TOV%_H', 'USG%_H', 'ORtg_H', 'Drtg_H', 'poss_H', 'pace_H', 'poss_per_poss_H', 'ass_per_poss_H','awayTeamAbbr', 'PlaceholderA', 'MP_A', 'FG_A', 'FGA_A', 'FG%_A', '3P_A', '3PA_A', 'EP%_A', 'FT_A', 'FTA_A', 'FT%_A', 'ORB_A', 'DRB_A', 'TRB_A', 'AST_A', 'STL_A', 'BLK_A', 'TOV_A', 'PF_A', 'PTS_A', 'TS%_A', 'eFG%_A', '3pAr_A', 'FTr_A', 'ORB%_A', 'DRB%_A', 'TRB%_A', 'AST%_A', 'STL%_A', 'BLK%_A', 'TOV%_A', 'USG%_A', 'ORtg_A', 'Drtg_A', 'poss_A', 'pace_A', 'poss_per_poss_A', 'ass_per_poss_A'])
    df['game_id'] = gameIdList
    df.set_index('game_id', inplace = True)
    df.drop(['PlaceholderH', 'PlaceholderA'], axis = 1, inplace = True)
    return df


     
     
