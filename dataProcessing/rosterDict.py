import numpy as np
import pandas as pd
import sys
from urllib.request import urlopen
import bs4 as bs

sys.path.insert(0, "..")
from utils.utils import *
import re


def convertMultiIndexing(years):
    df_all = pd.DataFrame()
    for year in years:
        df = pd.read_csv('../data/gameStats/game_data_player_stats_{}.csv'.format(year), index_col = 0)
        df['year'] = year
        df.set_index(['gameid', 'playerid'], inplace = True)
        df_all = pd.concat([df_all, df], axis = 0)
    return df_all

years = np.arange(2015, 2023)
convertMultiIndexing(years).to_csv('../data/gameStats/game_data_player_stats_all.csv')

def scrapeRoster(team, year):
    '''
    Scrapes roster information from basketball-reference.com
    Parameters
    ----------
    team = team abbreviation
    year = year of roster
    ----------

    Outputs
    ----------
    player_ids given a specific team and year
    player positions for each respective id 
    
    '''
    url = 'https://www.basketball-reference.com/teams/{}/{}.html'.format(team, year)

    soup = bs.BeautifulSoup(urlopen(url), features='lxml')
    rows = [p for p in soup.find('div', {'id': 'div_roster'}).findAll('tr')]
    rowList = []
    for row in rows:
        rowList.append([td for td in row.findAll(['td', 'th'])])
    playerids = []
    pos = []
    for i in range (0, len(rowList)):
        achildren = rowList[i][1].findChildren('a')
        if rowList[i][2].getText() != 'Pos':
            pos.append(rowList[i][2].getText())
        if len(achildren) == 1 and achildren[0].has_attr('href'):
            playerids.append(achildren[0]['href'].split("/")[3].split(".")[0])
    if len(pos) != len(playerids):
        print('Issue with length of playerids and positions')
    return playerids, pos

def rosterDictPos(year):
    '''
    Initializes dictionary for every team given a year of each playerid and their corresponding posiiton
    
    '''
    rosterDict = {}
    for team in getAllTeams():
        playerids, pos = scrapeRoster(team, year)
        rosterDict[team] = dict(zip(playerids, pos))

    return rosterDict

def getStatsBeforeGame(gameId, team):
    '''
    Outputs DataFrame of average player statistics that were announced in the roster

    1 if home is true, 0 if away is true
    
    '''
    year = getYearFromId(gameId)
    df = pd.read_csv('../data/gameStats/game_data_player_stats_all.csv', index_col = 0)
    gameIdList = getSeasonGames(gameId, team)
    playerIdList = getPlayerRoster(gameId, team)
    df = df[df.index.isin(gameIdList)]
    df = df[df['playerid'].isin(playerIdList)]
    
    


def getPlayerRoster(gameId, team): 
    df = pd.read_csv('../data/gameStats/game_data_player_stats_all.csv', index_col = [0,1])
    df = df.loc[gameId]
    homeTeam, awayTeam = getTeams(gameId)
    if team == homeTeam:
        df = df[df['home'] == 1]
    elif team == awayTeam:
        df = df[df['home'] == 0]
    return list(df.index)
    
    
