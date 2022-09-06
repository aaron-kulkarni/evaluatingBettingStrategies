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

def rosterDict(year):
    '''
    Initializes dictionary of player rosters for each team every year
    
    '''

    
    
    rosterDict = {}
    rosterDict.keys() = teams




            
            



        
        
