import sys
 
# adding Folder_2 to the system path
from functions import *

import time

from selenium import webdriver
import pandas as pd

sys.path.insert(0, "..")
from utils.utils import *
import os
import shutil
import itertools
import bs4 as bs

#scrape_oddsportal_current_season(sport = 'basketball', country = 'usa', league = 'nba', season = '2022-2023', max_page = 2)

#scrape_oddsportal_next_games(sport = 'basketball', country = 'usa', league = 'nba', season = '2022-2023', nmax= 30)


def temp_input_odds(gameIdList):
    year = getYearFromDate(dt.datetime.now())
    df_all = pd.read_csv('../data/bettingOddsData/closing_betting_odds_{}_clean.csv'.format(year), header = [0,1], index_col = 0)
    booker_list = ['10x10bet', '1xBet', 'Alphabet', 'bet-at-home', 'bet365', 'bwin', 'Coolbet', 'Curebet', 'GGBET', 'Lasbet', 'Marathonbet', 'Marsbet', 'Pinnacle', 'Unibet', 'VOBET', 'William Hill']
    df = pd.DataFrame(index = gameIdList, columns = df_all.columns)
    for gameId in gameIdList:
        print('Navigate to game_id {}'.format(gameId))
        for booker in booker_list:
            print('BOOKER NAME: {}'.format(booker))
            oddHome = int(input('Enter odd home: '))
            oddAway = int(input('Enter odd away: '))
            df['OddHome', booker].loc[gameId] = oddHome
            df['OddAway', booker].loc[gameId] = oddAway

    df_ = pd.concat([df_all, df], axis=0)
    df_.to_csv('../data/bettingOddsData/closing_betting_odds_{}_clean.csv'.format(year))
    return

def scrape_current_season_page_next_games(sport, country, league, game_list):
    driver = webdriver.Chrome(executable_path = DRIVER_LOCATION)
    url = 'https://www.oddsportal.com/{}/{}/{}/'.format(sport, country, league)
    driver.get(url)
    html = driver.page_source
    soup = bs.BeautifulSoup(html, 'html.parser')
    links = []
    div_app = soup.find_all('div', class_ = 'flex flex-col border-b border-black-borders min-h-[35px]')
    for div in div_app:
        link = div.find('a')   
        links.append(link['href'])
    time.sleep(2)
    print('We wait 2 seconds:')
    df_all = pd.DataFrame()
    for link in links:
        text, game_id = scrape_page_oddsportal(link, game_list)
        df = convert_dataframe(text, game_id)
        df_all = pd.concat([df, df_all], axis=0)
        driver.get(url)
        time.sleep(2)
        print('We wait 2 seconds:')
    driver.quit()
    return df_all

def scrape_page_oddsportal(link, game_list):

    driver.get(link)
    html = driver.page_source
    soup = bs.BeautifulSoup(html, 'html.parser')

    div_app = soup.find('div', id = 'app')
    text_list = []
 
    teams = div_app.find('li', class_= 'capitalize font-normal text-[0.70rem] leading-4 max-mt:!hidden').text
    home_team = teamDict[teams.split(' - ')[0]]
    
    for div in div_app.find_all('div', class_= 'flex text-xs font-normal text-gray-dark font-main item-center'):
        date_str = div.text.split(',')[1].strip()
        date = dt.datetime.strptime(date_str, '%d %b %Y')

        game_id = '{}0{}'.format(f'{date.year}{date.month:02d}{date.day:02d}', home_team)

    if game_id not in game_list:
            return None, None
    print(game_id)
    for div in div_app.find_all('div', class_ = 'flex text-xs max-sm:h-[60px] h-9 border-b'):
        for p in div.find_all('p'):
            text_list.append(p.text)
    text = '\n'.join(text_list)

    return text, game_id

    
def convert_dataframe(text, game_id):
    if text == None:
        print('Not in List')
        return 
    regex = "(?:([A-Z a-z0-9\-]+)\\n([-+\d]+)\\n([-+\d]+))+\\n"
    matches = re.findall(regex, text)
    bookmakers = [bookmaker[0] for bookmaker in matches]
    columns = [['OddHome'] *len(bookmakers)  + ['OddAway'] *len(bookmakers), bookmakers * 2]
    col = pd.MultiIndex.from_arrays(columns, names=['', 'Bookmaker'])
    df = pd.DataFrame(columns = col, index = [game_id])
    
    for firm in matches:
        df['OddHome', firm[0]].loc[game_id] = firm[0]
        df['OddAway', firm[0]].loc[game_id] = firm[1]
        print('{}: \n OddHome: {} \n OddAway: {}'.format(firm[0], firm[1], firm[2]))
    
    return df

def convert_betting_odds(year):
    scrape_oddsportal_next_games(sport = 'basketball', country = 'usa', league = 'nba', season = '2022-2023', nmax= 30)
    shutil.rmtree('../scrapeOP/basketball')
    for i in os.listdir('../scrapeOP/nba'):
        filename = '../scrapeOP/nba/{}'.format(i)
    teamDict = getTeamDict()

    df = pd.read_csv(filename, sep=';')
    df = df[df['Home_id'].isin(list(teamDict.keys()))]
    df = df[df['Away_id'].isin(list(teamDict.keys()))]

    df['game_id'] = df.apply(lambda d: '{}0{}'.format(pd.to_datetime(d['Date']).strftime('%Y%m%d'), teamDict[d['Home_id']]), axis = 1)
    gameIdList = getGameIdList(year)
    df = df[df['game_id'].isin(gameIdList)]

    df = df.pivot(index = 'game_id', columns = 'Bookmaker', values = ['OddHome', 'OddAway'])

    return df

def update_adj_prob(gameIdList):
    year = getYearFromDate(dt.datetime.now())
    df = convert_betting_odds(year)
    df = df[df.index.isin(gameIdList)]
    df_all = pd.read_csv('../data/bettingOddsData/closing_betting_odds_{}_clean.csv'.format(year), header = [0,1], index_col = 0)
    index_in = list(set(gameIdList).intersection(set(df_all.index)))
    if len(index_in) != 0:
        df_all.drop(index=index_in, inplace=True, axis=0)
    df_all = pd.concat([df_all, df], axis=0)
    df_all.to_csv('../data/bettingOddsData/closing_betting_odds_{}_clean.csv'.format(year))
    shutil.rmtree('../scrapeOP/nba')
    return df_all


def convert_betting_odds_post(year):
    scrape_oddsportal_current_season(sport = 'basketball', country = 'usa', league = 'nba', season = '2022-2023', max_page = 2)
    shutil.rmtree('../scrapeOP/basketball')
    for i in os.listdir('../scrapeOP/nba'):
        filename = '../scrapeOP/nba/{}'.format(i)
    teamDict = getTeamDict()

    df = pd.read_csv(filename, sep=';')
    df = df[df['Home_id'].isin(list(teamDict.keys()))]
    df = df[df['Away_id'].isin(list(teamDict.keys()))]

    df['game_id'] = df.apply(lambda d: '{}0{}'.format(pd.to_datetime(d['Date']).strftime('%Y%m%d'), teamDict[d['Home_id']]), axis = 1)
    gameIdList = getGameIdList(year)
    df = df[df['game_id'].isin(gameIdList)]

    df = df.pivot(index = 'game_id', columns = 'Bookmaker', values = ['OddHome', 'OddAway'])
    return df


def update_adj_prob_post():
    year = getYearFromDate(dt.datetime.now())
    df = convert_betting_odds_post(year)
    df_all = pd.read_csv('../data/bettingOddsData/closing_betting_odds_{}_clean.csv'.format(year), header = [0,1], index_col = 0)
    drop_index = list(set(df.index).intersection(set(df_all.index)))
    df_all.drop(index = drop_index, axis=0, inplace=True)
    df_all = pd.concat([df_all, df], axis=0)
    df_all.to_csv('../data/bettingOddsData/closing_betting_odds_{}_clean.csv'.format(year))
    shutil.rmtree('../scrapeOP/nba')
    shutil.rmtree('../scrapeOP/nba_FULL')
    return df_all
