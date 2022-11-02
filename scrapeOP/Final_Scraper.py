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


#scrape_oddsportal_current_season(sport = 'basketball', country = 'usa', league = 'nba', season = '2022-2023', max_page = 2)

#scrape_oddsportal_next_games(sport = 'basketball', country = 'usa', league = 'nba', season = '2022-2023', nmax= 30)

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
