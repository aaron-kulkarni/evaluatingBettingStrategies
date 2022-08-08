import re
import numpy as np
import pandas as pd
import datetime as dt
from sportsipy.nba.teams import Teams
from sportsipy.nba.boxscore import Boxscores


# filename = '../data/bettingOddsData/closing_betting_odds_2022_FIXED.csv'

def extract_lines(filename):
    startGameId = pd.read_csv(filename).head(1)['gameid'].iloc[0]
    endGameId = pd.read_csv(filename).tail(1)['gameid'].iloc[0]

    startDate = dt.datetime.strptime(startGameId[0:4] + ', ' + startGameId[4:6] + ', ' + startGameId[6:8], '%Y, %m, %d')
    endDate = dt.datetime.strptime(endGameId[0:4] + ', ' + endGameId[4:6] + ', ' + endGameId[6:8], '%Y, %m, %d')

    return startDate, endDate


def convertBettingOdds(filename):
    """
    function does following:
    1. Removes games that are not in regular season
    2. Adds GameID
    3. Checks if all games in betting odds file are in regular season and vice versa
    4. Pivots table
    """

    year = re.findall('[0-9]+', filename)[0]

    teamDict = {}
    for team in Teams():
        teamAbbr = re.search(r'\((.*?)\)', str(team)).group(1)
        teamDict[team.name] = teamAbbr

    df = pd.read_csv(filename, sep=';')
    df = df[df.Home_id.isin(list(teamDict.keys()))]
    df.drop('Season', inplace=True, axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True, inplace=True)

    # IMPORTANT: MUST CHANGE FOR SPECIFIC FILE WHERE CLEAN DATA FOR CORRESPONDING YEAR IS LOCATED 
    fileLocation = '../data/gameStats/game_data_player_stats_{}.csv'.format(year)

    # yeardates = {
      #  2015: ('2014-10-28', '2015-06-15'),
      #  2016: ('2015-10-27', '2016-06-19'),
      #  2017: ('2016-10-25', '2017-06-12'),
      #  2018: ('2017-10-17', '2018-06-08'),
      #  2019: ('2018-10-16', '2019-06-13'),
      #  2020: ('2019-10-22', '2020-10-11'),
      #  2021: ('2020-12-22', '2021-07-20'),
      #  2022: ('2021-10-19', '2022-06-16')
    # }

    startDate = str(extract_lines(fileLocation)[0])[0:10]
    endDate = str(extract_lines(fileLocation)[1])[0:10]

    # startDate = yeardates[int(year)][0]
    # endDate = yeardates[int(year)][1]

    df = df[(df['Date'] >= startDate) & (df['Date'] <= endDate)]
    df['game_id'] = df.apply(lambda d: str(d['Date'])[0:10].replace('-', '') + '0' + teamDict[d['Home_id']], axis=1)

    gameIdList = []
    allGames = Boxscores(dt.datetime.strptime(startDate, '%Y-%m-%d'), dt.datetime.strptime(endDate, '%Y-%m-%d')).games
    for key in allGames.keys():
        for i in range(len(allGames[key])):
            gameIdList.append(allGames[key][i]['boxscore'])

    if set(gameIdList) != set(df['game_id'].unique().tolist()):
        print('Issue with GameID')
        print(set(gameIdList).difference(set(df['game_id'].unique().tolist())))
        print(set(df['game_id'].unique().tolist()).difference(set(gameIdList)))
    else:
        print('No Issue with GameID')

    df.drop('Home_id', inplace=True, axis=1)
    df.drop('Away_id', inplace=True, axis=1)
    df.drop('Score_home', inplace=True, axis=1)
    df.drop('Score_away', inplace=True, axis=1)
    df.drop('Date', inplace=True, axis=1)
    
    df = df.pivot(index = 'game_id', columns = 'Bookmaker', values = ['OddHome', 'OddAway'])
    
    return df


def cleanBettingOdds(filename):
    '''
    1. Verifies that OddHome and OddAway are different signs
    2. Returns dataframe that displays betting odd accruacy of each betting bookmaker
    '''
    year = re.findall('[0-9]+', filename)[0]
    oddsDF = pd.read_csv('../data/bettingOddsData/closing_betting_odds_{0}_clean.csv'.format(year), header = [0,1], skipinitialspace = True)

    return df 


    
years = np.arange(2015, 2023)
for y in years:
    convertBettingOdds('../data/bettingOddsData/closing_betting_odds_{0}_FIXED.csv'.format(y)) \
        .to_csv('../data/bettingOddsData/closing_betting_odds_{0}_clean.csv'.format(y))
    print('Created betting odds CSV for year {0}'.format(y))
