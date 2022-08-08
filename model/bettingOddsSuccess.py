import re
import numpy as np
import pandas as pd
import datetime as dt

def checkBettingAccuracy(year):
    oddsFile = '/Users/jasonli/Projects/evaluatingBettingStrategies/data/bettingOddsData/closing_betting_odds_{}_clean.csv'.format(year)
    gameFile = '/Users/jasonli/Projects/evaluatingBettingStrategies/data/gameStats/game_state_data_{}.csv'.format(year)
    oddsDF = pd.read_csv(oddsFile)
    gameDF = pd.read_csv(gameFile)
    
    oddsDF['result'] = df.apply(lambda d: '1' if gameDF['winner']  axis=1)


    oddsDF = oddsDF.pivot(index = 'game_id', columns = 'Bookmaker', values = ['OddHome', 'OddAway'])
    if set(oddsDF.index) != set(gameDF['gameId'].tolist()):
        print('deleting extra rows')
        
    gameDF.set_index('gameId', inplace = True)
    df = pd.concat([oddsDF['OddHome'], gameDF[['winner', 'gameId']]], axis = 1, join = 'outer')
    df['result'] = df.apply(lambda d: '1' if gameDF['winner']
