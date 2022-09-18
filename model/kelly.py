import pandas as pd
import sys

sys.path.insert(0, "..")
from utils.utils import *
from dataProcessing.TeamPerformance import *
from dataProcessing.PCA import *

def kellyBet(Y_prob, alpha, prop_gained_home, prop_gained_away, n):
    '''
    Given probability of bet outputs proportion of bet for long term expected growth

    '''
    
    f_bet = alpha * (Y_prob - (1 - Y_prob)/(prop_gained_home))
    
    if f_bet <= 0:
        pass
    else:
        f_bet = avoidOdds(prop_gained_home, f_bet, n)
        return f_bet, True
    
    f_bet = alpha * ((1 - Y_prob) - Y_prob/(prop_gained_away))

    if f_bet <= 0:
        return 0, 0
    else:
        f_bet = avoidOdds(prop_gained_away, f_bet, n)
        return f_bet, False


def avoidOdds(prop_gained, f_bet, n):
    if prop_gained < n:
        f_bet = 0
    return f_bet


def convOdds(odd):
    if pd.isna(odd) == True:
        return odd

    if odd > 0:
        return odd/100 
    elif odd < 0:
        return - 100/odd 

def convAmericanOdds():
    df = pd.read_csv('../data/bettingOddsData/closing_betting_odds_all.csv', index_col = 0, header = [0,1])
    for col in df['OddHome'].columns:
        df['OddHome', '{}_return'.format(col)] = df.apply(lambda d: convOdds(d['OddHome', col]), axis = 1)
    for col in df['OddAway'].columns:
        df['OddAway', '{}_return'.format(col)] = df.apply(lambda d: convOdds(d['OddAway', col]), axis = 1)
        df.drop(col, axis = 1, inplace = True, level = 1)
    return df

def findProportionGained(select_x):
    df = pd.read_csv('../data/bettingOddsData/closing_betting_odds_returns.csv', index_col = 0, header = [0,1])
    
    oddHome = df['OddHome']
    oddAway = df['OddAway']
    oddHome = oddHome[select_x]
    oddAway = oddAway[select_x]
    
    return oddHome.max(axis = 1), oddAway.max(axis = 1)

def returnBet(per_bet, signal, retHome, retAway, home):

    if signal == 1 and home == True:
        return per_bet * retHome
    if signal == 0 and home == True:
        return -per_bet
    if signal == 1 and home == False:
        return -per_bet
    if signal == 0 and home == False:
        return per_bet * retAway

    
