import pandas as pd
import sys

sys.path.insert(0, "..")
from utils.utils import *
from dataProcessing.TeamPerformance import *
from dataProcessing.PCA import *

def kellyBet(Y_prob, alpha, prop_gained_home, prop_gained_away):
    '''
    Given probability of bet outputs proportion of bet for long term expected growth

    '''
    
    f_bet_home = alpha * (Y_prob - (1 - Y_prob)/(prop_gained_home))
    
    f_bet_away = alpha * ((1 - Y_prob) - Y_prob/(prop_gained_away))

    if f_bet_home < 0 and f_bet_away < 0 :
        return 0, 0
    if f_bet_home > 0 and f_bet_away < 0 :
        return f_bet_home, True
    if f_bet_home < 0 and f_bet_away > 0 :
        return f_bet_away, False
    if f_bet_home > 0 and f_bet_away > 0 :
        if max(f_bet_home, f_bet_away) == f_bet_home: 
            return f_bet_home, True
        if max(f_bet_home, f_bet_away) == f_bet_away: 
            return f_bet_away, False


def findProportionGained(select_x):
    df = pd.read_csv('../data/bettingOddsData/closing_betting_odds_returns.csv', index_col = 0, header = [0,1])
    
    oddHome = df['OddHome']
    oddAway = df['OddAway']
    oddHome = oddHome[select_x]
    oddAway = oddAway[select_x]
    
    return oddHome.max(axis = 1), oddAway.max(axis = 1)

def returnBettingFirm(select_x, index):
    df = pd.read_csv('../data/bettingOddsData/closing_betting_odds_returns.csv', index_col = 0, header = [0,1])
    oddHome, oddAway = df['OddHome'], df['OddAway']
    oddHome = oddHome[select_x]
    oddAway = oddAway[select_x]
    oddHome = oddHome[oddHome.index.isin(index)]
    oddAway = oddAway[oddAway.index.isin(index)]
    colHome, colAway = pd.DataFrame(oddHome.idxmax(axis=1)), pd.DataFrame(oddAway.idxmax(axis=1))    
    return colHome, colAway

def returnBet(per_bet, signal, retHome, retAway, home):

    if signal == 1 and home == True:
        return per_bet * retHome
    if signal == 0 and home == True:
        return -per_bet
    if signal == 1 and home == False:
        return -per_bet
    if signal == 0 and home == False:
        return per_bet * retAway
