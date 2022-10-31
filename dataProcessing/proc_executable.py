import sys
sys.path.insert(0, "..")
from utils.utils import *
from TeamPerformance import * 
from EloCalculator import *
from cleanBettingOdds import * 
from PerformanceMetric import * 

TeamPerformance(2023).getTeamPerformanceDF(5, True)
TeamPerformance(2023).getTeamPerformanceDF(5, False)
concat(5, np.arange(2015,2024))

convertEloCSVs.concatCSV()
updatePerMetric(2023)
concatAll(np.arange(2015,2024))


#addAdjProb('../data/bettingOddsData/closing_betting_odds_2023_clean.csv').to_csv('../data/bettingOddsData/adj_prob_2023.csv')
#concatBettingOdds(np.arange(2015, 2024)).to_csv('../data/bettingOddsData/closing_betting_odds_all.csv')
#fillBettingOdds(np.arange(2015,2024)).to_csv('../data/bettingOddsData/adj_prob_win_ALL.csv')
