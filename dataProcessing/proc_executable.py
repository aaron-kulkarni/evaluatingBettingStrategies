import sys
sys.path.insert(0, "..")
from utils.utils import *
from TeamPerformance import * 
from EloCalculator import *
from cleanBettingOdds import * 
from PerformanceMetric import * 


# can only be run after scraping odds
addAdjProb('../data/bettingOddsData/closing_betting_odds_2023_clean.csv').to_csv('../data/bettingOddsData/adj_prob_2023.csv')
concatBettingOdds(np.arange(2015, 2024)).to_csv('../data/bettingOddsData/closing_betting_odds_all.csv')
fillBettingOdds(np.arange(2015,2024)).to_csv('../data/bettingOddsData/adj_prob_win_ALL.csv')
#addReturns('../data/bettingOddsData/closing_betting_odds_all.csv')
convAmericanOdds('../data/bettingOddsData/closing_betting_odds_all.csv').to_csv('../data/bettingOddsData/closing_betting_odds_returns.csv')
# run after team performances
#TeamPerformance(2023).getTeamPerformanceDF(5, True)
#TeamPerformance(2023).getTeamPerformanceDF(5, False)
#concat(5, np.arange(2015,2024))

# can be run anytime (preferably run a bit before game starts)
convertEloCSVs.concatCSV()

# can only be run after odds
updatePerMetric(2023)
concatAll(np.arange(2015,2024))

