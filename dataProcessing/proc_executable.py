import sys

sys.path.insert(0, "..")
from utils.utils import *
from TeamPerformance import * 
from EloCalculator import *
from cleanBettingOdds import * 
from PerformanceMetric import *
import os

current_year = getYearHelper(dt.datetime.now().year, dt.datetime.now().month)

# run after team performances
if __name__ == "__main__":
    TeamPerformance(current_year).getTeamPerformanceDF(5, True)
    TeamPerformance(current_year).getTeamPerformanceDF(5, False)
    concat(5, np.arange(2015, current_year + 1))

# can be run anytime (preferably run a bit before game starts)
convertEloCSVs.concatCSV()

# can only be run after scraping odds
addAdjProb('../data/bettingOddsData/closing_betting_odds_{}_clean.csv'.format(current_year)).to_csv('../data/bettingOddsData/adj_prob_{}.csv'.format(current_year))
concatBettingOdds(np.arange(2015, current_year + 1)).to_csv('../data/bettingOddsData/closing_betting_odds_all.csv')
fillBettingOdds(np.arange(2015, current_year + 1)).to_csv('../data/bettingOddsData/adj_prob_win_ALL.csv')
addReturns('../data/bettingOddsData/closing_betting_odds_all.csv')
convAmericanOdds('../data/bettingOddsData/closing_betting_odds_all.csv').to_csv('../data/bettingOddsData/closing_betting_odds_returns.csv')

# can only be run after odds
updatePerMetric(current_year)
concatAll(np.arange(2015, current_year + 1))

# R script (run after raptor and odds)
os.system('Rscript {}'.format(os.path.abspath('MLE_prev_datetime.R')))

test_games = getGamesToday()

for gameId in test_games:
    print('{} - home team raptor-change: {}'.format(gameId, get_raptor_change(gameId, True)))
    print('{} - away team raptor-change: {}'.format(gameId, get_raptor_change(gameId, False)))
