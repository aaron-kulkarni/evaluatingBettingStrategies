import sys

sys.path.insert(0, "..")
from utils.utils import *
from TeamPerformance import * 
from EloCalculator import *
from cleanBettingOdds import * 
from PerformanceMetric import *
import os


# can only be run after scraping odds
addAdjProb('../data/bettingOddsData/closing_betting_odds_2023_clean.csv').to_csv('../data/bettingOddsData/adj_prob_2023.csv')
concatBettingOdds(np.arange(2015, 2024)).to_csv('../data/bettingOddsData/closing_betting_odds_all.csv')
fillBettingOdds(np.arange(2015,2024)).to_csv('../data/bettingOddsData/adj_prob_win_ALL.csv')
addReturns('../data/bettingOddsData/closing_betting_odds_all.csv')
convAmericanOdds('../data/bettingOddsData/closing_betting_odds_all.csv').to_csv('../data/bettingOddsData/closing_betting_odds_returns.csv')

# run after team performances
TeamPerformance(2023).getTeamPerformanceDF(5, True)
TeamPerformance(2023).getTeamPerformanceDF(5, False)
concat(5, np.arange(2015,2024))

# can be run anytime (preferably run a bit before game starts)
convertEloCSVs.concatCSV()

# can only be run after odds
updatePerMetric(2023)
concatAll(np.arange(2015,2024))

# R script (run after raptor and odds)
os.system('Rscript /Users/jasonli/Projects/evaluatingBettingStrategies/dataProcessing/MLE.R')

years=np.arange(2015,2024)
for year in years:
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header=[0,1], index_col=0)
    df['gameState', 'datetime'] = pd.to_datetime(df['gameState', 'datetime'])
    df['gameState', 'endtime'] = pd.to_datetime(df['gameState', 'endtime'])
    df.to_csv('../data/gameStats/game_state_data_{}.csv'.format(year))

# df = pd.read_csv('../data/gameStats/game_state_data_ALL.csv', header=[0,1], index_col=0)
# df['gameState', 'datetime'] = pd.to_datetime(df['gameState', 'datetime'])
# df['gameState', 'endtime'] = pd.to_datetime(df['gameState', 'endtime'])
# df.to_csv('../data/gameStats/game_state_data_ALL.csv')

