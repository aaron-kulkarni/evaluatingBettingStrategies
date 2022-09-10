import re
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt 
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

def computeImpliedProb(x):
    if x > 0:
        return 100/(x+100) * 100
    elif x < 0:
        return (-x)/((-x)+100) * 100
    else:
        return 'NaN'
        
def addImpliedProb(filename):
    '''
    Returns dataframe of implied probability 
    '''
    year = re.findall('[0-9]+', filename)[0]
    oddsDF = pd.read_csv('../data/bettingOddsData/closing_betting_odds_{0}_clean.csv'.format(year), header = [0,1], index_col = 0)
    #gameDF = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year))
    #gameDF.set_index('gameId', inplace = True)
    oddsHome = oddsDF['OddHome'] 
    for col in oddsHome.columns:
        oddsHome['{} (%)'.format(col)] = oddsHome[col].apply(computeImpliedProb)
        oddsHome.drop(col, inplace = True, axis = 1)
        
    oddsAway = oddsDF['OddAway']    
    for col in oddsAway.columns:
        oddsAway['{} (%)'.format(col)] = oddsAway[col].apply(computeImpliedProb)
        oddsAway.drop(col, inplace = True, axis = 1)
    
    df = pd.concat([oddsHome, oddsAway], axis = 1, keys = ['homeProb','awayProb'], join = 'inner')
    
    return df 

def returnFavoredWinner(x, y):
    '''
    1 if home team favored win, 0 if equally favored, -1 if away team win

    '''

    if pd.isna(x) == True:
        return x
    elif pd.isna(y) == True:
        return y
    
    if x > y:
        return 1
    elif x < y:
        return -1
    else:     
        return 0


def returnWinner(x, y):
    '''
    1 if home team wins, -1 if away team wins
    
    '''
    if x == y:
        return 1
    else:
        return -1

def findP(x, y):

    '''
    compares win with predicted win
    
    '''
    if pd.isna(x) == True or pd.isna(y) == True:
        return 'NaN'
    if x == y:
        return 1
    else:
        return 0

def bettingOddSuccess(filename):
    '''
    Returns percentage of betting odd success given the impliedProb files
    Note: if both teams are equaly likely to win, program will count as a failed prediction  
    
    '''
    year = re.findall('[0-9]+', filename)[0]
    probDF = pd.read_csv('../data/bettingOddsData/implied_prob_{0}.csv'.format(year), header = [0,1], index_col = 0)
    gameDF = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year))
    gameDF.set_index('gameId', inplace = True)
    for col in probDF['homeProb'].columns:
        probDF['PredWin', '{}'.format(col.replace(' (%)', ''))] = probDF.apply(lambda d: returnFavoredWinner(d['homeProb'][col], d['awayProb'][col]), axis=1)
    gameDF['win'] = gameDF.apply(lambda d: returnWinner(d['teamHome'], d['winner']), axis = 1)
    probDF['colWin', 'win'] = gameDF['win']
    
    for col in probDF['PredWin'].columns:
        probDF['betWin', '{}'.format(col)] = probDF.apply(lambda d:  findP(d['PredWin'][col], d['colWin']['win']), axis = 1)
        
    df = pd.DataFrame()
    
    for col in probDF['betWin'].columns:
       df['{}'.format(col)] = probDF['betWin'][col].value_counts()
    df = df.T
    df['Prob'] = df.apply(lambda d: d[1]/(d[1] + d[0]), axis = 1)
    return df

def adjProb(x, y):
    '''
    x = home implied probability
    y = away implied probability
    
    '''
    if pd.isna(x) == True:
        return x
    if pd.isna(y) == True:
        return y
    return x/(x+y)

def getAdjProb(filename):
    '''
    takes implied probability files as input

    '''
    year = re.findall('[0-9]+', filename)[0]
    probDF = pd.read_csv('../data/bettingOddsData/implied_prob_{0}.csv'.format(year), header = [0,1], index_col = 0)
    for col in probDF['homeProb'].columns:
        probDF['homeProbAdj', '{}'.format(col)] = probDF.apply(lambda d: adjProb(d['homeProb'][col], d['awayProb'][col]), axis = 1)
    for col in probDF['awayProb'].columns:
        probDF['awayProbAdj', '{}'.format(col)] = probDF.apply(lambda d: adjProb(d['awayProb'][col], d['homeProb'][col]), axis = 1)

    return probDF

def getProbCut(filename):
    '''
    uses adjusted probability files as input 
    sorts probability files by increments of 10 and evaluates success of each increment
    
    '''
    year = re.findall('[0-9]+', filename)[0]
    probDF = pd.read_csv('../data/bettingOddsData/adj_prob_{0}.csv'.format(year), header = [0,1], index_col = 0)
    gameDF = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year))
    gameDF.set_index('gameId', inplace = True)
    gameDF['win'] = gameDF.apply(lambda d: returnWinner(d['teamHome'], d['winner']), axis = 1)
    probDF['colWin', 'win'] = gameDF['win']
    df = pd.DataFrame()
    for col in probDF['homeProbAdj'].columns:
        probDF['probCut', col] = pd.cut(probDF['homeProb'][col],[0,10,20,30,40,50,60,70,80,90,100])
        df[col] = probDF.groupby([('probCut', col), ('colWin','win')]).size() * 100/probDF.groupby([('probCut', col)]).size()

    return df

def evaluateBettingOdds(filename, n):
    '''
    evaluates betting odd performance, uses implied probability files as inputs in increments of n
    '''
    year = re.findall('[0-9]+', filename)[0]
    probDF = pd.read_csv('../data/bettingOddsData/implied_prob_{0}.csv'.format(year), header = [0,1], index_col = 0)
    gameDF = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year))
    gameDF.set_index('gameId', inplace = True)
    for col in probDF['homeProb'].columns:
        probDF['PredWin', '{}'.format(col.replace(' (%)', ''))] = probDF.apply(lambda d: returnFavoredWinner(d['homeProb'][col], d['awayProb'][col]), axis=1)
    gameDF['win'] = gameDF.apply(lambda d: returnWinner(d['teamHome'], d['winner']), axis = 1)
    probDF['colWin', 'win'] = gameDF['win']
    
    for col in probDF['PredWin'].columns:
        probDF['betWin', '{}'.format(col)] = probDF.apply(lambda d:  findP(d['PredWin'][col], d['colWin']['win']), axis = 1)
    probDF['betWin', 'col'] = range(0, len(probDF))
    probDF['betWin', 'col'] =  probDF['betWin', 'col']//n
    listX = []
    listY = []
    listZ = []
    for i in range(0, probDF['betWin'].loc[probDF['betWin']['col'].idxmax()]['col']):     
        temp = probDF['betWin'].groupby(probDF['betWin']['col']).get_group(i)
        temp.drop('col', inplace = True, axis = 1)
        df = pd.DataFrame()
        for col in temp.columns:
            df = pd.concat([df, temp[col].value_counts()], axis = 1)
        try:
            success = df.T[1].sum()
        except:
            success = 0
        try:
            failure = df.T[0].sum()
        except:
            failure = 0
        try:
            non = df.T['NaN'].sum()
        except:
            non = 0
        listX.append(success)
        listY.append(failure)
        listZ.append(non)
    final = pd.DataFrame([listX, listY, listZ]).T
    final.columns = ['Success','Failure','NaN']
    return final

def plotSuccess(filename, n, cumSum = True):
    df = evaluateBettingOdds(filename, n)
    if cumSum == True:
        df['Success'] = df['Success'].cumsum(skipna = False)
        df['Failure'] = df['Failure'].cumsum(skipna = False)
    df['Rate'] = df.apply(lambda d: d['Success']/(d['Success'] + d['Failure']), axis = 1)
    rate = df['Rate'].array
    return rate

rate = plotSuccess('../data/bettingOddsData/implied_prob_2020.csv', 30, False)
x = list(range(0, len(rate)))

plt.plot(x, rate)  
plt.show()


        
#years = np.arange(2015, 2023)
#for year in years:
    #getProbCut('../data/bettingOddsData/adj_prob_{}.csv'.format(year)).to_csv('summary_of_betting_success_by_cut_{}.csv'.format(year))



def fillBettingOdds():
    '''

    INITIALIZES DATAFRAME FOR PCA => CONCATS ALL SEPERATE ADJPROB CSVS AND REPLACES ALL NAN VALUES WITH THE ROW AVERAGE

    '''
    
    df = pd.DataFrame()
    years = np.arange(2015, 2023)
    for year in years:
        adjProb = pd.read_csv('../data/bettingOddsData/adj_prob_{}.csv'.format(year), header = [0,1], index_col = 0)['homeProbAdj']
        df = pd.concat([df, adjProb], axis = 0)
    df = df.apply(lambda row: row.fillna(row.mean()), axis = 1)

    return df

fillBettingOdds().to_csv('adj_prob_home_win_ALL.csv')

from TeamPerformance import getSignal

df = pd.read_csv('../data/bettingOddsData/adj_prob_home_win_ALL.csv', index_col = 0)

df['signal'] = getSignal()

'STANDARIZE DATA FEATURES TO UNIT SCALE (mean = 0 and variance = 1)'

from sklearn.preprocessing import StandardScaler

features = list(df.columns[:-1])
x = df.loc[:, features].values
y = df.loc[:, ['signal']].values

x = StandardScaler().fit_transform(x)

'BEGIN PERFORMING PCA'

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

def performPCA(n):
    pca = PCA(n_components = n)
    principalComponents = pca.fit_transform(x)
    principalDF = pd.DataFrame(data = principalComponents, columns = ['PCA{}'.format(i) for i in range(1, n + 1)])
    print(pca.explained_variance_ratio_)
    return principalDF

dfFinal = performPCA(2).set_index(df.index)
dfFinal.to_csv('PCA_2_betting_odds_all.csv')


def concatBettingOdds(years):
    df_all = pd.DataFrame()
    for year in years:
        df = pd.read_csv('../data/bettingOddsData/closing_betting_odds_{}_clean.csv'.format(year), header = [0,1], index_col = 0)
        df_all = pd.concat([df, df_all], axis = 0)

    return df_all

concatBettingOdds(np.arange(2015, 2023)).to_csv('../data/bettingOddsData/closing_betting_odds_all.csv')
