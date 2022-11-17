import numpy as np
import pandas as pd
import requests
from lxml import html
from datetime import timedelta
import os
import datetime as dt
import sys
import bs4 as bs

sys.path.insert(0, "..")

from utils.utils import *
from dataCollection.collectGameAttendanceReferees import *
from dataCollection.collectStaticData import *


"""
The output from this file can be checked using the regex found in
the file validationRegex.txt
"""

teamRivalryDict = {
    "TOR": ["Eastern", "Atlantic"],
    "BOS": ["Eastern", "Atlantic"],
    "BRK": ["Eastern", "Atlantic"],
    "PHI": ["Eastern", "Atlantic"],
    "NYK": ["Eastern", "Atlantic"],
    "CLE": ["Eastern", "Central"],
    "CHI": ["Eastern", "Central"],
    "MIL": ["Eastern", "Central"],
    "IND": ["Eastern", "Central"],
    "DET": ["Eastern", "Central"],
    "ATL": ["Eastern", "Southeast"],
    "WAS": ["Eastern", "Southeast"],
    "MIA": ["Eastern", "Southeast"],
    "CHO": ["Eastern", "Southeast"],
    "ORL": ["Eastern", "Southeast"],
    "POR": ["Western", "Northwest"],
    "OKC": ["Western", "Northwest"],
    "UTA": ["Western", "Northwest"],
    "DEN": ["Western", "Northwest"],
    "MIN": ["Western", "Northwest"],
    "GSW": ["Western", "Pacific"],
    "LAC": ["Western", "Pacific"],
    "PHO": ["Western", "Pacific"],
    "SAC": ["Western", "Pacific"],
    "LAL": ["Western", "Pacific"],
    "HOU": ["Western", "Southwest"],
    "MEM": ["Western", "Southwest"],
    "SAS": ["Western", "Southwest"],
    "DAL": ["Western", "Southwest"],
    "NOP": ["Western", "Southwest"]
}


def getGameData(game_id, neutral):
    """
    This function returns a 1 x n array of data about the game
    represented by the given game id. See helper functions for
    additional details about the output.

    Parameters
    ----------
    game_id : the id of the game on basketball-reference.com
    neutral : indicates whether the game was played on neutral territory or not

    Returns
    -------
    An array of the following data:
    [ game id, game winner, home team, away team, location,
        Q1 home score, Q2 home score, Q3 home score, Q4 home score, overtime home score,
        total home points, home winning/losing streak, number of days since last home team game,
        home team roster, home win/loss record, home wins for this match-up,
        Q1 away score, Q2 away score, Q3 away score, Q4 away score, overtime away score,
        total away points, away winning/losing streak, number of days since last away team game,
        away team roster, away win/loss record, away wins for this match-up,
        rivalry type, home total salary, away total salary, home average salary, away average salary,
        home # of games played, away # of games played, game start time, game end time ]
    """

    # Checks that the game id is a valid game id
    if not gameIdIsValid(game_id):
        raise Exception('Issue with Game ID')

    # Get the boxscores of all games for the given date
    gamesToday = getGamesOnDate(game_id[0:8])
    a = next(item for item in gamesToday if item["boxscore"] == game_id)
    teamHome = a['home_abbr']
    teamAway = a['away_abbr']

    gameData = getBoxscoreData(game_id)

    # Gets the home and away past schedules and sorts the schedules by datetime
    year = getYearFromId(game_id)
    homeTeamSchedule = getTeamScheduleCSV(teamHome, year)
    awayTeamSchedule = getTeamScheduleCSV(teamAway, year)

    otherDict = scrapeGameAttendanceReferees(game_id)
    attendance = otherDict['att']
    referees = otherDict['ref']


    # Gets the number of points scored in each quarter
    ((q1ScoreHome, q2ScoreHome, q3ScoreHome, q4ScoreHome, overtimeScoresHome),
     (q1ScoreAway, q2ScoreAway, q3ScoreAway, q4ScoreAway, overtimeScoresAway)) = getQuarterScore(gameData.summary)

    # Gets the total number of points scored
    pointsHome = gameData.home_points
    pointsAway = gameData.away_points

    # Sets the winner based on the total number of points scored
    winner = teamHome if pointsHome > pointsAway else teamAway

    # Gets the location and rivalry type of the game
    location = gameData.location
    rivalry = getRivalry(teamHome, teamAway)

    # Gets each team's win/loss streak
    streakHome = getTeamStreak(homeTeamSchedule, game_id, teamHome)
    streakAway = getTeamStreak(awayTeamSchedule, game_id, teamAway)

    # Gets the record of each team
    homeRecord = getTeamRecord(homeTeamSchedule, game_id, teamHome)
    awayRecord = getTeamRecord(awayTeamSchedule, game_id, teamAway)

    # Gets all previous matchups between the two teams
    matchupWinsHome, matchupWinsAway = getPastMatchUpWinLoss(homeTeamSchedule, game_id, teamAway)

    # Gets the number of days since the last game for each team
    daysSinceLastGameHome = round(getDaysSinceLastGame(homeTeamSchedule, game_id))
    daysSinceLastGameAway = round(getDaysSinceLastGame(awayTeamSchedule, game_id))

    # Gets team rosters
    homePlayerRoster = [player.player_id for player in gameData.home_players]
    awayPlayerRoster = [player.player_id for player in gameData.away_players]

    # Gets coaches and location
    # TODO coaches are unused right now, might add to dataframe later?
    #homeCoach, awayCoach = getCoaches(teamHome, teamAway, game_id[0:8])

    # Gets player and team salaries
    homeTotalSalary, homeAverageSalary = getTeamSalaryData(teamHome, game_id, homePlayerRoster)
    awayTotalSalary, awayAverageSalary = getTeamSalaryData(teamAway, game_id, awayPlayerRoster)


    # Gets Number of Games Played
    homeGamesPlayed = getNumberGamesPlayed(teamHome, year, game_id)
    awayGamesPlayed = getNumberGamesPlayed(teamAway, year, game_id)

    # Times of game
    timeOfDay = pd.to_datetime(homeTeamSchedule.loc[game_id]['datetime'])
    endTime = timeOfDay + (timedelta(hours=(2 + .5 * len(overtimeScoresHome))))

    # Condenses all the information into an array to return
    gameData = [game_id, winner, teamHome, teamAway, location, rivalry, timeOfDay, endTime, referees, attendance, neutral,
                q1ScoreHome, q2ScoreHome, q3ScoreHome, q4ScoreHome, overtimeScoresHome,
                pointsHome, streakHome, daysSinceLastGameHome, homePlayerRoster, homeRecord, matchupWinsHome,
                homeTotalSalary, homeAverageSalary, homeGamesPlayed,
                q1ScoreAway, q2ScoreAway, q3ScoreAway, q4ScoreAway, overtimeScoresAway,
                pointsAway, streakAway, daysSinceLastGameAway, awayPlayerRoster, awayRecord, matchupWinsAway,
                awayTotalSalary, awayAverageSalary, awayGamesPlayed]

    return gameData


def getQuarterScore(game_summary):
    """
    Gets the score at each quarter for a given game.

    Parameters
    ----------
    game_summary : the sportsipy game summary dictionary
                   from the given game

    Returns
    -------
    A nested tuple containing the scores each quarter as:
        ( (q1 home score, ..., qn home score), (q1 away score, ..., qn away score) )
        where n is the number of quarters (4) + the number of overtime periods
    """

    q1ScoreHome = game_summary['home'][0]
    q1ScoreAway = game_summary['away'][0]
    q2ScoreHome = game_summary['home'][1]
    q2ScoreAway = game_summary['away'][1]
    q3ScoreHome = game_summary['home'][2]
    q3ScoreAway = game_summary['away'][2]
    q4ScoreHome = game_summary['home'][3]
    q4ScoreAway = game_summary['away'][3]
    overtimeScoresHome = []
    overtimeScoresAway = []

    overtimePeriods = len(game_summary['home']) - 4
    for x in range(4, 4 + overtimePeriods):
        overtimeScoresHome.append(game_summary['home'][x])
        overtimeScoresAway.append(game_summary['away'][x])

    return ((q1ScoreHome, q2ScoreHome, q3ScoreHome, q4ScoreHome, overtimeScoresHome),
            (q1ScoreAway, q2ScoreAway, q3ScoreAway, q4ScoreAway, overtimeScoresAway))


def getDaysSinceLastGame(schedule, game_id):
    """
    Gets the number of days it has been since a given
    team last played a game.

    Parameters
    ----------
    schedule : the schedule of the specific team
    game_id : the basketball-reference.com id of the game

    Returns
    -------
    The number of days it has been
    """

    prevHomeDate = pd.to_datetime(schedule['datetime'].shift().loc[game_id])
    currentDate = pd.to_datetime(schedule.loc[game_id]['datetime'])

    return (currentDate - prevHomeDate).total_seconds() / 86400


def getTeamRecord(schedule, game_id, team):
    """
    Gets a given team's win/loss record until a
    given game

    Parameters
    ----------
    schedule : the schedule of the specific team
    game_id : the basketball-reference.com id of the game
    team: the teams abbreviation

    Returns
    -------
    The win loss record as a tuple: (wins, losses)
    """
    prevIndex = schedule.index.get_loc(game_id) - 1
    if prevIndex == -1:
        # on the first game of the season, team has 0 wins and 0 losses
        return [0, 0]
    prevRecord = schedule.iloc[prevIndex]['record']
    prevRecord = (re.sub('\ |\[|\]', '', prevRecord)).split(',')
    wins = int(prevRecord[0])
    losses = int(prevRecord[1])
    prevWon = schedule.iloc[prevIndex]['winner']
    if prevWon == team:
        return [wins + 1, losses]
    else:
        return [wins, losses + 1]


def getTeamStreak(schedule, game_id, team):
    """
    Gets the winning or losing streak that the team is
    on for a game. The streak is > 0 for a win streak of
    that many games, and is < 0 for a losing streak of that
    many games.

    Example
    -------
    If a team wins 5 games in a row before the current game,
        getTeamStreak(...) returns 5
    If a team loses 8 games in a row before the current game,
        getTeamStreak(...) returns -8

    Parameters
    ----------
    schedule : the schedule of the specific team
    game_id : the basketball-reference.com id of the game
    team: the teams abbreviation

    Returns
    -------
    The win/loss streak before the current game
    """
    prevIndex = schedule.index.get_loc(game_id) - 1
    if prevIndex == -1:
        #first game of the season has a streak of 0
        return 0
    prevStreak = schedule.iloc[prevIndex]['streak']
    prevWon = schedule.iloc[prevIndex]['winner']
    if prevWon == team:
        if prevStreak < 0:
            return 1
        else:
            return prevStreak+1
    else:
        if prevStreak > 0:
            return -1
        else:
            return prevStreak-1



def getPastMatchUpWinLoss(schedule, game_id, away_team):
    """
    Gets how many times the home team has won against the
    away team previously and how many times the away team
    has won against the home team previously.

    Parameters
    ----------
    schedule : the schedule of the home team
    game_id : the basketball-reference.com id of the game
    away_team : the 3 letter basketball-reference.com
                abbreviation of the away team

    Returns
    -------
    The past match-up results between the two teams
    as a tuple: (home wins, away wins)
    """
    current_date = schedule.loc[game_id]['datetime']
    df1 = schedule.loc[schedule['teamAway'] == away_team]
    df2 = schedule.loc[schedule['teamHome'] == away_team]
    tempDf = pd.concat([df1, df2], axis=0)

    tempDf = tempDf.loc[schedule['datetime'] < current_date]
    tempDf = tempDf.sort_index()

    wins = tempDf.loc[schedule['winner'] != away_team].shape[0]
    losses = tempDf.loc[schedule['winner'] == away_team].shape[0]

    return [wins, losses]


def getTeamSalaryData(team_abbr, game_id, playerRoster):
    """
    Gets average salary for team as well as total salary for a specific game
    given the game roster.

    Parameters
    ----------
    team_abbr : the 3 letter basketball-reference abbreviation of the team
    game_id : the basketball-reference.com id of the game
    playerRoster : the list of players playing in the game for the team

    Returns
    -------
    The salary data as a tuple (totalSalary, averageSalary)
    """
    curSalary = 0
    totalSalary = 0
    avgSalary = 0
    i = 0
    year = getYearFromId(game_id)

    url = 'https://www.basketball-reference.com/teams/{}/{}.html'.format(team_abbr, year)

    try:
        soup = bs.BeautifulSoup(urlopen(url), features='lxml')
        salaryTable = re.split(r"<tr ><th scope=\"row\" class=\"center \" data-stat=\"ranker\" ",str(soup.find('div', {'id': 'all_salaries2'})))
        regex = r"<a href='/players/./([a-z0-9]+).*\$([0-9,]*).*"
        for row in salaryTable:
            matches = re.findall(regex, row)
            try:
                if len(matches) != 1:
                    raise Exception()
                else:
                    matches = matches[0]

                if len(matches) != 2:
                    raise Exception()
                else:
                    if matches[0] in playerRoster:
                        curSalary = int(matches[1].replace(',', ''))
                        if curSalary != 0:
                            i += 1
                        totalSalary += curSalary

            except Exception:
                z = 1 #dont actually want to do anything

        if i != 0:
            avgSalary = totalSalary / i
        return totalSalary, avgSalary

    except Exception:
        print("Some salary bullshit")

def getRivalry(team_home, team_away):
    """
    Gets whether the two teams have a conference, division,
    or no rivalry between them

    Parameters
    ----------
    team_home : the basketball-reference.com 3 letter abbreviation
                for the home team
    team_away : the basketball-reference.com 3 letter abbreviation
                for the away team

    Returns
    -------
    The rivalry as a string, either 'division', 'conference', or 'none'
    """
    if teamRivalryDict[team_home] == teamRivalryDict[team_away]:
        rivalry = 'division'
    elif teamRivalryDict[team_home][0] == teamRivalryDict[team_away][0]:
        rivalry = 'conference'
    else:
        rivalry = 'none'
    return rivalry


def getCoaches(team_home, team_away, game_date):
    """
    Gets the name of the home and away coaches for a given match from
    basketball-reference.com/teams/.

    Parameters
    ----------
    team_home : the basketball-reference.com 3 letter abbreviation
                for the home team
    team_away : the basketball-reference.com 3 letter abbreviation
                for the away team
    game_date : the date of the game in YYYYMMDD format

    Returns
    -------
    A tuple containing the coaches as: (home coach, away coach)
    """
    urlHome = f"https://www.basketball-reference.com/teams/{team_home}/{game_date[:4].lower()}.html"
    try:
        page = requests.get(urlHome)
        doc = html.fromstring(page.content)
        homeCoach = doc.xpath('//*[@id="meta"]/div[2]/p[2]/a/text()')
    except:
        raise Exception('Coach not found on basketball-reference.com for ' + Teams()(team_home).name)

    urlAway = f"https://www.basketball-reference.com/teams/{team_away}/{game_date[:4].lower()}.html"

    try:
        page = requests.get(urlAway)
        doc2 = html.fromstring(page.content)
        awayCoach = doc2.xpath('//*[@id="meta"]/div[2]/p[2]/a/text()')
    except:
        raise Exception('Coach not found on basketball-reference.com for ' + Teams()(team_away).name)

    return homeCoach, awayCoach


def getGameDataframe(start_time, end_time):
    """
    Creates a dataframe with all the information for every game between
    a given start date and a given end date.

    Parameters
    ----------
    start_time : the start date in YYYYMMDD format
    end_time : the end date in YYYYMMDD format

    Returns
    -------
    A dataframe containing the relevant information for each game, with
    the game_id column as the index.
    """

    # allGames = getGamesBetween(start_time, end_time)
    # gameIdList = []
    # for key in allGames.keys():
    #     for i in range(len(allGames[key])):
    #         gameIdList.append(allGames[key][i]['boxscore'])

    gameIdList = []
    gameDataList = []
    for game_id in gameIdList:
        gameDataList.append(getGameData(game_id))

    cols = ['game_id', 'winner', 'teamHome', 'teamAway', 'location', 'rivalry', 'datetime', 'endtime', 'neutral',
            'q1Score', 'q2Score', 'q3Score', 'q4Score', 'overtimeScores', 'points', 'streak', 'daysSinceLastGame',
            'playerRoster', 'record', 'matchupWins', 'salary', 'avgSalary', 'numberOfGamesPlayed',
            'q1Score', 'q2Score', 'q3Score', 'q4Score', 'overtimeScores', 'points', 'streak', 'daysSinceLastGame',
            'playerRoster', 'record', 'matchupWins', 'salary', 'avgSalary', 'numberOfGamesPlayed']

    df = pd.DataFrame(gameDataList, columns=cols)
    df.set_index('game_id', inplace=True)
    fileName = '../data/tempBullshit.csv'
    df.to_csv(fileName)
    makeMultiIndexing(fileName)
    return df


def getNumberGamesPlayedDF(year):
    """
    Gets dataframes for a given year with information
    about each game.

    Parameters
    ----------
    year : the year to get the games for

    Returns
    -------
    The pandas dataframe
    """
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1])
    df['gameState', 'index'] = df.index.to_series()
    df['home', 'numberOfGamesPlayed'] = df.apply(
        lambda d: getNumberGamesPlayed(d['gameState', 'teamHome'], year, d['gameState', 'index']), axis=1)
    df['away', 'numberOfGamesPlayed'] = df.apply(
        lambda d: getNumberGamesPlayed(d['gameState', 'teamAway'], year, d['gameState', 'index']), axis=1)
    df.drop('index', level=1, inplace=True, axis=1)

    return df


def convDateTimeDF(year):
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1])
    df['gameState', 'index'] = df.index
    df['gameState', 'datetime'] = df.apply(lambda d: convDateTime(d['gameState', 'index'], d['gameState', 'timeOfDay']),
                                           axis=1)
    df.drop(['index', 'timeOfDay'], level=1, inplace=True, axis=1)
    return df


def concatDF(years):
    df = pd.DataFrame()
    for year in years:
        dfCurrent = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), index_col=0, header=[0, 1])
        df = pd.concat([df, dfCurrent], axis=0)

    return df


def addEndTime(years):
    for year in years:
        df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header=[0, 1], index_col=0)
        df['gameState', 'datetime'] = pd.to_datetime(df['gameState']['datetime'])
        df['gameState', 'endtime'] = df.apply(
            lambda d: d['gameState', 'datetime'] + timedelta(hours=2 if d['home', 'overtimeScores'] == "[]" else
            (2 + .5 * (len((str(d['home', 'overtimeScores'])).split(","))))), axis=1)
        df.to_csv('../data/gameStats/game_state_data_{}.csv'.format(year))
    return df


def makeMultiIndexing(file):
    line = ',gameState,gameState,gameState,gameState,home,home,home,home,home,home,home,home,home,home,home,away,away,away,away,away,away,away,away,away,away,away,gameState,home,away,home,away,home,away,gameState,gameState'

    tempfile = file + '.abc'
    with open(file, 'r') as read_objc, open(tempfile, 'w') as write_objc:
        write_objc.write(line + '\n')

        for line in read_objc:
            write_objc.write(line)

    os.remove(file)
    os.rename(tempfile, file)

    return

def get_win_percentage(gameId):
    teamHome, teamAway = getTeamsCSV(gameId)
    year = getYearFromId(gameId)
    dfHome = getTeamScheduleCSVSplit(teamHome, year)[0]['gameState'][:gameId].drop([gameId], axis=0)
    dfAway = getTeamScheduleCSVSplit(teamAway, year)[1]['gameState'][:gameId].drop([gameId], axis=0)
    if dfHome.isnull().values.any() == True:
        return None, None 
    if dfAway.isnull().values.any() == True:
        return None, None
    try: 
        dfHome['signal'] = dfHome.apply(lambda d: 1 if d['winner'] == teamHome else 0, axis=1)
        home_wp = dfHome['signal'].sum()/len(dfHome['signal'])
    except:
        home_wp = None

    try:
        dfAway['signal'] = dfAway.apply(lambda d: 1 if d['winner'] == teamAway else 0, axis=1)
        away_wp = dfAway['signal'].sum()/len(dfAway['signal'])
    except:
        away_wp = None 
    return home_wp, away_wp


#addEndTime(np.arange(2015, 2023))
# years = np.arange(2015, 2023)
# concatDF(years).to_csv('../data/gameStats/game_state_data_ALL.csv')
# for year in years:
#     convDateTimeDF(year).to_csv('../data/gameStats/game_state_data_{}.csv'.format(year))

# addEndTime(years)

# print(getGameData("202102100LAL"))
#getGameDataframe('20201225', '20201225')
# TODO might be deleted
# def getGameStatYear(year):
#     """ 
#     Gets the game stats for a given year by extracting
#     the start and end dates from the game_data_player_stats
#     CSV files.
#
#     Parameters
#     ----------
#     year : the year to get the games for
#
#     Returns
#     -------
#     The pandas dataframe
#     """
#     fileLocation = '../data/gameStats/game_data_player_stats_{}_clean.csv'.format(year)
#
#     startDate = str(extract_lines(fileLocation)[0])[0:10]
#     endDate = str(extract_lines(fileLocation)[1])[0:10]
#
#     df = getGameDataframe(startDate, endDate)
#     return df

# years = np.arange(2015, 2023)
# for y in years:
#    getNumberGamesPlayedDF(y).to_csv('../data/gameStats/game_state_data_{}.csv'.format(y))
