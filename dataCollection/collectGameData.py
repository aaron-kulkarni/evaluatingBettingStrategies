import numpy as np
import pandas as pd
import requests
from lxml import html
from datetime import timedelta
import os
import datetime as dt
import sys

sys.path.insert(0, "..")

from utils.utils import *
#from collectStaticData import *
#from collectGameAttendanceReferees import *


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
    teamHomeSchedule = getTeamScheduleAPI(teamHome, game_id[0:8]).sort_values(by='datetime')
    teamAwaySchedule = getTeamScheduleAPI(teamAway, game_id[0:8]).sort_values(by='datetime')

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
    streakHome = getTeamStreak(teamHomeSchedule, game_id)
    streakAway = getTeamStreak(teamAwaySchedule, game_id)

    # Gets the record of each team
    homeRecord = getTeamRecord(teamHomeSchedule, game_id)
    awayRecord = getTeamRecord(teamAwaySchedule, game_id)

    # Gets all previous matchups between the two teams
    matchupWinsHome, matchupWinsAway = getPastMatchUpWinLoss(teamHomeSchedule, game_id, teamAway)

    # Gets the number of days since the last game for each team
    daysSinceLastGameHome = getDaysSinceLastGame(teamHomeSchedule, game_id)
    daysSinceLastGameAway = getDaysSinceLastGame(teamAwaySchedule, game_id)

    # Gets team rosters
    homePlayerRoster = [player.player_id for player in gameData.home_players]
    awayPlayerRoster = [player.player_id for player in gameData.away_players]

    # Gets coaches and location
    # TODO coaches are unused right now, might add to dataframe later?
    homeCoach, awayCoach = getCoaches(teamHome, teamAway, game_id[0:8])

    # Gets player and team salaries
    homeTotalSalary, homeAverageSalary = getTeamSalaryData(teamHome, game_id, homePlayerRoster)
    awayTotalSalary, awayAverageSalary = getTeamSalaryData(teamAway, game_id, awayPlayerRoster)


    # Gets Number of Games Played
    year = getYearFromId(game_id)
    homeGamesPlayed = getNumberGamesPlayed(teamHome, year, game_id)
    awayGamesPlayed = getNumberGamesPlayed(teamAway, year, game_id)

    # Times of game
    timeOfDay = convDateTime(game_id, teamHomeSchedule.loc[game_id][13])
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


def getDaysSinceLastGame(team_schedule, game_id):
    """
    Gets the number of days it has been since a given
    team last played a game.

    Parameters
    ----------
    team_schedule : the schedule of the specific team
    game_id : the basketball-reference.com id of the game

    Returns
    -------
    The number of days it has been
    """
    team_schedule.sort_values(by='datetime')

    prevHomeDate = team_schedule['datetime'].shift().loc[game_id]
    currentdate = team_schedule.loc[game_id]['datetime']

    return (currentdate - prevHomeDate).total_seconds() / 86400


def getTeamRecord(team_schedule, game_id):
    """
    Gets a given team's win/loss record until a
    given game

    Parameters
    ----------
    team_schedule : the schedule of the specific team
    game_id : the basketball-reference.com id of the game

    Returns
    -------
    The win loss record as a tuple: (wins, losses)
    """
    results = team_schedule.result.shift()
    results = results.loc[results.index[0]:game_id]
    homeRecord = results.value_counts(ascending=True)
    try:
        wins = homeRecord['Win']
    except KeyError:
        wins = 0
    try:
        losses = homeRecord['Loss']
    except KeyError:
        losses = 0

    return [wins, losses]


def getTeamStreak(team_schedule, game_id):
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
    team_schedule : the schedule of the specific team
    game_id : the basketball-reference.com id of the game

    Returns
    -------
    The win/loss streak before the current game
    """
    # since streak counts current game, look at streak based on last game
    streak = team_schedule.shift().loc[game_id][12]

    # takes care of first game of season problem
    # also changed format from 'L 5' to -5
    streak = 0 if pd.isna(streak) else int(streak[-1:]) if streak.startswith('W') else -int(
        streak[-1:])

    return streak


def getPastMatchUpWinLoss(home_team_schedule, game_id, away_team):
    """
    Gets how many times the home team has won against the
    away team previously and how many times the away team
    has won against the home team previously.

    Parameters
    ----------
    home_team_schedule : the schedule of the home team
    game_id : the basketball-reference.com id of the game
    away_team : the 3 letter basketball-reference.com
                abbreviation of the away team

    Returns
    -------
    The past match-up results between the two teams
    as a tuple: (home wins, away wins)
    """
    current_date = home_team_schedule.loc[game_id]['datetime']
    tempDf = home_team_schedule.loc[home_team_schedule['opponent_abbr'] == away_team]
    tempDf = tempDf.loc[home_team_schedule['datetime'] < current_date]

    wins = tempDf.loc[home_team_schedule['result'] == 'Win'].shape[0]
    losses = tempDf.loc[home_team_schedule['result'] == 'Loss'].shape[0]

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
    averageSalary = 0
    i = 0

    year = getYearFromId(game_id)
    for playerId in playerRoster:
        try:
            curSalary = scrapePlayerSalaryData(playerId, team_abbr)
        except Exception as e:
            #print(e)
            curSalary = 0
            i -= 1
        totalSalary += curSalary
        i += 1
    if i != 0:
        averageSalary = totalSalary / i

    return totalSalary, averageSalary


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
