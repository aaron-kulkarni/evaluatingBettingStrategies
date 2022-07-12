import numpy as np
import pandas as pd
from datetime import date
import datetime as dt
import matplotlib.pyplot as plt
import bs4 as bs
from urllib.request import urlopen
import requests
from dateutil.relativedelta import relativedelta
import pdb
from sportsipy.nba.teams import Teams
from sportsreference.nba.roster import Roster
from sportsreference.nba.roster import Player

from sportsreference.nba.schedule import Schedule
from sportsipy.nba.boxscore import Boxscore
from sportsipy.nba.boxscore import Boxscores
import re

listGames = 0
listGames = Boxscores(date(2021, 10, 19), date(2021, 10, 19)).games

df = pd.DataFrame()

teams, gameId, q1Score, q2Score, q3Score, q4Score, points, location, daysSinceLastGame, gamesInPastWeek, timeOfDay, roster, coach, record, winsAgainstTeam, streak = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

# wins against team is computed with past 5 years data

i = 0
gameDay = '10-19-2021'
game = listGames[gameDay][i]

gameId = [game['boxscore'], game['boxscore']]
teams = [game['home_abbr'], game['away_abbr']]
gameData = Boxscore(gameId[0])
q1Score = [gameData.summary['home'][0],gameData.summary['away'][0]]
q2Score = [gameData.summary['home'][1],gameData.summary['away'][1]]
q3Score = [gameData.summary['home'][2],gameData.summary['away'][2]]
q4Score = [gameData.summary['home'][3],gameData.summary['away'][3]]
points = [game['home_score'], game['away_score']]


teamHomeSchedule = Schedule(teams[0], year = 2022).dataframe
teamAwaySchedule = Schedule(teams[1], year = 2022).dataframe
timeOfDay = [teamHomeSchedule.loc[gameId[0]][13],teamAwaySchedule.loc[gameId[0]][13]]
streak = [teamHomeSchedule.loc[gameId[0]][12],teamAwaySchedule.loc[gameId[0]][12]]
# caution: streak might be included with the current loss/win

df['teams'] = teams
df['gameId'] = gameId

teamHomeSchedule.sort_values(by='datetime')
teamAwaySchedule.sort_values(by='datetime')

daysSinceLastGame = []
gamesInPastWeek = []

prevdate = teamHomeSchedule['datetime'].shift().loc[gameId[0]]
currentdate = teamHomeSchedule.loc[gameId[0]]['datetime']
daysSinceLastGame.append((currentdate-prevdate).dt.total_seconds()/86400)

temp = teamHomeSchedule[(teamHomeSchedule['datetime'] - currentdate).dt.total_seconds() < 86400*7]
temp = temp[temp['datetime'] < currentdate]
gamesInPastWeek.append(temp.shape[0])

prevdate = teamAwaySchedule['datetime'].shift().loc[gameId[0]]
daysSinceLastGame.append((currentdate-prevdate).dt.total_seconds()/86400)

temp = teamAwaySchedule[(teamAwaySchedule['datetime'] - currentdate).dt.total_seconds() < 86400*7]
temp = temp[temp['datetime'] < currentdate]
gamesInPastWeek.append(temp.shape[0])

df['teams'] = teams 
df['gameId'] = gameId
df['q1Score'] = q1Score
df['q2Score'] = q2Score
df['q3Score'] = q3Score
df['q4Score'] = q4Score
df['points'] = points
df['location'] = location
df['daysSinceLastGame'] = daysSinceLastGame
df['gamesInPastWeek'] = gamesInPastWeek
df['timeOfDay'] = timeOfDay
df['roster'] = roster
df['coach'] = coach
df['record'] = record
df['winsAgainstTeam'] = winsAgainstTeam

df.set_index(['teams'])

teamSchedule.to_csv('teamSchedule.csv', index = False)

def getGameData('gameid'):
    return dataframe
    
