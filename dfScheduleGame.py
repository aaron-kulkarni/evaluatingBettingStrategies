import numpy as np
import pandas as pd
from datetime import date
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

teams, gameId, q1Score, q2Score, q3Score, q4Score, points, location, daysSinceLastGame, gamesInPastWeek, timeOfDay, roster, coach, record, winsAgainstTeam = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []


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

teamSchedule = Schedule('GSW', year = 2022).dataframe
teamSchedule.to_csv('teamSchedule.csv', index = False)

    
