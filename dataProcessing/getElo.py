import re
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import re
import sys
import math

from sportsipy.nba.teams import Teams
from recentTeamPerformance import getFirstGame 

def win_probs(home_elo, away_elo, home_court_advantage) :

    h = math.pow(10, home_elo/400)
    r = math.pow(10, away_elo/400)
    a = math.pow(10, home_court_advantage/400) 

    denom = r + a * h
    home_prob = a * h / denom
    away_prob = r / denom 
  
    return home_prob, away_prob


def home_odds_on(home_elo, away_elo, home_court_advantage):
    
    h = math.pow(10, home_elo/400)
    r = math.pow(10, away_elo/400)
    a = math.pow(10, home_court_advantage/400)
    return a * h / r


def elo_k(MOV, elo_diff):
    k = 20
    if MOV > 0:
        multiplier = (MOV+3)**(0.8)/(7.5+0.006*(elo_diff))
    else:
        multiplier = (-MOV+3)**(0.8)/(7.5+0.006*(-elo_diff))

    return k * multiplier


def update_elo(home_score, away_score, home_elo, away_elo, home_court_advantage):

    home_prob, away_prob = win_probs(home_elo, away_elo, home_court_advantage) 

    if (home_score - away_score > 0) :
        home_win = 1
        away_win = 0 
    else :
        home_win = 0 
        away_win = 1 
  
    k = elo_k(home_score - away_score, home_elo - away_elo)

    updated_home_elo = home_elo + k * (home_win - home_prob) 
    updated_away_elo = away_elo + k * (away_win - away_prob)

    return updated_home_elo, updated_away_elo

def getElo(year):

    if year == 2015:
        eloDict = {
            'ATL' : 
            'BOS' :
            'BKN' :
            'CHA' :
            'CHI' :
            'CLE' :
            'DAL' :
            'DEN' :
            'DET' :
            'GSW' :
            'HOU' :
            'IND' :
            'LAC' :
            'LAL' :
            'MEM' :
            'MIA' :
            'MIL' :
            'MIN' :
            'NOP' :
            'NYK' :
            'OKC' :
            'ORL' :
            'PHI' :
            'PHX' :
            'POR' :
            'SAC' :
            'SAS' :
            'TOR' :
            'UTA' :
            'WSH' :
}

    if year == 2016:
        eloDict = {}
    if year = 2017:
        eloDict = {}
    if year = 2018:
        eloDict = {}
    if year = 2019:
        eloDict = {}    
    if year = 2020:
        eloDict = {}
    if year = 2021:
        eloDict = {}
    if year = 2022:
        eloDict = {}

    gameIdList =  pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0).index
    df = pd.DataFrame()
    for gameId in gameIdList:
        eloDict = getEloDict(eloDict, gameId)
        
        

def getEloDict(eloDict, gameId):
    teamHome, teamAway, pointsAway, pointsHome = getEloInputs(gameId)
    eloHome = eloDict[teamHome]
    eloAway = eloDict[teamAway]

    eloHome, eloAway = update_elo(pointsHome, pointsAway, eloHome, eloAway, 100)
    upDict = {teamHome: eloHome, teamAway: eloAway
        }
    eloDict = eloDict.update(upDict)
    return eloDict
    
    
def getEloInputs(gameId):
    df = pd.read_csv('../data/gameStats/game_state_data_{}.csv'.format(year), header = [0,1], index_col = 0)
    df =  df.loc[:, [('home', 'points'), ('away','points'), ('gameState', 'teamHome'), ('gameState', 'teamAway')]]
    pointsHome = df.loc[gameId]['home']['points']
    pointsAway = df.loc[gameId]['away']['points']

    teamHome = df.loc[gameId]['gameState']['teamHome']
    teamAway = df.loc[gameId]['gameState']['teamAway']

    return teamHome, teamAway, pointsAway, pointsHome

    

    
    
    
    
