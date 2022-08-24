import re
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import re
import sys
import math

def getSalarySum(playerIdList, year):
    df = pd.read_csv('../data/staticPlayerData/static_player_stats_{}.csv'.format(year), index_col = 0)
    df = df[df['Id'].isin(playerIdList)]
    return df['salary'].sum()
    
def getSalarySumFile(filename):
    year = re.findall('[0-9]+', filename)[0]
    df = pd.read_csv(filename, index_col = 0, header = [0,1])
    salaryHomeList = []
    salaryAwayList = []
    for gameId in df.index:
         salaryHome = getSalarySum(eval(df.loc[gameId]['home']['playerRoster']), year)
         salaryAway = getSalarySum(eval(df.loc[gameId]['away']['playerRoster']), year)
         salaryHomeList.append(salaryHome)
         salaryAwayList.append(salaryAway)
    df['home', 'salary'] = salaryHomeList
    df['away', 'salary'] = salaryAwayList
    return df
