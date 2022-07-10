import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bs4 as bs
from urllib.request import urlopen
import requests
from dateutil.relativedelta import relativedelta
import pdb
from sportsipy.nba.teams import Teams
from sportsreference.nba.roster import Roster
from sportsreference.nba.roster import Player

teams2022 = Teams(year = '2022')
ids = []
names = []
heights = []
weights = []
roles = []
salary = []
shoots = []

for team in teams2022:
    teamDict = Roster(team.abbreviation, year='2022',
                      slim=True).players
    for key, value in teamDict.items():
        if key not in ids:
            ids.append(key)
            names.append(value)
            salary.append(Player(key).salary)
            url = f"https://www.basketball-reference.com/players/{value.split(' ')[1][0:1].lower()}/{key}.html"
            print(url)
            soup = bs.BeautifulSoup(urlopen(url), features='lxml')
            stats = [p.getText() for p in soup.find('div', {'id':'meta'}).findAll('p')]
            print(stats[4])
            raise Exception('aaahhh')
            

df = pd.DataFrame()
df['Name'] = names
df['Id'] = ids
df['height'] = heights
df['weights'] = weights
df['role'] = roles
df['salary'] = salary
df['shoots'] = shoots



        
