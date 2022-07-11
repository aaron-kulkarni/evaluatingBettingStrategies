import numpy as np
import pandas as pd
import datetime
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
from sportsipy.nba.boxscore import Boxscores



listGames = Boxscores(datetime(2021, 10, 19))


teamSchedule = Schedule('GSW', year = 2022).dataframe
teamSchedule.to_csv('teamSchedule.csv', index = False)

